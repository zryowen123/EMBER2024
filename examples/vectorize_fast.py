#!/usr/bin/env python3
"""Fast raw-JSONL -> vectorized memmap conversion for EMBER2024.

This is a drop-in replacement for ``thrember.create_vectorized_features`` with
much lower per-row overhead:

* one multiprocessing task covers a contiguous run of complete JSON lines
  (default ~16 MiB of input), not one task per sample;
* the task carries only file offsets, so the raw JSON is never pickled between
  processes -- the worker reads it straight from disk;
* each worker imports the extractor and opens the ``X_*.dat`` / ``y_*.dat``
  memmaps once, then writes feature vectors directly into them (batched);
* ``.jsonl`` and ``.jsonl.gz`` inputs are both handled natively; gzip files are
  processed one file per task;
* progress is reported as rows/s with an ETA, and a ``vectorize_meta.json``
  sidecar records the feature dim, row counts, and (for non-binary label types)
  the label map, which the reference implementation does not persist;
* hard-linked duplicates (for example ``data/`` plus ``data/by_type/Win64/``)
  are detected by inode and counted once.

Output layout matches ``thrember.read_vectorized_features``::

    OUT/X_train.dat  OUT/y_train.dat     float32 (n, dim) / int32 (n,)
    OUT/X_test.dat   OUT/y_test.dat      test split
    OUT/X_challenge.dat OUT/y_challenge.dat

For tag label types (behavior/file_property/packer/exploit/group) the train and
test ``y_*.dat`` are ``int32 (n, nclasses)``; challenge labels always use the
binary ``label`` field, mirroring the reference implementation.

Usage::

    python examples/vectorize_fast.py data/ --out-dir data/vec/Win64 --include Win64
    python examples/vectorize_fast.py data/by_type/Win64 --out-dir data/vec/Win64
    python examples/vectorize_fast.py data/ --out-dir out --label-type family
    python examples/vectorize_fast.py data/ --out-dir out --scan-only

Splits are inferred from the file name: ``challenge`` wins over ``test``, which
wins over ``train``; use ``--force-split`` for files without those keywords.
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from thrember.features import PEFeatureExtractor

try:  # optional, ~2x faster than the stdlib decoder
    import orjson

    _loads = orjson.loads
except ImportError:  # pragma: no cover - depends on the environment
    _loads = json.loads


SPLITS = ("train", "test", "challenge")
FLAT_LABEL_TYPES = {"label", "family"}
TAG_LABEL_TYPES = {"behavior", "file_property", "packer", "exploit", "group"}
LABEL_TYPES = FLAT_LABEL_TYPES | TAG_LABEL_TYPES
IGNORE_TAGS = {"", "win32", "win64", "elf", "linux", "pdf", "apk", "android"}

# Rows buffered per worker before the batch is written into the memmap.
BATCH_ROWS = 2048
# Read blocks used for scanning / line iteration.
IO_BLOCK = 1 << 20


@dataclass
class Chunk:
    """A contiguous run of whole JSON lines inside one input file."""

    split: str
    path: str
    start: int  # byte offset (plain files); unused for gzip
    end: int  # byte offset (plain files); unused for gzip
    nrows: int
    row_off: int = 0


def fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class Progress:
    """Throttled rows/s + ETA reporter (rewrites one line on a tty)."""

    def __init__(self, total: int, label: str):
        self.total = max(int(total), 1)
        self.label = label
        self.tty = sys.stdout.isatty()
        self.start = self.last = time.time()
        self.done = 0
        self.width = 0

    def update(self, n: int = 0, force: bool = False) -> None:
        self.done += n
        now = time.time()
        interval = 0.5 if self.tty else 10.0
        if not force and now - self.last < interval:
            return
        self.last = now
        elapsed = max(now - self.start, 1e-9)
        rate = self.done / elapsed
        pct = 100.0 * self.done / self.total
        eta = (self.total - self.done) / rate if rate > 0 else 0.0
        line = (
            f"{self.label}: {self.done:,}/{self.total:,} rows "
            f"({pct:5.1f}%)  {rate:,.0f} rows/s  ETA {fmt_duration(eta)}"
        )
        self.width = max(self.width, len(line))
        if self.tty:
            print("\r" + line.ljust(self.width), end="", flush=True)
        else:
            print(line, flush=True)

    def close(self) -> None:
        if self.tty:
            self.update(force=True)
            print()


# ---------------------------------------------------------------------------
# Discovery and scanning
# ---------------------------------------------------------------------------


def _is_raw_jsonl(name: str) -> bool:
    return name.endswith(".jsonl") or name.endswith(".jsonl.gz")


def _infer_split(path: Path) -> str | None:
    name = path.name.lower()
    if "challenge" in name:
        return "challenge"
    if "test" in name:
        return "test"
    if "train" in name:
        return "train"
    return None


def discover_files(
    srcs: list[str],
    includes: list[str],
    excludes: list[str],
    force_split: str | None,
) -> list[tuple[Path, str]]:
    found: dict[str, tuple[Path, str]] = {}
    seen_inodes: set[tuple[int, int]] = set()
    duplicates = 0
    for src in srcs:
        path = Path(src)
        if path.is_dir():
            root = path.resolve()
            candidates = [
                p for p in sorted(path.rglob("*"))
                if p.is_file() and _is_raw_jsonl(p.name)
            ]
            # Skip hidden directories such as .vec_cache.
            candidates = [
                p for p in candidates
                if not any(part.startswith(".") for part in p.relative_to(path).parts)
            ]
        elif path.is_file():
            candidates = [path]
        else:
            raise SystemExit(f"no such file or directory: {src}")
        for candidate in candidates:
            resolved = candidate.resolve()
            if str(resolved) in found:
                continue
            text = str(candidate)
            if includes and not any(s in text for s in includes):
                continue
            if excludes and any(s in text for s in excludes):
                continue
            # Hard-linked copies (e.g. data/ and data/by_type/<type>/) would
            # otherwise be counted twice.
            try:
                stat = candidate.stat()
            except OSError:
                stat = None
            if stat is not None:
                inode = (stat.st_dev, stat.st_ino)
                if inode in seen_inodes:
                    duplicates += 1
                    continue
                seen_inodes.add(inode)
            split = force_split or _infer_split(candidate)
            if split is None:
                print(
                    f"[skip] cannot infer train/test/challenge from {candidate}; "
                    "use --force-split",
                    file=sys.stderr,
                )
                continue
            found[str(resolved)] = (candidate, split)
    if duplicates:
        print(
            f"[skip] {duplicates} duplicate file(s) sharing an inode with an "
            "earlier file (hard-linked copies)",
            file=sys.stderr,
        )
    return sorted(found.values(), key=lambda item: str(item[0]))


def scan_plain(path: Path, chunk_bytes: int) -> list[Chunk]:
    """Split a plain JSONL file into byte ranges that contain whole lines."""
    chunks: list[Chunk] = []
    pos = 0
    tail = b""
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk_bytes)
            if not data:
                break
            if tail:
                data = tail + data
                tail = b""
            cut = data.rfind(b"\n")
            if cut < 0:
                tail = data
                continue
            piece = data[: cut + 1]
            nrows = piece.count(b"\n")
            chunks.append(Chunk("", str(path), pos, pos + len(piece), nrows))
            pos += len(piece)
            tail = data[cut + 1:]
    if tail:  # last line without a trailing newline
        chunks.append(Chunk("", str(path), pos, pos + len(tail), 1))
    return chunks


def scan_gzip(path: Path) -> list[Chunk]:
    """Count rows in a gzip file; the whole file becomes a single task."""
    nrows = 0
    last = b""
    with gzip.open(path, "rb") as f:
        while True:
            block = f.read(IO_BLOCK)
            if not block:
                break
            nrows += block.count(b"\n")
            last = block[-1:]
    if last not in (b"", b"\n"):
        nrows += 1
    return [Chunk("", str(path), 0, 0, nrows)] if nrows else []


def scan_all(
    files: list[tuple[Path, str]], chunk_bytes: int
) -> tuple[list[Chunk], dict[str, int], dict[str, int]]:
    chunks: list[Chunk] = []
    rows = {split: 0 for split in SPLITS}
    raw_bytes = {split: 0 for split in SPLITS}
    t0 = time.time()
    progress = Progress(len(files), "scan")
    for index, (path, split) in enumerate(files, 1):
        file_chunks = (
            scan_gzip(path) if path.name.endswith(".gz") else scan_plain(path, chunk_bytes)
        )
        for chunk in file_chunks:
            chunk.split = split
            chunk.row_off = rows[split]
            rows[split] += chunk.nrows
        chunks.extend(file_chunks)
        try:
            raw_bytes[split] += path.stat().st_size
        except OSError:
            pass
        progress.update(1)
    progress.close()
    elapsed = time.time() - t0
    print(
        f"Scanned {len(files)} file(s) in {fmt_duration(elapsed)}: "
        + "  ".join(
            f"{split} {rows[split]:,} rows ({raw_bytes[split] / 1e9:.2f} GB)"
            for split in SPLITS
            if rows[split]
        )
    )
    return chunks, rows, raw_bytes


# ---------------------------------------------------------------------------
# Worker functions (module level so they pickle on spawn)
# ---------------------------------------------------------------------------


def _iter_lines(path: str, start: int, end: int):
    """Yield the lines covered by a chunk, without the trailing newline."""
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            tail = b""
            while True:
                block = f.read(IO_BLOCK)
                if not block:
                    break
                if tail:
                    block = tail + block
                    tail = b""
                cut = block.rfind(b"\n")
                if cut < 0:
                    tail = block
                    continue
                yield from block[: cut + 1].split(b"\n")[:-1]
                tail = block[cut + 1:]
            if tail:
                yield tail
    else:
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(end - start)
        parts = data.split(b"\n")
        if parts and parts[-1] == b"":
            parts.pop()
        yield from parts


def _flat_label(raw, label_type: str, label_map: dict | None) -> int:
    if not isinstance(raw, dict):
        return -1
    label = raw.get(label_type)
    if label is None:
        return -1
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, int):
        return label
    if isinstance(label, str) and label_map is not None:
        return label_map.get(label, -1)
    return -1


def _worker_init(config: dict) -> None:
    global _CONFIG, _X_MAPS, _Y_MAPS, _EXTRACTOR
    _CONFIG = config
    _X_MAPS = {}
    _Y_MAPS = {}
    _EXTRACTOR = PEFeatureExtractor()


def _maps(split: str):
    X = _X_MAPS.get(split)
    if X is None:
        X = np.memmap(
            _CONFIG["x_path"][split],
            dtype=np.float32,
            mode="r+",
            shape=(_CONFIG["rows"][split], _CONFIG["dim"]),
        )
        Y = np.memmap(
            _CONFIG["y_path"][split],
            dtype=np.int32,
            mode="r+",
            shape=_CONFIG["y_shape"][split],
        )
        _X_MAPS[split] = X
        _Y_MAPS[split] = Y
    return _X_MAPS[split], _Y_MAPS[split]


def _label_config(split: str):
    # The reference implementation vectorizes the challenge set with the
    # binary "label" field regardless of --label-type; mirror that.
    if split == "challenge":
        return "label", None, False
    label_type = _CONFIG["label_type"]
    if label_type in TAG_LABEL_TYPES:
        return label_type, _CONFIG["label_map"], True
    return label_type, _CONFIG["label_map"], False


def _process_chunk(chunk: Chunk):
    split = chunk.split
    dim = _CONFIG["dim"]
    label_type, label_map, is_tags = _label_config(split)
    X, Y = _maps(split)

    batch = min(chunk.nrows, BATCH_ROWS)
    if batch <= 0:
        return split, 0, 0, ""
    xbuf = np.empty((batch, dim), dtype=np.float32)
    ybuf = (
        np.zeros((batch, Y.shape[1]), dtype=np.int32)
        if is_tags
        else np.zeros((batch,), dtype=np.int32)
    )

    errors = 0
    first_error = ""
    written = 0
    buffered = 0

    for line in _iter_lines(chunk.path, chunk.start, chunk.end):
        raw = None
        try:
            raw = _loads(line)
            xbuf[buffered] = _EXTRACTOR.process_raw_features(raw)
        except Exception as exc:  # noqa: BLE001 - report and continue
            xbuf[buffered] = 0.0
            if not isinstance(raw, dict):
                raw = None
            errors += 1
            if not first_error:
                first_error = f"{type(exc).__name__}: {exc}"[:200]

        if is_tags:
            ybuf[buffered].fill(0)
            if raw is not None:
                tags = raw.get(label_type)
                if tags is not None:
                    if not isinstance(tags, list):
                        tags = [tags]
                    for tag in tags:
                        index = label_map.get(tag) if isinstance(tag, str) else None
                        if index is not None:
                            ybuf[buffered, index] = 1
        else:
            ybuf[buffered] = _flat_label(raw, label_type, label_map)

        buffered += 1
        if buffered == batch:
            X[chunk.row_off + written : chunk.row_off + written + buffered] = xbuf[:buffered]
            Y[chunk.row_off + written : chunk.row_off + written + buffered] = ybuf[:buffered]
            written += buffered
            buffered = 0

    if buffered:
        X[chunk.row_off + written : chunk.row_off + written + buffered] = xbuf[:buffered]
        Y[chunk.row_off + written : chunk.row_off + written + buffered] = ybuf[:buffered]
        written += buffered

    if written != chunk.nrows:
        raise RuntimeError(
            f"{chunk.path}: handled {written} rows but the scan counted {chunk.nrows}"
        )
    return split, written, errors, first_error


def _counter_init(label_type: str) -> None:
    global _COUNT_LABEL
    _COUNT_LABEL = label_type


def _count_chunk(chunk: Chunk):
    counts: dict[str, int] = {}
    for line in _iter_lines(chunk.path, chunk.start, chunk.end):
        try:
            raw = _loads(line)
        except Exception:  # noqa: BLE001 - bad lines are reported in the main pass
            continue
        if not isinstance(raw, dict):
            continue
        tags = raw.get(_COUNT_LABEL)
        if tags is None:
            continue
        if not isinstance(tags, list):
            tags = [tags]
        for tag in tags:
            if isinstance(tag, str):
                counts[tag] = counts.get(tag, 0) + 1
    return counts, chunk.nrows


def collect_tag_counts(
    chunks: list[Chunk], label_type: str, workers: int
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not chunks:
        return counts
    total = sum(chunk.nrows for chunk in chunks)
    progress = Progress(total, f"count {label_type}")
    pool = mp.get_context().Pool(
        processes=workers, initializer=_counter_init, initargs=(label_type,)
    )
    try:
        for chunk_counts, nrows in pool.imap_unordered(
            _count_chunk, chunks, chunksize=1
        ):
            progress.update(nrows)
            for tag, count in chunk_counts.items():
                counts[tag] = counts.get(tag, 0) + count
    finally:
        pool.terminate()
        pool.join()
    progress.close()
    return counts


def build_label_map(
    train_counts: dict[str, int], test_counts: dict[str, int], class_min: int
) -> dict[str, int]:
    """Deterministic version of the reference map (sorted tag order)."""
    label_map: dict[str, int] = {}

    def _add(source: dict[str, int]) -> None:
        for tag in sorted(source):
            if tag in IGNORE_TAGS or tag in label_map:
                continue
            if source[tag] >= class_min:
                label_map[tag] = len(label_map)

    _add(train_counts)
    _add(test_counts)
    return label_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast raw-JSONL -> memmap vectorization for EMBER2024.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("src", nargs="+", help="JSONL files or directories (searched recursively)")
    parser.add_argument("--out-dir", required=True, help="directory for X_*.dat / y_*.dat")
    parser.add_argument(
        "--label-type",
        default="label",
        choices=sorted(LABEL_TYPES),
        help="classification target stored in y_*.dat",
    )
    parser.add_argument(
        "--class-min", type=int, default=10,
        help="drop tags appearing fewer than this many times (non-binary label types)",
    )
    parser.add_argument(
        "--include", action="append", default=[],
        help="only use files whose path contains this substring (repeatable)",
    )
    parser.add_argument(
        "--exclude", action="append", default=[],
        help="skip files whose path contains this substring (repeatable)",
    )
    parser.add_argument(
        "--force-split", choices=SPLITS, default=None,
        help="assign every discovered file to this split instead of inferring it",
    )
    parser.add_argument(
        "--workers", type=int, default=min(16, os.cpu_count() or 1),
        help="worker processes",
    )
    parser.add_argument(
        "--chunk-mb", type=float, default=16.0,
        help="target input bytes per task (plain JSONL; gzip files are one task each)",
    )
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing .dat files")
    parser.add_argument(
        "--scan-only", action="store_true",
        help="count rows and report estimated output size, then exit",
    )
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.chunk_mb <= 0:
        parser.error("--chunk-mb must be > 0")
    if args.class_min < 0:
        parser.error("--class-min must be >= 0")
    if args.label_type in TAG_LABEL_TYPES and args.force_split == "challenge":
        parser.error("challenge labels are binary; --label-type with tags is not meaningful")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    t_start = time.time()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise SystemExit(f"--out-dir is not a directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        p.name for p in out_dir.iterdir()
        if p.name.startswith(("X_", "y_")) and p.name.endswith(".dat")
    )
    if existing and not args.overwrite:
        raise SystemExit(
            f"{out_dir} already contains {', '.join(existing)}; "
            "pass --overwrite to replace them"
        )

    files = discover_files(args.src, args.include, args.exclude, args.force_split)
    if not files:
        raise SystemExit("no .jsonl / .jsonl.gz files matched")

    dim = PEFeatureExtractor().dim
    chunk_bytes = int(args.chunk_mb * 1_000_000)
    chunks, rows, raw_bytes = scan_all(files, chunk_bytes)
    if not chunks:
        raise SystemExit("the matched files contain no rows")

    if args.scan_only:
        needed = sum(rows[s] * dim * 4 for s in SPLITS)
        print(f"Feature dim: {dim}  output X size: {needed / 1e9:.1f} GB (plus labels)")
        print("No files written (--scan-only).")
        return

    # Build the label map before allocating the multi-label y matrices.
    label_map: dict[str, int] = {}
    nclasses = 0
    if args.label_type != "label":
        train_chunks = [c for c in chunks if c.split == "train"]
        test_chunks = [c for c in chunks if c.split == "test"]
        train_counts = collect_tag_counts(train_chunks, args.label_type, args.workers)
        test_counts = collect_tag_counts(test_chunks, args.label_type, args.workers)
        label_map = build_label_map(train_counts, test_counts, args.class_min)
        nclasses = len(label_map)
        print(f"Label map: {nclasses:,} tag(s) kept (class_min={args.class_min})")
        if args.label_type in TAG_LABEL_TYPES and nclasses == 0:
            raise SystemExit("no tags survived --class-min; nothing to vectorize")

    y_shape: dict[str, tuple[int, ...]] = {s: (rows[s],) for s in SPLITS}
    if args.label_type in TAG_LABEL_TYPES:
        y_shape["train"] = (rows["train"], nclasses)
        y_shape["test"] = (rows["test"], nclasses)

    needed = sum(rows[s] * dim * 4 for s in SPLITS) + sum(
        int(np.prod(y_shape[s])) * 4 for s in SPLITS
    )
    free = shutil.disk_usage(out_dir).free
    print(f"Output required: {needed / 1e9:.1f} GB  (free: {free / 1e9:.1f} GB)")
    if needed > free:
        raise SystemExit(
            f"not enough free space in {out_dir}: need {needed / 1e9:.1f} GB, "
            f"have {free / 1e9:.1f} GB"
        )

    x_path = {s: str(out_dir / f"X_{s}.dat") for s in SPLITS}
    y_path = {s: str(out_dir / f"y_{s}.dat") for s in SPLITS}
    for split in SPLITS:
        if rows[split] == 0:
            continue
        X = np.memmap(
            x_path[split], dtype=np.float32, mode="w+", shape=(rows[split], dim)
        )
        del X
        Y = np.memmap(y_path[split], dtype=np.int32, mode="w+", shape=y_shape[split])
        del Y
        print(
            f"Allocated {x_path[split]} ({rows[split]:,} x {dim} float32) and "
            f"{y_path[split]} {y_shape[split]}"
        )

    config = {
        "dim": dim,
        "rows": rows,
        "x_path": x_path,
        "y_path": y_path,
        "y_shape": y_shape,
        "label_type": args.label_type,
        "label_map": label_map,
        "class_min": args.class_min,
    }

    total = sum(chunk.nrows for chunk in chunks)
    progress = Progress(total, "vectorize")
    errors_by_split = {s: 0 for s in SPLITS}
    error_samples: list[str] = []
    done = 0
    t_vec = time.time()

    pool = mp.get_context().Pool(
        processes=args.workers, initializer=_worker_init, initargs=(config,)
    )
    try:
        for split, written, nerrors, first_error in pool.imap_unordered(
            _process_chunk, chunks, chunksize=1
        ):
            done += written
            errors_by_split[split] += nerrors
            if first_error and len(error_samples) < 5:
                error_samples.append(f"{first_error}")
            progress.update(written)
    except KeyboardInterrupt:
        print("\nInterrupted; terminating workers.", file=sys.stderr)
        pool.terminate()
        pool.join()
        raise
    except Exception:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()
    finally:
        progress.close()

    vectorize_seconds = time.time() - t_vec

    print("Flushing memmaps...")
    for split in SPLITS:
        if rows[split] == 0:
            continue
        mm = np.memmap(
            x_path[split], dtype=np.float32, mode="r+", shape=(rows[split], dim)
        )
        mm.flush()
        del mm
        mm = np.memmap(
            y_path[split], dtype=np.int32, mode="r+", shape=y_shape[split]
        )
        mm.flush()
        del mm

    total_errors = sum(errors_by_split.values())
    print()
    for split in SPLITS:
        if rows[split] == 0:
            continue
        print(
            f"  {split:9} {rows[split]:>10,} rows  errors {errors_by_split[split]:>6,}"
        )
    if total_errors:
        print(
            f"Warning: {total_errors:,} line(s) could not be vectorized; they were "
            "written as zero vectors with label -1 (flat) / all-zero tags."
        )
        for sample in error_samples:
            print(f"  example: {sample}")

    elapsed = time.time() - t_start
    rate = total / vectorize_seconds if vectorize_seconds > 0 else 0.0
    print(
        f"Done: {total:,} rows in {fmt_duration(elapsed)} total "
        f"({fmt_duration(vectorize_seconds)} vectorize, {rate:,.0f} rows/s, "
        f"{args.workers} workers)"
    )

    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dim": dim,
        "label_type": args.label_type,
        "class_min": args.class_min,
        "workers": args.workers,
        "chunk_bytes": chunk_bytes,
        "splits": {
            split: {
                "rows": rows[split],
                "errors": errors_by_split[split],
                "input_bytes": raw_bytes[split],
            }
            for split in SPLITS
        },
        "label_map": label_map if label_map else None,
        "elapsed_seconds": round(elapsed, 3),
    }
    meta_path = out_dir / "vectorize_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
