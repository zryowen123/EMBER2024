#!/usr/bin/env python3
"""Merge EMBER2024 raw-feature JSONL files, deduplicating by sha256.

``examples/extract_features.py`` is already append-only and skips sha256 values
present in its output, so the normal workflow is to point every extraction run
at the same --output file. This script covers the other case: several JSONL
files produced independently (different machines, batches or directories) need
to become one clean file for evaluation or vectorization.

    # Merge into a new file
    python examples/merge_features.py merged.jsonl batch1.jsonl batch2.jsonl

    # Append to an existing merged file (resume, skip known sha256)
    python examples/merge_features.py merged.jsonl batch3.jsonl

    # Keep the first label when the same sample appears with different labels
    python examples/merge_features.py merged.jsonl batch4.jsonl --allow-label-conflict

Rules:
- One JSON object per line, each with a 64-hex ``sha256``. Invalid JSON, a
  missing/invalid sha256, or a blank line aborts the merge.
- The first occurrence of a sha256 wins; later duplicates are counted and
  skipped. If a duplicate carries a different label, the merge stops unless
  --allow-label-conflict is given (the existing record is kept).
- Records are copied byte-for-byte (minus the line terminator), so unknown
  fields survive. The output is opened in append mode and locked, and --resume
  (default) skips sha256 values already present in the output.
- This is for the raw ``.jsonl`` only. The ``.errors.jsonl`` and
  ``.skipped.jsonl`` side logs can simply be concatenated with ``cat`` if you
  need to keep them.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import stat
import sys
from typing import Dict, Iterator, Optional, Tuple


def iter_jsonl(path: str) -> Iterator[Tuple[int, str, Dict]]:
    try:
        handle = open(path, "rb")
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    with handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.endswith(b"\n"):
                raw = raw + b"\n"
            text = raw.decode("utf-8", "surrogateescape").rstrip("\r\n")
            if not text.strip():
                raise SystemExit(f"blank line in {path}:{line_no}")
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise SystemExit(f"record is not a JSON object at {path}:{line_no}")
            yield line_no, text, obj


def record_digest(obj: Dict, path: str, line_no: int) -> bytes:
    sha = obj.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise SystemExit(f"missing or invalid sha256 in {path}:{line_no}")
    try:
        return bytes.fromhex(sha)
    except ValueError as exc:
        raise SystemExit(f"invalid sha256 in {path}:{line_no}: {exc}") from exc


def record_label(obj: Dict) -> Optional[int]:
    label = obj.get("label")
    return label if isinstance(label, int) and not isinstance(label, bool) else None


def scan_existing(path: str, repair: bool) -> Dict[bytes, Optional[int]]:
    """Return {sha256_bytes: label} for records already in the output file."""
    completed: Dict[bytes, Optional[int]] = {}
    if not os.path.lexists(path):
        return completed
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"output is not a regular file: {path}")

    last_good = 0
    with open(path, "rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                if repair:
                    os.truncate(path, last_good)
                    print(
                        f"WARNING: truncated incomplete trailing line in {path}",
                        file=sys.stderr,
                    )
                    break
                raise SystemExit(
                    f"{path} ends with an incomplete line; rerun with --repair "
                    "or remove the partial line"
                )
            line_start = last_good
            last_good += len(raw)
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw.decode("utf-8", "surrogateescape"))
            except json.JSONDecodeError as exc:
                if repair:
                    os.truncate(path, line_start)
                    print(
                        f"WARNING: truncated invalid line in {path} ({exc})",
                        file=sys.stderr,
                    )
                    break
                raise SystemExit(
                    f"invalid JSON in {path} at byte {line_start}: {exc}; "
                    "rerun with --repair to truncate it"
                ) from exc
            if not isinstance(obj, dict):
                continue
            sha = obj.get("sha256")
            if not isinstance(sha, str) or len(sha) != 64:
                continue
            try:
                digest = bytes.fromhex(sha)
            except ValueError:
                continue
            completed[digest] = record_label(obj)
    return completed


def open_output(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, mode=0o700, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
    except OSError:
        pass
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        raise SystemExit(f"output is locked by another process: {path} ({exc})") from exc
    return fd


def write_line(fd: int, path: str, line: str) -> None:
    payload = line.encode("utf-8", "surrogateescape") + b"\n"
    view = memoryview(payload)
    while view:
        try:
            written = os.write(fd, view)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError(errno.EIO, f"short write to {path}")
        view = view[written:]


def run(args: argparse.Namespace) -> int:
    output = os.path.abspath(args.output)
    inputs = [os.path.abspath(path) for path in args.inputs]
    if output in inputs:
        raise SystemExit("--output must not also be an input file")
    for path in inputs:
        if not os.path.isfile(path):
            raise SystemExit(f"input does not exist: {path}")

    if args.resume:
        seen = scan_existing(output, args.repair)
    else:
        seen = {}
        if os.path.exists(output) and os.path.getsize(output) > 0:
            raise SystemExit(
                f"output exists and --no-resume was given: {output}; "
                "remove it or use --resume"
            )

    stats = {"read": 0, "written": 0, "duplicates": 0, "label_conflicts": 0}
    fd = open_output(output)
    try:
        for path in inputs:
            for line_no, text, obj in iter_jsonl(path):
                stats["read"] += 1
                digest = record_digest(obj, path, line_no)
                label = record_label(obj)
                if digest in seen:
                    previous = seen[digest]
                    if (
                        isinstance(previous, int)
                        and isinstance(label, int)
                        and previous != label
                    ):
                        message = (
                            f"label conflict: {digest.hex()} is already recorded "
                            f"with label {previous}, but {path}:{line_no} has "
                            f"label {label}; the existing record was kept"
                        )
                        stats["label_conflicts"] += 1
                        if not args.allow_label_conflict:
                            raise SystemExit(
                                message
                                + ". Use --allow-label-conflict to keep the "
                                "first label."
                            )
                        print(f"WARNING: {message}", file=sys.stderr)
                    stats["duplicates"] += 1
                    continue
                write_line(fd, output, text)
                seen[digest] = label
                stats["written"] += 1
        os.fsync(fd)
    finally:
        os.close(fd)

    print(
        f"merged {len(inputs)} file(s) -> {output}: "
        f"read={stats['read']} written={stats['written']} "
        f"duplicates={stats['duplicates']} "
        f"label_conflicts={stats['label_conflicts']} "
        f"total={len(seen)}",
        file=sys.stderr,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge EMBER2024 raw-feature JSONL files, deduplicating by sha256."
        )
    )
    parser.add_argument("output", help="merged JSONL path (appended to by default)")
    parser.add_argument("inputs", nargs="+", help="input JSONL file(s)")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="skip sha256 values already present in the output (default: on)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="do not scan the output; requires a fresh/empty output file",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="truncate an incomplete/invalid trailing line in the output",
    )
    parser.add_argument(
        "--allow-label-conflict",
        action="store_true",
        help="keep the first label when the same sha256 appears with a different label",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
