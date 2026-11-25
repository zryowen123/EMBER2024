#!/usr/bin/env python3
"""
Build per-file-type directories suitable for thrember.create_vectorized_features().

create_vectorized_features() scans a single directory and requires train, test
and challenge files to be present. This helper creates one directory per file
type:

    <data_dir>/by_type/Win64/
        <week>_Win64_train.jsonl      (hardlink to the original)
        <week>_Win64_test.jsonl       (hardlink)
        challenge_Win64_malicious.jsonl  (real file, filtered by file_type)

Hardlinks cost no extra space. The challenge set is mixed-type in the published
data, so it is split by the record's ``file_type`` field.

Usage:
    python examples/make_by_type_dirs.py <data_dir> \
        [--output-dir DIR] [--types Win32 Win64 ...] [--force]
"""

import argparse
import json
import os
from pathlib import Path

ALL_TYPES = ["Win32", "Win64", "Dot_Net", "APK", "ELF", "PDF"]


def link_or_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.link(src, dst)
    except OSError:
        dst.symlink_to(src.resolve())


def split_challenge(src_dir: Path, dst_dir: Path, file_type: str, force: bool) -> int:
    out = dst_dir / f"challenge_{file_type}_malicious.jsonl"
    if out.exists() and not force:
        return sum(1 for _ in out.open("rb"))
    kept = 0
    with out.open("w") as fout:
        for path in sorted(src_dir.glob("*challenge*.jsonl")):
            with path.open() as fin:
                for line in fin:
                    if json.loads(line).get("file_type") == file_type:
                        fout.write(line)
                        kept += 1
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create per-file-type directories for EMBER2024 vectorization."
    )
    parser.add_argument("data_dir", help="directory with the deduplicated weekly JSONL")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="where to create by_type/ (default: <data_dir>/by_type)",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=ALL_TYPES,
        choices=ALL_TYPES,
        help="file types to build (default: all six)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rewrite challenge files even if they already exist",
    )
    args = parser.parse_args()

    src = Path(args.data_dir)
    if not src.is_dir():
        raise SystemExit(f"not a directory: {src}")
    out_root = Path(args.output_dir) if args.output_dir else src / "by_type"

    for file_type in args.types:
        dst = out_root / file_type
        dst.mkdir(parents=True, exist_ok=True)

        train = sorted(src.glob(f"*_{file_type}_train.jsonl"))
        test = sorted(src.glob(f"*_{file_type}_test.jsonl"))
        if not train or not test:
            print(f"{file_type}: skipping (train={len(train)} test={len(test)})")
            continue
        for path in train + test:
            link_or_symlink(path, dst / path.name)

        challenge_rows = split_challenge(src, dst, file_type, args.force)
        print(
            f"{file_type}: {len(train)} train + {len(test)} test linked, "
            f"{challenge_rows} challenge rows -> {dst}"
        )


if __name__ == "__main__":
    main()
