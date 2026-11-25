#!/usr/bin/env python3
"""
Memory-efficient LightGBM training on EMBER2024 vectorized features.

The stock ``thrember.train_model()`` path materializes X several times:
``np.array(memmap)`` -> ``X[y != -1]`` -> ``train_test_split`` copies. On the
full 2.6M x 2568 training set that peaks around 60 GB. This script streams rows
from the ``.dat`` memmap into LightGBM in batches instead:

* ``X_train.dat`` is opened read-only as a memmap; rows are never all resident.
* Unlabeled rows (label == -1) are skipped with an index array, no full copy.
* The 10% validation split is index-based, not a copy of X.
* ``lightgbm.Sequence`` pushes rows to the Dataset in ``--batch-size`` chunks.

Peak memory is the binned Dataset (~1 byte per feature per row, ~6.7 GB for the
full training set), the bin-construction sample (200k rows as float64, ~4 GB),
one batch, and resident memmap pages (clean and reclaimable), instead of several
copies of X.

Usage:
    python examples/train_lgbm_memmap.py <data_dir> <model_path> \
        [--config-file examples/lgbm_config.json] \
        [--batch-size 4096] [--bin-sample-count 200000] \
        [--no-valid] [--threads N]
"""

import argparse
import json
import os
import resource
import sys
import time

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split

import thrember


# Same categorical columns as thrember.model.train_model().
CATEGORICAL_FEATURES = [2, 3, 4, 5, 6, 701, 702]


class IndexedMemmapSequence(lgb.Sequence):
    """Present selected rows of a 2-D memmap to LightGBM in batches.

    ``rows is None`` means "use every row"; otherwise ``rows`` holds memmap row
    indices. Slices are gathered per batch, so only ``batch_size`` rows are
    materialized at a time.

    ``__getitem__`` returns float64 because LightGBM's Dataset sampling
    (``LGBM_DatasetCreateFromSampledColumn``) requires doubles.
    """

    def __init__(
        self,
        X: npt.NDArray,
        rows: npt.NDArray | None = None,
        batch_size: int = 4096,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.X = X
        self.rows = rows
        self.batch_size = int(batch_size)

    def __len__(self) -> int:
        return self.X.shape[0] if self.rows is None else len(self.rows)

    def __getitem__(self, idx):
        if self.rows is None:
            data = self.X[idx]
        elif isinstance(idx, (int, np.integer)):
            data = self.X[self.rows[idx]]
        elif isinstance(idx, slice):
            data = self.X[self.rows[idx]]
        else:  # list / ndarray of row positions
            data = self.X[self.rows[np.asarray(idx)]]
        return np.asarray(data, dtype=np.float64)


def check_file(path: str) -> str:
    if not os.path.isfile(path):
        raise SystemExit(f"missing file: {path}")
    return path


def open_memmap(path: str, dtype, shape=None) -> np.memmap:
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def peak_rss_gb() -> float:
    # ru_maxrss is KiB on Linux and bytes on macOS.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1e9
    return rss * 1024 / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM on EMBER2024 .dat features with bounded memory."
    )
    parser.add_argument("data_dir", help="directory containing X_train.dat / y_train.dat")
    parser.add_argument("model_path", help="path to write the LightGBM model")
    parser.add_argument(
        "--config-file",
        default="examples/lgbm_config.json",
        help="LightGBM parameter JSON (default: examples/lgbm_config.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="rows pushed per batch; ~42 MB per batch at 2568 float32 (default: 4096)",
    )
    parser.add_argument(
        "--bin-sample-count",
        type=int,
        default=200000,
        help="rows sampled to build feature bins (LightGBM default 200000 uses ~4 GB "
        "as float64 and avoids its small-sample warning; 20000 uses ~0.4 GB)",
    )
    parser.add_argument(
        "--no-valid",
        action="store_true",
        help="skip the 10%% validation split (lowest memory, no per-iteration metrics)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="override LightGBM num_threads (default: use the config)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise SystemExit(f"not a directory: {args.data_dir}")
    if not os.path.isfile(args.config_file):
        raise SystemExit(f"not a file: {args.config_file}")

    extractor = thrember.PEFeatureExtractor()
    dim = extractor.dim

    X_path = check_file(os.path.join(args.data_dir, "X_train.dat"))
    y_path = check_file(os.path.join(args.data_dir, "y_train.dat"))

    X = open_memmap(X_path, np.float32)
    if X.size % dim != 0:
        raise SystemExit(
            f"{X_path} size {X.size} is not a multiple of the feature dim {dim}"
        )
    nrows = X.size // dim
    X = X.reshape(nrows, dim)

    y = open_memmap(y_path, np.int32)
    if y.size == nrows:
        y = np.asarray(y)  # ~10 MB for the full training set
    elif y.size > nrows and y.size % nrows == 0:
        raise SystemExit(
            "y_train.dat is multi-label (2-D); this script handles flat labels only. "
            "Use thrember.train_ovr_model() or a dedicated per-class script."
        )
    else:
        raise SystemExit(
            f"y_train.dat has {y.size} values but X_train.dat has {nrows} rows"
        )

    labeled = np.flatnonzero(y != -1)
    n_labeled = len(labeled)
    if n_labeled == 0:
        raise SystemExit("no labeled rows (all labels are -1)")
    if n_labeled == nrows:
        rows = None
        y_labeled = y
        print(f"Rows: {nrows:,} (all labeled)")
    else:
        rows = labeled
        y_labeled = y[labeled]
        print(f"Rows: {nrows:,} ({nrows - n_labeled:,} unlabeled skipped)")

    num_classes = int(y_labeled.max()) + 1
    if num_classes < 2:
        raise SystemExit("training data must contain at least two classes")

    # Index-based validation split: no copy of X.
    if args.no_valid:
        train_rows = rows
        val_rows = None
        y_train = y_labeled
        y_valid = None
    else:
        positions = np.arange(n_labeled)
        train_pos, valid_pos = train_test_split(
            positions, test_size=0.1, stratify=y_labeled, random_state=0
        )
        # Row order does not matter to LightGBM, and sorted positions make the
        # memmap reads sequential instead of random.
        train_pos.sort()
        valid_pos.sort()
        train_rows = train_pos if rows is None else rows[train_pos]
        val_rows = valid_pos if rows is None else rows[valid_pos]
        y_train = y_labeled[train_pos]
        y_valid = y_labeled[valid_pos]

    train_seq = IndexedMemmapSequence(X, train_rows, args.batch_size)
    train_set = lgb.Dataset(
        train_seq,
        label=y_train,
        categorical_feature=CATEGORICAL_FEATURES,
        params={"bin_construct_sample_cnt": args.bin_sample_count},
        free_raw_data=True,
    )

    valid_sets = []
    if val_rows is not None:
        valid_seq = IndexedMemmapSequence(X, val_rows, args.batch_size)
        valid_sets = [
            lgb.Dataset(
                valid_seq,
                label=y_valid,
                reference=train_set,
                categorical_feature=CATEGORICAL_FEATURES,
                params={"bin_construct_sample_cnt": args.bin_sample_count},
            )
        ]

    params = json.load(open(args.config_file, "r"))
    if args.threads is not None:
        params["num_threads"] = args.threads
    if num_classes > 2:
        params.update(
            {
                "objective": "multiclass",
                "num_class": num_classes,
                "metric": "multi_logloss",
            }
        )
        params.pop("is_unbalance", None)

    print(
        f"Features: {dim}  classes: {num_classes}  "
        f"train rows: {len(y_train):,}"
        + (f"  valid rows: {len(y_valid):,}" if y_valid is not None else "")
    )
    print(
        f"Streaming: batch_size={args.batch_size}  "
        f"bin_construct_sample_cnt={args.bin_sample_count}  "
        f"threads={params.get('num_threads', 'config')}"
    )

    t0 = time.time()
    booster = lgb.train(
        params,
        train_set,
        valid_sets=valid_sets if valid_sets else None,
    )
    elapsed = time.time() - t0

    booster.save_model(args.model_path)
    print(f"Saved {args.model_path}")
    print(
        f"Training time: {elapsed:.1f}s  "
        f"trees: {booster.num_trees()}  peak RSS: {peak_rss_gb():.1f} GB"
    )


if __name__ == "__main__":
    main()
