#!/usr/bin/env python3
"""Evaluate a binary EMBER2024 benchmark model on the test and challenge sets.

This follows the exact protocol of ``examples/eval_lgbm.py`` (same metrics, same
operating point, same challenge construction) but reads the raw feature JSONL
written by ``thrember.download_dataset()`` instead of the vectorized
``X_test.dat``/``X_challenge.dat`` files. ``create_vectorized_features()`` only
writes those files when a train split is present, so ``eval_lgbm.py`` cannot be
used with a test+challenge-only download; this script can.

Protocol, identical to ``eval_lgbm.py``:

* **test** - the model scores every row of the test split. Reported metrics:
  ROC AUC, PR AUC (trapezoid area under the precision-recall curve) and TPR at
  the target FPR (default 0.01).
* **challenge** - challenge rows (all malicious) are positives and the benign
  rows of the test split are negatives. Reported metrics: ROC AUC and PR AUC.

The published test JSONL files contain every record twice (the zips on the
HuggingFace hub store duplicate rows, while the challenge files do not), so
the script removes exact duplicates by ``sha256`` within each JSONL file
before evaluating. This reproduces the official per-week file counts
(606,000 test files, 6,315 challenge files) and matters for the challenge PR
AUC, where only the benign side would otherwise be double-counted. Pass
``--keep-duplicates`` to disable this.

Vectorized features are cached as memmaps under ``<data_dir>/.vec_cache``
(about 10 KB per sample), so rerunning with another model only re-predicts.
Pass ``--rebuild`` to re-vectorize or ``--no-cache`` for a temporary cache.

Only the binary malicious/benign models are supported (``all``, ``PE``,
``Win32``, ``Win64``, ``Dot_Net``, ``APK``, ``ELF``, ``PDF``). The multiclass
``family`` model and the pickled tag models are rejected.

Examples:
    python examples/eval_dataset.py data model/EMBER2024_all.model
    python examples/eval_dataset.py data model/EMBER2024_all.model \
        --json-out data/eval_results.json --plot-out Classifier_ROC_AUC.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import lightgbm as lgb
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

import thrember
from thrember.features import PEFeatureExtractor

FEATURE_DIM = 2568
DEDUP_MODE = "per-file-v1"


# --------------------------------------------------------------------------- #
# Data discovery and vectorization cache
# --------------------------------------------------------------------------- #


def gather_feature_paths(data_dir: Path, subset: str) -> List[Path]:
    """Match ``thrember.gather_feature_paths``: sorted *.jsonl with ``subset``
    in the file name."""
    paths = [
        data_dir / name
        for name in sorted(p.name for p in data_dir.glob("*.jsonl"))
        if subset in name
    ]
    if not paths:
        raise SystemExit(
            f"no '*{subset}*.jsonl' files in {data_dir}; download the dataset "
            "with thrember.download_dataset()"
        )
    return paths


def count_rows(paths: Sequence[Path]) -> int:
    total = 0
    for path in paths:
        with path.open("rb") as handle:
            total += sum(1 for _ in handle)
    return total


def file_signature(paths: Sequence[Path]) -> List[List[object]]:
    """Name/size/mtime of the source files, used to invalidate the cache."""
    signature: List[List[object]] = []
    for path in paths:
        stat = path.stat()
        signature.append([path.name, stat.st_size, stat.st_mtime_ns])
    return signature


def vectorize_or_load(
    paths: Sequence[Path],
    prefix: str,
    cache_dir: Path,
    extractor: PEFeatureExtractor,
    rebuild: bool,
) -> Tuple[np.memmap, np.memmap]:
    """Return memmapped features/labels, vectorizing only on a cache miss."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    X_path = cache_dir / f"{prefix}_X.dat"
    y_path = cache_dir / f"{prefix}_y.dat"
    meta_path = cache_dir / f"{prefix}_meta.json"
    signature = file_signature(paths)

    if not rebuild and meta_path.is_file() and X_path.is_file() and y_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = None
        if (
            meta
            and meta.get("feature_dim") == extractor.dim
            and meta.get("files") == signature
            and meta.get("nrows")
        ):
            nrows = int(meta["nrows"])
            X_ok = X_path.stat().st_size == nrows * extractor.dim * 4
            y_ok = y_path.stat().st_size == nrows * 4
            if X_ok and y_ok:
                print(f"Using cached vectors: {prefix} ({nrows} rows)")
                return open_memmaps(X_path, y_path, nrows, extractor.dim)

    nrows = count_rows(paths)
    if nrows == 0:
        raise SystemExit(f"no records in {', '.join(str(p) for p in paths)}")
    print(
        f"Vectorizing {prefix}: {nrows} rows from {len(paths)} file(s) "
        f"(cache: {cache_dir})"
    )
    try:
        thrember.vectorize_subset(X_path, y_path, list(paths), extractor, nrows)
    except Exception as exc:  # noqa: BLE001 - make the error actionable
        raise SystemExit(
            f"vectorization failed for {prefix}: {type(exc).__name__}: {exc}\n"
            "The JSONL must be EMBER2024 feature version 3 (2568 dims) produced "
            "by thrember.download_dataset(); check that every record contains "
            "all feature keys."
        ) from exc
    meta_path.write_text(
        json.dumps(
            {"feature_dim": extractor.dim, "nrows": nrows, "files": signature},
            indent=1,
        ),
        encoding="utf-8",
    )
    return open_memmaps(X_path, y_path, nrows, extractor.dim)


def open_memmaps(
    X_path: Path, y_path: Path, nrows: int, dim: int
) -> Tuple[np.memmap, np.memmap]:
    X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(nrows, dim))
    # thrember.vectorize() writes labels as int32 even though the file is
    # preallocated as float32, so it must be read back as int32.
    y = np.memmap(y_path, dtype=np.int32, mode="r", shape=nrows)
    return X, y


SHA256_PATTERN = re.compile(rb'"sha256"\s*:\s*"([0-9a-fA-F]{64})"')


def iter_sha256(paths: Sequence[Path]):
    """Yield (path, sha256) for every JSONL record, in vectorization order."""
    for path in paths:
        with path.open("rb") as handle:
            for line_no, line in enumerate(handle, 1):
                match = SHA256_PATTERN.search(line)
                if match is None:
                    raise SystemExit(f"missing sha256 in {path}:{line_no}")
                yield path, match.group(1).lower()


def dedup_or_load(
    X: np.memmap,
    y: np.memmap,
    paths: Sequence[Path],
    prefix: str,
    cache_dir: Path,
    dim: int,
    rebuild: bool,
) -> Tuple[np.memmap, np.memmap, int]:
    """Drop duplicated records (same sha256), matching the official counts.

    The test zips published on HuggingFace store each record twice. Duplicates
    are removed per JSONL file, which reproduces the official per-week file
    counts (606,000 test files, 6,315 challenge files). Returns the
    deduplicated memmaps and the number of dropped rows.
    """
    signature = file_signature(paths)
    X_path = cache_dir / f"{prefix}_unique_X.dat"
    y_path = cache_dir / f"{prefix}_unique_y.dat"
    meta_path = cache_dir / f"{prefix}_unique_meta.json"

    if not rebuild and meta_path.is_file() and X_path.is_file() and y_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = None
        if (
            meta
            and meta.get("feature_dim") == dim
            and meta.get("dedup_mode") == DEDUP_MODE
            and meta.get("files") == signature
            and meta.get("nrows")
        ):
            nrows = int(meta["nrows"])
            X_ok = X_path.stat().st_size == nrows * dim * 4
            y_ok = y_path.stat().st_size == nrows * 4
            if X_ok and y_ok:
                dropped = int(meta.get("raw_nrows", nrows)) - nrows
                print(f"Using cached deduplicated vectors: {prefix} ({nrows} rows)")
                return open_memmaps(X_path, y_path, nrows, dim) + (dropped,)

    raw_nrows = int(X.shape[0])
    keep = np.zeros(raw_nrows, dtype=bool)
    index = -1
    current_path: Optional[Path] = None
    seen = set()
    for index, (path, sha) in enumerate(iter_sha256(paths)):
        if index >= raw_nrows:
            break
        if current_path != path:
            # Reset per file: the official counts treat the same file appearing
            # in two weekly files as part of both weeks.
            current_path = path
            seen = set()
        if sha not in seen:
            seen.add(sha)
            keep[index] = True
    if index + 1 != raw_nrows:
        raise SystemExit(
            f"{prefix}: JSONL now has {index + 1} rows but the vector cache has "
            f"{raw_nrows}; rerun with --rebuild"
        )

    nrows = int(np.count_nonzero(keep))
    dropped = raw_nrows - nrows
    print(f"Deduplicating {prefix}: kept {nrows} of {raw_nrows} rows")
    X_unique = np.memmap(
        X_path, dtype=np.float32, mode="w+", shape=(nrows, dim)
    )
    y_unique = np.memmap(y_path, dtype=np.int32, mode="w+", shape=nrows)
    indices = np.flatnonzero(keep)
    step = 20_000
    for start in range(0, nrows, step):
        stop = min(start + step, nrows)
        rows = indices[start:stop]
        X_unique[start:stop] = np.asarray(X[rows])
        y_unique[start:stop] = np.asarray(y[rows])
    X_unique.flush()
    y_unique.flush()
    del X_unique, y_unique
    meta_path.write_text(
        json.dumps(
            {
                "feature_dim": dim,
                "dedup_mode": DEDUP_MODE,
                "nrows": nrows,
                "raw_nrows": raw_nrows,
                "files": signature,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return open_memmaps(X_path, y_path, nrows, dim) + (dropped,)


# --------------------------------------------------------------------------- #
# Model and scoring
# --------------------------------------------------------------------------- #


def load_binary_model(path: Path) -> lgb.Booster:
    """Load a plain LightGBM model and reject multiclass/pickled ones."""
    try:
        model = lgb.Booster(model_file=str(path))
    except Exception as exc:  # noqa: BLE001 - LightGBM raises several types
        raise SystemExit(
            f"cannot load {path} as a LightGBM model: {type(exc).__name__}: {exc}\n"
            "Only the binary benchmark models are supported. The family model "
            "is multiclass and the behavior/file_property/packer/exploit/group "
            "models are pickled lists of boosters."
        ) from exc
    if model.num_feature() != FEATURE_DIM:
        raise SystemExit(
            f"{path} expects {model.num_feature()} features, but EMBER2024 "
            f"feature version 3 has {FEATURE_DIM}"
        )
    try:
        models_per_iteration = int(model.num_model_per_iteration())
    except Exception:  # noqa: BLE001 - older LightGBM builds
        models_per_iteration = 1
    if models_per_iteration != 1:
        raise SystemExit(
            f"{path} is a multiclass model ({models_per_iteration} trees per "
            "iteration); only binary malicious/benign models are supported"
        )
    return model


def predict_scores(
    model: lgb.Booster, X: np.ndarray, chunk_size: int = 50_000
) -> np.ndarray:
    """Predict a memmapped feature matrix in bounded-memory chunks."""
    nrows = int(X.shape[0])
    scores = np.empty(nrows, dtype=np.float64)
    for start in range(0, nrows, chunk_size):
        end = min(start + chunk_size, nrows)
        scores[start:end] = model.predict(np.asarray(X[start:end]))
    return scores


# --------------------------------------------------------------------------- #
# Metrics (same definitions as examples/eval_lgbm.py)
# --------------------------------------------------------------------------- #


def check_binary_labels(y: np.ndarray, split: str) -> None:
    labels = set(np.unique(np.asarray(y)).tolist())
    if not labels <= {0, 1}:
        raise SystemExit(
            f"{split} contains labels {sorted(labels)}; the malicious/benign "
            "evaluation expects only 0/1 labels"
        )


def pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Trapezoid area under the precision-recall curve, as in eval_lgbm.py."""
    precision, recall, _ = precision_recall_curve(y, scores)
    return float(auc(recall, precision))


def tpr_at_fpr(
    y: np.ndarray, scores: np.ndarray, target_fpr: float
) -> Dict[str, float]:
    """TPR at the FPR closest to the target, as in eval_lgbm.py."""
    fpr, tpr, thresholds = roc_curve(y, scores)
    index = int(np.argmin(np.abs(fpr - target_fpr)))
    threshold = float(thresholds[index])
    return {
        "target_fpr": float(target_fpr),
        "achieved_fpr": float(fpr[index]),
        "tpr": float(tpr[index]),
        "threshold": threshold if np.isfinite(threshold) else None,
    }


def plot_roc(
    y: np.ndarray,
    scores: np.ndarray,
    target_fpr: float,
    tpr_value: float,
    output: Path,
) -> None:
    """Recreate the ROC figure written by eval_lgbm.py."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the ROC plot")
        return
    fpr, tpr, _ = roc_curve(y, scores)
    figure = plt.figure(figsize=(6, 6))
    plt.title("ROC Curve for EMBERv3 LightGBM Model")
    plt.plot(fpr, tpr, color="black")
    plt.xlim(0.00005, 1.0)
    plt.ylim(0.65, 1.02)
    plt.xscale("log")
    plt.plot(
        [target_fpr, target_fpr, 0],
        [0, tpr_value, tpr_value],
        color="red",
        linestyle="--",
        label=f"TPR at {target_fpr:.2%} FPR",
    )
    plt.xlabel("False Positive Rate (log scale)")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    print(f"Saved ROC curve plot to {output}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a binary EMBER2024 benchmark model on the test and "
            "challenge splits (eval_lgbm.py protocol, raw JSONL input)."
        )
    )
    parser.add_argument(
        "data_dir", help="directory with the EMBER2024 JSONL (e.g. ./data)"
    )
    parser.add_argument(
        "model_path",
        help="binary LightGBM model, e.g. model/EMBER2024_all.model",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.01,
        help="target false-positive rate for the reported TPR (default: 0.01)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="vector cache directory (default: <data_dir>/.vec_cache)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="vectorize into a temporary directory and delete it afterwards",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="ignore existing cached vectors and re-vectorize",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help=(
            "keep duplicate records; the published test JSONL stores each "
            "record twice, so the default is to deduplicate by sha256"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="prediction chunk size (default: 50000)",
    )
    parser.add_argument(
        "--plot-out",
        default="Classifier_ROC_AUC.pdf",
        help="ROC curve output path (default: Classifier_ROC_AUC.pdf, as "
        "eval_lgbm.py)",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="do not write the ROC plot"
    )
    parser.add_argument("--json-out", help="write all metrics as JSON")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not 0 <= args.target_fpr <= 1:
        raise SystemExit("--target-fpr must be in [0, 1]")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"not a directory: {data_dir}")
    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise SystemExit(
            f"model file not found: {model_path}\n"
            "Download the benchmark models with thrember.download_models() or: "
            "hf download joyce8/EMBER2024-benchmark-models --repo-type model"
        )

    extractor = PEFeatureExtractor()
    if extractor.dim != FEATURE_DIM:
        raise SystemExit(
            f"extractor produces {extractor.dim} features, expected {FEATURE_DIM}; "
            "a modified feature set is not compatible with the benchmark models"
        )
    model = load_binary_model(model_path)

    test_paths = gather_feature_paths(data_dir, "test")
    challenge_paths = gather_feature_paths(data_dir, "challenge")

    if args.no_cache:
        cache_dir = Path(tempfile.mkdtemp(prefix="ember_eval_cache_"))
        remove_cache = True
    else:
        cache_dir = Path(args.cache_dir) if args.cache_dir else data_dir / ".vec_cache"
        remove_cache = False

    try:
        # ------------------------------------------------------------------ #
        # Test set
        # ------------------------------------------------------------------ #
        X_test, y_test_mm = vectorize_or_load(
            test_paths, "test", cache_dir, extractor, args.rebuild
        )
        dropped_test = 0
        if not args.keep_duplicates:
            X_test, y_test_mm, dropped_test = dedup_or_load(
                X_test,
                y_test_mm,
                test_paths,
                "test",
                cache_dir,
                extractor.dim,
                args.rebuild,
            )
        y_test = np.asarray(y_test_mm, dtype=np.int64)
        check_binary_labels(y_test, "test")
        if y_test.size and int(np.count_nonzero(y_test == 0)) == 0:
            raise SystemExit("test set has no benign rows; challenge protocol "
                             "needs them as negatives")

        scores_test = predict_scores(model, X_test, args.chunk_size)
        test_roc_auc = float(roc_auc_score(y_test, scores_test))
        test_pr_auc = pr_auc(y_test, scores_test)
        operating_point = tpr_at_fpr(y_test, scores_test, args.target_fpr)

        # ------------------------------------------------------------------ #
        # Challenge set: challenge positives + benign test rows as negatives
        # ------------------------------------------------------------------ #
        X_challenge, y_challenge_mm = vectorize_or_load(
            challenge_paths, "challenge", cache_dir, extractor, args.rebuild
        )
        dropped_challenge = 0
        if not args.keep_duplicates:
            X_challenge, y_challenge_mm, dropped_challenge = dedup_or_load(
                X_challenge,
                y_challenge_mm,
                challenge_paths,
                "challenge",
                cache_dir,
                extractor.dim,
                args.rebuild,
            )
        y_challenge = np.asarray(y_challenge_mm, dtype=np.int64)
        check_binary_labels(y_challenge, "challenge")
        scores_challenge = predict_scores(model, X_challenge, args.chunk_size)
        del X_challenge

        benign_mask = y_test == 0
        y_combined = np.concatenate([y_test[benign_mask], y_challenge])
        scores_combined = np.concatenate(
            [scores_test[benign_mask], scores_challenge]
        )
        challenge_roc_auc = float(roc_auc_score(y_combined, scores_combined))
        challenge_pr_auc = pr_auc(y_combined, scores_combined)

        # ------------------------------------------------------------------ #
        # Report
        # ------------------------------------------------------------------ #
        result = {
            "data_dir": str(data_dir),
            "model": str(model_path),
            "feature_dim": FEATURE_DIM,
            "target_fpr": float(args.target_fpr),
            "test": {
                "count": int(y_test.size),
                "duplicates_removed": int(dropped_test),
                "positive": int(np.count_nonzero(y_test == 1)),
                "negative": int(np.count_nonzero(y_test == 0)),
                "roc_auc": test_roc_auc,
                "pr_auc": test_pr_auc,
                "tpr_at_fpr": operating_point,
            },
            "challenge": {
                "challenge_rows": int(y_challenge.size),
                "duplicates_removed": int(dropped_challenge),
                "benign_test_rows": int(np.count_nonzero(benign_mask)),
                "combined_count": int(y_combined.size),
                "roc_auc": challenge_roc_auc,
                "pr_auc": challenge_pr_auc,
            },
        }

        print(f"Data: {data_dir}")
        print(f"Model: {model_path}")
        if dropped_test or dropped_challenge:
            print(
                f"Duplicates removed: test={dropped_test}, "
                f"challenge={dropped_challenge}"
            )
        print(
            f"Test set: {result['test']['count']} rows "
            f"(positive={result['test']['positive']}, "
            f"negative={result['test']['negative']})"
        )
        print(f"ROC AUC on test set: {test_roc_auc:.6f}")
        print(f"PR AUC on test set: {test_pr_auc:.6f}")
        print(
            f"TPR at FPR {operating_point['achieved_fpr']:.6f} "
            f"(target {args.target_fpr:g}): {operating_point['tpr']:.6f}"
        )
        print(
            f"Challenge set: {y_challenge.size} challenge rows + "
            f"{result['challenge']['benign_test_rows']} benign test rows"
        )
        print(f"ROC AUC on challenge set: {challenge_roc_auc:.6f}")
        print(f"PR AUC on challenge set: {challenge_pr_auc:.6f}")

        if not args.no_plot:
            plot_roc(
                y_test,
                scores_test,
                args.target_fpr,
                operating_point["tpr"],
                Path(args.plot_out),
            )

        if args.json_out:
            out_path = Path(args.json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
            )
            print(f"Metrics JSON: {out_path}")
        if not args.no_cache:
            print(f"Vector cache: {cache_dir}")
    finally:
        if remove_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
