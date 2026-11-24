#!/usr/bin/env python3
"""Evaluate EMBER2024 raw-feature JSONL with a binary LightGBM model.

The input is the JSONL produced by ``examples/extract_features.py`` or the
``*.jsonl`` files of the EMBER2024 dataset. Records are vectorized in memory
with ``thrember.vectorize_subset`` (feature version 3, 2568 dims) and scored
with either a pretrained model or one trained here.

Examples:
    # Route each PE to its Win32/Win64/.NET model (all for other files)
    python examples/evaluate_features.py features.jsonl \
        --models-dir model --target-fpr 0.01 0.001

    # Score everything with one pretrained benchmark model
    python examples/evaluate_features.py features.jsonl \
        --model model/EMBER2024_all.model --target-fpr 0.01 0.001

    # Train on a separate feature file and evaluate on features.jsonl
    python examples/evaluate_features.py eval.jsonl \
        --train-features train.jsonl --save-model model.txt

    # Train/evaluate with a stratified split of one file
    python examples/evaluate_features.py features.jsonl \
        --train --test-size 0.3 --seed 42

With ``--models-dir``, records are routed the same way
``examples/detect_sample.py`` routes files: .NET (COM descriptor present) ->
``Dot_Net``, PE32+ / AMD64 / ARM64 -> ``Win64``, other PEs -> ``Win32``, and
non-PE or unparsable records -> ``all``.

Scores from separately trained models are not on a common scale, so routed
evaluation separates two kinds of results:

* Threshold-free metrics are reported on the pooled scores and, additionally,
  as sample-count-weighted averages of the per-type AUC/AP.
* Threshold-dependent operating points pick one threshold per model type (the
  type's own threshold at each target FPR, at 0.5, or at that type's best F1),
  apply it to that type's rows, then sum the per-type TP/FP/TN/FN into a single
  overall confusion matrix. The chosen thresholds and per-type metrics are
  printed and written to the JSON output.

All input files of a run share one sha256 set and only the first occurrence of a
record is kept, so duplicate JSONL rows (overlapping weekly files, the records
stored twice in the published test zips, hand-concatenated files) are not counted
twice. ``--keep-duplicates`` disables this. In ``--train-features`` mode the
training files are scanned first, so evaluation records whose sha256 also appears
in the training data are dropped.

The input must contain both label=0 and label=1 records. Unlabeled records
(label=-1) are ignored. Metrics are printed to stdout; LightGBM logs go to
stderr.

Only binary malicious/benign models are supported. The multiclass ``family``
model and the pickled tag models (``behavior``, ``packer``, ...) use different
label layouts and are rejected.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

import thrember
from thrember.features import PEFeatureExtractor

# Used only if examples/lgbm_config.json next to this script cannot be read.
EMBER2024_FALLBACK_TRAIN_PARAMS = {
    "boosting": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.1,
    "num_leaves": 64,
    "min_data_in_leaf": 100,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "feature_fraction": 0.9,
    "verbosity": -1,
}
EMBER_NUM_BOOST_ROUND = 1000
DEFAULT_TRAIN_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lgbm_config.json")
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "model"
FEATURE_DIM = 2568

# Per-type binary models used by --models-dir, mirroring detect_sample.py.
ROUTED_MODEL_FILES = {
    "Win32": "EMBER2024_Win32.model",
    "Win64": "EMBER2024_Win64.model",
    "Dot_Net": "EMBER2024_Dot_Net.model",
    "all": "EMBER2024_all.model",
}
FALLBACK_TYPE = "all"


def configure_lightgbm_logging(verbose: bool) -> None:
    logger = logging.getLogger("thrember_eval_lightgbm")
    logger.handlers = [logging.StreamHandler(sys.stderr)]
    logger.propagate = False
    logger.setLevel(logging.INFO if verbose else logging.ERROR)
    lgb.register_logger(logger)


def count_lines(paths: Sequence[str]) -> int:
    total = 0
    for path in paths:
        try:
            with open(path, "rb") as handle:
                for _ in handle:
                    total += 1
        except OSError as exc:
            raise SystemExit(f"cannot read {path}: {exc}") from exc
    return total


def iter_jsonl(paths: Sequence[str]) -> Iterator[Tuple[str, int, Dict[str, Any]]]:
    """Yield (path, line number, parsed object) for every JSONL line."""
    for path in paths:
        try:
            handle = open(path, encoding="utf-8")
        except OSError as exc:
            raise SystemExit(f"cannot read {path}: {exc}") from exc
        with handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    raise SystemExit(f"blank line in {path}:{line_no}")
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"invalid JSON in {path}:{line_no}: {exc}") from exc
                if not isinstance(obj, dict):
                    raise SystemExit(
                        f"record is not a JSON object at {path}:{line_no}"
                    )
                yield path, line_no, obj


def effective_label(obj: Dict[str, Any]) -> Optional[int]:
    """The binary label of a record, or None when it is missing/unlabeled."""
    label = obj.get("label")
    if isinstance(label, bool) or not isinstance(label, int):
        return None
    return label if label in (0, 1) else None


def dedup_masks(
    path_groups: Sequence[Sequence[str]],
    names: Sequence[str],
    *,
    enabled: bool = True,
) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
    """Keep only the first record for each sha256 across all input files.

    Returns one boolean keep-mask per path group, aligned with the physical lines
    that ``vectorize()`` will read, plus a JSON-serializable stats dict. With
    ``enabled=False`` the scan is skipped and no sha256 validation happens.

    Groups are scanned in order, sharing one ``seen`` map. An unlabeled occurrence
    is replaced by a later copy that has a binary label; duplicates with
    conflicting labels abort.
    """
    if len(path_groups) != len(names):
        raise ValueError("path_groups and names must have the same length")
    if not enabled:
        return None, {"enabled": False}

    masks = [bytearray() for _ in path_groups]
    seen: Dict[bytes, Tuple[int, int, Optional[int]]] = {}
    group_stats: List[Dict[str, Any]] = [
        {
            "files": list(paths),
            "records": 0,
            "kept": 0,
            "duplicates_removed": 0,
            "replaced_unlabeled": 0,
            "cross_group_duplicates": 0,
        }
        for paths in path_groups
    ]

    for group_index, paths in enumerate(path_groups):
        stats = group_stats[group_index]
        mask = masks[group_index]
        for path, line_no, obj in iter_jsonl(paths):
            raw_sha = obj.get("sha256")
            if not isinstance(raw_sha, str) or len(raw_sha) != 64:
                raise SystemExit(
                    f"missing or invalid sha256 in {path}:{line_no}; pass "
                    "--keep-duplicates to skip sha256 deduplication"
                )
            try:
                digest = bytes.fromhex(raw_sha)
            except ValueError:
                digest = b""
            if len(digest) != 32:
                raise SystemExit(
                    f"invalid sha256 in {path}:{line_no}; pass "
                    "--keep-duplicates to skip sha256 deduplication"
                )

            label = effective_label(obj)
            row = len(mask)
            stats["records"] += 1
            previous = seen.get(digest)
            if previous is None:
                seen[digest] = (group_index, row, label)
                mask.append(1)
                continue

            previous_group, previous_row, previous_label = previous
            if previous_label is None and label is not None:
                # Prefer the labeled copy: unlabeled rows are ignored downstream.
                masks[previous_group][previous_row] = 0
                group_stats[previous_group]["replaced_unlabeled"] += 1
                seen[digest] = (group_index, row, label)
                mask.append(1)
                continue

            if (
                previous_label is not None
                and label is not None
                and previous_label != label
            ):
                raise SystemExit(
                    f"label conflict for sha256 {raw_sha.lower()} in "
                    f"{path}:{line_no}: earlier record has label "
                    f"{previous_label}, this record has label {label}; run "
                    "examples/merge_features.py to resolve duplicates or pass "
                    "--keep-duplicates"
                )

            mask.append(0)
            stats["duplicates_removed"] += 1
            if previous_group != group_index:
                stats["cross_group_duplicates"] += 1

    result_masks: List[np.ndarray] = []
    for mask, stats in zip(masks, group_stats):
        stats["kept"] = int(sum(mask))
        result_masks.append(np.frombuffer(bytes(mask), dtype=np.uint8).astype(bool))

    return result_masks, {
        "enabled": True,
        "method": "sha256",
        "scope": "global-first-occurrence",
        "records": sum(stats["records"] for stats in group_stats),
        "kept": sum(stats["kept"] for stats in group_stats),
        "duplicates_removed": sum(
            stats["duplicates_removed"] for stats in group_stats
        ),
        "overlap_removed": sum(
            stats["cross_group_duplicates"] for stats in group_stats
        ),
        "unlabeled_replaced": sum(
            stats["replaced_unlabeled"] for stats in group_stats
        ),
        "groups": {
            name: group_stats[index] for index, name in enumerate(names)
        },
    }


def vectorize(
    paths: Sequence[str],
    workdir: str,
    prefix: str,
    nrows: Optional[int] = None,
) -> Tuple[np.memmap, np.memmap, PEFeatureExtractor]:
    if nrows is None:
        nrows = count_lines(paths)
    if nrows == 0:
        raise SystemExit(f"no records in {', '.join(paths)}")
    extractor = PEFeatureExtractor()
    X_path = os.path.join(workdir, f"{prefix}_X.dat")
    y_path = os.path.join(workdir, f"{prefix}_y.dat")
    try:
        thrember.vectorize_subset(
            X_path, y_path, [Path(p) for p in paths], extractor, nrows
        )
    except Exception as exc:  # noqa: BLE001 - make the error actionable
        raise SystemExit(
            "vectorization failed: "
            f"{type(exc).__name__}: {exc}\n"
            "The JSONL must be EMBER2024 feature version 3 (2568 dims) "
            "produced by examples/extract_features.py or the dataset; check "
            "that every record contains all feature keys and a binary label."
        ) from exc
    X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(nrows, extractor.dim))
    # thrember.vectorize() writes labels through an int32 memmap even though
    # vectorize_subset() preallocates the file as float32, so the file must be
    # read back as int32.
    y = np.memmap(y_path, dtype=np.int32, mode="r", shape=nrows)
    return X, y, extractor


def predict_scores(model: lgb.Booster, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n = int(np.count_nonzero(mask))
    scores = np.empty(n, dtype=np.float64)
    out = 0
    chunk_size = 100_000
    for start in range(0, X.shape[0], chunk_size):
        end = min(start + chunk_size, X.shape[0])
        chunk_mask = mask[start:end]
        if not chunk_mask.any():
            continue
        chunk = np.asarray(X[start:end])
        predictions = model.predict(chunk[chunk_mask])
        scores[out : out + len(predictions)] = predictions
        out += len(predictions)
    if out != n:
        raise RuntimeError(f"predicted {out} scores, expected {n}")
    return scores


def load_binary_model(path: str) -> lgb.Booster:
    """Load a plain LightGBM text model and reject non-binary boosters."""
    try:
        model = lgb.Booster(model_file=path)
    except Exception as exc:  # noqa: BLE001 - LightGBM raises several types
        raise SystemExit(
            f"cannot load {path} as a LightGBM model: {type(exc).__name__}: {exc}\n"
            "Only plain binary booster files are supported. The family model is "
            "multiclass and the behavior/file_property/packer/exploit/group "
            "models are pickled lists of boosters; evaluate those separately."
        ) from exc
    try:
        models_per_iteration = int(model.num_model_per_iteration())
    except Exception:  # noqa: BLE001 - older LightGBM builds
        models_per_iteration = 1
    if models_per_iteration != 1:
        raise SystemExit(
            f"{path} is a multiclass model ({models_per_iteration} trees per "
            "iteration); this script evaluates binary malicious/benign models only"
        )
    return model


def load_routed_models(models_dir: Path) -> Dict[str, lgb.Booster]:
    """Load the per-type binary models used by --models-dir."""
    models: Dict[str, lgb.Booster] = {}
    missing: List[str] = []
    for model_type, filename in ROUTED_MODEL_FILES.items():
        path = models_dir / filename
        if not path.is_file():
            missing.append(str(path))
            continue
        models[model_type] = load_binary_model(str(path))
    if missing:
        raise SystemExit(
            "model files not found:\n  "
            + "\n  ".join(missing)
            + "\nDownload them with thrember.download_models() or: hf download "
            "joyce8/EMBER2024-benchmark-models --repo-type model"
        )
    return models


WIN64_MACHINES = frozenset({"IMAGE_FILE_MACHINE_AMD64", "IMAGE_FILE_MACHINE_ARM64"})
PE32_PLUS_MAGIC = 0x20B
COM_DESCRIPTOR = "COM_DESCRIPTOR"


def record_file_type(record: Dict[str, Any]) -> str:
    """Route a raw-feature record like detect_sample.detect_file_type().

    The original bytes are not available here, so the same decision is made
    from the extracted PE features: pefile parse success, COM descriptor
    presence, machine type and optional-header magic.
    """
    general = record.get("general") or {}
    if not general.get("is_pe"):
        return FALLBACK_TYPE

    for directory in record.get("datadirectories") or []:
        if isinstance(directory, dict) and directory.get("name") == COM_DESCRIPTOR:
            if directory.get("virtual_address", 0):
                return "Dot_Net"
            break

    header = record.get("header") or {}
    machine = (header.get("coff") or {}).get("machine", "")
    magic = (header.get("optional") or {}).get("magic", 0)
    if machine in WIN64_MACHINES or magic == PE32_PLUS_MAGIC:
        return "Win64"
    return "Win32"


def read_record_types(paths: Sequence[str]) -> np.ndarray:
    """Per-record model routing keys, in the same order as the JSONL rows."""
    types: List[str] = []
    for _, _, obj in iter_jsonl(paths):
        types.append(record_file_type(obj))
    return np.asarray(types)


def predict_routed_scores(
    models: Dict[str, lgb.Booster],
    X: np.ndarray,
    mask: np.ndarray,
    types: np.ndarray,
) -> np.ndarray:
    """Score each selected row with the model chosen for its file type."""
    if len(types) != X.shape[0]:
        raise SystemExit(
            f"routing metadata covers {len(types)} records but {X.shape[0]} "
            "rows were vectorized"
        )
    n = int(np.count_nonzero(mask))
    scores = np.empty(n, dtype=np.float64)
    out = 0
    chunk_size = 100_000
    for start in range(0, X.shape[0], chunk_size):
        end = min(start + chunk_size, X.shape[0])
        rows = np.flatnonzero(mask[start:end]) + start
        if rows.size == 0:
            continue
        chunk_types = types[rows]
        chunk_scores = np.empty(rows.size, dtype=np.float64)
        for model_type, model in models.items():
            selected = chunk_types == model_type
            if not selected.any():
                continue
            chunk_scores[selected] = model.predict(np.asarray(X[rows[selected]]))
        scores[out : out + rows.size] = chunk_scores
        out += rows.size
    if out != n:
        raise RuntimeError(f"predicted {out} scores, expected {n}")
    return scores


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(y: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_pred & (y == 1)))
    fp = int(np.sum(y_pred & (y == 0)))
    tn = int(np.sum(~y_pred & (y == 0)))
    fn = int(np.sum(~y_pred & (y == 1)))
    tpr = safe_div(tp, tp + fn)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2 * precision * tpr, precision + tpr)
    accuracy = safe_div(tp + tn, tp + fp + tn + fn)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tpr,
        "fpr": fpr,
        "fnr": fnr,
        "precision": precision,
        "recall": tpr,
        "f1": f1,
        "accuracy": accuracy,
        "threshold_used": None,
    }


def threshold_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    result = classification_metrics(y, scores >= threshold)
    result["threshold_used"] = float(threshold)
    return result


def operating_point(y: np.ndarray, scores: np.ndarray, target_fpr: float) -> Dict[str, Any]:
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    target = min(max(float(target_fpr), 0.0), 1.0)
    idx = int(np.searchsorted(fpr, target, side="right") - 1)
    idx = min(max(idx, 0), len(fpr) - 1)

    achieved_fpr = float(fpr[idx])
    achieved_tpr = float(tpr[idx])
    tp = int(round(achieved_tpr * n_pos))
    fp = int(round(achieved_fpr * n_neg))
    tn = n_neg - fp
    fn = n_pos - tp
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2 * precision * achieved_tpr, precision + achieved_tpr)
    threshold = float(thresholds[idx])
    if not math.isfinite(threshold):
        threshold_out: Optional[float] = None
    else:
        threshold_out = threshold
    return {
        "target_fpr": target,
        "achieved_fpr": achieved_fpr,
        "tpr": achieved_tpr,
        "fnr": 1.0 - achieved_tpr,
        "threshold": threshold_out,
        "precision": precision,
        "recall": achieved_tpr,
        "f1": f1,
        "accuracy": safe_div(tp + tn, n_pos + n_neg),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "min_fpr_one_fp": safe_div(1, n_neg),
    }


def best_f1_point(y: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = np.zeros_like(precision)
    denom = precision + recall
    nonzero = denom > 0
    f1[nonzero] = 2 * precision[nonzero] * recall[nonzero] / denom[nonzero]
    idx = int(np.argmax(f1))
    if idx < len(thresholds):
        threshold = float(thresholds[idx])
        result = threshold_metrics(y, scores, threshold)
        result["f1"] = float(f1[idx])
        result["precision"] = float(precision[idx])
        result["recall"] = float(recall[idx])
        return result
    # Last PR point has no corresponding threshold; fall back to 0.0.
    result = threshold_metrics(y, scores, 0.0)
    result["f1"] = float(f1[idx])
    result["precision"] = float(precision[idx])
    result["recall"] = float(recall[idx])
    return result


def fpr_at_tpr(y: np.ndarray, scores: np.ndarray, target_tpr: float) -> Dict[str, Any]:
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    idx = int(np.searchsorted(tpr, target_tpr, side="left"))
    idx = min(max(idx, 0), len(tpr) - 1)
    threshold = float(thresholds[idx])
    return {
        "target_tpr": float(target_tpr),
        "achieved_tpr": float(tpr[idx]),
        "fpr": float(fpr[idx]),
        "threshold": threshold if math.isfinite(threshold) else None,
    }


def score_stats(values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {"count": 0}
    percentiles = np.percentile(values, [1, 5, 50, 95, 99])
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(percentiles[2]),
        "p1": float(percentiles[0]),
        "p5": float(percentiles[1]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    target_fprs: Sequence[float],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    aucs: List[float] = []
    aps: List[float] = []
    tprs: Dict[str, List[float]] = {f"{t:g}": [] for t in target_fprs}
    for _ in range(n_boot):
        take = np.concatenate(
            [
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            ]
        )
        yb = y[take]
        sb = scores[take]
        aucs.append(float(roc_auc_score(yb, sb)))
        aps.append(float(average_precision_score(yb, sb)))
        fpr, tpr, _ = roc_curve(yb, sb, drop_intermediate=False)
        for target in target_fprs:
            idx = int(np.searchsorted(fpr, target, side="right") - 1)
            idx = min(max(idx, 0), len(tpr) - 1)
            tprs[f"{target:g}"].append(float(tpr[idx]))

    def interval(values: Sequence[float]) -> Optional[List[float]]:
        if not values:
            return None
        low, high = np.percentile(values, [2.5, 97.5])
        return [float(low), float(high)]

    return {
        "n_boot": int(n_boot),
        "roc_auc": interval(aucs),
        "average_precision": interval(aps),
        "tpr_at_fpr": {key: interval(values) for key, values in tprs.items()},
    }


def iter_records(paths: Sequence[str]) -> Iterator[Dict[str, Any]]:
    for _, _, obj in iter_jsonl(paths):
        source = obj.get("source") or {}
        yield {
            "sha256": obj.get("sha256", ""),
            "label": obj.get("label"),
            "archive": source.get("archive", ""),
            "member": source.get("member", ""),
        }


def write_scores_csv(
    path: str,
    records: Iterable[Dict[str, Any]],
    scores: np.ndarray,
    indices: np.ndarray,
    split: str,
) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    order = np.argsort(indices)
    sorted_indices = indices[order]
    sorted_scores = scores[order]
    ptr = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sha256", "label", "score", "split", "archive", "member"])
        for record_index, record in enumerate(records):
            if ptr >= len(sorted_indices):
                break
            if record_index == int(sorted_indices[ptr]):
                writer.writerow(
                    [
                        record.get("sha256", ""),
                        record.get("label", ""),
                        f"{float(sorted_scores[ptr]):.10g}",
                        split,
                        record.get("archive", ""),
                        record.get("member", ""),
                    ]
                )
                ptr += 1


def load_train_params(path: Optional[str]) -> Tuple[Dict[str, Any], int]:
    params = dict(EMBER2024_FALLBACK_TRAIN_PARAMS)
    num_boost_round = EMBER_NUM_BOOST_ROUND
    if path is None and os.path.isfile(DEFAULT_TRAIN_CONFIG):
        # Match the configuration used to train the benchmark models.
        path = DEFAULT_TRAIN_CONFIG
    if path:
        try:
            with open(path, encoding="utf-8") as handle:
                user_params = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read {path}: {exc}") from exc
        if not isinstance(user_params, dict):
            raise SystemExit("--train-params-json must contain a JSON object")
        if "num_iterations" in user_params:
            num_boost_round = int(user_params.pop("num_iterations"))
        params.update(user_params)
    params.pop("num_iterations", None)
    return params, num_boost_round


def format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "inf"
    return f"{value:.{digits}f}"


def print_operating_points(points: List[Tuple[str, Dict[str, Any]]]) -> None:
    header = (
        f"{'point':<18}{'target':>8}{'FPR':>9}{'threshold':>11}"
        f"{'TPR':>9}{'FNR':>9}{'precision':>11}{'F1':>9}"
        f"{'TP':>7}{'FP':>7}{'TN':>7}{'FN':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, point in points:
        if name in ("threshold 0.5", "best F1"):
            target_text = "-"
        else:
            target = point.get("target_fpr", point.get("target_tpr"))
            if isinstance(target, float):
                target_text = f"{target:.4f}"
            elif target is None:
                target_text = "-"
            else:
                target_text = str(target)
        if "thresholds" in point:
            # Routed evaluation: each model type has its own threshold.
            threshold_text = "per-type"
        else:
            threshold = point.get("threshold", point.get("threshold_used"))
            if threshold is None and "threshold" in point:
                threshold_text = "inf"
            else:
                threshold_text = format_float(threshold, 4)
        print(
            f"{name:<18}{target_text:>8}"
            f"{point.get('fpr', point.get('achieved_fpr', 0.0)):>9.4f}"
            f"{threshold_text:>11}"
            f"{point['tpr']:>9.4f}"
            f"{point['fnr']:>9.4f}"
            f"{point.get('precision', 0.0):>11.4f}"
            f"{point.get('f1', 0.0):>9.4f}"
            f"{point.get('tp', 0):>7d}"
            f"{point.get('fp', 0):>7d}"
            f"{point.get('tn', 0):>7d}"
            f"{point.get('fn', 0):>7d}"
        )


def evaluate(
    y: np.ndarray,
    scores: np.ndarray,
    target_fprs: Sequence[float],
    partial_auc_max_fpr: float,
) -> Dict[str, Any]:
    if len(y) != len(scores):
        raise SystemExit(f"label/score length mismatch: {len(y)} vs {len(scores)}")
    if not np.isfinite(scores).all():
        raise SystemExit("model produced non-finite scores")
    if len(np.unique(y)) < 2:
        raise SystemExit("need both label=0 and label=1 records to evaluate")
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    eps = 1e-15
    clipped = np.clip(scores.astype(np.float64), eps, 1 - eps)
    metrics: Dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "brier_score": float(brier_score_loss(y, scores)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "prevalence": safe_div(n_pos, n_pos + n_neg),
    }
    if partial_auc_max_fpr is not None and 0 < partial_auc_max_fpr < 1:
        try:
            partial_auc = float(roc_auc_score(y, scores, max_fpr=partial_auc_max_fpr))
            metrics["partial_auc"] = partial_auc if math.isfinite(partial_auc) else None
            metrics["partial_auc_max_fpr"] = float(partial_auc_max_fpr)
        except ValueError:
            metrics["partial_auc"] = None
            metrics["partial_auc_max_fpr"] = float(partial_auc_max_fpr)
    metrics["fpr_at_tpr_0.95"] = fpr_at_tpr(y, scores, 0.95)

    points: List[Tuple[str, Dict[str, Any]]] = []
    for target in target_fprs:
        point = operating_point(y, scores, target)
        point["name"] = f"target_fpr_{target:g}"
        points.append((f"target FPR {target:g}", point))
    point = threshold_metrics(y, scores, 0.5)
    point["name"] = "threshold_0.5"
    points.append(("threshold 0.5", point))
    point = best_f1_point(y, scores)
    point["name"] = "best_f1"
    points.append(("best F1", point))
    metrics["operating_points"] = [point for _, point in points]
    return {
        "metrics": metrics,
        "operating_points_named": points,
        "counts": {"positive": n_pos, "negative": n_neg},
        "score_stats": {
            "positive": score_stats(scores[y == 1]),
            "negative": score_stats(scores[y == 0]),
        },
    }


def json_threshold(value: Optional[float]) -> Optional[float]:
    """Represent an infinite/absent threshold as null in JSON output."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def type_threshold_for_fpr(y: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Threshold for one model type at a target FPR.

    Scores from different models are not on a common scale, so each type gets
    its own threshold. Types without both classes fall back to 0.5.
    """
    if not (np.any(y == 0) and np.any(y == 1)):
        return 0.5
    threshold = operating_point(y, scores, target_fpr)["threshold"]
    # operating_point() reports an infinite threshold as None.
    return math.inf if threshold is None else float(threshold)


def type_best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    if not (np.any(y == 0) and np.any(y == 1)):
        return 0.5
    threshold = best_f1_point(y, scores).get("threshold_used")
    return 0.5 if threshold is None else float(threshold)


def aggregate_thresholds(
    labels: np.ndarray,
    scores: np.ndarray,
    types: np.ndarray,
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """Apply one threshold per model type and aggregate the confusion counts."""
    predictions = np.zeros(len(labels), dtype=bool)
    for model_type, threshold in thresholds.items():
        selected = types == model_type
        if selected.any():
            predictions[selected] = scores[selected] >= threshold
    result = classification_metrics(labels, predictions)
    result["thresholds"] = {
        model_type: json_threshold(threshold)
        for model_type, threshold in thresholds.items()
    }
    return result


def evaluate_routed(
    labels: np.ndarray,
    scores: np.ndarray,
    types: np.ndarray,
    target_fprs: Sequence[float],
    partial_auc_max_fpr: float,
) -> Dict[str, Any]:
    """Overall evaluation when samples were scored by different models.

    Threshold-free metrics (ROC AUC, AP, Brier, log loss) are computed on the
    pooled scores. Threshold-dependent operating points are built by choosing a
    threshold per model type and summing the per-type confusion counts, so each
    model is thresholded on its own score scale. Type-weighted AUC/AP are also
    reported because pooling scores from separately trained models can distort
    the global ranking.
    """
    base = evaluate(labels, scores, target_fprs, partial_auc_max_fpr)
    metrics = dict(base["metrics"])
    present_types = [mt for mt in ROUTED_MODEL_FILES if np.any(types == mt)]

    per_type: Dict[str, Dict[str, Any]] = {}
    fpr_thresholds: Dict[str, Dict[float, float]] = {}
    best_f1_thresholds: Dict[str, float] = {}
    for model_type in ROUTED_MODEL_FILES:
        selected = types == model_type
        n = int(np.count_nonzero(selected))
        entry: Dict[str, Any] = {
            "count": n,
            "positive": 0,
            "negative": 0,
            "roc_auc": None,
            "average_precision": None,
            "threshold_for_target_fpr": {},
            "best_f1_threshold": None,
        }
        numeric_targets: Dict[float, float] = {}
        if n:
            y_t = labels[selected]
            s_t = scores[selected]
            n_pos_t = int(np.sum(y_t == 1))
            n_neg_t = int(np.sum(y_t == 0))
            entry["positive"] = n_pos_t
            entry["negative"] = n_neg_t
            if n_pos_t and n_neg_t:
                entry["roc_auc"] = float(roc_auc_score(y_t, s_t))
                entry["average_precision"] = float(
                    average_precision_score(y_t, s_t)
                )
            for target in target_fprs:
                numeric_targets[target] = type_threshold_for_fpr(y_t, s_t, target)
            best_f1_thresholds[model_type] = type_best_f1_threshold(y_t, s_t)
            entry["threshold_for_target_fpr"] = {
                f"{target:g}": json_threshold(threshold)
                for target, threshold in numeric_targets.items()
            }
            entry["best_f1_threshold"] = json_threshold(
                best_f1_thresholds[model_type]
            )
        fpr_thresholds[model_type] = numeric_targets
        per_type[model_type] = entry

    points: List[Tuple[str, Dict[str, Any]]] = []
    for target in target_fprs:
        thresholds = {
            model_type: fpr_thresholds[model_type][target]
            for model_type in present_types
        }
        point = aggregate_thresholds(labels, scores, types, thresholds)
        point["name"] = f"target_fpr_{target:g}"
        point["target_fpr"] = float(target)
        point["achieved_fpr"] = point["fpr"]
        points.append((f"target FPR {target:g}", point))
    point = aggregate_thresholds(
        labels, scores, types, {mt: 0.5 for mt in present_types}
    )
    point["name"] = "threshold_0.5"
    points.append(("threshold 0.5", point))
    point = aggregate_thresholds(
        labels,
        scores,
        types,
        {mt: best_f1_thresholds[mt] for mt in present_types},
    )
    point["name"] = "best_f1"
    points.append(("best F1", point))

    def weighted_mean(key: str) -> Optional[float]:
        pairs = [
            (entry["count"], entry[key])
            for entry in per_type.values()
            if entry["count"] and entry[key] is not None
        ]
        total = sum(count for count, _ in pairs)
        if not total:
            return None
        return float(sum(count * value for count, value in pairs) / total)

    metrics["type_weighted_roc_auc"] = weighted_mean("roc_auc")
    metrics["type_weighted_average_precision"] = weighted_mean("average_precision")
    metrics["operating_points"] = [point for _, point in points]
    return {
        "metrics": metrics,
        "operating_points_named": points,
        "counts": base["counts"],
        "score_stats": base["score_stats"],
        "per_type": per_type,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate EMBER2024 (feature version 3) raw-feature JSONL with a "
            "binary LightGBM model."
        )
    )
    parser.add_argument("features", nargs="+", help="feature JSONL file(s) to evaluate")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model",
        help=(
            "pretrained binary LightGBM model (.txt), e.g. "
            "model/EMBER2024_all.model or one trained by train_lgbm.py"
        ),
    )
    source.add_argument(
        "--models-dir",
        nargs="?",
        const=str(DEFAULT_MODELS_DIR),
        default=None,
        metavar="DIR",
        help=(
            "route each record to its Win32/Win64/Dot_Net binary model "
            "(Dot_Net if a COM descriptor exists, Win64 for PE32+/AMD64/ARM64, "
            "Win32 otherwise) and fall back to the all model for non-PE or "
            f"unparsable records (default dir: {DEFAULT_MODELS_DIR})"
        ),
    )
    source.add_argument(
        "--train",
        action="store_true",
        help="train on a stratified split of the input and evaluate on the rest",
    )
    source.add_argument(
        "--train-features",
        action="append",
        help="train on this JSONL file (repeatable), evaluate on the positional features",
    )
    parser.add_argument(
        "--target-fpr",
        nargs="+",
        type=float,
        default=[0.01, 0.001],
        help="target false-positive rates (default: 0.01 0.001)",
    )
    parser.add_argument(
        "--partial-auc-max-fpr",
        type=float,
        default=0.005,
        help="partial ROC AUC restricted to FPR <= this value (default: 0.005; 0 disables)",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help=(
            "keep records whose sha256 duplicates an earlier record; by default "
            "all input files of the run share one sha256 set and only the first "
            "occurrence is kept"
        ),
    )
    parser.add_argument("--test-size", type=float, default=0.3, help="test fraction for --train")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--train-params-json",
        help=(
            "LightGBM training config JSON (default: examples/lgbm_config.json, "
            "the config used for the benchmark models)"
        ),
    )
    parser.add_argument("--save-model", help="save a model trained by --train/--train-features")
    parser.add_argument("--json-out", help="write all metrics as JSON")
    parser.add_argument("--scores-out", help="write per-sample scores as CSV")
    parser.add_argument("--bootstrap", type=int, default=0, help="bootstrap resamples for 95%% CI")
    parser.add_argument("--workdir", help="directory for vectorized memmaps (default: temporary)")
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="keep the vectorized memmaps after evaluation",
    )
    parser.add_argument("--verbose", action="store_true", help="show LightGBM logs")
    return parser


def run(args: argparse.Namespace) -> int:
    if any(target < 0 or target > 1 for target in args.target_fpr):
        raise SystemExit("--target-fpr values must be in [0, 1]")
    if not (0 < args.test_size < 1):
        raise SystemExit("--test-size must be between 0 and 1")
    if args.save_model and not (args.train or args.train_features):
        raise SystemExit("--save-model requires --train or --train-features")

    configure_lightgbm_logging(args.verbose)

    if args.workdir:
        os.makedirs(args.workdir, mode=0o700, exist_ok=True)
        workdir = tempfile.mkdtemp(prefix="thrember_eval_", dir=args.workdir)
    else:
        workdir = tempfile.mkdtemp(prefix="thrember_eval_")
    remove_workdir = not args.keep_workdir
    routing_counts: Optional[Dict[str, int]] = None
    routed_models_dir: Optional[Path] = None
    eval_types: Optional[np.ndarray] = None
    dedup_info: Optional[Dict[str, Any]] = None
    try:
        if args.model:
            if not os.path.exists(args.model):
                raise SystemExit(f"model does not exist: {args.model}")
            keep_masks, dedup_info = dedup_masks(
                [args.features], ["eval"], enabled=not args.keep_duplicates
            )
            keep = None if keep_masks is None else keep_masks[0]
            X, y, extractor = vectorize(
                args.features,
                workdir,
                "eval",
                nrows=None if keep is None else len(keep),
            )
            mask = np.asarray(y) != -1
            if keep is not None:
                mask &= keep
            eval_indices = np.flatnonzero(mask)
            labels = np.asarray(y[eval_indices], dtype=np.int64)
            if labels.size == 0:
                raise SystemExit("no labeled records (label 0/1) in the input")
            model = load_binary_model(args.model)
            if model.num_feature() != X.shape[1]:
                raise SystemExit(
                    f"model expects {model.num_feature()} features but the JSONL "
                    f"produced {X.shape[1]}; the input must be EMBER2024 feature "
                    "version 3 (2568 dims)"
                )
            scores = predict_scores(model, X, mask)
            train_info = None
            eval_files = args.features
            eval_split = "eval"
        elif args.models_dir:
            routed_models_dir = Path(args.models_dir)
            keep_masks, dedup_info = dedup_masks(
                [args.features], ["eval"], enabled=not args.keep_duplicates
            )
            keep = None if keep_masks is None else keep_masks[0]
            X, y, extractor = vectorize(
                args.features,
                workdir,
                "eval",
                nrows=None if keep is None else len(keep),
            )
            mask = np.asarray(y) != -1
            if keep is not None:
                mask &= keep
            eval_indices = np.flatnonzero(mask)
            labels = np.asarray(y[eval_indices], dtype=np.int64)
            if labels.size == 0:
                raise SystemExit("no labeled records (label 0/1) in the input")
            models = load_routed_models(routed_models_dir)
            for model_type, loaded in models.items():
                if loaded.num_feature() != X.shape[1]:
                    raise SystemExit(
                        f"{ROUTED_MODEL_FILES[model_type]} expects "
                        f"{loaded.num_feature()} features but the JSONL produced "
                        f"{X.shape[1]}; the input must be EMBER2024 feature "
                        "version 3 (2568 dims)"
                    )
            types = read_record_types(args.features)
            scores = predict_routed_scores(models, X, mask, types)
            eval_types = types[eval_indices]
            routing_counts = {
                model_type: int(np.count_nonzero(eval_types == model_type))
                for model_type in ROUTED_MODEL_FILES
            }
            # Types with no records are not passed to the evaluator.
            train_info = None
            eval_files = args.features
            eval_split = "eval"
        elif args.train:
            keep_masks, dedup_info = dedup_masks(
                [args.features], ["input"], enabled=not args.keep_duplicates
            )
            keep = None if keep_masks is None else keep_masks[0]
            X_all, y_all, extractor = vectorize(
                args.features,
                workdir,
                "all",
                nrows=None if keep is None else len(keep),
            )
            label_mask = np.asarray(y_all) != -1
            if keep is not None:
                label_mask &= keep
            labeled_positions = np.flatnonzero(label_mask)
            labels_all = np.asarray(y_all[labeled_positions], dtype=np.int64)
            if labels_all.size == 0:
                raise SystemExit("no labeled records (label 0/1) in the input")
            try:
                train_pos, test_pos = train_test_split(
                    labeled_positions,
                    test_size=args.test_size,
                    random_state=args.seed,
                    stratify=labels_all,
                )
            except ValueError as exc:
                raise SystemExit(
                    f"cannot create a stratified train/test split: {exc}; "
                    "use --train-features or a larger dataset"
                ) from exc
            X_train = np.asarray(X_all[train_pos])
            y_train = np.asarray(y_all[train_pos], dtype=np.int64)
            X_test = np.asarray(X_all[test_pos])
            y_test = np.asarray(y_all[test_pos], dtype=np.int64)
            params, num_boost_round = load_train_params(args.train_params_json)
            model = lgb.train(
                params,
                lgb.Dataset(X_train, label=y_train),
                num_boost_round=num_boost_round,
            )
            if args.save_model:
                model.save_model(args.save_model)
            eval_indices = np.asarray(test_pos, dtype=np.int64)
            labels = y_test
            scores = model.predict(X_test)
            train_info = {
                "train_count": int(len(train_pos)),
                "test_count": int(len(test_pos)),
                "train_params": params,
                "num_boost_round": int(num_boost_round),
            }
            eval_files = args.features
            eval_split = "test"
        else:
            keep_masks, dedup_info = dedup_masks(
                [args.train_features, args.features],
                ["train", "eval"],
                enabled=not args.keep_duplicates,
            )
            train_keep = None if keep_masks is None else keep_masks[0]
            eval_keep = None if keep_masks is None else keep_masks[1]
            X_train_mm, y_train_mm, extractor = vectorize(
                args.train_features,
                workdir,
                "train",
                nrows=None if train_keep is None else len(train_keep),
            )
            train_mask = np.asarray(y_train_mm) != -1
            if train_keep is not None:
                train_mask &= train_keep
            train_pos = np.flatnonzero(train_mask)
            X_train = np.asarray(X_train_mm[train_pos])
            y_train = np.asarray(y_train_mm[train_pos], dtype=np.int64)
            if len(np.unique(y_train)) < 2:
                raise SystemExit("training features need both label=0 and label=1")
            X_eval_mm, y_eval_mm, _ = vectorize(
                args.features,
                workdir,
                "eval",
                nrows=None if eval_keep is None else len(eval_keep),
            )
            eval_mask = np.asarray(y_eval_mm) != -1
            if eval_keep is not None:
                eval_mask &= eval_keep
            eval_indices = np.flatnonzero(eval_mask)
            labels = np.asarray(y_eval_mm[eval_indices], dtype=np.int64)
            X_eval = np.asarray(X_eval_mm[eval_indices])
            if labels.size == 0:
                raise SystemExit("no labeled records (label 0/1) in the evaluation input")
            params, num_boost_round = load_train_params(args.train_params_json)
            model = lgb.train(
                params,
                lgb.Dataset(X_train, label=y_train),
                num_boost_round=num_boost_round,
            )
            if args.save_model:
                model.save_model(args.save_model)
            scores = model.predict(X_eval)
            train_info = {
                "train_count": int(len(y_train)),
                "test_count": int(len(labels)),
                "train_params": params,
                "num_boost_round": int(num_boost_round),
            }
            eval_files = args.features
            eval_split = "eval"

        if labels.size == 0:
            raise SystemExit("no labeled records to evaluate")
        if eval_types is not None:
            results = evaluate_routed(
                labels, scores, eval_types, args.target_fpr, args.partial_auc_max_fpr
            )
        else:
            results = evaluate(
                labels, scores, args.target_fpr, args.partial_auc_max_fpr
            )
        n_pos = results["counts"]["positive"]
        n_neg = results["counts"]["negative"]
        warnings: List[str] = []
        if dedup_info and dedup_info.get("enabled"):
            if dedup_info["duplicates_removed"]:
                warnings.append(
                    f"deduplicated {dedup_info['duplicates_removed']} duplicate "
                    f"record(s) by sha256 (kept {dedup_info['kept']} of "
                    f"{dedup_info['records']}); pass --keep-duplicates to disable"
                )
            if dedup_info["overlap_removed"]:
                warnings.append(
                    f"removed {dedup_info['overlap_removed']} evaluation record(s) "
                    "whose sha256 also appeared in the training features"
                )
            if dedup_info["unlabeled_replaced"]:
                warnings.append(
                    f"{dedup_info['unlabeled_replaced']} earlier unlabeled "
                    "duplicate(s) were replaced by a later labeled copy"
                )
        if n_pos < 30 or n_neg < 30:
            warnings.append(
                f"small evaluation set (positive={n_pos}, negative={n_neg}); "
                "metrics are unstable"
            )
        for target in args.target_fpr:
            if n_neg > 0 and 0 < target < 1.0 / n_neg:
                warnings.append(
                    f"target FPR {target:g} is below the resolution of "
                    f"1/negative={1.0 / n_neg:.6g}; achieved FPR will be 0"
                )
        if args.bootstrap > 0:
            results["bootstrap"] = bootstrap_ci(
                labels, scores, args.target_fpr, args.bootstrap, args.seed
            )

        if args.model:
            model_info: Dict[str, Any] = {"source": "pretrained", "path": args.model}
        elif routed_models_dir is not None:
            model_info = {
                "source": "routed",
                "path": str(routed_models_dir),
                "files": dict(ROUTED_MODEL_FILES),
            }
        else:
            model_info = {"source": "trained", "path": args.save_model or None}

        summary = {
            "input": list(args.features),
            "dedup": dedup_info or {"enabled": False},
            "feature_version": 3,
            "feature_dim": int(extractor.dim),
            "model": model_info,
            "training": train_info,
            "counts": {
                "evaluated": int(len(labels)),
                **results["counts"],
            },
            "metrics": results["metrics"],
            "score_stats": results["score_stats"],
            "warnings": warnings,
        }
        if routing_counts is not None and "per_type" in results:
            summary["routing"] = {
                "counts": routing_counts,
                "per_type": results["per_type"],
            }
            warnings.append(
                "threshold-free metrics (ROC AUC, AP, partial AUC, FPR@TPR, "
                "bootstrap) pool scores from different models; compare them with "
                "type-weighted AUC/AP and the per-type table"
            )
            for model_type, count in routing_counts.items():
                if count == 0:
                    warnings.append(
                        f"no evaluated records routed to {model_type}; "
                        f"{ROUTED_MODEL_FILES[model_type]} was not used"
                    )
                    continue
                entry = results["per_type"][model_type]
                if entry["positive"] == 0 or entry["negative"] == 0:
                    warnings.append(
                        f"{model_type}: eval set has positive={entry['positive']}, "
                        f"negative={entry['negative']}; AUC/AP are undefined and "
                        "its threshold falls back to 0.5"
                    )
                elif entry["count"] < 30:
                    warnings.append(
                        f"{model_type}: only {entry['count']} evaluated records"
                    )
        if args.bootstrap > 0:
            summary["bootstrap"] = results.get("bootstrap", {})

        if args.scores_out:
            write_scores_csv(
                args.scores_out,
                iter_records(eval_files),
                np.asarray(scores),
                eval_indices,
                eval_split,
            )

        if args.json_out:
            parent = os.path.dirname(os.path.abspath(args.json_out))
            os.makedirs(parent, exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)

        print(f"Input: {', '.join(args.features)}")
        if dedup_info and dedup_info.get("enabled"):
            print(
                f"Dedup: kept {dedup_info['kept']} of {dedup_info['records']} "
                f"records ({dedup_info['duplicates_removed']} duplicates removed "
                "by sha256)"
            )
            if dedup_info["overlap_removed"]:
                print(
                    f"Dedup: {dedup_info['overlap_removed']} evaluation record(s) "
                    "also present in the training features were removed"
                )
        print(
            f"Model: {summary['model']['source']}"
            + (f" ({args.model})" if args.model else "")
            + (f" ({routed_models_dir})" if routed_models_dir is not None else "")
        )
        if train_info:
            print(
                f"Training: {train_info['train_count']} samples; "
                f"evaluating on {train_info['test_count']} samples"
            )
        print(
            f"Samples: {len(labels)} labeled "
            f"(positive={n_pos}, negative={n_neg}, "
            f"prevalence={n_pos / len(labels):.4f})"
        )
        if routing_counts is not None:
            print(
                "Routing: "
                + ", ".join(f"{k}={v}" for k, v in routing_counts.items())
            )
        metrics = results["metrics"]
        routed = "per_type" in results
        pooled_suffix = " (pooled scores)" if routed else ""
        print(f"ROC AUC{pooled_suffix}: {metrics['roc_auc']:.6f}")
        if metrics.get("partial_auc") is not None:
            print(
                f"Partial AUC (FPR<={metrics['partial_auc_max_fpr']:g})"
                f"{pooled_suffix}: {metrics['partial_auc']:.6f}"
            )
        print(
            f"PR AUC / average precision{pooled_suffix}: "
            f"{metrics['average_precision']:.6f}"
        )
        if metrics.get("type_weighted_roc_auc") is not None:
            print(
                f"Type-weighted ROC AUC: {metrics['type_weighted_roc_auc']:.6f}; "
                f"type-weighted AP: "
                f"{metrics['type_weighted_average_precision']:.6f}"
            )
        print(f"Brier score: {metrics['brier_score']:.6f}")
        print(f"Log loss: {metrics['log_loss']:.6f}")
        fpr95 = metrics["fpr_at_tpr_0.95"]
        print(
            f"FPR at TPR>=0.95: {fpr95['fpr']:.6f} "
            f"(achieved TPR={fpr95['achieved_tpr']:.6f}, "
            f"threshold={format_float(fpr95['threshold'])})"
        )
        if args.bootstrap > 0:
            boot = summary["bootstrap"]
            print(
                f"Bootstrap 95% CI ({boot['n_boot']} resamples): "
                f"ROC AUC {boot['roc_auc']}, AP {boot['average_precision']}"
            )
            for target, interval in boot.get("tpr_at_fpr", {}).items():
                print(f"  TPR at FPR {target}: {interval}")
        if routed:
            target_labels = [f"{target:g}" for target in args.target_fpr]
            header = (
                f"{'type':<9}{'n':>7}{'pos':>7}{'neg':>7}"
                f"{'AUC':>9}{'AP':>9}{'bestF1thr':>11}"
                + "".join(f"{'thr@' + label:>12}" for label in target_labels)
            )
            print()
            print("Per-type results (thresholds are per model):")
            print(header)
            print("-" * len(header))
            for model_type, entry in results["per_type"].items():
                if entry["count"] == 0:
                    continue
                roc_auc_text = (
                    "-" if entry["roc_auc"] is None else f"{entry['roc_auc']:.4f}"
                )
                average_precision_text = (
                    "-"
                    if entry["average_precision"] is None
                    else f"{entry['average_precision']:.4f}"
                )
                target_thresholds = "".join(
                    f"{format_float(entry['threshold_for_target_fpr'].get(label), 4):>12}"
                    for label in target_labels
                )
                print(
                    f"{model_type:<9}{entry['count']:>7}{entry['positive']:>7}"
                    f"{entry['negative']:>7}{roc_auc_text:>9}"
                    f"{average_precision_text:>9}"
                    f"{format_float(entry['best_f1_threshold'], 4):>11}"
                    f"{target_thresholds}"
                )
        print()
        if routed:
            print("Operating points (each model type thresholded independently):")
        else:
            print("Operating points:")
        print_operating_points(results["operating_points_named"])
        print()
        print("Score distributions:")
        for label_name in ("positive", "negative"):
            stats = results["score_stats"][label_name]
            print(
                f"  {label_name:<8} n={stats['count']} mean={stats['mean']:.6f} "
                f"median={stats['median']:.6f} p1={stats['p1']:.6f} "
                f"p5={stats['p5']:.6f} p95={stats['p95']:.6f} "
                f"p99={stats['p99']:.6f}"
            )
        if warnings:
            print()
            print("Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        if args.json_out:
            print()
            print(f"Metrics JSON: {args.json_out}")
        if args.scores_out:
            print(f"Per-sample scores: {args.scores_out}")
    finally:
        if remove_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        elif args.keep_workdir:
            print(f"Workdir kept: {workdir}", file=sys.stderr)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
