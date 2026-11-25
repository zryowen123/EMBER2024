#!/usr/bin/env python3
"""
Compare two binary LightGBM models on the same vectorized EMBER2024 test data.

Both models are evaluated on exactly the same rows with the same protocol:

* test set: all rows of ``X_test.dat`` / ``y_test.dat``
* challenge protocol: benign test rows (label 0) + all ``X_challenge.dat`` rows

Reported metrics: ROC AUC, PR AUC, TPR at fixed FPR (default 1% and 0.1%),
plus score correlation and model structure. Prediction is chunked, so X never
has to be fully materialized.

Usage:
    python examples/compare_models.py <data_dir> <model_a> <model_b> \
        [--names official train] [--chunk-size 50000] [--json result.json]

``data_dir`` must contain X_test.dat, y_test.dat, X_challenge.dat, y_challenge.dat.
"""

import argparse
import json
import os
import sys

import lightgbm as lgb
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

import thrember


def open_X(path: str, dim: int) -> np.memmap:
    X = np.memmap(path, dtype=np.float32, mode="r")
    if X.size % dim != 0:
        raise SystemExit(f"{path} size {X.size} is not a multiple of dim {dim}")
    return X.reshape(X.size // dim, dim)


def open_y(path: str, nrows: int) -> np.ndarray:
    y = np.memmap(path, dtype=np.int32, mode="r")
    if y.size != nrows:
        raise SystemExit(f"{path} has {y.size} labels for {nrows} rows")
    return np.asarray(y)


def predict(booster, X, rows, chunk_size, num_iteration):
    n = len(X) if rows is None else len(rows)
    scores = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        batch = np.asarray(X[start:end]) if rows is None else np.asarray(X[rows[start:end]])
        scores[start:end] = booster.predict(batch, num_iteration=num_iteration)
    return scores


def metrics(y, scores, fpr_targets):
    y = np.asarray(y)
    roc_auc = float(roc_auc_score(y, scores))
    precision, recall, _ = precision_recall_curve(y, scores)
    pr_auc = float(auc(recall, precision))
    fpr, tpr, _ = roc_curve(y, scores)
    out = {"roc_auc": roc_auc, "pr_auc": pr_auc}
    for target in fpr_targets:
        # Interpolate TPR at the requested FPR (fpr is non-decreasing).
        out[f"tpr@fpr={target:g}"] = float(np.interp(target, fpr, tpr))
    return out


def model_info(path: str) -> dict:
    booster = lgb.Booster(model_file=path)
    if booster.num_model_per_iteration() != 1:
        raise SystemExit(f"{path} is not a binary model")
    dump = booster.dump_model()
    leaves = [tree["num_leaves"] for tree in dump["tree_info"]]
    return {
        "path": path,
        "file_mb": os.path.getsize(path) / 1e6,
        "num_trees": booster.num_trees(),
        "num_features": booster.num_feature(),
        "avg_leaves": float(np.mean(leaves)) if leaves else 0.0,
        "max_leaves": int(max(leaves)) if leaves else 0,
        "best_iteration": int(getattr(booster, "best_iteration", -1) or -1),
    }, booster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    parser.add_argument("model_a")
    parser.add_argument("model_b")
    parser.add_argument("--names", nargs=2, default=None, help="labels for the two models")
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--fpr", nargs="+", type=float, default=[0.01, 0.001])
    parser.add_argument("--json", default=None, help="write results to this JSON file")
    args = parser.parse_args()

    dim = thrember.PEFeatureExtractor().dim
    data_dir = args.data_dir
    X_test = open_X(os.path.join(data_dir, "X_test.dat"), dim)
    y_test = open_y(os.path.join(data_dir, "y_test.dat"), X_test.shape[0])
    X_chal = open_X(os.path.join(data_dir, "X_challenge.dat"), dim)
    y_chal = open_y(os.path.join(data_dir, "y_challenge.dat"), X_chal.shape[0])

    benign_rows = np.flatnonzero(y_test == 0)
    y_benign = np.zeros(len(benign_rows), dtype=np.int32)
    y_challenge = np.concatenate([y_benign, np.asarray(y_chal)])
    print(
        f"test rows: {len(y_test):,} (benign {len(benign_rows):,})  "
        f"challenge rows: {len(y_chal):,}"
    )

    names = args.names or [os.path.basename(args.model_a), os.path.basename(args.model_b)]
    results = {}
    for name, path in zip(names, [args.model_a, args.model_b]):
        info, booster = model_info(path)
        if info["num_features"] != dim:
            raise SystemExit(
                f"{path} expects {info['num_features']} features, data has {dim}"
            )
        num_iteration = info["best_iteration"] if info["best_iteration"] > 0 else None
        scores_test = predict(booster, X_test, None, args.chunk_size, num_iteration)
        scores_chal = predict(booster, X_chal, None, args.chunk_size, num_iteration)
        scores_benign = predict(
            booster, X_test, benign_rows, args.chunk_size, num_iteration
        )
        scores_combined = np.concatenate([scores_benign, scores_chal])

        info["test"] = metrics(y_test, scores_test, args.fpr)
        info["challenge"] = metrics(y_challenge, scores_combined, args.fpr)
        info["scores_test"] = scores_test
        results[name] = info

    a, b = names
    corr = float(np.corrcoef(results[a]["scores_test"], results[b]["scores_test"])[0, 1])

    rows = []
    for group in ("test", "challenge"):
        for metric in results[a][group]:
            va = results[a][group][metric]
            vb = results[b][group][metric]
            rows.append((group, metric, va, vb, vb - va))
    rows.append(("test", "score_correlation", corr, corr, 0.0))

    print()
    print(f"{'':10} {'metric':22} {a:>12} {b:>12} {'delta(b-a)':>12}")
    for group, metric, va, vb, delta in rows:
        print(f"{group:10} {metric:22} {va:12.6f} {vb:12.6f} {delta:+12.6f}")

    print()
    print(f"{'':10} {'structure':22} {a:>12} {b:>12}")
    for key in ("num_trees", "num_features", "avg_leaves", "max_leaves", "file_mb"):
        va, vb = results[a][key], results[b][key]
        print(f"{'model':10} {key:22} {va:12.2f} {vb:12.2f}")

    if args.json:
        out = {
            name: {k: v for k, v in info.items() if k != "scores_test"}
            for name, info in results.items()
        }
        out["score_correlation"] = corr
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
