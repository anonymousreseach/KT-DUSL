#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Metrics (same style as your evaluator)
# -------------------------

def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """Rank-based AUC (no sklearn). Returns None if only one class."""
    if len(np.unique(y_true)) < 2:
        return None

    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1, dtype=float)

    # average ranks for ties
    sorted_p = y_prob[order]
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            avg_rank = float(np.mean(ranks[order[i:j + 1]]))
            ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    pos = (y_true == 1)
    n_pos = int(np.sum(pos))
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None

    sum_ranks_pos = float(np.sum(ranks[pos]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def accuracy_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> float:
    y_pred = (y_prob >= float(thr)).astype(int)
    return float(np.mean(y_pred == y_true))


def f1_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> float:
    y_pred = (y_prob >= float(thr)).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float((2 * tp) / denom)


def logloss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


# -------------------------
# Loading + merging
# -------------------------

def load_all_predictions(pred_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(pred_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {pred_dir} matching {pattern}")

    dfs: List[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        df["__source_file__"] = f.name
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    # Defensive: keep only required columns if present
    required = ["IsCorrect", "p_leaf_mean", "p_leaf_sl", "p_prop_mean", "p_prop_sl"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"Missing columns in merged predictions: {missing}\nAvailable: {list(out.columns)}")

    # Ensure types
    out["IsCorrect"] = out["IsCorrect"].astype(int)
    for c in required[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    bad = out[required].isna().any(axis=1).sum()
    if bad > 0:
        # Drop rows with NaNs in essential columns
        out = out.dropna(subset=required).reset_index(drop=True)

    return out


def compute_global(df: pd.DataFrame, acc_threshold: float) -> pd.DataFrame:
    y = df["IsCorrect"].to_numpy(dtype=int)

    method_cols: Dict[str, str] = {
        "leaf_mean": "p_leaf_mean",
        "leaf_sl": "p_leaf_sl",
        "prop_mean": "p_prop_mean",
        "prop_sl": "p_prop_sl",
    }

    rows = []
    for method, col in method_cols.items():
        p = df[col].to_numpy(dtype=float)
        n = int(len(y))
        auc = roc_auc(y, p)
        acc = accuracy_at_threshold(y, p, thr=acc_threshold)
        f1 = f1_at_threshold(y, p, thr=acc_threshold)
        ll = logloss(y, p)

        rows.append({
            "method": method,
            "n_predictions": n,
            "auc": None if auc is None else float(auc),
            "accuracy": float(acc),
            "f1": float(f1),
            "logloss": float(ll),
            "acc_threshold": float(acc_threshold),
        })

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute global metrics by concatenating shard prediction CSVs.")
    ap.add_argument("--pred_dir", required=True, help="Directory containing predictions_P1_shard*.csv")
    ap.add_argument("--pattern", default="predictions_P1_shard*.csv", help="Glob pattern for shard prediction files")
    ap.add_argument("--acc_threshold", type=float, default=0.5)
    ap.add_argument("--out_csv", default=None, help="Where to write global metrics CSV")
    ap.add_argument("--out_merged_pred_csv", default=None, help="Optional: write merged predictions CSV")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir).expanduser().resolve()
    df = load_all_predictions(pred_dir, args.pattern)

    metrics = compute_global(df, acc_threshold=float(args.acc_threshold))

    # Print nicely
    for r in metrics.itertuples(index=False):
        auc_str = "NA" if pd.isna(r.auc) else f"{r.auc:.6f}"
        print(f"{r.method:10s} n={int(r.n_predictions):8d}  AUC={auc_str}  "
              f"ACC@{args.acc_threshold:.2f}={r.accuracy:.6f}  "
              f"F1@{args.acc_threshold:.2f}={r.f1:.6f}  LogLoss={r.logloss:.6f}")

    if args.out_merged_pred_csv:
        outp = Path(args.out_merged_pred_csv).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outp, index=False)
        print("Wrote merged predictions:", str(outp))

    if args.out_csv:
        outm = Path(args.out_csv).expanduser().resolve()
        outm.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(outm, index=False)
        print("Wrote global metrics:", str(outm))


if __name__ == "__main__":
    main()
