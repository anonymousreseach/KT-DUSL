#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Metrics (same as evaluators)
# -------------------------

def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
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


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Tuple[int, Optional[float], float, float, float]:
    n = int(len(y_true))
    if n == 0:
        return 0, None, 0.0, 0.0, 0.0
    auc = roc_auc(y_true, y_prob)
    acc = accuracy_at_threshold(y_true, y_prob, thr=thr)
    f1 = f1_at_threshold(y_true, y_prob, thr=thr)
    ll = logloss(y_true, y_prob)
    return n, auc, acc, f1, ll


# -------------------------
# Loading / merging
# -------------------------

def load_concat_predictions(pred_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(pred_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {pred_dir} matching pattern '{pattern}'")

    dfs: List[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        df["__source_file__"] = f.name
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    return out


def compute_global_from_df(df: pd.DataFrame, acc_threshold: float) -> pd.DataFrame:
    required = ["IsCorrect", "p_leaf_mean", "p_leaf_sl", "p_prop_mean", "p_prop_sl"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    df = df.copy()
    df["IsCorrect"] = df["IsCorrect"].astype(int)

    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)

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
        n, auc, acc, f1, ll = compute_metrics(y, p, thr=float(acc_threshold))
        rows.append({
            "method": method,
            "n_predictions": int(n),
            "auc": None if auc is None else float(auc),
            "accuracy": float(acc),
            "f1": float(f1),
            "logloss": float(ll),
            "acc_threshold": float(acc_threshold),
        })
    return pd.DataFrame(rows)


def pretty_print(metrics: pd.DataFrame, title: str, thr: float) -> None:
    def fmt_auc(x):
        return "NA" if pd.isna(x) else f"{x:.6f}"
    print(f"\n===== {title} =====")
    for r in metrics.itertuples(index=False):
        print(f"{r.method:12s} n={int(r.n_predictions):9d}  AUC={fmt_auc(r.auc)}  "
              f"ACC@{thr:.2f}={r.accuracy:.6f}  F1@{thr:.2f}={r.f1:.6f}  LogLoss={r.logloss:.6f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute GLOBAL metrics for EEDI mono vs multi concepts by merging shard predictions.")
    ap.add_argument("--root", default=".", help="Repo root (where RESULTS_EEDI lives). Default: current dir")
    ap.add_argument("--policy", default="P1", help="Policy folder name (e.g., P1)")
    ap.add_argument("--acc_threshold", type=float, default=0.5)

    ap.add_argument("--mono_dir", default="RESULTS_EEDI", help="Folder for mono-concept setting")
    ap.add_argument("--multi_dir", default="RESULTS_EEDI_ALL_CONCEPTS", help="Folder for multi-concepts setting")

    ap.add_argument("--mono_pattern", default="predictions_P1_shard*.csv", help="Glob for mono predictions")
    ap.add_argument("--multi_pattern", default="predictions_allconcepts_P1_shard*.csv", help="Glob for multi predictions")

    ap.add_argument("--write_merged", action="store_true", help="Also write merged prediction CSVs")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()

    # MONO
    mono_pdir = root / args.mono_dir / args.policy
    mono_df = load_concat_predictions(mono_pdir, args.mono_pattern)
    mono_metrics = compute_global_from_df(mono_df, acc_threshold=float(args.acc_threshold))
    pretty_print(mono_metrics, f"EEDI MONO ({args.mono_dir}/{args.policy})", float(args.acc_threshold))

    mono_out = mono_pdir / "metrics_eedi_P1_ALL.csv"
    mono_metrics.to_csv(mono_out, index=False)
    print("Wrote:", str(mono_out))

    if args.write_merged:
        mono_merged = mono_pdir / "predictions_P1_ALL.csv"
        mono_df.to_csv(mono_merged, index=False)
        print("Wrote merged:", str(mono_merged))

    # MULTI
    multi_pdir = root / args.multi_dir / args.policy
    multi_df = load_concat_predictions(multi_pdir, args.multi_pattern)
    multi_metrics = compute_global_from_df(multi_df, acc_threshold=float(args.acc_threshold))
    pretty_print(multi_metrics, f"EEDI MULTI ({args.multi_dir}/{args.policy})", float(args.acc_threshold))

    multi_out = multi_pdir / "metrics_eedi_allconcepts_P1_ALL.csv"
    multi_metrics.to_csv(multi_out, index=False)
    print("Wrote:", str(multi_out))

    if args.write_merged:
        multi_merged = multi_pdir / "predictions_allconcepts_P1_ALL.csv"
        multi_df.to_csv(multi_merged, index=False)
        print("Wrote merged:", str(multi_merged))


if __name__ == "__main__":
    main()
