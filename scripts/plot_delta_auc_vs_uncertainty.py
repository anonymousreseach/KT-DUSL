#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# AUC (no sklearn)
# ============================================================
def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """AUC via rank method (no sklearn). Returns None if only one class."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if y_true.size == 0:
        return None
    if len(np.unique(y_true)) < 2:
        return None

    order = np.argsort(y_prob, kind="mergesort")
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
            avg_rank = float(np.mean(ranks[order[i : j + 1]]))
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    pos = (y_true == 1)
    n_pos = int(np.sum(pos))
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None

    sum_ranks_pos = float(np.sum(ranks[pos]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ============================================================
# Rolling AUC computation
# ============================================================
def compute_rolling_auc(
    df: pd.DataFrame,
    u_col: str,
    prob_usl: str,
    prob_dusl: str,
    window_size: int,
    stride: int,
    min_pos: int,
    min_neg: int,
) -> pd.DataFrame:
    needed = ["IsCorrect", u_col, prob_usl, prob_dusl]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}. Available: {list(df.columns)}")

    d = df.copy()
    d["IsCorrect"] = pd.to_numeric(d["IsCorrect"], errors="coerce")
    d[u_col] = pd.to_numeric(d[u_col], errors="coerce")
    d[prob_usl] = pd.to_numeric(d[prob_usl], errors="coerce")
    d[prob_dusl] = pd.to_numeric(d[prob_dusl], errors="coerce")
    d = d.dropna(subset=["IsCorrect", u_col, prob_usl, prob_dusl]).reset_index(drop=True)

    # sort by uncertainty (monotonic x-axis)
    d = d.sort_values(u_col, kind="mergesort").reset_index(drop=True)

    y = d["IsCorrect"].astype(int).to_numpy()
    u = d[u_col].to_numpy(dtype=float)
    p_usl = d[prob_usl].to_numpy(dtype=float)
    p_dusl = d[prob_dusl].to_numpy(dtype=float)

    n_total = len(d)
    if n_total < window_size:
        raise ValueError(
            f"Not enough rows after NaN drop: {n_total} < window_size={window_size}"
        )

    rows: List[Dict[str, float]] = []
    for start in range(0, n_total - window_size + 1, stride):
        end = start + window_size

        ys = y[start:end]
        us = u[start:end]
        pusl = p_usl[start:end]
        pdusl = p_dusl[start:end]

        n_pos = int(np.sum(ys == 1))
        n_neg = int(np.sum(ys == 0))
        if n_pos < min_pos or n_neg < min_neg:
            continue

        auc_usl = roc_auc(ys, pusl)
        auc_dusl = roc_auc(ys, pdusl)
        if auc_usl is None or auc_dusl is None:
            continue

        rows.append({
            "start": float(start),
            "end": float(end),
            "n": float(window_size),
            "n_pos": float(n_pos),
            "n_neg": float(n_neg),
            "u_mean": float(np.mean(us)),
            "u_median": float(np.median(us)),
            "auc_usl": float(auc_usl),
            "auc_dusl": float(auc_dusl),
            "delta_auc": float(auc_dusl - auc_usl),
        })

    return pd.DataFrame(rows)


# ============================================================
# Plot: AUC(KT-USL) vs AUC(KT-DUSL)
# ============================================================
def plot_auc_two_models(
    df_roll: pd.DataFrame,
    out_png: Path,
    title: str,
) -> None:
    if df_roll.empty:
        raise ValueError("df_roll is empty (no valid windows).")

    out_png.parent.mkdir(parents=True, exist_ok=True)

    x = df_roll["u_mean"].to_numpy(dtype=float)
    auc_usl = df_roll["auc_usl"].to_numpy(dtype=float)
    auc_dusl = df_roll["auc_dusl"].to_numpy(dtype=float)

    # ensure clean monotonic curves
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    auc_usl = auc_usl[order]
    auc_dusl = auc_dusl[order]

    plt.figure()
    plt.plot(
        x,
        auc_usl,
        marker="o",
        linewidth=1.8,
        label="KT-USL",
    )
    plt.plot(
        x,
        auc_dusl,
        marker="o",
        linewidth=1.8,
        label="KT-DUSL",
    )

    plt.xlabel("uncertainty u (window mean)")
    plt.ylabel("AUC")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ============================================================
# Main
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--pred_csv", required=True,
                    help="Merged predictions CSV (e.g., predictions_P1_ALL.csv)")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--u_col", default="u_leaf_sl_mean",
                    help="Uncertainty column (KT-USL) used to order windows")
    ap.add_argument("--prob_usl", default="p_leaf_sl",
                    help="Probability column for KT-USL")
    ap.add_argument("--prob_dusl", default="p_prop_sl",
                    help="Probability column for KT-DUSL")

    ap.add_argument("--window_size", type=int, default=200000)
    ap.add_argument("--stride", type=int, default=50000)
    ap.add_argument("--min_pos", type=int, default=1000)
    ap.add_argument("--min_neg", type=int, default=1000)

    ap.add_argument("--tag", default="run")

    args = ap.parse_args()

    pred_csv = Path(args.pred_csv).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_csv)

    df_roll = compute_rolling_auc(
        df=df,
        u_col=str(args.u_col),
        prob_usl=str(args.prob_usl),
        prob_dusl=str(args.prob_dusl),
        window_size=int(args.window_size),
        stride=int(args.stride),
        min_pos=int(args.min_pos),
        min_neg=int(args.min_neg),
    )

    out_csv = out_dir / f"rolling_auc_{args.tag}.csv"
    df_roll.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

    out_png = out_dir / f"rolling_auc_two_models_{args.tag}.png"
    plot_auc_two_models(
        df_roll=df_roll,
        out_png=out_png,
        title=f"{args.tag}",
    )
    print("Wrote:", out_png)


if __name__ == "__main__":
    main()
