#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Helpers: robust stats
# ============================================================
def _mad(x: np.ndarray) -> float:
    """Median Absolute Deviation (unscaled)."""
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def summarize_series(x: pd.Series, prefix: str) -> Dict[str, float]:
    """Return a robust summary dict for a numeric series."""
    a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {f"{prefix}__n": 0}

    qs = [0.01, 0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]
    qv = np.quantile(a, qs)

    q25 = float(np.quantile(a, 0.25))
    q75 = float(np.quantile(a, 0.75))

    out = {
        f"{prefix}__n": int(a.size),
        f"{prefix}__mean": float(np.mean(a)),
        f"{prefix}__std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        f"{prefix}__min": float(np.min(a)),
        f"{prefix}__max": float(np.max(a)),
        f"{prefix}__median": float(np.median(a)),
        f"{prefix}__q25": q25,
        f"{prefix}__q75": q75,
        f"{prefix}__iqr": float(q75 - q25),
        f"{prefix}__mad": _mad(a),
        f"{prefix}__pct_pos": float(np.mean(a > 0.0)),
        f"{prefix}__pct_neg": float(np.mean(a < 0.0)),
        f"{prefix}__pct_zero": float(np.mean(a == 0.0)),
    }

    for qq, vv in zip(qs, qv):
        out[f"{prefix}__q{int(round(qq*100)):02d}"] = float(vv)

    return out


# ============================================================
# Delta computation
# ============================================================
def compute_delta(df: pd.DataFrame, col_a: str, col_b: str) -> pd.Series:
    """Compute signed delta = B - A."""
    a = pd.to_numeric(df[col_a], errors="coerce")
    b = pd.to_numeric(df[col_b], errors="coerce")
    return b - a


def compute_abs_delta(delta: pd.Series) -> pd.Series:
    """Absolute value of a delta."""
    return delta.abs()


def add_bins(df: pd.DataFrame, col: str, nbins: int, label: str) -> None:
    """Add quantile bins for a numeric column."""
    s = pd.to_numeric(df[col], errors="coerce")
    df[label] = pd.qcut(s, q=nbins, duplicates="drop")


# ============================================================
# Core analysis
# ============================================================
def analyze(
    pred_csv: Path,
    out_dir: Path,
    mastery_pairs: List[Tuple[str, str, str]],
    uncertainty_pairs: List[Tuple[str, str, str]],
    group_cols: List[str],
    make_u_bins: Optional[Tuple[str, int]],
    make_k_bins: Optional[Tuple[str, int]],
    drop_na_cols: List[str],
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(pred_csv)

    # ---- validation
    required = set(drop_na_cols + group_cols)
    for a, b, _ in mastery_pairs + uncertainty_pairs:
        required.update([a, b])

    missing = [c for c in required if c and c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    if drop_na_cols:
        df = df.dropna(subset=drop_na_cols).reset_index(drop=True)

    # ---- optional binning
    if make_u_bins is not None:
        col, nb = make_u_bins
        add_bins(df, col, nb, f"{col}__qbin{nb}")
        group_cols = group_cols + [f"{col}__qbin{nb}"]

    if make_k_bins is not None:
        col, nb = make_k_bins
        add_bins(df, col, nb, f"{col}__qbin{nb}")
        group_cols = group_cols + [f"{col}__qbin{nb}"]

    # ---- compute deltas (signed + absolute)
    signed_cols: List[str] = []
    abs_cols: List[str] = []

    for a, b, name in mastery_pairs:
        df[name] = compute_delta(df, a, b)
        df[f"abs_{name}"] = compute_abs_delta(df[name])
        signed_cols.append(name)
        abs_cols.append(f"abs_{name}")

    for a, b, name in uncertainty_pairs:
        df[name] = compute_delta(df, a, b)
        df[f"abs_{name}"] = compute_abs_delta(df[name])
        signed_cols.append(name)
        abs_cols.append(f"abs_{name}")

    # ========================================================
    # 1) Global summaries (ABSOLUTE deltas)
    # ========================================================
    global_rows: List[Dict[str, float]] = []
    for col in abs_cols:
        global_rows.append(summarize_series(df[col], prefix=col))

    df_global = pd.DataFrame(global_rows)
    df_global.insert(0, "metric", abs_cols)

    out_global = out_dir / "delta_summary_global.csv"
    df_global.to_csv(out_global, index=False)

    # ========================================================
    # 2) Grouped summaries (ABSOLUTE deltas)
    # ========================================================
    if group_cols:
        rows: List[Dict[str, float]] = []
        for keys, sub in df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_dict = dict(zip(group_cols, keys))

            for col in abs_cols:
                row = {"metric": col}
                row.update(key_dict)
                row.update(summarize_series(sub[col], prefix="d"))
                rows.append(row)

        df_grouped = pd.DataFrame(rows)
        df_grouped.to_csv(out_dir / "delta_summary_grouped.csv", index=False)

    # ========================================================
    # 3) Per-row output (SIGNED + ABS)
    # ========================================================
    keep_cols = [c for c in (group_cols + drop_na_cols) if c in df.columns]
    df_out = df[keep_cols + signed_cols + abs_cols].copy()
    df_out.to_csv(out_dir / "deltas_per_row.csv", index=False)

    print("Wrote:", out_global)
    if group_cols:
        print("Wrote:", out_dir / "delta_summary_grouped.csv")
    print("Wrote:", out_dir / "deltas_per_row.csv")


# ============================================================
# CLI
# ============================================================
def parse_pairs(pairs: List[str]) -> List[Tuple[str, str, str]]:
    out = []
    for s in pairs:
        a, b, name = [x.strip() for x in s.split(",")]
        out.append((a, b, name))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--mastery_pair", action="append", default=[])
    ap.add_argument("--uncert_pair", action="append", default=[])

    ap.add_argument("--drop_na", default="")
    ap.add_argument("--group_cols", default="")
    ap.add_argument("--u_bins", default="")
    ap.add_argument("--k_bins", default="")

    args = ap.parse_args()

    drop_na_cols = [c.strip() for c in args.drop_na.split(",") if c.strip()]
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]

    mastery_pairs = parse_pairs(args.mastery_pair)
    uncert_pairs = parse_pairs(args.uncert_pair)

    u_bins = None
    if args.u_bins:
        c, n = args.u_bins.split(",")
        u_bins = (c.strip(), int(n))

    k_bins = None
    if args.k_bins:
        c, n = args.k_bins.split(",")
        k_bins = (c.strip(), int(n))

    analyze(
        pred_csv=Path(args.pred_csv).expanduser(),
        out_dir=Path(args.out_dir).expanduser(),
        mastery_pairs=mastery_pairs,
        uncertainty_pairs=uncert_pairs,
        group_cols=group_cols,
        make_u_bins=u_bins,
        make_k_bins=k_bins,
        drop_na_cols=drop_na_cols,
    )


if __name__ == "__main__":
    main()
