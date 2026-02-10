#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
metrics_by_evidence_user_concept_eedi.py

Compute metrics per evidence count at the (UserId, ConceptId) level for EEDI,
for ALL prediction CSV files.

ConceptId definition:
- EEDI mono-concept: ConceptId = SubjectId_single (from question_metadata_task_1_2_with_single_leaf.csv)
- EEDI multi-concepts: ConceptId = each SubjectId in the question's subject list (from raw question_metadata_task_1_2.csv)

Evidence definition:
- For each (UserId, ConceptId), sort chronologically and define:
    evidence_k = 1, 2, 3, ... as the rank of the interaction for that pair.

Two reporting modes:
1) exact: metrics computed only on rows with evidence_k == k
2) upto:  metrics computed on rows with evidence_k <= k (cumulative truncation)

Required columns in predictions:
- UserId, QuestionId, IsCorrect
- u_leaf_sl_mean
- p_leaf_sl, p_prop_sl
Optional:
- step (chronological order)
- policy

Required columns in question metadata:
- QuestionId
- either:
  - SubjectId_single (mono), or
  - SubjectId / SubjectIds / subjects (multi; possibly string-encoded lists)

Notes:
- In multi-concepts setting, each interaction contributes once per tagged concept
  (we "explode" concepts). This matches the notion of evidence per (learner, concept).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, List, Iterable, Any

import numpy as np
import pandas as pd
import ast


# ============================================================
# Metrics (no sklearn)
# ============================================================

def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None

    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1, dtype=float)

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
    denom = (2 * tp + fp + fn)
    if denom == 0:
        return 0.0
    return float((2 * tp) / denom)


def logloss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def compute_metrics_arrays(y_true: np.ndarray, y_prob: np.ndarray, acc_thr: float) -> Dict[str, object]:
    m = np.isfinite(y_prob)
    y_true = y_true[m].astype(int)
    y_prob = y_prob[m].astype(float)

    n = int(len(y_true))
    if n == 0:
        return {"n": 0, "auc": None, "accuracy": 0.0, "f1": 0.0, "logloss": 0.0}

    auc = roc_auc(y_true, y_prob)
    acc = accuracy_at_threshold(y_true, y_prob, acc_thr)
    f1 = f1_at_threshold(y_true, y_prob, acc_thr)
    ll = logloss(y_true, y_prob)
    return {
        "n": n,
        "auc": None if auc is None else float(auc),
        "accuracy": float(acc),
        "f1": float(f1),
        "logloss": float(ll),
    }


# ============================================================
# Concept mapping (QuestionId -> list of SubjectIds)
# ============================================================

def _parse_subject_list(x: Any) -> List[int]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (int, np.integer)):
        return [int(x)]
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return []

    # list-like string
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return [int(v) for v in obj if str(v).strip() != ""]
            if isinstance(obj, (int, np.integer)):
                return [int(obj)]
        except Exception:
            pass

    # separators
    for sep in ["|", ";", ",", " "]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip() != ""]
            out = []
            for p in parts:
                try:
                    out.append(int(float(p)))
                except Exception:
                    continue
            return out

    try:
        return [int(float(s))]
    except Exception:
        return []


def build_q2subjects(question_metadata_csv: str, mode: str) -> Dict[int, List[int]]:
    """
    mode:
      - "mono": expects SubjectId_single (or close variant)
      - "multi": expects subject list column (or SubjectId)
      - "auto": tries mono first then multi
    """
    df = pd.read_csv(question_metadata_csv)
    if "QuestionId" not in df.columns:
        raise KeyError(f"Missing QuestionId in {question_metadata_csv}. Columns={list(df.columns)}")

    cols = list(df.columns)

    mono_candidates = ["SubjectId_single", "SubjectIdSingle", "subject_id_single"]
    multi_candidates = ["SubjectIds", "SubjectIdList", "SubjectId_list", "subject_ids", "subjects", "SubjectId", "SubjectID", "subject_id"]

    def pick_col(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in cols:
                return c
        return None

    mono_col = pick_col(mono_candidates)
    multi_col = pick_col(multi_candidates)

    chosen = None
    if mode == "mono":
        chosen = mono_col
        if chosen is None:
            raise KeyError(f"mode=mono but none of {mono_candidates} found. Columns={cols}")
        parse_fn = lambda v: _parse_subject_list(v)[:1]  # ensure 1
    elif mode == "multi":
        chosen = multi_col
        if chosen is None:
            raise KeyError(f"mode=multi but none of {multi_candidates} found. Columns={cols}")
        parse_fn = _parse_subject_list
    elif mode == "auto":
        if mono_col is not None:
            chosen = mono_col
            parse_fn = lambda v: _parse_subject_list(v)[:1]
        elif multi_col is not None:
            chosen = multi_col
            parse_fn = _parse_subject_list
        else:
            raise KeyError(f"mode=auto but no subject column found. Tried mono={mono_candidates} multi={multi_candidates}. Columns={cols}")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    q2s: Dict[int, List[int]] = {}
    for r in df.itertuples(index=False):
        qid = int(getattr(r, "QuestionId"))
        raw = getattr(r, chosen)
        sids = parse_fn(raw)
        # unique preserve order
        seen = set()
        out = []
        for s in sids:
            if int(s) not in seen:
                seen.add(int(s))
                out.append(int(s))
        q2s[qid] = out

    return q2s


# ============================================================
# Build (UserId, ConceptId) rows + evidence rank
# ============================================================

def explode_to_user_concept(df_pred: pd.DataFrame, q2s: Dict[int, List[int]]) -> pd.DataFrame:
    df = df_pred.copy()
    required = ["UserId", "QuestionId", "IsCorrect"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Predictions missing required columns: {missing}")

    # map QuestionId -> SubjectIds list
    df["_concepts"] = df["QuestionId"].map(q2s)
    df["_concepts"] = df["_concepts"].apply(lambda x: x if isinstance(x, list) else [])

    # explode
    df = df.explode("_concepts", ignore_index=True)
    df = df[df["_concepts"].notna()].copy()
    df["ConceptId"] = df["_concepts"].astype(int)
    df.drop(columns=["_concepts"], inplace=True)

    return df


def add_evidence_rank_user_concept(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "step" in df.columns:
        df = df.sort_values(["UserId", "ConceptId", "step"], kind="mergesort")
    else:
        df["_idx"] = np.arange(len(df), dtype=int)
        df = df.sort_values(["UserId", "ConceptId", "_idx"], kind="mergesort")

    df["evidence_k"] = df.groupby(["UserId", "ConceptId"]).cumcount() + 1
    df.drop(columns=["_idx"], inplace=True, errors="ignore")
    return df


def uncertainty_mask(df: pd.DataFrame, u_th: float) -> np.ndarray:
    u = pd.to_numeric(df["u_leaf_sl_mean"], errors="coerce").to_numpy()
    u = np.where(np.isfinite(u), u, -np.inf)  # NaN -> certain
    return (u > float(u_th))


# ============================================================
# Metrics by k
# ============================================================

def compute_by_k(
    df: pd.DataFrame,
    file_name: str,
    u_th: float,
    acc_thr: float,
    k_values: Iterable[int],
    mode: str,  # "exact" or "upto"
) -> pd.DataFrame:

    methods = {
        "leaf_sl": "p_leaf_sl",
        "prop_sl": "p_prop_sl",
    }

    needed = ["IsCorrect", "evidence_k", "u_leaf_sl_mean"] + list(methods.values())
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{file_name}: missing columns {missing}")

    policy = (
        str(df["policy"].dropna().iloc[0])
        if "policy" in df.columns and df["policy"].notna().any()
        else "NA"
    )

    unc = uncertainty_mask(df, u_th=u_th)
    split_masks = {
        "global": np.ones(len(df), dtype=bool),
        "concepts_certain": ~unc,
        "concepts_uncertain": unc,
    }

    y_true_all = pd.to_numeric(df["IsCorrect"], errors="coerce").to_numpy().astype(int)

    rows: List[Dict[str, object]] = []

    ev = df["evidence_k"].to_numpy()

    for k in k_values:
        if mode == "exact":
            mk = (ev == int(k))
        elif mode == "upto":
            mk = (ev <= int(k))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not np.any(mk):
            continue

        for group, mg in split_masks.items():
            m = mk & mg
            if not np.any(m):
                continue

            y_true = y_true_all[m]

            for method, pcol in methods.items():
                y_prob = pd.to_numeric(df.loc[m, pcol], errors="coerce").to_numpy()
                met = compute_metrics_arrays(y_true, y_prob, acc_thr)

                rows.append({
                    "source_file": file_name,
                    "policy": policy,
                    "mode": mode,
                    "k": int(k),
                    "group": group,
                    "u_th": float(u_th),
                    "acc_threshold": float(acc_thr),
                    "method": method,
                    **met,
                })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--glob", default="predictions_*.csv")
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--question_metadata_csv", required=True,
                    help="EEDI question metadata (raw multi OR processed mono single leaf).")
    ap.add_argument("--concept_mode", default="auto", choices=["auto", "mono", "multi"],
                    help="How to interpret concepts from question metadata.")

    ap.add_argument("--u_th", type=float, default=0.5)
    ap.add_argument("--acc_threshold", type=float, default=0.5)

    # ---- k control ----
    ap.add_argument("--k_list", type=str, default=None,
                    help="Comma-separated list of k values, e.g. 1,3,5,10,20 (overrides --k_max)")
    ap.add_argument("--k_max", type=int, default=20,
                    help="Use k=1..k_max if --k_list is not provided")

    ap.add_argument("--modes", default="exact,upto")
    args = ap.parse_args()

    # build k values
    if args.k_list is not None:
        k_values = sorted({int(k) for k in args.k_list.split(",")})
    else:
        k_values = list(range(1, int(args.k_max) + 1))

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    # load mapping
    q2s = build_q2subjects(args.question_metadata_csv, mode=args.concept_mode)

    pred_dir = Path(args.pred_dir)
    files = sorted(pred_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched {args.glob} in {pred_dir}")

    all_frames: List[pd.DataFrame] = []

    for fp in files:
        print(f"[INFO] Processing {fp}")
        dfp = pd.read_csv(fp)

        # explode to (UserId, ConceptId)
        df_uc = explode_to_user_concept(dfp, q2s=q2s)

        # add evidence rank per (UserId, ConceptId)
        df_uc = add_evidence_rank_user_concept(df_uc)

        for mode in modes:
            out = compute_by_k(
                df=df_uc,
                file_name=fp.name,
                u_th=args.u_th,
                acc_thr=args.acc_threshold,
                k_values=k_values,
                mode=mode,
            )
            all_frames.append(out)

    res = pd.concat(all_frames, axis=0, ignore_index=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)

    print(f"[DONE] Wrote {out_path}")
    print(res.head(20))


if __name__ == "__main__":
    main()
