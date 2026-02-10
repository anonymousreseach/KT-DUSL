#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
metrics_by_evidence_user_concept.py

Compute prediction metrics by evidence count k at the (UserId, KC) level,
for ALL prediction CSV files.

Evidence definition (KC-level):
- For each interaction (UserId, QuestionId), we first associate the interaction
  to one or more KCs (depending on the dataset).
- We then sort chronologically and define evidence_k = 1, 2, 3, ... as the rank
  of the interaction for the (UserId, KC) pair.

Two reporting modes:
1) exact: metrics computed only on rows with evidence_k == k
2) upto:  metrics computed only on rows with evidence_k <= k (cumulative truncation)

Supports:
- Explicit k list: --k_list 1,3,5,10,20
- Or range:        --k_max 30  (used only if k_list is not provided)

KC assignment options:
A) If your prediction CSV already contains a KC column:
   - Provide --kc_col <COLUMN_NAME>
   - For multi-KC, the column may contain "kc1|kc2|kc3" or "kc1,kc2,kc3"

B) If the CSV does NOT contain KCs (e.g., Junyi predictions),
   provide a mapping from QuestionId -> KC via:
   - --qc_map_csv <path_to_csv>   (with two columns: QuestionId, KC)
   OR
   - --junyi_tree_json <subject_tree_with_questions.json>
     (we extract QuestionId -> subject id as KC)

Required columns in prediction CSV:
    UserId, QuestionId, IsCorrect,
    u_leaf_sl_mean, p_leaf_sl, p_prop_sl

Chronological order:
- Uses "step" if available, otherwise row order.

Output:
- One CSV with metrics per k, per group (global / concepts_certain / concepts_uncertain),
  per method (leaf_sl / prop_sl), and per mode (exact / upto).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List, Iterable, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Metrics (no sklearn)
# ============================================================

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
# KC mapping utilities
# ============================================================

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # also try case-insensitive
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def load_qc_map_csv(path: Path) -> pd.DataFrame:
    """
    Load a QuestionId -> KC mapping from a CSV.
    The CSV must contain two columns: one for QuestionId, one for KC/Subject/Concept.
    We try to auto-detect common column names.
    """
    m = pd.read_csv(path)

    qcol = _find_col(m, ["QuestionId", "question_id", "qid", "question", "QuestionID"])
    kcol = _find_col(m, ["KC", "KCId", "ConceptId", "concept_id", "SubjectId", "subject_id", "skill_id"])

    if qcol is None or kcol is None:
        raise ValueError(
            f"Cannot auto-detect mapping columns in {path}. "
            f"Found columns: {list(m.columns)}. "
            f"Expected something like (QuestionId, SubjectId) or (QuestionId, ConceptId)."
        )

    out = m[[qcol, kcol]].copy()
    out.columns = ["QuestionId", "KC"]
    out["QuestionId"] = pd.to_numeric(out["QuestionId"], errors="ignore")
    out["KC"] = out["KC"].astype(str)
    out = out.dropna(subset=["QuestionId", "KC"]).drop_duplicates()
    return out


def load_junyi_tree_json(path: Path) -> pd.DataFrame:
    """
    Build QuestionId -> KC mapping from Junyi 'subject_tree_with_questions.json'.

    We interpret each (leaf) subject as a KC and map all its questions to this KC.
    This is robust to the fact that one subject(KC) has multiple questions.

    The JSON structure can vary; we recursively traverse all dict/list nodes
    and collect:
      - a subject identifier (id / subject_id / name fallback)
      - a list of question ids (questions / question_ids / problem_ids, etc.)

    IMPORTANT:
    In some Junyi exports, the "questions" list may contain dict objects rather
    than scalar IDs. Pandas cannot drop_duplicates() on dict values (unhashable),
    so we normalize each question element into a scalar QuestionId.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    pairs: List[Tuple[object, object]] = []

    def get_subject_id(node: dict) -> Optional[str]:
        for key in ["subject_id", "id", "subjectId", "name", "subject"]:
            if key in node and node[key] is not None:
                return str(node[key])
        return None

    def get_questions(node: dict) -> Optional[List[object]]:
        for key in ["questions", "question_ids", "questionIds", "problem_ids", "problemIds", "items"]:
            if key in node and node[key] is not None:
                if isinstance(node[key], list):
                    return node[key]
        return None

    def normalize_question_id(q) -> Optional[object]:
        """
        Convert a question element into a hashable scalar id.
        - If it's a dict, try common id keys.
        - If it's already scalar (int/str), keep it.
        """
        if isinstance(q, dict):
            for k in ["QuestionId", "question_id", "questionId", "id", "uid", "problem_id", "problemId"]:
                if k in q and q[k] is not None:
                    return q[k]
            return None
        return q

    def traverse(node, current_subject: Optional[str] = None):
        if isinstance(node, dict):
            sid = get_subject_id(node) or current_subject
            qs = get_questions(node)
            if sid is not None and qs is not None:
                for q in qs:
                    qid = normalize_question_id(q)
                    if qid is not None:
                        pairs.append((qid, sid))

            # traverse children keys commonly used
            for key in ["children", "child", "nodes", "subtree"]:
                if key in node:
                    traverse(node[key], sid)

            # traverse everything else defensively
            for v in node.values():
                if isinstance(v, (dict, list)):
                    traverse(v, sid)

        elif isinstance(node, list):
            for it in node:
                traverse(it, current_subject)

    traverse(obj)

    if not pairs:
        raise ValueError(f"No (question, subject) pairs extracted from {path}. JSON structure may differ.")

    df = pd.DataFrame(pairs, columns=["QuestionId", "KC"])

    # Ensure hashable / consistent types
    df["KC"] = df["KC"].astype(str)

    # Align QuestionId type with most prediction CSVs (usually int)
    # If your prediction files store QuestionId as string, remove the int cast.
    df["QuestionId"] = pd.to_numeric(df["QuestionId"], errors="coerce")
    df = df.dropna(subset=["QuestionId", "KC"]).drop_duplicates()
    df["QuestionId"] = df["QuestionId"].astype(int)

    return df


def attach_kc(
    df: pd.DataFrame,
    kc_col: Optional[str],
    qc_map: Optional[pd.DataFrame],
    multi_kc_sep: Optional[str],
) -> pd.DataFrame:
    """
    Ensure a 'KC' column exists.
    - If kc_col is provided and exists in df, use it.
    - Else merge QuestionId -> KC mapping (qc_map).
    """
    d = df.copy()

    if kc_col is not None:
        if kc_col not in d.columns:
            raise ValueError(f"--kc_col='{kc_col}' not found in prediction file columns: {list(d.columns)}")
        d["KC"] = d[kc_col]
    else:
        if qc_map is None:
            raise ValueError("No KC source provided. Use --kc_col OR --qc_map_csv / --junyi_tree_json.")
        if "QuestionId" not in d.columns:
            raise ValueError(f"Prediction file missing 'QuestionId'. Available: {list(d.columns)}")
        d = d.merge(qc_map, on="QuestionId", how="left")
        # Some QuestionIds might not be in map -> drop later when exploding

    # Normalize to string
    d["KC"] = d["KC"].astype(str)

    # If multi-KC is encoded in a single string, split + explode
    if multi_kc_sep is not None and multi_kc_sep.strip():
        sep = multi_kc_sep.strip()
        d["KC"] = d["KC"].str.split(sep)
        d = d.explode("KC")
        d["KC"] = d["KC"].astype(str).str.strip()

    # Remove empty / nan-like
    d = d[~d["KC"].isin(["", "nan", "None", "NA"])].reset_index(drop=True)
    return d


# ============================================================
# Evidence rank per (UserId, KC)
# ============================================================

def add_evidence_rank_kc(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    required = ["UserId", "KC", "IsCorrect"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required columns after KC attach: {missing}")

    if "step" in d.columns:
        d = d.sort_values(["UserId", "KC", "step"], kind="mergesort")
    else:
        d["_idx"] = np.arange(len(d), dtype=int)
        d = d.sort_values(["UserId", "KC", "_idx"], kind="mergesort")

    d["evidence_k"] = d.groupby(["UserId", "KC"]).cumcount() + 1
    d.drop(columns=["_idx"], inplace=True, errors="ignore")
    return d


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

    for k in k_values:
        if mode == "exact":
            mk = (df["evidence_k"].to_numpy() == int(k))
        elif mode == "upto":
            mk = (df["evidence_k"].to_numpy() <= int(k))
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

    ap.add_argument("--u_th", type=float, default=0.5)
    ap.add_argument("--acc_threshold", type=float, default=0.5)

    # ---- KC definition ----
    ap.add_argument(
        "--kc_col",
        default=None,
        help="Column name in predictions containing the KC id. If not provided, a QuestionId->KC map must be given.",
    )
    ap.add_argument(
        "--multi_kc_sep",
        default=None,
        help="If a row can have multiple KCs in kc_col, provide separator (e.g., '|' or ',').",
    )
    ap.add_argument(
        "--qc_map_csv",
        default=None,
        help="CSV mapping QuestionId -> KC (e.g., QuestionId,SubjectId). Used if --kc_col is not provided.",
    )
    ap.add_argument(
        "--junyi_tree_json",
        default=None,
        help="Junyi subject_tree_with_questions.json to build QuestionId -> KC mapping (used if --kc_col is not provided).",
    )

    # ---- k control ----
    ap.add_argument(
        "--k_list",
        type=str,
        default=None,
        help="Comma-separated list of k values, e.g. 1,3,5,10,20 (overrides --k_max)",
    )
    ap.add_argument(
        "--k_max",
        type=int,
        default=20,
        help="Use k=1..k_max if --k_list is not provided",
    )
    ap.add_argument("--modes", default="exact,upto")

    args = ap.parse_args()

    # ---- build k values ----
    if args.k_list is not None:
        k_values = sorted({int(k) for k in args.k_list.split(",")})
    else:
        k_values = list(range(1, int(args.k_max) + 1))

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    pred_dir = Path(args.pred_dir)
    files = sorted(pred_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched {args.glob} in {pred_dir}")

    # ---- build QuestionId -> KC map if needed ----
    qc_map: Optional[pd.DataFrame] = None
    if args.kc_col is None:
        if args.qc_map_csv is not None:
            qc_map = load_qc_map_csv(Path(args.qc_map_csv))
        elif args.junyi_tree_json is not None:
            qc_map = load_junyi_tree_json(Path(args.junyi_tree_json))
        else:
            raise ValueError("You must provide --kc_col OR (--qc_map_csv / --junyi_tree_json).")

    all_frames: List[pd.DataFrame] = []

    for fp in files:
        print(f"[INFO] Processing {fp}")
        df = pd.read_csv(fp)

        # Attach KC ids (KC-level evaluation)
        df = attach_kc(
            df=df,
            kc_col=args.kc_col,
            qc_map=qc_map,
            multi_kc_sep=args.multi_kc_sep,
        )

        # Evidence rank per (UserId, KC)
        df = add_evidence_rank_kc(df)

        for mode in modes:
            out = compute_by_k(
                df=df,
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
