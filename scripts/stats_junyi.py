#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Junyi dataset statistics and hierarchy stats from a domain JSON:
- counts: interactions/users/questions/concepts
- correctness rate
- interactions per user / per question (describe)
- concepts-per-question from domain mapping
- hierarchy depth distribution (from domain tree)
Optionally writes a one-row CSV summary.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def _norm_col(c: str) -> str:
    """Normalize a column name for robust matching."""
    return "".join(ch.lower() for ch in str(c).strip() if ch.isalnum())


def resolve_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> Optional[str]:
    """Resolve a column from candidate names (robust to case/punctuation)."""
    lookup = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_col(cand)
        if key in lookup:
            return lookup[key]
    if required:
        raise ValueError(f"Missing required column among {candidates}. Available: {list(df.columns)}")
    return None


def load_domain_with_depth(domain_json: Path) -> tuple[dict[int, list[int]], set[int], dict[int, int]]:
    """
    Parse domain JSON tree and extract:
      - q2s: question_id -> [subject_id, ...]
      - subjects: all subject_ids present
      - depth_map: subject_id -> depth (root depth=0)

    Expected structure:
      dom["pkg"]["roots"] or dom["roots"]
      node: {"subject_id": ..., "children": [...], "questions": [...]}.
    """
    with open(domain_json, "r", encoding="utf-8") as f:
        dom = json.load(f)

    roots = dom.get("pkg", {}).get("roots")
    if roots is None:
        roots = dom.get("roots", [])
    roots = roots or []

    q2s: dict[int, list[int]] = defaultdict(list)
    subjects: set[int] = set()
    depth_map: dict[int, int] = {}

    def visit(node: dict[str, Any], depth: int) -> None:
        sid = int(node["subject_id"])
        subjects.add(sid)
        depth_map.setdefault(sid, int(depth))

        for q in node.get("questions", []) or []:
            qid = q.get("question_id", q.get("id"))
            if qid is not None:
                q2s[int(qid)].append(sid)

        for ch in node.get("children", []) or []:
            visit(ch, depth + 1)

    for r in roots:
        if isinstance(r, dict) and "subject_id" in r:
            visit(r, depth=0)

    return dict(q2s), subjects, depth_map


def describe_counts(s: pd.Series) -> dict[str, float]:
    """Return a compact set of summary stats for a count series."""
    d = s.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    return {k: float(d[k]) for k in ["min", "25%", "50%", "75%", "max", "mean"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_csv", required=True, help="Junyi interactions CSV (logs)")
    ap.add_argument("--domain_json", required=True, help="Domain JSON with subject hierarchy + question mapping")
    ap.add_argument("--question_id_map", required=True, help="CSV mapping raw question ids to QuestionId")
    ap.add_argument("--out_csv", default=None, help="Optional path to write a one-row summary CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.log_csv)
    qmap = pd.read_csv(args.question_id_map)
    q2s, subjects, depth_map = load_domain_with_depth(Path(args.domain_json))

    # Resolve key columns in logs
    user_col = resolve_col(df, ["uuid", "user", "userid"])
    qraw_col = resolve_col(df, ["ucid", "uc_id", "ucidraw", "ucid_", "uc id", "ucid "])
    y_col = resolve_col(df, ["is_correct", "iscorrect", "correct", "label"])

    # Basic counts
    n_interactions = int(len(df))
    n_users = int(df[user_col].nunique())

    # Mapping file: raw -> QuestionId
    qraw_map_col = resolve_col(qmap, ["UCId", "ucid", "uc_id", "uc id"], required=False)
    qid_map_col = resolve_col(qmap, ["QuestionId", "questionid", "question_id"], required=True)
    n_questions_mapped = int(qmap[qid_map_col].nunique())

    n_subjects_hierarchy = int(len(subjects))

    # Global correctness
    y = pd.to_numeric(df[y_col], errors="coerce")
    acc_global = float(y.mean())

    # Interaction count distributions
    ipu = df.groupby(user_col).size()
    ipq = df.groupby(qraw_col).size()
    ipu_stats = describe_counts(ipu)
    ipq_stats = describe_counts(ipq)

    # Concepts-per-question (from domain mapping)
    if q2s:
        n_concepts_per_q_all = np.array([len(set(v)) for v in q2s.values()], dtype=float)
        mean_cpq_all = float(np.mean(n_concepts_per_q_all))
        min_cpq_all = int(np.min(n_concepts_per_q_all))
        max_cpq_all = int(np.max(n_concepts_per_q_all))
        n_concepts_from_qmeta = len({sid for sids in q2s.values() for sid in sids})
    else:
        mean_cpq_all = float("nan")
        min_cpq_all = -1
        max_cpq_all = -1
        n_concepts_from_qmeta = 0

    # Restrict CPQ to questions appearing in the log (if raw->QuestionId exists)
    mean_cpq_in_log = float("nan")
    n_questions_in_log_mapped = 0
    if qraw_map_col is not None:
        qmap_small = qmap[[qraw_map_col, qid_map_col]].dropna().copy()

        # Make ids comparable when possible
        for c in [qraw_map_col, qid_map_col]:
            try:
                qmap_small[c] = qmap_small[c].astype(int)
            except Exception:
                pass

        qraw_in_log = df[qraw_col].dropna().unique().tolist()
        qraw_in_log_set = set(qraw_in_log)

        qids_in_log = (
            qmap_small[qmap_small[qraw_map_col].isin(qraw_in_log_set)][qid_map_col]
            .dropna()
            .unique()
            .tolist()
        )
        qids_in_log_set = {int(q) for q in qids_in_log}
        n_questions_in_log_mapped = len(qids_in_log_set)

        cpq_list = []
        for qid in qids_in_log_set:
            sids = q2s.get(int(qid), [])
            if sids:
                cpq_list.append(len(set(sids)))

        if cpq_list:
            mean_cpq_in_log = float(np.mean(np.array(cpq_list, dtype=float)))

    # Hierarchy depth stats
    max_depth = max(depth_map.values()) if depth_map else 0
    n_levels = int(max_depth + 1) if depth_map else 0

    counts_per_depth: dict[int, int] = defaultdict(int)
    for d in depth_map.values():
        counts_per_depth[int(d)] += 1
    counts_per_depth = dict(sorted(counts_per_depth.items()))

    # Simple sparsity proxies
    avg_interactions_per_user = float(n_interactions / max(1, n_users))
    avg_interactions_per_question = float(n_interactions / max(1, n_questions_mapped))

    # Console report
    print("\n=== JUNYI DATASET STATISTICS ===\n")
    print("Detected columns:")
    print(f"  user_col     = {user_col}")
    print(f"  question_col = {qraw_col}")
    print(f"  label_col    = {y_col}")
    print(f"  qmap_raw_col = {qraw_map_col}")
    print(f"  qmap_qid_col = {qid_map_col}\n")

    print(f"Interactions: {n_interactions:,}")
    print(f"Learners:     {n_users:,}")
    print(f"Questions (mapped): {n_questions_mapped:,}")
    print(f"Concepts (hierarchy subjects): {n_subjects_hierarchy:,}")
    print(f"Global accuracy: {acc_global:.4f}\n")

    print("Interactions per user (summary):")
    for k, v in ipu_stats.items():
        print(f"  {k:>5s}: {v:.2f}")

    print("\nInteractions per raw-question in log (summary):")
    for k, v in ipq_stats.items():
        print(f"  {k:>5s}: {v:.2f}")

    print("\nConcept stats (from domain question->subjects mapping):")
    print(f"  Concepts covered by mapping: {n_concepts_from_qmeta:,}")
    print(f"  Mean concepts/question (all mapped questions): {mean_cpq_all:.3f}")
    print(f"  Min/Max concepts/question (all mapped questions): {min_cpq_all} / {max_cpq_all}")
    if not np.isnan(mean_cpq_in_log):
        print(f"  Mean concepts/question (questions in log): {mean_cpq_in_log:.3f} (n_q={n_questions_in_log_mapped:,})")

    print("\nHierarchy depth stats (from domain tree):")
    print(f"  Max depth: {max_depth}  (levels={n_levels})")
    print("  Concepts per depth:")
    for d, cnt in counts_per_depth.items():
        print(f"    depth {d:2d}: {cnt:,}")

    print("\nSparsity indicators:")
    print(f"  Avg interactions/user     : {avg_interactions_per_user:.2f}")
    print(f"  Avg interactions/question : {avg_interactions_per_question:.2f}")

    # Optional one-row CSV output
    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)

        row: dict[str, Any] = {
            "n_users": n_users,
            "n_questions_mapped": n_questions_mapped,
            "n_subjects_hierarchy": n_subjects_hierarchy,
            "n_interactions": n_interactions,
            "global_accuracy": acc_global,
            "concepts_from_q2s_mapping": n_concepts_from_qmeta,
            "mean_concepts_per_question_all": mean_cpq_all,
            "min_concepts_per_question_all": min_cpq_all,
            "max_concepts_per_question_all": max_cpq_all,
            "mean_concepts_per_question_in_log": None if np.isnan(mean_cpq_in_log) else mean_cpq_in_log,
            "n_questions_in_log_mapped": n_questions_in_log_mapped,
            "hierarchy_max_depth": max_depth,
            "hierarchy_n_levels": n_levels,
            "avg_interactions_per_user": avg_interactions_per_user,
            "avg_interactions_per_question": avg_interactions_per_question,
            "ipu_min": ipu_stats["min"],
            "ipu_p25": ipu_stats["25%"],
            "ipu_median": ipu_stats["50%"],
            "ipu_p75": ipu_stats["75%"],
            "ipu_max": ipu_stats["max"],
            "ipu_mean": ipu_stats["mean"],
            "ipq_min": ipq_stats["min"],
            "ipq_p25": ipq_stats["25%"],
            "ipq_median": ipq_stats["50%"],
            "ipq_p75": ipq_stats["75%"],
            "ipq_max": ipq_stats["max"],
            "ipq_mean": ipq_stats["mean"],
            "detected_user_col": user_col,
            "detected_question_col": qraw_col,
            "detected_label_col": y_col,
            "qmap_raw_col": qraw_map_col,
            "qmap_qid_col": qid_map_col,
        }

        # Flatten depth counts (n_concepts_depth_0, n_concepts_depth_1, ...)
        for d, cnt in counts_per_depth.items():
            row[f"n_concepts_depth_{d}"] = int(cnt)

        pd.DataFrame([row]).to_csv(out, index=False)
        print(f"\nWrote summary CSV to {out}")


if __name__ == "__main__":
    main()
