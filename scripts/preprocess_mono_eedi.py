#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create question_metadata_task_1_2_with_single_leaf.csv

Goal: enforce exactly ONE leaf concept (subject) per question.

Inputs:
  - subject_metadata.csv : must contain (SubjectId, ParentId) or equivalents
  - question_metadata_task_1_2.csv : must contain QuestionId and at least one subject field:
        * SubjectId (single)
        * or SubjectIds / subject_ids (list-like string)
        * or Subjects / subjects (list-like string)

Strategy:
  1) For each question, collect candidate subjects.
  2) Pick the deepest subject (max depth). Tie-break by smaller id.
  3) If not a leaf, replace by a descendant leaf (deterministic: smallest leaf id).
  4) Write output with a single subject id per row (column: SingleLeafSubjectId).
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# -------------------------
# Column resolution helpers
# -------------------------

def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def parse_subjects_cell(x: Any) -> list[int]:
    """
    Parse a subject field that can be:
      - int / float
      - string "123"
      - string "[1,2,3]" or "1,2,3"
      - NaN / None
    Returns list[int].
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    # Already numeric
    if isinstance(x, (int,)):
        return [int(x)]
    if isinstance(x, float) and not pd.isna(x):
        return [int(x)]

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return []

    # Try literal_eval for list-like strings
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple, set)):
            out = []
            for t in v:
                try:
                    out.append(int(t))
                except Exception:
                    pass
            return out
        # single scalar
        try:
            return [int(v)]
        except Exception:
            pass
    except Exception:
        pass

    # Fallback: comma-separated
    if "," in s:
        out = []
        for part in s.split(","):
            part = part.strip()
            if part == "":
                continue
            try:
                out.append(int(part))
            except Exception:
                pass
        return out

    # Fallback: just int string
    try:
        return [int(s)]
    except Exception:
        return []


# -------------------------
# Build hierarchy structures
# -------------------------

def build_hierarchy(subject_meta: pd.DataFrame) -> tuple[dict[int, Optional[int]], dict[int, list[int]], dict[int, int], set[int]]:
    """
    Returns:
      parent_of: child -> parent (or None)
      children_of: parent -> [children]
      depth: node -> depth (roots depth=0)
      leaves: set of leaf nodes
    """
    sid_col = pick_first_existing(subject_meta, ["SubjectId", "subject_id", "SubjectID", "id", "Id"])
    pid_col = pick_first_existing(subject_meta, ["ParentId", "parent_id", "ParentID", "parentId"])
    if sid_col is None or pid_col is None:
        raise ValueError("subject_metadata.csv must have SubjectId and ParentId (or equivalents).")

    df = subject_meta[[sid_col, pid_col]].copy()
    df = df[df[sid_col].notna()].copy()
    df[sid_col] = df[sid_col].astype(int)

    parent_of: dict[int, Optional[int]] = {}
    children_of: dict[int, list[int]] = defaultdict(list)
    all_ids = set(df[sid_col].unique())

    for r in df.itertuples(index=False):
        sid = int(getattr(r, sid_col))
        pid = getattr(r, pid_col)
        if pd.isna(pid):
            parent_of[sid] = None
        else:
            pid = int(pid)
            parent_of[sid] = pid
            children_of[pid].append(sid)

    # Roots: no parent or parent outside known ids
    roots = [sid for sid, pid in parent_of.items() if pid is None or pid not in all_ids]

    # Depth via BFS
    depth: dict[int, int] = {}
    q = deque(roots)
    for r in roots:
        depth[r] = 0

    while q:
        cur = q.popleft()
        for ch in children_of.get(cur, []):
            if ch not in depth:
                depth[ch] = depth[cur] + 1
                q.append(ch)

    leaves = {sid for sid in all_ids if sid not in children_of}
    return parent_of, children_of, depth, leaves


def smallest_leaf_descendant(node: int, children_of: dict[int, list[int]], leaves: set[int]) -> Optional[int]:
    """
    Deterministic: returns the smallest leaf id in the subtree rooted at node.
    If node itself is a leaf, returns node.
    """
    if node in leaves:
        return node

    best: Optional[int] = None
    stack = [node]
    seen = set()

    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)

        ch = children_of.get(cur, [])
        if not ch:
            # leaf encountered (in case leaves set incomplete)
            best = cur if best is None else min(best, cur)
        else:
            for c in ch:
                if c in leaves:
                    best = c if best is None else min(best, c)
                else:
                    stack.append(c)

    return best


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject_meta", required=True, help="Path to subject_metadata.csv")
    ap.add_argument("--question_meta", required=True, help="Path to question_metadata_task_1_2.csv")
    ap.add_argument("--out_csv", required=True, help="Path to output CSV with single leaf per question")
    args = ap.parse_args()

    subject_meta = pd.read_csv(args.subject_meta)
    qmeta = pd.read_csv(args.question_meta)

    parent_of, children_of, depth, leaves = build_hierarchy(subject_meta)

    # Resolve question id column
    qid_col = pick_first_existing(qmeta, ["QuestionId", "question_id", "QuestionID", "id", "Id"])
    if qid_col is None:
        raise ValueError(f"Cannot find QuestionId column in {args.question_meta}. Columns={list(qmeta.columns)}")

    # Resolve subject column(s)
    # Common patterns: SubjectId (single) OR SubjectIds/list-like OR subjects/list-like
    subj_single_col = pick_first_existing(qmeta, ["SubjectId", "subject_id", "SubjectID"])
    subj_multi_col = pick_first_existing(qmeta, ["SubjectIds", "subject_ids", "Subjects", "subjects", "subject"])

    if subj_single_col is None and subj_multi_col is None:
        raise ValueError(
            "Cannot find a subject column. Expected SubjectId or SubjectIds/Subjects-like column. "
            f"Columns={list(qmeta.columns)}"
        )

    # Build candidate subjects per row
    candidates_list: list[list[int]] = []
    for _, r in qmeta.iterrows():
        cand = []
        if subj_multi_col is not None:
            cand = parse_subjects_cell(r.get(subj_multi_col))
        if not cand and subj_single_col is not None:
            cand = parse_subjects_cell(r.get(subj_single_col))
        candidates_list.append(cand)

    # Choose one leaf subject per question
    chosen_leaf: list[Optional[int]] = []
    dropped = 0

    for cand in candidates_list:
        cand = [c for c in cand if c in depth or c in parent_of or c in leaves]  # keep known-ish
        if not cand:
            chosen_leaf.append(None)
            dropped += 1
            continue

        # Pick deepest subject (max depth), tie-break by smallest id
        def d(x: int) -> int:
            return depth.get(x, -1)

        best = sorted(cand, key=lambda x: (-d(x), x))[0]

        # Enforce leaf
        leaf = smallest_leaf_descendant(best, children_of, leaves)
        chosen_leaf.append(leaf)

    out = qmeta.copy()
    out["SingleLeafSubjectId"] = chosen_leaf

    # Optionally also overwrite a canonical SubjectId column
    # (useful if downstream expects SubjectId)
    out["SubjectId_single_leaf"] = chosen_leaf

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    n_total = len(out)
    n_missing = int(out["SingleLeafSubjectId"].isna().sum())
    print(f"[OK] wrote: {out_path}")
    print(f"[STATS] rows={n_total} | missing_subject={n_missing} | hierarchy_leaves={len(leaves)}")


if __name__ == "__main__":
    main()
