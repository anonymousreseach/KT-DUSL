#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import defaultdict, deque

import pandas as pd

EEDI_DIR = "data/eedi/raw"
SUBJECT_META_CSV = os.path.join(EEDI_DIR, "subject_metadata.csv")


def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name (case-insensitive), or None."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def pick_name_column(df: pd.DataFrame) -> str | None:
    """Best-effort column name for human-readable concept labels."""
    return pick_first_existing(
        df,
        ["SubjectName", "subject_name", "Name", "name", "Title", "title", "Label", "label"],
    )


def compute_hierarchy_stats(submeta: pd.DataFrame) -> dict:
    """
    Compute simple hierarchy stats from subject metadata:
      - roots, leaves, max depth, counts per depth, leaf counts per root
    Assumes a parent-pointer representation (SubjectId, ParentId).
    """
    sid_col = pick_first_existing(submeta, ["SubjectId", "subject_id", "SubjectID", "id", "Id"])
    pid_col = pick_first_existing(submeta, ["ParentId", "parent_id", "ParentID", "parentId"])
    if sid_col is None or pid_col is None:
        raise ValueError("subject_metadata.csv must contain SubjectId and ParentId columns (or equivalents).")

    df = submeta[[sid_col, pid_col]].copy()
    df = df[df[sid_col].notna()].copy()
    df[sid_col] = df[sid_col].astype(int)

    parent_of: dict[int, int | None] = {}
    children_of: dict[int, list[int]] = defaultdict(list)
    all_ids = set(df[sid_col].unique())

    # Build parent/children maps
    for r in df.itertuples(index=False):
        sid = int(getattr(r, sid_col))
        pid = getattr(r, pid_col)
        if pd.isna(pid):
            parent_of[sid] = None
        else:
            pid_int = int(pid)
            parent_of[sid] = pid_int
            children_of[pid_int].append(sid)

    # Roots: parent is missing/NaN OR parent not in known ids
    roots = sorted({sid for sid, pid in parent_of.items() if pid is None or pid not in all_ids})

    # BFS for depths from all roots
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

    # Leaves: nodes without children
    leaves = [sid for sid in all_ids if sid not in children_of]

    def leaf_count_under(node: int) -> int:
        """Count leaf descendants under a root (including itself if it is a leaf)."""
        stack = [node]
        cnt = 0
        while stack:
            cur = stack.pop()
            ch = children_of.get(cur, [])
            if not ch:
                cnt += 1
            else:
                stack.extend(ch)
        return cnt

    leaf_count_per_root = {r: leaf_count_under(r) for r in roots}

    counts_per_depth: dict[int, int] = defaultdict(int)
    for d in depth.values():
        counts_per_depth[int(d)] += 1

    return {
        "subject_id_col": sid_col,
        "roots": roots,
        "n_roots": len(roots),
        "n_concepts": len(all_ids),
        "n_leaves": len(leaves),
        "leaf_count_per_root": leaf_count_per_root,
        "max_depth": max(depth.values()) if depth else 0,
        "counts_per_depth": dict(sorted(counts_per_depth.items())),
    }


def main() -> None:
    if not os.path.exists(SUBJECT_META_CSV):
        raise FileNotFoundError(SUBJECT_META_CSV)

    submeta = pd.read_csv(SUBJECT_META_CSV)
    stats = compute_hierarchy_stats(submeta)

    sid_col = stats["subject_id_col"]
    name_col = pick_name_column(submeta)

    print("\n=== EEDI hierarchy stats ===")
    print(f"Total concepts: {stats['n_concepts']}")
    print(f"Roots:          {stats['n_roots']}")
    print(f"Leaves:         {stats['n_leaves']}")

    if name_col is not None:
        id_to_name = (
            submeta[[sid_col, name_col]]
            .dropna()
            .drop_duplicates()
            .set_index(sid_col)[name_col]
            .to_dict()
        )
        print("\nRoots (id -> name):")
        for r in stats["roots"]:
            print(f"  {r}: {id_to_name.get(r, 'UNKNOWN_NAME')}")
    else:
        print("\n[WARN] No name/label column found for concepts.")

    print("\nLeaf descendants per root:")
    for r, cnt in stats["leaf_count_per_root"].items():
        print(f"  root {r}: {cnt}")

    print("\nDepth summary:")
    print(f"Max depth: {stats['max_depth']}")
    for d, c in stats["counts_per_depth"].items():
        print(f"  depth {d}: {c}")


if __name__ == "__main__":
    main()
