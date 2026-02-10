#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocess_junyi_domain.py

Build a domain subject tree for Junyi from info_content.csv.

Junyi IDs (ucid, level*_id) are often base64-like strings (non-numeric).
We therefore FACTORIZE question ids to integers so that evaluate_next_correctness.py can cast to int.
We also output a question_id_map.csv (ucid -> QuestionId int) to reuse in log preprocessing.

Hierarchy strategy:
1) If level1_id..level4_id exist and have non-null values -> build multi-level tree using those ids (as strings).
2) Else fallback to subject (flat).
3) Else fallback to learning_stage (flat).
4) Else single root.

Outputs:
- subject_tree_with_questions.json  (compatible with scripts/evaluate_next_correctness.py)
- subject_id_map.csv                (maps hierarchy keys -> subject_id int)
- question_id_map.csv               (maps ucid (raw) -> QuestionId int)
- question_index.csv (optional)      (question_id -> attached subject_id)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


LEVEL_COLS_DEFAULT = ["level1_id", "level2_id", "level3_id", "level4_id"]


def _norm_col(c: str) -> str:
    return c.strip().replace("\t", " ").replace("\u00a0", " ")


def _canon(c: str) -> str:
    return _norm_col(c).lower().replace(" ", "").replace("_", "")


def _as_str_or_none(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    if s == "0":
        return None
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--info_content_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_subject_map", required=True)
    ap.add_argument("--out_question_map", required=True,
                    help="CSV mapping ucid -> QuestionId (int). Use this in log preprocessing.")
    ap.add_argument("--out_question_index", default=None)

    ap.add_argument("--ucid_col", default="ucid")
    ap.add_argument("--pretty_name_col", default="content_pretty_name")
    ap.add_argument("--kind_col", default="content_kind")
    ap.add_argument("--difficulty_col", default="difficulty")
    ap.add_argument("--subject_col", default="subject")
    ap.add_argument("--learning_stage_col", default="learning_stage")

    ap.add_argument("--level_cols", nargs="+", default=LEVEL_COLS_DEFAULT)
    args = ap.parse_args()

    in_path = Path(args.info_content_csv)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    df = pd.read_csv(in_path)
    df.columns = [_norm_col(c) for c in df.columns]
    col_lookup = {_canon(c): c for c in df.columns}

    def resolve(col: str) -> Optional[str]:
        return col_lookup.get(_canon(col))

    ucid_col = resolve(args.ucid_col)
    if ucid_col is None:
        raise ValueError(f"Cannot resolve ucid column '{args.ucid_col}'. Available: {list(df.columns)}")

    pretty_col = resolve(args.pretty_name_col)
    kind_col = resolve(args.kind_col)
    diff_col = resolve(args.difficulty_col)
    subj_col = resolve(args.subject_col)
    stage_col = resolve(args.learning_stage_col)

    # Resolve level columns (if present)
    level_cols: List[str] = []
    level_cols_ok = True
    for c in args.level_cols:
        rc = resolve(c)
        if rc is None:
            level_cols_ok = False
            break
        level_cols.append(rc)

    # --------- Factorize question ids (ucid -> int) ---------
    ucids_norm = df[ucid_col].map(_as_str_or_none)
    valid_mask = ucids_norm.notna()
    if int(valid_mask.sum()) == 0:
        raise ValueError("No valid ucid found in info_content.csv (all empty/NA).")

    # factorize only valid ucids; deterministic with sort=True
    q_codes, q_uniques = pd.factorize(ucids_norm[valid_mask].astype(str), sort=True)

    df = df.copy()
    df["ucid_norm"] = ucids_norm
    df["QuestionId"] = pd.NA
    df.loc[valid_mask, "QuestionId"] = q_codes.astype(int)
    df["QuestionId"] = df["QuestionId"].astype("Int64")

    # Write question map
    q_map_df = pd.DataFrame({
        "ucid": q_uniques.astype(str),
        "QuestionId": list(range(len(q_uniques))),
    })
    out_qmap = Path(args.out_question_map)
    out_qmap.parent.mkdir(parents=True, exist_ok=True)
    q_map_df.to_csv(out_qmap, index=False)
    print("[INFO] Wrote question_id_map:", str(out_qmap), "n=", len(q_map_df))

    # --------- Determine hierarchy mode ---------
    def level_path_from_series(row: pd.Series) -> List[Tuple[int, str]]:
        path: List[Tuple[int, str]] = []
        for i, c in enumerate(level_cols, start=1):
            v = _as_str_or_none(row.get(c))
            if v is not None:
                path.append((i, v))
        return path

    hierarchy_mode = "levels"
    usable_levels = False
    if level_cols_ok:
        sample = df.head(5000)
        for _, r in sample.iterrows():
            if level_path_from_series(r):
                usable_levels = True
                break

    if not usable_levels:
        if subj_col is not None:
            hierarchy_mode = "subject"
        elif stage_col is not None:
            hierarchy_mode = "learning_stage"
        else:
            hierarchy_mode = "single_root"

    print(f"[INFO] hierarchy_mode={hierarchy_mode}")
    if hierarchy_mode == "levels":
        print(f"[INFO] using level columns: {level_cols}")
    else:
        print(f"[INFO] fallback columns: subject={subj_col} learning_stage={stage_col}")

    # --------- Build nodes (subject tree) ---------
    nodes: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    children_of: Dict[Tuple[Any, Any], set] = {}
    subject_id_of: Dict[Tuple[Any, Any], int] = {}
    next_sid = 1

    def ensure_node(key: Tuple[Any, Any], name: str, raw_level: int, raw_id: str) -> int:
        nonlocal next_sid
        if key not in subject_id_of:
            subject_id_of[key] = next_sid
            nodes[key] = {
                "subject_id": next_sid,
                "name": name,
                "raw_level": int(raw_level),
                "raw_id": str(raw_id),
                "children": [],
                "questions": [],
            }
            next_sid += 1
        return subject_id_of[key]

    # Roots depending on mode
    if hierarchy_mode in {"subject", "learning_stage"}:
        ensure_node(("root", "junyi"), "junyi", raw_level=0, raw_id="junyi")
    elif hierarchy_mode == "single_root":
        ensure_node(("root", "junyi_all"), "junyi_all", raw_level=0, raw_id="junyi_all")

    q_index_rows: List[Dict[str, Any]] = []

    # Iterate rows safely with iterrows (simple + robust for column names)
    for _, r in df.iterrows():
        qid = r.get("QuestionId")
        if pd.isna(qid):
            continue
        qid = int(qid)

        q: Dict[str, Any] = {"question_id": qid}
        if pretty_col:
            q["name"] = str(r.get(pretty_col))
        if kind_col:
            q["kind"] = str(r.get(kind_col))
        if diff_col:
            q["difficulty"] = r.get(diff_col)
        if subj_col:
            sv = _as_str_or_none(r.get(subj_col))
            if sv is not None:
                q["subject"] = sv
        if stage_col:
            stv = _as_str_or_none(r.get(stage_col))
            if stv is not None:
                q["learning_stage"] = stv

        if hierarchy_mode == "levels":
            path = level_path_from_series(r)
            if not path:
                continue

            prev_key = None
            for lvl, rid in path:
                key = (lvl, rid)  # rid is string
                ensure_node(key, name=f"level{lvl}_{rid}", raw_level=lvl, raw_id=rid)
                if prev_key is not None:
                    children_of.setdefault(prev_key, set()).add(key)
                prev_key = key
            leaf_key = (path[-1][0], path[-1][1])

        elif hierarchy_mode == "subject":
            sv = _as_str_or_none(r.get(subj_col)) if subj_col else None
            if sv is None:
                continue
            leaf_key = ("subject", sv)
            ensure_node(leaf_key, name=sv, raw_level=1, raw_id=sv)
            children_of.setdefault(("root", "junyi"), set()).add(leaf_key)

        elif hierarchy_mode == "learning_stage":
            stv = _as_str_or_none(r.get(stage_col)) if stage_col else None
            if stv is None:
                continue
            leaf_key = ("learning_stage", stv)
            ensure_node(leaf_key, name=stv, raw_level=1, raw_id=stv)
            children_of.setdefault(("root", "junyi"), set()).add(leaf_key)

        else:
            leaf_key = ("root", "junyi_all")

        nodes[leaf_key]["questions"].append(q)
        q_index_rows.append({
            "question_id": qid,
            "attached_subject_id": int(nodes[leaf_key]["subject_id"]),
            "hierarchy_mode": hierarchy_mode,
        })

    # Guard: at least one question attached
    n_attached = sum(len(n.get("questions", [])) for n in nodes.values())
    if n_attached == 0:
        raise ValueError(
            "No questions were attached to any node. "
            "If hierarchy_mode=levels, check that level*_id columns are populated per-row. "
            "Otherwise fallback to subject/learning_stage."
        )

    # Deduplicate questions per node
    for n in nodes.values():
        seen = set()
        uniq = []
        for qq in n["questions"]:
            if qq["question_id"] in seen:
                continue
            seen.add(qq["question_id"])
            uniq.append(qq)
        n["questions"] = uniq

    # Roots selection
    if hierarchy_mode == "levels":
        all_children = set()
        for cs in children_of.values():
            all_children.update(cs)
        roots_keys = [k for k in nodes if isinstance(k[0], int) and k[0] == 1 and k not in all_children]
        if not roots_keys:
            roots_keys = [k for k in nodes if isinstance(k[0], int) and k[0] == 1]
    elif hierarchy_mode in {"subject", "learning_stage"}:
        roots_keys = [("root", "junyi")]
    else:
        roots_keys = [("root", "junyi_all")]

    def build(key: Tuple[Any, Any]) -> Dict[str, Any]:
        n = nodes[key]
        ch = [build(k) for k in sorted(children_of.get(key, set()), key=lambda x: (str(x[0]), str(x[1])))]
        return {
            "subject_id": int(n["subject_id"]),
            "raw_level": int(n.get("raw_level", 0)),
            "raw_id": str(n.get("raw_id", "")),
            "name": n.get("name"),
            "children": ch,
            "questions": n.get("questions", []),
        }

    roots = [build(k) for k in sorted(roots_keys, key=lambda x: (str(x[0]), str(x[1])))]
    out_json = {"pkg": {"dataset": "junyi", "hierarchy_mode": hierarchy_mode, "roots": roots}}

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print("[INFO] Wrote domain json:", str(out_path))

    # Subject map
    sm_rows = []
    for key, sid in subject_id_of.items():
        sm_rows.append({
            "subject_id": int(sid),
            "key_type": str(key[0]),
            "key_value": str(key[1]),
            "name": nodes[key].get("name"),
            "raw_level": int(nodes[key].get("raw_level", 0)),
            "raw_id": str(nodes[key].get("raw_id", "")),
        })

    sm_path = Path(args.out_subject_map)
    sm_path.parent.mkdir(parents=True, exist_ok=True)
    sm_df = pd.DataFrame(sm_rows)
    if not sm_df.empty:
        sm_df = sm_df.sort_values(["raw_level", "key_type", "key_value", "subject_id"])
    else:
        sm_df = pd.DataFrame(columns=["subject_id", "key_type", "key_value", "name", "raw_level", "raw_id"])
    sm_df.to_csv(sm_path, index=False)
    print("[INFO] Wrote subject_id_map:", str(sm_path))

    # Question index
    if args.out_question_index:
        qi_path = Path(args.out_question_index)
        qi_path.parent.mkdir(parents=True, exist_ok=True)
        qi_df = pd.DataFrame(q_index_rows).drop_duplicates("question_id")
        qi_df.to_csv(qi_path, index=False)
        print("[INFO] Wrote question_index:", str(qi_path))

    print(
        f"[STATS] mode={hierarchy_mode} | subjects={len(subject_id_of)} | "
        f"attached_questions={len(set(r['question_id'] for r in q_index_rows))} | roots={len(roots)}"
    )


if __name__ == "__main__":
    main()
