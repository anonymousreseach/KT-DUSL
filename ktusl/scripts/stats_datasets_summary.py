from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd


# ===================== Paths =====================
# ASSIST2015
ASSIST_RAW = Path("ktusl/data_processing/assist2015/raw/assist2015_raw.csv")
ASSIST_SEP = ","

# EEDI
EEDI_RAW_DIR = Path("ktusl/data_processing/eedi/raw")
EEDI_TASK = "task_1_2"
EEDI_QUESTION_META = EEDI_RAW_DIR / f"question_metadata_{EEDI_TASK}.csv"
EEDI_SUBJECT_META = EEDI_RAW_DIR / "subject_metadata.csv"
EEDI_SEQ_FILE = Path(f"ktusl/data_processing/eedi/processed/eedi_{EEDI_TASK}_sequences.txt")

# JUNYI
JUNYI_CONTENT = Path("ktusl/data_processing/junyi/raw/Info_Content.csv")
JUNYI_SEQ_FILE = Path("ktusl/data_processing/junyi/processed/junyi_sequences.txt")

# PREDS (used for learner–concept mean)
PREDS_DIR = Path("outputs")
PREDS = {
    "eedi": PREDS_DIR / "preds_bkt_eedi.csv",
    "junyi": PREDS_DIR / "preds_bkt_junyi.csv",
}


# ===================== Generic helpers =====================
def detect_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def normalize_id_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out


def read_sequences_txt_counts(path: Path) -> pd.DataFrame:
    """Return DataFrame(learner_id, n_interactions) from sequences file."""
    if not path.exists():
        raise FileNotFoundError(f"Sequence file not found: {path}")

    learner_ids, n_ints = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            learner = tokens[0]
            length = None
            for t in tokens[1:6]:
                if t.isdigit():
                    length = int(t)
                    break
            if length is None:
                continue
            learner_ids.append(str(learner))
            n_ints.append(int(length))
    return pd.DataFrame({"learner_id": learner_ids, "n_interactions": n_ints})


def correct_rate_from_sequences_txt(path: Path) -> float:
    """
    Global correct rate from any long-enough 0/1 block in sequences file.
    (Same method you used; consistent across EEDI/Junyi.)
    """
    if not path.exists():
        raise FileNotFoundError(f"Sequence file not found: {path}")
    n_pos, n_tot = 0, 0
    pat = re.compile(r"(?:\b[01]\b[, ]*){6,}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            vals = [int(v) for v in re.findall(r"[01]", m.group(0))]
            n_pos += sum(vals)
            n_tot += len(vals)
    return float(n_pos / n_tot) if n_tot > 0 else 0.0


def mean_lc_from_preds_with_mapping(preds_csv: Path, q2c: Dict[str, str]) -> float:
    """Mean over observed (learner, concept) pairs: mean_{(l,c)} count(l,c)."""
    if not preds_csv.exists():
        print(f"[WARN] Missing preds file: {preds_csv}")
        return np.nan

    df = pd.read_csv(preds_csv)
    ucol = detect_col(df, ["UserId", "user_id", "learner_id"])
    qcol = detect_col(df, ["QuestionId", "question_id", "ucid", "item_id"])
    if ucol is None or qcol is None:
        print(f"[WARN] Could not detect user/question columns in {preds_csv}. Columns={list(df.columns)}")
        return np.nan

    df = df.dropna(subset=[ucol, qcol]).copy()
    df[ucol] = normalize_id_series(df[ucol])
    df[qcol] = normalize_id_series(df[qcol])

    df["concept_id"] = df[qcol].map(q2c)
    df = df.dropna(subset=["concept_id"])
    if len(df) == 0:
        print(f"[WARN] No rows after mapping QuestionId->Concept for {preds_csv}")
        return np.nan

    return float(df.groupby([ucol, "concept_id"]).size().mean())


# ===================== EEDI mapping =====================
def parse_subject_list(s) -> List[int]:
    if pd.isna(s):
        return []
    try:
        xs = eval(str(s))
        return [int(x) for x in xs] if isinstance(xs, (list, tuple, set)) else []
    except Exception:
        return []


def build_q2c_eedi_lvl3() -> Dict[str, str]:
    df_sub = pd.read_csv(EEDI_SUBJECT_META)
    keep_lvl3: Set[int] = set(df_sub.loc[df_sub["Level"] == 3, "SubjectId"].astype(int))

    df_q = pd.read_csv(EEDI_QUESTION_META)
    qcol = detect_col(df_q, ["QuestionId", "question_id", "questionId"])
    if qcol is None:
        raise ValueError(f"[EEDI] Cannot find question id column in {EEDI_QUESTION_META}. Columns={list(df_q.columns)}")
    if "SubjectId" not in df_q.columns:
        raise ValueError(f"[EEDI] Missing SubjectId in {EEDI_QUESTION_META}. Columns={list(df_q.columns)}")

    df_q["subjects"] = df_q["SubjectId"].apply(parse_subject_list)

    def pick_lvl3(xs):
        for x in xs:
            if x in keep_lvl3:
                return str(x)
        return None

    df_q["concept_lvl3"] = df_q["subjects"].apply(pick_lvl3)
    df_q = df_q.dropna(subset=["concept_lvl3"]).copy()

    qids = normalize_id_series(df_q[qcol])
    cids = normalize_id_series(df_q["concept_lvl3"])
    return dict(zip(qids.tolist(), cids.tolist()))


# ===================== Junyi mapping =====================
def choose_finest(row) -> str:
    for col in ["level4_id", "level3_id", "level2_id", "level1_id"]:
        v = row.get(col, None)
        if pd.notna(v) and str(v) not in ["", "nan", "None"]:
            return str(v)
    return str(row["ucid"])


def build_ucid_to_concept_junyi() -> Dict[str, str]:
    df = pd.read_csv(JUNYI_CONTENT)
    for c in ["level1_id", "level2_id", "level3_id", "level4_id"]:
        if c not in df.columns:
            df[c] = pd.NA
    if "ucid" not in df.columns:
        raise ValueError(f"[Junyi] Missing 'ucid' in {JUNYI_CONTENT}. Columns={list(df.columns)}")

    df = df.dropna(subset=["ucid"]).copy()
    df["concept"] = df.apply(choose_finest, axis=1)

    ucid = normalize_id_series(df["ucid"])
    concept = normalize_id_series(df["concept"])
    return dict(zip(ucid.tolist(), concept.tolist()))


def build_q2c_junyi_best(preds_csv: Path) -> Dict[str, str]:
    """
    Builds QuestionId->concept mapping for Junyi:
      - Try direct mapping (QuestionId==ucid)
      - Else try indexed mapping (QuestionId==0..N-1) with two ordering options
    Returns the mapping with best coverage on preds.
    """
    ucid_to_concept = build_ucid_to_concept_junyi()

    preds = pd.read_csv(preds_csv)
    qcol = detect_col(preds, ["QuestionId", "question_id", "ucid", "item_id"])
    if qcol is None:
        return {}

    qids = normalize_id_series(preds[qcol].dropna())

    # --- (1) direct: QuestionId is ucid
    direct = {qid: ucid_to_concept.get(qid) for qid in pd.unique(qids)}
    direct = {k: v for k, v in direct.items() if v is not None and str(v) != "nan"}
    cov_direct = float(qids.map(direct).notna().mean()) if len(qids) else 0.0

    # --- (2) indexed: QuestionId is 0..N-1
    # Build unique ucids lists
    base = pd.read_csv(JUNYI_CONTENT)
    base = base.dropna(subset=["ucid"]).copy()
    ucids_raw = normalize_id_series(base["ucid"])
    ucids_appearance = pd.unique(ucids_raw).tolist()

    # sorted order (numeric if possible)
    def _to_int_or_none(x: str):
        try:
            return int(x)
        except Exception:
            return None

    ints = [(_to_int_or_none(x), x) for x in ucids_appearance]
    if all(a is not None for a, _ in ints):
        ucids_sorted = [x for _, x in sorted(ints, key=lambda t: t[0])]
    else:
        ucids_sorted = sorted(ucids_appearance)

    q_int = pd.to_numeric(qids, errors="coerce").dropna().astype(int)
    cov_app = 0.0
    cov_sort = 0.0
    indexed_app: Dict[str, str] = {}
    indexed_sort: Dict[str, str] = {}

    if len(q_int) > 0:
        max_q = int(q_int.max())
        min_q = int(q_int.min())
        if min_q >= 0 and max_q < len(ucids_appearance):
            idx_to_concept_app = {
                str(i): ucid_to_concept.get(ucids_appearance[i]) for i in range(len(ucids_appearance))
            }
            idx_to_concept_app = {k: v for k, v in idx_to_concept_app.items() if v is not None and str(v) != "nan"}
            cov_app = float(qids.map(idx_to_concept_app).notna().mean())
            indexed_app = idx_to_concept_app

            idx_to_concept_sort = {
                str(i): ucid_to_concept.get(ucids_sorted[i]) for i in range(len(ucids_sorted))
            }
            idx_to_concept_sort = {k: v for k, v in idx_to_concept_sort.items() if v is not None and str(v) != "nan"}
            cov_sort = float(qids.map(idx_to_concept_sort).notna().mean())
            indexed_sort = idx_to_concept_sort

    # pick best
    best_cov = max(cov_direct, cov_app, cov_sort)
    if best_cov == cov_direct:
        print(f"[INFO] Junyi mapping: direct coverage={cov_direct:.3f} (chosen)")
        return direct
    if best_cov == cov_app:
        print(f"[INFO] Junyi mapping: indexed-appearance coverage={cov_app:.3f} (chosen)")
        return indexed_app
    print(f"[INFO] Junyi mapping: indexed-sorted coverage={cov_sort:.3f} (chosen)")
    return indexed_sort


# ===================== Dataset summaries (minimal) =====================
def summary_assist2015() -> dict:
    df = pd.read_csv(ASSIST_RAW, sep=ASSIST_SEP, low_memory=False)
    df = df.dropna(subset=["user_id", "sequence_id"]).copy()

    n_learners = int(df["user_id"].nunique())
    n_interactions = int(len(df))
    n_concepts = int(df["sequence_id"].nunique())
    correct_rate = correct_rate_from_df(df, ["correct", "is_correct", "IsCorrect"])
    mean_lc = float(df.groupby(["user_id", "sequence_id"]).size().mean())

    return dict(
        dataset="assist2015",
        n_learners=n_learners,
        n_interactions=n_interactions,
        n_concepts=n_concepts,
        correct_rate=correct_rate,
        mean_interactions_per_learner_concept=mean_lc,
    )


def summary_eedi() -> dict:
    seq = read_sequences_txt_counts(EEDI_SEQ_FILE)
    q2c = build_q2c_eedi_lvl3()

    n_learners = int(len(seq))
    n_interactions = int(seq["n_interactions"].sum())
    n_concepts = int(len(set(q2c.values())))
    correct_rate = correct_rate_from_sequences_txt(EEDI_SEQ_FILE)

    mean_lc = mean_lc_from_preds_with_mapping(PREDS["eedi"], q2c)

    return dict(
        dataset="eedi",
        n_learners=n_learners,
        n_interactions=n_interactions,
        n_concepts=n_concepts,
        correct_rate=correct_rate,
        mean_interactions_per_learner_concept=mean_lc,
    )


def summary_junyi() -> dict:
    seq = read_sequences_txt_counts(JUNYI_SEQ_FILE)

    n_learners = int(len(seq))
    n_interactions = int(seq["n_interactions"].sum())
    correct_rate = correct_rate_from_sequences_txt(JUNYI_SEQ_FILE)

    preds_path = PREDS["junyi"]
    if not preds_path.exists():
        print(f"[WARN] Missing preds file: {preds_path}")
        return dict(
            dataset="junyi",
            n_learners=n_learners,
            n_interactions=n_interactions,
            n_concepts=0,
            correct_rate=correct_rate,
            mean_interactions_per_learner_concept=np.nan,
        )

    q2c = build_q2c_junyi_best(preds_path)
    n_concepts = int(len(set(q2c.values()))) if q2c else 0
    mean_lc = mean_lc_from_preds_with_mapping(preds_path, q2c)

    return dict(
        dataset="junyi",
        n_learners=n_learners,
        n_interactions=n_interactions,
        n_concepts=n_concepts,
        correct_rate=correct_rate,
        mean_interactions_per_learner_concept=mean_lc,
    )
def correct_rate_from_df(df: pd.DataFrame, candidates: List[str]) -> float:
    """
    Compute global correct rate from a dataframe by detecting
    a binary correctness column.
    """
    col = detect_col(df, candidates)
    if col is None:
        print(f"[WARN] No correctness column found among {candidates}")
        return 0.0

    y = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(y) == 0:
        return 0.0

    return float((y == 1).mean())


# ===================== Main =====================
def main():
    rows = [summary_eedi(), summary_junyi(), summary_assist2015()]
    out = pd.DataFrame(rows, columns=[
        "dataset",
        "n_learners",
        "n_interactions",
        "n_concepts",
        "correct_rate",
        "mean_interactions_per_learner_concept",
    ])

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

    print("\n=== DATASETS SUMMARY (MIN) ===")
    print(out.to_string(index=False))

    out_path = Path("outputs/datasets_summary_min.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\n[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
