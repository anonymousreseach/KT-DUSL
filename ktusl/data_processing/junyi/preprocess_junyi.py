# data_processing/junyi/preprocess_junyi.py
# -*- coding: utf-8 -*-

import os
import sys
import logging

import pandas as pd

# ➜ Add the parent folder (data_processing) to PYTHONPATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # .../ktusl/data_processing
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.io_utils import write_txt
from utils.time_utils import change2timestamp
from utils.stats_utils import sta_infos

# Key columns for stats
KEYS = ["user_id", "topic", "exercise"]


# ---------------------------------------------------------------------------
# 1) Load the content and create integer ID mappings
# ---------------------------------------------------------------------------

def load_q2c(content_path: str):
    """
    Load Info_Content.csv and build:
      - the most fine-grained concept per exercise (level4 > level3 > level2 > level1)
      - a mapping exercise -> (qid_int, cid_int).
    """
    print(f"[JUNYI] Loading content mapping from: {content_path}")
    df = pd.read_csv(content_path)

    # Ensure columns exist
    for col in ["level1_id", "level2_id", "level3_id", "level4_id"]:
        if col not in df.columns:
            df[col] = pd.NA

    def choose_finest(row):
        # Traverse from the most granular to the coarsest
        for col in ["level4_id", "level3_id", "level2_id", "level1_id"]:
            val = row[col]
            if pd.notna(val) and str(val) not in ["", "nan", "None"]:
                return str(val)
        # If no level provided: use ucid as the concept
        return str(row["ucid"])

    df = df.dropna(subset=["ucid"])
    df["concept_str"] = df.apply(choose_finest, axis=1)

    # Encode as integers
    ex_values = sorted(df["ucid"].unique())
    ex2id = {v: i for i, v in enumerate(ex_values)}

    concept_values = sorted(df["concept_str"].unique())
    concept2id = {v: i for i, v in enumerate(concept_values)}

    dq2c = {}
    for _, row in df.iterrows():
        ucid = row["ucid"]
        concept = row["concept_str"]
        qid = ex2id[ucid]
        cid = concept2id[concept]
        dq2c[ucid] = (qid, cid)

    print(
        f"[JUNYI] Loaded {len(ex2id)} unique exercises "
        f"and {len(concept2id)} finest-level concepts."
    )
    return dq2c, ex2id, concept2id


# ---------------------------------------------------------------------------
# 2) Read the log and build sequences
# ---------------------------------------------------------------------------

def read_data_from_csv(log_path: str, write_file: str, dq2c: dict):
    """
    Preprocess JUNYI from Log_Problem.csv + dq2c mapping,
    and write user sequences to write_file.
    Here dq2c: ucid (string) -> (exercise_id_int, topic_id_int)
    """
    stares = []

    print(f"[JUNYI] Loading interaction log from: {log_path}")
    df = pd.read_csv(log_path)

    # Rename columns into the pipeline's "standard" format
    rename_map = {
        "uuid": "user_id",
        "ucid": "exercise_raw",  # keep raw ucid for the mapping
        "timestamp_TW": "time_done",
        "is_correct": "correct",
        "total_sec_taken": "time_taken_attempts",
        "total_attempt_cnt": "count_attempts",
    }
    missing_cols = [c for c in rename_map.keys() if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing expected columns in Log_Problem.csv: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.rename(columns=rename_map)

    # ------------------------------------------------------------------
    # Map exercise/concept to integer IDs
    # ------------------------------------------------------------------
    def map_exercise(ucid):
        if ucid in dq2c:
            return dq2c[ucid][0]
        else:
            return None

    def map_topic(ucid):
        if ucid in dq2c:
            return dq2c[ucid][1]
        else:
            return None

    df["exercise"] = df["exercise_raw"].map(map_exercise)
    df["topic"] = df["exercise_raw"].map(map_topic)

    # Drop interactions without a known mapping
    df = df.dropna(subset=["exercise", "topic"])

    # Convert to int
    df["exercise"] = df["exercise"].astype(int)
    df["topic"] = df["topic"].astype(int)

    # Initial stats
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(
        f"[JUNYI] Original (mapped): interactions={ins}, users={us}, questions={qs}, "
        f"concepts={cs}, avg_ins/user={avgins:.2f}, avg_concepts/question={avgcq:.2f}, na={na}"
    )

    # Index to stabilize sorting
    df["index"] = range(df.shape[0])
    print(f"[JUNYI] Original records shape: {df.shape}")

    # Columns used to build sequences
    usedf = df[
        [
            "index",
            "user_id",
            "exercise",
            "time_done",
            "time_taken_attempts",
            "correct",
            "count_attempts",
            "topic",
        ]
    ]

    # Cleaning: drop essential NAs
    usedf = usedf.dropna(
        subset=["user_id", "exercise", "time_done", "correct"]
    )

    # Keep only correct values in {0,1,True,False}
    usedf = usedf[usedf["correct"].isin([0, 1, True, False])]

    # Convert attempt time: total_sec_taken (sec) -> ms and string
    usedf["time_taken_attempts"] = (
        usedf["time_taken_attempts"].fillna(-100).astype(float) * 1000.0
    ).astype(int).astype(str)

    # Convert 'time_done'
    if usedf["time_done"].dtype == object:
        usedf["time_done"] = usedf["time_done"].apply(
            lambda x: change2timestamp(x, hasf="." in str(x))
        )
    usedf = usedf.dropna(subset=["time_done"])
    usedf["time_done"] = usedf["time_done"].astype(int)

    # Stats after cleaning
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(usedf, KEYS, stares)
    print(
        f"[JUNYI] After cleaning: interactions={ins}, users={us}, questions={qs}, "
        f"concepts={cs}, avg_ins/user={avgins:.2f}, avg_concepts/question={avgcq:.2f}, na={na}"
    )

    # ------------------------------------------------------------------
    # Encode user_id (uuid) as integers
    # ------------------------------------------------------------------
    unique_users = usedf["user_id"].unique()
    user2id = {u: i for i, u in enumerate(unique_users)}
    print(f"[JUNYI] Unique users: {len(user2id)}")

    usedf["user_id_int"] = usedf["user_id"].map(user2id)

    data = []
    uids = usedf["user_id_int"].nunique()
    problems = usedf["exercise"].nunique()
    print(f"[JUNYI] Usedf shape: {usedf.shape}, unique users={uids}, unique exercises={problems}")

    # Group by encoded user
    ui_df = usedf.groupby("user_id_int", sort=False)

    for uid_int, curdf in ui_df:
        # Chronological sort + index to break ties
        curdf = curdf.sort_values(by=["time_done", "index"])

        # time_done as string for writing
        curdf["time_done"] = curdf["time_done"].astype(int).astype(str)

        questions = curdf["exercise"].astype(int).astype(str).tolist()
        concepts = curdf["topic"].astype(int).astype(str).tolist()
        rs = curdf["correct"].astype(int).astype(str).tolist()
        ts = curdf["time_done"].tolist()
        uts = curdf["time_taken_attempts"].tolist()
        seq_len = len(rs)

        # uc = [user_id_int, seq_len]
        uc = [str(uid_int), str(seq_len)]
        data.append([uc, questions, concepts, rs, ts, uts])

        if len(data) % 1000 == 0:
            print(f"[JUNYI] Processed {len(data)} user sequences so far...")

    # Write sequences to write_file
    print(f"[JUNYI] Writing sequences to: {write_file}")
    write_txt(write_file, data)

    print("[JUNYI] Done.")
    print("[JUNYI] Stats log:")
    for line in stares:
        print("  ", line)

    return


# ---------------------------------------------------------------------------
# 3) main
# ---------------------------------------------------------------------------

def main():
    # Hard-coded, simple, and robust paths (run from data_processing/)
    content_path = os.path.join("junyi", "raw", "Info_Content.csv")
    log_path = os.path.join("junyi", "raw", "Log_Problem.csv")
    out_path = os.path.join("junyi", "processed", "junyi_sequences.txt")
    log_dir = os.path.join("junyi", "logs")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "junyi_preprocess.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("[JUNYI] Starting preprocessing...")
    print(f"[JUNYI] Content file : {content_path}")
    print(f"[JUNYI] Log file     : {log_path}")
    print(f"[JUNYI] Output file  : {out_path}")
    print(f"[JUNYI] Log file     : {log_file}")

    dq2c, ex2id, concept2id = load_q2c(content_path)
    read_data_from_csv(log_path, out_path, dq2c)


if __name__ == "__main__":
    main()
