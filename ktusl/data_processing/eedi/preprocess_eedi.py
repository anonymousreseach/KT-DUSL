# data_processing/eedi/preprocess_eedi.py
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

from utils.io_utils import write_txt, format_list2str
from utils.time_utils import change2timestamp
from utils.stats_utils import sta_infos

KEYS = ["UserId", "SubjectId_level3_str", "QuestionId"]


def load_nips_data(primary_data_path, meta_data_dir, task_name):
    """
    Load and merge the EEDI NeurIPS data (task_1_2),
    with debug prints to catch issues.
    """
    print("[EEDI] --- Entering load_nips_data ---")
    print(f"[EEDI] Received primary_data_path = {primary_data_path}")
    print(f"[EEDI] Received meta_data_dir     = {meta_data_dir}")
    print(f"[EEDI] Received task_name         = {task_name}")

    # Build paths
    answer_metadata_path = os.path.join(meta_data_dir, f"answer_metadata_{task_name}.csv")
    question_metadata_path = os.path.join(meta_data_dir, f"question_metadata_{task_name}.csv")
    subject_metadata_path = os.path.join(meta_data_dir, "subject_metadata.csv")

    print(f"[EEDI] Expecting primary_data_path      = {primary_data_path}")
    print(f"[EEDI] Expecting answer_metadata_path   = {answer_metadata_path}")
    print(f"[EEDI] Expecting question_metadata_path = {question_metadata_path}")
    print(f"[EEDI] Expecting subject_metadata_path  = {subject_metadata_path}")

    # File checks
    for p in [primary_data_path, answer_metadata_path, question_metadata_path, subject_metadata_path]:
        print(f"[EEDI] Exists? {p} -> {os.path.exists(p)}")

    # If a file is missing, show it clearly and exit
    missing = [p for p in [primary_data_path, answer_metadata_path, question_metadata_path, subject_metadata_path]
               if not os.path.exists(p)]
    if missing:
        print("[EEDI][ERROR] The following files are missing:")
        for p in missing:
            print("   -", p)
        raise FileNotFoundError("Some required EEDI files are missing. See prints above.")

    print("[EEDI] Reading primary data...")
    df_primary = pd.read_csv(primary_data_path)
    print(f"[EEDI] len(df_primary) = {len(df_primary)}")
    print(f"[EEDI] df_primary columns: {list(df_primary.columns)}")

    print("[EEDI] Reading answer metadata...")
    df_answer = pd.read_csv(answer_metadata_path)
    print(f"[EEDI] len(df_answer) = {len(df_answer)}")
    print(f"[EEDI] df_answer columns: {list(df_answer.columns)}")

    if "DateAnswered" not in df_answer.columns:
        raise KeyError("[EEDI][ERROR] 'DateAnswered' column not found in answer metadata.")

    df_answer["answer_timestamp"] = df_answer["DateAnswered"].apply(change2timestamp)

    print("[EEDI] Reading question metadata...")
    df_question = pd.read_csv(question_metadata_path)
    print(f"[EEDI] len(df_question) = {len(df_question)}")
    print(f"[EEDI] df_question columns: {list(df_question.columns)}")

    print("[EEDI] Reading subject metadata...")
    df_subject = pd.read_csv(subject_metadata_path)
    print(f"[EEDI] len(df_subject) = {len(df_subject)}")
    print(f"[EEDI] df_subject columns: {list(df_subject.columns)}")

    if "SubjectId" not in df_subject.columns or "Level" not in df_subject.columns:
        raise KeyError("[EEDI][ERROR] 'SubjectId' or 'Level' column missing in subject_metadata.csv")

    # only keep level 3 subjects
    keep_subject_ids = set(df_subject[df_subject["Level"] == 3]["SubjectId"])
    print(f"[EEDI] Number of level-3 SubjectIds to keep = {len(keep_subject_ids)}")

    if "SubjectId" not in df_question.columns:
        raise KeyError("[EEDI][ERROR] 'SubjectId' column missing in question_metadata.")

    # SubjectId is a string representing a list, e.g., "[1, 2, 3]"
    def _filter_level3(subject_str):
        try:
            s = set(eval(subject_str))
            return s & keep_subject_ids
        except Exception as e:
            print(f"[EEDI][WARN] Could not parse SubjectId='{subject_str}': {e}")
            return set()

    df_question["SubjectId_level3"] = df_question["SubjectId"].apply(_filter_level3)

    # merge data
    print("[EEDI] Merging primary with answer metadata...")
    if "AnswerId" not in df_primary.columns or "AnswerId" not in df_answer.columns:
        raise KeyError("[EEDI][ERROR] 'AnswerId' column missing in primary or answer metadata.")

    df_merge = df_primary.merge(
        df_answer[["AnswerId", "answer_timestamp"]], how="left"
    )

    print("[EEDI] Merging with question metadata...")
    if "QuestionId" not in df_primary.columns or "QuestionId" not in df_question.columns:
        raise KeyError("[EEDI][ERROR] 'QuestionId' column missing in primary or question metadata.")

    df_merge = df_merge.merge(
        df_question[["QuestionId", "SubjectId_level3"]], how="left"
    )

    print("[EEDI] Creating 'SubjectId_level3_str' column...")
    df_merge["SubjectId_level3_str"] = df_merge["SubjectId_level3"].apply(
        lambda x: "_".join([str(i) for i in x]) if isinstance(x, (set, list)) else ""
    )

    print(f"[EEDI] len(df_merge) = {len(df_merge)}")
    print("[EEDI] df_merge columns:", list(df_merge.columns))

    print(f"[EEDI] Num of students  = {df_merge['UserId'].nunique()}")
    print(f"[EEDI] Num of questions = {df_merge['QuestionId'].nunique()}")

    kcs = []
    for item in df_merge["SubjectId_level3"].values:
        if isinstance(item, (set, list)):
            kcs.extend(item)
    print(f"[EEDI] Num of level-3 knowledge concepts = {len(set(kcs))}")

    print("[EEDI] --- Leaving load_nips_data ---")
    return df_merge


def get_user_inters(df):
    """
    Convert the merged dataframe into user sequences.
    """
    user_inters = []
    for user, group in df.groupby("UserId", sort=False):
        group = group.sort_values(["answer_timestamp", "tmp_index"], ascending=True)

        seq_skills = group["SubjectId_level3_str"].tolist()
        seq_ans = group["IsCorrect"].tolist()
        seq_response_cost = ["NA"]
        seq_start_time = group["answer_timestamp"].tolist()
        seq_problems = group["QuestionId"].tolist()
        seq_len = len(group)

        user_inters.append(
            [
                [str(user), str(seq_len)],
                format_list2str(seq_problems),
                format_list2str(seq_skills),
                format_list2str(seq_ans),
                format_list2str(seq_start_time),
                format_list2str(seq_response_cost),
            ]
        )
    return user_inters


def read_data_from_csv(primary_data_path, meta_data_dir, task_name, write_file):
    stares = []

    print("[EEDI] --- Entering read_data_from_csv ---")
    df = load_nips_data(primary_data_path, meta_data_dir, task_name)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(
        f"[EEDI] original interaction num: {ins}, user num: {us}, "
        f"question num: {qs}, concept num: {cs}, "
        f"avg(ins) per user: {avgins:.2f}, avg(concepts) per question: {avgcq:.2f}, na: {na}"
    )

    df["tmp_index"] = range(len(df))

    df = df.dropna(
        subset=[
            "UserId",
            "answer_timestamp",
            "SubjectId_level3_str",
            "IsCorrect",
            "QuestionId",
        ]
    )

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(
        f"[EEDI] after drop interaction num: {ins}, user num: {us}, "
        f"question num: {qs}, concept num: {cs}, "
        f"avg(ins) per user: {avgins:.2f}, avg(concepts) per question: {avgcq:.2f}, na: {na}"
    )

    user_inters = get_user_inters(df)

    print(f"[EEDI] Writing sequences to: {write_file}")
    write_txt(write_file, user_inters)

    print("[EEDI] Done.")
    print("[EEDI] Stats log:")
    for line in stares:
        print("  ", line)

    print("[EEDI] --- Leaving read_data_from_csv ---")


def main():
    # IMPORTANT: adapt this name if your train file is named differently
    primary_data_path = os.path.join("eedi", "raw", "train_task_1_2.csv")
    meta_data_dir = os.path.join("eedi", "raw")
    task_name = "task_1_2"
    out_path = os.path.join("eedi", "processed", "eedi_task_1_2_sequences.txt")
    log_dir = os.path.join("eedi", "logs")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "eedi_task_1_2_preprocess.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("[EEDI] === Starting preprocessing task_1_2 ===")
    print(f"[EEDI] Working directory: {os.getcwd()}")
    print(f"[EEDI] Listing eedi/raw:")
    raw_dir = os.path.join("eedi", "raw")
    if os.path.exists(raw_dir):
        print(os.listdir(raw_dir))
    else:
        print("[EEDI][ERROR] eedi/raw does not exist!")

    print(f"[EEDI] primary_data_path = {primary_data_path}")
    print(f"[EEDI] meta_data_dir     = {meta_data_dir}")
    print(f"[EEDI] output file       = {out_path}")
    print(f"[EEDI] log file          = {log_file}")

    read_data_from_csv(primary_data_path, meta_data_dir, task_name, out_path)

    print("[EEDI] === Finished preprocessing task_1_2 ===")


if __name__ == "__main__":
    main()
