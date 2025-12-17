# ktusl/data_processing/assist2015/preprocess_assist2015.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import logging
import pandas as pd

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../assist2015
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                # .../data_processing
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.io_utils import write_txt, format_list2str

# ------------------------------------------------------------
# Main preprocessing
# ------------------------------------------------------------
def read_data_from_csv(raw_file: str, out_file: str):
    print(">>> preprocess_assist2015.py is running")
    print(f"[INFO] Raw file: {raw_file}")
    print(f"[INFO] Out file: {out_file}")

    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Raw file not found: {raw_file}")

    df = pd.read_csv(raw_file)

    # Expected columns (based on your example)
    required = ["user_id", "log_id", "sequence_id", "correct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}. Available columns: {list(df.columns)}"
        )

    # Nettoyage minimal
    df = df.dropna(subset=required).copy()
    df["correct"] = df["correct"].astype(int)
    df = df[df["correct"].isin([0, 1])]

    # log_id / sequence_id should be numeric (otherwise keep as str)
    for col in ["log_id", "sequence_id"]:
        try:
            df[col] = df[col].astype(int)
        except Exception:
            df[col] = df[col].astype(str)

    # Stats rapides
    print(f"#Interactions : {len(df):,}")
    print(f"#Students     : {df['user_id'].nunique():,}")
    print(f"#Concepts(KC) : {df['sequence_id'].nunique():,}")

    # On suppose que log_id est ordonnable (proxy temporel)
    df["tmp_index"] = range(len(df))

    # --------------------------------------------------------
    # Sequences = EVERYTHING the learner did (per user)
    # --------------------------------------------------------
    user_inters = []
    for user, g in df.groupby("user_id", sort=False):
        g = g.sort_values(["log_id", "tmp_index"], ascending=True)

        # Dans ton pipeline : colonne "questions" attendue.
        # Ici, on met log_id comme identifiant d'interaction (proxy question).
        seq_questions = g["log_id"].astype(str).tolist()
        # sequence_id = skill / concept (1 concept par interaction dans ce format)
        seq_concepts  = g["sequence_id"].astype(str).tolist()
        seq_correct   = g["correct"].astype(int).astype(str).tolist()
        # timestamps: reuse log_id as a proxy (or tmp_index)
        seq_time      = g["log_id"].astype(str).tolist()
        seq_rt        = ["NA"] * len(g)

        seq_len = len(seq_correct)

        user_inters.append(
            [
                [str(user), str(seq_len)],
                format_list2str(seq_questions),
                format_list2str(seq_concepts),
                format_list2str(seq_correct),
                format_list2str(seq_time),
                format_list2str(seq_rt),
            ]
        )

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    write_txt(out_file, user_inters)

    print(f"[OK] Wrote {len(user_inters):,} user sequences to: {out_file}")


def main():
    # Paths aligned with your directory hierarchy
    raw_path = os.path.join(CURRENT_DIR, "raw", "assist2015_raw.csv")
    out_path = os.path.join(CURRENT_DIR, "processed", "assist2015_sequences.txt")
    log_dir  = os.path.join(CURRENT_DIR, "logs")

    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(log_dir, "assist2015_preprocess.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    read_data_from_csv(raw_path, out_path)


if __name__ == "__main__":
    main()
