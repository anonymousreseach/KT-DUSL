# data_processing/utils/stats_utils.py
# -*- coding: utf-8 -*-

import pandas as pd


def sta_infos(df: pd.DataFrame, keys, stares: list):
    """
    Compute simple statistics on the dataframe.
    keys: [user_col, concept_col, question_col]
    stares: list of strings for logging.
    """
    user_col, concept_col, question_col = keys

    ins = len(df)
    us = df[user_col].nunique(dropna=True)
    cs = df[concept_col].nunique(dropna=True)
    qs = df[question_col].nunique(dropna=True)

    avg_ins_per_user = ins / us if us > 0 else 0.0
    avg_concepts_per_q = cs / qs if qs > 0 else 0.0

    # Number of NAs in the key columns
    na = int(
        df[[user_col, concept_col, question_col]]
        .isna()
        .sum()
        .sum()
    )

    msg = (
        f"ins={ins}, users={us}, questions={qs}, concepts={cs}, "
        f"avg_ins_per_user={avg_ins_per_user:.2f}, "
        f"avg_concepts_per_q={avg_concepts_per_q:.2f}, "
        f"na={na}"
    )
    stares.append(msg)

    return ins, us, qs, cs, avg_ins_per_user, avg_concepts_per_q, na
