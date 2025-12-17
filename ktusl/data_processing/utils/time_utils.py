# data_processing/utils/time_utils.py
# -*- coding: utf-8 -*-

import pandas as pd


def change2timestamp(x, hasf=False):
    """
    Convert a datetime string into a timestamp (seconds).
    hasf is ignored but kept for compatibility with existing code.
    """
    if pd.isna(x):
        return None
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return None
    # timestamp in seconds
    return int(ts.value // 10**9)
