# data_processing/utils/io_utils.py
# -*- coding: utf-8 -*-

import os


def format_list2str(lst, sep=" "):
    """Convert a list [a, b, c] into 'a b c' (all strings)."""
    return sep.join(str(x) for x in lst)


def write_txt(path, data):
    """
    Write sequences into a text file.
    data is a list of "rows", each being a list of columns.
    Columns are separated with tabs.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            # Row may contain a sub-list [user_id, seq_len] + other columns already as str
            flat_cols = []
            for col in row:
                if isinstance(col, (list, tuple)):
                    flat_cols.append(format_list2str(col))
                else:
                    flat_cols.append(str(col))
            f.write("\t".join(flat_cols) + "\n")
