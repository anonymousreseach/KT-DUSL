# ktusl/experiments/run.py
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from ..training.trainer import evaluate_model

# Models
from ..models.ktusl import KTUSL
from ..models.bkt import BKT
from ..models.pfa import PFA
from ..models.ukt import UKT
from ..models.sakt import SAKT

# Optional DKT (if available)
try:
    from ..models.dkt import DKT
    _HAS_DKT = True
except Exception:
    _HAS_DKT = False


# -------------------------------------------------------------------
# General helpers
# -------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file '{path}' is empty or invalid (top-level YAML must be a mapping).")
    return cfg


def _ensure_dir(path: str | None):
    if not path:
        return
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _format_path(template: str | None, dataset: str, model: str) -> str | None:
    if template is None:
        return None
    return template.format(dataset=dataset, model=model)


def _append_metrics_csv(path_csv: str, row: dict):
    # Classification-oriented schema
    SCHEMA = [
        "timestamp","config","model","params_json","dataset",
        "traces_csv","subjects_csv","test_frac","level","first_attempt_only",
        "combine_mode","threshold",
        "accuracy","precision","recall","f1","auc","n_samples",
    ]
    _ensure_dir(path_csv)
    if os.path.exists(path_csv) and os.path.getsize(path_csv) > 0:
        header_cols = list(pd.read_csv(path_csv, nrows=0).columns)
        df = pd.DataFrame([row]).reindex(columns=header_cols)
        df.to_csv(path_csv, mode="a", index=False, header=False)
    else:
        df = pd.DataFrame([row])
        for c in SCHEMA:
            if c not in df.columns:
                df[c] = np.nan
        df = df.reindex(columns=SCHEMA)
        df.to_csv(path_csv, mode="w", index=False, header=True)


def _flatten_concepts_from_traces(traces: pd.DataFrame) -> list[int]:
    all_lists = traces["SubjectId"].tolist()
    return sorted({int(c) for lst in all_lists for c in lst})


# -------------------------------------------------------------------
# Load sequence files (.txt) → flat DataFrame
# -------------------------------------------------------------------

# Base path = ktusl/ folder
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_PROC_DIR = os.path.join(_BASE_DIR, "data_processing")

# Adjust paths if your filenames differ
SEQ_FILES = {
    "assist2012": os.path.join(_DATA_PROC_DIR, "assist2012", "processed", "assist2012_sequences.txt"),
    "assist2015": os.path.join(_DATA_PROC_DIR, "assist2015", "processed", "assist2015_sequences.txt"),
    "junyi":      os.path.join(_DATA_PROC_DIR, "junyi", "processed", "junyi_sequences.txt"),
    "eedi":       os.path.join(_DATA_PROC_DIR, "eedi", "processed", "eedi_task_1_2_sequences.txt"),
}


def _parse_int_list(s) -> list[int]:
    """
    Parse a string into a list of integers.
    Ignore empty or non-numeric tokens.
    """
    if pd.isna(s):
        return []
    vals: list[int] = []
    for x in str(s).split():
        x = x.strip()
        if not x or x.upper() == "NA":
            continue
        try:
            vals.append(int(x))
        except ValueError:
            # Ignore non-integer values
            continue
    return vals


def _parse_str_list(s) -> list[str]:
    """
    Parse a string into a list of space-separated tokens.
    Does not convert to int.
    """
    if pd.isna(s):
        return []
    return [x for x in str(s).split() if x.strip() != ""]



def _parse_skill_token(token: str) -> list[int]:
    """Parse '181_439' -> [181, 439]. Single skill '181' -> [181]."""
    if pd.isna(token):
        return []
    parts = str(token).split("_")
    out: list[int] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            # Silently ignore non-numeric debris
            continue
    return out


def load_traces_from_sequences(dataset: str) -> pd.DataFrame:
    """
    Load a sequence file in the custom text format and return a
    "flat" DataFrame with one row per interaction.

    Expected file format (tab-separated '\t'):
      col0: "userId seq_len"   (e.g., "123 57" or "FLy+lvig... 42")
      col1: list of QuestionId (space-separated, string or int)
      col2: list of SubjectId (tokens 'a_b_c' for multi-skills, space-separated)
      col3: list of IsCorrect (0/1, space-separated)
      col4: list of timestamps (int, space-separated)
      col5: (optional) response time / NA (ignored here)

    Output DataFrame columns:
      - UserId        (string)
      - QuestionId    (string or int, depending on dataset)
      - SubjectId     (list[int])
      - IsCorrect     (int 0/1)
      - DateAnswered  (int timestamp)
    """
    dataset = dataset.lower()
    if dataset not in SEQ_FILES:
        raise ValueError(f"No sequence file configured for dataset='{dataset}' in SEQ_FILES.")

    path = SEQ_FILES[dataset]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sequence file for dataset='{dataset}' not found: {path}")

    print(f"  [INFO] Loading sequence file: {path}")
    df_seq = pd.read_csv(path, sep="\t", header=None)

    if df_seq.shape[1] < 5:
        raise ValueError(f"Sequence file {path} has {df_seq.shape[1]} columns, expected at least 5.")

    rows = []
    for _, row in df_seq.iterrows():
        # col0: "userId seq_len"
        parts = str(row[0]).split()
        if len(parts) == 0:
            continue

        # UserId can be alphanumeric (Junyi), keep as string
        user_id = parts[0].strip()
        # seq_len = int(parts[1])  # not needed for flat reconstruction

        # col1: QuestionIds — keep as string (some are non-numeric)
        q_list = _parse_str_list(row[1])

        # col2: tokens de skills, ex: "12_34 45 7_8_9"
        skill_tokens = str(row[2]).split() if not pd.isna(row[2]) else []

        # col3: answers (0/1)
        ans_list = _parse_int_list(row[3])

        # col4: timestamps
        ts_list = _parse_int_list(row[4])

        T = min(len(q_list), len(skill_tokens), len(ans_list), len(ts_list))
        if T == 0:
            continue

        for t in range(T):
            qid = q_list[t]  # string
            skills = _parse_skill_token(skill_tokens[t]) if t < len(skill_tokens) else []
            y = ans_list[t]
            ts = ts_list[t]

            rows.append(
                {
                    "UserId": user_id,          # string
                    "QuestionId": qid,          # string (or int if numeric dataset)
                    "SubjectId": skills,        # list of skill IDs (int)
                    "IsCorrect": int(y),
                    "DateAnswered": int(ts),    # used for temporal sorting
                }
            )

    traces = pd.DataFrame(rows)
    print(
        f"  [INFO] Built flat traces: {len(traces)} interactions, "
        f"{traces['UserId'].nunique()} students, {traces['QuestionId'].nunique()} questions."
    )
    return traces



# -------------------------------------------------------------------
# Model construction
# -------------------------------------------------------------------

def _build_model(name: str, params: dict, traces: pd.DataFrame, subjects: pd.DataFrame | None):
    """
    Construct an untrained model from its name and parameters.
    Deep models are trained in main(), not here.
    """
    name = name.lower()
    if name == "ktusl":
        return KTUSL(**params)
    if name == "bkt":
        return BKT(**params)
    if name == "pfa":
        return PFA(**params)
    if name == "ukt":
        # Concept vocab
        all_lists = traces["SubjectId"].tolist()
        concept_ids = sorted({int(c) for lst in all_lists for c in lst})
        question_ids = traces["QuestionId"].astype(str).unique().tolist()
        return UKT(concept_ids=concept_ids, question_ids=question_ids, **params)
    if name == "sakt":
        concept_ids = _flatten_concepts_from_traces(traces)
        return SAKT(concept_ids=concept_ids, **params)
    if name == "dkt":
        if not _HAS_DKT:
            raise ValueError("DKT unavailable (module or torch missing).")
        concept_ids = _flatten_concepts_from_traces(traces)
        return DKT(concept_ids=concept_ids, **params)
    raise ValueError(f"Unknown model: {name}")


def _split_train_test_by_user(traces: pd.DataFrame, test_frac: float, seed: int = 42):
    """
    Split train / test PER STUDENT.
    Returns train_df, test_df.
    """
    if test_frac <= 0.0 or test_frac >= 1.0:
        return traces.copy(), traces.iloc[0:0].copy()

    users = traces["UserId"].astype(int).unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(users)
    n_users = len(users)
    n_test = int(round(test_frac * n_users))
    test_users = set(users[:n_test])
    train_users = set(users[n_test:])

    train_df = traces[traces["UserId"].astype(int).isin(train_users)].copy()
    test_df  = traces[traces["UserId"].astype(int).isin(test_users)].copy()
    return train_df, test_df


# -------------------------------------------------------------------
# main()
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str, help="YAML path")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    # ----- lecture config -----
    data_cfg = cfg.get("data", {})
    split_cfg = cfg.get("split", {})
    filt_cfg = cfg.get("filter", {})
    eval_cfg = cfg.get("eval", {})
    model_cfg = cfg.get("model", {})

    model_name = model_cfg.get("name", "ktusl").lower()
    params = model_cfg.get("params", {}) or {}

    datasets_list = data_cfg.get("datasets", None)
    if datasets_list is None:
        datasets_list = ["eedi"]

    test_frac = float(split_cfg.get("test_frac", 0.2))
    level = filt_cfg.get("level", 3)  # not really used anymore; kept for compatibility
    level_by_ds = filt_cfg.get("level_by_dataset", {})
    first_only = bool(filt_cfg.get("first_attempt_only", True))
    combine_mode = eval_cfg.get("combine_mode", "mean")
    threshold = float(eval_cfg.get("threshold", 0.5))
    seed = int(eval_cfg.get("seed", 42))

    save_preds_tpl = eval_cfg.get("save_predictions", None)
    save_metrics = eval_cfg.get("save_metrics", "outputs/metrics_log.csv")

    traces_csv = data_cfg.get("traces_csv", None)
    subjects_csv = data_cfg.get("subjects_csv", None)

    # ----- boucle datasets -----
    for dataset in datasets_list:
        dataset = str(dataset).lower()
        print(f"\n=== RUN {model_name} | dataset={dataset} ===")

        eff_level = level_by_ds.get(dataset, level)  # not really used here

        # ---------- load from sequence files ----------
        traces = load_traces_from_sequences(dataset)
        subjects = None   # no separate table anymore

        # skip if empty
        if traces is None or len(traces) == 0:
            print(f"  [WARN] dataset={dataset}: 0 lignes -> skip.")
            rec = {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "config": args.config,
                "model": model_name,
                "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
                "dataset": dataset,
                "traces_csv": f"[seq:{dataset}]",
                "subjects_csv": "[none]",
                "test_frac": test_frac,
                "level": eff_level,
                "first_attempt_only": first_only,
                "combine_mode": combine_mode,
                "threshold": threshold,
                "n_samples": 0,
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "auc": float("nan"),
            }
            _append_metrics_csv(save_metrics, rec)
            print(f"  -> metrics appended (empty dataset): {save_metrics}")
            continue

        # --------- per-student train / test split ---------
        train_traces, test_traces = _split_train_test_by_user(traces, test_frac=test_frac, seed=seed)
        print(f"  train interactions: {len(train_traces)}, test interactions: {len(test_traces)}")

        # build model (untrained)
        model = _build_model(model_name, params, train_traces, subjects)

        # --------- training for deep models ---------
        if model_name in ["dkt", "sakt", "ukt"]:
            print("  [INFO] Training deep model...")
            model.fit(
                train_df=train_traces,
                subjects_df=subjects,
                show_progress=not args.no_progress,
            )
        else:
            print("  [INFO] Model without offline training (online-only baseline).")

        # --------- ONLINE EVALUATION ON THE TEST ---------
        metrics, preds, roc_df = evaluate_model(
            traces=traces, 
            subjects=subjects,
            model=model,
            test_frac=test_frac,          # already split; 0.0 = no re-split
            level=eff_level,
            first_only=first_only,
            combine_mode=combine_mode,
            show_progress=not args.no_progress,
            threshold=threshold,
        )

        # Console output -> goes into the .out log
        print(f"  n_samples={metrics.get('n_samples')}")
        for k in ["accuracy", "precision", "recall", "f1", "auc"]:
            if k in metrics:
                print(f"  {k:>8s}: {metrics[k]}")

        # save predictions
        save_preds = _format_path(save_preds_tpl, dataset=dataset, model=model_name)
        if save_preds:
            _ensure_dir(save_preds)
            preds.to_csv(save_preds, index=False)
            print(f"  -> preds: {save_preds}")

        # metrics row
        rec = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": args.config,
            "model": model_name,
            "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
            "dataset": dataset,
            "traces_csv": str(traces_csv) if traces_csv else f"[seq:{dataset}]",
            "subjects_csv": str(subjects_csv) if subjects_csv else "[none]",
            "test_frac": test_frac,
            "level": eff_level,
            "first_attempt_only": first_only,
            "combine_mode": combine_mode,
            "threshold": threshold,
        }
        rec.update(metrics)
        _append_metrics_csv(save_metrics, rec)
        print(f"  -> metrics appended: {save_metrics}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
