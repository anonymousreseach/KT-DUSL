# ktusl/training/trainer.py
from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
from sklearn.metrics import roc_curve

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)

try:
    from tqdm.auto import tqdm  # type: ignore

    def _progress(iterable, enable: bool, **kwargs):
        return tqdm(iterable, **kwargs) if enable else iterable
except Exception:  # pragma: no cover
    def _progress(iterable, enable: bool, **kwargs):
        return iterable


# ---------------- Multi-concept aggregators ----------------
def combine_concepts(p_list, w_list=None, mode: str = "mean") -> float:
    """
    Aggregates multiple concept-level probabilities into a single
    question-level probability.
    Supported modes: {mean, noisy_or, noisy_and, logit_linear}.
    """
    arr = np.asarray(list(p_list), dtype=float)
    if arr.size == 0:
        return 0.5

    mode = (mode or "mean").lower()

    if mode == "mean":
        if w_list is None:
            return float(arr.mean())
        w = np.asarray(list(w_list), dtype=float)
        s = w.sum()
        w = w / s if s > 0 else np.ones_like(arr) / len(arr)
        return float((w * arr).sum())

    if mode == "noisy_or":
        arr = np.clip(arr, 1e-9, 1 - 1e-9)
        return float(1.0 - np.prod(1.0 - arr))

    if mode == "noisy_and":
        arr = np.clip(arr, 1e-9, 1 - 1e-9)
        return float(np.prod(arr))

    if mode == "logit_linear":
        p = np.clip(arr, 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p))
        if w_list is None:
            w = np.ones_like(logit) / len(logit)
        else:
            w = np.asarray(list(w_list), dtype=float)
            s = w.sum()
            w = (w / s) if s > 0 else (np.ones_like(logit) / len(logit))
        z = float((w * logit).sum())
        return float(1.0 / (1.0 + np.exp(-z)))

    raise ValueError(f"Unknown combine mode: {mode}")


# ---------------- Time helpers (timestamps in seconds) ----------------
def _ts_to_seconds(ts) -> Optional[float]:
    if ts is None:
        return None
    try:
        return float(pd.Timestamp(ts).timestamp())
    except Exception:
        return None


def _maybe_decay(model, user: int, concepts: List[int], tsec: Optional[float]):
    """
    Calls model.decay_to(user, concept, tsec) if the method exists.
    Otherwise, does nothing.
    This ensures compatibility with BKT, PFA, KTUSL, DKT, SAKT, and UKT.
    """
    decay_fn = getattr(model, "decay_to", None)
    if decay_fn is None or tsec is None:
        return
    for c in concepts:
        decay_fn(user, c, tsec)


# ---------------- Helpers for trace preprocessing ----------------
def _ensure_subject_list_column(df: pd.DataFrame, col: str = "SubjectId") -> pd.DataFrame:
    """
    Ensures that df[col] contains a list of integers for each row.
    If the value is already a list or tuple, it is kept as is.
    If it is a scalar, it is wrapped into a single-element list.
    """
    def _to_list(x):
        if isinstance(x, (list, tuple)):
            return [int(c) for c in x]
        if pd.isna(x):
            return []
        return [int(x)]

    df = df.copy()
    df[col] = df[col].apply(_to_list)
    return df


def _ensure_sorted_by_user_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts interactions by (UserId, DateAnswered) when available,
otherwise by (UserId, QuestionId).
    """
    if "DateAnswered" in df.columns:
        return df.sort_values(["UserId", "DateAnswered", "QuestionId"], kind="mergesort")
    return df.sort_values(["UserId", "QuestionId"], kind="mergesort")


def _first_attempt_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only the first attempt per (UserId, QuestionId).
    """
    if "DateAnswered" in df.columns:
        df = df.sort_values(["UserId", "QuestionId", "DateAnswered"], kind="mergesort")
    else:
        df = df.sort_values(["UserId", "QuestionId"], kind="mergesort")
    return df.drop_duplicates(subset=["UserId", "QuestionId"], keep="first")


def _temporal_user_split(df: pd.DataFrame, test_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-user temporal split: the first (1 - test_frac) interactions
    are assigned to the training set, and the remaining interactions
    to the test set, while preserving temporal order.
    """
    test_frac = float(test_frac)
    test_frac = min(max(test_frac, 0.0), 0.9)

    train_parts = []
    test_parts = []

    for _, g in df.groupby("UserId", sort=False):
        g = _ensure_sorted_by_user_time(g)
        n = len(g)
        if n <= 1:
            train_parts.append(g)
            continue
        split_idx = int(round((1.0 - test_frac) * n))
        split_idx = min(max(split_idx, 1), n - 1)
        train_parts.append(g.iloc[:split_idx])
        test_parts.append(g.iloc[split_idx:])

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).reset_index(drop=True) if test_parts else train_df.iloc[0:0].copy()
    return train_df, test_df


# ---------------- Generic evaluation ----------------
def evaluate_model(
    traces: pd.DataFrame,
    subjects: Optional[pd.DataFrame],
    model,
    test_frac: float = 0.2,
    level: Optional[int] = None,
    first_only: bool = True,
    combine_mode: str = "mean",
    show_progress: bool = True,
    threshold: float = 0.5,
):
    """
    Evaluate a KT model on a pre-built `traces` DataFrame coming from
    preprocessed data.

    Expected columns in traces:
      - UserId
      - QuestionId
      - SubjectId: list of ints (concepts) or convertible scalar
      - IsCorrect: 0/1
      - DateAnswered (optional): timestamp or integer
    """

    # Column normalization
    df = _ensure_subject_list_column(traces, "SubjectId")
    df = _ensure_sorted_by_user_time(df)
    if first_only:
        df = _first_attempt_only(df)

    # Per-user temporal split
    train, test = _temporal_user_split(df, test_frac=test_frac)

    # Reset model state before training
    if hasattr(model, "reset_state"):
        model.reset_state()

    # ------------------------- TRAIN: update-only -------------------------
    train_iter = _progress(
        train.itertuples(index=False),
        enable=show_progress,
        desc=f"Train (update-only) — {len(train):,} rows",
        total=len(train),
        unit="row",
        mininterval=0.5,
        smoothing=0.1,
    )

    for row in train_iter:
        u = row.UserId
        y = int(row.IsCorrect)
        cids = list(row.SubjectId) if isinstance(row.SubjectId, (list, tuple)) else []
        if len(cids) == 0:
            continue
        tsec = _ts_to_seconds(getattr(row, "DateAnswered", None))
        _maybe_decay(model, u, cids, tsec)
        # Default uniform weights (question_concept_weights unavailable here)
        w = [1.0 / len(cids)] * len(cids)
        model.update(u, cids, y, w)

    # ------------------------- TEST: predict → update -----------------------
    y_true: List[int] = []
    y_prob: List[float] = []
    recs: List[Dict[str, Any]] = []

    test_iter = _progress(
        test.itertuples(index=False),
        enable=show_progress,
        desc=f"Test (predict→update) — {len(test):,} rows",
        total=len(test),
        unit="row",
        mininterval=0.5,
        smoothing=0.1,
    )

    for row in test_iter:
        u = row.UserId
        q = row.QuestionId
        y = int(row.IsCorrect)
        cids = list(row.SubjectId) if isinstance(row.SubjectId, (list, tuple)) else []
        if len(cids) == 0:
            continue

        tsec = _ts_to_seconds(getattr(row, "DateAnswered", None))
        _maybe_decay(model, u, cids, tsec)

        # Concept-wise prediction, then aggregation
        p_list = [float(model.predict_concept_proba(u, c)) for c in cids]
        w = [1.0 / len(cids)] * len(cids)
        p_hat = combine_concepts(p_list, w_list=w, mode=combine_mode)

        y_true.append(y)
        y_prob.append(p_hat)
        recs.append(
            {
                "UserId": u,
                "QuestionId": q,
                "IsCorrect": y,
                "y_prob": p_hat,
                "n_concepts": len(cids),
            }
        )

        # Update after seeing the true answer
        model.update(u, cids, y, w)

    preds = pd.DataFrame(recs)

    if len(y_true) == 0:
        metrics = {
            "n_samples": 0,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
        }
        roc_df = pd.DataFrame({"threshold": [], "fpr": [], "tpr": []})
        return metrics, preds, roc_df

    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred_arr = (y_prob_arr >= float(threshold)).astype(int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="binary", zero_division=0
    )
    try:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except Exception:
        auc = float("nan")

    metrics = {
        "n_samples": int(len(y_true_arr)),
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": auc,
    }

    # ROC points
    try:
        fpr, tpr, thr = roc_curve(y_true_arr, y_prob_arr)
        roc_df = pd.DataFrame({"threshold": thr, "fpr": fpr, "tpr": tpr})
    except Exception:
        roc_df = pd.DataFrame({"threshold": [], "fpr": [], "tpr": []})

    return metrics, preds, roc_df
