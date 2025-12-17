# ktusl/training/metrics.py
from __future__ import annotations
from typing import Sequence, Dict, Any
import numpy as np

# On utilise sklearn si dispo (sinon fallback naïf)
try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )
    _HAS_SK = True
except Exception:
    _HAS_SK = False


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    
    if not _HAS_SK:
        return float("nan")
    classes = np.unique(y_true)
    if len(classes) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def all_metrics(
    y_true_in: Sequence[int],
    y_prob_in: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Calculates the main binary classification metrics
    from probabilities (y_prob) + decision threshold (threshold).
    Returns: accuracy, precision, recall, f1, auc, n_samples, threshold
    """
    y_true = np.asarray(y_true_in, dtype=int)
    y_prob = np.asarray(y_prob_in, dtype=float)

    n = int(len(y_true))
    if n == 0:
        return {
            "threshold": float(threshold),
            "n_samples": 0,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
        }

    y_pred = (y_prob >= float(threshold)).astype(int)

    if _HAS_SK:
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
    else:
        # Fallback simple (sans sklearn)
        acc = float((y_pred == y_true).mean())
        tp = float(((y_pred == 1) & (y_true == 1)).sum())
        fp = float(((y_pred == 1) & (y_true == 0)).sum())
        fn = float(((y_pred == 0) & (y_true == 1)).sum())
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    auc = _safe_auc(y_true, y_prob)

    return {
        "threshold": float(threshold),
        "n_samples": n,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }
