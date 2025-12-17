# ktusl/scripts/bootstrap_f1_test.py

import numpy as np
import pandas as pd
from pathlib import Path


THRESHOLD = 0.5
N_BOOT = 2000
SEED = 42


def load_preds(path: Path):
    """
    Expected columns:
      UserId, QuestionId, IsCorrect, y_prob, n_concepts
    Returns:
      y_true (0/1), y_pred (0/1)
    """
    df = pd.read_csv(path)

    required = {"IsCorrect", "y_prob"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing columns {required}. Found: {list(df.columns)}")

    y_true = df["IsCorrect"].astype(int).to_numpy()
    y_pred = (df["y_prob"].to_numpy() >= THRESHOLD).astype(int)

    return y_true, y_pred


def f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    F1 for positive class (label=1).
    (Equivalent to sklearn f1_score(y_true, y_pred, pos_label=1) for binary data.)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    denom = (2 * tp + fp + fn)
    return float((2 * tp) / denom) if denom > 0 else 0.0


def paired_bootstrap_f1(y_true, pred_a, pred_b, n_boot=N_BOOT, seed=SEED):
    """
    Paired bootstrap on ΔF1 = F1(B) - F1(A).
    Here we will call:
      A = BKT, B = UKT-SL  -> delta = F1(UKT-SL) - F1(BKT)

    Returns:
      delta_obs, (ci_low, ci_high), p_value, deltas
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    f1_a = f1_binary(y_true, pred_a)
    f1_b = f1_binary(y_true, pred_b)
    delta_obs = f1_b - f1_a

    deltas = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = f1_binary(y_true[idx], pred_b[idx]) - f1_binary(y_true[idx], pred_a[idx])

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])

    # two-sided bootstrap p-value
    p_value = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))

    return delta_obs, (ci_low, ci_high), p_value, deltas


def run(dataset: str):
    base = Path("outputs")

    y_true, pred_bkt = load_preds(base / f"preds_bkt_{dataset}.csv")
    _,      pred_uktsl = load_preds(base / f"preds_ktusl_{dataset}.csv")  # UKT-SL

    delta, ci, p, _ = paired_bootstrap_f1(y_true, pred_bkt, pred_uktsl)

    print(f"\n=== {dataset.upper()} ===")
    print(f"Δ F1 (UKT-SL − BKT) = {delta:.4f}")
    print(f"IC95% = [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"p-value (bootstrap) = {p:.4g}")

    if ci[0] > 0:
        print("→ UKT-SL significantly better than BKT ✅")
    elif ci[1] < 0:
        print("→ BKT significantly better than UKT-SL ❌")
    else:
        print("→ No significant difference")


if __name__ == "__main__":
    for ds in ["assist2015", "eedi", "junyi"]:
        run(ds)
