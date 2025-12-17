import numpy as np
import pandas as pd
from pathlib import Path


THRESHOLD = 0.5
N_BOOT = 2000
SEED = 42


def load_preds(path):
    """
    Columns expected:
    UserId, QuestionId, IsCorrect, y_prob, n_concepts
    """
    df = pd.read_csv(path)

    required = {"IsCorrect", "y_prob"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing columns {required}")

    y_true = df["IsCorrect"].astype(int).values
    y_pred = (df["y_prob"].values >= THRESHOLD).astype(int)

    return y_true, y_pred


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def paired_bootstrap_accuracy(y_true, pred_a, pred_b, n_boot=N_BOOT, seed=SEED):
    """
    Paired bootstrap on the accuracy difference.
    Returns:
      - delta_obs
      - 95% CI
      - two-sided bootstrap p-value
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    # Observed difference
    acc_a = accuracy(y_true, pred_a)
    acc_b = accuracy(y_true, pred_b)
    delta_obs = acc_b - acc_a   # UKT-SL − BKT

    deltas = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = (
            accuracy(y_true[idx], pred_b[idx]) -
            accuracy(y_true[idx], pred_a[idx])
        )

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])

    # Two-sided bootstrap p-value
    p_value = 2 * min(
        np.mean(deltas <= 0),
        np.mean(deltas >= 0),
    )

    return delta_obs, (ci_low, ci_high), p_value


def run(dataset):
    base = Path("outputs")

    y_true, pred_bkt = load_preds(base / f"preds_bkt_{dataset}.csv")
    _,      pred_ktusl = load_preds(base / f"preds_ktusl_{dataset}.csv")

    delta, ci, p = paired_bootstrap_accuracy(y_true, pred_bkt, pred_ktusl)

    print(f"\n=== {dataset.upper()} ===")
    print(f"Δ Accuracy (UKT-SL − BKT) = {delta:.4f}")
    print(f"95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"Bootstrap p-value = {p:.4g}")

    if ci[0] > 0:
        print("→ UKT-SL significantly better ✅")
    elif ci[1] < 0:
        print("→ BKT significantly better ❌")
    else:
        print("→ Difference not significant")


if __name__ == "__main__":
    for ds in ["assist2015", "eedi", "junyi"]:
        run(ds)
