# ktusl/scripts/plot_roc_curves.py
# Usage:
#   cd kt_repo
#   python ktusl/scripts/plot_roc_curves.py
#
# Expects prediction files in: outputs/
# with columns: UserId,QuestionId,IsCorrect,y_prob,n_concepts

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc

# --- Config ---
DATASETS = ["assist2015", "eedi", "junyi"]

MODELS = {
    "ktusl":       {"pattern": "preds_ktusl_{ds}.csv",       "color": "red",    "label": "UKT-SL"},
    "ukt":         {"pattern": "preds_ukt_{ds}.csv",         "color": "green",  "label": "UKT"},
    "pfa":         {"pattern": "preds_pfa_{ds}.csv",         "color": "gray",   "label": "PFA"},
    "bkt":         {"pattern": "preds_bkt_{ds}.csv",         "color": "orange", "label": "BKT"},
}

# <repo_root>/outputs (repo_root inferred from this file location)
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]          # .../kt_repo
BASE_DIR = REPO_ROOT / "outputs"          # .../kt_repo/outputs

# Where to save figures (same folder)
FIG_DIR = BASE_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_preds(csv_path: Path):
    df = pd.read_csv(csv_path)
    needed = {"IsCorrect", "y_prob"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{csv_path} missing {needed}. Found: {list(df.columns)}")
    y_true = df["IsCorrect"].astype(int).to_numpy()
    y_score = df["y_prob"].astype(float).to_numpy()
    return y_true, y_score


def plot_dataset(dataset: str, show: bool = True):
    fig = plt.figure()

    any_curve = False
    for _, cfg in MODELS.items():
        path = BASE_DIR / cfg["pattern"].format(ds=dataset)
        if not path.exists():
            print(f"[WARN] Missing: {path}")
            continue

        y_true, y_score = load_preds(path)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=cfg["color"], label=f'{cfg["label"]} (AUC={roc_auc:.3f})')
        any_curve = True

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {dataset.upper()}")

    if any_curve:
        plt.legend(loc="lower right")
    else:
        print(f"[ERROR] No curves plotted for {dataset} (no files found in {BASE_DIR}).")

    plt.tight_layout()

    # Save figure
    out_png = FIG_DIR / f"roc_{dataset}.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {out_png}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    print(f"[INFO] Using BASE_DIR={BASE_DIR}")
    for ds in DATASETS:
        plot_dataset(ds, show=True)  # set show=False if you don't want windows popping up


if __name__ == "__main__":
    main()
