import matplotlib.pyplot as plt
import numpy as np

def plot_reliability(y_true, y_prob, n_bins: int = 10, title: str = "Calibration"):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    bin_acc, bin_conf = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins-1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.any():
            bin_acc.append(y_true[mask].mean())
            bin_conf.append(y_prob[mask].mean())
        else:
            bin_acc.append(np.nan); bin_conf.append(np.nan)

    plt.figure(figsize=(5.5,4.5))
    plt.plot([0,1],[0,1], linestyle="--")
    plt.scatter(bin_conf, bin_acc)
    plt.xlabel("Confiance moyenne (p̂)")
    plt.ylabel("Exactitude moyenne")
    plt.title(title)
    plt.tight_layout()
    return plt.gca()
