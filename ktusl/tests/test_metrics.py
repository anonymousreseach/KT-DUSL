import numpy as np
from ktusl.training.metrics import all_metrics

def test_metrics_shapes():
    y = np.array([0,1,1,0,1])
    p = np.array([0.2,0.7,0.8,0.4,0.6])
    m = all_metrics(y, p)
    assert set(["accuracy","auroc","nll","brier","ece","n_samples"]).issubset(m.keys())
