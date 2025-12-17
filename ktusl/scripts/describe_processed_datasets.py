from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

# BASE_DIR = ktusl/ directory
BASE_DIR = Path(__file__).resolve().parents[1]
DP_DIR = BASE_DIR / "data_processing"

SEQ_FILES = {
    "assist2012": DP_DIR / "assist2012" / "processed" / "assist2012_sequences.txt",
    "eedi":       DP_DIR / "eedi" / "processed" / "eedi_task_1_2_sequences.txt",
    "junyi":      DP_DIR / "junyi" / "processed" / "junyi_sequences.txt",
}

def _len_tokens(x) -> int:
    """Count the number of space-separated tokens in a cell."""
    if pd.isna(x):
        return 0
    s = str(x).strip()
    if not s:
        return 0
    return len(s.split())

def _parse_ints_fast(x) -> np.ndarray:
    """Fast parsing of a '0 1 0 1 ...' column into an int8 array."""
    if pd.isna(x):
        return np.array([], dtype=np.int8)
    s = str(x).strip()
    if not s:
        return np.array([], dtype=np.int8)
    try:
        return np.fromstring(s, sep=" ", dtype=np.int8)
    except Exception:
        return np.array([int(t) for t in s.split() if t.strip().isdigit()], dtype=np.int8)

def describe_dataset(name: str, path: Path, sep: str = "\t"):
    print("\n" + "=" * 72)
    print(f"DATASET: {name}")
    print("=" * 72)

    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return

    # Raw read: 5 columns without headers (same as the current parsing)
    df = pd.read_csv(path, sep=sep, header=None, dtype=str, engine="python")
    # Expected columns (according to your format):
    # 0 = "UserId ..." (sometimes only the user + other tokens)
    # 1 = question sequence
    # 2 = skill sequence (tokens like "12_45 9 3_7")
    # 3 = answer sequence (0/1)
    # 4 = timestamps sequence

    n_rows = len(df)

    # UserId (first token of column 0)
    user_ids = df[0].fillna("").astype(str).str.split().str[0]
    n_users = user_ids.nunique()

    # Lengths per column (in tokens)
    L_q  = df[1].apply(_len_tokens).to_numpy()
    L_s  = df[2].apply(_len_tokens).to_numpy()
    L_a  = df[3].apply(_len_tokens).to_numpy()
    L_ts = df[4].apply(_len_tokens).to_numpy()

    # Effective sequence length per row = min of the 4 lengths (synchronous)
    L = np.minimum.reduce([L_q, L_s, L_a, L_ts])

    # Length statistics
    def q(x, p): return float(np.quantile(x, p)) if len(x) else float("nan")

    print(f"Rows (sequences): {n_rows:,}")
    print(f"Unique students : {n_users:,}")

    print("\nSequence length (T per learner-row) using min(len(q),len(s),len(a),len(ts)):")
    print(f"  Mean   : {L.mean():.2f}")
    print(f"  Median : {np.median(L):.2f}")
    print(f"  Min/Max: {L.min():.0f} / {L.max():.0f}")
    print(f"  Q10/Q25: {q(L,0.10):.0f} / {q(L,0.25):.0f}")
    print(f"  Q75/Q90: {q(L,0.75):.0f} / {q(L,0.90):.0f}")

    # % of short / long sequences
    for thr in [5, 10, 20, 50, 100, 200]:
        pct = 100.0 * float((L >= thr).mean())
        print(f"  % sequences with T >= {thr:<3}: {pct:5.1f}%")

    # Correct/incorrect proportions (over all answers, without flattening rows)
    # Concatenate the answer arrays (fast and memory-safe)
    answers_all = []
    for x in df[3].tolist():
        arr = _parse_ints_fast(x)
        if arr.size:
            answers_all.append(arr)
    if answers_all:
        y = np.concatenate(answers_all)
        p_correct = float(y.mean())
        print("\nCorrect / wrong proportions (over all interactions):")
        print(f"  P(correct) = {p_correct:.4f}")
        print(f"  P(wrong)   = {1.0 - p_correct:.4f}")
        print(f"  #answers   = {len(y):,}")
    else:
        print("\n[WARN] Could not parse answers to compute P(correct).")

    # Number of concepts per interaction (approx):
    # average number of concepts per skill token.
    # Token example "12_45" => 2 concepts; "9" => 1 concept.
    # Sample when massive to keep it fast.
    skill_tokens = df[2].fillna("").astype(str).tolist()
    sample = skill_tokens
    if len(sample) > 5000:
        rng = np.random.default_rng(42)
        sample = list(rng.choice(sample, size=5000, replace=False))

    counts = []
    for seq in sample:
        toks = [t for t in seq.split() if t.strip()]
        for t in toks[:200]:  # per-line cap (avoid extremely long sequences)
            counts.append(1 + t.count("_"))
    if counts:
        print("\nConcept multiplicity (approx, sampled):")
        print(f"  Avg #concepts per interaction ≈ {np.mean(counts):.3f}")
        print(f"  % multi-concept interactions  ≈ {100.0*np.mean(np.array(counts)>1):.1f}%")
    else:
        print("\n[WARN] Could not parse skills for concept multiplicity.")

    print("=" * 72)


def main():
    print(f"[DEBUG] BASE_DIR = {BASE_DIR}")
    print(f"[DEBUG] DP_DIR   = {DP_DIR}")
    for name, path in SEQ_FILES.items():
        describe_dataset(name, path)


if __name__ == "__main__":
    main()
