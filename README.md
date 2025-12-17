# KTUSL Knowledge Tracing Experiments

This repository contains all the utilities I use to benchmark **KTUSL** (Uncertainty Aware Knowledge tracing using Subjective Logic) and several reference models (BKT, PFA, SAKT, UKT, DKVMN, optional DKT). It provides:

- preprocessing scripts that convert raw datasets (Assist/2015, EEDI task 1-2, Junyi) into tab separated interaction sequences,
- an evaluation harness that loads those sequences, splits them per learner, trains the requested model, and logs metrics/predictions,
- analysis scripts to compute bootstrap confidence intervals, draw ROC curves, and describe dataset statistics.
---

## Repository Layout

| Path | Description |
| --- | --- |
| `ktusl/models/` | Implementations of KTUSL, BKT, PFA, SAKT, UKT, DKVMN and optional DKT. |
| `ktusl/training/` | Online evaluation helpers (`trainer.py`) plus metrics/calibration utilities. |
| `ktusl/data_processing/` | Dataset-specific scripts that turn the raw CSV exports into `*_sequences.txt`. |
| `ktusl/experiments/` | The configurable experiment runner (`run.py`) and grid helpers. |
| `ktusl/scripts/` | One-off utilities: ROC plots, bootstrap tests, dataset summaries, etc. |
| `ktusl/config/` | Ready-to-use YAML configs for each model. |
| `outputs/` | Default location for metrics CSVs, prediction dumps, and generated plots. |

---

## Requirements & Installation

1. Create/activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
---

## Preparing the Data

The experiment runner expects tab-separated sequence files with columns:

1. `"user_id seq_len"`
2. question ids (space separated)
3. subject/concept tokens (`"12_37"` for multi-skill questions)
4. correctness flags (`0/1`)
5. timestamps (integer, aligned with the other columns)

Scripts under `ktusl/data_processing/<dataset>/` show how to build these files from the original raw exports. For example:

```bash
python ktusl/data_processing/eedi/preprocess_eedi.py
```

After preprocessing, the sequence files should exist at:

- `ktusl/data_processing/eedi/processed/eedi_task_1_2_sequences.txt`

You can inspect the resulting datasets with:

```bash
python ktusl/scripts/describe_processed_datasets.py
python ktusl/scripts/stats_datasets_summary.py
```

The raw data can be downloaded here 
https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy 
https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data 
https://www.kaggle.com/datasets/alejopaullier/eedi-external-dataset 

---

## Running Experiments

The main entry point is `ktusl/experiments/run.py`. It takes a single YAML config describing the data split, model type, and output locations.

```bash
python -m ktusl.experiments.run ktusl/config/ktusl_multi.yaml
```


## Evaluating Results

Once predictions are available in `outputs/`, you can run:

- `python ktusl/scripts/bootstrap_accuracy_test.py` – paired bootstrap on accuracy (KTUSL vs. BKT).
- `python ktusl/scripts/bootstrap_f1_test.py` – paired bootstrap on F1 score.
- `python ktusl/scripts/plot_roc_curves.py` – generate ROC plots per dataset (saved under `outputs/roc_<dataset>.png`).

These scripts expect files named `preds_<model>_<dataset>.csv` with columns `UserId, QuestionId, IsCorrect, y_prob, n_concepts`.
