## repository overview

this repository contains the code used for **kt-dusl**, a **domain-informed and uncertainty-aware knowledge tracing model**.

---

## repository structure

├── data/
│ ├── eedi/
│ │ ├── raw/
│ │ └── processed/
│ └── junyi/
│ ├── raw/
│ └── processed/
│
├── scripts/
└── README.md

---

## scripts directory

all experiment-related code is located in the `scripts/` directory.

### preprocessing

- `preprocess_junyi_domain.py`  
  preprocesses the junyi dataset and builds the domain hierarchy.

- `preprocess_mono_eedi.py`  
  creates a mono-concept version of the eedi dataset by keeping a single leaf
  concept per question.

---

### dataset statistics

- `stats_eedi.py`  
  computes descriptive statistics for the eedi dataset.

- `stats_junyi.py`  
  computes descriptive statistics for the junyi dataset.

---

### next-question correctness evaluation

scripts to run online next-question correctness prediction:

- `evaluate_next_correctness_junyi.py`
- `evaluate_next_correctness_eedi_multi.py`
- `evaluate_next_correctness_eedi_mono.py`

these scripts evaluate kt-a, kt-usl, kt-da, and kt-dusl and produce
per-interaction prediction files.

---

### global metrics

scripts computing global performance metrics (auc, acc@0.5, logloss):

- `compute_global_metrics_eedi_both.py`
- `compute_global_metrics_junyi.py`

---

### metrics by evidence

- `metrics_by_evidence_user_concept.py`  

analyzes performance as a function of the amount of evidence, defined as the
number of past interactions per (learner, concept) pair.

---

### impact on the learner model

- `analyze_mastery_uncertainty_deltas.py`  

analyzes the impact of domain-informed propagation on knowledge mastery
and epistemic uncertainty.

---

### uncertainty-based analysis

- `plot_delta_auc_vs_uncertainty.py`  

produces figures showing auc as a function of uncertainty.

---

## raw data

### eedi (task 1 & 2)
https://www.kaggle.com/datasets/alejopaullier/eedi-external-dataset 
located in `data/eedi/raw/`:

- `answer_metadata_task_1_2.csv`
- `question_metadata_task_1_2.csv`
- `student_metadata_task_1_2.csv`
- `subject_metadata.csv`
- `train_task_1_2.csv`

---

### junyi 
https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy 
located in `data/junyi/raw/`:

- `Info_Content.csv`
- `Info_UserData.csv`
- `Log_Problem.csv`
