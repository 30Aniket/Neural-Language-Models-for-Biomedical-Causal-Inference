# Neural Language Models for Biomedical Causal Inference

MSc Computer Science dissertation, University of Warwick.

This project extends the InferBERT-style causal-inference framework for pharmacovigilance
by adding domain-specific encoder models to the existing comparison suite, and evaluates
all models on two FAERS-derived datasets.

**Supervisor:** Dr Gabriele Pergola
**Base work:** [`hsdslab/biomedical-causal-inference`](https://github.com/hsdslab/biomedical-causal-inference)
(Kiss et al.), which itself builds on
[InferBERT](https://github.com/XingqiaoWang/DeepCausalPV-master) (Wang et al., 2021).

---

## Overview

| | |
|---|---|
| **Datasets** | Analgesics-induced Acute Liver Failure (AILF), Tramadol-related Mortalities (TRAM) |
| **Source** | FDA Adverse Event Reporting System (FAERS) |
| **Models** | XGBoost (baseline), ALBERT, BioBERT, PubMedBERT |
| **Protocol** | 5-fold / 20-run cross-validation, isotonic calibration, per-term Z-score causal attribution |

PubMedBERT is added by this work; XGBoost, ALBERT and BioBERT reproduce the base repository.

---

## Repository structure

```
.
├── data/                 preprocessing script + prepared CSVs (raw exports not included)
├── dat/                  pipeline-ready layout: <dataset>/proc/{df_together,split}.csv
│                         and per-model prediction CSVs (cross_val_*)
├── src/
│   ├── <dataset>/        training scripts, one pair per model
│   │                     <model>_train.py      driver: CV loop, logging, artefact saving
│   │                     <model>_train_src.py  helpers: splits, training, calibration
│   └── run_evaluation.py full evaluation pipeline (all four stages)
├── slurm/                SLURM job scripts for the GPU cluster + cluster guide
├── env/                  environment setup, pinned requirements, model prefetch
├── transfer/             upload/download helpers for the cluster
├── eda/                  exploratory data analysis
└── evaluation/           computed metrics, significance tests, Z-score graphs
```

---

## Data availability

The raw FAERS exports (~460 MB) are **not** included in this repository: they exceed
GitHub's 100 MB per-file limit, and are better obtained from source than redistributed.

FAERS data is publicly available from the
[FDA](https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers/fda-adverse-event-reporting-system-faers-public-dashboard).
Cohort definitions for both datasets follow the base paper.

The **prepared** files needed to reproduce every result here are included:

- `dat/<dataset>/proc/df_together.csv` — preprocessed records with clinical term columns
- `dat/<dataset>/proc/split.csv` — fold assignments (0–4) per record

Preprocessing is reproduced by `data/prepare_data.py`.

---

## Reproducing the results

### 1. Environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r env/requirements_albert.txt   # transformer models
pip install -r env/requirements_xgb.txt      # XGBoost baseline
```

The transformer stack is version-pinned (`transformers==4.41.2`) because the training
code from the base repository uses an API that later releases changed.

### 2. Training

**XGBoost** runs on CPU:

```bash
cd src/Analgesics-induced_acute_liver_failure
python xgb_train.py --n_trials 30
```

**Transformer models** need a GPU. On a SLURM cluster:

```bash
bash env/setup_env.sh          # once
bash env/prefetch_models.sh    # once: warm the model cache
bash submit_all.sh <email> albert       # also: biobert, pubmedbert
```

See `slurm/README_CLUSTER.md` for the full cluster workflow. To run locally instead,
execute `src/<dataset>/<model>_train.py` directly.

Each run writes prediction CSVs to `dat/<dataset>/proc/cross_val_<model>*/`, plus
weights, training curves and logs under `outputs/` (git-ignored).

### 3. Evaluation

```bash
python src/run_evaluation.py --root . --compare /path/to/base/repo
```

This reproduces all four analysis stages from the base repository:

| Stage | Produces |
|---|---|
| `metrics` | AUC, precision, recall, F1, accuracy, 10-bin ECE per split and model |
| `zscores` | per-term Z-scores plus PRR / ROR / EBGM disproportionality baselines |
| `ttest` | paired one-sided t-tests between all models, with p-value heatmaps |
| `trees` | hierarchical Z-score pyramid graphs |

Run stages selectively with `--stages metrics ttest`.

---

## Notes on methodology

**Fold independence.** The training loop in the base repository instantiates each
transformer once and reuses it across all 20 cross-validation runs, so from the second
run onward a model has already trained on data it is later evaluated on. XGBoost cannot
carry state this way — a fresh model is fitted per run — which makes it a useful control.

The evaluation therefore reports two views: the 20-run medians, for comparability with
the published tables, and the first run alone, in which every model starts from its
pretrained checkpoint. Both are written to `evaluation/`. This is discussed in the
dissertation.

**Cased vs uncased.** The pipeline lowercases input text. This matches PubMedBERT
(uncased) but not BioBERT (cased); the behaviour is preserved from the base repository
so that results remain comparable, and is noted in the relevant scripts.

---

## Acknowledgements

Computing facilities provided by the Scientific Computing Research Technology Platform,
University of Warwick. Thanks to Dr Gabriele Pergola for supervision and to Csaba Kiss
for clarifications on the original implementation.

## References

- Wang X. et al. (2021). *InferBERT: A Transformer-Based Causal Inference Framework for
  Enhancing Pharmacovigilance.* Frontiers in Artificial Intelligence.
- Lee J. et al. (2020). *BioBERT: a pre-trained biomedical language representation model.*
  Bioinformatics.
- Gu Y. et al. (2021). *Domain-Specific Language Model Pretraining for Biomedical NLP.*
  ACM Transactions on Computing for Healthcare.
- Lan Z. et al. (2020). *ALBERT: A Lite BERT for Self-supervised Learning of Language
  Representations.* ICLR.
