"""
Helper module containing functions to set up data splits and train the XGBoost baseline.

This mirrors the repo's albert_train_src.py / biobert_train_src.py structure one
function at a time (get_split -> train -> predict -> calibration) so the XGBoost
baseline sits in exactly the same pipeline as the transformer models.

FIDELITY NOTES
--------------
* get_split()      : identical logic to the transformer version, except it keeps the
                     tabular feature columns instead of "Temp_sentence".
* calibration()    : COPIED VERBATIM from albert_train_src.py (isotonic regression
                     fitted on the dev fold, applied to test).
* The 5-fold / 20-run cross-validation scheme, and the role of the dev fold
  ("hyperparameter tuning, early stopping, and calibrator fitting"), follow
  section 2.2 of the 2026 paper.
* Features are multi-hot encoded directly from the tabular data, per section 2.1
  of the paper ("for XGBoost, features are multi-hot encoded directly from the
  tabular data").
* Hyperparameters are optimised on the validation fold with Optuna, per the
  co-author's note: "the XGBoost version was hyperparameter-optimized on the
  validation set using the Optuna package".

A NOTE ON FOLD INDEPENDENCE
---------------------------
The transformer scripts in the repo reuse a single model object across all 20
splits, so from split 02 onwards the network has already seen data it is later
tested on. XGBoost cannot carry state this way: a fresh booster is fitted for
every split. The XGBoost numbers are therefore fold-independent by construction,
which makes this baseline the cleanest point of comparison in the suite.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

# Columns that are never features
NON_FEATURE_COLS = ["Temp_sentence", "label"]
# Columns holding a single categorical value per report
SINGLE_VALUE_COLS = ["age", "dose", "gender"]


def get_split(split, dataset, project_root=None):
    """
    Identical to the transformer version, except the tabular feature columns are
    retained instead of the generated sentence.
    """
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dat_dir = os.path.join(project_root, "dat")

    df = pd.read_csv(os.path.join(dat_dir, dataset, "proc", "df_together.csv"), index_col=0)
    split_df = pd.read_csv(os.path.join(dat_dir, dataset, "proc", "split.csv"), index_col=0)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    df = df[feature_cols + ["label"]]

    df_train = df.loc[split_df[split_df["SPLIT"].isin(split["train"])].index]
    df_dev = df.loc[split_df[split_df["SPLIT"].isin(split["dev"])].index]
    df_test = df.loc[split_df[split_df["SPLIT"].isin(split["test"])].index]

    return df_train, df_dev, df_test


# ---------------------------------------------------------------------------
# Multi-hot encoding
# ---------------------------------------------------------------------------
def build_vocabulary(df_train, feature_cols):
    """
    Build the multi-hot vocabulary from the TRAINING fold only.

    Fitting the vocabulary on train alone (rather than the full dataset) keeps
    the dev and test folds genuinely unseen: a term that never appears in
    training contributes no column, exactly as it would at deployment time.

    Multi-value columns (psd, ssd, ccd, idrug, indication, outcome/ade) hold
    comma-separated lists and expand to one binary column per distinct term.
    Single-value columns (age, dose, gender) expand to one binary column per level.
    """
    vocab = {}
    for col in feature_cols:
        if col in SINGLE_VALUE_COLS:
            terms = sorted({str(v).strip() for v in df_train[col].dropna()})
        else:
            terms = set()
            for cell in df_train[col].dropna():
                for t in str(cell).split(","):
                    t = t.strip()
                    if t:
                        terms.add(t)
            terms = sorted(terms)
        vocab[col] = {t: i for i, t in enumerate(terms)}
    return vocab


def multi_hot_encode(df, vocab, feature_cols):
    """
    Encode a frame against a fixed vocabulary -> CSR sparse matrix.

    Vectorised with pandas explode/map rather than a per-row loop: on the AILF
    training fold (~22k rows) the row-wise version takes minutes, this takes
    well under a second.
    """
    offsets, total = {}, 0
    for col in feature_cols:
        offsets[col] = total
        total += len(vocab[col])

    n = len(df)
    pos = pd.Series(np.arange(n), index=df.index)   # row index -> matrix row
    rows_all, cols_all = [], []

    for col in feature_cols:
        s = df[col].dropna()
        if s.empty or not vocab[col]:
            continue
        s = s.astype(str).str.strip()
        if col in SINGLE_VALUE_COLS:
            terms = s
        else:
            terms = s.str.split(",").explode().str.strip()
            terms = terms[terms != ""]
        if terms.empty:
            continue
        j = terms.map(vocab[col])
        keep = j.notna()
        if not keep.any():
            continue
        j = j[keep].astype(np.int64).to_numpy() + offsets[col]
        r = pos.loc[terms[keep].index].to_numpy()
        rows_all.append(r)
        cols_all.append(j)

    if rows_all:
        rows = np.concatenate(rows_all)
        cols = np.concatenate(cols_all)
    else:
        rows = np.empty(0, dtype=np.int64)
        cols = np.empty(0, dtype=np.int64)

    data = np.ones(len(rows), dtype=np.float32)
    X = sparse.csr_matrix((data, (rows, cols)), shape=(n, total), dtype=np.float32)
    X.sum_duplicates()
    X.data[:] = 1.0          # multi-hot: presence, not count
    return X


def feature_names(vocab, feature_cols):
    """Column names aligned with multi_hot_encode, e.g. 'psd_Paracetamol'."""
    names = []
    for col in feature_cols:
        inv = {i: t for t, i in vocab[col].items()}
        names.extend(f"{col}_{inv[i]}" for i in range(len(inv)))
    return names


# ---------------------------------------------------------------------------
# Training with Optuna hyperparameter search on the validation fold
# ---------------------------------------------------------------------------
def tune_hyperparameters(X_train, y_train, X_dev, y_dev, n_trials=30, seed=42, verbose=False):
    """
    Optimise hyperparameters on the validation fold, per the co-author's note.

    The objective is dev-set AUC. Early stopping also uses the dev fold, matching
    the paper's description of the validation fold's role.
    """
    import optuna
    import xgboost as xgb

    optuna.logging.set_verbosity(optuna.logging.WARNING if not verbose else optuna.logging.INFO)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    ddev = xgb.DMatrix(X_dev, label=y_dev)

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "seed": seed,
        }
        booster = xgb.train(
            params, dtrain, num_boost_round=1000,
            evals=[(ddev, "dev")], early_stopping_rounds=50, verbose_eval=False,
        )
        trial.set_user_attr("best_iteration", booster.best_iteration)
        return booster.best_score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    best.update({"objective": "binary:logistic", "eval_metric": "auc",
                 "tree_method": "hist", "seed": seed})
    return best, study.best_trial.user_attrs.get("best_iteration", 100), study


def train_xgb(X_train, y_train, X_dev, y_dev, params, num_boost_round=1000, seed=42):
    """Fit the final booster with early stopping on the dev fold."""
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train)
    ddev = xgb.DMatrix(X_dev, label=y_dev)
    evals_result = {}
    booster = xgb.train(
        params, dtrain, num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (ddev, "dev")],
        early_stopping_rounds=50, evals_result=evals_result, verbose_eval=False,
    )
    return booster, evals_result


def predict(booster, X_dev, X_test):
    """Return (dev probabilities, test probabilities) — mirrors the transformer predict()."""
    import xgboost as xgb

    it = getattr(booster, "best_iteration", None)
    rng = (0, it + 1) if it is not None else None
    kw = {"iteration_range": rng} if rng else {}
    p_dev = booster.predict(xgb.DMatrix(X_dev), **kw)
    p_test = booster.predict(xgb.DMatrix(X_test), **kw)
    return p_dev.tolist(), p_test.tolist()


# ---------------------------------------------------------------------------
# VERBATIM from albert_train_src.py / biobert_train_src.py
# ---------------------------------------------------------------------------
def calibration(probabilities_dev, labels_dev, probabilities_test):
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(probabilities_dev, labels_dev)
    probabilities_test_calibrated = ir.transform(probabilities_test)

    return probabilities_test_calibrated
