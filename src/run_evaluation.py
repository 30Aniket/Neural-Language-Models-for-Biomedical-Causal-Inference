#!/usr/bin/env python3
"""
run_evaluation.py -- the 2026 repo's full evaluation pipeline, as one script.

Reimplements all four notebooks named in the repo README, with every statistical
function copied VERBATIM from the originals:

  stage 1  metrics   <- extra_evaluation_codes/notebooks/calc_metrics.ipynb
                       AUC / precision / recall / F1 / accuracy / 10-bin ECE
  stage 2  zscores   <- <dataset>/calc_zscores.ipynb
                       per-term Z-scores + PRR / ROR / EBGM disproportionality
  stage 3  ttest     <- <dataset>/student_test.ipynb
                       paired one-sided t-tests between models, p-value heatmaps
  stage 4  trees     <- extra_evaluation_codes/notebooks/vis_trees.ipynb
                       hierarchical Z-score pyramid graphs (PDF per model/split)

WHAT IS UNCHANGED
    classification_scores(), calculate_z_scores(), calculate_ppr(),
    calculate_ror(), calculate_ebgm(), calc_student() and the graph-building
    logic are byte-for-byte the repo's. Thresholds (Z>1.64, PPR>=2 & chi2>=4,
    ROR lower CI>1, EBGM lower CI>2, n>=30 per group) are untouched.

WHAT IS ADAPTED (plumbing only)
    * model list is discovered from whatever cross_val_*_temp folders exist,
      instead of being hardcoded to xgb/biobert/albert/llama. That is what lets
      the same code score albert, biobert and pubmedbert.
    * the notebooks' absolute /home/hsdslab/... paths become --root.
    * stage 1 writes all_results.json, which stage 3 requires; the shipped
      calc_metrics.ipynb defines its functions but never runs or saves them.
    * AILF uses an 'outcome' column, TRAM uses 'ade'; handled automatically.

USAGE
    python run_evaluation.py --root ~/biocausal_results                # all stages
    python run_evaluation.py --root ~/biocausal_results --stages metrics ttest
    python run_evaluation.py --root ~/biocausal_results --compare repo2026/
    python run_evaluation.py --root ~/biocausal_results --stages trees --splits 10

REQUIRES
    dat/<dataset>/proc/df_together.csv          (for stages 2 and 4)
    dat/<dataset>/proc/cross_val_<model>_temp/df_res<ij>.csv
    pip install numpy pandas scipy scikit-learn matplotlib seaborn networkx
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, accuracy_score)

warnings.filterwarnings("ignore")

DATASETS = {"ailf": "Analgesics-induced_acute_liver_failure",
            "tram": "Tramadol-related_mortalities"}
# The 5-fold CV actually run on the cluster writes exactly these folds
# (df_res<test><dev>): each fold is the test set once. The old 20-combo
# generator produced 15 ids that never exist on disk - load_split skipped them,
# but the "split 01" output came out empty and medians were mislabelled.
SPLITS = ["10", "21", "32", "43", "04"]
METRICS = {"auc": 0, "precision": 1, "recall": 2, "f1": 3, "accuracy": 4, "ece": 5}
METRIC_COLS = list(METRICS.keys())


# ==========================================================================
# VERBATIM from calc_metrics.ipynb
# ==========================================================================
def classification_scores(df, score_column, n_bins=10):
    y_true = df["label"].values
    y_prob = df[score_column].values
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_correct_sums = np.zeros(n_bins)
    bin_total_sums = np.zeros(n_bins)
    for i in range(n_bins):
        bin_samples = y_true[bin_indices == i]
        bin_total_sums[i] = len(bin_samples)
        bin_correct_sums[i] = bin_samples.sum()

    bin_accs = np.divide(bin_correct_sums, bin_total_sums,
                         out=np.zeros_like(bin_correct_sums),
                         where=bin_total_sums != 0)
    bin_confs = (np.bincount(bin_indices, weights=y_prob, minlength=n_bins)
                 / (np.bincount(bin_indices, minlength=n_bins) + 1e-12))
    ece = np.sum(np.abs(bin_confs - bin_accs) * bin_total_sums) / len(y_true)

    auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_prob.round())
    recall = recall_score(y_true, y_prob.round())
    f1 = f1_score(y_true, y_prob.round())
    accuracy = accuracy_score(y_true, y_prob.round())
    return [auc, precision, recall, f1, accuracy, ece]


# ==========================================================================
# VERBATIM from calc_zscores.ipynb / vis_trees.ipynb
# `as_frame=True` returns the DataFrame form used by vis_trees; the default
# list form is what calc_zscores uses. Both come from the same original code.
# ==========================================================================
def calculate_z_scores(df, terms_dict, score_col, as_frame=False):
    results = []
    for column_name, terms in terms_dict.items():
        z_scores = {}
        filtered_df = df[df[column_name].notna()]
        for term in terms:
            mask_present = filtered_df[column_name].str.contains(term, na=False, regex=False)
            prob_present = filtered_df.loc[mask_present, score_col]
            prob_absent = filtered_df.loc[~mask_present, score_col]
            if len(prob_present) >= 30 and len(prob_absent) >= 30:
                mean_present = prob_present.mean()
                mean_absent = prob_absent.mean()
                std_present = prob_present.std(ddof=1)
                std_absent = prob_absent.std(ddof=1)
                z_score = ((mean_present - mean_absent)
                           / np.sqrt((std_present ** 2 / len(prob_present))
                                     + (std_absent ** 2 / len(prob_absent))))
                z_scores[term] = z_score
        z_scores = dict(sorted(z_scores.items(), key=lambda item: item[1], reverse=True))
        for term, z_score in z_scores.items():
            results.append({'Column': column_name, 'Term': term, 'Z-score': z_score})

    z_scores_df = pd.DataFrame(results)
    try:
        z_scores_df = z_scores_df.sort_values(by='Z-score', ascending=False)
        z_scores_df = z_scores_df[(z_scores_df['Z-score'] > 1.64) | (z_scores_df['Z-score'].isna())]
    except Exception:
        z_scores_df = None

    if as_frame:
        if z_scores_df is None or len(z_scores_df) == 0:
            return None
        out = z_scores_df.copy()
        out['Term'] = out['Column'] + '_' + out['Term']
        return out.reset_index(drop=True)

    if z_scores_df is not None:
        return (z_scores_df["Column"] + "_" + z_scores_df["Term"]).tolist()
    return []


def calculate_ppr(df, terms_dict):
    results = []
    pprs_df = None
    for column_name, terms in terms_dict.items():
        pprs = {}
        filtered_df = df[df[column_name].notna()]
        for term in terms:
            mask_present = filtered_df[column_name].str.contains(term, na=False, regex=False)
            labels_present = filtered_df.loc[mask_present, "label"]
            labels_absent = filtered_df.loc[~mask_present, "label"]
            if len(labels_present) >= 30 and len(labels_absent) >= 30:
                A = labels_present.sum(); B = len(labels_present) - A
                C = labels_absent.sum();  D = len(labels_absent) - C
                if A > 3:
                    ppr = (A * (C + D)) / ((A + B) * C)
                    EA = (A + B) * (A + C) / (A + B + C + D)
                    EB = (A + B) * (B + D) / (A + B + C + D)
                    EC = (C + D) * (A + C) / (A + B + C + D)
                    ED = (C + D) * (B + D) / (A + B + C + D)
                    chi2 = ((A - EA) ** 2 / EA + (B - EB) ** 2 / EB
                            + (C - EC) ** 2 / EC + (D - ED) ** 2 / ED)
                    pprs[term] = (ppr, chi2)
        for term, ppr in pprs.items():
            results.append({'Column': column_name, 'Term': term, 'ppr': ppr[0], 'chi2': ppr[1]})
        pprs_df = pd.DataFrame(results)
    try:
        pprs_df = pprs_df.sort_values(by='ppr', ascending=False)
        pprs_df = pprs_df[pprs_df['ppr'] >= 2]
        pprs_df = pprs_df[pprs_df['chi2'] >= 4]
    except Exception:
        pprs_df = None
    return (pprs_df["Column"] + "_" + pprs_df["Term"]).tolist() if pprs_df is not None else []


def calculate_ror(df, terms_dict):
    results = []
    rors_df = None
    for column_name, terms in terms_dict.items():
        rors = {}
        filtered_df = df[df[column_name].notna()]
        for term in terms:
            mask_present = filtered_df[column_name].str.contains(term, na=False, regex=False)
            labels_present = filtered_df.loc[mask_present, "label"]
            labels_absent = filtered_df.loc[~mask_present, "label"]
            if len(labels_present) >= 30 and len(labels_absent) >= 30 and labels_present.sum() > 3:
                A = labels_present.sum(); B = len(labels_present) - A
                C = labels_absent.sum();  D = len(labels_absent) - B   # (repo's expression, kept as-is)
                ror = (A * D) / (B * C)
                conf = np.exp([np.log(ror) - 1.96 * np.sqrt(1/A + 1/B + 1/C + 1/D),
                               np.log(ror) + 1.96 * np.sqrt(1/A + 1/B + 1/C + 1/D)])
                rors[term] = conf
        for term, conf in rors.items():
            results.append({'Column': column_name, 'Term': term,
                            'conf_l': conf[0], 'conf_u': conf[1]})
        rors_df = pd.DataFrame(results)
    try:
        rors_df = rors_df.sort_values(by='conf_l', ascending=False)
        rors_df = rors_df[rors_df['conf_l'] > 1]
    except Exception:
        rors_df = None
    return (rors_df["Column"] + "_" + rors_df["Term"]).tolist() if rors_df is not None else []


def calculate_ebgm(df, terms_dict):
    results = []
    ebgm_df = None
    for column_name, terms in terms_dict.items():
        ebgms = {}
        filtered_df = df[df[column_name].notna()]
        for term in terms:
            mask_present = filtered_df[column_name].str.contains(term, na=False, regex=False)
            labels_present = filtered_df.loc[mask_present, "label"]
            labels_absent = filtered_df.loc[~mask_present, "label"]
            if len(labels_present) >= 30 and len(labels_absent) >= 30 and labels_present.sum() > 3:
                A = labels_present.sum(); B = len(labels_present) - A
                C = labels_absent.sum();  D = len(labels_absent) - B   # (repo's expression, kept as-is)
                ebgm = (A * (A + B + C + D)) / ((A + B) * (A + C))
                conf = np.exp(np.log(ebgm) - 1.64 * np.sqrt(1/A + 1/B + 1/C + 1/D))
                ebgms[term] = conf
        for term, conf in ebgms.items():
            results.append({'Column': column_name, 'Term': term, 'conf_l': conf})
        ebgm_df = pd.DataFrame(results)
    try:
        ebgm_df = ebgm_df.sort_values(by='conf_l', ascending=False)
        ebgm_df = ebgm_df[ebgm_df['conf_l'] > 2]
    except Exception:
        ebgm_df = None
    return (ebgm_df["Column"] + "_" + ebgm_df["Term"]).tolist() if ebgm_df is not None else []


# ==========================================================================
# helpers (plumbing)
# ==========================================================================
def proc_dir(root, short):
    return os.path.join(root, "dat", DATASETS[short], "proc")


def find_together(root, short):
    """Locate df_together.csv for stages 2 and 4. The fetch script leaves
    per-dataset copies as <short>_df_together.csv in root/ and root/root_csvs/,
    so check those too instead of forcing a manual copy into proc/."""
    for c in (os.path.join(proc_dir(root, short), "df_together.csv"),
              os.path.join(root, f"{short}_df_together.csv"),
              os.path.join(root, "root_csvs", f"{short}_df_together.csv")):
        if os.path.isfile(c):
            return c
    return None


def find_model_dirs(pdir):
    found = {}
    if not os.path.isdir(pdir):
        return found
    for d in sorted(os.listdir(pdir)):
        full = os.path.join(pdir, d)
        if os.path.isdir(full):
            if d.startswith("cross_val_") and d.endswith("_temp"):
                found[d[len("cross_val_"):-len("_temp")]] = full
            elif d == "cross_val_xgb":
                found["xgb"] = full
    return found


def load_split(pdir, model_dirs, split):
    """
    Concatenate every model's score columns for one split, plus the label.

    All models are trained on the same fold, so their prediction CSVs share an
    index. That is asserted rather than assumed: if the indices ever disagree
    the common subset is used and a warning is printed, because silently
    concatenating misaligned frames would inject NaNs and corrupt every metric.
    """
    frames, label, idx = [], None, None
    for key, path in model_dirs.items():
        f = os.path.join(path, f"df_res{split}.csv")
        if not os.path.isfile(f):
            continue
        d = pd.read_csv(f, index_col=0)
        if label is None and "label" in d.columns:
            label = d["label"]
        idx = d.index if idx is None else idx.intersection(d.index)
        sc = d.drop(columns=["label"], errors="ignore")
        # Each df_res carries the model's raw score plus an isotonic-calibrated
        # twin (…_cal), e.g. xgb -> {xgb, xgb_cal}. Rename by FOLDER key so the
        # three PEFT rank configs (…_paper/_r64/_r128) - which share an internal
        # column name - don't overwrite each other on concat, while keeping the
        # _cal suffix so calibrated and uncalibrated metrics stay separable.
        newnames = [f"{key}_cal" if str(c).endswith("_cal") else key
                    for c in sc.columns]
        if len(set(newnames)) != len(newnames):
            # unexpected layout (>1 raw column) -> namespace every column safely
            newnames = [f"{key}_{c}" for c in sc.columns]
        sc.columns = newnames
        frames.append(sc)
    if not frames or label is None:
        return None

    if any(len(fr) != len(idx) for fr in frames):
        print(f"    warn: split {split} indices differ across models; "
              f"using the {len(idx)} shared rows", file=sys.stderr)
    frames = [fr.loc[idx] for fr in frames]
    return pd.concat(frames + [label.loc[idx]], axis=1)


def build_terms_dict(df):
    """The repo's term extraction, with outcome/ade chosen by what exists."""
    def uniq(col):
        t = df[col].dropna().str.split(',').explode().str.strip()
        return sorted(t.unique().tolist())

    td = {}
    for c in ['psd', 'ssd', 'ccd', 'idrug', 'indication']:
        if c in df.columns:
            td[c] = uniq(c)
    for c in ['outcome', 'ade']:          # AILF has outcome, TRAM has ade
        if c in df.columns:
            td[c] = uniq(c)
    if 'age' in df.columns:
        td['age'] = sorted(set(df["age"].dropna()))
    if 'dose' in df.columns:
        td['dose'] = sorted(set(df["dose"].dropna()))
    if 'gender' in df.columns:
        td['gender'] = sorted(["Male", "Female"])
    return td


# ==========================================================================
# STAGE 1 -- metrics
# ==========================================================================
def stage_metrics(root, short, outdir, compare=None):
    pdir = proc_dir(root, short)
    model_dirs = find_model_dirs(pdir)
    if not model_dirs:
        print(f"  [{short}] no cross_val_* dirs in {pdir}", file=sys.stderr)
        return None
    print(f"  models: {', '.join(sorted(model_dirs))}")

    rows, all_results = [], {}
    for split in SPLITS:
        df = load_split(pdir, model_dirs, split)
        if df is None:
            continue
        per = {}
        for col in df.columns:
            if col == "label":
                continue
            sc = classification_scores(df, col)
            per[col] = sc
            rows.append([split, col] + sc)
        all_results[split] = per

    if not rows:
        return None
    scores = pd.DataFrame(rows, columns=["cv_id", "model"] + METRIC_COLS)
    scores.to_csv(os.path.join(outdir, f"{short}_classification_scores.csv"), index=False)

    # all_results.json is what stage 3 consumes
    with open(os.path.join(outdir, f"{short}_all_results.json"), "w") as f:
        json.dump(all_results, f)

    grp = scores.groupby("model")[METRIC_COLS]
    med  = grp.median().reset_index()
    mean = grp.mean().reset_index()
    std  = grp.std(ddof=1).reset_index()

    med.to_csv(os.path.join(outdir,  f"{short}_median_classification_results.csv"), index=False)
    mean.to_csv(os.path.join(outdir, f"{short}_mean_classification_results.csv"),   index=False)

    # dissertation-ready combined table: mean and std side by side per metric
    summary = (mean.set_index("model").add_suffix("_mean")
               .join(std.set_index("model").add_suffix("_std")))
    summary = summary[[f"{m}_{s}" for m in METRIC_COLS
                       for s in ("mean", "std")]].reset_index()
    summary.to_csv(os.path.join(outdir, f"{short}_summary_mean_std.csv"), index=False)

    # Fold coverage per score column. Anything below 5 means a job did not
    # finish all folds, and its mean/median is over fewer values.
    cov = scores.groupby("model")["cv_id"].nunique()
    cov.rename("n_folds").reset_index().to_csv(
        os.path.join(outdir, f"{short}_fold_coverage.csv"), index=False)
    short_cov = cov[cov < len(SPLITS)]
    if len(short_cov):
        print(f"\n  INCOMPLETE score columns (fewer than {len(SPLITS)} folds):")
        for m, k in short_cov.items():
            got = sorted(scores.loc[scores.model == m, "cv_id"].unique())
            print(f"      {m:<28} {k}/{len(SPLITS)} folds: {','.join(got)}")

    fmt = lambda v: f"{v:.4f}"
    print(f"\n  MEAN over {len(SPLITS)} folds")
    print(mean.to_string(index=False, float_format=fmt))
    print(f"\n  MEDIAN over {len(SPLITS)} folds (comparable with the paper's tables)")
    print(med.to_string(index=False, float_format=fmt))

    # per-fold tables: one CSV per fold + printed, for every fold that has data.
    # Each carries all six metrics for every model (same layout as the fold view).
    present = [s for s in SPLITS if (scores.cv_id == s).any()]
    for split in present:
        sf = scores[scores.cv_id == split].drop(columns="cv_id").reset_index(drop=True)
        sf.to_csv(os.path.join(outdir, f"{short}_split{split}_classification_results.csv"), index=False)
        print(f"\n  FOLD {split} only (single-fold view)")
        print(sf.to_string(index=False, float_format=fmt))

    if compare:
        pf = os.path.join(compare, f"{short}_classification_scores.csv")
        if os.path.isfile(pf):
            pub = pd.read_csv(pf)
            pub_med = pub.groupby("model")[METRIC_COLS].median()
            shared = sorted(set(pub_med.index) & set(med.set_index("model").index))
            if shared:
                print("\n  REPRODUCTION CHECK vs published medians (yours - published)")
                diff = med.set_index("model").loc[shared, METRIC_COLS] - pub_med.loc[shared, METRIC_COLS]
                print(diff.to_string(float_format=lambda v: f"{v:+.4f}"))
    return scores


# ==========================================================================
# STAGE 2 -- z-scores + disproportionality
# ==========================================================================
def stage_zscores(root, short, outdir):
    pdir = proc_dir(root, short)
    together = find_together(root, short)
    if together is None:
        print(f"  [{short}] df_together.csv not found - looked in proc/, "
              f"{short}_df_together.csv and root_csvs/. Stages 2 and 4 need "
              f"the clinical term columns.", file=sys.stderr)
        return None
    model_dirs = find_model_dirs(pdir)
    base = pd.read_csv(together, index_col=0).drop(columns=["Temp_sentence"], errors="ignore")

    all_results = {}
    for split in SPLITS:
        preds = load_split(pdir, model_dirs, split)
        if preds is None:
            continue
        df = base.loc[preds.index].copy()
        df = pd.concat([df.drop(columns=["label"], errors="ignore"), preds], axis=1)
        td = build_terms_dict(df)

        res = {}
        for col in preds.columns:
            if col == "label":
                continue
            res[col] = calculate_z_scores(df, td, col)
        res["prr"] = calculate_ppr(df, td)
        res["ror"] = calculate_ror(df, td)
        res["ebgm"] = calculate_ebgm(df, td)
        all_results[split] = res
        print(f"    split {split}: " + "  ".join(
            f"{k}={len(v)}" for k, v in list(res.items())[:4]) + "  ...")

    out = os.path.join(outdir, f"{short}_all_sigterms.json")
    with open(out, "w") as f:
        json.dump(all_results, f)

    # convenience summary: how many significant terms each model finds, median over splits
    counts = {}
    for split, res in all_results.items():
        for k, v in res.items():
            counts.setdefault(k, []).append(len(v))
    summ = pd.DataFrame({"method": list(counts), 
                         "median_n_sig_terms": [int(np.median(v)) for v in counts.values()]})
    summ = summ.sort_values("median_n_sig_terms", ascending=False)
    summ.to_csv(os.path.join(outdir, f"{short}_sigterm_counts.csv"), index=False)
    print("\n  significant-term counts (median over splits)")
    print(summ.to_string(index=False))
    print(f"\n  written -> {out}")
    return all_results


# ==========================================================================
# STAGE 3 -- paired t-tests  (VERBATIM calc_student from student_test.ipynb)
# ==========================================================================
def calc_student(results, model1, model2, metric):
    model1_vals, model2_vals = [], []
    for key in results.keys():
        model1_vals.append(results[key][model1][METRICS[metric]])
        model2_vals.append(results[key][model2][METRICS[metric]])
    model1_vals = np.array(model1_vals)
    model2_vals = np.array(model2_vals)
    if metric == "ece":
        model1_vals = model1_vals * -1
        model2_vals = model2_vals * -1
    differences = model1_vals - model2_vals
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    n = len(differences)
    t_stat = mean_diff / (std_diff / np.sqrt(n))
    df = n - 1
    return 1 - stats.t.cdf(t_stat, df)


def stage_ttest(root, short, outdir):
    rf = os.path.join(outdir, f"{short}_all_results.json")
    if not os.path.isfile(rf):
        print(f"  [{short}] {rf} missing - run the metrics stage first.", file=sys.stderr)
        return None
    with open(rf) as f:
        results = json.load(f)

    # A paired t-test needs the SAME folds for both models. Taking the model
    # list from the first split alone raises KeyError as soon as any model is
    # missing a fold (e.g. Gemma4-31B AILF stopping at 4/5 on walltime).
    # Use only models present in EVERY fold, and say which were dropped.
    splits_present = list(results.keys())
    complete = set(results[splits_present[0]].keys())
    for sp in splits_present[1:]:
        complete &= set(results[sp].keys())
    all_seen = set()
    for sp in splits_present:
        all_seen |= set(results[sp].keys())
    dropped = sorted(all_seen - complete)
    model_names = sorted(complete)
    if dropped:
        print(f"  NOTE: {len(dropped)} score column(s) excluded from the paired "
              f"t-tests because they are missing at least one fold:")
        for d in dropped:
            have = sorted(sp for sp in splits_present if d in results[sp])
            print(f"        {d:<28} folds present: {','.join(have)}")
        print("        (their per-fold, mean and median metrics are still in the "
              "stage-1 CSVs)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import ListedColormap
        have_plot = True
    except ImportError:
        have_plot = False
        print("  (matplotlib/seaborn not installed - writing CSVs only)")

    hm_dir = os.path.join(outdir, f"{short}_pvalue_matrices")
    os.makedirs(hm_dir, exist_ok=True)

    for metric in METRICS:
        M = np.zeros((len(model_names), len(model_names)))
        for i in range(len(model_names)):
            for j in range(len(model_names)):
                M[i, j] = 1 if i == j else calc_student(results, model_names[i], model_names[j], metric)
        pdf = pd.DataFrame(M, index=model_names, columns=model_names)
        pdf.to_csv(os.path.join(hm_dir, f"{metric}_pvalues.csv"))

        if have_plot:
            # 11 models gave a 22x22 grid; 27 models give 54x54. Annotating 2916
            # cells in a 10x10in figure is unreadable, so scale the canvas and
            # drop the numbers once the matrix is large (the CSV keeps them).
            n = len(model_names)
            side = max(10, min(40, 0.42 * n))
            annot = n <= 20
            tick_fs = max(4, min(10, 260 / max(n, 1)))
            plt.figure(figsize=(side, side))
            sns.heatmap(M, annot=annot, fmt=".2f" if annot else "",
                        annot_kws={"size": max(4, tick_fs - 1)} if annot else None,
                        xticklabels=model_names, yticklabels=model_names,
                        cmap=ListedColormap(['red', 'green']), cbar=False,
                        vmin=0, vmax=0.05, center=0.05, linewidth=0.05)
            plt.xticks(fontsize=tick_fs, rotation=90)
            plt.yticks(fontsize=tick_fs, rotation=0)
            plt.title(f"Student's t-test p-values for {DATASETS[short]}, {metric}")
            plt.tight_layout()
            plt.savefig(os.path.join(hm_dir, f"{metric}_pvalues.png"), dpi=150)
            plt.close()

        sig = [(model_names[i], model_names[j])
               for i in range(len(model_names)) for j in range(len(model_names))
               if i != j and M[i, j] < 0.05]
        print(f"    {metric:<10} significant (row > col, p<0.05): {len(sig)} pairs")
        for a, b in sig[:6]:
            print(f"        {a} > {b}   p={pdf.loc[a, b]:.2e}")
        if len(sig) > 6:
            print(f"        ... and {len(sig)-6} more (see {metric}_pvalues.csv)")
    print(f"\n  written -> {hm_dir}/")
    return True


# ==========================================================================
# STAGE 4 -- z-score pyramid trees  (logic from vis_trees.ipynb)
# ==========================================================================
def stage_trees(root, short, outdir, only_splits=None, only_models=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("  needs networkx + matplotlib:  pip install networkx matplotlib", file=sys.stderr)
        return None

    pdir = proc_dir(root, short)
    together = find_together(root, short)
    if together is None:
        print(f"  [{short}] df_together.csv not found - skipping trees.", file=sys.stderr)
        return None
    model_dirs = find_model_dirs(pdir)
    base = pd.read_csv(together, index_col=0).drop(columns=["Temp_sentence"], errors="ignore")
    tree_root = os.path.join(outdir, f"{short}_trees")
    made = 0

    splits = only_splits or SPLITS
    print(f"  building trees for up to {len(find_model_dirs(pdir))*2} score columns "
          f"x {len(splits)} folds (skips any that fail the z-score thresholds)")
    for split in splits:
        preds = load_split(pdir, model_dirs, split)
        if preds is None:
            continue
        df = base.loc[preds.index].copy()
        df = pd.concat([df.drop(columns=["label"], errors="ignore"), preds], axis=1)
        td = build_terms_dict(df)
        models = [c for c in preds.columns if c != "label"]
        if only_models:
            models = [m for m in models if m in only_models]

        print(f"    fold {split}: {len(models)} score columns", flush=True)
        for model in models:
            try:
                z = calculate_z_scores(df, td, model, as_frame=True)
                if z is None or len(z) < 1:
                    continue
                first_term = {'Term': z['Term'].iloc[0], 'Z-score': float(z['Z-score'].iloc[0])}
                colname, term_value = first_term['Term'].split('_', 1)
                reduced_df = df[df[colname].str.contains(term_value, na=False, regex=False)]

                z = calculate_z_scores(reduced_df, td, model, as_frame=True)
                if z is None or len(z) < 2:
                    continue
                second_terms = {0: {'Term': z['Term'].iloc[0], 'Z-score': float(z['Z-score'].iloc[0])},
                                1: {'Term': z['Term'].iloc[1], 'Z-score': float(z['Z-score'].iloc[1])}}

                c0, v0 = second_terms[0]['Term'].split('_', 1)
                rd0 = reduced_df[reduced_df[c0].str.contains(v0, na=False, regex=False)]
                z0 = calculate_z_scores(rd0, td, model, as_frame=True)
                if z0 is not None:
                    z0 = z0.rename(columns={'Z-score': 'Z-score 0'})
                    z0 = z0[z0['Term'] != second_terms[1]['Term']]

                c1, v1 = second_terms[1]['Term'].split('_', 1)
                rd1 = reduced_df[reduced_df[c1].str.contains(v1, na=False, regex=False)]
                z1 = calculate_z_scores(rd1, td, model, as_frame=True)
                if z1 is not None:
                    z1 = z1.rename(columns={'Z-score': 'Z-score 1'})
                    z1 = z1[z1['Term'] != second_terms[0]['Term']]

                parts = []
                if z0 is not None and len(z0):
                    parts.append(z0.set_index('Term')[['Z-score 0']])
                if z1 is not None and len(z1):
                    parts.append(z1.set_index('Term')[['Z-score 1']])
                combined = pd.concat(parts, axis=1) if parts else None
                if combined is not None:
                    zc = [c for c in combined.columns if c.startswith('Z-score')]
                    combined['max Z-score'] = combined[zc].max(axis=1)
                    combined = combined.sort_values('max Z-score', ascending=False)
                    combined.drop('max Z-score', axis=1, inplace=True)
                    combined = combined.reset_index().head(3)
                    for c in ['Z-score 0', 'Z-score 1']:
                        if c not in combined.columns:
                            combined[c] = np.nan

                G = nx.DiGraph()
                if combined is not None:
                    for t in combined['Term']:
                        G.add_node(t)
                G.add_node(second_terms[0]['Term']); G.add_node(second_terms[1]['Term'])
                G.add_node(first_term['Term']); G.add_node(short)

                if combined is not None:
                    for _, row in combined.iterrows():
                        if not pd.isna(row['Z-score 0']):
                            G.add_edge(row['Term'], second_terms[0]['Term'], weight=row['Z-score 0'])
                        if not pd.isna(row['Z-score 1']):
                            G.add_edge(row['Term'], second_terms[1]['Term'], weight=row['Z-score 1'])
                G.add_edge(second_terms[0]['Term'], first_term['Term'], weight=second_terms[0]['Z-score'])
                G.add_edge(second_terms[1]['Term'], first_term['Term'], weight=second_terms[1]['Z-score'])
                G.add_edge(first_term['Term'], short, weight=first_term['Z-score'])

                if combined is not None:
                    levels = {0: list(combined['Term']),
                              1: [second_terms[0]['Term'], second_terms[1]['Term']],
                              2: [first_term['Term']], 3: [short]}
                else:
                    levels = {0: [first_term['Term']],
                              1: [second_terms[0]['Term'], second_terms[1]['Term']],
                              2: [short]}

                pos, y = {}, {0: 0, 1: 1, 2: 2, 3: 3}
                for lvl, nodes in levels.items():
                    x0 = -(len(nodes) - 1) * 1.0 / 2
                    for i, node in enumerate(nodes):
                        pos[node] = (x0 + i * 1.0, y[lvl])

                plt.figure(figsize=(10, 8))
                nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000,
                        font_size=6, font_weight='bold', arrows=True,
                        arrowstyle='->', arrowsize=20)
                nx.draw_networkx_edge_labels(
                    G, pos, {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)},
                    font_size=8)
                plt.title('Hierarchical Pyramid Graph of Z-Scores')
                d = os.path.join(tree_root, model)
                os.makedirs(d, exist_ok=True)
                plt.savefig(os.path.join(d, f'graph_{split}.pdf'))
                plt.close()
                made += 1
            except Exception:
                plt.close('all')
                continue
    print(f"  generated {made} tree PDFs -> {tree_root}/")
    return made


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.expanduser("~/biocausal_results"))
    ap.add_argument("--dataset", choices=["ailf", "tram", "both"], default="both")
    ap.add_argument("--stages", nargs="+",
                    choices=["metrics", "zscores", "ttest", "trees", "all"], default=["all"])
    ap.add_argument("--outdir", default=None, help="default: <root>/evaluation")
    ap.add_argument("--compare", default=None, help="path to a clone of the 2026 repo")
    ap.add_argument("--splits", nargs="+", default=None, help="limit trees to these splits, e.g. 01 12")
    ap.add_argument("--models", nargs="+", default=None, help="limit trees to these score columns")
    args = ap.parse_args()

    stages = ["metrics", "zscores", "ttest", "trees"] if "all" in args.stages else args.stages
    outdir = args.outdir or os.path.join(args.root, "evaluation")
    os.makedirs(outdir, exist_ok=True)
    todo = ["ailf", "tram"] if args.dataset == "both" else [args.dataset]

    for short in todo:
        print(f"\n{'='*74}\n{DATASETS[short]}\n{'='*74}")
        if "metrics" in stages:
            print("\n[1/4] METRICS  (calc_metrics.ipynb)")
            stage_metrics(args.root, short, outdir, args.compare)
        if "zscores" in stages:
            print("\n[2/4] Z-SCORES + PRR/ROR/EBGM  (calc_zscores.ipynb)")
            stage_zscores(args.root, short, outdir)
        if "ttest" in stages:
            print("\n[3/4] PAIRED T-TESTS  (student_test.ipynb)")
            stage_ttest(args.root, short, outdir)
        if "trees" in stages:
            print("\n[4/4] Z-SCORE PYRAMID TREES  (vis_trees.ipynb)")
            stage_trees(args.root, short, outdir, args.splits, args.models)

    print(f"\nAll output in: {outdir}")


if __name__ == "__main__":
    main()
