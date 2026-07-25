"""
=============================================================================
FAERS Dataset — Exploratory Data Analysis & Preprocessing
Replicating the InferBERT / Kiss et al. (2026) dissertation analysis

Datasets : AILF_dataset.csv  (Analgesics-Induced Acute Liver Failure)
           TRAM_dataset.csv  (Tramadol-Related Mortalities)

Run in   : Google Colab  OR  VS Code (Python 3.8+)
=============================================================================
"""

# ── 0. INSTALL / IMPORT ───────────────────────────────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn",
            "wordcloud", "tqdm"]:
    try:
        __import__(pkg.split("[")[0].replace("-","_"))
    except ImportError:
        install(pkg)

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

# ── COLOUR PALETTE (matches dissertation PPT) ─────────────────────────────────
C_DEEP   = "#065A82"   # deep blue  (positive bars)
C_MINT   = "#02C39A"   # mint green (negative bars)
C_TEAL   = "#1C7293"
C_DARK   = "#21295C"
C_ORANGE = "#FF6B35"
C_BG     = "#F0F8FF"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  SECTION 1 — Loading Raw Datasets")
print("=" * 70)

# ➜ Adjust these paths if running locally
AILF_PATH = "Analgesics-induced acute liver failure/AILF_dataset.csv"
TRAM_PATH = "Tramadol-related mortalities/TRAM_dataset.csv"

# Google Colab: upload files or mount Drive, then update paths above.

ailf_raw = pd.read_csv(AILF_PATH, low_memory=False, encoding="utf-8-sig")
tram_raw = pd.read_csv(TRAM_PATH, low_memory=False, encoding="utf-8-sig")

print(f"AILF raw shape : {ailf_raw.shape[0]:,} rows × {ailf_raw.shape[1]:,} cols")
print(f"TRAM raw shape : {tram_raw.shape[0]:,} rows × {tram_raw.shape[1]:,} cols")

# Keep only the clinically relevant columns
CORE_COLS = ["Primary Suspect Drugs", "Other Administered Drugs",
             "Adverse Events", "Outcomes", "Gender", "Age", "Age Units",
             "Dose", "Indications", "Reporter Occupation", "Location"]

def select_core(df, extra=None):
    cols = [c for c in CORE_COLS + (extra or []) if c in df.columns]
    return df[cols].copy()

ailf = select_core(ailf_raw)
tram = select_core(tram_raw)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  LABEL CREATION (InferBERT methodology)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 2 — Label Creation")
print("=" * 70)

# ── AILF: positive = presence of any MedDRA PT from the HLGT
#   'Acute liver failure and associated disorders' in the Adverse Events field ──
#
#   The paper says they labelled positives using "acute liver failure" as the
#   endpoint term. In PharmaPendium, "Acute liver failure and associated
#   disorders" is a MedDRA High-Level Group Term (HLGT). Its constituent
#   Preferred Terms (PTs) as they appear in the Adverse Events free-text column
#   are:
#     - "Hepatic failure"              (the generic/parent PT — most frequent)
#     - "Acute hepatic failure"        (the specific acute PT)
#     - "Subacute hepatic failure"
#     - "Acute on chronic liver failure"
#
#   NOTE: "Chronic hepatic failure" belongs to a DIFFERENT HLGT
#   ('Chronic hepatic failure') and must be excluded.
#
#   The string "acute liver failure" itself does NOT appear in the dataset
#   (PharmaPendium normalises to MedDRA PTs), which is why the original
#   code only matched ~4,800 rows instead of the expected ~15,000+.

ALF_TERMS = [
    "hepatic failure",           # catches "Hepatic failure" (~12,000 records)
    "acute hepatic failure",     # catches "Acute hepatic failure" (~4,700)
    "subacute hepatic failure",
    "acute on chronic liver failure",
]
ALF_EXCLUDE = ["chronic hepatic failure"]   # NOT in the HLGT — exclude it

def contains_any(series, terms):
    mask = pd.Series([False] * len(series), index=series.index)
    for t in terms:
        mask = mask | series.str.lower().str.contains(t, na=False, regex=False)
    return mask

ailf["label"] = (
    contains_any(ailf["Adverse Events"].fillna(""), ALF_TERMS) &
    ~contains_any(ailf["Adverse Events"].fillna(""), ALF_EXCLUDE)
).astype(int)

# ── TRAM: positive = 'death' appears in Outcomes column ─────────────────────
#   Drop rows with empty outcomes first (paper: "excluded case reports without
#   clinical outcome information")
tram = tram[tram["Outcomes"].fillna("").str.strip() != ""].copy()
tram["label"] = tram["Outcomes"].fillna("").str.lower().str.contains(
    "death", na=False, regex=False).astype(int)

print(f"\nAILF  — Total: {len(ailf):,}  |  "
      f"Positive (ALF):  {ailf['label'].sum():,}  |  "
      f"Negative: {(ailf['label']==0).sum():,}  |  "
      f"Ratio: {ailf['label'].mean():.2f}")
print(f"TRAM  — Total: {len(tram):,}  |  "
      f"Positive (Death): {tram['label'].sum():,}  |  "
      f"Negative: {(tram['label']==0).sum():,}  |  "
      f"Ratio: {tram['label'].mean():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 3 — Preprocessing & Feature Engineering")
print("=" * 70)

# ── 3a. Age normalisation ─────────────────────────────────────────────────────
def parse_age(age_str, unit_str="Year"):
    """Convert any age to years as a float."""
    try:
        age = float(str(age_str).strip())
        unit = str(unit_str).strip().lower()
        if "month" in unit:
            return age / 12
        elif "week" in unit:
            return age / 52
        elif "day" in unit:
            return age / 365
        return age
    except Exception:
        return np.nan

for df in [ailf, tram]:
    unit_col = "Age Units" if "Age Units" in df.columns else None
    if unit_col:
        df["age_years"] = df.apply(
            lambda r: parse_age(r["Age"], r[unit_col]), axis=1)
    else:
        df["age_years"] = pd.to_numeric(df["Age"], errors="coerce")
    # Clip unrealistic ages
    df["age_years"] = df["age_years"].clip(0, 120)

# ── 3b. Age groups (matching InferBERT paper) ─────────────────────────────────
age_bins   = [0, 18, 40, 65, 120]
age_labels = ["<18", "18–39", "40–64", "≥65"]

for df in [ailf, tram]:
    df["age_group"] = pd.cut(df["age_years"], bins=age_bins,
                             labels=age_labels, right=False)

# ── 3c. Dose normalisation ────────────────────────────────────────────────────
def classify_dose(dose_str):
    """Classify dose as >100 mg, ≤100 mg, Drug Abuse, or Unknown."""
    s = str(dose_str).strip().lower()
    if any(t in s for t in ["abuse", "misuse", "recreational", "overdose"]):
        return "Drug Abuse"
    import re
    numbers = re.findall(r"(\d+(?:\.\d+)?)\s*mg", s)
    if numbers:
        val = max(float(n) for n in numbers)
        return ">100 mg" if val > 100 else "≤100 mg"
    return "Unknown"

for df in [ailf, tram]:
    df["dose_group"] = df["Dose"].fillna("").apply(classify_dose)

# ── 3d. Gender cleaning ───────────────────────────────────────────────────────
VALID_GENDERS = {"Female", "Male"}
for df in [ailf, tram]:
    df["gender_clean"] = df["Gender"].where(
        df["Gender"].isin(VALID_GENDERS), other="Unknown")

# ── 3e. Missing value summary ─────────────────────────────────────────────────
print("\n--- Missing Values (%) ---")
for name, df in [("AILF", ailf), ("TRAM", tram)]:
    miss = (df[CORE_COLS + ["label"]].isnull() |
            df[CORE_COLS + ["label"]].apply(
                lambda col: col.astype(str).str.strip() == "")).mean() * 100
    miss = miss[miss > 0].sort_values(ascending=False)
    print(f"\n{name}:")
    print(miss.round(1).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 4.  SENTENCE TEMPLATE GENERATION  (InferBERT input format)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 4 — Sentence Template Generation")
print("=" * 70)

def clean_field(val):
    """Strip newlines and extra whitespace from a raw FAERS field."""
    return str(val).replace("\n", ", ").replace("  ", " ").strip()

def make_sentence(row, endpoint_col, is_ailf=True):
    """
    Generate InferBERT-style sentence:
    'Patient (gender and age) takes primary drug to treat disease,
     causing adverse events, leading to outcomes.'
    For AILF: endpoint is the adverse event field.
    For TRAM: endpoint is the outcome field.
    """
    gender  = str(row.get("gender_clean", "unknown")).lower()
    age     = row.get("age_years", "unknown")
    age_str = f"{int(age)}" if pd.notna(age) and age != "unknown" else "unknown"

    drug     = clean_field(row.get("Primary Suspect Drugs", "unknown drug"))
    disease  = clean_field(row.get("Indications",           "unknown disease"))
    ade      = clean_field(row.get("Adverse Events",        "unknown adverse events"))
    outcome  = clean_field(row.get("Outcomes",              "unknown"))

    if is_ailf:
        return (f"Patient ({gender} and {age_str}) takes {drug} to treat "
                f"{disease} and cause some adverse events, "
                f"leading to {ade}.")
    else:
        return (f"Patient ({gender} and {age_str}) takes {drug} to treat "
                f"{disease} and cause some adverse events {ade}, "
                f"leading to {outcome}.")

ailf["sentence"] = ailf.apply(
    lambda r: make_sentence(r, "Adverse Events", is_ailf=True), axis=1)
tram["sentence"] = tram.apply(
    lambda r: make_sentence(r, "Outcomes", is_ailf=False), axis=1)

ailf["seq_len"] = ailf["sentence"].str.split().str.len()
tram["seq_len"] = tram["sentence"].str.split().str.len()

print(f"\nAILF sentence length — mean: {ailf['seq_len'].mean():.1f} ± {ailf['seq_len'].std():.1f}")
print(f"TRAM sentence length — mean: {tram['seq_len'].mean():.1f} ± {tram['seq_len'].std():.1f}")
print("\nSample AILF sentence:")
print(" ", ailf["sentence"].iloc[0][:200])
print("\nSample TRAM sentence:")
print(" ", tram["sentence"].iloc[0][:200])

# ─────────────────────────────────────────────────────────────────────────────
# 5.  TRAIN / DEV / TEST SPLIT  (64 : 16 : 20 — stratified)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 5 — Train / Dev / Test Split")
print("=" * 70)

def stratified_split(df, label_col="label", test_size=0.20, dev_size=0.16,
                     random_state=42):
    """Stratified 64/16/20 split matching the InferBERT paper."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[label_col],
        random_state=random_state)
    val_frac = dev_size / (1 - test_size)
    train, dev = train_test_split(
        train_val, test_size=val_frac, stratify=train_val[label_col],
        random_state=random_state)
    return train, dev, test

ailf_train, ailf_dev, ailf_test = stratified_split(ailf)
tram_train, tram_dev, tram_test = stratified_split(tram)

print(f"\nAILF  — Train: {len(ailf_train):,}  Dev: {len(ailf_dev):,}  "
      f"Test: {len(ailf_test):,}")
print(f"TRAM  — Train: {len(tram_train):,}  Dev: {len(tram_dev):,}  "
      f"Test: {len(tram_test):,}")

for split_name, split_df in [("AILF Train", ailf_train), ("AILF Dev", ailf_dev),
                               ("AILF Test", ailf_test),  ("TRAM Train", tram_train),
                               ("TRAM Dev",  tram_dev),   ("TRAM Test",  tram_test)]:
    pos = split_df["label"].sum()
    neg = (split_df["label"] == 0).sum()
    print(f"  {split_name:12s} — Pos: {pos:,}  Neg: {neg:,}  "
          f"Ratio: {pos/len(split_df):.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 6 — Generating Visualisations")
print("=" * 70)

os.makedirs("eda_output", exist_ok=True)

# ── Helper ────────────────────────────────────────────────────────────────────
def save_and_show(fig, filename):
    path = os.path.join("eda_output", filename)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {path}")
    plt.show()
    plt.close(fig)

# ── 6.1  Class Distribution (side-by-side) ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("FAERS Dataset — Class Distribution", fontsize=14,
             fontweight="bold", color=C_DARK)

for ax, (name, df), pos_label, neg_label in zip(
        axes,
        [("AILF", ailf), ("TRAM", tram)],
        ["Acute Liver\nFailure (ALF)", "Patient\nDeath"],
        ["No ALF",                     "No Death"]):

    pos = df["label"].sum()
    neg = (df["label"] == 0).sum()
    bars = ax.bar([pos_label, neg_label], [pos, neg],
                  color=[C_DEEP, C_MINT], width=0.5,
                  edgecolor="white", linewidth=1.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 150,
                f"{h:,}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=C_DARK)
    ax.set_title(name, fontsize=13, fontweight="bold", color=C_DARK, pad=8)
    ax.set_ylabel("Number of Reports", fontsize=10)
    ax.set_ylim(0, max(pos, neg) * 1.2)
    ax.set_xlabel(f"Total: {len(df):,} reports", fontsize=9,
                  color=C_TEAL, style="italic")
    ax.tick_params(labelsize=11)

plt.tight_layout()
save_and_show(fig, "01_class_distribution.png")

# ── 6.2  Gender Distribution ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Gender Distribution by Dataset & Label",
             fontsize=14, fontweight="bold", color=C_DARK)

for ax, (name, df) in zip(axes, [("AILF", ailf), ("TRAM", tram)]):
    gender_label = df.groupby(["gender_clean", "label"]).size().unstack(fill_value=0)
    gender_label.columns = ["Negative", "Positive"]
    gender_label = gender_label.reindex(["Female", "Male", "Unknown"])
    gender_label.plot(kind="bar", ax=ax, color=[C_MINT, C_DEEP],
                      edgecolor="white", width=0.6)
    ax.set_title(name, fontsize=13, fontweight="bold", color=C_DARK)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0, labelsize=11)
    ax.legend(title="Label", fontsize=10)

plt.tight_layout()
save_and_show(fig, "02_gender_distribution.png")

# ── 6.3  Age Group Distribution ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Age Group Distribution by Dataset & Label",
             fontsize=14, fontweight="bold", color=C_DARK)

for ax, (name, df) in zip(axes, [("AILF", ailf), ("TRAM", tram)]):
    age_lbl = df.dropna(subset=["age_group"])
    ag = age_lbl.groupby(["age_group", "label"]).size().unstack(fill_value=0)
    ag.columns = ["Negative", "Positive"]
    ag.plot(kind="bar", ax=ax, color=[C_MINT, C_DEEP],
            edgecolor="white", width=0.65)
    ax.set_title(name, fontsize=13, fontweight="bold", color=C_DARK)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0, labelsize=11)
    ax.legend(title="Label", fontsize=10)

plt.tight_layout()
save_and_show(fig, "03_age_group_distribution.png")

# ── 6.4  Age Distribution (continuous histogram) ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Age Distribution (continuous)",
             fontsize=14, fontweight="bold", color=C_DARK)

for ax, (name, df) in zip(axes, [("AILF", ailf), ("TRAM", tram)]):
    pos_ages = df[df["label"] == 1]["age_years"].dropna()
    neg_ages = df[df["label"] == 0]["age_years"].dropna()
    ax.hist(neg_ages, bins=40, color=C_MINT, alpha=0.7, label="Negative")
    ax.hist(pos_ages, bins=40, color=C_DEEP, alpha=0.7, label="Positive")
    ax.set_title(name, fontsize=13, fontweight="bold", color=C_DARK)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=10)
    ax.axvline(pos_ages.mean(), color=C_DEEP, linestyle="--", linewidth=1.5,
               label=f"Pos mean: {pos_ages.mean():.1f}")
    ax.axvline(neg_ages.mean(), color=C_MINT, linestyle="--", linewidth=1.5,
               label=f"Neg mean: {neg_ages.mean():.1f}")

plt.tight_layout()
save_and_show(fig, "04_age_histogram.png")

# ── 6.5  Outcome Distribution ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("Outcome Category Distribution",
             fontsize=14, fontweight="bold", color=C_DARK)

colours = [C_DEEP, C_TEAL, C_MINT, C_ORANGE, "#E91E63", "#8E24AA", "#FF8F00"]

for ax, (name, df) in zip(axes, [("AILF", ailf), ("TRAM", tram)]):
    # Explode comma-separated outcomes into individual values
    outcomes = (df["Outcomes"].fillna("Unknown")
                .str.replace(r"\n", ",", regex=True)
                .str.split(",")
                .explode()
                .str.strip()
                .replace("", "Unknown"))
    top = outcomes.value_counts().head(7)
    ax.barh(top.index[::-1], top.values[::-1],
            color=colours[:len(top)], edgecolor="white")
    ax.set_title(name, fontsize=13, fontweight="bold", color=C_DARK)
    ax.set_xlabel("Count")
    for i, (v, lbl) in enumerate(zip(top.values[::-1], top.index[::-1])):
        ax.text(v + 50, i, f"{v:,}", va="center", fontsize=9, color=C_DARK)

plt.tight_layout()
save_and_show(fig, "05_outcome_distribution.png")

# ── 6.6  Top Primary Suspect Drugs ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Top 10 Primary Suspect Drugs",
             fontsize=14, fontweight="bold", color=C_DARK)

for ax, (name, df) in zip(axes, [("AILF", ailf), ("TRAM", tram)]):
    drug_counts = df["Primary Suspect Drugs"].value_counts().head(10)
    ax.barh(drug_counts.index[::-1], drug_counts.values[::-1],
            color=C_DEEP, edgecolor="white")
    ax.set_title(name, fontsize=13, fontweight="bold", color=C_DARK)
    ax.set_xlabel("Count")
    for i, v in enumerate(drug_counts.values[::-1]):
        ax.text(v + 30, i, f"{v:,}", va="center", fontsize=9, color=C_DARK)

plt.tight_layout()
save_and_show(fig, "06_top_drugs.png")

# ── 6.7  Dose Group Distribution ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Dose Group Distribution by Label",
             fontsize=14, fontweight="bold", color=C_DARK)

dose_order = [">100 mg", "≤100 mg", "Drug Abuse", "Unknown"]

for ax, (name, df) in zip(axes, [("AILF", ailf), ("TRAM", tram)]):
    dg = df.groupby(["dose_group", "label"]).size().unstack(fill_value=0)
    dg.columns = ["Negative", "Positive"]
    dg = dg.reindex([d for d in dose_order if d in dg.index])
    dg.plot(kind="bar", ax=ax, color=[C_MINT, C_DEEP], edgecolor="white", width=0.65)
    ax.set_title(name, fontsize=13, fontweight="bold", color=C_DARK)
    ax.set_xlabel("Dose Category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=15, labelsize=10)
    ax.legend(title="Label", fontsize=10)

plt.tight_layout()
save_and_show(fig, "07_dose_distribution.png")

# ── 6.8  Sentence / Sequence Length Distribution ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Template Sentence Length Distribution (Tokens)",
             fontsize=14, fontweight="bold", color=C_DARK)

for ax, (name, df) in zip(axes, [("AILF", ailf), ("TRAM", tram)]):
    seq = df["seq_len"].dropna()
    ax.hist(seq, bins=60, color=C_DEEP, alpha=0.85, edgecolor="white")
    ax.axvline(seq.mean(), color=C_ORANGE, linewidth=2, linestyle="--",
               label=f"Mean: {seq.mean():.1f}")
    ax.axvline(seq.median(), color=C_MINT, linewidth=2, linestyle="-.",
               label=f"Median: {seq.median():.1f}")
    ax.set_title(f"{name}  (μ={seq.mean():.1f} ± {seq.std():.1f})",
                 fontsize=13, fontweight="bold", color=C_DARK)
    ax.set_xlabel("Sequence length (words)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=10)

plt.tight_layout()
save_and_show(fig, "08_sequence_length.png")

# ── 6.9  Positive vs Negative ratio comparison (summary bar) ─────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
fig.suptitle("Dataset Size & Class Balance",
             fontsize=14, fontweight="bold", color=C_DARK)

datasets = ["AILF", "TRAM"]
pos_counts = [ailf["label"].sum(),       tram["label"].sum()]
neg_counts = [(ailf["label"]==0).sum(),  (tram["label"]==0).sum()]

x = np.arange(len(datasets))
w = 0.35
r1 = ax.bar(x - w/2, neg_counts, w, label="Negative", color=C_MINT, edgecolor="white")
r2 = ax.bar(x + w/2, pos_counts, w, label="Positive", color=C_DEEP, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=13)
ax.set_ylabel("Number of Reports")
ax.legend(fontsize=11)
for bar in list(r1) + list(r2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 100, f"{h:,}",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color=C_DARK)
ax.set_ylim(0, max(neg_counts) * 1.2)
plt.tight_layout()
save_and_show(fig, "09_dataset_summary.png")

# ── 6.10  Missing Values Heatmap ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Missing / Empty Value Rate (%) per Feature",
             fontsize=14, fontweight="bold", color=C_DARK)

check_cols = ["Primary Suspect Drugs", "Adverse Events", "Outcomes",
              "Gender", "Age", "Dose", "Indications"]

for ax, (name, df) in zip(axes, [("AILF", ailf), ("TRAM", tram)]):
    miss_pct = pd.Series({
        c: (df[c].isnull() | (df[c].fillna("").astype(str).str.strip() == "")).mean() * 100
        for c in check_cols if c in df.columns
    }).sort_values(ascending=False)
    colours_miss = [C_ORANGE if v > 30 else C_DEEP if v > 5 else C_MINT
                    for v in miss_pct.values]
    ax.barh(miss_pct.index[::-1], miss_pct.values[::-1],
            color=colours_miss[::-1], edgecolor="white")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Missing %")
    ax.set_title(name, fontsize=13, fontweight="bold", color=C_DARK)
    for i, v in enumerate(miss_pct.values[::-1]):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)

plt.tight_layout()
save_and_show(fig, "10_missing_values.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 7 — Dataset Summary (matches dissertation proposal / PPT)")
print("=" * 70)

summary = pd.DataFrame({
    "Metric": [
        "Total reports (raw)",
        "Positive class",
        "Negative class",
        "Positive ratio",
        "Train set",
        "Dev set",
        "Test set",
        "Avg sentence length (words)",
        "Std sentence length",
        "% Female",
        "% Male",
        "% Age <18",
        "% Age 18–39",
        "% Age 40–64",
        "% Age ≥65",
    ],
    "AILF (Acute Liver Failure)": [
        f"{len(ailf):,}",
        f"{ailf['label'].sum():,}",
        f"{(ailf['label']==0).sum():,}",
        f"{ailf['label'].mean():.2f}",
        f"{len(ailf_train):,}",
        f"{len(ailf_dev):,}",
        f"{len(ailf_test):,}",
        f"{ailf['seq_len'].mean():.1f}",
        f"{ailf['seq_len'].std():.1f}",
        f"{(ailf['gender_clean']=='Female').mean()*100:.1f}%",
        f"{(ailf['gender_clean']=='Male').mean()*100:.1f}%",
        f"{(ailf['age_group']=='<18').mean()*100:.1f}%",
        f"{(ailf['age_group']=='18–39').mean()*100:.1f}%",
        f"{(ailf['age_group']=='40–64').mean()*100:.1f}%",
        f"{(ailf['age_group']=='≥65').mean()*100:.1f}%",
    ],
    "TRAM (Tramadol Mortality)": [
        f"{len(tram):,}",
        f"{tram['label'].sum():,}",
        f"{(tram['label']==0).sum():,}",
        f"{tram['label'].mean():.2f}",
        f"{len(tram_train):,}",
        f"{len(tram_dev):,}",
        f"{len(tram_test):,}",
        f"{tram['seq_len'].mean():.1f}",
        f"{tram['seq_len'].std():.1f}",
        f"{(tram['gender_clean']=='Female').mean()*100:.1f}%",
        f"{(tram['gender_clean']=='Male').mean()*100:.1f}%",
        f"{(tram['age_group']=='<18').mean()*100:.1f}%",
        f"{(tram['age_group']=='18–39').mean()*100:.1f}%",
        f"{(tram['age_group']=='40–64').mean()*100:.1f}%",
        f"{(tram['age_group']=='≥65').mean()*100:.1f}%",
    ],
})
print(summary.to_string(index=False))
summary.to_csv("eda_output/dataset_summary.csv", index=False)
print("\n  Saved → eda_output/dataset_summary.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  EXPORT PREPROCESSED DATASETS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 8 — Exporting Preprocessed Files")
print("=" * 70)

EXPORT_COLS = ["sentence", "label", "gender_clean", "age_years",
               "age_group", "dose_group", "seq_len",
               "Primary Suspect Drugs", "Adverse Events", "Outcomes",
               "Indications"]

for tag, train_, dev_, test_ in [
    ("ailf", ailf_train, ailf_dev, ailf_test),
    ("tram", tram_train, tram_dev, tram_test),
]:
    for split_name, split_df in [("train", train_), ("dev", dev_), ("test", test_)]:
        out_cols = [c for c in EXPORT_COLS if c in split_df.columns]
        path = f"eda_output/{tag}_{split_name}.csv"
        split_df[out_cols].to_csv(path, index=False)
        print(f"  Saved → {path}  ({len(split_df):,} rows)")

print("\n" + "=" * 70)
print("  ✅  EDA COMPLETE  —  All outputs saved to ./eda_output/")
print("=" * 70)
print("""
Files generated:
  eda_output/
  ├── 01_class_distribution.png
  ├── 02_gender_distribution.png
  ├── 03_age_group_distribution.png
  ├── 04_age_histogram.png
  ├── 05_outcome_distribution.png
  ├── 06_top_drugs.png
  ├── 07_dose_distribution.png
  ├── 08_sequence_length.png
  ├── 09_dataset_summary.png
  ├── 10_missing_values.png
  ├── dataset_summary.csv
  ├── ailf_train.csv / ailf_dev.csv / ailf_test.csv
  └── tram_train.csv / tram_dev.csv / tram_test.csv
""")