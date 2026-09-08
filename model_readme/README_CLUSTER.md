# Biomedical Causal Inference — Cluster Guide (Step 1: ALBERT)

This package makes the 2026 InferBERT-style **ALBERT** training run on the **Blythe** HPC/GPU
cluster (SCRTP, Warwick). The model training/prediction/calibration **logic is unchanged** from
the repo (`hsdslab/biomedical-causal-inference`); the only additions are logging, saving of
learned weights, and persistence of training results so you can download them.

Blythe facts used here: login node `blythe.scrtp.warwick.ac.uk`, data node
`blythedata.scrtp.warwick.ac.uk`, GPU partition `gpu` with 3× Nvidia **L40** (48 GB) per node
(`--gres=gpu:lovelace_l40:N`, max 10 CPUs/GPU), flat module system, 48 h max runtime.

---

## Folder layout

```
biocausal_cluster/
├── README_CLUSTER.md
├── env/
│   ├── requirements_albert.txt      # pinned deps (transformers 4.41.2 etc.)
│   └── setup_env.sh                 # build the venv on the login node (run once)
├── data/
│   └── prepare_data.py              # arrange CSVs -> dat/<dataset>/proc/{df_together,split}.csv
├── slurm/
│   └── albert.slurm                 # GPU job: sbatch slurm/albert.slurm <dataset>
├── transfer/
│   ├── upload_to_cluster.sh         # LOCAL -> Blythe (rsync via blythedata)
│   └── download_from_cluster.sh     # Blythe -> LOCAL (weights, logs, results)
└── src/
    ├── Analgesics-induced_acute_liver_failure/
    │   ├── albert_train.py          # repo logic + logging/saving  (default dataset = AILF)
    │   └── albert_train_src.py      # UNCHANGED from repo (byte-identical)
    └── Tramadol-related_mortalities/
        ├── albert_train.py          # same, default dataset = TRAM
        └── albert_train_src.py      # UNCHANGED from repo
```

Put your 4 CSVs (`ailf_df_together.csv`, `ailf_split.csv`, `tram_df_together.csv`,
`tram_split.csv`) in the **project root** (next to this README) before uploading.

---

## One-time prerequisites (local machine)

You need an SSH key registered with SCRTP and 2FA set up (see the Blythe "Creating and using
ssh keys" docs). Test it: `ssh <user>@blythe.scrtp.warwick.ac.uk`. If your key has a non-default
name, `export SSH_KEY=~/.ssh/yourkey` before running the transfer scripts.

---

## Step-by-step

### 1. Prepare the data locally (optional but recommended)
From the project root:
```bash
python data/prepare_data.py --src .
```
This creates `dat/<dataset>/proc/df_together.csv` and `split.csv` for both datasets. (You can
skip this and run it on the cluster instead — either works, the paths are identical.)

### 2. Upload code + data to Blythe
From the project root, on your **local** machine:
```bash
bash transfer/upload_to_cluster.sh <scrtp_username>
```
This rsyncs everything to `~/biocausal_cluster/` on Blythe (via `blythedata`), excluding
caches/outputs.

### 3. Build the Python environment (once, on the login node)
```bash
ssh <scrtp_username>@blythe.scrtp.warwick.ac.uk
cd ~/biocausal_cluster
mkdir -p logs                     # SLURM needs this dir to exist for job logs
module av Python                  # check the exact Python module name available
# if it differs from the default, edit PYTHON_MODULE in env/setup_env.sh AND slurm/albert.slurm
bash env/setup_env.sh
```
This creates `~/biocausal_cluster/venv` with torch 2.5 (CUDA 12.4) + the pinned stack, and
prints a verification block. (`torch.cuda.is_available()` is `False` on the login node — normal,
there's no GPU there; it will be `True` inside the GPU job.)

### 4. If you skipped step 1, prepare data on the cluster now
```bash
cd ~/biocausal_cluster
module load Python/3.11.5-GCCcore-13.2.0 && source venv/bin/activate
python data/prepare_data.py --src .
```

### 5. Submit the ALBERT training job(s)
From `~/biocausal_cluster`:
```bash
sbatch slurm/albert.slurm Analgesics-induced_acute_liver_failure
# and/or
sbatch slurm/albert.slurm Tramadol-related_mortalities
```
Each job runs the **full 20-split cross-validation** (5×5 minus the diagonal), exactly as in the
repo. `sbatch` prints a job ID.

### 6. Monitor
```bash
squeue -u <scrtp_username>                 # queue / running status
tail -f logs/albert-<jobid>.out            # live stdout (prints + logging)
cat    logs/albert-<jobid>.err             # errors, if any
```

### 7. Download results to your local machine
On your **local** machine, from the project root:
```bash
bash transfer/download_from_cluster.sh <scrtp_username> Analgesics-induced_acute_liver_failure
```
Everything lands under `./downloaded/`.

---

## What gets produced (and downloaded)

For each dataset, under `outputs/albert/<dataset>/`:

| Path | Contents |
|---|---|
| `weights/split_<dev><test>/` | Learned ALBERT weights per CV split: `model.safetensors`, `config.json`, tokenizer files. Reloadable directly with `from_pretrained(...)`. Add `--save_state_dict` to also emit `pytorch_state_dict.pt`. |
| `train_logs/train_curve_<dev><test>.csv` | Per-split training curve (loss, eval_loss, lr, step) from `trainer.state.log_history`. |
| `run_logs/albert_<dataset>_<jobid>.log` | Full text log (all prints + progress) for the whole run. |

Plus the **prediction CSVs** the downstream causal-inference / metrics code consumes, unchanged
in location:
```
dat/<dataset>/proc/cross_val_albert_temp/df_res<dev><test>.csv   # cols: albert_temp, albert_temp_cal, label
```
And the SLURM captures in `logs/albert-<jobid>.out` / `.err`.

## Reloading weights locally for evaluation
```python
from transformers import AlbertForSequenceClassification, AutoTokenizer
p = "downloaded/outputs/albert/Analgesics-induced_acute_liver_failure/weights/split_13"
model = AlbertForSequenceClassification.from_pretrained(p)
tok   = AutoTokenizer.from_pretrained(p)
```
Then run the repo's local evaluation notebooks (`extra_evaluation_codes/`) as-is.

---

## Notes & knobs
- **Weights format:** safetensors (via `trainer.save_model`) is the recommended format — safe,
  fast, and directly reloadable. `pytorch_state_dict.pt` is available via `--save_state_dict` if
  you specifically want a raw torch/pickle-style file.
- **Runtime:** `--time=24:00:00` is generous. The repo uses `max_steps=30000` with early
  stopping (patience 10 @ eval every 200 steps), so most splits stop well before that. If a job
  ever approaches 48 h, split the work (run one dataset per job) rather than raising the limit.
- **Model is loaded once and reused across splits** — this is the repo's original behaviour and
  is preserved deliberately. Do not "fix" it unless you intend to change the experiment.
- **CPUs/GPU:** the job requests 1 L40 + 10 CPUs (the max allowed per GPU on Blythe).
- **Don't set `CUDA_VISIBLE_DEVICES` manually** — SLURM sets it from the GRes request.
- To also generate the optional raw state dict, edit the `srun` line in `slurm/albert.slurm` to
  `srun python albert_train.py --dataset "${DATASET}" --save_state_dict`.
