# BioMistral-7B QLoRA (PEFT) — 5 configs × 2 datasets, 20-fold CV (encoder-style)

Single-GPU QLoRA fine-tuning of BioMistral-7B, reusing the base paper's Med-LLaMA PEFT
recipe (4-bit nf4 + LoRA + `paged_adamw_8bit`). ENCODER STYLE: **one job trains all 20
folds** on one GPU, so the queue is tidy — **10 plain jobs**, not 200 array tasks.
No DeepSpeed / ZeRO-3 / NCCL / P2P.

## The five configs

| config | rank r | alpha | target modules | ~trainable params |
|--------|--------|-------|----------------|-------------------|
| `paper`| 8      | 16    | q_proj, v_proj | ~3.4 M (exact paper repro) |
| `r16`  | 16     | 32    | all-linear     | ~42 M |
| `r64`  | 64     | 128   | all-linear     | ~168 M |
| `r128` | 128    | 256   | all-linear     | ~336 M |
| `r256` | 256    | 512   | all-linear     | ~671 M |

Each config writes to `cross_val_biomistral_<config>_temp/`, so `run_evaluation.py`
treats the five as separate models automatically.

## Files

```
src/<dataset>/biomistral_peft_train.py       # driver: loops all 20 folds, RESUMABLE
src/<dataset>/biomistral_peft_train_src.py   # train/predict/calibration helper
env/requirements_peft.txt                    # peft + bitsandbytes
env/setup_env_peft.sh                        # adds them to venv_biomistral
slurm/biomistral_peft.slurm                  # single-GPU job, no array, 48h walltime
submit_biomistral_peft.sh                    # fires all 10 jobs
```

## Run order

**One-time (login node):**
```bash
cd $SHARE/$USER/biocausal_cluster
bash env/setup_env_peft.sh          # adds peft + bitsandbytes to venv_biomistral
```

**Fire all 10 jobs (each runs its 20 folds):**
```bash
bash submit_biomistral_peft.sh you@warwick.ac.uk
```

**Or one config at a time:**
```bash
sbatch slurm/biomistral_peft.slurm Analgesics-induced_acute_liver_failure paper
```

## Monitoring — just 10 lines now

```bash
squeue -u $USER

# progress — each cell should reach 20 (updates live as folds finish)
for d in Analgesics-induced_acute_liver_failure Tramadol-related_mortalities; do
  for c in paper r16 r64 r128 r256; do
    echo -n "$d $c: "; ls dat/$d/proc/cross_val_biomistral_${c}_temp/df_res*.csv 2>/dev/null | wc -l
  done
done

# live log of a running config
tail -f outputs/biomistral_<config>/<dataset>/run_logs/*.log
```

## Resumable — walltime safety

Each job trains 20 folds sequentially (BioMistral folds are ~1-2h each, so 20 folds can
approach the 48h walltime). The driver **skips folds whose df_res already exists**, so if
a job is killed at walltime part-way through, just **resubmit the same line** and it
continues from where it stopped:
```bash
sbatch slurm/biomistral_peft.slurm Tramadol-related_mortalities r256   # resumes, skips done folds
```

## Evaluate

```bash
python run_evaluation.py --root "$(pwd)"
```
All five PEFT configs appear as separate models (`biomistral_paper_temp`, … `biomistral_r256_temp`,
plus `_cal` variants) alongside the encoders and XGBoost.

## Notes

- **Classification head is trained** (`modules_to_save=["score"]`) — the paper omitted this.
- **max_length=128** matches the rest of the study (paper used 480 for Med-LLaMA; change `MAX_LEN`).
- **lr=2e-4**, **effective batch ≈32** (per_device 8 × accum 4). Override via driver flags if needed.
- **No prefetch** — BioMistral-7B is already cached; 4-bit quantizes on load.
- **Reuses for Med-LLaMA-8B** by changing `MODEL_ID`.
