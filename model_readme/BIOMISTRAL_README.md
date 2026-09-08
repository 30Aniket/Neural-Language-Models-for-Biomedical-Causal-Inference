# BioMistral-7B — full fine-tuning on Blythe (ZeRO-3, 20-fold CV)

Full-parameter fine-tuning of `BioMistral/BioMistral-7B` as a sequence classifier,
20-fold (5×5) cross-validation on both datasets, predictions-only, then one saved
checkpoint per dataset via "re-train the winner". Training/prediction/calibration
logic is identical to the encoder pipeline; only what a 7B decoder under DeepSpeed
ZeRO-3 forces was changed (see the headers in `biomistral_train_src.py` /
`biomistral_train.py`).

## Where the new files go (preserve this structure under your project root)

```
biocausal_cluster/
├── src/Analgesics-induced_acute_liver_failure/biomistral_train.py
├── src/Analgesics-induced_acute_liver_failure/biomistral_train_src.py
├── src/Tramadol-related_mortalities/biomistral_train.py        # identical, DEFAULT_DATASET differs
├── src/Tramadol-related_mortalities/biomistral_train_src.py    # identical copy
├── env/deepspeed_zero3.json          # ZeRO-3 bf16, no offload
├── env/requirements_biomistral.txt
├── env/setup_env_biomistral.sh
├── env/prefetch_biomistral.sh
├── slurm/biomistral.slurm            # 20-fold array, 5 concurrent, 3×L40 each
├── slurm/biomistral_best.slurm       # re-train winning fold, save 1 checkpoint
├── pick_best_fold.py                 # picks best fold by test AUC
└── submit_biomistral.sh              # submits both datasets
```

The `_src` module must sit in **each** dataset folder (the driver does
`import biomistral_train_src` from its own folder), exactly like the encoders.

## Prerequisite (already true for you)

`dat/<dataset>/proc/df_together.csv` and `dat/<dataset>/proc/split.csv` must exist for
both datasets. Nothing else is needed — no checkpoints, no XGBoost, no encoders.

## Run order

**On the login node (once):**

```bash
cd $SHARE/$USER/biocausal_cluster

# 1. build the isolated venv (torch 2.5 cu124 + transformers + deepspeed + accelerate)
bash env/setup_env_biomistral.sh

# 2. warm the model cache so the array jobs run offline and never race to download
bash env/prefetch_biomistral.sh
```

**Submit the 20-fold CV for both datasets:**

```bash
bash submit_biomistral.sh you@warwick.ac.uk      # email optional
```

This launches two arrays (one per dataset), each 20 tasks, **5 running at once**
(5 × 3 = 15 L40s = your GPU cap). Each task trains one fold on 3 L40s and writes
`dat/<dataset>/proc/cross_val_biomistral_temp/df_res<ij>.csv`. No checkpoints saved.

**After BOTH arrays finish (40 df_res files total), save one checkpoint per dataset:**

```bash
sbatch slurm/biomistral_best.slurm Analgesics-induced_acute_liver_failure
sbatch slurm/biomistral_best.slurm Tramadol-related_mortalities
```

Each re-trains the highest-AUC fold and saves it to
`outputs/biomistral/<dataset>/best_checkpoint/` (~15 GB, ~30 GB for both).

## Monitoring

```bash
# queue (array tasks show as <jobid>_<taskid>)
squeue -u $USER

# progress — count predictions written (each dataset -> 20)
for d in Analgesics-induced_acute_liver_failure Tramadol-related_mortalities; do
  echo -n "$d: "; ls dat/$d/proc/cross_val_biomistral_temp/df_res*.csv 2>/dev/null | wc -l
done

# live log of a running fold (rank-0 persistent log)
tail -f outputs/biomistral/<dataset>/run_logs/biomistral_<dataset>_<split>_<jobid>.log

# a fold's SLURM stdout/err
tail -f logs/bmi_<arrayjobid>_<taskid>.out
```

Watch the first fold's log for a normal DeepSpeed start:
- `model loaded: BioMistral/BioMistral-7B (7,...,... parameters, all trainable / full FT)`
- DeepSpeed ZeRO-3 init lines, then `loss` / `eval_loss` every 200 steps
- early stopping halts each fold once dev loss plateaus (patience 10).

## After it's done — evaluate

Download `dat/*/proc/cross_val_biomistral_temp/` alongside your other models, then
`run_evaluation.py` will discover `biomistral` automatically (it globs
`cross_val_*_temp`). No evaluation code changes needed.

---

## Assumptions / watch-items (flagged deliberately)

1. **GPU cap = 15.** The array uses `--array=0-19%5` (5 × 3 = 15 GPUs) and
   `--cpus-per-task=30` (5 × 30 = 150 ≤ your 160 CPU cap). If your cap is actually 3
   GPUs, change `%5` to `%1` (one fold at a time) — it will still complete, just serially.

2. **Node-local scratch must hold DeepSpeed's transient checkpoints.** ZeRO-3
   checkpoints (kept for `load_best_model_at_end`, bounded by `save_total_limit=1`) go
   to `$TMPDIR`/`/tmp` on the compute node's SSD, **not** `$SHARE`, and are deleted when
   the task ends. If a node's `/tmp` is too small and a fold dies writing a checkpoint,
   either point `FOLD_SCRATCH` at a `$SHARE/scratch/...` path in `biomistral.slurm`
   (uses shared quota transiently) or raise `save_steps`/`eval_steps` to checkpoint less
   often.

3. **Effective batch ≈ 36, not exactly 32.** 3 GPUs × `per_device_train_batch_size=2`
   × `gradient_accumulation_steps=6` = 36 (closest clean value to the encoders' 32 on
   3 GPUs). If a fold OOMs, drop `--per_device_train_batch_size` to 1 (raise accum to
   12 to keep the effective batch) in `biomistral.slurm`.

4. **lr = 5e-5 is faithful to the encoders but aggressive for 7B full FT.** If you see
   loss diverge / NaNs in the early steps of a fold, lower the learning rate (it's
   `learning_rate=5e-5` in `biomistral_train_src.py`, fed to DeepSpeed via "auto").

5. **Per-fold walltime.** `--time=48:00:00` (Blythe max). A single 7B fold is expected
   in single-digit hours; the 48h ceiling is generous headroom, not an expected runtime.

6. **This mirrors for Med-LLaMA-8B** by changing `MODEL_ID`, the job tags, and (if
   needed) a slightly larger memory margin — the driver is otherwise model-agnostic.
