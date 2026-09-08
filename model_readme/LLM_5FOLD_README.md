# BioMistral-7B + Med-LLaMA-8B QLoRA — TRUE 5-fold CV (12 jobs)

Single-GPU QLoRA (4-bit nf4 + LoRA + paged_adamw_8bit), encoder style: one job runs
all 5 folds. 2 models x 3 configs x 2 datasets = 12 jobs. No DeepSpeed/NCCL/P2P.

## Configs — each rank at a stable, rank-appropriate learning rate

| config | r | alpha | targets | lr | dropout | ~trainable |
|--------|---|-------|---------|-----|---------|-----------|
| paper  | 8 | 16 | q/v | 2e-4 | 0.05 | ~3.4M (paper repro) |
| r64    | 64| 128| all-linear | 1e-4 | 0.10 | ~168M |
| r128   |128| 256| all-linear | 5e-5 | 0.10 | ~336M |

(The 20-fold sweep showed a fixed lr=2e-4 destabilises high-rank LoRA; here each rank
gets a lower lr so it trains cleanly and is judged on merit.)

TRUE 5-fold CV: tags 10/21/32/43/04 (each split tested once) — identical to the
encoders + XGBoost, so joint statistics are valid.

## Files
```
src/<ds>/biomistral_peft_train.py + _src.py     # BioMistral driver + helper
src/<ds>/med_llama_peft_train.py  + _src.py     # Med-LLaMA driver + helper
slurm/biomistral_peft.slurm                     # bmis_<rank>_<ds>, 48h, single GPU
slurm/med_llama_peft.slurm                      # mlm_<rank>_<ds>, 48h, HF token passthrough
submit_llms_5fold.sh                            # fires all 12 (or one model)
```
Reuses venv_biomistral (peft + bitsandbytes) and .hf_cache (both models already cached).

## Run
```bash
cd $SHARE/$USER/biocausal_cluster

# all 12 at once:
bash submit_llms_5fold.sh Aniket.Kumar@warwick.ac.uk

# or one model at a time (6 jobs each):
bash submit_llms_5fold.sh Aniket.Kumar@warwick.ac.uk biomistral
bash submit_llms_5fold.sh Aniket.Kumar@warwick.ac.uk med_llama
```

## Monitor (each cell -> 5)
```bash
for m in biomistral med_llama; do for c in paper r64 r128; do
  for d in Analgesics-induced_acute_liver_failure Tramadol-related_mortalities; do
    echo -n "$m $c $d: "; ls dat/$d/proc/cross_val_${m}_${c}_temp/df_res*.csv 2>/dev/null | wc -l
done;done;done
```

## Resumable
If a job hits 48h mid-run, resubmit that one line; it skips finished folds:
```bash
sbatch slurm/biomistral_peft.slurm Tramadol-related_mortalities r128
```

## Efficiency
SDPA attention + real batch 16 (effective batch 32) + dataloader workers.
At 5 folds each, expect roughly a few hours per job.

## Evaluate
Predictions land in cross_val_<model>_<config>_temp/ -> discovered automatically as
separate models by run_evaluation.py / eval_lora_configs.py alongside the encoders + XGBoost.
