"""
Phi-4 (microsoft/phi-4, ~14.7B) QLoRA driver -- true 5-fold CV, one L40.
r8 / "paper" config only.

Purpose in the suite: the general-purpose counterpart to Gemma4-12B at matched
scale, and a different PRETRAINING-DATA axis -- Phi is trained predominantly on
curated synthetic "textbook-quality" data where Gemma and Qwen are web-scale.

The plain checkpoint, NOT Phi-4-reasoning or -reasoning-plus. Reasoning
post-training teaches the model to emit long chain-of-thought before answering;
this task truncates at 128 tokens and pools the final position, so there is no
generation for that training to help, and it is a stage Gemma4-12B never had.

MIT licensed, ungated, text-only. No HF_TOKEN, no licence acceptance -- the
simplest model to run in the whole suite.

ARCHITECTURE NOTE -- like all Phi models this fuses Q/K/V into `qkv_proj`, so
the engine falls back from ["q_proj","v_proj"] to the fused projection and
therefore also adapts K. Deviation from the paper config; note it in write-up.

GEOMETRY -- 8 x 4 with NO gradient checkpointing, matching the Gemma4-12B
driver. Effective batch 32.

MEMORY -- the Gemma 12B at this geometry peaked at 24.2 GiB on fold 10. Phi-4
is slightly larger in layers (40 vs 48 is fewer, but hidden 5120 and
intermediate 17920) yet has a far smaller vocabulary (100,352 vs 262,144), so
its fp32 embedding is 1.91 GiB against 3.77. Expect a peak in the low-to-mid
20s GiB. Gradient checkpointing stays OFF; flip it with GRAD_CKPT=1 if a probe
says otherwise.

Two lessons from the Gemma runs are already baked in:
  * one process per fold (SLURM loop) -- the fp32 embedding survived
    `del model` and accumulated 3.77 GiB per fold on jobs 1262603-06
  * per_device_eval_batch_size is separate from the train batch -- pinning them
    together made the 31B evaluate one sample at a time, ~26% of wall clock

Expected trainable params at r8 (qkv_proj + score head):
    40 layers x 8*(5120 + 7680) + 5120*2 = 4,106,240

  python phi4_14b_peft_train.py --dataset <ds> --config paper
"""
import gemma_large_peft_src as src

MODEL_SPEC = {
    "key": "phi4_14b",
    "model_id": "microsoft/phi-4",
    "loaders": ("AutoModelForCausalLM",),
    "placement": "single",        # 14B at 4-bit fits directly; no streaming needed
    "gpu_max_mem": "40GiB",       # unused when placement == "single"
    "grad_checkpointing": False,
    "train_bs": 8,
    "accum": 4,                   # 8 x 4 = 32, same as Gemma4-12B
    "eval_bs": 16,
}

if __name__ == "__main__":
    src.main(MODEL_SPEC)
