"""
MediPhi (microsoft/MediPhi, 3.8B) QLoRA driver -- true 5-fold CV, one L40.
r8 / "paper" config only.

Purpose in the suite: the medically-adapted counterpart to MedGemma-1.5-4B at
matched scale. MediPhi is Phi-3.5-mini-instruct with five domain experts
(PubMed, Clinical, MedWiki, Guidelines, MedCode) merged via BreadCrumbs. The
unified merge is used rather than a single expert, because MedGemma-1.5 is a
broad medical generalist too -- comparing it to a single-corpus expert would be
a category mismatch. MediPhi-Instruct is deliberately NOT used: its MediFlow
SFT+DPO stage targets generative clinical tasks with JSON output, which cannot
transfer to a pooled linear classifier and has no counterpart on the Gemma side.

MIT licensed, ungated, text-only. No HF_TOKEN, no licence acceptance.

ARCHITECTURE NOTE -- Phi fuses Q, K and V into one `qkv_proj`, so the paper
config's ["q_proj","v_proj"] matches nothing. The engine detects this and falls
back to `qkv_proj`, which also adapts K. That is a genuine deviation from the
paper LoRA config and belongs in the write-up. Watch for the "FUSED ATTENTION"
warning in the log on fold 10 -- it confirms the fallback fired.

GEOMETRY -- 16 x 2, matching MedGemma-1.5-4B and Gemma3-4B exactly (those runs
used per_device_train_batch_size=16, gradient_accumulation_steps=2,
per_device_eval_batch_size=16, no gradient checkpointing). Effective batch 32.

MEMORY -- comfortable. Weights ~2.5 GiB at 4-bit; activations at 16 x ~97
tokens over 32 layers of hidden 3072 / intermediate 8192 come to ~4.3 GiB, so
peak should land near 8 GiB of the L40's 44.4. The fp32 embedding that leaked
3.77 GiB per fold on the Gemma 12B is only 0.37 GiB here (vocab 32,064 vs
262,144) -- and the SLURM script runs one process per fold regardless, so it
cannot accumulate at all.

Expected trainable params at r8 (qkv_proj + score head):
    32 layers x 8*(3072 + 9216) + 3072*2 = 3,151,872

  python mediphi_4b_peft_train.py --dataset <ds> --config paper
"""
import gemma_large_peft_src as src

MODEL_SPEC = {
    "key": "mediphi_4b",
    "model_id": "microsoft/MediPhi",
    "loaders": ("AutoModelForCausalLM",),   # text-only; no vision stack
    "placement": "auto",                    # matches the existing 4B runs
    "gpu_max_mem": "40GiB",
    "grad_checkpointing": False,            # 3.8B has ample room; ~35% faster
    "train_bs": 16,
    "accum": 2,                             # 16 x 2 = 32, same as MedGemma-1.5-4B
    "eval_bs": 16,
}

if __name__ == "__main__":
    src.main(MODEL_SPEC)
