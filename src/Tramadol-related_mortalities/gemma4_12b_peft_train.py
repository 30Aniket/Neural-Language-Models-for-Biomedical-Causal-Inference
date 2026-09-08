"""
Gemma4-12B (google/gemma-4-12B) QLoRA driver -- true 5-fold CV, one L40.

Runs in venv_gemma4 (transformers 5.16 / torch 2.14 / CUDA-13 wheels).
Gated: accept the licence on HF and export HF_TOKEN.

Same story as the Gemma3-12B driver: the 3 Sep run at per-device batch 16 with
checkpointing off sat at 99.8% of the card and fell over at the start of fold
21 on a 114 MB allocation. Batch 8 x accum 4 keeps the effective batch at 32.

  python gemma4_12b_peft_train.py --dataset <ds> --config paper
"""
import gemma_large_peft_src as src

MODEL_SPEC = {
    "key": "gemma4_12b",
    "model_id": "google/gemma-4-12B",
    "loaders": ("AutoModelForImageTextToText", "AutoModelForCausalLM"),
    "placement": "single",
    "gpu_max_mem": "40GiB",
    "grad_checkpointing": False,
    "train_bs": 8,
    "accum": 4,
    "eval_bs": 16,
}

if __name__ == "__main__":
    src.main(MODEL_SPEC)
