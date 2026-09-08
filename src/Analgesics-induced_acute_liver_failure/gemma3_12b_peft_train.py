"""
Gemma3-12B (google/gemma-3-12b-it) QLoRA driver -- true 5-fold CV, one L40.

Runs in venv_gemma (the debugged Gemma3/MedGemma stack, older transformers, so
the loader falls back to torch_dtype= automatically).

Why this geometry: the 3 Sep run used per-device batch 16 with gradient
checkpointing off and peaked at ~44.3 GiB of the L40's 44.42 GiB usable. Fold
10 survived on luck -- DataCollatorWithPadding pads each batch to its own
longest sequence, so peak memory is a lottery per step -- and fold 21 drew a
long batch on step 1 and died. Batch 8 x accum 4 is the same effective batch of
32 with roughly half the activation footprint.

  python gemma3_12b_peft_train.py --dataset <ds> --config paper
"""
import gemma_large_peft_src as src

MODEL_SPEC = {
    "key": "gemma3_12b",
    "model_id": "google/gemma-3-12b-it",
    "loaders": ("AutoModelForImageTextToText", "Gemma3ForConditionalGeneration",
                "AutoModelForCausalLM"),
    "placement": "single",        # 12B 4-bit fits on the card; no streaming needed
    "gpu_max_mem": "40GiB",       # unused when placement == "single"
    "grad_checkpointing": False,  # 12B has room without it -> ~35% faster
    "train_bs": 8,
    "accum": 4,                   # 8 x 4 = 32, unchanged effective batch
    "eval_bs": 16,
}

if __name__ == "__main__":
    src.main(MODEL_SPEC)
