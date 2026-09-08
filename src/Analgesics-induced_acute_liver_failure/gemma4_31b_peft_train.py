"""
Gemma4-31B (google/gemma-4-31B) QLoRA driver -- true 5-fold CV, one L40.
This is the 20-30B scale point Dr Pergola asked for.

Runs in venv_gemma4. Gated: accept the licence on HF and export HF_TOKEN.

Placement: device_map="auto" + low_cpu_mem_usage + max_memory, because forcing
device_map={"":0} disabled streaming and OOM'd at load. Keep GPU_MAX_MEM at
40GiB -- nvidia-smi on the 4 Sep run showed 25153 MiB of 46068 MiB in use, so
nothing spilled to CPU and lowering the cap would only start forcing spill.

Geometry: 8 x 4 with eval batch 16 -- the SAME micro-batch geometry as every
other LLM in the study. BioMistral-7B, Med-LLaMA-8B, Qwen3.5-4B/9B, Gemma3-4B
and MedGemma all trained at per_device_train_batch_size=16,
gradient_accumulation_steps=2, per_device_eval_batch_size=16. The 1 x 32 /
eval-1 setting used on 3 Sep was the outlier, adopted to survive a load-time
OOM that turned out not to exist once device_map="auto" was in place.

Why 8 and not 16: measured on the real data, Temp_sentence tokenises to a mean
of ~53 tokens with only ~1% of records hitting the 128-token cap, so dynamic
padding to the batch maximum costs 1.63x more tokens at batch 8 and 1.84x at
batch 16. Batch 8 keeps the padding overhead moderate while cutting the number
of 4-bit dequantisation passes per optimiser step from 32 to 4.

Optional 30-minute confirmation before committing ~100 GPU-hours:

  python gemma4_31b_peft_train.py --dataset <ds> --probe 20 \
      --per_device_train_batch_size 8 --gradient_accumulation_steps 4

  python gemma4_31b_peft_train.py --dataset <ds> --config paper
"""
import gemma_large_peft_src as src

MODEL_SPEC = {
    "key": "gemma4_31b",
    "model_id": "google/gemma-4-31B",
    "loaders": ("AutoModelForImageTextToText", "AutoModelForCausalLM"),
    "placement": "auto",          # streaming loader; dense 31B loads fine this way
    "gpu_max_mem": "40GiB",       # proven; override with GPU_MAX_MEM if ever needed
    "grad_checkpointing": True,
    "train_bs": 8,
    "accum": 4,                   # 8 x 4 = 32, matching BioMistral/Med-LLaMA/Qwen/Gemma3-4B
    "eval_bs": 16,
}

if __name__ == "__main__":
    src.main(MODEL_SPEC)
