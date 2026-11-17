#!/usr/bin/env python3
"""Benchmarks a 4-bit quantized load of the instruction adapter against the
bf16 baseline: load time, peak memory, throughput, and whether quality
degrades on the fixed engineering prompts. Answers "does this still run on
a laptop-class GPU" without requiring one -- memory footprint generalizes.
"""
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_DIR = ROOT / "oceangpt_base"
ADAPTER_DIR = ROOT / "checkpoints" / "oceangpt_lora_instruct" / "best_adapter"

PROMPT = "Explain the key factors affecting ship hull resistance and powering during preliminary ship design."


def run(label, quantize):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

    t0 = time.time()
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_DIR, quantization_config=bnb_config, device_map="cuda",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_DIR, dtype=torch.bfloat16, device_map="cuda",
        )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    load_time = time.time() - t0
    mem_loaded = torch.cuda.max_memory_allocated() / 1e9

    messages = [{"role": "user", "content": PROMPT}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t_gen0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    gen_time = time.time() - t_gen0
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    mem_gen = torch.cuda.max_memory_allocated() / 1e9
    text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print(f"\n=== {label} ===")
    print(f"Load time:            {load_time:.2f} s")
    print(f"Peak memory (load):   {mem_loaded:.2f} GB")
    print(f"Peak memory (gen):    {mem_gen:.2f} GB")
    print(f"Tokens generated:     {n_new}")
    print(f"Throughput:           {n_new/gen_time:.1f} tokens/sec")
    print(f"Sample output:        {text[:200]}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run("bf16 (baseline)", quantize=False)
    run("4-bit NF4 (bitsandbytes)", quantize=True)
