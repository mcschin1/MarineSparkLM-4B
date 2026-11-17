#!/usr/bin/env python3
"""Measures adapter load time, generation throughput, and peak GPU memory
for the instruction adapter, to back the "lightweight" claim with numbers
rather than just asserting it.
"""
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_DIR = ROOT / "oceangpt_base"
ADAPTER_DIR = ROOT / "checkpoints" / "oceangpt_lora_instruct" / "best_adapter"

PROMPT = "Explain the key factors affecting ship hull resistance and powering during preliminary ship design."


def main():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR, dtype=torch.bfloat16, device_map="cuda",
    )
    t_base_loaded = time.time()
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    t_adapter_loaded = time.time()

    base_load_s = t_base_loaded - t0
    adapter_load_s = t_adapter_loaded - t_base_loaded
    mem_after_load = torch.cuda.max_memory_allocated() / 1e9

    messages = [{"role": "user", "content": PROMPT}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")

    # warmup (first generate() includes kernel compilation / cache warmup, not representative)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    n_new = 200
    t_gen0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=n_new, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    t_gen1 = time.time()

    n_generated = out.shape[1] - inputs["input_ids"].shape[1]
    gen_time = t_gen1 - t_gen0
    tokens_per_sec = n_generated / gen_time
    mem_during_gen = torch.cuda.max_memory_allocated() / 1e9

    print(f"Base model load time:       {base_load_s:.2f} s")
    print(f"Adapter load time:          {adapter_load_s:.2f} s")
    print(f"Peak GPU memory (loaded):   {mem_after_load:.2f} GB")
    print(f"Peak GPU memory (gen):      {mem_during_gen:.2f} GB")
    print(f"Tokens generated:           {n_generated}")
    print(f"Generation time:            {gen_time:.2f} s")
    print(f"Throughput:                 {tokens_per_sec:.1f} tokens/sec")


if __name__ == "__main__":
    main()
