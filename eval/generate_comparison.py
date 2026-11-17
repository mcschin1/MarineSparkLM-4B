#!/usr/bin/env python3
"""Generate answers to the held-out comparison eval set under 4 conditions:
base (no adapter), continuation adapter, instruction adapter, instruction
adapter + RAG. Used to compare MarineSparkLM-4B against the plain base model
and its own earlier (continuation-only) version.

Usage: python eval/generate_comparison.py
"""
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[0] / "package" / "MarineSparkLM"))

BASE_MODEL_DIR = ROOT / "oceangpt_base"
EVAL_SET = ROOT / "eval" / "comparison_eval_set.jsonl"
OUT = ROOT / "eval" / "comparison_outputs.jsonl"

CONDITIONS = [
    ("base", None, False),
    ("continuation_adapter", ROOT / "checkpoints" / "oceangpt_lora" / "best_adapter", False),
    ("instruction_adapter", ROOT / "checkpoints" / "oceangpt_lora_instruct" / "best_adapter", False),
    ("instruction_adapter_rag", ROOT / "checkpoints" / "oceangpt_lora_instruct" / "best_adapter", True),
]


def main():
    eval_rows = [json.loads(l) for l in open(EVAL_SET)]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

    retrieve = None
    out_f = open(OUT, "w")
    for name, adapter_dir, use_rag in CONDITIONS:
        if use_rag and retrieve is None:
            from rag import retrieve as _retrieve
            retrieve = _retrieve

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_DIR, dtype=torch.bfloat16, device_map="cuda",
        )
        if adapter_dir is not None:
            model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()

        for row in eval_rows:
            prompt = row["question"]
            augmented = prompt
            if use_rag:
                hits = retrieve(prompt, k=3)
                if hits:
                    ctx = "\n".join(f"[{h['theme']}] {h['text'][:300]}" for h in hits)
                    augmented = f"Source passages:\n{ctx}\n\n{prompt}"

            messages = [{"role": "user", "content": augmented}]
            inputs = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=150, do_sample=False,
                    repetition_penalty=1.2, no_repeat_ngram_size=3,
                    pad_token_id=tok.eos_token_id,
                )
            text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            out_f.write(json.dumps({
                "condition": name, "theme": row["theme"], "doc_id": row["doc_id"],
                "question": prompt, "reference_answer": row["reference_answer"],
                "generated": text.strip(),
            }) + "\n")
            out_f.flush()
            print(f"[{name}] {row['theme']}: {prompt[:60]}...", flush=True)

        del model
        torch.cuda.empty_cache()

    out_f.close()
    print(f"Done. Wrote {len(eval_rows) * len(CONDITIONS)} generations to {OUT}")


if __name__ == "__main__":
    main()
