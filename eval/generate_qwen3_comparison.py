#!/usr/bin/env python3
"""Compares plain Qwen3-4B-Instruct-2507 (the generic instruct model
OceanGPT itself was built from) against OceanGPT base and our instruction
adapter, with and without RAG, on the fresh (leak-free) Evaluation B set.
Isolates whether OceanGPT's own ocean-domain pretraining matters at all, or
whether a generic instruct model of the same size does just as well.

Usage: python eval/generate_qwen3_comparison.py
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = ROOT.parents[0] / "package" / "MarineSparkLM"
sys.path.insert(0, str(PKG_DIR))

QWEN_DIR = ROOT / "qwen3_4b_instruct"
EVAL_SET = ROOT / "eval" / "fresh_test_set.jsonl"
OUT = ROOT / "eval" / "comparison_outputs_qwen3.jsonl"

CONDITIONS = [
    ("qwen3_instruct", False),
    ("qwen3_instruct_rag_tfidf", True),
]


def main():
    eval_rows = [json.loads(l) for l in open(EVAL_SET)]
    tok = AutoTokenizer.from_pretrained(QWEN_DIR)
    model = AutoModelForCausalLM.from_pretrained(QWEN_DIR, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    retrieve = None
    out_f = open(OUT, "w")
    for name, use_rag in CONDITIONS:
        if use_rag and retrieve is None:
            from rag import retrieve

        for row in eval_rows:
            prompt = row["question"]
            augmented = prompt
            if use_rag:
                hits = retrieve(prompt, k=3, exclude_doc_ids={row["doc_id"]})
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

    out_f.close()
    print(f"Done. Wrote {len(eval_rows) * len(CONDITIONS)} generations to {OUT}")


if __name__ == "__main__":
    main()
