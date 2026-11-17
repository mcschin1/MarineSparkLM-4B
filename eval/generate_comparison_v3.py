#!/usr/bin/env python3
"""Expanded comparison harness, round 2: fixes the selection-leakage issue
(this round's eval set -- fresh_test_set.jsonl -- is built from corpus
passages never present in qa_seed.jsonl, so it was never seen in training,
validation, or checkpoint selection of any adapter), and adds 3 new
conditions: base+RAG (isolates whether RAG helps independent of instruction
tuning), the synthetic-only adapter (isolates whether the 114 hand-authored
pairs matter), and instruction_adapter+dense-RAG (embedding retrieval vs.
TF-IDF).

Usage: python eval/generate_comparison_v3.py
"""
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = ROOT.parents[0] / "package" / "MarineSparkLM"
sys.path.insert(0, str(PKG_DIR))

BASE_MODEL_DIR = ROOT / "oceangpt_base"
EVAL_SET = ROOT / "eval" / "fresh_test_set.jsonl"
OUT = ROOT / "eval" / "comparison_outputs_v3.jsonl"

CKPT = ROOT / "checkpoints"
# (name, adapter_dir or None, retrieval mode: None / "tfidf" / "dense")
CONDITIONS = [
    ("base", None, None),
    ("continuation_adapter", CKPT / "oceangpt_lora" / "best_adapter", None),
    ("instruction_adapter", CKPT / "oceangpt_lora_instruct" / "best_adapter", None),
    ("instruction_adapter_rag_tfidf", CKPT / "oceangpt_lora_instruct" / "best_adapter", "tfidf"),
    ("base_rag_tfidf", None, "tfidf"),
    ("instruction_adapter_synthetic_only", CKPT / "oceangpt_lora_instruct_synthetic_only" / "best_adapter", None),
    ("instruction_adapter_rag_dense", CKPT / "oceangpt_lora_instruct" / "best_adapter", "dense"),
]


def main():
    eval_rows = [json.loads(l) for l in open(EVAL_SET)]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

    retrieve_tfidf = None
    retrieve_dense = None
    out_f = open(OUT, "w")
    for name, adapter_dir, retrieval in CONDITIONS:
        if retrieval == "tfidf" and retrieve_tfidf is None:
            from rag import retrieve as retrieve_tfidf
        if retrieval == "dense" and retrieve_dense is None:
            from rag_dense import retrieve_dense as retrieve_dense

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_DIR, dtype=torch.bfloat16, device_map="cuda",
        )
        if adapter_dir is not None:
            model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()

        for row in eval_rows:
            prompt = row["question"]
            augmented = prompt
            if retrieval == "tfidf":
                hits = retrieve_tfidf(prompt, k=3, exclude_doc_ids={row["doc_id"]})
            elif retrieval == "dense":
                hits = retrieve_dense(prompt, k=3, exclude_doc_ids={row["doc_id"]})
            else:
                hits = None
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
