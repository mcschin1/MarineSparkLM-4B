#!/usr/bin/env python3
"""Tier 3 item #9: the enlarged (100+) held-out comparison round. Same 4
core conditions as round 1 (base, continuation_adapter, instruction_adapter,
instruction_adapter + TF-IDF RAG) against fresh_test_set_v2.jsonl -- the
3 ablation conditions added in round 2 (base+RAG, synthetic-only, dense RAG)
already got clear null/negative results at n=20 and are not re-run here to
keep generation and auto-scoring cost proportionate to what a larger set
is actually meant to test: statistical power on the core instruction-tuning
claim, using the automatic metric instead of full hand-scoring (validated
via a hand-scored spot-check in score_auto_v4.py / compute_stats_v4.py).

Usage: python eval/generate_comparison_v4.py
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
EVAL_SET = ROOT / "eval" / "fresh_test_set_v2.jsonl"
OUT = ROOT / "eval" / "comparison_outputs_v4.jsonl"

CKPT = ROOT / "checkpoints"
CONDITIONS = [
    ("base", None, None),
    ("continuation_adapter", CKPT / "oceangpt_lora" / "best_adapter", None),
    ("instruction_adapter", CKPT / "oceangpt_lora_instruct" / "best_adapter", None),
    ("instruction_adapter_rag_tfidf", CKPT / "oceangpt_lora_instruct" / "best_adapter", "tfidf"),
]


def main():
    eval_rows = [json.loads(l) for l in open(EVAL_SET)]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

    retrieve_tfidf = None
    out_f = open(OUT, "w")
    for name, adapter_dir, retrieval in CONDITIONS:
        if retrieval == "tfidf" and retrieve_tfidf is None:
            from rag import retrieve as retrieve_tfidf

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_DIR, dtype=torch.bfloat16, device_map="cuda",
        )
        if adapter_dir is not None:
            model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()

        for i, row in enumerate(eval_rows):
            prompt = row["question"]
            augmented = prompt
            if retrieval == "tfidf":
                hits = retrieve_tfidf(prompt, k=3, exclude_doc_ids={row["doc_id"]})
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
            if (i + 1) % 20 == 0:
                print(f"[{name}] {i+1}/{len(eval_rows)}", flush=True)

        del model
        torch.cuda.empty_cache()

    out_f.close()
    print(f"Done. Wrote {len(eval_rows) * len(CONDITIONS)} generations to {OUT}")


if __name__ == "__main__":
    main()
