#!/usr/bin/env python3
"""Validates the upload-RAG feature's core design claim (small 4-bit
backbone + strong retrieval ~= most of a larger pipeline's accuracy) by
simulating 20 single-document uploads: for each of 20 held-out questions
(2/theme, from fresh_test_set_v2.jsonl), the question's own corpus
passage is fed through the real upload_rag.py pipeline (dense retrieve ->
rerank -> 4-bit backbone), exactly as a user's upload would be. This is a
best-case retrieval scenario (the "document" trivially contains the
answer, no multi-document mixing or irrelevant content), not a test of
retrieval difficulty -- the point is to check generation quality on a
short, correctly-retrieved context, which is what the design actually
controls for.

Usage: python eval/generate_upload_rag_test.py
"""
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = ROOT.parents[0] / "package" / "MarineSparkLM"
sys.path.insert(0, str(PKG_DIR))
from upload_rag import build_index, retrieve  # noqa: E402

BASE_MODEL_DIR = ROOT / "oceangpt_base"
ADAPTER_DIR = ROOT / "checkpoints" / "oceangpt_lora_instruct" / "best_adapter"
ITEMS_PATH = ROOT / "eval" / "upload_rag_test_items.jsonl"
CHUNKS_PATH = PKG_DIR / "rag" / "chunks.jsonl"
OUT = ROOT / "eval" / "upload_rag_test_outputs.jsonl"


def main():
    items = [json.loads(l) for l in open(ITEMS_PATH)]
    doc_ids = {r["doc_id"] for r in items}
    passages = {}
    with open(CHUNKS_PATH) as f:
        for line in f:
            c = json.loads(line)
            if c["doc_id"] in doc_ids and c["doc_id"] not in passages:
                passages[c["doc_id"]] = c["text"]

    tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR, quantization_config=bnb_config, device_map="cuda",
    )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()

    out_f = open(OUT, "w")
    for i, row in enumerate(items):
        doc_text = passages[row["doc_id"]]
        # Each "upload" is one document, the real corpus passage this
        # question's reference answer was drawn from -- written to a temp
        # file so build_index exercises the real extract_text() path.
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(doc_text)
            tmp_path = f.name

        chunks, embeddings = build_index([tmp_path])
        hits = retrieve(row["question"], chunks, embeddings, k_dense=20, k_final=4)
        if hits:
            ctx = "\n".join(f"[{h['doc_id']}] {h['text'][:400]}" for h in hits)
            augmented = f"Source passages:\n{ctx}\n\n{row['question']}"
        else:
            augmented = row["question"]

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
            "condition": "upload_rag", "theme": row["theme"], "doc_id": row["doc_id"],
            "question": row["question"], "reference_answer": row["reference_answer"],
            "generated": text.strip(), "n_hits": len(hits),
        }) + "\n")
        out_f.flush()
        print(f"[{i+1}/{len(items)}] {row['theme']}/{row['doc_id']}: {len(hits)} hits", flush=True)

        Path(tmp_path).unlink()

    out_f.close()
    print(f"Done. Wrote {len(items)} generations to {OUT}")


if __name__ == "__main__":
    main()
