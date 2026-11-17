#!/usr/bin/env python3
"""Catastrophic-forgetting check: does the instruction adapter lose general
knowledge relative to the base model? Scores base model vs. instruction
adapter on the same random MMLU subset (multiple choice, auto-scored, no
manual rubric needed).

Usage: python mmlu_forgetting_check.py [--n 200] [--seed 42]
"""
import argparse
import random
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_DIR = ROOT / "oceangpt_base"
ADAPTER_DIR = ROOT / "checkpoints" / "oceangpt_lora_instruct" / "best_adapter"

LETTERS = ["A", "B", "C", "D"]


def format_prompt(item):
    choices = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(item["choices"]))
    return (f"Answer this multiple-choice question. Respond with only the letter "
            f"of the correct choice.\n\n{item['question']}\n{choices}\n\nAnswer:")


def score_model(model, tok, items, device):
    # Standard MMLU methodology (Hendrycks et al.): compare the model's next-token
    # logit for each choice letter directly, rather than generating free text and
    # parsing it -- avoids ambiguity when the model reasons before answering.
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    correct = 0
    for item in items:
        messages = [{"role": "user", "content": format_prompt(item)}]
        inputs = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
        next_token_logits = out.logits[0, -1]
        letter_logits = [next_token_logits[tid].item() for tid in letter_ids]
        pred_idx = max(range(4), key=lambda i: letter_logits[i])
        if pred_idx == item["answer"]:
            correct += 1
    return correct, len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset("cais/mmlu", "all", split="test")
    random.seed(args.seed)
    idxs = random.sample(range(len(ds)), args.n)
    items = [ds[i] for i in idxs]

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

    print(f"Scoring base model on {args.n} MMLU questions...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR, dtype=torch.bfloat16, device_map=device)
    base_model.eval()
    base_correct, n = score_model(base_model, tok, items, device)
    print(f"Base model: {base_correct}/{n} = {base_correct/n:.3f}")
    del base_model
    torch.cuda.empty_cache()

    print(f"Scoring instruction adapter on the same {args.n} questions...")
    adapted_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR, dtype=torch.bfloat16, device_map=device)
    adapted_model = PeftModel.from_pretrained(adapted_model, ADAPTER_DIR)
    adapted_model.eval()
    adapted_correct, n = score_model(adapted_model, tok, items, device)
    print(f"Instruction adapter: {adapted_correct}/{n} = {adapted_correct/n:.3f}")

    print(f"\nDelta: {adapted_correct - base_correct:+d} questions ({(adapted_correct-base_correct)/n:+.3f} accuracy)")


if __name__ == "__main__":
    main()
