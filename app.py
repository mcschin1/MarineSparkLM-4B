#!/usr/bin/env python3
"""Web UI for the OceanGPT LoRA fine-tune (run_500m_v1). Run: python app.py
(then open the printed local URL).

Loads the base model and adapter straight from run_500m_v1 rather than
bundling an 8GB base model into this package -- run this from a checkout
that still has ../../run_500m_v1/ next to it.
"""
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from upload_rag import build_index, retrieve as retrieve_uploaded

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parents[1] / "run_500m_v1"
BASE_MODEL_DIR = RUN_DIR / "oceangpt_base"
ADAPTER_DIR = RUN_DIR / "checkpoints" / "oceangpt_lora_instruct" / "best_adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR, dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map=DEVICE,
)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

# Second, 4-bit model for the "upload your own documents" tab -- loaded
# lazily on first use of that tab, not at startup, since most sessions
# only use the main (bf16) marine-corpus tab. Same base+adapter, just
# quantized: validated in quantization_benchmark.py (8.3GB -> 3.1GB,
# comparable output quality), so retrieval quality (not model size) is
# what this tab's accuracy depends on.
_quantized_model = {}


def _get_quantized_model():
    if "model" not in _quantized_model:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        m = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_DIR, quantization_config=bnb_config, device_map=DEVICE,
        )
        m = PeftModel.from_pretrained(m, ADAPTER_DIR)
        m.eval()
        _quantized_model["model"] = m
    return _quantized_model["model"]

EXAMPLES = [
    "Explain the key factors affecting ship hull resistance and powering during preliminary ship design.",
    "Describe how added mass and wave radiation damping affect a floating offshore structure's motion response.",
    "Discuss the primary causes of fatigue damage in offshore jacket structures and how they are mitigated.",
    "Describe how a dynamic positioning system maintains vessel position using thruster allocation and feedback control.",
    "Explain the key design considerations for ROV thruster allocation and maneuverability in deep water operations.",
]


def run(prompt, max_new_tokens, temperature, top_k, repetition_penalty, no_repeat_ngram_size, use_rag, rag_k, retrieval_method):
    if not prompt.strip():
        return "", ""
    augmented = prompt
    retrieved_text = ""
    if use_rag:
        if retrieval_method == "Dense (embedding)":
            from rag_dense import retrieve_dense as retrieve
        else:
            from rag import retrieve
        hits = retrieve(prompt, k=int(rag_k))
        if hits:
            context_lines = [f"[{h['theme']}] {h['text'][:300]}" for h in hits]
            augmented = "Source passages:\n" + "\n".join(context_lines) + f"\n\n{prompt}"
            retrieved_text = "\n\n".join(
                f"[{h['score']:.3f}] {h['theme']}/{h['doc_id']}: {h['text'][:200]}" for h in hits
            )
        else:
            retrieved_text = "(no index match -- generating unprimed)"

    messages = [{"role": "user", "content": augmented}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=int(max_new_tokens), do_sample=True,
            temperature=float(temperature), top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, retrieved_text


def build_session_index(files):
    if not files:
        return None, "No files uploaded."
    chunks, embeddings = build_index([f.name for f in files])
    if not chunks:
        return None, "No extractable text found in the uploaded file(s)."
    n_docs = len({c["doc_id"] for c in chunks})
    return (chunks, embeddings), f"Indexed {len(chunks)} chunks from {n_docs} document(s). Ready to query."


def run_uploaded(prompt, session_data, max_new_tokens, temperature, top_k,
                  repetition_penalty, no_repeat_ngram_size, k_final):
    if not prompt.strip():
        return "", ""
    if session_data is None:
        return "Upload and index a document first.", ""
    chunks, embeddings = session_data
    hits = retrieve_uploaded(prompt, chunks, embeddings, k_dense=20, k_final=int(k_final))
    if not hits:
        return "No relevant passage found in the uploaded document(s) for this query.", ""
    context_lines = [f"[{h['doc_id']}] {h['text'][:400]}" for h in hits]
    augmented = "Source passages:\n" + "\n".join(context_lines) + f"\n\n{prompt}"
    retrieved_text = "\n\n".join(
        f"[rerank score {h['score']:.3f}] {h['doc_id']}: {h['text'][:200]}" for h in hits
    )

    quantized_model = _get_quantized_model()
    messages = [{"role": "user", "content": augmented}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(DEVICE)
    with torch.no_grad():
        out = quantized_model.generate(
            **inputs, max_new_tokens=int(max_new_tokens), do_sample=True,
            temperature=float(temperature), top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, retrieved_text


with gr.Blocks(title="MarineSparkLM") as demo:
    with gr.Tabs():
        with gr.Tab("Marine corpus"):
            gr.Markdown(
                "# MarineSparkLM-4B\n"
                "A LoRA instruction fine-tune of OceanGPT-basic-4B-Instruct, trained on "
                "2,295 grounded question/answer pairs from a 10-theme marine and "
                "offshore engineering corpus (run_500m_v1), on top of the base model's "
                "existing domain knowledge. Domain-biased, not a verified source of "
                "engineering fact -- check anything load-bearing independently."
            )
            prompt = gr.Textbox(label="Prompt", lines=2, placeholder="Type an engineering prompt...")
            with gr.Row():
                max_new_tokens = gr.Slider(10, 400, value=120, step=10, label="Max new tokens",
                                            info="How much text to generate")
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature",
                                         info="Lower = safer/repetitive; higher (1.0+) = more varied/less coherent")
                top_k = gr.Slider(1, 200, value=50, step=1, label="Top-k",
                                   info="Restricts sampling to the k most likely next tokens")
            with gr.Row():
                repetition_penalty = gr.Slider(1.0, 2.0, value=1.2, step=0.05, label="Repetition penalty",
                                                info="1.0 = off; higher discourages repeating already-generated tokens")
                no_repeat_ngram_size = gr.Slider(0, 6, value=3, step=1, label="No-repeat n-gram size",
                                                  info="0 = off; bans repeating any n-token sequence of this size")
            gr.Markdown(
                "Both repetition knobs default on to discourage looping; set "
                "repetition penalty to 1.0 and no-repeat n-gram to 0 to disable them."
            )
            with gr.Row():
                use_rag = gr.Checkbox(label="Ground with retrieval (RAG)", value=False,
                                       info="Retrieve real passages from the current 10-theme "
                                            "run_500m_v1 corpus and prime generation with them -- "
                                            "recommended for anything factual.")
                rag_k = gr.Slider(1, 8, value=3, step=1, label="Passages to retrieve")
                retrieval_method = gr.Radio(["TF-IDF", "Dense (embedding)"], value="TF-IDF",
                                             label="Retrieval method",
                                             info="TF-IDF is the default. Dense (bge-small) trended "
                                                  "slightly better and avoided a contamination case "
                                                  "TF-IDF fell into in testing, but the difference "
                                                  "wasn't statistically significant -- see eval/README.md.")
            btn = gr.Button("Generate", variant="primary")
            output = gr.Textbox(label="Output", lines=30, max_lines=1000)
            retrieved = gr.Textbox(label="Retrieved passages (if RAG enabled)", lines=10, max_lines=1000)
            gr.Examples(EXAMPLES, inputs=prompt)

            inputs = [prompt, max_new_tokens, temperature, top_k, repetition_penalty, no_repeat_ngram_size, use_rag, rag_k, retrieval_method]
            btn.click(run, inputs=inputs, outputs=[output, retrieved])
            prompt.submit(run, inputs=inputs, outputs=[output, retrieved])

        with gr.Tab("Upload your own documents"):
            gr.Markdown(
                "# Retrieval-augmented generation over your own documents\n"
                "Upload .txt or .pdf files, build a per-session index (in memory, "
                "not saved anywhere), then ask questions. Retrieval does the heavy "
                "lifting -- dense search over your documents finds candidate passages, "
                "a cross-encoder reranker picks the most relevant few, and only those "
                "go into the prompt. Generation runs on a 4-bit quantized version of "
                "the same 4B-parameter model (~3GB instead of ~8GB), so most of the "
                "work is retrieval quality, not model size."
            )
            session_state = gr.State(None)
            files = gr.File(label="Upload documents", file_count="multiple", file_types=[".txt", ".pdf"])
            index_btn = gr.Button("Build index")
            index_status = gr.Textbox(label="Index status", interactive=False)
            index_btn.click(build_session_index, inputs=files, outputs=[session_state, index_status])

            up_prompt = gr.Textbox(label="Question", lines=2, placeholder="Ask about the uploaded document(s)...")
            with gr.Row():
                up_max_new_tokens = gr.Slider(10, 400, value=120, step=10, label="Max new tokens")
                up_temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                up_top_k = gr.Slider(1, 200, value=50, step=1, label="Top-k")
            with gr.Row():
                up_repetition_penalty = gr.Slider(1.0, 2.0, value=1.2, step=0.05, label="Repetition penalty")
                up_no_repeat_ngram_size = gr.Slider(0, 6, value=3, step=1, label="No-repeat n-gram size")
                up_k_final = gr.Slider(1, 8, value=4, step=1, label="Passages after reranking")
            up_btn = gr.Button("Generate", variant="primary")
            up_output = gr.Textbox(label="Output", lines=30, max_lines=1000)
            up_retrieved = gr.Textbox(label="Retrieved passages (post-rerank)", lines=10, max_lines=1000)

            up_inputs = [up_prompt, session_state, up_max_new_tokens, up_temperature, up_top_k,
                         up_repetition_penalty, up_no_repeat_ngram_size, up_k_final]
            up_btn.click(run_uploaded, inputs=up_inputs, outputs=[up_output, up_retrieved])
            up_prompt.submit(run_uploaded, inputs=up_inputs, outputs=[up_output, up_retrieved])

if __name__ == "__main__":
    demo.launch()
