# Breast Cancer Fine-Tune — Project Guide

## What we're building
A QLoRA fine-tuned version of Mistral 7B, specialized on breast cancer medical Q&A.
Trained on free Google Colab. Published on HuggingFace.

## Three deliverables
1. **LoRA adapter** — published on HuggingFace Hub
2. **Model card** — explains training data, config, and limitations
3. **Benchmarks** — ROUGE-L comparison: fine-tuned vs base Mistral

## How we work
Step by step. Diego learns by doing. Claude explains concepts before writing code.
No dumping finished solutions — we build and understand together.

---

## The Learning Roadmap

### Step 1 — Understand the data
**Goal:** Find a real breast cancer Q&A dataset, explore it, understand its structure.
**Concepts:** HuggingFace `datasets`, dataset splits, what makes good fine-tuning data.
**Output:** `prepare_data.py` — filters and saves breast cancer Q&A pairs locally.

### Step 2 — Understand QLoRA
**Goal:** Know what we're actually doing before touching a model.
**Concepts:** Quantization (4-bit), LoRA adapters, why QLoRA fits in 15GB VRAM.
**Output:** No code — just solid mental model. Can explain it to someone else.

### Step 3 — Load the model
**Goal:** Load Mistral 7B in 4-bit on Colab without OOM errors.
**Concepts:** `BitsAndBytesConfig`, `AutoModelForCausalLM`, tokenizer setup.
**Output:** Colab cell that loads the model and prints memory usage.

### Step 4 — Apply LoRA
**Goal:** Wrap the quantized model with a trainable LoRA adapter.
**Concepts:** PEFT, rank (r), alpha, target modules, trainable vs frozen params.
**Output:** Colab cell that applies LoRA and prints param counts.

### Step 5 — Format the data for training
**Goal:** Convert raw Q&A pairs into the instruction format Mistral expects.
**Concepts:** Chat templates, instruction tuning, tokenization, sequence length.
**Output:** Dataset with a `text` column in `<s>[INST]...[/INST]...</s>` format.

### Step 6 — Train
**Goal:** Run SFTTrainer and watch the loss go down.
**Concepts:** SFTTrainer, TrainingArguments, gradient accumulation, learning rate schedule.
**Output:** Trained adapter saved to Google Drive.

### Step 7 — Evaluate
**Goal:** Compare fine-tuned vs base model on real breast cancer questions.
**Concepts:** ROUGE-L, inference with `generate()`, temperature, greedy decoding.
**Output:** Benchmark table. Does the fine-tuned model actually do better?

### Step 8 — Publish
**Goal:** Push adapter + model card to HuggingFace Hub.
**Concepts:** `push_to_hub`, model cards, README format on HF.
**Output:** Public HuggingFace repo anyone can use.

---

## Tech stack
- **Model:** `mistralai/Mistral-7B-Instruct-v0.2`
- **Fine-tuning:** QLoRA via `peft` + `bitsandbytes` + `trl`
- **Data:** PubMedQA (`qiaojin/PubMedQA`, `pqa_labeled` split), filtered to breast cancer
- **Training:** Google Colab (T4 GPU, free tier)
- **Evaluation:** ROUGE-L via `evaluate` library
- **Publishing:** HuggingFace Hub — username: `DiegoDomLarr`
- **Model repo will be:** `DiegoDomLarr/mistral-7b-breast-cancer-qlora`

## Key libraries
```
transformers   — load models and tokenizers
peft           — apply LoRA adapters
bitsandbytes   — 4-bit quantization
trl            — SFTTrainer (supervised fine-tuning)
datasets       — load and process data
evaluate       — ROUGE-L scoring
huggingface_hub — push to HF
```

## Current step
→ **Step 1: Understand the data**
