# Mistral-7B Breast Cancer QLoRA Fine-Tune

> A domain-specialized language model trained on breast cancer medical Q&A using QLoRA — fine-tuned on a single free GPU in under 40 minutes.

**Author:** [DiegoDomLarr](https://huggingface.co/DiegoDomLarr)  
**Status:** ✅ Complete

| Artifact | Link |
|---|---|
| LoRA adapter | [DiegoDomLarr/mistral-7b-breast-cancer-qlora](https://huggingface.co/DiegoDomLarr/mistral-7b-breast-cancer-qlora) |
| Training dataset | [DiegoDomLarr/breast-cancer-qa](https://huggingface.co/datasets/DiegoDomLarr/breast-cancer-qa) |
| Base model | [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |

---

## Overview

Large language models are powerful generalists, but generalists have limits. A model trained on internet-scale data doesn't automatically excel at a narrow medical domain — it dilutes its answers across everything it knows.

This project fine-tunes Mistral-7B-Instruct specifically on breast cancer Q&A, producing a model that answers domain questions in a more structured, clinically-oriented way than the base model.

**Method:** QLoRA — 4-bit quantization + LoRA adapters. Only ~1% of the model's parameters are trained. The rest stay frozen. This makes it possible to fine-tune a 7B model on a single NVIDIA T4 (Google Colab free tier) without running out of memory.

**Result:** ROUGE-L of 0.45 between fine-tuned and base model responses — confirming the fine-tuned model generates meaningfully different, more domain-focused answers.

---

## Quick Start

```bash
pip install transformers peft bitsandbytes accelerate
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, "DiegoDomLarr/mistral-7b-breast-cancer-qlora")
model.eval()

prompt = "<s>[INST] What are the side effects of tamoxifen? [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Training Data

The adapter was trained on **[DiegoDomLarr/breast-cancer-qa](https://huggingface.co/datasets/DiegoDomLarr/breast-cancer-qa)** — 1,061 breast cancer Q&A pairs curated from two public sources:

| Dataset | Raw size | After filter | Type |
|---|---|---|---|
| [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) (`pqa_labeled`) | 1,000 | 29 | Human-verified biomedical Q&A from PubMed abstracts |
| [lavita/ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) | 112,165 | 1,032 | Real patient–doctor consultations |
| **Total** | — | **1,061** | |

**Filter keywords:** `breast cancer`, `breast carcinoma`, `BRCA1`, `BRCA2`, `HER2`, `tamoxifen`, `mastectomy`, `lumpectomy`, `mammogram`, `ductal carcinoma`, `lobular carcinoma`, `triple negative breast`, `aromatase inhibitor`, `trastuzumab`

All examples were converted to Mistral's instruction format before training:

```
<s>[INST] {question} [/INST] {answer} </s>
```

---

## Model & Training Config

| Parameter | Value |
|---|---|
| Base model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Quantization | 4-bit NF4 with double quantization |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Epochs | 3 |
| Effective batch size | 8 (2 per device × 4 gradient accumulation steps) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Max sequence length | 512 tokens |
| Hardware | NVIDIA T4 — Google Colab free tier |
| Training time | ~35 minutes |

---

## Evaluation

Fine-tuned vs base Mistral on: *"What are the main risk factors for breast cancer?"*

**Fine-tuned model:**
> Several factors can increase the risk of developing breast cancer:
> 1. **Gender** — Being female is the greatest risk factor.
> 2. **Age** — Most breast cancers are diagnosed in women over 50.
> 3. **Genetic Factors** — BRCA1 and BRCA2 mutations significantly increase risk. Family history in first-degree relatives also matters.
> 4. **Lifestyle Factors** — Physical inactivity, high-saturated-fat diet, obesity, and smoking all contribute.

**Base Mistral:**
> Breast cancer is the most common cancer among women worldwide. Several risk factors can increase a woman's chance of developing it:
> 1. **Age** — Risk increases with age. Most cases are diagnosed after 50.
> 2. **Genetic factors** — BRCA1 and BRCA2 mutations. Family history in first-degree relatives.
> 3. **Hormonal factors** — Extended exposure to estrogen and progesterone. Early menstruation, late menopause, never having given birth.

**ROUGE-L: `0.4509`** — confirming the fine-tuned model produces structurally different, more patient-oriented answers that cover additional factors (gender, lifestyle) not prominently addressed by the base model.

---

## Architecture

```
HF Datasets Hub ──► Google Colab (T4 GPU) ──► HF Hub (adapter)
                         │
                         └──► Google Drive (checkpoints)
```

```
BreastCancerFineTune.ipynb
├── Session Setup      — mount Drive, authenticate HF
├── Step 1             — load & filter dataset, push to HF
├── Step 3             — load Mistral-7B in 4-bit
├── Step 4             — apply LoRA adapter
├── Step 5             — format data (Mistral chat template)
├── Step 6             — train with SFTTrainer
├── Step 7             — evaluate vs base model (ROUGE-L)
└── Step 8             — push adapter + model card to HF Hub
```

- Dataset on HF Hub — `load_dataset()` works from any session, no manual uploads
- Checkpoints to Drive — Colab sessions die; Drive survives disconnects
- Adapter on HF Hub — permanent, public, versioned, usable in one line

---

## Project Structure

```
.
├── BreastCancerFineTune.ipynb   # Main Colab notebook
├── model_card.md                # HuggingFace model card source
├── config/
│   └── training_config.yaml     # Hyperparameters
├── scripts/
│   ├── prepare_data.py          # Data pipeline reference
│   ├── evaluate.py              # ROUGE-L evaluation
│   └── push_to_hub.py           # Standalone publisher
├── data/
│   └── .gitkeep                 # Data lives on HF Hub
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Library | Role |
|---|---|
| `transformers` | Load Mistral-7B and tokenizer |
| `peft` | Apply and load LoRA adapters |
| `bitsandbytes` | 4-bit quantization |
| `trl` | SFTTrainer for supervised fine-tuning |
| `datasets` | Load and process training data |
| `evaluate` | ROUGE-L scoring |
| `huggingface_hub` | Push adapter and dataset to HF Hub |

---

## Running the Notebook

**Prerequisites:**
- Google account (Colab + Drive)
- HuggingFace account with a write token
- No local GPU required

**Steps:**
1. Open `BreastCancerFineTune.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU)
3. Add your HF token to Colab Secrets (left sidebar → 🔑) with the name `HF_TOKEN`
4. Run the **Session Setup** cell first, then proceed top to bottom

If you already have a trained adapter saved to Drive, skip to the **"Load from Drive"** shortcut cell in Step 7 to avoid re-training.

---

## Limitations

This model is for **educational and research purposes only**. It is not a medical device and must not be used for clinical diagnosis, treatment decisions, or patient care. Always consult a licensed healthcare professional.

---

## License

MIT
