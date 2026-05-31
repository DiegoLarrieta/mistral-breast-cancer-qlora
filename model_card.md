---
language:
- en
license: apache-2.0
tags:
- medical
- breast-cancer
- qlora
- lora
- fine-tuned
- mistral
- peft
- causal-lm
base_model: mistralai/Mistral-7B-Instruct-v0.2
datasets:
- DiegoDomLarr/breast-cancer-qa
pipeline_tag: text-generation
---

# mistral-7b-breast-cancer-qlora

A QLoRA fine-tuned version of [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) specialized on breast cancer medical question answering.

Trained on a curated dataset of 1,061 breast cancer Q&A pairs assembled from PubMedQA and real patient–doctor consultations. The adapter runs on a single GPU in 4-bit and can be loaded on top of the base Mistral model in minutes.

---

## Why Fine-Tune a Model?

Large language models like Mistral 7B are trained on broad, internet-scale data. That makes them capable generalists — but generalists have limitations when applied to specialized domains.

**Fine-tuning** is the process of continuing the training of a pre-trained model on a smaller, domain-specific dataset. Instead of learning from scratch (which requires massive compute and data), we teach an already-capable model to *speak the language* of a specific field — in this case, breast cancer medicine.

The goal is not to make the model "smarter" in a general sense, but more:
- **Accurate** — using the right clinical terminology and referencing real medical concepts
- **Consistent** — answering breast cancer questions in the structured, informative style of medical Q&A
- **Relevant** — focusing its generation on domain knowledge rather than generic preambles

### Why QLoRA?

Training all 7 billion parameters requires dozens of GB of VRAM and days of compute — out of reach without expensive hardware. **QLoRA (Quantized Low-Rank Adaptation)** makes it accessible with two techniques:

1. **4-bit quantization** — model weights are compressed from 32-bit floats to 4-bit integers, reducing VRAM from ~28 GB to ~5 GB with minimal quality loss.
2. **LoRA adapters** — instead of updating all 7B parameters, small trainable matrices (adapters) are injected into the attention layers. Only ~1–2% of total parameters are trained. The rest stay frozen.

The result: a meaningful domain fine-tune on a single T4 GPU (Google Colab free tier) in approximately 35 minutes.

---

## Model Details

| Property | Value |
|---|---|
| Base model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Fine-tuning method | QLoRA (4-bit quantization + LoRA) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Training epochs | 3 |
| Effective batch size | 8 (2 per device × 4 gradient accumulation steps) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Max sequence length | 512 tokens |
| Quantization type | NF4 (NormalFloat4) with double quantization |
| Training hardware | NVIDIA T4 (Google Colab free tier) |
| Training time | ~35 minutes |

---

## Training Data

Trained on **[DiegoDomLarr/breast-cancer-qa](https://huggingface.co/datasets/DiegoDomLarr/breast-cancer-qa)** — 1,061 breast cancer Q&A pairs from two sources:

| Source | Examples | Description |
|---|---|---|
| [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) (`pqa_labeled`) | 29 | Human-verified biomedical questions from PubMed abstracts |
| [ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) | 1,032 | Real patient–doctor consultations filtered for breast cancer |
| **Total** | **1,061** | |

**Filter keywords:** `breast cancer`, `breast carcinoma`, `BRCA1`, `BRCA2`, `HER2`, `tamoxifen`, `mastectomy`, `lumpectomy`, `mammogram`, `ductal carcinoma`, `lobular carcinoma`, `triple negative breast`, `aromatase inhibitor`, `trastuzumab`

**Dataset statistics:**
- Average question length: 541 characters
- Average answer length: 621 characters

**Training format** — all examples were converted to Mistral's instruction template:

```
<s>[INST] {question} [/INST] {answer} </s>
```

---

## Evaluation

The fine-tuned model was compared against the base Mistral-7B-Instruct on the prompt:

> *"What are the main risk factors for breast cancer?"*

**Fine-tuned model response:**
> Several factors can increase the risk of developing breast cancer. Here are some of the most common risk factors:
>
> 1. **Gender:** Being female is the greatest risk factor for breast cancer.
> 2. **Age:** The risk of breast cancer increases as women get older. Most breast cancers are diagnosed in women over the age of 50.
> 3. **Genetic Factors:** Certain genetic mutations, such as those in the BRCA1 and BRCA2 genes, can significantly increase the risk. Women with a family history of breast cancer in first-degree relatives are also at higher risk.
> 4. **Lifestyle Factors:** A lack of physical activity, a diet high in saturated fat, being overweight or obese, and smoking all contribute to increased risk.

**Base Mistral response:**
> Breast cancer is the most common cancer among women worldwide. Several risk factors can increase a woman's chance of developing breast cancer.
>
> 1. **Age:** The risk increases as women get older. Most breast cancers are diagnosed after age 50.
> 2. **Genetic factors:** A family history of breast cancer increases the risk. Inherited mutations in BRCA1 and BRCA2 significantly increase risk.
> 3. **Hormonal factors:** Extended exposure to estrogen and progesterone can increase risk. Factors include early menstruation, late menopause, and never having given birth.

**ROUGE-L score (base vs fine-tuned): `0.4509`**

A ROUGE-L of ~0.45 confirms the fine-tuned model generates answers that are meaningfully different from the base — more structured, more patient-oriented, and covering additional factors (gender, lifestyle) not prominently addressed by the base model.

---

## How to Use

### Requirements

```bash
pip install transformers peft bitsandbytes accelerate
```

### Inference

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
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, "DiegoDomLarr/mistral-7b-breast-cancer-qlora")
model.eval()
model.config.use_cache = True

prompt = "<s>[INST] What are the side effects of tamoxifen? [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Limitations and Intended Use

> **This model is for educational and research purposes only. It is not a medical device and must not be used for clinical diagnosis, treatment decisions, or patient care.**

- Responses may contain inaccuracies or outdated medical information — always verify with a licensed healthcare professional.
- The model was trained on ~1,000 examples, which is small by fine-tuning standards. It may hallucinate or generalize poorly on edge-case questions.
- Coverage is limited to breast cancer topics represented in the training data.
- This model has not been audited, validated, or certified for any medical use.

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

## About This Project

This model was built as an end-to-end learning project covering the full fine-tuning pipeline:

1. Curating and publishing a domain-specific dataset to HF Hub
2. Loading a 7B model in 4-bit on consumer hardware (Colab T4)
3. Applying LoRA adapters with PEFT
4. Training with SFTTrainer (TRL library)
5. Evaluating with ROUGE-L against the base model
6. Publishing the adapter and model card to HuggingFace Hub

**Author:** [DiegoDomLarr](https://huggingface.co/DiegoDomLarr)  
**Dataset:** [DiegoDomLarr/breast-cancer-qa](https://huggingface.co/datasets/DiegoDomLarr/breast-cancer-qa)  
**Base model:** [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

---

## License

Apache 2.0 — same as the base model.
