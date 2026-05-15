# Mistral-7B Breast Cancer QLoRA Fine-Tune

> Fine-tuning Mistral-7B-Instruct on breast cancer medical Q&A using QLoRA — specialized model that outperforms the base on domain-specific questions.

**Author:** [DiegoDomLarr](https://huggingface.co/DiegoDomLarr)  
**Status:** 🔄 In progress — Step 2 of 8

---

## What we're building

A domain-specialized version of Mistral-7B trained exclusively on breast cancer medical Q&A. The hypothesis: a smaller model fine-tuned on a specific domain beats a general-purpose large model on that domain's questions.

**Three deliverables:**

| Artifact | Location | Status |
|---|---|---|
| LoRA adapter | [`DiegoDomLarr/mistral-7b-breast-cancer-qlora`](https://huggingface.co/DiegoDomLarr/mistral-7b-breast-cancer-qlora) | ⬜ Pending |
| Training dataset | [`DiegoDomLarr/breast-cancer-qa`](https://huggingface.co/datasets/DiegoDomLarr/breast-cancer-qa) | ✅ Live (1,061 examples) |
| Benchmark results | `results/benchmark.json` | ⬜ Pending |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Google Colab (T4 GPU)              │
│                                                      │
│   BreastCancerFineTune.ipynb                         │
│   ├── Session Setup (Drive mount + HF auth)          │
│   ├── Step 1: Load dataset from HF Hub              │
│   ├── Step 3: Load Mistral-7B in 4-bit (QLoRA)      │
│   ├── Step 4: Apply LoRA adapter                    │
│   ├── Step 5: Format data (Mistral chat template)   │
│   ├── Step 6: Train with SFTTrainer                 │
│   │     └── Checkpoints ──► Google Drive            │
│   ├── Step 7: Evaluate (ROUGE-L vs base)            │
│   └── Step 8: Push adapter ──► HuggingFace Hub      │
└─────────────────────────────────────────────────────┘

Data flow:
  HF Datasets Hub ──► Colab (training) ──► HF Hub (adapter)
                                       └──► Google Drive (checkpoints)
```

**Why this architecture:**
- **Dataset on HF Hub** — `load_dataset()` works from any session, no manual uploads
- **Checkpoints to Drive** — Colab sessions die unexpectedly; Drive survives disconnects
- **Adapter to HF Hub** — permanent, public, versioned, usable by anyone in one line
- **Hyperparams in `config/training_config.yaml`** — change a number in one place, not hunting through a notebook

---

## Project structure

```
.
├── BreastCancerFineTune.ipynb   # Main Colab notebook — run this on GPU
├── config/
│   └── training_config.yaml    # All hyperparameters in one place
├── scripts/
│   ├── prepare_data.py         # Data sourcing logic (reference + docs)
│   ├── evaluate.py             # ROUGE-L evaluation vs base model
│   └── push_to_hub.py          # Standalone adapter publisher
├── data/
│   └── .gitkeep                # Folder tracked; actual data lives on HF
├── requirements.txt            # Python dependencies
├── CLAUDE.md                   # AI collaboration guide
└── README.md
```

---

## Roadmap

- [x] **Step 1 — Data** — Collected 1,061 breast cancer Q&A pairs from PubMedQA + ChatDoctor. Pushed to HF Datasets Hub.
- [ ] **Step 2 — Understand QLoRA** — Concepts: 4-bit quantization, LoRA adapters, why it fits in 15GB VRAM. No code.
- [ ] **Step 3 — Load the model** — Mistral-7B-Instruct in 4-bit on Colab without OOM.
- [ ] **Step 4 — Apply LoRA** — Wrap quantized model with trainable adapter. Print param counts.
- [ ] **Step 5 — Format data** — Convert Q&A pairs to Mistral chat template format.
- [ ] **Step 6 — Train** — SFTTrainer, watch the loss curve, save adapter to Drive.
- [ ] **Step 7 — Evaluate** — ROUGE-L comparison: fine-tuned vs base Mistral on breast cancer questions.
- [ ] **Step 8 — Publish** — Push adapter + model card to HuggingFace Hub.

---

## Dataset

**Source:** Two public datasets, filtered for breast cancer content.

| Dataset | Total examples | After filter | Type |
|---|---|---|---|
| [`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA) (`pqa_labeled`) | 1,000 | 29 | Biomedical research Q&A |
| [`lavita/ChatDoctor-HealthCareMagic-100k`](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) | 112,165 | 1,032 | Patient–doctor consultations |
| **Combined** | — | **1,061** | — |

**Filter keywords:** `breast cancer`, `breast carcinoma`, `BRCA1`, `BRCA2`, `HER2`, `tamoxifen`, `mastectomy`, `lumpectomy`, `mammogram`, `ductal carcinoma`, `lobular carcinoma`, `triple negative breast`, `aromatase inhibitor`, `trastuzumab`

**Format:**
```json
{
  "question": "If you are a doctor, please answer... I was diagnosed with...",
  "answer": "Based on your description, the recommended approach is..."
}
```

---

## Model

**Base:** [`mistralai/Mistral-7B-Instruct-v0.2`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)  
**Method:** QLoRA — 4-bit quantization + LoRA adapters (only ~1% of parameters are trained)  
**Target repo:** [`DiegoDomLarr/mistral-7b-breast-cancer-qlora`](https://huggingface.co/DiegoDomLarr/mistral-7b-breast-cancer-qlora)

### Key hyperparameters

| Parameter | Value | Why |
|---|---|---|
| Quantization | 4-bit (NF4) | Fits Mistral-7B in ~12GB VRAM |
| LoRA rank (`r`) | 16 | Balances expressiveness vs parameter count |
| LoRA alpha | 32 | Standard 2× rank scaling |
| Target modules | `q_proj`, `v_proj` | Attention layers — where domain adaptation matters most |
| Learning rate | 2e-4 | Standard for LoRA fine-tuning |
| Epochs | 3 | Enough for 1,061 examples without overfitting |
| Max sequence length | 512 tokens | Covers most Q&A pairs |

All hyperparams live in `config/training_config.yaml`.

---

## Tech stack

| Library | Role |
|---|---|
| `transformers` | Load Mistral-7B and tokenizer |
| `peft` | Apply LoRA adapters |
| `bitsandbytes` | 4-bit quantization |
| `trl` | SFTTrainer (supervised fine-tuning) |
| `datasets` | Load and process data |
| `evaluate` | ROUGE-L scoring |
| `huggingface_hub` | Push adapter + dataset to HF |

---

## Setup

### Prerequisites
- Google account (for Colab + Drive)
- HuggingFace account with a **write token**
- No local GPU needed — everything runs on Colab's free T4

### Running the notebook

1. Upload `BreastCancerFineTune.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU)
3. Run **Session Setup** cell first — it will prompt for your HF write token
4. Run cells top to bottom

> **For contributors:** Add your HF token to Colab Secrets (left sidebar → 🔑) with the name `HF_TOKEN`. The session setup cell picks it up automatically so you don't paste it every session.

### Local development

```bash
git clone https://github.com/DiegoLarrieta/mistral-breast-cancer-qlora
cd mistral-breast-cancer-qlora
pip install -r requirements.txt
```

Create a `.env` file:
```
HF_TOKEN=your_huggingface_write_token
```

---

## Using the trained model

Once Step 8 is complete, the adapter will be usable like this:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = PeftModel.from_pretrained(base, "DiegoDomLarr/mistral-7b-breast-cancer-qlora")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

prompt = "[INST] What are the side effects of tamoxifen? [/INST]"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Contributing

This is an active learning project. If you want to contribute:

1. Fork the repo
2. Check the roadmap above — pick an unchecked step
3. Open a PR with your changes and a short description of what you tested

**Key files to understand first:**
- `BreastCancerFineTune.ipynb` — the main notebook, read top to bottom
- `config/training_config.yaml` — all tunable parameters
- `scripts/evaluate.py` — where ROUGE-L benchmarks will live

---

## License

MIT — use it, fork it, build on it.
