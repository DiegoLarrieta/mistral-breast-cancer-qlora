"""
Step 1 — Data preparation (reference script).

The actual execution lives in BreastCancerFineTune.ipynb (Colab).
The final dataset is published at: https://huggingface.co/datasets/DiegoDomLarr/breast-cancer-qa

This script documents the filtering logic for transparency and reproducibility.

Sources:
  - qiaojin/PubMedQA (pqa_labeled split) — 1,000 biomedical Q&A, filtered to 29 breast cancer
  - lavita/ChatDoctor-HealthCareMagic-100k — 112,165 patient Q&A, filtered to 1,032 breast cancer

Output: 1,061 breast cancer Q&A pairs pushed to HuggingFace Datasets Hub.
"""
