"""
Step 7 — Evaluation (to be implemented).

Compares the fine-tuned adapter against base Mistral-7B-Instruct-v0.2
on a held-out set of breast cancer questions.

Metrics:
  - ROUGE-L (lexical overlap with reference answers)

Output: benchmark table printed to stdout + saved to results/benchmark.json

Usage (once implemented):
  python scripts/evaluate.py \
    --adapter DiegoDomLarr/mistral-7b-breast-cancer-qlora \
    --dataset DiegoDomLarr/breast-cancer-qa \
    --split test
"""
