"""
data/prepare.py

Downloads and tokenizes:
  - Pretraining  : FineWeb-Edu (400M token slice)
  - Fine-tuning  : Sam-genz-omni + genz_brainrot_dataset (Gen Z column)
  - Evaluation   : WikiText-103 validation split (perplexity benchmark)

Outputs:
  data/pretrain/train.bin
  data/pretrain/val.bin
  data/finetune/genz.bin
  data/finetune/genz_{n}k.bin   ← 1k, 5k, 20k, 50k, 100k, 200k, 500k, 1000k
  data/wikitext_val.bin         ← used by finetune.py and evaluate.py
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset

# paths
# Use env vars with fallbacks for local testing
DATA_DIR = os.environ.get("DATA_DIR", "data")

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

PRETRAIN_TOKENS  = 400_000_000
FINETUNE_SUBSETS = [1_000, 5_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]

os.makedirs(f"{DATA_DIR}/pretrain", exist_ok=True)
os.makedirs(f"{DATA_DIR}/finetune", exist_ok=True)


def tokenize(text: str) -> list[int]:
    return enc.encode_ordinary(text.strip()) + [EOT]


def save_bin(tokens: list[int], path: str):
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(path)
    print(f"  saved {len(arr):,} tokens → {path}")


# 1. Pretraining — FineWeb-Edu (400M token slice)
print("downloading FineWeb-Edu...")
tokens  = []
dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
)
for doc in dataset:
    tokens.extend(tokenize(doc["text"]))
    if len(tokens) >= PRETRAIN_TOKENS:
        break

tokens = tokens[:PRETRAIN_TOKENS]
split  = int(0.9 * len(tokens))
save_bin(tokens[:split], f"{DATA_DIR}/pretrain/train.bin")
save_bin(tokens[split:], f"{DATA_DIR}/pretrain/val.bin")
del tokens


# 2. Fine-tuning — Sam-genz-omni + brainrot Gen Z
print("\ndownloading Gen Z data...")
genz = []

for row in load_dataset("Smilyai-labs/Sam-genz-omni", split="train"):
    genz.extend(tokenize(row["prompt"] + " " + row["response"]))

for row in load_dataset("projolx/genz_brainrot_dataset", split="train"):
    genz.extend(tokenize(row["gen_z"]))

print(f"  total Gen Z tokens: {len(genz):,}")
save_bin(genz, f"{DATA_DIR}/finetune/genz.bin")

for n in FINETUNE_SUBSETS:
    if n <= len(genz):
        save_bin(genz[:n], f"{DATA_DIR}/finetune/genz_{n//1000}k.bin")
    else:
        print(f"  skipping {n//1000}k — not enough tokens ({len(genz):,} available)")

del genz


# 3. WikiText-103 validation — perplexity benchmark
print("\ndownloading WikiText-103 validation...")
wikitext = []
for row in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation"):
    if row["text"].strip():
        wikitext.extend(tokenize(row["text"]))

save_bin(wikitext, f"{DATA_DIR}/wikitext_val.bin")
del wikitext


print("\n done.")
print(f"  {DATA_DIR}/pretrain/train.bin          ~360M tokens")
print(f"  {DATA_DIR}/pretrain/val.bin             ~40M tokens")
print(f"  {DATA_DIR}/finetune/genz.bin            ~1M tokens")
print(f"  {DATA_DIR}/finetune/genz_1k.bin ... genz_1000k.bin")
print(f"  {DATA_DIR}/wikitext_val.bin             validation split")