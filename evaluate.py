import csv
import json
import math
import os
import numpy as np
import torch
import tiktoken
from model import GPT

# paths
# Use env vars with fallbacks for local testing
DATA_DIR = os.environ.get("DATA_DIR", "data")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")

# config
device       = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size   = 50257
batch_size   = 32
block_size   = 64
eval_iters   = 300
drive_path   = CHECKPOINT_DIR
results_path = 'paper/results/results.csv'

SUBSETS      = [1, 5, 20, 50, 100, 200, 500, 1000]  # thousands
STYLE_PROMPTS = ["\n", "yo so basically", "ngl this is", "it's giving"]
STYLE_TOKENS  = 300   # tokens per prompt for style measurement

enc = tiktoken.get_encoding("gpt2")
os.makedirs('paper/results', exist_ok=True)

# data
genz_data = np.memmap(f'{DATA_DIR}/finetune/genz.bin',      dtype=np.uint16, mode='r')
wiki_data = np.memmap(f'{DATA_DIR}/wikitext_val.bin',        dtype=np.uint16, mode='r')

# slang vocabulary
def build_slang_vocab(genz_data, wiki_data, top_k=200):
    """
    Token-level TF-IDF: rank tokens by how much more often they appear
    in the Gen Z corpus than in WikiText, normalised by corpus length.
    Normalisation is critical — without it larger corpora dominate the ratio.
    """
    freq_genz = np.bincount(genz_data.astype(np.int64), minlength=vocab_size).astype(float)
    freq_wiki = np.bincount(wiki_data.astype(np.int64), minlength=vocab_size).astype(float)

    # Normalise by corpus length before computing ratio
    freq_genz /= len(genz_data)
    freq_wiki /= len(wiki_data)
    freq_wiki += 1e-8   # avoid division by zero

    ratio = freq_genz / freq_wiki
    return set(np.argsort(ratio)[-top_k:].tolist())

slang_vocab = build_slang_vocab(genz_data, wiki_data, top_k=200)
print(f"slang vocab built | top tokens: "
      f"{[enc.decode([t]) for t in list(slang_vocab)[:10]]}")

# metrics
@torch.no_grad()
def compute_perplexity(model, data):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x  = torch.stack([torch.from_numpy(data[i  :i+block_size  ].astype(np.int64)) for i in ix])
        y  = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return math.exp(losses.mean().item())

@torch.no_grad()
def compute_style_shift(model, slang_vocab):
    """
    Average slang-token proportion across multiple prompts.
    Single-prompt measurement is too noisy for a paper metric.
    """
    model.eval()
    slang_counts = []
    for prompt in STYLE_PROMPTS:
        ctx   = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
        tokens = model.generate(ctx, max_new_tokens=STYLE_TOKENS)[0].tolist()
        slang_counts.append(sum(1 for t in tokens if t in slang_vocab) / len(tokens))
    model.train()
    return float(np.mean(slang_counts))

def load_model(ckpt_path):
    model = GPT(vocab_size).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    return model

# evaluate baseline (pretrained, no fine-tuning)
print("\nevaluating baseline...")
baseline_model = load_model(f'{drive_path}/best_model.pt')
baseline_ppl   = compute_perplexity(baseline_model, wiki_data)
baseline_style = compute_style_shift(baseline_model, slang_vocab)
del baseline_model
print(f"  baseline | ppl {baseline_ppl:.2f} | style {baseline_style:.4f}")

# evaluate each fine-tuned subset
print(f"\n{'subset':>8} | {'ppl':>8} | {'Δppl':>7} | {'style':>8} | {'Δstyle':>8}")
print("─" * 55)

rows = [{'subset_k': 'baseline', 'perplexity': round(baseline_ppl, 4),
         'delta_ppl': 0.0, 'style_shift': round(baseline_style, 6),
         'delta_style': 0.0}]

for n in SUBSETS:
    ckpt_path = f'{drive_path}/finetune_genz_{n}k_best.pt'
    if not os.path.exists(ckpt_path):
        print(f"{n:>7}k | not found, skipping")
        continue

    model = load_model(ckpt_path)
    ppl   = compute_perplexity(model, wiki_data)
    style = compute_style_shift(model, slang_vocab)
    del model

    d_ppl   = ppl   - baseline_ppl
    d_style = style - baseline_style

    rows.append({'subset_k': n, 'perplexity': round(ppl, 4),
                 'delta_ppl': round(d_ppl, 4), 'style_shift': round(style, 6),
                 'delta_style': round(d_style, 6)})

    print(f"{n:>7}k | {ppl:>8.2f} | {d_ppl:>+7.2f} | {style:>8.4f} | {d_style:>+8.4f}")

# also read finetune_metrics.json for step-level curves
metrics_path = f'{drive_path}/finetune_metrics.json'
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        finetune_curves = json.load(f)

    curve_rows = []
    for tag, steps in finetune_curves.items():
        for entry in steps:
            curve_rows.append({'subset': tag, **entry})

    curve_path = 'paper/results/forgetting_curves.csv'
    with open(curve_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subset', 'step', 'slang_loss',
                                               'wiki_ppl', 'forgetting'])
        writer.writeheader()
        writer.writerows(curve_rows)
    print(f"\n✓ forgetting curves → {curve_path}")

# write summary CSV
with open(results_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['subset_k', 'perplexity', 'delta_ppl',
                                           'style_shift', 'delta_style'])
    writer.writeheader()
    writer.writerows(rows)

print(f"✓ summary → {results_path}")