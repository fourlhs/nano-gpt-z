import csv
import math
import os
import numpy as np
import torch
import tiktoken
from model import GPT

device     = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = 50257
batch_size = 32
block_size = 64
eval_iters = 300
drive_path = '/content/drive/MyDrive/nanogpt'
results_path = 'paper/results/results.csv'

CHECKPOINTS = [1, 5, 20, 50, 100, 200, 500, 1000]  # in thousands

enc = tiktoken.get_encoding("gpt2")
os.makedirs('paper/results', exist_ok=True)


def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([torch.from_numpy(data[i   : i+block_size  ].astype(np.int64)) for i in ix])
    y  = torch.stack([torch.from_numpy(data[i+1 : i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def build_slang_vocab(genz_data, wiki_data, top_k=200):
    freq_genz = np.bincount(genz_data.astype(np.int64), minlength=vocab_size).astype(float)
    freq_wiki = np.bincount(wiki_data.astype(np.int64), minlength=vocab_size).astype(float)
    freq_wiki += 1  # avoid division by zero
    ratio = freq_genz / freq_wiki
    return set(np.argsort(ratio)[-top_k:].tolist())


@torch.no_grad()
def compute_perplexity(model, data):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return math.exp(losses.mean().item())


@torch.no_grad()
def compute_style_shift(model, slang_vocab):
    model.eval()
    ctx = torch.tensor([enc.encode("\n")], dtype=torch.long, device=device)
    generated = model.generate(ctx, max_new_tokens=500)[0].tolist()
    model.train()
    return sum(1 for t in generated if t in slang_vocab) / len(generated)


# main
genz_data   = np.memmap('data/finetune/genz.bin',       dtype=np.uint16, mode='r')
wiki_data   = np.memmap('data/coherence/wikitext.bin',  dtype=np.uint16, mode='r')
slang_vocab = build_slang_vocab(genz_data, wiki_data, top_k=200)

print(f"{'subset':>8} | {'perplexity':>10} | {'style_shift':>11}")

results = []
for n in CHECKPOINTS:
    ckpt = f'{drive_path}/finetune_{n}k.pt'
    if not os.path.exists(ckpt):
        print(f"{n:>7}k | checkpoint not found, skipping")
        continue

    model = GPT(vocab_size).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    ppl   = compute_perplexity(model, wiki_data)
    style = compute_style_shift(model, slang_vocab)

    results.append({'subset_k': n, 'perplexity': round(ppl, 4), 'style_shift': round(style, 6)})
    print(f"{n:>7}k | {ppl:>10.4f} | {style:>11.6f}")

with open(results_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['subset_k', 'perplexity', 'style_shift'])
    writer.writeheader()
    writer.writerows(results)
