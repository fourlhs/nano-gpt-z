import json
import math
import os
import numpy as np
import torch
from model import GPT

# paths
# Use env vars with fallbacks for local testing
DATA_DIR = os.environ.get("DATA_DIR", "data")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")

# hyperparameters
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size    = 50257
batch_size    = 32
block_size    = 64
eval_iters    = 100
eval_interval = 200        # measure forgetting frequently — this is the data
max_steps     = 5000
learning_rate = 3e-5
min_lr        = 3e-6
warmup_steps  = 50
grad_clip     = 1.0
drive_path    = CHECKPOINT_DIR

SUBSETS = [1, 5, 20, 50, 100, 200, 500, 1000]  # thousands of tokens

# data helpers
def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([torch.from_numpy(data[i  :i+block_size  ].astype(np.int64)) for i in ix])
    y  = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, data):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

@torch.no_grad()
def perplexity(model, data):
    """WikiText perplexity — the forgetting metric."""
    return math.exp(estimate_loss(model, data))

# lr: linear warmup → cosine decay
def get_lr(step):
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    return min_lr + 0.5 * (learning_rate - min_lr) * (
        1 + math.cos(math.pi * (step - warmup_steps) / (max_steps - warmup_steps))
    )

# load base checkpoint (saved by train.py as a dict) 
def load_base(model):
    path = f'{drive_path}/best_model.pt'
    ckpt = torch.load(path, map_location=device)
    # train.py saves {'model': ..., 'optimizer': ..., 'step': ...}
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)

# WikiText validation data (measures forgetting)
wikitext_val = np.memmap(f'{DATA_DIR}/wikitext_val.bin', dtype=np.uint16, mode='r')

# fine-tuning loop
all_results = {}   # { "genz_1k": [{"step": 0, "slang_loss": x, "wiki_ppl": y}, ...] }

for n in SUBSETS:
    tag  = f'genz_{n}k'
    print(f"\n{'─'*50}\nsubset: {tag}")

    slang_data = np.memmap(f'{DATA_DIR}/finetune/{tag}.bin', dtype=np.uint16, mode='r')

    # Fresh model from base weights every run
    model = GPT(vocab_size).to(device)
    load_base(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  betas=(0.9, 0.95), weight_decay=0.1)

    # Baseline perplexity before any fine-tuning
    baseline_ppl = perplexity(model, wikitext_val)
    print(f"  baseline wiki ppl: {baseline_ppl:.2f}")

    results   = []
    best_loss = float('inf')
    os.makedirs(f'{drive_path}/finetune_ckpts/{tag}', exist_ok=True)

    for step in range(max_steps):

        # LR update
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Evaluate — measure both slang loss AND WikiText perplexity
        if step % eval_interval == 0:
            slang_loss = estimate_loss(model, slang_data)
            wiki_ppl   = perplexity(model, wikitext_val)
            forgetting = wiki_ppl - baseline_ppl   # positive = forgetting

            results.append({
                'step':       step,
                'slang_loss': slang_loss,
                'wiki_ppl':   wiki_ppl,
                'forgetting': forgetting,
            })

            print(f"  step {step:5d} | slang {slang_loss:.4f} | "
                  f"wiki ppl {wiki_ppl:.1f} | Δppl {forgetting:+.1f}")

            # Save checkpoint at each eval — needed for forgetting curve
            torch.save({
                'model': model.state_dict(),
                'step':  step,
                'wiki_ppl':   wiki_ppl,
                'slang_loss': slang_loss,
            }, f'{drive_path}/finetune_ckpts/{tag}/step_{step:05d}.pt')

            if slang_loss < best_loss:
                best_loss = slang_loss
                torch.save({
                    'model': model.state_dict(),
                    'step':  step,
                }, f'{drive_path}/finetune_{tag}_best.pt')

        # Forward + backward
        x, y         = get_batch(slang_data)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    all_results[tag] = results
    print(f"  done | best slang loss {best_loss:.4f} | "
          f"final wiki ppl {results[-1]['wiki_ppl']:.1f}")

# save all metrics to JSON for evaluate.py / plotting 
out_path = f'{drive_path}/finetune_metrics.json'
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n✓ metrics saved → {out_path}")
print("all subsets complete.")