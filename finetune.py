"""
Paper 2: Mitigations for catastrophic forgetting during style fine-tuning.
Implements: baseline, EWC (Elastic Weight Consolidation), LoRA, and data mixing.

Usage:
    python finetune.py --method baseline     # Baseline (no mitigation)
    python finetune.py --method ewc          # EWC with Fisher penalty
    python finetune.py --method lora         # LoRA adapters (r=8)
    python finetune.py --method mixing --mix_alpha 0.5  # Mix English + GenZ
"""
import json
import math
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from model import GPT

# ─────────────────────────────────────────────────────────────────────────────
# Paths & Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "data")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")

device        = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size    = 50257
batch_size    = 32
block_size    = 64
eval_iters    = 100
eval_interval = 200
max_steps     = 5000
learning_rate = 3e-5
min_lr        = 3e-6
warmup_steps  = 50
grad_clip     = 1.0

SUBSETS = [1, 5, 20, 50, 100, 200, 500, 1000]  # thousands of tokens

# ─────────────────────────────────────────────────────────────────────────────
# LoRA Modules
# ─────────────────────────────────────────────────────────────────────────────
class LoRA(nn.Module):
    """Low-rank adapter: y = W @ x + lora_up(lora_down(x))"""
    def __init__(self, in_features, out_features, r=8):
        super().__init__()
        self.r = r
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up = nn.Linear(r, out_features, bias=False)

        # Initialize: down to small random, up to zero
        nn.init.normal_(self.lora_down.weight, std=1.0 / math.sqrt(r))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.lora_up(self.lora_down(x))

class LinearWithLoRA(nn.Module):
    """Wraps a frozen linear layer with LoRA adapter."""
    def __init__(self, linear_layer, r=8):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRA(linear_layer.in_features, linear_layer.out_features, r=r)

        # Freeze the base linear layer
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)

def apply_lora_to_model(model, r=8):
    """
    Wrap QKV projection + output projection with LoRA adapters.
    Recursively finds attention modules and wraps their projections.
    """
    for name, module in model.named_modules():
        # Target: attention modules (have c_attn and c_proj)
        if hasattr(module, 'c_attn') and hasattr(module, 'c_proj'):
            # Wrap QKV projection (c_attn)
            if isinstance(module.c_attn, nn.Linear):
                module.c_attn = LinearWithLoRA(module.c_attn, r=r)

            # Wrap output projection (c_proj)
            if isinstance(module.c_proj, nn.Linear):
                module.c_proj = LinearWithLoRA(module.c_proj, r=r)

    return model

# ─────────────────────────────────────────────────────────────────────────────
# EWC Fisher Computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_fisher_matrix(model, data, num_samples=10000, device='cpu'):
    """
    Compute diagonal Fisher Information Matrix on first num_samples tokens.
    Returns: dict of {param_name: fisher_diagonal_values}
    """
    print(f"  computing Fisher matrix on {num_samples} tokens...")
    model.eval()

    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param, device=device)

    # Process data in chunks
    tokens_processed = 0
    idx = 0

    with torch.enable_grad():
        while tokens_processed < num_samples:
            # Get batch
            remaining = num_samples - tokens_processed
            batch_tokens = min(remaining, block_size * batch_size)

            # Sample batch from data
            start_idx = idx % (len(data) - block_size)
            idx = start_idx + block_size

            x = torch.from_numpy(data[start_idx:start_idx + block_size].astype(np.int64)).unsqueeze(0).to(device)
            y = torch.from_numpy(data[start_idx + 1:start_idx + block_size + 1].astype(np.int64)).unsqueeze(0).to(device)

            # Forward + backward to get gradients
            model.zero_grad()
            logits, loss = model(x, y)
            loss.backward()

            # Accumulate squared gradients (Fisher diagonal approximation)
            for name, param in model.named_parameters():
                if param.grad is not None and name in fisher:
                    fisher[name] += param.grad ** 2

            tokens_processed += block_size

    # Normalize by number of samples
    for name in fisher:
        fisher[name] /= max(1, tokens_processed // block_size)

    model.train()
    return fisher

# ─────────────────────────────────────────────────────────────────────────────
# Data & Training Utilities
# ─────────────────────────────────────────────────────────────────────────────
def get_batch(data, block_size=block_size, batch_size=batch_size, device='cpu'):
    """Sample a batch from data."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y  = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def get_mixed_batch(slang_data, english_data, mix_alpha, block_size=block_size,
                     batch_size=batch_size, device='cpu'):
    """
    Sample mixed batch: per-batch coin flip to choose English or GenZ.
    mix_alpha: probability of sampling from English data.
    """
    use_english = torch.rand(1).item() < mix_alpha
    data = english_data if use_english else slang_data
    return get_batch(data, block_size=block_size, batch_size=batch_size, device=device)

@torch.no_grad()
def estimate_loss(model, data, eval_iters=eval_iters):
    """Estimate loss on data."""
    model.eval()
    losses = torch.zeros(eval_iters, device=device)
    for k in range(eval_iters):
        x, y = get_batch(data, device=device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

@torch.no_grad()
def perplexity(model, data):
    """WikiText perplexity — the forgetting metric."""
    return math.exp(estimate_loss(model, data))

def get_lr(step, learning_rate=learning_rate, min_lr=min_lr,
           warmup_steps=warmup_steps, max_steps=max_steps):
    """Cosine decay with warmup."""
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    return min_lr + 0.5 * (learning_rate - min_lr) * (
        1 + math.cos(math.pi * (step - warmup_steps) / (max_steps - warmup_steps))
    )

def load_base(model, device='cpu'):
    """Load pretrained base model."""
    path = f'{CHECKPOINT_DIR}/best_model.pt'
    ckpt = torch.load(path, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    clean_state = {}
    for k, v in state.items():
        new_key = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
        clean_state[new_key] = v

    model.load_state_dict(clean_state)
    return model

# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train_one_subset(tag, slang_data, wikitext_val, english_data=None, method='baseline',
                     mix_alpha=0.5, ewc_fisher=None, ewc_lambda=1000.0):
    """
    Train on one dataset subset with specified mitigation method.
    """
    print(f"\n{'─'*60}")
    print(f"subset: {tag} | method: {method}")
    print(f"{'─'*60}")

    # Fresh model from base weights
    model = GPT(vocab_size).to(device)
    model = load_base(model, device=device)

    # Apply LoRA if method is 'lora'
    if method == 'lora':
        model = apply_lora_to_model(model, r=8)
        print("  ✓ LoRA adapters applied (r=8, QKV+output only)")

    # Store base params for EWC penalty (BEFORE torch.compile to avoid _orig_mod. prefixes)
    base_params = {n: p.detach().clone() for n, p in model.named_parameters()}

    # Compile model
    model = torch.compile(model)

    # Optimizer: only trainable parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )

    # Baseline perplexity before fine-tuning
    baseline_ppl = perplexity(model, wikitext_val)
    print(f"  baseline wiki ppl: {baseline_ppl:.2f}")

    results = []
    best_loss = float('inf')
    ewc_penalty_log = 0.0  # Track EWC penalty for logging
    os.makedirs(f'{CHECKPOINT_DIR}/finetune_ckpts/{tag}', exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Training steps
    # ─────────────────────────────────────────────────────────────────────────
    for step in range(max_steps):
        # Update learning rate
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Evaluation
        if step % eval_interval == 0:
            slang_loss = estimate_loss(model, slang_data)
            wiki_ppl = perplexity(model, wikitext_val)
            forgetting = wiki_ppl - baseline_ppl

            result = {
                'step': step,
                'slang_loss': slang_loss,
                'wiki_ppl': wiki_ppl,
                'forgetting': forgetting,
            }

            # Add EWC penalty to result if available
            if method == 'ewc':
                result['ewc_penalty'] = ewc_penalty_log

            results.append(result)

            log_str = f"  step {step:5d} | slang {slang_loss:.4f} | wiki ppl {wiki_ppl:.1f} | Δppl {forgetting:+.1f}"
            if method == 'ewc':
                log_str += f" | ewc_pen {ewc_penalty_log:.2f}"
            print(log_str)

            # Save checkpoint
            torch.save({
                'model': model.state_dict(),
                'step': step,
                'wiki_ppl': wiki_ppl,
                'slang_loss': slang_loss,
                'method': method,
            }, f'{CHECKPOINT_DIR}/finetune_ckpts/{tag}/step_{step:05d}.pt')

            if slang_loss < best_loss:
                best_loss = slang_loss
                torch.save({
                    'model': model.state_dict(),
                    'step': step,
                    'method': method,
                }, f'{CHECKPOINT_DIR}/finetune_{tag}_best.pt')

        # Forward + backward
        optimizer.zero_grad()

        # Get batch based on method
        if method == 'mixing':
            x, y = get_mixed_batch(slang_data, english_data, mix_alpha, device=device)
        else:
            x, y = get_batch(slang_data, device=device)

        with autocast(dtype=torch.bfloat16):
            logits, loss = model(x, y)

        # Add EWC penalty if method is 'ewc'
        if method == 'ewc' and ewc_fisher is not None:
            ewc_penalty = torch.tensor(0.0, device=device)
            for name, param in model.named_parameters():
                if param.requires_grad and name in ewc_fisher:
                    # penalty = lambda * sum((w - w_base)^2 * fisher)
                    param_diff = param - base_params[name]
                    fisher_diag = ewc_fisher[name]
                    ewc_penalty += (fisher_diag * (param_diff ** 2)).sum()

            ewc_penalty = ewc_lambda * ewc_penalty
            ewc_penalty_log = ewc_penalty.item()  # Log for debugging
            loss = loss + ewc_penalty

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    print(f"  done | best slang loss {best_loss:.4f} | final wiki ppl {results[-1]['wiki_ppl']:.1f}")
    return results, model

# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Fine-tune with catastrophic forgetting mitigations')
    parser.add_argument('--method', type=str, default='baseline',
                        choices=['baseline', 'ewc', 'lora', 'mixing'],
                        help='Mitigation method')
    parser.add_argument('--mix_alpha', type=float, default=0.5,
                        help='Probability of sampling English data (for mixing method)')
    parser.add_argument('--ewc_lambda', type=float, default=1000.0,
                        help='EWC penalty weight')
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA rank')
    args = parser.parse_args()

    method = args.method
    mix_alpha = args.mix_alpha
    ewc_lambda = args.ewc_lambda
    lora_r = args.lora_r

    print(f"\n{'='*60}")
    print(f"Fine-tuning with method: {method.upper()}")
    print(f"{'='*60}")

    # Load data
    wikitext_val = np.memmap(f'{DATA_DIR}/wikitext_val.bin', dtype=np.uint16, mode='r')
    english_data = None

    if method in ['ewc', 'mixing']:
        # Load English data for EWC Fisher computation or mixing
        # Use pretraining data (held-out from eval)
        english_path = f'{DATA_DIR}/pretrain/train.bin'
        english_data = np.memmap(english_path, dtype=np.uint16, mode='r')
        print(f"  ✓ Loaded English data: {english_path}")

    # Pre-compute EWC Fisher if needed
    ewc_fisher = None
    if method == 'ewc':
        print("\nPre-computing EWC Fisher matrix...")
        temp_model = GPT(vocab_size).to(device)
        temp_model = load_base(temp_model, device=device)
        temp_model = torch.compile(temp_model)
        ewc_fisher = compute_fisher_matrix(temp_model, english_data, num_samples=10000, device=device)
        del temp_model
        print("  ✓ Fisher matrix computed")

    # Training loop over subsets
    all_results = {}

    for n in SUBSETS:
        tag = f'genz_{n}k'
        slang_data = np.memmap(f'{DATA_DIR}/finetune/{tag}.bin', dtype=np.uint16, mode='r')

        results, _ = train_one_subset(
            tag, slang_data, wikitext_val,
            english_data=english_data,
            method=method,
            mix_alpha=mix_alpha,
            ewc_fisher=ewc_fisher,
            ewc_lambda=ewc_lambda
        )

        all_results[tag] = results

    # Save results
    out_path = f'{CHECKPOINT_DIR}/finetune_metrics_{method}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ metrics saved → {out_path}")
    print("all subsets complete.")

if __name__ == '__main__':
    main()