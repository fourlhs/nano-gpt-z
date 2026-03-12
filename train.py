import math
import os
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from model import GPT

# paths
# Use env vars with fallbacks for local testing
DATA_DIR = os.environ.get("DATA_DIR", "data")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")

# hyperparameters
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size    = 50257       # GPT-2 tokenizer (tiktoken)
batch_size    = 64
block_size    = 64
eval_iters    = 300
eval_interval = 500
max_steps     = 240000
learning_rate = 6e-4
min_lr        = 5e-5
warmup_steps  = 1000
grad_clip     = 1.0
drive_path    = CHECKPOINT_DIR

# Log-spaced save steps + every 10k after 50k.
# Dense early coverage is critical for the forgetting curve —
# most catastrophic forgetting happens in the first few thousand steps.
SAVE_STEPS = sorted(set(
    [int(10 ** (i * 0.25)) for i in range(1, 25) if int(10 ** (i * 0.25)) <= 50000]
    + list(range(0, max_steps + 1, 10000))
))

# data
def get_batch(split):
    path = f'{DATA_DIR}/pretrain/{"train" if split == "train" else "val"}.bin'
    data = np.memmap(path, dtype=np.uint16, mode='r')
    ix   = torch.randint(len(data) - block_size, (batch_size,))
    x    = torch.stack([torch.from_numpy(data[i  :i+block_size  ].astype(np.int64)) for i in ix])
    y    = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# evaluation
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# learning rate: linear warmup → cosine decay
def get_lr(step):
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    return min_lr + 0.5 * (learning_rate - min_lr) * (
        1 + math.cos(math.pi * (step - warmup_steps) / (max_steps - warmup_steps))
    )

# checkpoint helpers
def save_checkpoint(tag: str):
    os.makedirs(drive_path, exist_ok=True)
    path = f'{drive_path}/{tag}.pt'
    torch.save({
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step':      step,
        'val_loss':  last_val_loss,
    }, path)
    print(f"  ✓ saved {path}")

def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate 
    
    start_step = ckpt['step']
    return start_step, ckpt.get('val_loss', float('inf'))

# model + optimizer
model     = GPT(vocab_size).to(device)
model     = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                              betas=(0.9, 0.95), weight_decay=0.1)

n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params/1e6:.2f}M | device: {device}")
print(f"save steps: {len(SAVE_STEPS)} checkpoints planned")

# Resume from checkpoint if one exists
start_step    = 0
last_val_loss = float('inf')
best_val_loss = float('inf')
resume_path   = f'{drive_path}/latest.pt'
if os.path.exists(resume_path):
    start_step, last_val_loss = load_checkpoint(resume_path)
    best_val_loss = last_val_loss
    print(f"resumed from step {start_step}")

# training loop
for step in range(start_step, max_steps):

    # LR update
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # Evaluate
    if step % eval_interval == 0:
        losses        = estimate_loss()
        last_val_loss = losses['val']
        print(f"step {step:6d} | train {losses['train']:.4f} | "
              f"val {losses['val']:.4f} | lr {lr:.2e}")

        if last_val_loss < best_val_loss:
            best_val_loss = last_val_loss
            save_checkpoint('best_model')

    # Scheduled checkpoint (log-spaced early, then every 10k)
    if step in SAVE_STEPS:
        save_checkpoint(f'ckpt_{step:06d}')
        # Always overwrite latest for easy resume
        save_checkpoint('latest')

    # Forward + backward with bfloat16
    x, y = get_batch('train')
    optimizer.zero_grad()
    with autocast(dtype=torch.bfloat16):
        logits, loss = model(x, y)
        
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step() 

print("training complete.")