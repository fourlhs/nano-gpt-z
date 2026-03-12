import math
import numpy as np
import torch
from model import GPT

# hyperparameters
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size    = 50257
batch_size    = 32
block_size    = 64
eval_iters    = 100
eval_interval = 500
max_steps     = 5000
learning_rate = 3e-5
min_lr        = 3e-6
drive_path    = '/content/drive/MyDrive/nanogpt'

SUBSETS = [1, 5, 20, 50, 100, 200, 500, 1000]  # in thousands


def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([torch.from_numpy(data[i   : i+block_size  ].astype(np.int64)) for i in ix])
    y  = torch.stack([torch.from_numpy(data[i+1 : i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def get_lr(step):
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * step / max_steps))

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


# fine-tuning loop — one run per subset, always from same base
for n in SUBSETS:
    print(f"\nsubset: genz_{n}k")

    # load data
    data = np.memmap(f'data/finetune/genz_{n}k.bin', dtype=np.uint16, mode='r')

    # reload pretrained weights fresh every run
    model = GPT(vocab_size).to(device)
    model.load_state_dict(torch.load(f'{drive_path}/best_model.pt', map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_loss = float('inf')

    for step in range(max_steps):

        # evaluate
        if step % eval_interval == 0:
            loss = estimate_loss(model, data)
            print(f"  {step:5d} | {loss:.4f}")

            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), f'{drive_path}/finetune_{n}k.pt')

        # lr schedule
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # forward + backward
        x, y         = get_batch(data)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"  done | best loss {best_loss:.4f}")

print("\nall subsets complete.")