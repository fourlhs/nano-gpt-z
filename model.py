import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
block_size = 64
n_embd     = 256
n_head     = 8
n_layer    = 6
dropout    = 0.1


class Head(nn.Module):
    """Single self-attention head."""
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)  # what am i looking for?
        self.key   = nn.Linear(n_embd, head_size, bias=False)  # what do i contain?
        self.value = nn.Linear(n_embd, head_size, bias=False)  # what do i give if attended to?
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # scaled dot-product attention
        scores = q @ k.transpose(-2, -1) * self.head_size ** -0.5  # (B, T, T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # type: ignore
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        return weights @ v  # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Multiple attention heads in parallel."""
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        return self.dropout(self.proj(out))


class MLP(nn.Module):
    """Position-wise feedforward network."""
    def __init__(self):
        super().__init__()
        self.expand  = nn.Linear(n_embd, 4 * n_embd)
        self.gelu    = nn.GELU()
        self.proj    = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.expand(x)
        x = self.gelu(x)
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Single transformer block: attention + MLP with residual connections."""
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.mlp  = MLP()
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # residual + attention
        x = x + self.mlp(self.ln2(x))   # residual + MLP
        return x


class GPT(nn.Module):
    """Full GPT language model."""
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        tok_emb = self.tok_emb(idx)                                    # (B, T, n_embd)
        pos_emb = self.pos_emb(torch.arange(T, device=device))        # (T, n_embd)
        x = tok_emb + pos_emb                                          # (B, T, n_embd)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                       # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx