import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
block_size = 64
n_embd     = 256
n_head     = 8
n_layer    = 6
dropout    = 0.1


class MultiHeadAttention(nn.Module):
    """
    Fused multi-head self-attention.

    Instead of 8 separate Head modules each doing (256→32) projections,
    we use a single (256→768) linear to compute Q, K, V for all heads
    in one matmul, then split. This is how GPT-2 is actually implemented:
    faster to train, cleaner state dict, and maps directly to the C++
    runtime without any export gymnastics.
    """
    def __init__(self):
        super().__init__()
        assert n_embd % n_head == 0
        self.head_size = n_embd // n_head

        # Fused QKV projection: one matmul instead of 3 × n_head
        self.qkv     = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj    = nn.Linear(n_embd, n_embd, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Causal mask — registered as buffer so it moves with .to(device)
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        # Single matmul → split into Q, K, V  each (B, T, n_embd)
        q, k, v = self.qkv(x).split(n_embd, dim=2)

        # Reshape into (B, n_head, T, head_size) for per-head attention
        q = q.view(B, T, n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, n_head, self.head_size).transpose(1, 2)

        # Scaled dot-product attention
        scores = q @ k.transpose(-2, -1) * self.head_size ** -0.5   # (B, n_head, T, T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Weighted sum, merge heads back → (B, T, n_embd)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))


class MLP(nn.Module):
    """Position-wise feedforward network."""
    def __init__(self):
        super().__init__()
        self.expand  = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.gelu    = nn.GELU()
        self.proj    = nn.Linear(4 * n_embd, n_embd, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(self.gelu(self.expand(x))))


class Block(nn.Module):
    """Single transformer block: pre-norm attention + MLP with residuals."""
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.mlp  = MLP()
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
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

        # Weight tying: token embedding and lm_head share weights.
        # Halves parameters, standard in GPT-2, improves small model quality.
        self.lm_head.weight = self.tok_emb.weight

        # Init weights (GPT-2 convention)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                          # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_p=0.9):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs  = F.softmax(logits, dim=-1)

            # Top-p nucleus sampling — matches the C++ runtime
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # Remove tokens beyond the nucleus
            sorted_probs[cumulative > top_p] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            idx_next = sorted_idx.gather(-1, sampled)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx