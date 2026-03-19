# NanoGPT-Z

**Training small language models to talk like you — a study in style adaptation and catastrophic forgetting.**

A GPT trained on 1B tokens of standard English, then fine-tuned on Gen Z slang across 8 data regimes. We measure *when* the model learns to talk differently and *when* it starts forgetting standard English — producing quantitative forgetting curves across the full fine-tuning trajectory.

**[Live Demo](https://nikosfourlis.com/nano-gpt-z)** · **[Paper](https://nikosfourlis.com/papers/nano-gpt-z-catastrophic-forgetting.pdf)** · **[GitHub](https://github.com/fourlhs/nano-gpt-z)**

> Try the live demo — chat with the fine-tuned model hosted on Hugging Face Spaces.

---

## What We Found

Catastrophic forgetting is immediate and severe. After fine-tuning on just **1k tokens** of Gen Z slang, WikiText perplexity jumps from **258 → 12,611** — a **47x increase** — while slang loss drops to near zero. The model learns the new style almost instantly, and forgets standard English just as fast.

| Fine-tune tokens | WikiText Perplexity | Δ Perplexity | Style Shift |
|-----------------|-------------------|-------------|-------------|
| Baseline (none) | 258 | — | 1.4% |
| 1k | 12,611 | +12,353 | 18.1% |
| 5k | 12,339 | +12,081 | 19.0% |
| 20k | 13,009 | +12,750 | 17.7% |
| 50k | 24,988 | +24,730 | 17.9% |
| 100k | 23,736 | +23,478 | 18.7% |
| 200k | 34,129 | +33,871 | 20.1% |
| 500k | 40,518 | +40,260 | 16.5% |
| 1000k | 39,535 | +39,277 | 19.9% |

### Forgetting Curves
![Forgetting Curves](paper/results/forgetting_curves.png)
*WikiText perplexity (log scale) across fine-tuning steps. Each line represents a different fine-tuning set size (1k–1M tokens). Baseline perplexity (dashed line) is quickly exceeded within the first 200 steps.*

### Style-Forgetting Tradeoff
![Style Tradeoff](paper/results/tradeoff.png)
*Scatter plot showing the relationship between style shift (Δ, x-axis) and perplexity increase (Δ, y-axis, log scale). Points are labeled by fine-tuning set size. Larger datasets do not reduce forgetting; catastrophic forgetting is inevitable.*

**Key finding:** forgetting is not gradual — it is catastrophic within the first 200 fine-tuning steps, regardless of fine-tuning corpus size. Style shift plateaus around 18-20% across all subset sizes, suggesting a ceiling on how much slang a 17M parameter model can adopt from this corpus.

---

## Architecture

A 17M parameter GPT trained from scratch:

- 6 transformer layers, 8 attention heads, 256 embedding dim
- Fused QKV attention with `F.scaled_dot_product_attention` (FlashAttention)
- Weight-tied token embeddings and LM head
- GPT-2 tokenizer (tiktoken, vocab size 50,257)
- Context window: 64 tokens

Custom C++ inference engine (compiled to WebAssembly — see `inference/`):
- Int8 quantised weights (halves `.bin` size for network load)
- KV cache (O(T) decode instead of O(T²))
- WASM SIMD via `wasm_f32x4_*` intrinsics
- Top-p nucleus sampling

> The C++ engine is a standalone technical contribution and is not used in the live demo. The demo runs a Flask backend hosted on Hugging Face Spaces.

---

## Training

**Pretraining:** 1B tokens from FineWeb-Edu (`sample-10BT`), batch 512, lr 1e-3 with cosine decay, ~30k steps on RTX 4090.

**Fine-tuning:** 8 subset runs from the same pretrained checkpoint — 1k, 5k, 20k, 50k, 100k, 200k, 500k, 1000k tokens from two Gen Z datasets (`Sam-genz-omni` + `genz_brainrot_dataset`). Each run measured WikiText-103 perplexity every 200 steps.

---

## Reproduce

```bash
git clone https://github.com/fourlhs/nano-gpt-z
cd nano-gpt-z
pip install -r requirements.txt

# prepare data
python3 data/prepare.py

# pretrain
python3 train.py

# finetune all 8 subsets + measure forgetting
python3 finetune.py

# evaluate + export results
python3 evaluate.py

# export weights for inference
python3 export_weights.py --ckpt checkpoints/best_model.pt --out demo/weights.bin
```

RunPod RTX 4090 recommended. Full pipeline runs in ~2 hours and costs under $1.

---

## Datasets & Metrics

Raw experiment data:
- **[results.csv](paper/results/results.csv)** — Final metrics per subset: perplexity, style shift, deltas
- **[finetune_metrics.json](paper/results/finetune_metrics.json)** — Step-level metrics during fine-tuning: loss, perplexity, forgetting curves

---

## Repo Structure

```
model.py              — GPT architecture
train.py              — pretraining loop
finetune.py           — fine-tuning + forgetting measurement
evaluate.py           — perplexity + style shift metrics
export_weights.py     — fp32 → int8 binary for C++ runtime
data/prepare.py       — download + tokenise all datasets
inference/
  model.h             — GPT/Block/KVCache structs
  primitives.h        — SIMD matmul, layernorm, gelu, softmax
  attention.cpp       — KV-cached multi-head attention
  mlp.cpp             — feedforward block
  forward.cpp         — full forward pass + weight loading
  generate.cpp        — sampling, generate loop, global state
  bindings.cpp        — Emscripten JS bindings
  build.sh            — compiles to WASM
paper/results/
  results.csv                — perplexity + style shift per subset
  finetune_metrics.json      — step-level metrics for all 8 runs
  forgetting_curves.png      — publication-quality forgetting plot
  tradeoff.png               — style-forgetting tradeoff scatter plot
plot.py                      — generate figures from metrics & results
```

---

## Built by

[Nikos Fourlis](https://nikosfourlis.com) · Athens · 2026
