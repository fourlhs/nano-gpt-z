"""
export.py
Converts a stylift checkpoint (state_dict) into the binary format
expected by inference/forward.cpp.

Binary layout
─────────────
fp32 : tok_emb, pos_emb
per layer (x N_LAYER):
    fp32    : ln1.weight, ln1.bias, ln2.weight, ln2.bias
    QWeight : q_w  ← split from fused qkv (rows 0      .. N_EMBD)
    QWeight : k_w  ← split from fused qkv (rows N_EMBD .. 2*N_EMBD)
    QWeight : v_w  ← split from fused qkv (rows 2*N_EMBD .. 3*N_EMBD)
    QWeight : proj_w
    fp32    : proj_b
    QWeight : expand_w
    fp32    : expand_b
    QWeight : contract_w
    fp32    : contract_b
fp32 : ln_f.weight, ln_f.bias, lm_head  (lm_head kept fp32)

QWeight on disk: [int32 rows][int32 cols][int8 data][float32 scales]

Usage
─────
python export_weights.py --ckpt checkpoints/step_10000.pt --out weights/step_10000.bin
python export_weights.py --ckpt checkpoints/step_10000.pt --out weights/step_10000.bin --no-quant
"""

import argparse
import os
import struct
import numpy as np
import torch

# must match model.h
N_EMBD     = 256
N_HEAD     = 8
N_LAYER    = 6
BLOCK_SIZE = 64
VOCAB_SIZE = 50257
FF_DIM     = 4 * N_EMBD
#


def quantise(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row int8 symmetric quantisation. w ≈ data * scales[:, None]."""
    assert w.ndim == 2
    amax   = np.abs(w).max(axis=1, keepdims=True).clip(min=1e-9)
    scales = (amax / 127.0).astype(np.float32).squeeze(1)
    data   = np.clip(np.round(w / amax), -127, 127).astype(np.int8)
    return data, scales


def write_f32(f, arr: np.ndarray):
    f.write(arr.astype(np.float32).flatten().tobytes())


def write_qweight(f, w: np.ndarray, do_quant: bool):
    """Write a 2-D weight as a QWeight block."""
    assert w.ndim == 2, f"Expected 2D, got {w.shape}"
    rows, cols = w.shape
    f.write(struct.pack("ii", rows, cols))
    if do_quant:
        data, scales = quantise(w)
    else:
        scales = np.ones(rows, dtype=np.float32)
        data   = w.clip(-127, 127).astype(np.int8)
    f.write(data.tobytes())
    f.write(scales.tobytes())


def export(ckpt_path: str, out_path: str, do_quant: bool = True):
    print(f"Loading: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    sd = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    def get(key) -> np.ndarray:
        if key not in sd:
            raise KeyError(
                f"Missing key: '{key}'\n"
                f"First 10 keys: {list(sd.keys())[:10]}"
            )
        return sd[key].float().numpy()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"Writing → {out_path}  (quant={'int8' if do_quant else 'off'})")

    with open(out_path, "wb") as f:

        # Embeddings
        write_f32(f, get("tok_emb.weight"))   # (VOCAB_SIZE, N_EMBD)
        write_f32(f, get("pos_emb.weight"))   # (BLOCK_SIZE, N_EMBD)

        # Transformer blocks
        for l in range(N_LAYER):
            p = f"blocks.{l}"

            # LayerNorm (fp32 — small and precision-sensitive)
            write_f32(f, get(f"{p}.ln1.weight"))
            write_f32(f, get(f"{p}.ln1.bias"))
            write_f32(f, get(f"{p}.ln2.weight"))
            write_f32(f, get(f"{p}.ln2.bias"))

            # Split fused QKV (3*N_EMBD, N_EMBD) → three (N_EMBD, N_EMBD)
            qkv = get(f"{p}.attn.qkv.weight")   # (3*N_EMBD, N_EMBD)
            q_w, k_w, v_w = np.split(qkv, 3, axis=0)
            write_qweight(f, q_w,  do_quant)
            write_qweight(f, k_w,  do_quant)
            write_qweight(f, v_w,  do_quant)

            write_qweight(f, get(f"{p}.attn.proj.weight"), do_quant)
            write_f32(f,    get(f"{p}.attn.proj.bias"))

            write_qweight(f, get(f"{p}.mlp.expand.weight"), do_quant)
            write_f32(f,    get(f"{p}.mlp.expand.bias"))
            write_qweight(f, get(f"{p}.mlp.proj.weight"),   do_quant)
            write_f32(f,    get(f"{p}.mlp.proj.bias"))

        # Final norm + output projection
        write_f32(f, get("ln_f.weight"))
        write_f32(f, get("ln_f.bias"))
        # lm_head shares weights with tok_emb (weight tying) —
        # write tok_emb.weight transposed as lm_head (N_EMBD, VOCAB_SIZE)
        write_f32(f, get("tok_emb.weight").T)

    # Size report
    size_mb  = os.path.getsize(out_path) / 1024 / 1024
    fp32_mb  = sum(v.numel() * 4 for v in sd.values()) / 1024 / 1024
    print(f"\n  fp32 size  : {fp32_mb:.1f} MB")
    print(f"  export size: {size_mb:.1f} MB  ({100*size_mb/fp32_mb:.0f}% of fp32)")
    print(f"\n✓ Done: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     required=True, help="Path to .pt checkpoint")
    parser.add_argument("--out",      required=True, help="Output .bin path")
    parser.add_argument("--no-quant", action="store_true",
                        help="Disable int8 quantisation (for debugging)")
    args = parser.parse_args()
    export(args.ckpt, args.out, do_quant=not args.no_quant)