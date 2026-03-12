#include <cstring>
#include <cmath>
#include "model.h"
#include "primitives.h"

// ─────────────────────────────────────────────────────────────
// Cached multi-head self-attention — single new token forward.
//
//   x      : (N_EMBD,) — new token embedding, pre-normed by caller.
//            Attention output is ADDED into x (residual handled
//            in forward.cpp, not here).
//   bl     : transformer block holding Q/K/V/proj weights
//   cache  : layer KV store; k[layer] and v[layer] are updated
//   layer  : index into cache arrays
//   pos    : absolute position of the new token (0-indexed)
//
// Complexity: O(pos * HEAD_SIZE) per head — linear in sequence
// length because past K/V are read from cache, never recomputed.
// ─────────────────────────────────────────────────────────────
void attention_cached(float* x, Block& bl, KVCache& cache,
                      int layer, int pos)
{
    const int T = pos + 1;  // total tokens including new one

    float q[N_EMBD], k_new[N_EMBD], v_new[N_EMBD], out[N_EMBD];
    memset(out, 0, sizeof(out));

    // Project new token → Q, K, V
    matmul_q(x, bl.q_w, q,     1, N_EMBD, N_EMBD);
    matmul_q(x, bl.k_w, k_new, 1, N_EMBD, N_EMBD);
    matmul_q(x, bl.v_w, v_new, 1, N_EMBD, N_EMBD);

    // Append new K, V into cache at position `pos`
    memcpy(cache.k[layer] + pos * N_EMBD, k_new, N_EMBD * sizeof(float));
    memcpy(cache.v[layer] + pos * N_EMBD, v_new, N_EMBD * sizeof(float));

    const float scale = 1.f / sqrtf((float)HEAD_SIZE);

    for (int h = 0; h < N_HEAD; h++) {
        const int off = h * HEAD_SIZE;

        // Attention scores: Q_h · K_h^T over all cached positions
        // Shape: (T,) — new token attends to [0 .. pos] (causal)
        float scores[BLOCK_SIZE];
        for (int j = 0; j < T; j++) {
            float dot = 0.f;
            const float* kj = cache.k[layer] + j * N_EMBD + off;
            for (int d = 0; d < HEAD_SIZE; d++)
                dot += q[off + d] * kj[d];
            scores[j] = dot * scale;
        }
        softmax(scores, T);

        // Weighted sum of cached V_h → out_h
        for (int j = 0; j < T; j++) {
            const float* vj = cache.v[layer] + j * N_EMBD + off;
            for (int d = 0; d < HEAD_SIZE; d++)
                out[off + d] += scores[j] * vj[d];
        }
    }

    // Output projection + bias, accumulated into x
    float proj_out[N_EMBD];
    matmul_q(out, bl.proj_w, proj_out, 1, N_EMBD, N_EMBD);
    for (int i = 0; i < N_EMBD; i++)
        x[i] += proj_out[i] + bl.proj_b[i];
}