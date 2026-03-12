#pragma once
#include <cstdint>
#include <cmath>
#include <algorithm>

// ─────────────────────────────────────────────────────────────
// Model dimensions — must match model.py
// ─────────────────────────────────────────────────────────────
static constexpr int N_EMBD     = 256;
static constexpr int N_HEAD     = 8;
static constexpr int N_LAYER    = 6;
static constexpr int BLOCK_SIZE = 64;
static constexpr int VOCAB_SIZE = 50257;
static constexpr int HEAD_SIZE  = N_EMBD / N_HEAD;  // 32
static constexpr int FF_DIM     = 4 * N_EMBD;       // 1024

// ─────────────────────────────────────────────────────────────
// Int8 quantised weight matrix
// Each output row has its own float scale: w_float = w_int8 * scale
// Halves .bin size vs fp32 — critical for WASM network load time.
// lm_head is kept fp32 (output projection precision matters).
// ─────────────────────────────────────────────────────────────
struct QWeight {
    int8_t* data   = nullptr;
    float*  scales = nullptr;
    int     rows   = 0;
    int     cols   = 0;

    // Called at load time: quantise a row-major fp32 matrix in place
    void quantise(const float* src, int R, int C) {
        rows = R; cols = C;
        data   = new int8_t[R * C];
        scales = new float[R];
        for (int r = 0; r < R; r++) {
            float amax = 1e-9f;
            for (int c = 0; c < C; c++)
                amax = std::max(amax, fabsf(src[r*C + c]));
            scales[r]  = amax / 127.f;
            float inv  = 127.f / amax;
            for (int c = 0; c < C; c++)
                data[r*C + c] = (int8_t)std::max(-127.f,
                                 std::min(127.f, src[r*C + c] * inv));
        }
    }

    ~QWeight() { delete[] data; delete[] scales; }

    // Non-copyable — owns heap memory
    QWeight()                          = default;
    QWeight(const QWeight&)            = delete;
    QWeight& operator=(const QWeight&) = delete;
};

// ─────────────────────────────────────────────────────────────
// One transformer block
// ─────────────────────────────────────────────────────────────
struct Block {
    // LayerNorm params (fp32 — small, kept exact)
    float* ln1_w = nullptr;  float* ln1_b = nullptr;
    float* ln2_w = nullptr;  float* ln2_b = nullptr;

    // Attention projections (quantised)
    QWeight q_w, k_w, v_w;
    QWeight proj_w;
    float*  proj_b = nullptr;   // bias stays fp32

    // MLP projections (quantised)
    QWeight expand_w;
    float*  expand_b   = nullptr;
    QWeight contract_w;
    float*  contract_b = nullptr;
};

// ─────────────────────────────────────────────────────────────
// Full model
// ─────────────────────────────────────────────────────────────
struct GPT {
    Block  blocks[N_LAYER];
    float* tok_emb = nullptr;   // (VOCAB_SIZE, N_EMBD)
    float* pos_emb = nullptr;   // (BLOCK_SIZE, N_EMBD)
    float* ln_f_w  = nullptr;
    float* ln_f_b  = nullptr;
    float* lm_head = nullptr;   // (VOCAB_SIZE, N_EMBD) — fp32
};

// ─────────────────────────────────────────────────────────────
// Per-layer KV cache — avoids recomputing K/V for past tokens.
// Decode step is O(T) instead of O(T²).
// ─────────────────────────────────────────────────────────────
struct KVCache {
    float k[N_LAYER][BLOCK_SIZE * N_EMBD];
    float v[N_LAYER][BLOCK_SIZE * N_EMBD];
    int   len = 0;  // number of tokens currently cached
};