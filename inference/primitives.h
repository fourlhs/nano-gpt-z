#pragma once
#include <cmath>
#include <cstring>
#include <wasm_simd128.h>
#include "model.h"

// All functions are inline — avoids linker overhead and lets the
// compiler fold these into their call sites in the hot path.

// ─────────────────────────────────────────────────────────────
// LayerNorm  x = (x - mean) / std * w + b   (in-place)
// ─────────────────────────────────────────────────────────────
inline void layernorm(float* x, const float* w, const float* b, int n) {
    float mean = 0.f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    float var = 0.f;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    float inv = 1.f / sqrtf(var / n + 1e-5f);
    for (int i = 0; i < n; i++)
        x[i] = (x[i] - mean) * inv * w[i] + b[i];
}

// ─────────────────────────────────────────────────────────────
// GELU activation (OpenAI tanh approximation)
// ─────────────────────────────────────────────────────────────
inline void gelu(float* x, int n) {
    constexpr float C = 0.044715f;
    constexpr float S = 0.7978845608f;  // sqrt(2/pi)
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.f + tanhf(S * (v + C * v * v * v)));
    }
}

// ─────────────────────────────────────────────────────────────
// Softmax (in-place, numerically stable)
// ─────────────────────────────────────────────────────────────
inline void softmax(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0.f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    float inv = 1.f / s;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

// ─────────────────────────────────────────────────────────────
// SIMD int8 matmul   A(M×K) · B(K×N) → C(M×N)
// B is a QWeight (rows=N, cols=K) — each output neuron's weights
// are contiguous, which is SIMD-friendly.
// Processes 16 int8 elements per iteration via wasm_f32x4_*.
// ─────────────────────────────────────────────────────────────
inline void matmul_q(const float* __restrict__ A,
                     const QWeight& B,
                     float* __restrict__ C,
                     int M, int K, int N)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            const int8_t* brow  = B.data   + n * K;
            const float   scale = B.scales[n];

            v128_t acc = wasm_f32x4_splat(0.f);
            int k = 0;
            for (; k <= K - 16; k += 16) {
                // Dequantise 16 int8 → four f32x4 vectors
                v128_t b8    = wasm_v128_load(brow + k);
                v128_t b16lo = wasm_i16x8_extend_low_i8x16(b8);
                v128_t b16hi = wasm_i16x8_extend_high_i8x16(b8);
                v128_t s     = wasm_f32x4_splat(scale);
                v128_t bf0   = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(b16lo)),  s);
                v128_t bf1   = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(b16lo)), s);
                v128_t bf2   = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(b16hi)),  s);
                v128_t bf3   = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(b16hi)), s);
                // FMA with input row
                acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(A + m*K + k),      bf0));
                acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(A + m*K + k + 4),  bf1));
                acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(A + m*K + k + 8),  bf2));
                acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(A + m*K + k + 12), bf3));
            }
            // Horizontal reduce
            float tmp[4];
            wasm_v128_store(tmp, acc);
            float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            // Scalar tail for K % 16 != 0
            for (; k < K; k++)
                sum += A[m*K + k] * brow[k] * scale;
            C[m*N + n] = sum;
        }
    }
}

// ─────────────────────────────────────────────────────────────
// SIMD fp32 matmul   A(M×K) · B(K×N) → C(M×N)
// Used only for lm_head (kept fp32 for output precision).
// B is row-major: B[n, k] = B[n*K + k].
// ─────────────────────────────────────────────────────────────
inline void matmul_f(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C,
                     int M, int K, int N)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            v128_t acc = wasm_f32x4_splat(0.f);
            int k = 0;
            for (; k <= K - 4; k += 4) {
                acc = wasm_f32x4_add(acc,
                    wasm_f32x4_mul(wasm_v128_load(A + m*K + k),
                                   wasm_v128_load(B + n*K + k)));
            }
            float tmp[4];
            wasm_v128_store(tmp, acc);
            float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            for (; k < K; k++) sum += A[m*K + k] * B[n*K + k];
            C[m*N + n] = sum;
        }
    }
}