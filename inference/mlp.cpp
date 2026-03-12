#include "model.h"
#include "primitives.h"

// MLP sub-block — single token, pre-normed input.
//
//   x  : (N_EMBD,) pre-normed. MLP output is ADDED into x.
//        Residual is handled in forward.cpp.
//
// Architecture: expand (N_EMBD → FF_DIM) → GELU → contract (FF_DIM → N_EMBD)
// Both weight matrices are int8-quantised; biases stay fp32.
void mlp_single(float* x, Block& bl) {
    float h[FF_DIM], out[N_EMBD];

    // Expand + bias
    matmul_q(x, bl.expand_w, h, 1, N_EMBD, FF_DIM);
    for (int i = 0; i < FF_DIM; i++) h[i] += bl.expand_b[i];

    gelu(h, FF_DIM);

    // Contract + bias, accumulated into x
    matmul_q(h, bl.contract_w, out, 1, FF_DIM, N_EMBD);
    for (int i = 0; i < N_EMBD; i++)
        x[i] += out[i] + bl.contract_b[i];
}