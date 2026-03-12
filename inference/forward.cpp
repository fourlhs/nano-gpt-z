#include <cstdio>
#include <cstring>
#include "model.h"
#include "primitives.h"

// Declarations from their respective translation units
void attention_cached(float* x, Block& bl, KVCache& cache, int layer, int pos);
void mlp_single(float* x, Block& bl);

// Weight loading
// Binary format (written by export_weights.py):
//   fp32 arrays  : tok_emb, pos_emb
//   per layer    : ln1_w, ln1_b, ln2_w, ln2_b  (fp32)
//                  q_w, k_w, v_w, proj_w        (QWeight)
//                  proj_b                        (fp32)
//                  expand_w, contract_w          (QWeight)
//                  expand_b, contract_b          (fp32)
//   fp32 arrays  : ln_f_w, ln_f_b, lm_head
//
// QWeight on disk: [int32 rows][int32 cols][int8 data][float scales]
#define READ_F32(ptr, n, f) \
    ptr = new float[n];     \
    fread(ptr, sizeof(float), n, f);

static void read_qweight(QWeight& w, FILE* f) {
    fread(&w.rows,   sizeof(int), 1, f);
    fread(&w.cols,   sizeof(int), 1, f);
    w.data   = new int8_t[w.rows * w.cols];
    w.scales = new float[w.rows];
    fread(w.data,   sizeof(int8_t), w.rows * w.cols, f);
    fread(w.scales, sizeof(float),  w.rows,           f);
}

void load_weights(GPT& model, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open weights file: %s\n", path);
        return;
    }

    READ_F32(model.tok_emb, VOCAB_SIZE * N_EMBD, f)
    READ_F32(model.pos_emb, BLOCK_SIZE * N_EMBD, f)

    for (int l = 0; l < N_LAYER; l++) {
        Block& b = model.blocks[l];
        READ_F32(b.ln1_w, N_EMBD, f)  READ_F32(b.ln1_b, N_EMBD, f)
        READ_F32(b.ln2_w, N_EMBD, f)  READ_F32(b.ln2_b, N_EMBD, f)
        read_qweight(b.q_w,        f);
        read_qweight(b.k_w,        f);
        read_qweight(b.v_w,        f);
        read_qweight(b.proj_w,     f);  READ_F32(b.proj_b,     N_EMBD,  f)
        read_qweight(b.expand_w,   f);  READ_F32(b.expand_b,   FF_DIM,  f)
        read_qweight(b.contract_w, f);  READ_F32(b.contract_b, N_EMBD,  f)
    }

    READ_F32(model.ln_f_w, N_EMBD,             f)
    READ_F32(model.ln_f_b, N_EMBD,             f)
    READ_F32(model.lm_head, N_EMBD * VOCAB_SIZE, f)  // lm_head kept fp32

    fclose(f);
}

// Single-token forward pass (KV-cached)
//
//   model  : loaded GPT weights
//   cache  : KV state; updated in place for the new token
//   token  : token id of the new token
//   pos    : absolute position in the sequence (0-indexed)
//   logits : output buffer of size VOCAB_SIZE — caller owns
//
// Pre-norm residual stream (GPT-2 convention):
//   xn = layernorm(x)         ← normed copy
//   xn = xn + attn/mlp(xn)   ← transform accumulates into xn
//   x  = x  + xn              ← add back to residual stream
void forward_step(GPT& model, KVCache& cache,
                  int token, int pos, float* logits)
{
    // Token + positional embedding
    float x[N_EMBD];
    for (int d = 0; d < N_EMBD; d++)
        x[d] = model.tok_emb[token * N_EMBD + d]
             + model.pos_emb[pos   * N_EMBD + d];

    for (int l = 0; l < N_LAYER; l++) {
        Block& bl = model.blocks[l];
        float  xn[N_EMBD];

        // Attention sub-block
        memcpy(xn, x, sizeof(x));
        layernorm(xn, bl.ln1_w, bl.ln1_b, N_EMBD);  // norm the copy
        attention_cached(xn, bl, cache, l, pos);      // xn += attn(xn)
        for (int i = 0; i < N_EMBD; i++) x[i] += xn[i]; // residual

        // MLP sub-block
        memcpy(xn, x, sizeof(x));
        layernorm(xn, bl.ln2_w, bl.ln2_b, N_EMBD);
        mlp_single(xn, bl);
        for (int i = 0; i < N_EMBD; i++) x[i] += xn[i]; // residual
    }

    // Final layernorm + project to vocab
    layernorm(x, model.ln_f_w, model.ln_f_b, N_EMBD);
    matmul_f(x, model.lm_head, logits, 1, N_EMBD, VOCAB_SIZE);
}