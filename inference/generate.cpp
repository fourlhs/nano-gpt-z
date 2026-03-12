#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "model.h"

void load_weights(GPT& model, const char* path);
void forward_step(GPT& model, KVCache& cache, int token, int pos, float* logits);

// ─────────────────────────────────────────────────────────────
// Global model + cache state
// Only this translation unit touches these — keeps global state
// contained and easy to reason about.
// ─────────────────────────────────────────────────────────────
static GPT     g_model;
static KVCache g_cache;
static bool    g_loaded = false;

void model_load(const char* path) {
    load_weights(g_model, path);
    g_loaded = true;
}

// ─────────────────────────────────────────────────────────────
// Top-p nucleus sampling
//
// temperature : scales logit sharpness (0 = greedy)
// top_p       : cumulative probability cutoff (e.g. 0.9)
//
// Sorts vocab by probability, keeps the smallest set summing
// to >= top_p, renormalises, then samples. Produces more
// coherent text than temperature alone.
// ─────────────────────────────────────────────────────────────
struct IndexedLogit {
    float val; int idx;
    bool operator<(const IndexedLogit& o) const { return val > o.val; }
};

static int sample(float* logits, float temperature, float top_p) {
    if (temperature <= 0.f) {
        // Greedy — argmax
        int best = 0;
        for (int i = 1; i < VOCAB_SIZE; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    // Apply temperature then softmax
    for (int i = 0; i < VOCAB_SIZE; i++) logits[i] /= temperature;
    float mx = logits[0];
    for (int i = 1; i < VOCAB_SIZE; i++) if (logits[i] > mx) mx = logits[i];
    float s = 0.f;
    for (int i = 0; i < VOCAB_SIZE; i++) { logits[i] = expf(logits[i] - mx); s += logits[i]; }
    for (int i = 0; i < VOCAB_SIZE; i++) logits[i] /= s;

    // Sort descending to find nucleus
    static IndexedLogit buf[VOCAB_SIZE];
    for (int i = 0; i < VOCAB_SIZE; i++) buf[i] = {logits[i], i};
    std::sort(buf, buf + VOCAB_SIZE);

    // Find cutoff: smallest k such that sum(buf[0..k]) >= top_p
    float cum = 0.f;
    int cutoff = VOCAB_SIZE;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        cum += buf[i].val;
        if (cum >= top_p) { cutoff = i + 1; break; }
    }

    // Renormalise over nucleus
    float ns = 0.f;
    for (int i = 0; i < cutoff; i++) ns += buf[i].val;
    float inv = 1.f / ns;

    // Sample
    float r = (float)rand() / RAND_MAX;
    float c = 0.f;
    for (int i = 0; i < cutoff; i++) {
        c += buf[i].val * inv;
        if (r < c) return buf[i].idx;
    }
    return buf[cutoff - 1].idx;
}

// ─────────────────────────────────────────────────────────────
// Prefill: run all prompt tokens through the model, populating
// the KV cache. Must be called before decode_step.
// ─────────────────────────────────────────────────────────────
void prefill(const int* tokens, int len) {
    g_cache.len = 0;
    float logits[VOCAB_SIZE];
    for (int i = 0; i < len; i++) {
        forward_step(g_model, g_cache, tokens[i], g_cache.len, logits);
        g_cache.len++;
    }
}

// ─────────────────────────────────────────────────────────────
// Decode one token given the last token and current cache state.
// Returns the sampled token id.
// Caller is responsible for stopping at EOS or BLOCK_SIZE.
// ─────────────────────────────────────────────────────────────
int decode_step(int last_token, float temperature, float top_p) {
    if (g_cache.len >= BLOCK_SIZE) return -1;  // context window full
    float logits[VOCAB_SIZE];
    forward_step(g_model, g_cache, last_token, g_cache.len, logits);
    int next = sample(logits, temperature, top_p);
    g_cache.len++;
    return next;
}

// ─────────────────────────────────────────────────────────────
// Convenience: full generate (prefill + decode loop).
// Returns heap-allocated int[prompt_len + max_new_tokens].
// Caller must free with seq_free().
// ─────────────────────────────────────────────────────────────
int* generate(const int* prompt, int prompt_len, int max_new_tokens,
              float temperature, float top_p)
{
    int total = prompt_len + max_new_tokens;
    int* seq  = new int[total];
    memcpy(seq, prompt, prompt_len * sizeof(int));

    prefill(prompt, prompt_len);

    for (int step = 0; step < max_new_tokens; step++) {
        int last = seq[prompt_len + step - 1 + (step == 0 ? 0 : 0)];
        last = (step == 0) ? seq[prompt_len - 1] : seq[prompt_len + step - 1];
        int next = decode_step(last, temperature, top_p);
        if (next < 0) break;  // context window hit
        seq[prompt_len + step] = next;
    }

    return seq;
}

void seq_free(int* seq) { delete[] seq; }