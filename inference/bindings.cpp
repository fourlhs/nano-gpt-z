#include <emscripten.h>

// Declarations from generate.cpp — this file contains zero logic.
// If you ever swap Emscripten for a different runtime (Node native
// addon, Python bindings, etc.) you delete this file and write a
// new one. Everything else stays untouched.
void  model_load(const char* path);
void  prefill(const int* tokens, int len);
int   decode_step(int last_token, float temperature, float top_p);
int*  generate(const int* prompt, int prompt_len, int max_new_tokens,
               float temperature, float top_p);
void  seq_free(int* seq);

extern "C" {

// Load weights from a binary file path (called once at startup)
EMSCRIPTEN_KEEPALIVE
void wasm_model_load(const char* path) {
    model_load(path);
}

// Prefill the KV cache with a prompt token sequence
EMSCRIPTEN_KEEPALIVE
void wasm_prefill(int* tokens, int len) {
    prefill(tokens, len);
}

// Decode one token; call repeatedly to stream output
// Returns -1 when context window is full
EMSCRIPTEN_KEEPALIVE
int wasm_decode_step(int last_token, float temperature, float top_p) {
    return decode_step(last_token, temperature, top_p);
}

// Full generate loop — returns pointer to int[prompt_len + max_new_tokens]
// JS must call wasm_seq_free() when done to avoid memory leaks
EMSCRIPTEN_KEEPALIVE
int* wasm_generate(int* prompt, int prompt_len, int max_new_tokens,
                   float temperature, float top_p) {
    return generate(prompt, prompt_len, max_new_tokens, temperature, top_p);
}

EMSCRIPTEN_KEEPALIVE
void wasm_seq_free(int* seq) {
    seq_free(seq);
}

} // extern "C"