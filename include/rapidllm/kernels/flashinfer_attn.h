#pragma once

#include <cstddef>
#include <cstdint>

#if defined(RAPIDLLM_FLASHINFER)
// FlashInfer SinglePrefill. hd must be 256, F16 KV NHD. Group size must be
// 1/2/3/4/8 — official Qwen3.8-27B is 24/4=6 and always returns false.
bool fi_prefill_gqa(const float* q, const void* k_f16, const void* v_f16, float* o, int qo_len,
                    int kv_len, int n_q, int n_kv, int hd, float* scratch, size_t scratch_floats);
bool fi_available();
#else
inline bool fi_prefill_gqa(const float*, const void*, const void*, float*, int, int, int, int, int,
                           float*, size_t) {
    return false;
}
inline bool fi_available() { return false; }
#endif
