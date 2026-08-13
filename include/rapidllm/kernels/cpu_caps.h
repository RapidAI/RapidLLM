#pragma once

#include <cstdint>

namespace rapidllm::ops {

struct CpuCaps {
    bool sse2 = false;
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512dq = false;
    bool avx512bw = false;
    bool avx512vl = false;
};

const CpuCaps& cpu_caps();

enum class SimdKind { Scalar = 0, Avx2 = 1, Avx512 = 2 };

// Highest ISA that is both compiled in and present at runtime.
// Override with env RAPIDLLM_SIMD=scalar|avx2|avx512 (must still be available).
SimdKind active_simd();

void gemv_f32_avx2(const float* W, const float* x, float* y, int m, int n);
void gemv_f32_avx512(const float* W, const float* x, float* y, int m, int n);
void gemv_q8_0_avx2(const uint8_t* W, const float* x, float* y, int m, int n);
void gemv_q8_0_avx512(const uint8_t* W, const float* x, float* y, int m, int n);
void gemv_f16_avx2(const uint16_t* W, const float* x, float* y, int m, int n);
void gemv_f16_avx512(const uint16_t* W, const float* x, float* y, int m, int n);
void gemv_bf16_avx2(const uint16_t* W, const float* x, float* y, int m, int n);
void gemv_bf16_avx512(const uint16_t* W, const float* x, float* y, int m, int n);
void qwen3_rmsnorm_avx2(const float* x, const float* gamma, float* y, int n, float eps);
void qwen3_rmsnorm_avx512(const float* x, const float* gamma, float* y, int n, float eps);
void delta_recurrent_step_avx2(float* S, const float* q, const float* k, const float* v,
                               const float* beta, const float* g_log, float* o, int n_v, int dk,
                               int dv, float eps_l2);
void delta_recurrent_step_avx512(float* S, const float* q, const float* k, const float* v,
                                 const float* beta, const float* g_log, float* o, int n_v, int dk,
                                 int dv, float eps_l2);

} // namespace rapidllm::ops
