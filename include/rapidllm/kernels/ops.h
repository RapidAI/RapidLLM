#pragma once

#include "rapidllm/ir/model_desc.h"

#include <cstddef>
#include <cstdint>

namespace rapidllm::ops {

float silu(float x);
float fp16_to_f32(uint16_t h);
uint16_t f32_to_fp16(float f);
float e4m3_to_f32(uint8_t b);
uint8_t f32_to_e4m3(float f);

void rmsnorm(const float* x, const float* w, float* y, int n, float eps);
void qwen3_rmsnorm(const float* x, const float* gamma, float* y, int n, float eps);
void gated_rmsnorm(const float* x, const float* z, const float* gamma, float* y, int n, float eps,
                   int gamma_n = 0);
void l2norm(const float* x, float* y, int n, float eps);

struct QuantW {
    const uint8_t* data = nullptr;
    const float* scale = nullptr;
    QuantKind quant = QuantKind::F32;
};

void gemv_f32_scalar(const float* W, const float* x, float* y, int m, int n);
void gemv_f32_simd(const float* W, const float* x, float* y, int m, int n);
void gemv_f32(const float* W, const float* x, float* y, int m, int n);
void gemv_f16(const uint16_t* W, const float* x, float* y, int m, int n);
void gemm_f32(const float* W, const float* X, float* Y, int m, int n, int batch);

void delta_recurrent_step_scalar(float* S, const float* q, const float* k, const float* v,
                                 const float* beta, const float* g_log, float* o,
                                 int n_v, int dk, int dv, float eps_l2);
void delta_recurrent_step_simd(float* S, const float* q, const float* k, const float* v,
                               const float* beta, const float* g_log, float* o,
                               int n_v, int dk, int dv, float eps_l2);

void fused_swiglu(const float* gate, const float* up, float* y, int n);
void fused_rmsnorm_residual(const float* x, const float* y, const float* w, float* out, int n, float eps);

struct FusedDeltaArgs {
    const float* x;
    float* x_inout; // residual in-place
    const float* rms_w;
    QuantW Wqkv, Wz, Wa, Wb, Wo;
    const float* A_log;
    const float* dt_bias;
    const float* conv_w;
    const float* gnorm;
    float* S;
    float* conv_state;
    int hidden, nk, nv, dk, dv, conv_k;
    int gnorm_n = 0;
    float eps;
    bool use_simd;
};
void fused_delta_decode(const FusedDeltaArgs& a);

struct FusedAttnArgs {
    const float* x;
    float* x_inout;
    const float* rms_w;
    QuantW Wq, Wk, Wv, Wo;
    const float* q_norm, *k_norm;
    float* k_cache, *v_cache;
    int hidden, n_q, n_kv, head_dim, rotary_dim, pos;
    float theta, eps;
    bool use_simd;
};
void fused_attn_decode(const FusedAttnArgs& a);

struct FusedMlpArgs {
    const float* x;
    float* x_inout;
    const float* rms_w;
    QuantW Wg, Wu, Wd;
    int hidden, intermediate;
    float eps;
    bool use_simd;
};
void fused_mlp_decode(const FusedMlpArgs& a);

const char* simd_isa_name();
// cpu_caps.h: runtime AVX2 / AVX-512 detect + gemv_f32_avx2 / gemv_f32_avx512

void gemv_fp8(const uint8_t* W, const float* scale, const float* x, float* y,
              int m, int n, int block);

void dequant_q8_0(const uint8_t* packed, float* out, int n);
void gemv_q8_0_scalar(const uint8_t* packed, const float* x, float* y, int m, int n);
void gemv_q8_0(const uint8_t* packed, const float* x, float* y, int m, int n);
void gemv_f16_scalar(const uint16_t* W, const float* x, float* y, int m, int n);

void dequant_q4_k(const uint8_t* packed, float* out, int n);
void dequant_q5_k(const uint8_t* packed, float* out, int n);
void dequant_q6_k(const uint8_t* packed, float* out, int n);
void gemv_q4_k(const uint8_t* packed, const float* x, float* y, int m, int n);

void linear(QuantKind q, const uint8_t* w, const float* scale, const float* x, float* y,
            int m, int n);
void linear(const QuantW& w, const float* x, float* y, int m, int n);

void gemv_bf16(const uint16_t* W, const float* x, float* y, int m, int n);
void gemv_q4_k_scalar(const uint8_t* packed, const float* x, float* y, int m, int n);
void gemv_q5_k_scalar(const uint8_t* packed, const float* x, float* y, int m, int n);
void gemv_q6_k_scalar(const uint8_t* packed, const float* x, float* y, int m, int n);
void gemv_q5_k(const uint8_t* packed, const float* x, float* y, int m, int n);
void gemv_q6_k(const uint8_t* packed, const float* x, float* y, int m, int n);

void conv1d_causal_prefill(const float* x, const float* w, float* y, int seq, int dim, int k);
void conv1d_update(const float* x_t, const float* w, float* state, float* y, int dim, int k);

void rope_partial(float* q, float* k, int n_q, int n_kv, int head_dim, int rotary_dim,
                  int pos, float theta);

void delta_recurrent_step(float* S, const float* q, const float* k, const float* v,
                          const float* beta, const float* g_log, float* o,
                          int n_v, int dk, int dv, float eps_l2);

void attn_prefill(const float* q, const float* k, const float* v, float* o,
                  int seq, int n_q, int n_kv, int head_dim);
void attn_decode(const float* q, const float* k_cache, const float* v_cache, float* o,
                 int pos, int n_q, int n_kv, int head_dim);

void swiglu(const float* gate, const float* up, float* y, int n);

} // namespace rapidllm::ops
