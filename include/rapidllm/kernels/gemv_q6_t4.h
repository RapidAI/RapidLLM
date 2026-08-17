#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace rapidllm::cuda_gemv {

// T=2 Q6 ild + Q4 int (isolated TU). Same dequant as T=3 ild / T=3 Q4.
void launch_q6k_f32_t2_ild(const uint8_t* W, const float* X, float* Y, int m, int n, int add);
void launch_q4k_q8_t2(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n, int add);
void launch_q4k_q8_t2_dual(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc, float* Y1,
                           float* Y2, int m, int n);
void launch_q6k_f32_t2_1row(const uint8_t* W, const float* X, float* Y, int m, int n, int add);
void launch_q4k_q8_t2_1row(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n, int add);

// T=3/T=4 Q6 float GEMV (isolated TU, cp.async W pipeline).
void launch_q6k_f32_t3_ild(const uint8_t* W, const float* X, float* Y, int m, int n, int add);
void launch_q6k_f32_t4_ild(const uint8_t* W, const float* X, float* Y, int m, int n, int add);
void launch_q6k_f32_t6_ild(const uint8_t* W, const float* X, float* Y, int m, int n, int add);
void launch_q6k_f32_t12_ild(const uint8_t* W, const float* X, float* Y, int m, int n, int add);
void launch_q6k_f32_t12_1row(const uint8_t* W, const float* X, float* Y, int m, int n, int add);
void launch_q4k_q8_t12_1row(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n,
                            int add);

// T=4 Q4 integer dual (wg+wu): one x, two W. fuse_swiglu is applied by the caller.
void launch_q4k_q8_t4_dual(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                           float* Y1, float* Y2, int m, int n);
void launch_q4k_q8_t6(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n, int add);
void launch_q4k_q8_t6_dual(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                           float* Y1, float* Y2, int m, int n);
void launch_q4k_q8_t12(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n, int add);
void launch_q4k_q8_t12_dual(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc, float* Y1,
                            float* Y2, int m, int n);

// T=4 mixed Q6 ild + Q4 int (wqkv + wz).
void launch_q6q4_t4_dual(const uint8_t* W6, const uint8_t* W4, const float* X, const int8_t* xq,
                         const __half* xsc, float* Y6, float* Y4, int m6, int m4, int n);

void warmup_gdn_decode_t12();
void launch_gdn_decode_t12(const float* aa, const float* bb, uint16_t* S, const float* A_log,
                           const float* dt_bias, float* o, int nk, int nv, const float* qkv_raw,
                           const float* conv_w, float* conv_st, uint16_t* S_bak, float* conv_bak,
                           int qkv_dim, int v_tiled);

} // namespace rapidllm::cuda_gemv
