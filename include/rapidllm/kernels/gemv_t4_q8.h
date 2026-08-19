#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace rapidllm::cuda_gemv {

// Isolated T=4 Q8 1-row / dual. Same SoA dequant as gemm_q8_soa_k
// (scale * int8·float32). No block-wide X smem / __syncthreads.
// Do not add to gemv_q6_t4.h.
void launch_q8_f32_t4_1row(const int8_t* Q, const __half* scales, const float* X, float* Y, int m, int n,
                           int add);
void launch_q8_f32_t4_1row_dual(const int8_t* Q1, const __half* S1, const int8_t* Q2, const __half* S2,
                                const float* X, float* Y1, float* Y2, int m, int n);
void launch_q8_f32_t12(const int8_t* Q, const __half* scales, const float* X, float* Y, int m, int n, int add);

} // namespace rapidllm::cuda_gemv
