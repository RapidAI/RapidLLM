#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace rapidllm::cuda_gemv {

// Isolated T=4 GEMV: 4 warps/block (128 thr) so more blocks fill Ada SMs.
// Same dequant as gemv_t4_1row / gemv_t4_pipe. Do not add to gemv_q6_t4.h.
void launch_q6k_f32_t4_occ(const uint8_t* W, const float* X, float* Y, int m, int n, int add);
void launch_q4k_q8_t4_occ(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n,
                          int add);
void launch_q4k_q8_t4_dual_occ(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                               float* Y1, float* Y2, int m, int n);

} // namespace rapidllm::cuda_gemv
