#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace rapidllm::cuda_gemv {

// Isolated T=2 1-row HBM path. Same Q6/Q4 dequant as launch_q6k_f32_t2_1row /
// launch_q4k_q8_t2_1row. Do not declare these in gemv_q6_t4.h.
void launch_q6k_f32_t2_hbm(const uint8_t* W, const float* X, float* Y, int m, int n, int add);
void launch_q4k_q8_t2_hbm(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n, int add);

} // namespace rapidllm::cuda_gemv
