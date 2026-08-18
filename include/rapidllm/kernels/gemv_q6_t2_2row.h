#pragma once

#include <cstdint>

namespace rapidllm::cuda_gemv {

// Isolated T=2 Q6 2-row + cp.async. Same dequant as launch_q6k_f32_t2_1row.
// Do not add this to gemv_q6_t4.h.
void launch_q6k_f32_t2_2row(const uint8_t* W, const float* X, float* Y, int m, int n, int add);

} // namespace rapidllm::cuda_gemv
