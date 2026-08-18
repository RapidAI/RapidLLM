#pragma once

#include <cstdint>

namespace rapidllm::cuda_gemv {

// Isolated T=1 Q6 2-row + cp.async. Do not add this to gemv_q6_t4.h.
void launch_q6k_f32_t1_2row(const uint8_t* W, const float* x, float* y, int m, int n, int add);

} // namespace rapidllm::cuda_gemv
