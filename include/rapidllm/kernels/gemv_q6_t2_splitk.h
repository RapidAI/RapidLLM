#pragma once

#include <cstdint>

namespace rapidllm::cuda_gemv {

// Isolated T=2 Q6 split-K for long-K down-proj (add=1). Same dequant as
// launch_q6k_f32_t2_1row. Do not add this to gemv_q6_t4.h.
void launch_q6k_f32_t2_splitk(const uint8_t* W, const float* X, float* Y, int m, int n);

} // namespace rapidllm::cuda_gemv
