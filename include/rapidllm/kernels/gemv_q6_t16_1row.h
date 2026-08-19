#pragma once

#include <cstdint>

namespace rapidllm::cuda_gemv {

// Isolated T=16 Q6 1-row. Same acc_q6_ild_from_blk as T=12 1-row (four T=4
// groups). Do not add to gemv_q6_t4.h.
void launch_q6k_f32_t16_1row(const uint8_t* W, const float* X, float* Y, int m, int n, int add);

} // namespace rapidllm::cuda_gemv
