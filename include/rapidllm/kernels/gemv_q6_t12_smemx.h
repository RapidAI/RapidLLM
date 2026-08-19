#pragma once

#include <cstdint>

namespace rapidllm::cuda_gemv {

// Isolated T=12 Q6 GEMV with a block-shared 12x256 X tile.
// Do not add this declaration to gemv_q6_t4.h.
void launch_q6k_f32_t12_smemx(const uint8_t* W, const float* X, float* Y, int m, int n, int add);

} // namespace rapidllm::cuda_gemv
