#pragma once

#include <cstdint>

namespace rapidllm::cuda_gemv {

// Isolated T=4 Q6 1-row: X tile in smem so the 16 x-regs can drop and
// occupancy can rise. Same dequant as gemv_t4_1row acc_q6_ild_t4.
// Do not add this to gemv_q6_t4.h.
void launch_q6k_f32_t4_xs(const uint8_t* W, const float* X, float* Y, int m, int n, int add);

} // namespace rapidllm::cuda_gemv
