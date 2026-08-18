#pragma once

#include <cstdint>

namespace rapidllm::cuda_gemv {

// Isolated T=4 Q6 1-row, same fma order as gemv_t4_1row.cu acc_q6_ild_t4
// but X via __ldg (no 16 live X regs) so occupancy can rise.
// Do not add to gemv_q6_t4.h.
void launch_q6k_f32_t4_1row_ho(const uint8_t* W, const float* X, float* Y, int m, int n, int add);

} // namespace rapidllm::cuda_gemv
