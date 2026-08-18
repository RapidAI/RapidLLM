#pragma once

#include <cstdint>

namespace rapidllm::cuda_gemv {

// T=4 Q6 1-row, same dequant as gemv_t4_1row. K-unroll 2 + occ 4.
// Isolated TU — do not add to gemv_q6_t4.h.
void launch_q6k_f32_t4_1row_ku(const uint8_t* W, const float* X, float* Y, int m, int n, int add);

} // namespace rapidllm::cuda_gemv
