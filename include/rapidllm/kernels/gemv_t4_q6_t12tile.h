#pragma once

#include <cstdint>

namespace rapidllm::cuda_gemv {

// T=4 Q6 1-row using T=12's acc_q6_ild_from_blk tile + occ 3.
// Isolated TU — do not add to gemv_q6_t4.h.
void launch_q6k_f32_t4_1row_t12tile(const uint8_t* W, const float* X, float* Y, int m, int n, int add);

} // namespace rapidllm::cuda_gemv
