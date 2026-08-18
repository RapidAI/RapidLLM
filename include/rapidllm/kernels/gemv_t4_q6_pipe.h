#pragma once

#include <cstdint>

namespace rapidllm::cuda_gemv {

// Isolated T=4 Q6 1-row, 3-stage cp.async W. Same float dequant as
// gemv_t4_1row.cu. Do not add to gemv_q6_t4.h.
void launch_q6k_f32_t4_1row_pipe(const uint8_t* W, const float* X, float* Y, int m, int n, int add);

} // namespace rapidllm::cuda_gemv
