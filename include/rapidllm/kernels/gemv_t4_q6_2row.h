#pragma once

#include <cstdint>

namespace rapidllm::cuda_gemv {

// Isolated T=4 Q6 2-row. Same float dequant as gemv_t4_1row.cu acc_q6_ild_t4.
// Do not add to gemv_q6_t4.h (T=3 ild codegen).
void launch_q6k_f32_t4_2row(const uint8_t* W, const float* X, float* Y, int m, int n, int add);

} // namespace rapidllm::cuda_gemv
