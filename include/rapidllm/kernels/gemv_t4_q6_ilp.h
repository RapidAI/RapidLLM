#pragma once

#include <cstdint>

namespace rapidllm::cuda_gemv {

// T=4 Q6 1-row, same dequant as gemv_t4_1row.cu, split FMA chains for ILP.
void launch_q6k_f32_t4_1row_ilp(const uint8_t* W, const float* X, float* Y, int m, int n, int add);

} // namespace rapidllm::cuda_gemv
