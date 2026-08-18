#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace rapidllm::cuda_gemv {

// Isolated T=2 Q4 dual 1-row (wg+wu). Same dequant as launch_q4k_q8_t2_1row.
// Do not add this to gemv_q6_t4.h.
void launch_q4k_q8_t2_dual_1row(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                                float* Y1, float* Y2, int m, int n);

} // namespace rapidllm::cuda_gemv
