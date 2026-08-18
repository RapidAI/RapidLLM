#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace rapidllm::cuda_gemv {

// Isolated T=4 Q4 1-row / dual. Same dequant as gemv_t4_pipe.cu, but the
// x-only dp4a(1,xq) sum is computed once per K-block instead of per row.
// Do not add to gemv_q6_t4.h.
void launch_q4k_q8_t4_1row_sx(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n,
                              int add);
void launch_q4k_q8_t4_1row_dual_sx(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                                   float* Y1, float* Y2, int m, int n);

} // namespace rapidllm::cuda_gemv
