#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace rapidllm::cuda_gemv {

// Isolated T=4 Q4: same pipe dequant as gemv_t4_pipe, but group xsum comes
// from ensure_xq (g_xsum) instead of 8 extra dp4a. One lane per group applies
// the dm term. Do not add to gemv_q6_t4.h.
void launch_q4k_q8_t4_1row_xs(const uint8_t* W, const int8_t* xq, const __half* xsc, const int32_t* xsum,
                              float* Y, int m, int n, int add);
void launch_q4k_q8_t4_1row_dual_xs(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                                   const int32_t* xsum, float* Y1, float* Y2, int m, int n);

} // namespace rapidllm::cuda_gemv
