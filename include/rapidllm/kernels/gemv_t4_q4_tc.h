#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace rapidllm::cuda_gemv {

// Isolated T=4 Q4 tensor-core 16-row MMA. Same Q4 SOA dequant as the 1-row
// pipe (ds*q - dm)* (xs*xq); float acc is per-group along K. Do not add to
// gemv_q6_t4.h.
void launch_q4k_q8_t4_1row_tc(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n,
                              int add);
void launch_q4k_q8_t4_1row_dual_tc(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                                   float* Y1, float* Y2, int m, int n);

} // namespace rapidllm::cuda_gemv
