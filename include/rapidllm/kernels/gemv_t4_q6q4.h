#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace rapidllm::cuda_gemv {

// Isolated T=4 fused Q6+Q4 1-row pair. Same Q6 acc as gemv_t4_1row.cu and
// same Q4 acc as gemv_t4_pipe.cu. Do not add to gemv_q6_t4.h.
void launch_q6q4_t4_1row_pair(const uint8_t* W6, const float* X, float* Y6, int m6, const uint8_t* W4,
                              const int8_t* xq, const __half* xsc, float* Y4, int m4, int n);

} // namespace rapidllm::cuda_gemv
