#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace rapidllm::cuda_gemv {

// Isolated T=4 Q6 × Q8 1-row + dp4a. Same pack as acc_q6k_soa_q8 in
// cuda_engine.cu. Do not add to gemv_q6_t4.h.
void launch_q6k_q8_t4_1row(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n,
                           int add);

} // namespace rapidllm::cuda_gemv
