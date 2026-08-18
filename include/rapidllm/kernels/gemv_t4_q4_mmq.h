#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace rapidllm::cuda_gemv {

// Isolated T=4 Q4 mmq: 16-row MMA per warp, one 256-K unpack, no per-group
// syncthreads. Same Q4 SOA nibble + (ds*q - dm)*xs as the 1-row pipe.
// Do not add to gemv_q6_t4.h.
void launch_q4k_q8_t4_mmq(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n,
                          int add);
void launch_q4k_q8_t4_dual_mmq(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                               float* Y1, float* Y2, int m, int n);

} // namespace rapidllm::cuda_gemv
