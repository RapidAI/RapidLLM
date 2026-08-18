#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace rapidllm::cuda_gemv {

// Isolated T=4 Q6: unpack SOA once to int8+(q-32) and per-16 fp16 scales, then
// GEMV with the same lane/FMA order as gemv_t4_1row. Do not add to gemv_q6_t4.h.
void launch_q6k_unpack_i8(const uint8_t* Wsoa, int8_t* Qi, __half* Sc, int m, int n);
void launch_q6k_i8_f32_t4(const int8_t* Qi, const __half* Sc, const float* X, float* Y, int m, int n, int add);

} // namespace rapidllm::cuda_gemv
