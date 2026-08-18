#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace rapidllm::cuda_gemv {

// Isolated T=4 Q4: one W tile, two T=2 acc passes (fewer live accs / higher
// occ). Same dequant as gemv_t4_pipe acc_q4k_smem_4x. Do not add to gemv_q6_t4.h.
void launch_q4k_q8_t4_1row_p2(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n,
                              int add);
void launch_q4k_q8_t4_1row_dual_p2(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                                   float* Y1, float* Y2, int m, int n);

// Exact pipe 4-token acc, optional stream (for Q6||Q4 overlap). Same math as gemv_t4_pipe.
void launch_q4k_q8_t4_1row_st(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n,
                              int add, void* stream);

} // namespace rapidllm::cuda_gemv
