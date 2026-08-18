#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace rapidllm::cuda_gemv {

// Isolated T=4 Q4 1-row / dual: 3-stage cp.async + group-sx from ensure_xq.
// Same dequant as gemv_t4_pipe acc_q4k_smem_4x (sx folded to one group sum).
void launch_q4k_q8_t4_1row_p3(const uint8_t* W, const int8_t* xq, const __half* xsc, const int32_t* xsum,
                              float* Y, int m, int n, int add);
void launch_q4k_q8_t4_1row_dual_p3(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                                   const int32_t* xsum, float* Y1, float* Y2, int m, int n);

} // namespace rapidllm::cuda_gemv
