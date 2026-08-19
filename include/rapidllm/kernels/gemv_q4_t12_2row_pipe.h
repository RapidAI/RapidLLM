#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace rapidllm::cuda_gemv {

// Isolated T=12 Q4 2-row + dual with cp.async W. Same dequant as T=4 pipe
// acc_q4k_smem_4x (three T=4 groups per row). Do not add to gemv_q6_t4.h.
void launch_q4k_q8_t12_2row_pipe(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n,
                                 int add);
void launch_q4k_q8_t12_2row_dual_pipe(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                                      float* Y1, float* Y2, int m, int n);

} // namespace rapidllm::cuda_gemv
