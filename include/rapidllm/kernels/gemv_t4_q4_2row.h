#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace rapidllm::cuda_gemv {

// Isolated T=4 Q4 2-row dual. Same dequant as gemv_t4_pipe.cu acc_q4k_smem_4x.
// Do not add to gemv_q6_t4.h.
void launch_q4k_q8_t4_2row_dual(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                                float* Y1, float* Y2, int m, int n);

} // namespace rapidllm::cuda_gemv
