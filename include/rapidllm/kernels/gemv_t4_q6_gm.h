#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace rapidllm::cuda_gemv {

// Isolated T=4 Q6 group-major: pre-unpacked int8 (q-32) + 16 fp16 scales per
// 256-wide superblock (288 B). 2 lanes/group, __dp4a × Q8 X. Do not add to
// gemv_q6_t4.h.
void launch_q6k_gm_q8_t4(const int8_t* Wgm, const int8_t* xq, const __half* xsc, float* Y, int m, int n,
                         int add);

} // namespace rapidllm::cuda_gemv
