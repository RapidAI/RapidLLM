#pragma once

// CUDA-free public header. NVIDIA GEMV / fused GEMV host-ref + optional .cu.

namespace rapidllm::nv {

// INT8 weight + per-row scale, activations quantized to int8 per call (dp4a contract).
void gemv_dp4a_ref(const signed char* W, const float* row_scale, const float* x, float* y, int m,
                   int n);

// Same contract as the CUDA kernel (documented PTX/dp4a). Returns true if a device ran.
bool gemv_dp4a_cuda(const signed char* W, const float* row_scale, const float* x, float* y, int m,
                    int n);

void pack_weight_int8(const float* W, signed char* out, float* row_scale, int m, int n);

const char* nv_kernel_name();

} // namespace rapidllm::nv
