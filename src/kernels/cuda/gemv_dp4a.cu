// NVIDIA GEMV using sm_61+ __dp4a (4×int8 MAC). Isolated from public headers.
#include "rapidllm/kernels/nv_gemv.h"

#include <cuda_runtime.h>
#include <vector>

namespace {

__global__ void gemv_dp4a_kernel(const signed char* W, const float* row_scale, const int8_t* xq,
                                 float xs, float* y, int m, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const signed char* row = W + static_cast<size_t>(i) * n;
    int acc = 0;
    int j = 0;
    const int n4 = n & ~3;
    for (; j < n4; j += 4) {
        int wa, xa;
        // pack 4 int8 → int (little-endian, matches host-ref)
        wa = (static_cast<unsigned char>(row[j + 0])) |
             (static_cast<unsigned char>(row[j + 1]) << 8) |
             (static_cast<unsigned char>(row[j + 2]) << 16) |
             (static_cast<unsigned char>(row[j + 3]) << 24);
        xa = (static_cast<unsigned char>(xq[j + 0])) |
             (static_cast<unsigned char>(xq[j + 1]) << 8) |
             (static_cast<unsigned char>(xq[j + 2]) << 16) |
             (static_cast<unsigned char>(xq[j + 3]) << 24);
        acc = __dp4a(wa, xa, acc);
    }
    for (; j < n; ++j) acc += static_cast<int>(row[j]) * static_cast<int>(xq[j]);
    y[i] = static_cast<float>(acc) * row_scale[i] * xs;
}

} // namespace

namespace rapidllm::nv {

bool gemv_dp4a_cuda(const signed char* W, const float* row_scale, const float* x, float* y, int m,
                    int n) {
    float amax = 0.f;
    for (int j = 0; j < n; ++j) amax = fmaxf(amax, fabsf(x[j]));
    const float xs = amax > 0.f ? amax / 127.f : 1.f;
    std::vector<int8_t> xq(static_cast<size_t>(n));
    for (int j = 0; j < n; ++j) {
        int q = static_cast<int>(lroundf(x[j] / xs));
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        xq[j] = static_cast<int8_t>(q);
    }
    signed char* dW = nullptr;
    float* dS = nullptr;
    int8_t* dX = nullptr;
    float* dY = nullptr;
    const size_t wbytes = static_cast<size_t>(m) * n;
    if (cudaMalloc(&dW, wbytes) != cudaSuccess) return false;
    if (cudaMalloc(&dS, sizeof(float) * m) != cudaSuccess) {
        cudaFree(dW);
        return false;
    }
    if (cudaMalloc(&dX, n) != cudaSuccess) {
        cudaFree(dW);
        cudaFree(dS);
        return false;
    }
    if (cudaMalloc(&dY, sizeof(float) * m) != cudaSuccess) {
        cudaFree(dW);
        cudaFree(dS);
        cudaFree(dX);
        return false;
    }
    cudaMemcpy(dW, W, wbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dS, row_scale, sizeof(float) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(dX, xq.data(), n, cudaMemcpyHostToDevice);
    const int threads = 128;
    const int blocks = (m + threads - 1) / threads;
    gemv_dp4a_kernel<<<blocks, threads>>>(dW, dS, dX, xs, dY, m, n);
    cudaMemcpy(y, dY, sizeof(float) * m, cudaMemcpyDeviceToHost);
    cudaFree(dW);
    cudaFree(dS);
    cudaFree(dX);
    cudaFree(dY);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace rapidllm::nv
