// Isolated T=4 Q8 1-row. Same dequant as cuda_engine.cu gemm_q8_soa_k.
#include "rapidllm/kernels/gemv_t4_q8.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ void write_y(float* y, int row, float acc, int add) {
    if (add) y[row] += acc;
    else y[row] = acc;
}

// Same math as q8_dot32 in cuda_engine.cu: 32 int8 · 32 float.
__device__ __forceinline__ float q8_dot32_ldg(const int8_t* q, const float* x) {
    float s = 0.f;
    const int4 a = __ldcs(reinterpret_cast<const int4*>(q));
    const int4 b = __ldcs(reinterpret_cast<const int4*>(q + 16));
    const signed char* qa = reinterpret_cast<const signed char*>(&a);
    const signed char* qb = reinterpret_cast<const signed char*>(&b);
#pragma unroll
    for (int k = 0; k < 16; k += 4) {
        const float4 xv = __ldg(reinterpret_cast<const float4*>(x + k));
        s += static_cast<float>(qa[k]) * xv.x;
        s += static_cast<float>(qa[k + 1]) * xv.y;
        s += static_cast<float>(qa[k + 2]) * xv.z;
        s += static_cast<float>(qa[k + 3]) * xv.w;
    }
#pragma unroll
    for (int k = 0; k < 16; k += 4) {
        const float4 xv = __ldg(reinterpret_cast<const float4*>(x + 16 + k));
        s += static_cast<float>(qb[k]) * xv.x;
        s += static_cast<float>(qb[k + 1]) * xv.y;
        s += static_cast<float>(qb[k + 2]) * xv.z;
        s += static_cast<float>(qb[k + 3]) * xv.w;
    }
    return s;
}

__global__ void __launch_bounds__(256, 4) gemv_q8_f32_t4_1row_k(const int8_t* Q, const __half* scales,
                                                                const float* X, float* Y, int m, int n,
                                                                int add) {
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb = n / 32;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    if (row < m && nb > 0) {
        const int8_t* rowq = Q + static_cast<size_t>(row) * n;
        const __half* rows = scales + static_cast<size_t>(row) * nb;
        const float* x0 = X;
        const float* x1 = X + n;
        const float* x2 = X + 2 * n;
        const float* x3 = X + 3 * n;
        for (int b = lane; b < nb; b += 32) {
            const int off = b * 32;
            const float d = __half2float(__ldg(rows + b));
            a0 = fmaf(d, q8_dot32_ldg(rowq + off, x0 + off), a0);
            a1 = fmaf(d, q8_dot32_ldg(rowq + off, x1 + off), a1);
            a2 = fmaf(d, q8_dot32_ldg(rowq + off, x2 + off), a2);
            a3 = fmaf(d, q8_dot32_ldg(rowq + off, x3 + off), a3);
        }
    }
    if (row >= m) return;
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    a2 = warp_sum(a2);
    a3 = warp_sum(a3);
    if (lane != 0) return;
    write_y(Y, row, a0, add);
    write_y(Y + m, row, a1, add);
    write_y(Y + 2 * m, row, a2, add);
    write_y(Y + 3 * m, row, a3, add);
}

__global__ void __launch_bounds__(256, 2) gemv_q8_f32_t4_1row_dual_k(const int8_t* Q1, const __half* S1,
                                                                     const int8_t* Q2, const __half* S2,
                                                                     const float* X, float* Y1, float* Y2, int m,
                                                                     int n) {
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb = n / 32;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
    if (row < m && nb > 0) {
        const int8_t* r1 = Q1 + static_cast<size_t>(row) * n;
        const int8_t* r2 = Q2 + static_cast<size_t>(row) * n;
        const __half* s1 = S1 + static_cast<size_t>(row) * nb;
        const __half* s2 = S2 + static_cast<size_t>(row) * nb;
        const float* x0 = X;
        const float* x1 = X + n;
        const float* x2 = X + 2 * n;
        const float* x3 = X + 3 * n;
        for (int b = lane; b < nb; b += 32) {
            const int off = b * 32;
            const float d1 = __half2float(__ldg(s1 + b));
            const float d2 = __half2float(__ldg(s2 + b));
            a0 = fmaf(d1, q8_dot32_ldg(r1 + off, x0 + off), a0);
            a1 = fmaf(d1, q8_dot32_ldg(r1 + off, x1 + off), a1);
            a2 = fmaf(d1, q8_dot32_ldg(r1 + off, x2 + off), a2);
            a3 = fmaf(d1, q8_dot32_ldg(r1 + off, x3 + off), a3);
            b0 = fmaf(d2, q8_dot32_ldg(r2 + off, x0 + off), b0);
            b1 = fmaf(d2, q8_dot32_ldg(r2 + off, x1 + off), b1);
            b2 = fmaf(d2, q8_dot32_ldg(r2 + off, x2 + off), b2);
            b3 = fmaf(d2, q8_dot32_ldg(r2 + off, x3 + off), b3);
        }
    }
    if (row >= m) return;
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    a2 = warp_sum(a2);
    a3 = warp_sum(a3);
    b0 = warp_sum(b0);
    b1 = warp_sum(b1);
    b2 = warp_sum(b2);
    b3 = warp_sum(b3);
    if (lane != 0) return;
    write_y(Y1, row, a0, 0);
    write_y(Y1 + m, row, a1, 0);
    write_y(Y1 + 2 * m, row, a2, 0);
    write_y(Y1 + 3 * m, row, a3, 0);
    if (Y2) {
        write_y(Y2, row, b0, 0);
        write_y(Y2 + m, row, b1, 0);
        write_y(Y2 + 2 * m, row, b2, 0);
        write_y(Y2 + 3 * m, row, b3, 0);
    }
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q8_f32_t4_1row(const int8_t* Q, const __half* scales, const float* X, float* Y, int m, int n,
                           int add) {
    if (!Q || !scales || !X || !Y || m <= 0 || n <= 0 || (n % 32) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw - 1) / tw;
    gemv_q8_f32_t4_1row_k<<<pb, th>>>(Q, scales, X, Y, m, n, add);
}

void launch_q8_f32_t4_1row_dual(const int8_t* Q1, const __half* S1, const int8_t* Q2, const __half* S2,
                                const float* X, float* Y1, float* Y2, int m, int n) {
    if (!Q1 || !S1 || !Q2 || !S2 || !X || !Y1 || m <= 0 || n <= 0 || (n % 32) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw - 1) / tw;
    gemv_q8_f32_t4_1row_dual_k<<<pb, th>>>(Q1, S1, Q2, S2, X, Y1, Y2, m, n);
}

} // namespace rapidllm::cuda_gemv
