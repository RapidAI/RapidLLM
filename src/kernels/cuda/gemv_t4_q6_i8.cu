// Isolated T=4 Q6 int8 unpack + GEMV. Same dequant/lane/FMA order as
// gemv_t4_1row acc_q6_ild_t4 so tokens stay on the 1-row path.
#include "rapidllm/kernels/gemv_t4_q6_i8.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kQ6KSoaBsz = 224;

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ void write_y(float* y, int row, float acc, int add) {
    if (add) y[row] += acc;
    else y[row] = acc;
}

__global__ void q6k_unpack_i8_k(const uint8_t* W, int8_t* Qi, __half* Sc, int m, int n) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (row >= m) return;
    const int nb = n / 256;
    const int nsc = n >> 4;
    const uint8_t* r0 = W + static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ6KSoaBsz;
    int8_t* qrow = Qi + static_cast<size_t>(row) * static_cast<size_t>(n);
    __half* srow = Sc + static_cast<size_t>(row) * static_cast<size_t>(nsc);
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = r0 + static_cast<size_t>(b) * kQ6KSoaBsz;
        const __half* ds = reinterpret_cast<const __half*>(blk);
        if (lane < 16) srow[b * 16 + lane] = ds[lane];
        const uint8_t* ql = blk + 32;
        const uint8_t* qh = blk + 160;
        const int is = lane / 16;
#pragma unroll
        for (int n128 = 0; n128 < 2; ++n128) {
            const uint8_t qlo = ql[lane];
            const uint8_t qhi = qh[lane];
            const uint8_t qlo2 = ql[32 + lane];
            const int q1 = static_cast<int>((qlo & 0xF) | (((qhi >> 0) & 3) << 4)) - 32;
            const int q2 = static_cast<int>((qlo2 & 0xF) | (((qhi >> 2) & 3) << 4)) - 32;
            const int q3 = static_cast<int>((qlo >> 4) | (((qhi >> 4) & 3) << 4)) - 32;
            const int q4 = static_cast<int>((qlo2 >> 4) | (((qhi >> 6) & 3) << 4)) - 32;
            const int base = b * 256 + n128 * 128;
            qrow[base + lane] = static_cast<int8_t>(q1);
            qrow[base + 32 + lane] = static_cast<int8_t>(q2);
            qrow[base + 64 + lane] = static_cast<int8_t>(q3);
            qrow[base + 96 + lane] = static_cast<int8_t>(q4);
            (void)is;
            ql += 64;
            qh += 32;
        }
    }
}

__global__ void __launch_bounds__(256, 4) q6k_i8_f32_t4_k(const int8_t* Qi, const __half* Sc, const float* X,
                                                          float* Y, int m, int n, int add) {
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 4;
    const int is = lane / 16;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    if (row < m && nb > 0) {
        const int8_t* qrow = Qi + static_cast<size_t>(row) * static_cast<size_t>(n);
        const __half* srow = Sc + static_cast<size_t>(row) * static_cast<size_t>(nsc);
        const float* x0 = X;
        const float* x1 = X + n;
        const float* x2 = X + 2 * n;
        const float* x3 = X + 3 * n;
        for (int b = 0; b < nb; ++b) {
            const int off = b * 256;
            const __half* ds = srow + b * 16;
#pragma unroll
            for (int n128 = 0; n128 < 2; ++n128) {
                const int base = off + n128 * 128;
                const int q1 = static_cast<int>(qrow[base + lane]);
                const int q2 = static_cast<int>(qrow[base + 32 + lane]);
                const int q3 = static_cast<int>(qrow[base + 64 + lane]);
                const int q4 = static_cast<int>(qrow[base + 96 + lane]);
                const float s1 = __half2float(ds[is]) * static_cast<float>(q1);
                const float s2 = __half2float(ds[is + 2]) * static_cast<float>(q2);
                const float s3 = __half2float(ds[is + 4]) * static_cast<float>(q3);
                const float s4 = __half2float(ds[is + 6]) * static_cast<float>(q4);
                const float* p0 = x0 + base;
                const float* p1 = x1 + base;
                const float* p2 = x2 + base;
                const float* p3 = x3 + base;
                a0 = fmaf(s1, p0[lane], a0);
                a0 = fmaf(s2, p0[32 + lane], a0);
                a0 = fmaf(s3, p0[64 + lane], a0);
                a0 = fmaf(s4, p0[96 + lane], a0);
                a1 = fmaf(s1, p1[lane], a1);
                a1 = fmaf(s2, p1[32 + lane], a1);
                a1 = fmaf(s3, p1[64 + lane], a1);
                a1 = fmaf(s4, p1[96 + lane], a1);
                a2 = fmaf(s1, p2[lane], a2);
                a2 = fmaf(s2, p2[32 + lane], a2);
                a2 = fmaf(s3, p2[64 + lane], a2);
                a2 = fmaf(s4, p2[96 + lane], a2);
                a3 = fmaf(s1, p3[lane], a3);
                a3 = fmaf(s2, p3[32 + lane], a3);
                a3 = fmaf(s3, p3[64 + lane], a3);
                a3 = fmaf(s4, p3[96 + lane], a3);
                ds += 8;
            }
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

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q6k_unpack_i8(const uint8_t* Wsoa, int8_t* Qi, __half* Sc, int m, int n) {
    if (!Wsoa || !Qi || !Sc || m <= 0 || n <= 0 || (n % 256) != 0) return;
    q6k_unpack_i8_k<<<m, 32>>>(Wsoa, Qi, Sc, m, n);
}

void launch_q6k_i8_f32_t4(const int8_t* Qi, const __half* Sc, const float* X, float* Y, int m, int n, int add) {
    if (!Qi || !Sc || !X || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw - 1) / tw;
    q6k_i8_f32_t4_k<<<pb, th>>>(Qi, Sc, X, Y, m, n, add);
}

} // namespace rapidllm::cuda_gemv
