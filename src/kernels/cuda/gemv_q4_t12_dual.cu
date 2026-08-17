// T=12 Q4 dual (wg+wu). Isolated TU — do not edit gemv_q6_t12.cu.
// Same acc_q4k_soa_q8_12x as the token-correct 2-row T=12 single; W1 then W2
// reuses the 24 acc registers so occupancy matches the single.
#include "rapidllm/kernels/gemv_q6_t4.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kQ4KSoaBsz = 160;

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ void write_y(float* y, int row, float acc, int add) {
    if (add) y[row] += acc;
    else y[row] = acc;
}

__device__ __forceinline__ void acc_q4k_soa_q8_12x(const uint8_t* row, const int8_t* xq, const __half* xsc, int n,
                                                   int nsc, int nb, int lane, float& a0, float& a1, float& a2,
                                                   float& a3, float& a4, float& a5, float& a6, float& a7, float& a8,
                                                   float& a9, float& aa, float& ab) {
    const int group = lane >> 2;
    const int sub = lane & 3;
    const int gpair = group >> 1;
    const int hi = group & 1;
    const int8_t* x0 = xq;
    const int8_t* x1 = xq + n;
    const int8_t* x2 = xq + 2 * n;
    const int8_t* x3 = xq + 3 * n;
    const int8_t* x4 = xq + 4 * n;
    const int8_t* x5 = xq + 5 * n;
    const int8_t* x6 = xq + 6 * n;
    const int8_t* x7 = xq + 7 * n;
    const int8_t* x8 = xq + 8 * n;
    const int8_t* x9 = xq + 9 * n;
    const int8_t* xa = xq + 10 * n;
    const int8_t* xb = xq + 11 * n;
    const __half* s0 = xsc;
    const __half* s1 = xsc + nsc;
    const __half* s2 = xsc + 2 * nsc;
    const __half* s3 = xsc + 3 * nsc;
    const __half* s4 = xsc + 4 * nsc;
    const __half* s5 = xsc + 5 * nsc;
    const __half* s6 = xsc + 6 * nsc;
    const __half* s7 = xsc + 7 * nsc;
    const __half* s8 = xsc + 8 * nsc;
    const __half* s9 = xsc + 9 * nsc;
    const __half* sa = xsc + 10 * nsc;
    const __half* sb = xsc + 11 * nsc;
#pragma unroll 2
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ4KSoaBsz;
        const __half* ds = reinterpret_cast<const __half*>(blk);
        const __half* dm = reinterpret_cast<const __half*>(blk + 16);
        const uint8_t* qs = blk + 32 + gpair * 32 + sub * 8;
        const uint2 q8 = __ldcs(reinterpret_cast<const uint2*>(qs));
        const int q0 = static_cast<int>(hi ? ((q8.x >> 4) & 0x0f0f0f0f) : (q8.x & 0x0f0f0f0f));
        const int q1 = static_cast<int>(hi ? ((q8.y >> 4) & 0x0f0f0f0f) : (q8.y & 0x0f0f0f0f));
        const float dsg = __half2float(__ldg(ds + group));
        const float dmg = __half2float(__ldg(dm + group));
        const int off = b * 256 + group * 32 + sub * 8;
        auto dp = [&](const int8_t* xq_t, const __half* sc, float& acc) {
            const int u0 = *reinterpret_cast<const int*>(xq_t + off);
            const int u1 = *reinterpret_cast<const int*>(xq_t + off + 4);
            int s = __dp4a(q0, u0, 0);
            s = __dp4a(q1, u1, s);
            int sx = __dp4a(0x01010101, u0, 0);
            sx = __dp4a(0x01010101, u1, sx);
            const float xs = __half2float(sc[b * 8 + group]);
            acc = fmaf(dsg * xs, static_cast<float>(s), acc);
            acc = fmaf(-dmg * xs, static_cast<float>(sx), acc);
        };
        dp(x0, s0, a0);
        dp(x1, s1, a1);
        dp(x2, s2, a2);
        dp(x3, s3, a3);
        dp(x4, s4, a4);
        dp(x5, s5, a5);
        dp(x6, s6, a6);
        dp(x7, s7, a7);
        dp(x8, s8, a8);
        dp(x9, s9, a9);
        dp(xa, sa, aa);
        dp(xb, sb, ab);
    }
}

__global__ void __launch_bounds__(256, 2) gemv_q4k_soa_q8_t12_dual_k(const uint8_t* W1, const uint8_t* W2,
                                                                     const int8_t* xq, const __half* xsc, float* Y1,
                                                                     float* Y2, int m, int n) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    auto emit12 = [&](float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8,
                      float v9, float va, float vb, float* Y, int row) {
        v0 = warp_sum(v0);
        v1 = warp_sum(v1);
        v2 = warp_sum(v2);
        v3 = warp_sum(v3);
        v4 = warp_sum(v4);
        v5 = warp_sum(v5);
        v6 = warp_sum(v6);
        v7 = warp_sum(v7);
        v8 = warp_sum(v8);
        v9 = warp_sum(v9);
        va = warp_sum(va);
        vb = warp_sum(vb);
        if (lane != 0) return;
        write_y(Y, row, v0, 0);
        write_y(Y + m, row, v1, 0);
        write_y(Y + 2 * m, row, v2, 0);
        write_y(Y + 3 * m, row, v3, 0);
        write_y(Y + 4 * m, row, v4, 0);
        write_y(Y + 5 * m, row, v5, 0);
        write_y(Y + 6 * m, row, v6, 0);
        write_y(Y + 7 * m, row, v7, 0);
        write_y(Y + 8 * m, row, v8, 0);
        write_y(Y + 9 * m, row, v9, 0);
        write_y(Y + 10 * m, row, va, 0);
        write_y(Y + 11 * m, row, vb, 0);
    };
    if (row0 < m && nb > 0) {
        float a00 = 0.f, a01 = 0.f, a02 = 0.f, a03 = 0.f, a04 = 0.f, a05 = 0.f;
        float a06 = 0.f, a07 = 0.f, a08 = 0.f, a09 = 0.f, a0a = 0.f, a0b = 0.f;
        float a10 = 0.f, a11 = 0.f, a12 = 0.f, a13 = 0.f, a14 = 0.f, a15 = 0.f;
        float a16 = 0.f, a17 = 0.f, a18 = 0.f, a19 = 0.f, a1a = 0.f, a1b = 0.f;
        const uint8_t* r0 = W1 + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        acc_q4k_soa_q8_12x(r0, xq, xsc, n, nsc, nb, lane, a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a0a,
                           a0b);
        if (row1 < m) {
            const uint8_t* r1 = W1 + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz;
            acc_q4k_soa_q8_12x(r1, xq, xsc, n, nsc, nb, lane, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                               a1a, a1b);
        }
        emit12(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a0a, a0b, Y1, row0);
        if (row1 < m) emit12(a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a1a, a1b, Y1, row1);
        a00 = a01 = a02 = a03 = a04 = a05 = a06 = a07 = a08 = a09 = a0a = a0b = 0.f;
        a10 = a11 = a12 = a13 = a14 = a15 = a16 = a17 = a18 = a19 = a1a = a1b = 0.f;
        const uint8_t* s0 = W2 + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        acc_q4k_soa_q8_12x(s0, xq, xsc, n, nsc, nb, lane, a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a0a,
                           a0b);
        if (row1 < m) {
            const uint8_t* s1 = W2 + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz;
            acc_q4k_soa_q8_12x(s1, xq, xsc, n, nsc, nb, lane, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                               a1a, a1b);
        }
        if (Y2) {
            emit12(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a0a, a0b, Y2, row0);
            if (row1 < m) emit12(a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a1a, a1b, Y2, row1);
        }
    }
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q4k_q8_t12_dual(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc, float* Y1,
                            float* Y2, int m, int n) {
    if (!W1 || !W2 || !xq || !xsc || !Y1 || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int pairs = (m + 1) / 2;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (pairs + tw - 1) / tw;
    gemv_q4k_soa_q8_t12_dual_k<<<pb, th>>>(W1, W2, xq, xsc, Y1, Y2, m, n);
}

} // namespace rapidllm::cuda_gemv
