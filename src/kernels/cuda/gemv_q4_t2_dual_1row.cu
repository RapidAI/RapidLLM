// Isolated T=2 Q4 dual 1-row. Same dequant as gemv_q6_t2_1row acc_q4k_soa_q8_2x.
// Higher occupancy than the 2-row dual (8 accs, bounds 256,2).
#include "rapidllm/kernels/gemv_q4_t2_dual_1row.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kQ4KSoaBsz = 160;

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ void write_y(float* y, int row, float acc) {
    y[row] = acc;
}

__device__ __forceinline__ void acc_q4k_soa_q8_2x(const uint8_t* row, const int8_t* xq0, const __half* sc0,
                                                  const int8_t* xq1, const __half* sc1, int nb, int lane,
                                                  float& a0, float& a1) {
    const int group = lane >> 2;
    const int sub = lane & 3;
    const int gpair = group >> 1;
    const int hi = group & 1;
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
        const int u0 = *reinterpret_cast<const int*>(xq0 + off);
        const int u1 = *reinterpret_cast<const int*>(xq0 + off + 4);
        const int v0 = *reinterpret_cast<const int*>(xq1 + off);
        const int v1 = *reinterpret_cast<const int*>(xq1 + off + 4);
        int s0 = __dp4a(q0, u0, 0);
        s0 = __dp4a(q1, u1, s0);
        int sx0 = __dp4a(0x01010101, u0, 0);
        sx0 = __dp4a(0x01010101, u1, sx0);
        int s1 = __dp4a(q0, v0, 0);
        s1 = __dp4a(q1, v1, s1);
        int sx1 = __dp4a(0x01010101, v0, 0);
        sx1 = __dp4a(0x01010101, v1, sx1);
        const float xs0 = __half2float(sc0[b * 8 + group]);
        const float xs1 = __half2float(sc1[b * 8 + group]);
        a0 = fmaf(dsg * xs0, static_cast<float>(s0), a0);
        a0 = fmaf(-dmg * xs0, static_cast<float>(sx0), a0);
        a1 = fmaf(dsg * xs1, static_cast<float>(s1), a1);
        a1 = fmaf(-dmg * xs1, static_cast<float>(sx1), a1);
    }
}

__global__ void __launch_bounds__(256, 4) gemv_q4k_soa_q8_t2_dual_1row_k(const uint8_t* W1, const uint8_t* W2,
                                                                        const int8_t* xq, const __half* xsc,
                                                                        float* Y1, float* Y2, int m, int n) {
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    float a0 = 0.f, a1 = 0.f, b0 = 0.f, b1 = 0.f;
    if (row < m && nb > 0) {
        const uint8_t* r1 = W1 + static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        const uint8_t* r2 = W2 + static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        // Interleave W1/W2 per K-tile so the same xq tile stays in L1.
        for (int b = 0; b < nb; ++b) {
            acc_q4k_soa_q8_2x(r1 + static_cast<size_t>(b) * kQ4KSoaBsz, xq + b * 256, xsc + b * 8,
                              xq + n + b * 256, xsc + nsc + b * 8, 1, lane, a0, a1);
            acc_q4k_soa_q8_2x(r2 + static_cast<size_t>(b) * kQ4KSoaBsz, xq + b * 256, xsc + b * 8,
                              xq + n + b * 256, xsc + nsc + b * 8, 1, lane, b0, b1);
        }
    }
    if (row >= m) return;
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    b0 = warp_sum(b0);
    b1 = warp_sum(b1);
    if (lane != 0) return;
    write_y(Y1, row, a0);
    write_y(Y1 + m, row, a1);
    write_y(Y2, row, b0);
    write_y(Y2 + m, row, b1);
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q4k_q8_t2_dual_1row(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                                float* Y1, float* Y2, int m, int n) {
    if (!W1 || !W2 || !xq || !xsc || !Y1 || !Y2 || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw - 1) / tw;
    gemv_q4k_soa_q8_t2_dual_1row_k<<<pb, th>>>(W1, W2, xq, xsc, Y1, Y2, m, n);
}

} // namespace rapidllm::cuda_gemv
