// Isolated T=12 Q6 GEMV. Block-shared X tile so 8 warps do not each
// reload 12x256 X from HBM. Same Q6 dequant as gemv_q6_t12_1row.
#include "rapidllm/kernels/gemv_q6_t12_smemx.h"

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

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
#if __CUDA_ARCH__ >= 800
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(addr), "l"(gmem));
#else
    *reinterpret_cast<uint4*>(smem) = *reinterpret_cast<const uint4*>(gmem);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
#endif
}

__device__ __forceinline__ void cp_async_wait0() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n");
#endif
}

__device__ __forceinline__ void cp_async_q6_blk(uint8_t* dst, const uint8_t* src, int lane) {
    if (src && lane < 14) cp_async16(dst + lane * 16, src + lane * 16);
}

// Same dequant + fmaf order as acc_q6_ild_from_blk in t12_1row.
__device__ __forceinline__ void acc_q6_from_xsm(const uint8_t* blk, int lane, int is, const float* x0,
                                                const float* x1, const float* x2, const float* x3, int T,
                                                float& a0, float& a1, float& a2, float& a3) {
    const __half* ds = reinterpret_cast<const __half*>(blk);
    const uint8_t* ql = blk + 32;
    const uint8_t* qh = blk + 160;
#pragma unroll
    for (int n128 = 0; n128 < 2; ++n128) {
        const uint8_t qlo = ql[lane];
        const uint8_t qhi = qh[lane];
        const uint8_t qlo2 = ql[32 + lane];
        const int q1 = static_cast<int>((qlo & 0xF) | (((qhi >> 0) & 3) << 4)) - 32;
        const int q2 = static_cast<int>((qlo2 & 0xF) | (((qhi >> 2) & 3) << 4)) - 32;
        const int q3 = static_cast<int>((qlo >> 4) | (((qhi >> 4) & 3) << 4)) - 32;
        const int q4 = static_cast<int>((qlo2 >> 4) | (((qhi >> 6) & 3) << 4)) - 32;
        const float s1 = __half2float(ds[is]) * static_cast<float>(q1);
        const float s2 = __half2float(ds[is + 2]) * static_cast<float>(q2);
        const float s3 = __half2float(ds[is + 4]) * static_cast<float>(q3);
        const float s4 = __half2float(ds[is + 6]) * static_cast<float>(q4);
        const int base = n128 * 128;
        a0 = fmaf(s1, x0[base + lane], a0);
        a0 = fmaf(s2, x0[base + 32 + lane], a0);
        a0 = fmaf(s3, x0[base + 64 + lane], a0);
        a0 = fmaf(s4, x0[base + 96 + lane], a0);
        a1 = fmaf(s1, x1[base + lane], a1);
        a1 = fmaf(s2, x1[base + 32 + lane], a1);
        a1 = fmaf(s3, x1[base + 64 + lane], a1);
        a1 = fmaf(s4, x1[base + 96 + lane], a1);
        a2 = fmaf(s1, x2[base + lane], a2);
        a2 = fmaf(s2, x2[base + 32 + lane], a2);
        a2 = fmaf(s3, x2[base + 64 + lane], a2);
        a2 = fmaf(s4, x2[base + 96 + lane], a2);
        if (T >= 4) {
            a3 = fmaf(s1, x3[base + lane], a3);
            a3 = fmaf(s2, x3[base + 32 + lane], a3);
            a3 = fmaf(s3, x3[base + 64 + lane], a3);
            a3 = fmaf(s4, x3[base + 96 + lane], a3);
        }
        ql += 64;
        qh += 32;
        ds += 8;
    }
}

__global__ void __launch_bounds__(256, 3) gemv_q6k_soa_f32_t12_smemx_k(const uint8_t* W, const float* X, float* Y,
                                                                        int m, int n, int add) {
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int ln = threadIdx.x & 31;
    const int nb = n / 256;
    const int is = ln / 16;
    const int warp = threadIdx.x / 32;
    extern __shared__ uint8_t sm[];
    float* xsm = reinterpret_cast<float*>(sm);
    uint8_t* wbase = sm + sizeof(float) * 12 * 256;
    uint8_t* wsm = wbase + warp * (2 * kQ6KSoaBsz);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f;
    float a6 = 0.f, a7 = 0.f, a8 = 0.f, a9 = 0.f, aa = 0.f, ab = 0.f, ad = 0.f;
    const uint8_t* r0 =
        (row < m && nb > 0) ? (W + static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ6KSoaBsz) : nullptr;
    auto load_w = [&](int st, int b) {
        uint8_t* s0 = wsm + st * kQ6KSoaBsz;
        cp_async_q6_blk(s0, r0 ? r0 + static_cast<size_t>(b) * kQ6KSoaBsz : nullptr, ln);
        cp_async_commit();
    };
    if (nb > 0) load_w(0, 0);
    int stage = 0;
    for (int b = 0; b < nb; ++b) {
#pragma unroll
        for (int t = 0; t < 12; ++t)
            xsm[t * 256 + threadIdx.x] = __ldg(X + static_cast<size_t>(t) * n + b * 256 + threadIdx.x);
        if (b + 1 < nb) load_w(1 - stage, b + 1);
        cp_async_wait0();
        __syncthreads();
        const uint8_t* blk0 = wsm + stage * kQ6KSoaBsz;
        if (r0) {
            acc_q6_from_xsm(blk0, ln, is, xsm + 0 * 256, xsm + 1 * 256, xsm + 2 * 256, xsm + 3 * 256, 4, a0, a1,
                            a2, a3);
            acc_q6_from_xsm(blk0, ln, is, xsm + 4 * 256, xsm + 5 * 256, xsm + 5 * 256, xsm + 5 * 256, 3, a4, a5,
                            ad, ad);
            acc_q6_from_xsm(blk0, ln, is, xsm + 6 * 256, xsm + 7 * 256, xsm + 8 * 256, xsm + 9 * 256, 4, a6, a7,
                            a8, a9);
            acc_q6_from_xsm(blk0, ln, is, xsm + 10 * 256, xsm + 11 * 256, xsm + 11 * 256, xsm + 11 * 256, 3, aa, ab,
                            ad, ad);
        }
        stage ^= 1;
        __syncthreads();
    }
    if (row >= m) return;
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    a2 = warp_sum(a2);
    a3 = warp_sum(a3);
    a4 = warp_sum(a4);
    a5 = warp_sum(a5);
    a6 = warp_sum(a6);
    a7 = warp_sum(a7);
    a8 = warp_sum(a8);
    a9 = warp_sum(a9);
    aa = warp_sum(aa);
    ab = warp_sum(ab);
    if (ln != 0) return;
    write_y(Y, row, a0, add);
    write_y(Y + m, row, a1, add);
    write_y(Y + 2 * m, row, a2, add);
    write_y(Y + 3 * m, row, a3, add);
    write_y(Y + 4 * m, row, a4, add);
    write_y(Y + 5 * m, row, a5, add);
    write_y(Y + 6 * m, row, a6, add);
    write_y(Y + 7 * m, row, a7, add);
    write_y(Y + 8 * m, row, a8, add);
    write_y(Y + 9 * m, row, a9, add);
    write_y(Y + 10 * m, row, aa, add);
    write_y(Y + 11 * m, row, ab, add);
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q6k_f32_t12_smemx(const uint8_t* W, const float* X, float* Y, int m, int n, int add) {
    if (!W || !X || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw - 1) / tw;
    const size_t smem = sizeof(float) * 12 * 256 + static_cast<size_t>(tw) * 2 * kQ6KSoaBsz;
    gemv_q6k_soa_f32_t12_smemx_k<<<pb, th, smem>>>(W, X, Y, m, n, add);
}

} // namespace rapidllm::cuda_gemv
