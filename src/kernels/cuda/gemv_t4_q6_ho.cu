// Isolated T=4 Q6 1-row. Same dequant / fma order as gemv_t4_1row.cu
// acc_q6_ild_t4. X is __ldg at use (T=1 2-row style) so launch_bounds
// can target 4 blocks/SM instead of 2.
#include "rapidllm/kernels/gemv_t4_q6_ho.h"

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

__device__ __forceinline__ void cp_async_wait(int n) {
#if __CUDA_ARCH__ >= 800
    if (n <= 0) asm volatile("cp.async.wait_group 0;\n");
    else asm volatile("cp.async.wait_group 1;\n");
#endif
}

__device__ __forceinline__ void cp_async_q6_blk(uint8_t* dst, const uint8_t* src, int lane) {
    if (src && lane < 14) cp_async16(dst + lane * 16, src + lane * 16);
}

__device__ __forceinline__ void acc_q6_ild_t4_ho(const uint8_t* blk, int lane, int is, const float* p0,
                                                 const float* p1, const float* p2, const float* p3, float& a0,
                                                 float& a1, float& a2, float& a3) {
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
        a0 = fmaf(s1, __ldg(p0 + base + lane), a0);
        a0 = fmaf(s2, __ldg(p0 + base + 32 + lane), a0);
        a0 = fmaf(s3, __ldg(p0 + base + 64 + lane), a0);
        a0 = fmaf(s4, __ldg(p0 + base + 96 + lane), a0);
        a1 = fmaf(s1, __ldg(p1 + base + lane), a1);
        a1 = fmaf(s2, __ldg(p1 + base + 32 + lane), a1);
        a1 = fmaf(s3, __ldg(p1 + base + 64 + lane), a1);
        a1 = fmaf(s4, __ldg(p1 + base + 96 + lane), a1);
        a2 = fmaf(s1, __ldg(p2 + base + lane), a2);
        a2 = fmaf(s2, __ldg(p2 + base + 32 + lane), a2);
        a2 = fmaf(s3, __ldg(p2 + base + 64 + lane), a2);
        a2 = fmaf(s4, __ldg(p2 + base + 96 + lane), a2);
        a3 = fmaf(s1, __ldg(p3 + base + lane), a3);
        a3 = fmaf(s2, __ldg(p3 + base + 32 + lane), a3);
        a3 = fmaf(s3, __ldg(p3 + base + 64 + lane), a3);
        a3 = fmaf(s4, __ldg(p3 + base + 96 + lane), a3);
        ql += 64;
        qh += 32;
        ds += 8;
    }
}

__global__ void __launch_bounds__(256, 4) gemv_q6k_soa_f32_t4_1row_ho_k(const uint8_t* W, const float* X,
                                                                        float* Y, int m, int n, int add) {
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int is = lane / 16;
    const int warp = threadIdx.x / 32;
    extern __shared__ uint8_t smem_w[];
    uint8_t* wsm = smem_w + warp * (2 * kQ6KSoaBsz);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    const uint8_t* r0 =
        (row < m && nb > 0) ? (W + static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ6KSoaBsz) : nullptr;
    const float* x0 = X;
    const float* x1 = X + n;
    const float* x2 = X + 2 * n;
    const float* x3 = X + 3 * n;
    auto load_stage = [&](int st, int b) {
        cp_async_q6_blk(wsm + st * kQ6KSoaBsz, r0 ? r0 + static_cast<size_t>(b) * kQ6KSoaBsz : nullptr, lane);
        cp_async_commit();
    };
    if (nb > 0) load_stage(0, 0);
    int stage = 0;
    for (int b = 0; b < nb; ++b) {
        if (b + 1 < nb) {
            load_stage(1 - stage, b + 1);
            cp_async_wait(1);
        } else {
            cp_async_wait(0);
        }
        __syncwarp();
        if (r0)
            acc_q6_ild_t4_ho(wsm + stage * kQ6KSoaBsz, lane, is, x0 + b * 256, x1 + b * 256, x2 + b * 256,
                             x3 + b * 256, a0, a1, a2, a3);
        stage ^= 1;
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

void launch_q6k_f32_t4_1row_ho(const uint8_t* W, const float* X, float* Y, int m, int n, int add) {
    if (!W || !X || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * kQ6KSoaBsz;
    gemv_q6k_soa_f32_t4_1row_ho_k<<<pb, th, smem>>>(W, X, Y, m, n, add);
}

} // namespace rapidllm::cuda_gemv
