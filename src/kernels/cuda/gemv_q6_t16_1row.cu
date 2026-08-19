// Isolated T=16 Q6 1-row GEMV. Same acc_q6_ild_from_blk as T=12 1-row (four T=4
// groups). Do not edit gemv_q6_t12_1row.cu.
#include "rapidllm/kernels/gemv_q6_t16_1row.h"

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

__device__ __forceinline__ void acc_q6_ild_from_blk(const uint8_t* blk, int lane, int is, const float* p0,
                                                    const float* p1, const float* p2, const float* p3, int T,
                                                    float x0a, float x0b, float x0c, float x0d, float x1a, float x1b,
                                                    float x1c, float x1d, float x2a, float x2b, float x2c, float x2d,
                                                    float x3a, float x3b, float x3c, float x3d, float& a0, float& a1,
                                                    float& a2, float& a3) {
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
        if (n128 == 0) {
            a0 = fmaf(s1, x0a, a0);
            a0 = fmaf(s2, x0b, a0);
            a0 = fmaf(s3, x0c, a0);
            a0 = fmaf(s4, x0d, a0);
            a1 = fmaf(s1, x1a, a1);
            a1 = fmaf(s2, x1b, a1);
            a1 = fmaf(s3, x1c, a1);
            a1 = fmaf(s4, x1d, a1);
            a2 = fmaf(s1, x2a, a2);
            a2 = fmaf(s2, x2b, a2);
            a2 = fmaf(s3, x2c, a2);
            a2 = fmaf(s4, x2d, a2);
            if (T >= 4) {
                a3 = fmaf(s1, x3a, a3);
                a3 = fmaf(s2, x3b, a3);
                a3 = fmaf(s3, x3c, a3);
                a3 = fmaf(s4, x3d, a3);
            }
        } else {
            a0 = fmaf(s1, __ldg(p0 + 128 + lane), a0);
            a0 = fmaf(s2, __ldg(p0 + 160 + lane), a0);
            a0 = fmaf(s3, __ldg(p0 + 192 + lane), a0);
            a0 = fmaf(s4, __ldg(p0 + 224 + lane), a0);
            a1 = fmaf(s1, __ldg(p1 + 128 + lane), a1);
            a1 = fmaf(s2, __ldg(p1 + 160 + lane), a1);
            a1 = fmaf(s3, __ldg(p1 + 192 + lane), a1);
            a1 = fmaf(s4, __ldg(p1 + 224 + lane), a1);
            a2 = fmaf(s1, __ldg(p2 + 128 + lane), a2);
            a2 = fmaf(s2, __ldg(p2 + 160 + lane), a2);
            a2 = fmaf(s3, __ldg(p2 + 192 + lane), a2);
            a2 = fmaf(s4, __ldg(p2 + 224 + lane), a2);
            if (T >= 4) {
                a3 = fmaf(s1, __ldg(p3 + 128 + lane), a3);
                a3 = fmaf(s2, __ldg(p3 + 160 + lane), a3);
                a3 = fmaf(s3, __ldg(p3 + 192 + lane), a3);
                a3 = fmaf(s4, __ldg(p3 + 224 + lane), a3);
            }
        }
        ql += 64;
        qh += 32;
        ds += 8;
    }
}

__global__ void __launch_bounds__(256, 2) gemv_q6k_soa_f32_t16_1row_k(const uint8_t* W, const float* X, float* Y,
                                                                      int m, int n, int add) {
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int is = lane / 16;
    const int warp = threadIdx.x / 32;
    extern __shared__ uint8_t smem_w[];
    uint8_t* wsm = smem_w + warp * (2 * kQ6KSoaBsz);
    float a[16] = {};
    const uint8_t* r0 =
        (row < m && nb > 0) ? (W + static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ6KSoaBsz) : nullptr;
    const float* xs[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) xs[t] = X + t * n;
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
        const uint8_t* blk0 = wsm + stage * kQ6KSoaBsz;
        auto acc4 = [&](int t0, float& b0, float& b1, float& b2, float& b3) {
            const float* p0 = xs[t0] + b * 256;
            const float* p1 = xs[t0 + 1] + b * 256;
            const float* p2 = xs[t0 + 2] + b * 256;
            const float* p3 = xs[t0 + 3] + b * 256;
            const float x0a = __ldg(p0 + lane), x0b = __ldg(p0 + 32 + lane);
            const float x0c = __ldg(p0 + 64 + lane), x0d = __ldg(p0 + 96 + lane);
            const float x1a = __ldg(p1 + lane), x1b = __ldg(p1 + 32 + lane);
            const float x1c = __ldg(p1 + 64 + lane), x1d = __ldg(p1 + 96 + lane);
            const float x2a = __ldg(p2 + lane), x2b = __ldg(p2 + 32 + lane);
            const float x2c = __ldg(p2 + 64 + lane), x2d = __ldg(p2 + 96 + lane);
            const float x3a = __ldg(p3 + lane), x3b = __ldg(p3 + 32 + lane);
            const float x3c = __ldg(p3 + 64 + lane), x3d = __ldg(p3 + 96 + lane);
            acc_q6_ild_from_blk(blk0, lane, is, p0, p1, p2, p3, 4, x0a, x0b, x0c, x0d, x1a, x1b, x1c, x1d, x2a,
                                x2b, x2c, x2d, x3a, x3b, x3c, x3d, b0, b1, b2, b3);
        };
        if (r0) {
            acc4(0, a[0], a[1], a[2], a[3]);
            acc4(4, a[4], a[5], a[6], a[7]);
            acc4(8, a[8], a[9], a[10], a[11]);
            acc4(12, a[12], a[13], a[14], a[15]);
        }
        stage ^= 1;
    }
    if (row >= m) return;
#pragma unroll
    for (int t = 0; t < 16; ++t) a[t] = warp_sum(a[t]);
    if (lane != 0) return;
#pragma unroll
    for (int t = 0; t < 16; ++t) write_y(Y + t * m, row, a[t], add);
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q6k_f32_t16_1row(const uint8_t* W, const float* X, float* Y, int m, int n, int add) {
    if (!W || !X || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * kQ6KSoaBsz;
    gemv_q6k_soa_f32_t16_1row_k<<<pb, th, smem>>>(W, X, Y, m, n, add);
}

} // namespace rapidllm::cuda_gemv
