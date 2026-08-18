// Isolated T=2 Q6 split-K. Same dequant as acc_q6_ild_t2 in gemv_q6_t2_1row.cu.
// K is split across blockIdx.y; each part atomicAdds into Y (wd residual).
#include "rapidllm/kernels/gemv_q6_t2_splitk.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kQ6KSoaBsz = 224;
constexpr int kParts = 4;

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
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

__device__ __forceinline__ void acc_q6_ild_t2(const uint8_t* blk, int lane, int is, const float* p0,
                                              const float* p1, float x0a, float x0b, float x0c, float x0d,
                                              float x1a, float x1b, float x1c, float x1d, float& a0, float& a1) {
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
        } else {
            a0 = fmaf(s1, __ldg(p0 + 128 + lane), a0);
            a0 = fmaf(s2, __ldg(p0 + 160 + lane), a0);
            a0 = fmaf(s3, __ldg(p0 + 192 + lane), a0);
            a0 = fmaf(s4, __ldg(p0 + 224 + lane), a0);
            a1 = fmaf(s1, __ldg(p1 + 128 + lane), a1);
            a1 = fmaf(s2, __ldg(p1 + 160 + lane), a1);
            a1 = fmaf(s3, __ldg(p1 + 192 + lane), a1);
            a1 = fmaf(s4, __ldg(p1 + 224 + lane), a1);
        }
        ql += 64;
        qh += 32;
        ds += 8;
    }
}

__global__ void __launch_bounds__(256, 3) gemv_q6k_soa_f32_t2_splitk_k(const uint8_t* W, const float* X, float* Y,
                                                                      int m, int n) {
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int part = static_cast<int>(blockIdx.y);
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int tiles = (nb + kParts - 1) / kParts;
    const int b0 = part * tiles;
    const int b1 = b0 + tiles < nb ? b0 + tiles : nb;
    if (b0 >= nb || row >= m) return;
    const int is = lane / 16;
    const int warp = threadIdx.x / 32;
    extern __shared__ uint8_t smem_w[];
    uint8_t* wsm = smem_w + warp * (2 * kQ6KSoaBsz);
    float a0 = 0.f, a1 = 0.f;
    const uint8_t* r0 = W + static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ6KSoaBsz;
    const float* x0 = X;
    const float* x1 = X + n;
    auto load_stage = [&](int st, int b) {
        uint8_t* s0 = wsm + st * kQ6KSoaBsz;
        if (lane < 14) cp_async16(s0 + lane * 16, r0 + static_cast<size_t>(b) * kQ6KSoaBsz + lane * 16);
        cp_async_commit();
    };
    load_stage(0, b0);
    int stage = 0;
    for (int b = b0; b < b1; ++b) {
        if (b + 1 < b1) {
            load_stage(1 - stage, b + 1);
            cp_async_wait(1);
        } else {
            cp_async_wait(0);
        }
        __syncwarp();
        const float* p0 = x0 + b * 256;
        const float* p1 = x1 + b * 256;
        const float x0a = __ldg(p0 + lane), x0b = __ldg(p0 + 32 + lane);
        const float x0c = __ldg(p0 + 64 + lane), x0d = __ldg(p0 + 96 + lane);
        const float x1a = __ldg(p1 + lane), x1b = __ldg(p1 + 32 + lane);
        const float x1c = __ldg(p1 + 64 + lane), x1d = __ldg(p1 + 96 + lane);
        acc_q6_ild_t2(wsm + stage * kQ6KSoaBsz, lane, is, p0, p1, x0a, x0b, x0c, x0d, x1a, x1b, x1c, x1d, a0, a1);
        stage ^= 1;
    }
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    if (lane != 0) return;
    atomicAdd(Y + row, a0);
    atomicAdd(Y + m + row, a1);
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q6k_f32_t2_splitk(const uint8_t* W, const float* X, float* Y, int m, int n) {
    if (!W || !X || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * kQ6KSoaBsz;
    dim3 grid(pb, kParts);
    gemv_q6k_soa_f32_t2_splitk_k<<<grid, th, smem>>>(W, X, Y, m, n);
}

} // namespace rapidllm::cuda_gemv
