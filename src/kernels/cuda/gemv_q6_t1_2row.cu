// Isolated T=1 Q6 2-row GEMV for large lm_head / MTP. 3-stage cp.async W.
// X via __ldg (same as T=2 1-row) — no block X tile, no __syncthreads.
#include "rapidllm/kernels/gemv_q6_t1_2row.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kQ6KSoaBsz = 224;

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

__device__ __forceinline__ void cp_async_wait0() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n");
#endif
}

__device__ __forceinline__ void cp_async_wait1() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;\n");
#endif
}

__device__ __forceinline__ void cp_async_wait2() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 2;\n");
#endif
}

// Same dequant + fmaf order as the smem-x kernel; x is the K-block base in global.
__device__ __forceinline__ void acc_q6_blk(const uint8_t* blk, int lane, int is, const float* x, float& acc) {
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
        const float* p = x + n128 * 128;
        acc = fmaf(s1, __ldg(p + lane), acc);
        acc = fmaf(s2, __ldg(p + 32 + lane), acc);
        acc = fmaf(s3, __ldg(p + 64 + lane), acc);
        acc = fmaf(s4, __ldg(p + 96 + lane), acc);
        ql += 64;
        qh += 32;
        ds += 8;
    }
}

__global__ void __launch_bounds__(256, 4) gemv_q6k_soa_f32_t1_2row_k(const uint8_t* W, const float* x, float* y,
                                                                    int m, int n, int add) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int is = lane / 16;
    const int warp = threadIdx.x / 32;
    extern __shared__ uint8_t smem_w[];
    uint8_t* wsm = smem_w + static_cast<size_t>(warp) * 6 * kQ6KSoaBsz;
    float a0 = 0.f, a1 = 0.f;
    const uint8_t* r0 =
        (row0 < m && nb > 0) ? (W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ6KSoaBsz) : nullptr;
    const uint8_t* r1 =
        (row1 < m && nb > 0) ? (W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ6KSoaBsz) : nullptr;
    auto load_stage = [&](int st, int b) {
        uint8_t* s0 = wsm + st * 2 * kQ6KSoaBsz;
        if (lane < 14) {
            if (r0) cp_async16(s0 + lane * 16, r0 + static_cast<size_t>(b) * kQ6KSoaBsz + lane * 16);
            if (r1) cp_async16(s0 + kQ6KSoaBsz + lane * 16, r1 + static_cast<size_t>(b) * kQ6KSoaBsz + lane * 16);
        }
        cp_async_commit();
    };
    if (nb > 0) load_stage(0, 0);
    if (nb > 1) load_stage(1, 1);
    int stage = 0;
    for (int b = 0; b < nb; ++b) {
        if (b + 2 < nb) {
            load_stage((stage + 2) % 3, b + 2);
            cp_async_wait2();
        } else if (b + 1 < nb) {
            cp_async_wait1();
        } else {
            cp_async_wait0();
        }
        __syncwarp();
        const float* xb = x + b * 256;
        const uint8_t* s0 = wsm + stage * 2 * kQ6KSoaBsz;
        if (r0) acc_q6_blk(s0, lane, is, xb, a0);
        if (r1) acc_q6_blk(s0 + kQ6KSoaBsz, lane, is, xb, a1);
        stage = (stage + 1) % 3;
    }
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    if (lane != 0) return;
    if (row0 < m) {
        if (add) y[row0] += a0;
        else y[row0] = a0;
    }
    if (row1 < m) {
        if (add) y[row1] += a1;
        else y[row1] = a1;
    }
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q6k_f32_t1_2row(const uint8_t* W, const float* x, float* y, int m, int n, int add) {
    if (!W || !x || !y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pairs = (m + 1) / 2;
    const int pb = (pairs + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 6 * kQ6KSoaBsz;
    gemv_q6k_soa_f32_t1_2row_k<<<pb, th, smem>>>(W, x, y, m, n, add);
}

} // namespace rapidllm::cuda_gemv
