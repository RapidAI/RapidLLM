// Isolated T=2 Q6 2-row + cp.async. Same Q6 dequant as acc_q6_ild_t2 in
// gemv_q6_t2_1row.cu. Do not edit locked TUs or gemv_q6_t4.h.
#include "rapidllm/kernels/gemv_q6_t2_2row.h"

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

__device__ __forceinline__ void cp_async_wait1() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;\n");
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

__global__ void __launch_bounds__(256, 3) gemv_q6k_soa_f32_t2_2row_k(const uint8_t* W, const float* X, float* Y,
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
    uint8_t* wsm = smem_w + static_cast<size_t>(warp) * 4 * kQ6KSoaBsz;
    float a00 = 0.f, a01 = 0.f, a10 = 0.f, a11 = 0.f;
    const uint8_t* r0 =
        (row0 < m && nb > 0) ? (W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ6KSoaBsz) : nullptr;
    const uint8_t* r1 =
        (row1 < m && nb > 0) ? (W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ6KSoaBsz) : nullptr;
    const float* x0 = X;
    const float* x1 = X + n;
    auto load_stage = [&](int st, int b) {
        uint8_t* s0 = wsm + st * 2 * kQ6KSoaBsz;
        if (lane < 14) {
            if (r0) cp_async16(s0 + lane * 16, r0 + static_cast<size_t>(b) * kQ6KSoaBsz + lane * 16);
            if (r1) cp_async16(s0 + kQ6KSoaBsz + lane * 16, r1 + static_cast<size_t>(b) * kQ6KSoaBsz + lane * 16);
        }
        cp_async_commit();
    };
    if (nb > 0) load_stage(0, 0);
    int stage = 0;
    for (int b = 0; b < nb; ++b) {
        if (b + 1 < nb) {
            load_stage(1 - stage, b + 1);
            cp_async_wait1();
        } else {
            cp_async_wait0();
        }
        __syncwarp();
        const float* p0 = x0 + b * 256;
        const float* p1 = x1 + b * 256;
        const float x0a = __ldg(p0 + lane), x0b = __ldg(p0 + 32 + lane);
        const float x0c = __ldg(p0 + 64 + lane), x0d = __ldg(p0 + 96 + lane);
        const float x1a = __ldg(p1 + lane), x1b = __ldg(p1 + 32 + lane);
        const float x1c = __ldg(p1 + 64 + lane), x1d = __ldg(p1 + 96 + lane);
        const uint8_t* s0 = wsm + stage * 2 * kQ6KSoaBsz;
        if (r0) acc_q6_ild_t2(s0, lane, is, p0, p1, x0a, x0b, x0c, x0d, x1a, x1b, x1c, x1d, a00, a01);
        if (r1) acc_q6_ild_t2(s0 + kQ6KSoaBsz, lane, is, p0, p1, x0a, x0b, x0c, x0d, x1a, x1b, x1c, x1d, a10, a11);
        stage ^= 1;
    }
    a00 = warp_sum(a00);
    a01 = warp_sum(a01);
    a10 = warp_sum(a10);
    a11 = warp_sum(a11);
    if (lane != 0) return;
    if (row0 < m) {
        write_y(Y, row0, a00, add);
        write_y(Y + m, row0, a01, add);
    }
    if (row1 < m) {
        write_y(Y, row1, a10, add);
        write_y(Y + m, row1, a11, add);
    }
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q6k_f32_t2_2row(const uint8_t* W, const float* X, float* Y, int m, int n, int add) {
    if (!W || !X || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pairs = (m + 1) / 2;
    const int pb = (pairs + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 4 * kQ6KSoaBsz;
    gemv_q6k_soa_f32_t2_2row_k<<<pb, th, smem>>>(W, X, Y, m, n, add);
}

} // namespace rapidllm::cuda_gemv
