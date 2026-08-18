// Isolated T=4 fused Q6+Q4 1-row. Independent accs copied from
// gemv_t4_1row.cu / gemv_t4_pipe.cu so reduction order matches sequential
// launches. Q4 W is fetched in the same K-loop as Q6.
#include "rapidllm/kernels/gemv_t4_q6q4.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kQ6KSoaBsz = 224;
constexpr int kQ4KSoaBsz = 160;
constexpr int kStage = kQ6KSoaBsz + kQ4KSoaBsz;

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ void write_y(float* y, int row, float acc) {
    y[row] = acc;
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

__device__ __forceinline__ void acc_q6_ild_t4(const uint8_t* blk, int lane, int is, const float* p0,
                                              const float* p1, const float* p2, const float* p3, float x0a,
                                              float x0b, float x0c, float x0d, float x1a, float x1b, float x1c,
                                              float x1d, float x2a, float x2b, float x2c, float x2d, float x3a,
                                              float x3b, float x3c, float x3d, float& a0, float& a1, float& a2,
                                              float& a3) {
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
            a3 = fmaf(s1, x3a, a3);
            a3 = fmaf(s2, x3b, a3);
            a3 = fmaf(s3, x3c, a3);
            a3 = fmaf(s4, x3d, a3);
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
            a3 = fmaf(s1, __ldg(p3 + 128 + lane), a3);
            a3 = fmaf(s2, __ldg(p3 + 160 + lane), a3);
            a3 = fmaf(s3, __ldg(p3 + 192 + lane), a3);
            a3 = fmaf(s4, __ldg(p3 + 224 + lane), a3);
        }
        ql += 64;
        qh += 32;
        ds += 8;
    }
}

__device__ __forceinline__ void acc_q4k_smem_4x(const uint8_t* blk, const int8_t* xq0, const __half* sc0,
                                                const int8_t* xq1, const __half* sc1, const int8_t* xq2,
                                                const __half* sc2, const int8_t* xq3, const __half* sc3, int b,
                                                int lane, float& a0, float& a1, float& a2, float& a3) {
    const int group = lane >> 2;
    const int sub = lane & 3;
    const int gpair = group >> 1;
    const int hi = group & 1;
    const __half* ds = reinterpret_cast<const __half*>(blk);
    const __half* dm = reinterpret_cast<const __half*>(blk + 16);
    const uint8_t* qs = blk + 32 + gpair * 32 + sub * 8;
    const uint2 q8 = *reinterpret_cast<const uint2*>(qs);
    const int q0 = static_cast<int>(hi ? ((q8.x >> 4) & 0x0f0f0f0f) : (q8.x & 0x0f0f0f0f));
    const int q1 = static_cast<int>(hi ? ((q8.y >> 4) & 0x0f0f0f0f) : (q8.y & 0x0f0f0f0f));
    const float dsg = __half2float(ds[group]);
    const float dmg = __half2float(dm[group]);
    const int off = b * 256 + group * 32 + sub * 8;
    const int u0 = *reinterpret_cast<const int*>(xq0 + off);
    const int u1 = *reinterpret_cast<const int*>(xq0 + off + 4);
    const int v0 = *reinterpret_cast<const int*>(xq1 + off);
    const int v1 = *reinterpret_cast<const int*>(xq1 + off + 4);
    const int w0i = *reinterpret_cast<const int*>(xq2 + off);
    const int w1i = *reinterpret_cast<const int*>(xq2 + off + 4);
    const int z0 = *reinterpret_cast<const int*>(xq3 + off);
    const int z1 = *reinterpret_cast<const int*>(xq3 + off + 4);
    int s0 = __dp4a(q0, u0, 0);
    s0 = __dp4a(q1, u1, s0);
    int sx0 = __dp4a(0x01010101, u0, 0);
    sx0 = __dp4a(0x01010101, u1, sx0);
    int s1 = __dp4a(q0, v0, 0);
    s1 = __dp4a(q1, v1, s1);
    int sx1 = __dp4a(0x01010101, v0, 0);
    sx1 = __dp4a(0x01010101, v1, sx1);
    int s2 = __dp4a(q0, w0i, 0);
    s2 = __dp4a(q1, w1i, s2);
    int sx2 = __dp4a(0x01010101, w0i, 0);
    sx2 = __dp4a(0x01010101, w1i, sx2);
    int s3 = __dp4a(q0, z0, 0);
    s3 = __dp4a(q1, z1, s3);
    int sx3 = __dp4a(0x01010101, z0, 0);
    sx3 = __dp4a(0x01010101, z1, sx3);
    const float xs0 = __half2float(sc0[b * 8 + group]);
    const float xs1 = __half2float(sc1[b * 8 + group]);
    const float xs2 = __half2float(sc2[b * 8 + group]);
    const float xs3 = __half2float(sc3[b * 8 + group]);
    a0 = fmaf(dsg * xs0, static_cast<float>(s0), a0);
    a0 = fmaf(-dmg * xs0, static_cast<float>(sx0), a0);
    a1 = fmaf(dsg * xs1, static_cast<float>(s1), a1);
    a1 = fmaf(-dmg * xs1, static_cast<float>(sx1), a1);
    a2 = fmaf(dsg * xs2, static_cast<float>(s2), a2);
    a2 = fmaf(-dmg * xs2, static_cast<float>(sx2), a2);
    a3 = fmaf(dsg * xs3, static_cast<float>(s3), a3);
    a3 = fmaf(-dmg * xs3, static_cast<float>(sx3), a3);
}

__global__ void __launch_bounds__(256, 2) gemv_q6q4_t4_1row_pair_k(const uint8_t* W6, const float* X, float* Y6,
                                                                   int m6, const uint8_t* W4, const int8_t* xq,
                                                                   const __half* xsc, float* Y4, int m4, int n) {
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    const int is = lane / 16;
    const int warp = threadIdx.x / 32;
    extern __shared__ uint8_t smem[];
    uint8_t* wsm = smem + warp * (2 * kStage);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
    const uint8_t* r6 =
        (row < m6 && nb > 0) ? (W6 + static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ6KSoaBsz) : nullptr;
    const uint8_t* r4 =
        (row < m4 && nb > 0) ? (W4 + static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ4KSoaBsz) : nullptr;
    auto load_stage = [&](int st, int b) {
        uint8_t* s = wsm + st * kStage;
        if (r6 && lane < 14) cp_async16(s + lane * 16, r6 + static_cast<size_t>(b) * kQ6KSoaBsz + lane * 16);
        if (r4 && lane < 10)
            cp_async16(s + kQ6KSoaBsz + lane * 16, r4 + static_cast<size_t>(b) * kQ4KSoaBsz + lane * 16);
        cp_async_commit();
    };
    if (nb > 0) load_stage(0, 0);
    int stage = 0;
    const float* x0 = X;
    const float* x1 = X + n;
    const float* x2 = X + 2 * n;
    const float* x3 = X + 3 * n;
    for (int b = 0; b < nb; ++b) {
        if (b + 1 < nb) {
            load_stage(1 - stage, b + 1);
            cp_async_wait(1);
        } else {
            cp_async_wait(0);
        }
        __syncwarp();
        const uint8_t* s = wsm + stage * kStage;
        if (r6) {
            const float* p0 = x0 + b * 256;
            const float* p1 = x1 + b * 256;
            const float* p2 = x2 + b * 256;
            const float* p3 = x3 + b * 256;
            const float x0a = __ldg(p0 + lane), x0b = __ldg(p0 + 32 + lane);
            const float x0c = __ldg(p0 + 64 + lane), x0d = __ldg(p0 + 96 + lane);
            const float x1a = __ldg(p1 + lane), x1b = __ldg(p1 + 32 + lane);
            const float x1c = __ldg(p1 + 64 + lane), x1d = __ldg(p1 + 96 + lane);
            const float x2a = __ldg(p2 + lane), x2b = __ldg(p2 + 32 + lane);
            const float x2c = __ldg(p2 + 64 + lane), x2d = __ldg(p2 + 96 + lane);
            const float x3a = __ldg(p3 + lane), x3b = __ldg(p3 + 32 + lane);
            const float x3c = __ldg(p3 + 64 + lane), x3d = __ldg(p3 + 96 + lane);
            acc_q6_ild_t4(s, lane, is, p0, p1, p2, p3, x0a, x0b, x0c, x0d, x1a, x1b, x1c, x1d, x2a, x2b, x2c,
                          x2d, x3a, x3b, x3c, x3d, a0, a1, a2, a3);
        }
        if (r4)
            acc_q4k_smem_4x(s + kQ6KSoaBsz, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n,
                            xsc + 3 * nsc, b, lane, b0, b1, b2, b3);
        stage ^= 1;
    }
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    a2 = warp_sum(a2);
    a3 = warp_sum(a3);
    b0 = warp_sum(b0);
    b1 = warp_sum(b1);
    b2 = warp_sum(b2);
    b3 = warp_sum(b3);
    if (lane != 0) return;
    if (row < m6) {
        write_y(Y6, row, a0);
        write_y(Y6 + m6, row, a1);
        write_y(Y6 + 2 * m6, row, a2);
        write_y(Y6 + 3 * m6, row, a3);
    }
    if (row < m4 && Y4) {
        write_y(Y4, row, b0);
        write_y(Y4 + m4, row, b1);
        write_y(Y4 + 2 * m4, row, b2);
        write_y(Y4 + 3 * m4, row, b3);
    }
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q6q4_t4_1row_pair(const uint8_t* W6, const float* X, float* Y6, int m6, const uint8_t* W4,
                              const int8_t* xq, const __half* xsc, float* Y4, int m4, int n) {
    if (!W6 || !X || !Y6 || !W4 || !xq || !xsc || !Y4 || m6 <= 0 || m4 <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int m = m6 > m4 ? m6 : m4;
    const int pb = (m + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * kStage;
    gemv_q6q4_t4_1row_pair_k<<<pb, th, smem>>>(W6, X, Y6, m6, W4, xq, xsc, Y4, m4, n);
}

} // namespace rapidllm::cuda_gemv
