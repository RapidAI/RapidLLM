// T=12 Q6/Q4 GEMV. Isolated TU so T=3/T=4/T=6 codegen in gemv_q6_t4.cu is unchanged.
// Q6 dequant matches acc_q6_ild_from_blk (T=4 then T=3, twice). Not a fused 12-way acc.
#include "rapidllm/kernels/gemv_q6_t4.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kQ6KSoaBsz = 224;
constexpr int kQ4KSoaBsz = 160;

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

// One W stream, twelve x. Two T=6 double-passes on the same smem tile.
__global__ void __launch_bounds__(256, 2) gemv_q6k_soa_f32_t12_pipe_k(const uint8_t* W, const float* X, float* Y,
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
    uint8_t* wsm = smem_w + warp * (2 * 2 * kQ6KSoaBsz);
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a03 = 0.f, a04 = 0.f, a05 = 0.f;
    float a06 = 0.f, a07 = 0.f, a08 = 0.f, a09 = 0.f, a0a = 0.f, a0b = 0.f, ad0 = 0.f;
    float a10 = 0.f, a11 = 0.f, a12 = 0.f, a13 = 0.f, a14 = 0.f, a15 = 0.f;
    float a16 = 0.f, a17 = 0.f, a18 = 0.f, a19 = 0.f, a1a = 0.f, a1b = 0.f, ad1 = 0.f;
    const uint8_t* r0 = (row0 < m && nb > 0)
                            ? (W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ6KSoaBsz)
                            : nullptr;
    const uint8_t* r1 = (row1 < m && nb > 0)
                            ? (W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ6KSoaBsz)
                            : nullptr;
    const float* xs[12];
#pragma unroll
    for (int t = 0; t < 12; ++t) xs[t] = X + t * n;
    auto load_stage = [&](int st, int b) {
        uint8_t* s0 = wsm + st * 2 * kQ6KSoaBsz;
        uint8_t* s1 = s0 + kQ6KSoaBsz;
        cp_async_q6_blk(s0, r0 ? r0 + static_cast<size_t>(b) * kQ6KSoaBsz : nullptr, lane);
        cp_async_q6_blk(s1, r1 ? r1 + static_cast<size_t>(b) * kQ6KSoaBsz : nullptr, lane);
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
        const uint8_t* blk0 = wsm + stage * 2 * kQ6KSoaBsz;
        const uint8_t* blk1 = blk0 + kQ6KSoaBsz;
        auto acc6 = [&](int t0, float& a0, float& a1, float& a2, float& a3, float& a4, float& a5, float& ad,
                        const uint8_t* blk) {
            const float* p0 = xs[t0] + b * 256;
            const float* p1 = xs[t0 + 1] + b * 256;
            const float* p2 = xs[t0 + 2] + b * 256;
            const float* p3 = xs[t0 + 3] + b * 256;
            const float* p4 = xs[t0 + 4] + b * 256;
            const float* p5 = xs[t0 + 5] + b * 256;
            const float x0a = __ldg(p0 + lane), x0b = __ldg(p0 + 32 + lane);
            const float x0c = __ldg(p0 + 64 + lane), x0d = __ldg(p0 + 96 + lane);
            const float x1a = __ldg(p1 + lane), x1b = __ldg(p1 + 32 + lane);
            const float x1c = __ldg(p1 + 64 + lane), x1d = __ldg(p1 + 96 + lane);
            const float x2a = __ldg(p2 + lane), x2b = __ldg(p2 + 32 + lane);
            const float x2c = __ldg(p2 + 64 + lane), x2d = __ldg(p2 + 96 + lane);
            const float x3a = __ldg(p3 + lane), x3b = __ldg(p3 + 32 + lane);
            const float x3c = __ldg(p3 + 64 + lane), x3d = __ldg(p3 + 96 + lane);
            const float x4a = __ldg(p4 + lane), x4b = __ldg(p4 + 32 + lane);
            const float x4c = __ldg(p4 + 64 + lane), x4d = __ldg(p4 + 96 + lane);
            const float x5a = __ldg(p5 + lane), x5b = __ldg(p5 + 32 + lane);
            const float x5c = __ldg(p5 + 64 + lane), x5d = __ldg(p5 + 96 + lane);
            acc_q6_ild_from_blk(blk, lane, is, p0, p1, p2, p3, 4, x0a, x0b, x0c, x0d, x1a, x1b, x1c, x1d, x2a,
                                x2b, x2c, x2d, x3a, x3b, x3c, x3d, a0, a1, a2, a3);
            acc_q6_ild_from_blk(blk, lane, is, p4, p5, p5, p5, 3, x4a, x4b, x4c, x4d, x5a, x5b, x5c, x5d, x5a,
                                x5b, x5c, x5d, 0.f, 0.f, 0.f, 0.f, a4, a5, ad, ad);
        };
        if (r0) {
            acc6(0, a00, a01, a02, a03, a04, a05, ad0, blk0);
            acc6(6, a06, a07, a08, a09, a0a, a0b, ad0, blk0);
        }
        if (r1) {
            acc6(0, a10, a11, a12, a13, a14, a15, ad1, blk1);
            acc6(6, a16, a17, a18, a19, a1a, a1b, ad1, blk1);
        }
        stage ^= 1;
    }
    auto emit12 = [&](float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8,
                      float v9, float va, float vb, int row) {
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
        write_y(Y, row, v0, add);
        write_y(Y + m, row, v1, add);
        write_y(Y + 2 * m, row, v2, add);
        write_y(Y + 3 * m, row, v3, add);
        write_y(Y + 4 * m, row, v4, add);
        write_y(Y + 5 * m, row, v5, add);
        write_y(Y + 6 * m, row, v6, add);
        write_y(Y + 7 * m, row, v7, add);
        write_y(Y + 8 * m, row, v8, add);
        write_y(Y + 9 * m, row, v9, add);
        write_y(Y + 10 * m, row, va, add);
        write_y(Y + 11 * m, row, vb, add);
    };
    if (row0 < m) emit12(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a0a, a0b, row0);
    if (row1 < m) emit12(a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a1a, a1b, row1);
}

__global__ void __launch_bounds__(256, 2) gemv_q4k_soa_q8_t12_k(const uint8_t* W, const int8_t* xq,
                                                                const __half* xsc, float* Y, int m, int n, int add) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a03 = 0.f, a04 = 0.f, a05 = 0.f;
    float a06 = 0.f, a07 = 0.f, a08 = 0.f, a09 = 0.f, a0a = 0.f, a0b = 0.f;
    float a10 = 0.f, a11 = 0.f, a12 = 0.f, a13 = 0.f, a14 = 0.f, a15 = 0.f;
    float a16 = 0.f, a17 = 0.f, a18 = 0.f, a19 = 0.f, a1a = 0.f, a1b = 0.f;
    if (row0 < m && nb > 0) {
        const uint8_t* r0 = W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        acc_q4k_soa_q8_12x(r0, xq, xsc, n, nsc, nb, lane, a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a0a,
                           a0b);
        if (row1 < m) {
            const uint8_t* r1 = W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz;
            acc_q4k_soa_q8_12x(r1, xq, xsc, n, nsc, nb, lane, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                               a1a, a1b);
        }
    }
    auto emit12 = [&](float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8,
                      float v9, float va, float vb, int row) {
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
        write_y(Y, row, v0, add);
        write_y(Y + m, row, v1, add);
        write_y(Y + 2 * m, row, v2, add);
        write_y(Y + 3 * m, row, v3, add);
        write_y(Y + 4 * m, row, v4, add);
        write_y(Y + 5 * m, row, v5, add);
        write_y(Y + 6 * m, row, v6, add);
        write_y(Y + 7 * m, row, v7, add);
        write_y(Y + 8 * m, row, v8, add);
        write_y(Y + 9 * m, row, v9, add);
        write_y(Y + 10 * m, row, va, add);
        write_y(Y + 11 * m, row, vb, add);
    };
    if (row0 < m) emit12(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a0a, a0b, row0);
    if (row1 < m) emit12(a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a1a, a1b, row1);
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q6k_f32_t12_ild(const uint8_t* W, const float* X, float* Y, int m, int n, int add) {
    if (!W || !X || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int pairs = (m + 1) / 2;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (pairs + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * 2 * 224;
    gemv_q6k_soa_f32_t12_pipe_k<<<pb, th, smem>>>(W, X, Y, m, n, add);
}

void launch_q4k_q8_t12(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n, int add) {
    if (!W || !xq || !xsc || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int pairs = (m + 1) / 2;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (pairs + tw - 1) / tw;
    gemv_q4k_soa_q8_t12_k<<<pb, th>>>(W, xq, xsc, Y, m, n, add);
}

} // namespace rapidllm::cuda_gemv
