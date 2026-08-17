// T=4 Q6 SOA float GEMV. Isolated TU so nvcc codegen of the T=3 ild kernel
// in cuda_engine.cu is unchanged. Dequant matches gemv_q6k_soa_f32_t3_2row_k.
#include "rapidllm/kernels/gemv_q6_t4.h"

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

// 224-byte Q6 SOA block: 14×16B. Warp-local smem, no block sync.
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


template <int T>
__global__ void __launch_bounds__(256, 2) gemv_q6k_soa_f32_tn_pipe_k(const uint8_t* W, const float* X, float* Y,
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
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a03 = 0.f, a10 = 0.f, a11 = 0.f, a12 = 0.f, a13 = 0.f;
    const uint8_t* r0 = (row0 < m && nb > 0)
                            ? (W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ6KSoaBsz)
                            : nullptr;
    const uint8_t* r1 = (row1 < m && nb > 0)
                            ? (W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ6KSoaBsz)
                            : nullptr;
    const float* x0 = X;
    const float* x1 = X + n;
    const float* x2 = X + 2 * n;
    const float* x3 = T >= 4 ? X + 3 * n : X;
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
        const float x3a = T >= 4 ? __ldg(p3 + lane) : 0.f;
        const float x3b = T >= 4 ? __ldg(p3 + 32 + lane) : 0.f;
        const float x3c = T >= 4 ? __ldg(p3 + 64 + lane) : 0.f;
        const float x3d = T >= 4 ? __ldg(p3 + 96 + lane) : 0.f;
        const uint8_t* blk0 = wsm + stage * 2 * kQ6KSoaBsz;
        const uint8_t* blk1 = blk0 + kQ6KSoaBsz;
        if (r0)
            acc_q6_ild_from_blk(blk0, lane, is, p0, p1, p2, p3, T, x0a, x0b, x0c, x0d, x1a, x1b, x1c, x1d, x2a,
                                x2b, x2c, x2d, x3a, x3b, x3c, x3d, a00, a01, a02, a03);
        if (r1)
            acc_q6_ild_from_blk(blk1, lane, is, p0, p1, p2, p3, T, x0a, x0b, x0c, x0d, x1a, x1b, x1c, x1d, x2a,
                                x2b, x2c, x2d, x3a, x3b, x3c, x3d, a10, a11, a12, a13);
        stage ^= 1;
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        a01 = warp_sum(a01);
        a02 = warp_sum(a02);
        if (T >= 4) a03 = warp_sum(a03);
        if (lane == 0) {
            write_y(Y, row0, a00, add);
            write_y(Y + m, row0, a01, add);
            write_y(Y + 2 * m, row0, a02, add);
            if (T >= 4) write_y(Y + 3 * m, row0, a03, add);
        }
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        a11 = warp_sum(a11);
        a12 = warp_sum(a12);
        if (T >= 4) a13 = warp_sum(a13);
        if (lane == 0) {
            write_y(Y, row1, a10, add);
            write_y(Y + m, row1, a11, add);
            write_y(Y + 2 * m, row1, a12, add);
            if (T >= 4) write_y(Y + 3 * m, row1, a13, add);
        }
    }
}

#if 0
__global__ void gemv_q6k_soa_f32_t4_ild_old(const uint8_t* W, const float* X, float* Y, int m, int n, int add) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int is = lane / 16;
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a03 = 0.f, a10 = 0.f, a11 = 0.f, a12 = 0.f, a13 = 0.f;
    const uint8_t* r0 = nullptr;
    const uint8_t* r1 = nullptr;
    const float* x0 = X;
    const float* x1 = X + n;
    const float* x2 = X + 2 * n;
    const float* x3 = X + 3 * n;
    for (int b = 0; b < nb; ++b) {
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
        auto acc_row = [&](const uint8_t* row, float& a0, float& a1, float& a2, float& a3) {
            const uint8_t* blk = row + static_cast<size_t>(b) * kQ6KSoaBsz;
            const __half* ds = reinterpret_cast<const __half*>(blk);
            const uint8_t* ql = blk + 32;
            const uint8_t* qh = blk + 160;
#pragma unroll
            for (int n128 = 0; n128 < 2; ++n128) {
                const uint8_t qlo = __ldcs(ql + lane);
                const uint8_t qhi = __ldcs(qh + lane);
                const uint8_t qlo2 = __ldcs(ql + 32 + lane);
                const int q1 = static_cast<int>((qlo & 0xF) | (((qhi >> 0) & 3) << 4)) - 32;
                const int q2 = static_cast<int>((qlo2 & 0xF) | (((qhi >> 2) & 3) << 4)) - 32;
                const int q3 = static_cast<int>((qlo >> 4) | (((qhi >> 4) & 3) << 4)) - 32;
                const int q4 = static_cast<int>((qlo2 >> 4) | (((qhi >> 6) & 3) << 4)) - 32;
                const float s1 = __half2float(__ldg(ds + is)) * static_cast<float>(q1);
                const float s2 = __half2float(__ldg(ds + is + 2)) * static_cast<float>(q2);
                const float s3 = __half2float(__ldg(ds + is + 4)) * static_cast<float>(q3);
                const float s4 = __half2float(__ldg(ds + is + 6)) * static_cast<float>(q4);
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
        };
        if (r0) acc_row(r0, a00, a01, a02, a03);
        if (r1) acc_row(r1, a10, a11, a12, a13);
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        a01 = warp_sum(a01);
        a02 = warp_sum(a02);
        a03 = warp_sum(a03);
        if (lane == 0) {
            write_y(Y, row0, a00, add);
            write_y(Y + m, row0, a01, add);
            write_y(Y + 2 * m, row0, a02, add);
            write_y(Y + 3 * m, row0, a03, add);
        }
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        a11 = warp_sum(a11);
        a12 = warp_sum(a12);
        a13 = warp_sum(a13);
        if (lane == 0) {
            write_y(Y, row1, a10, add);
            write_y(Y + m, row1, a11, add);
            write_y(Y + 2 * m, row1, a12, add);
            write_y(Y + 3 * m, row1, a13, add);
        }
    }
}
#endif

constexpr int kQ4KSoaBsz = 160;

__device__ __forceinline__ void acc_q4k_soa_q8_4x(const uint8_t* row, const int8_t* xq0, const __half* sc0,
                                                  const int8_t* xq1, const __half* sc1, const int8_t* xq2,
                                                  const __half* sc2, const int8_t* xq3, const __half* sc3, int nb,
                                                  int lane, float& a0, float& a1, float& a2, float& a3) {
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
        const int w0i = *reinterpret_cast<const int*>(xq2 + off);
        const int w1i = *reinterpret_cast<const int*>(xq2 + off + 4);
        const int z0 = *reinterpret_cast<const int*>(xq3 + off);
        const int z1 = *reinterpret_cast<const int*>(xq3 + off + 4);
        int s0 = 0, sx0 = 0, s1 = 0, sx1 = 0, s2 = 0, sx2 = 0, s3 = 0, sx3 = 0;
        s0 = __dp4a(q0, u0, 0);
        s0 = __dp4a(q1, u1, s0);
        sx0 = __dp4a(0x01010101, u0, 0);
        sx0 = __dp4a(0x01010101, u1, sx0);
        s1 = __dp4a(q0, v0, 0);
        s1 = __dp4a(q1, v1, s1);
        sx1 = __dp4a(0x01010101, v0, 0);
        sx1 = __dp4a(0x01010101, v1, sx1);
        s2 = __dp4a(q0, w0i, 0);
        s2 = __dp4a(q1, w1i, s2);
        sx2 = __dp4a(0x01010101, w0i, 0);
        sx2 = __dp4a(0x01010101, w1i, sx2);
        s3 = __dp4a(q0, z0, 0);
        s3 = __dp4a(q1, z1, s3);
        sx3 = __dp4a(0x01010101, z0, 0);
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
}


__global__ void __launch_bounds__(256, 2) gemv_q4k_soa_q8_t4_dual_k(const uint8_t* W1, const uint8_t* W2,
                                                                    const int8_t* xq, const __half* xsc, float* Y1,
                                                                    float* Y2, int m, int n) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    const int8_t* xq0 = xq;
    const int8_t* xq1 = xq + n;
    const int8_t* xq2 = xq + 2 * n;
    const int8_t* xq3 = xq + 3 * n;
    const __half* sc0 = xsc;
    const __half* sc1 = xsc + nsc;
    const __half* sc2 = xsc + 2 * nsc;
    const __half* sc3 = xsc + 3 * nsc;
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a03 = 0.f, b00 = 0.f, b01 = 0.f, b02 = 0.f, b03 = 0.f;
    float a10 = 0.f, a11 = 0.f, a12 = 0.f, a13 = 0.f, b10 = 0.f, b11 = 0.f, b12 = 0.f, b13 = 0.f;
    if (row0 < m && nb > 0) {
        const uint8_t* r0 = W1 + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        const uint8_t* s0 = W2 + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        acc_q4k_soa_q8_4x(r0, xq0, sc0, xq1, sc1, xq2, sc2, xq3, sc3, nb, lane, a00, a01, a02, a03);
        acc_q4k_soa_q8_4x(s0, xq0, sc0, xq1, sc1, xq2, sc2, xq3, sc3, nb, lane, b00, b01, b02, b03);
        if (row1 < m) {
            const uint8_t* r1 = W1 + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz;
            const uint8_t* s1 = W2 + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz;
            acc_q4k_soa_q8_4x(r1, xq0, sc0, xq1, sc1, xq2, sc2, xq3, sc3, nb, lane, a10, a11, a12, a13);
            acc_q4k_soa_q8_4x(s1, xq0, sc0, xq1, sc1, xq2, sc2, xq3, sc3, nb, lane, b10, b11, b12, b13);
        }
    }
    auto emit = [&](float g, float u, float* yg, float* yu, int row) {
        g = warp_sum(g);
        u = warp_sum(u);
        if (lane != 0) return;
        write_y(yg, row, g, 0);
        if (yu) write_y(yu, row, u, 0);
    };
    if (row0 < m) {
        emit(a00, b00, Y1, Y2, row0);
        emit(a01, b01, Y1 + m, Y2 ? Y2 + m : nullptr, row0);
        emit(a02, b02, Y1 + 2 * m, Y2 ? Y2 + 2 * m : nullptr, row0);
        emit(a03, b03, Y1 + 3 * m, Y2 ? Y2 + 3 * m : nullptr, row0);
    }
    if (row1 < m) {
        emit(a10, b10, Y1, Y2, row1);
        emit(a11, b11, Y1 + m, Y2 ? Y2 + m : nullptr, row1);
        emit(a12, b12, Y1 + 2 * m, Y2 ? Y2 + 2 * m : nullptr, row1);
        emit(a13, b13, Y1 + 3 * m, Y2 ? Y2 + 3 * m : nullptr, row1);
    }
}

// T=4 mixed: one row/warp, Q6 ild + Q4 int. Same math as sequential launches.
__global__ void __launch_bounds__(256, 2) gemv_q6q4_t4_dual_k(const uint8_t* W6, const uint8_t* W4, const float* X,
                                                              const int8_t* xq, const __half* xsc, float* Y6,
                                                              float* Y4, int m6, int m4, int n) {
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int is = lane / 16;
    const int mmax = m6 > m4 ? m6 : m4;
    if (row >= mmax || nb <= 0) return;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
    const uint8_t* r6 = (row < m6) ? (W6 + static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ6KSoaBsz)
                                   : nullptr;
    const uint8_t* r4 = (row < m4) ? (W4 + static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ4KSoaBsz)
                                   : nullptr;
    const float* x0 = X;
    const float* x1 = X + n;
    const float* x2 = X + 2 * n;
    const float* x3 = X + 3 * n;
    const int nsc = n >> 5;
    if (r4)
        acc_q4k_soa_q8_4x(r4, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n, xsc + 3 * nsc,
                          nb, lane, b0, b1, b2, b3);
    if (r6) {
        for (int b = 0; b < nb; ++b) {
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
            const uint8_t* blk = r6 + static_cast<size_t>(b) * kQ6KSoaBsz;
            const __half* ds = reinterpret_cast<const __half*>(blk);
            const uint8_t* ql = blk + 32;
            const uint8_t* qh = blk + 160;
#pragma unroll
            for (int n128 = 0; n128 < 2; ++n128) {
                const uint8_t qlo = __ldcs(ql + lane);
                const uint8_t qhi = __ldcs(qh + lane);
                const uint8_t qlo2 = __ldcs(ql + 32 + lane);
                const int q1 = static_cast<int>((qlo & 0xF) | (((qhi >> 0) & 3) << 4)) - 32;
                const int q2 = static_cast<int>((qlo2 & 0xF) | (((qhi >> 2) & 3) << 4)) - 32;
                const int q3 = static_cast<int>((qlo >> 4) | (((qhi >> 4) & 3) << 4)) - 32;
                const int q4 = static_cast<int>((qlo2 >> 4) | (((qhi >> 6) & 3) << 4)) - 32;
                const float s1 = __half2float(__ldg(ds + is)) * static_cast<float>(q1);
                const float s2 = __half2float(__ldg(ds + is + 2)) * static_cast<float>(q2);
                const float s3 = __half2float(__ldg(ds + is + 4)) * static_cast<float>(q3);
                const float s4 = __half2float(__ldg(ds + is + 6)) * static_cast<float>(q4);
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
    }
    if (row < m6) {
        a0 = warp_sum(a0);
        a1 = warp_sum(a1);
        a2 = warp_sum(a2);
        a3 = warp_sum(a3);
        if (lane == 0) {
            write_y(Y6, row, a0, 0);
            write_y(Y6 + m6, row, a1, 0);
            write_y(Y6 + 2 * m6, row, a2, 0);
            write_y(Y6 + 3 * m6, row, a3, 0);
        }
    }
    if (row < m4) {
        b0 = warp_sum(b0);
        b1 = warp_sum(b1);
        b2 = warp_sum(b2);
        b3 = warp_sum(b3);
        if (lane == 0) {
            write_y(Y4, row, b0, 0);
            write_y(Y4 + m4, row, b1, 0);
            write_y(Y4 + 2 * m4, row, b2, 0);
            write_y(Y4 + 3 * m4, row, b3, 0);
        }
    }
}


// T=6: one W stream, six x. Reuses T=4 acc on the smem tile then T=3 on x4/x5.
__global__ void __launch_bounds__(256, 2) gemv_q6k_soa_f32_t6_pipe_k(const uint8_t* W, const float* X, float* Y,
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
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a03 = 0.f, a04 = 0.f, a05 = 0.f, ad0 = 0.f;
    float a10 = 0.f, a11 = 0.f, a12 = 0.f, a13 = 0.f, a14 = 0.f, a15 = 0.f, ad1 = 0.f;
    const uint8_t* r0 = (row0 < m && nb > 0)
                            ? (W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ6KSoaBsz)
                            : nullptr;
    const uint8_t* r1 = (row1 < m && nb > 0)
                            ? (W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ6KSoaBsz)
                            : nullptr;
    const float* x0 = X;
    const float* x1 = X + n;
    const float* x2 = X + 2 * n;
    const float* x3 = X + 3 * n;
    const float* x4 = X + 4 * n;
    const float* x5 = X + 5 * n;
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
        const float* p0 = x0 + b * 256;
        const float* p1 = x1 + b * 256;
        const float* p2 = x2 + b * 256;
        const float* p3 = x3 + b * 256;
        const float* p4 = x4 + b * 256;
        const float* p5 = x5 + b * 256;
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
        const uint8_t* blk0 = wsm + stage * 2 * kQ6KSoaBsz;
        const uint8_t* blk1 = blk0 + kQ6KSoaBsz;
        if (r0) {
            acc_q6_ild_from_blk(blk0, lane, is, p0, p1, p2, p3, 4, x0a, x0b, x0c, x0d, x1a, x1b, x1c, x1d, x2a,
                                x2b, x2c, x2d, x3a, x3b, x3c, x3d, a00, a01, a02, a03);
            acc_q6_ild_from_blk(blk0, lane, is, p4, p5, p5, p5, 3, x4a, x4b, x4c, x4d, x5a, x5b, x5c, x5d, x5a,
                                x5b, x5c, x5d, 0.f, 0.f, 0.f, 0.f, a04, a05, ad0, ad0);
        }
        if (r1) {
            acc_q6_ild_from_blk(blk1, lane, is, p0, p1, p2, p3, 4, x0a, x0b, x0c, x0d, x1a, x1b, x1c, x1d, x2a,
                                x2b, x2c, x2d, x3a, x3b, x3c, x3d, a10, a11, a12, a13);
            acc_q6_ild_from_blk(blk1, lane, is, p4, p5, p5, p5, 3, x4a, x4b, x4c, x4d, x5a, x5b, x5c, x5d, x5a,
                                x5b, x5c, x5d, 0.f, 0.f, 0.f, 0.f, a14, a15, ad1, ad1);
        }
        stage ^= 1;
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        a01 = warp_sum(a01);
        a02 = warp_sum(a02);
        a03 = warp_sum(a03);
        a04 = warp_sum(a04);
        a05 = warp_sum(a05);
        if (lane == 0) {
            write_y(Y, row0, a00, add);
            write_y(Y + m, row0, a01, add);
            write_y(Y + 2 * m, row0, a02, add);
            write_y(Y + 3 * m, row0, a03, add);
            write_y(Y + 4 * m, row0, a04, add);
            write_y(Y + 5 * m, row0, a05, add);
        }
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        a11 = warp_sum(a11);
        a12 = warp_sum(a12);
        a13 = warp_sum(a13);
        a14 = warp_sum(a14);
        a15 = warp_sum(a15);
        if (lane == 0) {
            write_y(Y, row1, a10, add);
            write_y(Y + m, row1, a11, add);
            write_y(Y + 2 * m, row1, a12, add);
            write_y(Y + 3 * m, row1, a13, add);
            write_y(Y + 4 * m, row1, a14, add);
            write_y(Y + 5 * m, row1, a15, add);
        }
    }
}

__device__ __forceinline__ void acc_q4k_soa_q8_6x(const uint8_t* row, const int8_t* xq0, const __half* sc0,
                                                  const int8_t* xq1, const __half* sc1, const int8_t* xq2,
                                                  const __half* sc2, const int8_t* xq3, const __half* sc3,
                                                  const int8_t* xq4, const __half* sc4, const int8_t* xq5,
                                                  const __half* sc5, int nb, int lane, float& a0, float& a1,
                                                  float& a2, float& a3, float& a4, float& a5) {
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
        auto dp = [&](const int8_t* xq, const __half* sc, float& acc) {
            const int u0 = *reinterpret_cast<const int*>(xq + off);
            const int u1 = *reinterpret_cast<const int*>(xq + off + 4);
            int s = __dp4a(q0, u0, 0);
            s = __dp4a(q1, u1, s);
            int sx = __dp4a(0x01010101, u0, 0);
            sx = __dp4a(0x01010101, u1, sx);
            const float xs = __half2float(sc[b * 8 + group]);
            acc = fmaf(dsg * xs, static_cast<float>(s), acc);
            acc = fmaf(-dmg * xs, static_cast<float>(sx), acc);
        };
        dp(xq0, sc0, a0);
        dp(xq1, sc1, a1);
        dp(xq2, sc2, a2);
        dp(xq3, sc3, a3);
        dp(xq4, sc4, a4);
        dp(xq5, sc5, a5);
    }
}

__global__ void __launch_bounds__(256, 2) gemv_q4k_soa_q8_t6_k(const uint8_t* W, const int8_t* xq, const __half* xsc,
                                                               float* Y, int m, int n, int add) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a03 = 0.f, a04 = 0.f, a05 = 0.f;
    float a10 = 0.f, a11 = 0.f, a12 = 0.f, a13 = 0.f, a14 = 0.f, a15 = 0.f;
    if (row0 < m && nb > 0) {
        const uint8_t* r0 = W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        acc_q4k_soa_q8_6x(r0, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n, xsc + 3 * nsc,
                          xq + 4 * n, xsc + 4 * nsc, xq + 5 * n, xsc + 5 * nsc, nb, lane, a00, a01, a02, a03,
                          a04, a05);
        if (row1 < m) {
            const uint8_t* r1 = W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz;
            acc_q4k_soa_q8_6x(r1, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n,
                              xsc + 3 * nsc, xq + 4 * n, xsc + 4 * nsc, xq + 5 * n, xsc + 5 * nsc, nb, lane, a10,
                              a11, a12, a13, a14, a15);
        }
    }
    auto emit6 = [&](float v0, float v1, float v2, float v3, float v4, float v5, int row) {
        v0 = warp_sum(v0);
        v1 = warp_sum(v1);
        v2 = warp_sum(v2);
        v3 = warp_sum(v3);
        v4 = warp_sum(v4);
        v5 = warp_sum(v5);
        if (lane != 0) return;
        write_y(Y, row, v0, add);
        write_y(Y + m, row, v1, add);
        write_y(Y + 2 * m, row, v2, add);
        write_y(Y + 3 * m, row, v3, add);
        write_y(Y + 4 * m, row, v4, add);
        write_y(Y + 5 * m, row, v5, add);
    };
    if (row0 < m) emit6(a00, a01, a02, a03, a04, a05, row0);
    if (row1 < m) emit6(a10, a11, a12, a13, a14, a15, row1);
}

__global__ void __launch_bounds__(256, 2) gemv_q4k_soa_q8_t6_dual_k(const uint8_t* W1, const uint8_t* W2,
                                                                    const int8_t* xq, const __half* xsc, float* Y1,
                                                                    float* Y2, int m, int n) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a03 = 0.f, a04 = 0.f, a05 = 0.f;
    float b00 = 0.f, b01 = 0.f, b02 = 0.f, b03 = 0.f, b04 = 0.f, b05 = 0.f;
    float a10 = 0.f, a11 = 0.f, a12 = 0.f, a13 = 0.f, a14 = 0.f, a15 = 0.f;
    float b10 = 0.f, b11 = 0.f, b12 = 0.f, b13 = 0.f, b14 = 0.f, b15 = 0.f;
    if (row0 < m && nb > 0) {
        const uint8_t* r0 = W1 + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        const uint8_t* s0 = W2 + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        acc_q4k_soa_q8_6x(r0, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n, xsc + 3 * nsc,
                          xq + 4 * n, xsc + 4 * nsc, xq + 5 * n, xsc + 5 * nsc, nb, lane, a00, a01, a02, a03,
                          a04, a05);
        acc_q4k_soa_q8_6x(s0, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n, xsc + 3 * nsc,
                          xq + 4 * n, xsc + 4 * nsc, xq + 5 * n, xsc + 5 * nsc, nb, lane, b00, b01, b02, b03,
                          b04, b05);
        if (row1 < m) {
            const uint8_t* r1 = W1 + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz;
            const uint8_t* s1 = W2 + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz;
            acc_q4k_soa_q8_6x(r1, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n,
                              xsc + 3 * nsc, xq + 4 * n, xsc + 4 * nsc, xq + 5 * n, xsc + 5 * nsc, nb, lane, a10,
                              a11, a12, a13, a14, a15);
            acc_q4k_soa_q8_6x(s1, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n,
                              xsc + 3 * nsc, xq + 4 * n, xsc + 4 * nsc, xq + 5 * n, xsc + 5 * nsc, nb, lane, b10,
                              b11, b12, b13, b14, b15);
        }
    }
    auto emit = [&](float g, float u, float* yg, float* yu, int row) {
        g = warp_sum(g);
        u = warp_sum(u);
        if (lane != 0) return;
        write_y(yg, row, g, 0);
        if (yu) write_y(yu, row, u, 0);
    };
    if (row0 < m) {
        emit(a00, b00, Y1, Y2, row0);
        emit(a01, b01, Y1 + m, Y2 ? Y2 + m : nullptr, row0);
        emit(a02, b02, Y1 + 2 * m, Y2 ? Y2 + 2 * m : nullptr, row0);
        emit(a03, b03, Y1 + 3 * m, Y2 ? Y2 + 3 * m : nullptr, row0);
        emit(a04, b04, Y1 + 4 * m, Y2 ? Y2 + 4 * m : nullptr, row0);
        emit(a05, b05, Y1 + 5 * m, Y2 ? Y2 + 5 * m : nullptr, row0);
    }
    if (row1 < m) {
        emit(a10, b10, Y1, Y2, row1);
        emit(a11, b11, Y1 + m, Y2 ? Y2 + m : nullptr, row1);
        emit(a12, b12, Y1 + 2 * m, Y2 ? Y2 + 2 * m : nullptr, row1);
        emit(a13, b13, Y1 + 3 * m, Y2 ? Y2 + 3 * m : nullptr, row1);
        emit(a14, b14, Y1 + 4 * m, Y2 ? Y2 + 4 * m : nullptr, row1);
        emit(a15, b15, Y1 + 5 * m, Y2 ? Y2 + 5 * m : nullptr, row1);
    }
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q6k_f32_t3_ild(const uint8_t* W, const float* X, float* Y, int m, int n, int add) {
    if (!W || !X || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int pairs = (m + 1) / 2;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (pairs + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * 2 * 224;
    gemv_q6k_soa_f32_tn_pipe_k<3><<<pb, th, smem>>>(W, X, Y, m, n, add);
}

void launch_q6k_f32_t4_ild(const uint8_t* W, const float* X, float* Y, int m, int n, int add) {
    if (!W || !X || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int pairs = (m + 1) / 2;
    const int t4th = 256;
    const int t4w = t4th / 32;
    const int t4pb = (pairs + t4w - 1) / t4w;
    const size_t smem = static_cast<size_t>(t4w) * 2 * 2 * 224;
    gemv_q6k_soa_f32_tn_pipe_k<4><<<t4pb, t4th, smem>>>(W, X, Y, m, n, add);
}

void launch_q6k_f32_t6_ild(const uint8_t* W, const float* X, float* Y, int m, int n, int add) {
    if (!W || !X || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int pairs = (m + 1) / 2;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (pairs + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * 2 * 224;
    gemv_q6k_soa_f32_t6_pipe_k<<<pb, th, smem>>>(W, X, Y, m, n, add);
}

void launch_q4k_q8_t6(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n, int add) {
    if (!W || !xq || !xsc || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int pairs = (m + 1) / 2;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (pairs + tw - 1) / tw;
    gemv_q4k_soa_q8_t6_k<<<pb, th>>>(W, xq, xsc, Y, m, n, add);
}

void launch_q4k_q8_t6_dual(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                           float* Y1, float* Y2, int m, int n) {
    if (!W1 || !W2 || !xq || !xsc || !Y1 || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int pairs = (m + 1) / 2;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (pairs + tw - 1) / tw;
    gemv_q4k_soa_q8_t6_dual_k<<<pb, th>>>(W1, W2, xq, xsc, Y1, Y2, m, n);
}

void launch_q4k_q8_t4_dual(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                           float* Y1, float* Y2, int m, int n) {
    if (!W1 || !W2 || !xq || !xsc || !Y1 || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int pairs = (m + 1) / 2;
    const int t4th = 256;
    const int t4w = t4th / 32;
    const int t4pb = (pairs + t4w - 1) / t4w;
    gemv_q4k_soa_q8_t4_dual_k<<<t4pb, t4th>>>(W1, W2, xq, xsc, Y1, Y2, m, n);
}


void launch_q6q4_t4_dual(const uint8_t* W6, const uint8_t* W4, const float* X, const int8_t* xq,
                         const __half* xsc, float* Y6, float* Y4, int m6, int m4, int n) {
    if (!W6 || !W4 || !X || !xq || !xsc || !Y6 || !Y4 || m6 <= 0 || m4 <= 0 || n <= 0 || (n % 256) != 0)
        return;
    const int mmax = m6 > m4 ? m6 : m4;
    const int t4th = 256;
    const int t4w = t4th / 32;
    const int t4pb = (mmax + t4w - 1) / t4w;
    gemv_q6q4_t4_dual_k<<<t4pb, t4th>>>(W6, W4, X, xq, xsc, Y6, Y4, m6, m4, n);
}

} // namespace rapidllm::cuda_gemv
