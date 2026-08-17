// T=2 Q6 ild + Q4 int GEMV. Isolated TU so nvcc codegen of T=3/T=4/T=6
// ild in gemv_q6_t4.cu and cuda_engine.cu is unchanged. Q6 dequant matches
// gemv_q6k_soa_f32_tn_pipe_k; Q4 dequant matches acc_q4k_soa_q8_2x.
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

// Same Q6 SOA dequant as acc_q6_ild_from_blk; only two x vectors.
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

__global__ void __launch_bounds__(256, 2) gemv_q6k_soa_f32_t2_pipe_k(const uint8_t* W, const float* X, float* Y,
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
    float a00 = 0.f, a01 = 0.f, a10 = 0.f, a11 = 0.f;
    const uint8_t* r0 = (row0 < m && nb > 0)
                            ? (W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ6KSoaBsz)
                            : nullptr;
    const uint8_t* r1 = (row1 < m && nb > 0)
                            ? (W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ6KSoaBsz)
                            : nullptr;
    const float* x0 = X;
    const float* x1 = X + n;
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
        const float x0a = __ldg(p0 + lane), x0b = __ldg(p0 + 32 + lane);
        const float x0c = __ldg(p0 + 64 + lane), x0d = __ldg(p0 + 96 + lane);
        const float x1a = __ldg(p1 + lane), x1b = __ldg(p1 + 32 + lane);
        const float x1c = __ldg(p1 + 64 + lane), x1d = __ldg(p1 + 96 + lane);
        const uint8_t* blk0 = wsm + stage * 2 * kQ6KSoaBsz;
        const uint8_t* blk1 = blk0 + kQ6KSoaBsz;
        if (r0) acc_q6_ild_t2(blk0, lane, is, p0, p1, x0a, x0b, x0c, x0d, x1a, x1b, x1c, x1d, a00, a01);
        if (r1) acc_q6_ild_t2(blk1, lane, is, p0, p1, x0a, x0b, x0c, x0d, x1a, x1b, x1c, x1d, a10, a11);
        stage ^= 1;
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        a01 = warp_sum(a01);
        if (lane == 0) {
            write_y(Y, row0, a00, add);
            write_y(Y + m, row0, a01, add);
        }
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        a11 = warp_sum(a11);
        if (lane == 0) {
            write_y(Y, row1, a10, add);
            write_y(Y + m, row1, a11, add);
        }
    }
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

__global__ void __launch_bounds__(256, 2) gemv_q4k_soa_q8_t2_2row_k(const uint8_t* W, const int8_t* xq,
                                                                   const __half* xsc, float* Y, int m, int n,
                                                                   int add) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    const int8_t* xq0 = xq;
    const int8_t* xq1 = xq + n;
    const __half* sc0 = xsc;
    const __half* sc1 = xsc + nsc;
    float a00 = 0.f, a01 = 0.f, a10 = 0.f, a11 = 0.f;
    if (row0 < m && nb > 0) {
        const uint8_t* r0 = W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        const uint8_t* r1 = (row1 < m) ? (W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz)
                                       : nullptr;
        for (int b0 = 0; b0 < nb; ++b0) {
            acc_q4k_soa_q8_2x(r0 + static_cast<size_t>(b0) * kQ4KSoaBsz, xq0 + b0 * 256, sc0 + b0 * 8,
                              xq1 + b0 * 256, sc1 + b0 * 8, 1, lane, a00, a01);
            if (r1)
                acc_q4k_soa_q8_2x(r1 + static_cast<size_t>(b0) * kQ4KSoaBsz, xq0 + b0 * 256, sc0 + b0 * 8,
                                  xq1 + b0 * 256, sc1 + b0 * 8, 1, lane, a10, a11);
        }
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        a01 = warp_sum(a01);
        if (lane == 0) {
            write_y(Y, row0, a00, add);
            write_y(Y + m, row0, a01, add);
        }
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        a11 = warp_sum(a11);
        if (lane == 0) {
            write_y(Y, row1, a10, add);
            write_y(Y + m, row1, a11, add);
        }
    }
}

__global__ void __launch_bounds__(256, 2) gemv_q4k_soa_q8_t2_dual_k(const uint8_t* W1, const uint8_t* W2,
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
    const __half* sc0 = xsc;
    const __half* sc1 = xsc + nsc;
    float a00 = 0.f, a01 = 0.f, b00 = 0.f, b01 = 0.f;
    float a10 = 0.f, a11 = 0.f, b10 = 0.f, b11 = 0.f;
    if (row0 < m && nb > 0) {
        const uint8_t* r0 = W1 + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        const uint8_t* s0 = W2 + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        const uint8_t* r1 = (row1 < m) ? (W1 + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz)
                                       : nullptr;
        const uint8_t* s1 = (row1 < m) ? (W2 + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz)
                                       : nullptr;
        for (int b0 = 0; b0 < nb; ++b0) {
            acc_q4k_soa_q8_2x(r0 + static_cast<size_t>(b0) * kQ4KSoaBsz, xq0 + b0 * 256, sc0 + b0 * 8,
                              xq1 + b0 * 256, sc1 + b0 * 8, 1, lane, a00, a01);
            acc_q4k_soa_q8_2x(s0 + static_cast<size_t>(b0) * kQ4KSoaBsz, xq0 + b0 * 256, sc0 + b0 * 8,
                              xq1 + b0 * 256, sc1 + b0 * 8, 1, lane, b00, b01);
            if (r1)
                acc_q4k_soa_q8_2x(r1 + static_cast<size_t>(b0) * kQ4KSoaBsz, xq0 + b0 * 256, sc0 + b0 * 8,
                                  xq1 + b0 * 256, sc1 + b0 * 8, 1, lane, a10, a11);
            if (s1)
                acc_q4k_soa_q8_2x(s1 + static_cast<size_t>(b0) * kQ4KSoaBsz, xq0 + b0 * 256, sc0 + b0 * 8,
                                  xq1 + b0 * 256, sc1 + b0 * 8, 1, lane, b10, b11);
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
    }
    if (row1 < m) {
        emit(a10, b10, Y1, Y2, row1);
        emit(a11, b11, Y1 + m, Y2 ? Y2 + m : nullptr, row1);
    }
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q6k_f32_t2_ild(const uint8_t* W, const float* X, float* Y, int m, int n, int add) {
    if (!W || !X || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int pairs = (m + 1) / 2;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (pairs + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * 2 * kQ6KSoaBsz;
    gemv_q6k_soa_f32_t2_pipe_k<<<pb, th, smem>>>(W, X, Y, m, n, add);
}

void launch_q4k_q8_t2(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n, int add) {
    if (!W || !xq || !xsc || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int pairs = (m + 1) / 2;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (pairs + tw - 1) / tw;
    gemv_q4k_soa_q8_t2_2row_k<<<pb, th>>>(W, xq, xsc, Y, m, n, add);
}

void launch_q4k_q8_t2_dual(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc, float* Y1,
                           float* Y2, int m, int n) {
    if (!W1 || !W2 || !xq || !xsc || !Y1 || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int pairs = (m + 1) / 2;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (pairs + tw - 1) / tw;
    gemv_q4k_soa_q8_t2_dual_k<<<pb, th>>>(W1, W2, xq, xsc, Y1, Y2, m, n);
}

} // namespace rapidllm::cuda_gemv
