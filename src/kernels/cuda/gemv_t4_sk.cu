// Isolated T=4 1-row split-K. Two warps share a row: each runs the same
// acc_q6_ild_t4 / acc_q4k_smem_4x over a K-half, then one add of the two
// warp sums. Per-weight math matches gemv_t4_1row / gemv_t4_pipe.
#include "rapidllm/kernels/gemv_t4_sk.h"

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

__device__ __forceinline__ void cp_async_q4_blk(uint8_t* dst, const uint8_t* src, int lane) {
    if (src && lane < 10) cp_async16(dst + lane * 16, src + lane * 16);
}

// Exact copy of gemv_t4_1row.cu acc_q6_ild_t4.
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

// Exact copy of gemv_t4_pipe.cu acc_q4k_smem_4x.
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

__global__ void __launch_bounds__(256, 3) gemv_q6k_soa_f32_t4_1row_sk_k(const uint8_t* W, const float* X, float* Y,
                                                                       int m, int n, int add) {
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int pair = warp >> 1;
    const int half = warp & 1;
    const int nrows = warps >> 1;
    const int row = blockIdx.x * nrows + pair;
    const int nb = n / 256;
    const int mid = (nb + 1) >> 1;
    const int b0 = half ? mid : 0;
    const int b1 = half ? nb : mid;
    const int is = lane / 16;
    extern __shared__ uint8_t smem_raw[];
    uint8_t* wsm = smem_raw + warp * (2 * kQ6KSoaBsz);
    float* part = reinterpret_cast<float*>(smem_raw + warps * (2 * kQ6KSoaBsz));
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
    if (b0 < b1) load_stage(0, b0);
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
        if (r0)
            acc_q6_ild_t4(wsm + stage * kQ6KSoaBsz, lane, is, p0, p1, p2, p3, x0a, x0b, x0c, x0d, x1a, x1b,
                          x1c, x1d, x2a, x2b, x2c, x2d, x3a, x3b, x3c, x3d, a0, a1, a2, a3);
        stage ^= 1;
    }
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    a2 = warp_sum(a2);
    a3 = warp_sum(a3);
    if (lane == 0) {
        part[warp * 4 + 0] = a0;
        part[warp * 4 + 1] = a1;
        part[warp * 4 + 2] = a2;
        part[warp * 4 + 3] = a3;
    }
    __syncthreads();
    if (row >= m || half != 0 || lane != 0) return;
    write_y(Y, row, part[warp * 4 + 0] + part[(warp + 1) * 4 + 0], add);
    write_y(Y + m, row, part[warp * 4 + 1] + part[(warp + 1) * 4 + 1], add);
    write_y(Y + 2 * m, row, part[warp * 4 + 2] + part[(warp + 1) * 4 + 2], add);
    write_y(Y + 3 * m, row, part[warp * 4 + 3] + part[(warp + 1) * 4 + 3], add);
}

__global__ void __launch_bounds__(256, 3) gemv_q4k_soa_q8_t4_1row_sk_k(const uint8_t* W, const int8_t* xq,
                                                                       const __half* xsc, float* Y, int m, int n,
                                                                       int add) {
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int pair = warp >> 1;
    const int half = warp & 1;
    const int nrows = warps >> 1;
    const int row = blockIdx.x * nrows + pair;
    const int nb = n / 256;
    const int nsc = n >> 5;
    const int mid = (nb + 1) >> 1;
    const int b0 = half ? mid : 0;
    const int b1 = half ? nb : mid;
    extern __shared__ uint8_t smem_q4[];
    uint8_t* wsm = smem_q4 + warp * (2 * kQ4KSoaBsz);
    float* part = reinterpret_cast<float*>(smem_q4 + warps * (2 * kQ4KSoaBsz));
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    const uint8_t* r0 =
        (row < m && nb > 0) ? (W + static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ4KSoaBsz) : nullptr;
    auto load_stage = [&](int st, int b) {
        cp_async_q4_blk(wsm + st * kQ4KSoaBsz, r0 ? r0 + static_cast<size_t>(b) * kQ4KSoaBsz : nullptr, lane);
        cp_async_commit();
    };
    if (b0 < b1) load_stage(0, b0);
    int stage = 0;
    for (int b = b0; b < b1; ++b) {
        if (b + 1 < b1) {
            load_stage(1 - stage, b + 1);
            cp_async_wait(1);
        } else {
            cp_async_wait(0);
        }
        __syncwarp();
        if (r0)
            acc_q4k_smem_4x(wsm + stage * kQ4KSoaBsz, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc,
                            xq + 3 * n, xsc + 3 * nsc, b, lane, a0, a1, a2, a3);
        stage ^= 1;
    }
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    a2 = warp_sum(a2);
    a3 = warp_sum(a3);
    if (lane == 0) {
        part[warp * 4 + 0] = a0;
        part[warp * 4 + 1] = a1;
        part[warp * 4 + 2] = a2;
        part[warp * 4 + 3] = a3;
    }
    __syncthreads();
    if (row >= m || half != 0 || lane != 0) return;
    write_y(Y, row, part[warp * 4 + 0] + part[(warp + 1) * 4 + 0], add);
    write_y(Y + m, row, part[warp * 4 + 1] + part[(warp + 1) * 4 + 1], add);
    write_y(Y + 2 * m, row, part[warp * 4 + 2] + part[(warp + 1) * 4 + 2], add);
    write_y(Y + 3 * m, row, part[warp * 4 + 3] + part[(warp + 1) * 4 + 3], add);
}

__global__ void __launch_bounds__(256, 2) gemv_q4k_soa_q8_t4_dual_sk_k(const uint8_t* W1, const uint8_t* W2,
                                                                      const int8_t* xq, const __half* xsc,
                                                                      float* Y1, float* Y2, int m, int n) {
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int pair = warp >> 1;
    const int half = warp & 1;
    const int nrows = warps >> 1;
    const int row = blockIdx.x * nrows + pair;
    const int nb = n / 256;
    const int nsc = n >> 5;
    const int mid = (nb + 1) >> 1;
    const int b0 = half ? mid : 0;
    const int b1 = half ? nb : mid;
    extern __shared__ uint8_t smem_d[];
    uint8_t* wsm = smem_d + warp * (2 * 2 * kQ4KSoaBsz);
    float* part = reinterpret_cast<float*>(smem_d + warps * (2 * 2 * kQ4KSoaBsz));
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    const uint8_t* r1 =
        (row < m && nb > 0) ? (W1 + static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ4KSoaBsz) : nullptr;
    const uint8_t* r2 =
        (row < m && nb > 0) ? (W2 + static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ4KSoaBsz) : nullptr;
    auto load_stage = [&](int st, int b) {
        uint8_t* s = wsm + st * (2 * kQ4KSoaBsz);
        cp_async_q4_blk(s, r1 ? r1 + static_cast<size_t>(b) * kQ4KSoaBsz : nullptr, lane);
        cp_async_q4_blk(s + kQ4KSoaBsz, r2 ? r2 + static_cast<size_t>(b) * kQ4KSoaBsz : nullptr, lane);
        cp_async_commit();
    };
    if (b0 < b1) load_stage(0, b0);
    int stage = 0;
    for (int b = b0; b < b1; ++b) {
        if (b + 1 < b1) {
            load_stage(1 - stage, b + 1);
            cp_async_wait(1);
        } else {
            cp_async_wait(0);
        }
        __syncwarp();
        const uint8_t* s = wsm + stage * (2 * kQ4KSoaBsz);
        if (r1)
            acc_q4k_smem_4x(s, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n, xsc + 3 * nsc, b,
                            lane, a0, a1, a2, a3);
        if (r2)
            acc_q4k_smem_4x(s + kQ4KSoaBsz, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n,
                            xsc + 3 * nsc, b, lane, c0, c1, c2, c3);
        stage ^= 1;
    }
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    a2 = warp_sum(a2);
    a3 = warp_sum(a3);
    c0 = warp_sum(c0);
    c1 = warp_sum(c1);
    c2 = warp_sum(c2);
    c3 = warp_sum(c3);
    if (lane == 0) {
        part[warp * 8 + 0] = a0;
        part[warp * 8 + 1] = a1;
        part[warp * 8 + 2] = a2;
        part[warp * 8 + 3] = a3;
        part[warp * 8 + 4] = c0;
        part[warp * 8 + 5] = c1;
        part[warp * 8 + 6] = c2;
        part[warp * 8 + 7] = c3;
    }
    __syncthreads();
    if (row >= m || half != 0 || lane != 0) return;
    write_y(Y1, row, part[warp * 8 + 0] + part[(warp + 1) * 8 + 0], 0);
    write_y(Y1 + m, row, part[warp * 8 + 1] + part[(warp + 1) * 8 + 1], 0);
    write_y(Y1 + 2 * m, row, part[warp * 8 + 2] + part[(warp + 1) * 8 + 2], 0);
    write_y(Y1 + 3 * m, row, part[warp * 8 + 3] + part[(warp + 1) * 8 + 3], 0);
    if (Y2) {
        write_y(Y2, row, part[warp * 8 + 4] + part[(warp + 1) * 8 + 4], 0);
        write_y(Y2 + m, row, part[warp * 8 + 5] + part[(warp + 1) * 8 + 5], 0);
        write_y(Y2 + 2 * m, row, part[warp * 8 + 6] + part[(warp + 1) * 8 + 6], 0);
        write_y(Y2 + 3 * m, row, part[warp * 8 + 7] + part[(warp + 1) * 8 + 7], 0);
    }
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q6k_f32_t4_1row_sk(const uint8_t* W, const float* X, float* Y, int m, int n, int add) {
    if (!W || !X || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int nrows = tw >> 1;
    const int pb = (m + nrows - 1) / nrows;
    const size_t smem = static_cast<size_t>(tw) * 2 * kQ6KSoaBsz + static_cast<size_t>(tw) * 4 * sizeof(float);
    gemv_q6k_soa_f32_t4_1row_sk_k<<<pb, th, smem>>>(W, X, Y, m, n, add);
}

void launch_q4k_q8_t4_1row_pipe_sk(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n,
                                   int add) {
    if (!W || !xq || !xsc || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int nrows = tw >> 1;
    const int pb = (m + nrows - 1) / nrows;
    const size_t smem = static_cast<size_t>(tw) * 2 * kQ4KSoaBsz + static_cast<size_t>(tw) * 4 * sizeof(float);
    gemv_q4k_soa_q8_t4_1row_sk_k<<<pb, th, smem>>>(W, xq, xsc, Y, m, n, add);
}

void launch_q4k_q8_t4_1row_dual_pipe_sk(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                                        float* Y1, float* Y2, int m, int n) {
    if (!W1 || !W2 || !xq || !xsc || !Y1 || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int nrows = tw >> 1;
    const int pb = (m + nrows - 1) / nrows;
    const size_t smem = static_cast<size_t>(tw) * 2 * 2 * kQ4KSoaBsz + static_cast<size_t>(tw) * 8 * sizeof(float);
    gemv_q4k_soa_q8_t4_dual_sk_k<<<pb, th, smem>>>(W1, W2, xq, xsc, Y1, Y2, m, n);
}

} // namespace rapidllm::cuda_gemv
