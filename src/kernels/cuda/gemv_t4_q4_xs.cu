// Isolated T=4 Q4: pipe W + group xsum from ensure_xq (no extra dp4a).
// Same nibble dequant as gemv_t4_pipe acc_q4k_smem_4x.
#include "rapidllm/kernels/gemv_t4_q4_xs.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

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

__device__ __forceinline__ void cp_async_q4_blk(uint8_t* dst, const uint8_t* src, int lane) {
    if (src && lane < 10) cp_async16(dst + lane * 16, src + lane * 16);
}

__device__ __forceinline__ void acc_q4k_smem_4x_xs(const uint8_t* blk, const int8_t* xq0, const __half* sc0,
                                                   const int8_t* xq1, const __half* sc1, const int8_t* xq2,
                                                   const __half* sc2, const int8_t* xq3, const __half* sc3,
                                                   const int32_t* xs0, const int32_t* xs1, const int32_t* xs2,
                                                   const int32_t* xs3, int b, int lane, float& a0, float& a1,
                                                   float& a2, float& a3) {
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
    int s1 = __dp4a(q0, v0, 0);
    s1 = __dp4a(q1, v1, s1);
    int s2 = __dp4a(q0, w0i, 0);
    s2 = __dp4a(q1, w1i, s2);
    int s3 = __dp4a(q0, z0, 0);
    s3 = __dp4a(q1, z1, s3);
    const int gix = b * 8 + group;
    const float xs0v = __half2float(sc0[gix]);
    const float xs1v = __half2float(sc1[gix]);
    const float xs2v = __half2float(sc2[gix]);
    const float xs3v = __half2float(sc3[gix]);
    a0 = fmaf(dsg * xs0v, static_cast<float>(s0), a0);
    a1 = fmaf(dsg * xs1v, static_cast<float>(s1), a1);
    a2 = fmaf(dsg * xs2v, static_cast<float>(s2), a2);
    a3 = fmaf(dsg * xs3v, static_cast<float>(s3), a3);
    // Full 32-wide group xsum lives on one lane; warp_sum folds it later.
    if (sub == 0) {
        a0 = fmaf(-dmg * xs0v, static_cast<float>(xs0[gix]), a0);
        a1 = fmaf(-dmg * xs1v, static_cast<float>(xs1[gix]), a1);
        a2 = fmaf(-dmg * xs2v, static_cast<float>(xs2[gix]), a2);
        a3 = fmaf(-dmg * xs3v, static_cast<float>(xs3[gix]), a3);
    }
}

__global__ void __launch_bounds__(256, 4) gemv_q4k_soa_q8_t4_xs_k(const uint8_t* W, const int8_t* xq,
                                                                  const __half* xsc, const int32_t* xsum, float* Y,
                                                                  int m, int n, int add) {
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    const int warp = threadIdx.x / 32;
    extern __shared__ uint8_t smem_w[];
    uint8_t* wsm = smem_w + warp * (2 * kQ4KSoaBsz);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    const uint8_t* r0 =
        (row < m && nb > 0) ? (W + static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ4KSoaBsz) : nullptr;
    auto load_stage = [&](int st, int b) {
        cp_async_q4_blk(wsm + st * kQ4KSoaBsz, r0 ? r0 + static_cast<size_t>(b) * kQ4KSoaBsz : nullptr, lane);
        cp_async_commit();
    };
    if (nb > 0) load_stage(0, 0);
    int stage = 0;
    const int32_t* t0s = xsum;
    const int32_t* t1s = xsum + nsc;
    const int32_t* t2s = xsum + 2 * nsc;
    const int32_t* t3s = xsum + 3 * nsc;
    for (int b = 0; b < nb; ++b) {
        if (b + 1 < nb) {
            load_stage(1 - stage, b + 1);
            cp_async_wait(1);
        } else {
            cp_async_wait(0);
        }
        __syncwarp();
        if (r0)
            acc_q4k_smem_4x_xs(wsm + stage * kQ4KSoaBsz, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc,
                               xq + 3 * n, xsc + 3 * nsc, t0s, t1s, t2s, t3s, b, lane, a0, a1, a2, a3);
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

__global__ void __launch_bounds__(256, 3) gemv_q4k_soa_q8_t4_dual_xs_k(const uint8_t* W1, const uint8_t* W2,
                                                                       const int8_t* xq, const __half* xsc,
                                                                       const int32_t* xsum, float* Y1, float* Y2,
                                                                       int m, int n) {
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    const int warp = threadIdx.x / 32;
    extern __shared__ uint8_t smem_d[];
    uint8_t* wsm = smem_d + warp * (2 * 2 * kQ4KSoaBsz);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
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
    if (nb > 0) load_stage(0, 0);
    int stage = 0;
    const int32_t* t0s = xsum;
    const int32_t* t1s = xsum + nsc;
    const int32_t* t2s = xsum + 2 * nsc;
    const int32_t* t3s = xsum + 3 * nsc;
    for (int b = 0; b < nb; ++b) {
        if (b + 1 < nb) {
            load_stage(1 - stage, b + 1);
            cp_async_wait(1);
        } else {
            cp_async_wait(0);
        }
        __syncwarp();
        const uint8_t* s = wsm + stage * (2 * kQ4KSoaBsz);
        if (r1)
            acc_q4k_smem_4x_xs(s, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n, xsc + 3 * nsc,
                               t0s, t1s, t2s, t3s, b, lane, a0, a1, a2, a3);
        if (r2)
            acc_q4k_smem_4x_xs(s + kQ4KSoaBsz, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n,
                               xsc + 3 * nsc, t0s, t1s, t2s, t3s, b, lane, b0, b1, b2, b3);
        stage ^= 1;
    }
    if (row >= m) return;
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    a2 = warp_sum(a2);
    a3 = warp_sum(a3);
    b0 = warp_sum(b0);
    b1 = warp_sum(b1);
    b2 = warp_sum(b2);
    b3 = warp_sum(b3);
    if (lane != 0) return;
    write_y(Y1, row, a0, 0);
    write_y(Y1 + m, row, a1, 0);
    write_y(Y1 + 2 * m, row, a2, 0);
    write_y(Y1 + 3 * m, row, a3, 0);
    if (Y2) {
        write_y(Y2, row, b0, 0);
        write_y(Y2 + m, row, b1, 0);
        write_y(Y2 + 2 * m, row, b2, 0);
        write_y(Y2 + 3 * m, row, b3, 0);
    }
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q4k_q8_t4_1row_xs(const uint8_t* W, const int8_t* xq, const __half* xsc, const int32_t* xsum,
                              float* Y, int m, int n, int add) {
    if (!W || !xq || !xsc || !xsum || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * kQ4KSoaBsz;
    gemv_q4k_soa_q8_t4_xs_k<<<pb, th, smem>>>(W, xq, xsc, xsum, Y, m, n, add);
}

void launch_q4k_q8_t4_1row_dual_xs(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                                   const int32_t* xsum, float* Y1, float* Y2, int m, int n) {
    if (!W1 || !W2 || !xq || !xsc || !xsum || !Y1 || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * 2 * kQ4KSoaBsz;
    gemv_q4k_soa_q8_t4_dual_xs_k<<<pb, th, smem>>>(W1, W2, xq, xsc, xsum, Y1, Y2, m, n);
}

} // namespace rapidllm::cuda_gemv
