// Isolated T=16 Q4 GEMV. cp.async W pipeline; acc is four T=4 groups so
// dequant matches gemv_t4_pipe acc_q4k_smem_4x.
#include "rapidllm/kernels/gemv_q4_t16_pipe.h"

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

__device__ __forceinline__ void acc16(const uint8_t* blk, const int8_t* xq, const __half* xsc, int n, int nsc,
                                      int b, int lane, float* a) {
    acc_q4k_smem_4x(blk, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n, xsc + 3 * nsc, b,
                    lane, a[0], a[1], a[2], a[3]);
    acc_q4k_smem_4x(blk, xq + 4 * n, xsc + 4 * nsc, xq + 5 * n, xsc + 5 * nsc, xq + 6 * n, xsc + 6 * nsc,
                    xq + 7 * n, xsc + 7 * nsc, b, lane, a[4], a[5], a[6], a[7]);
    acc_q4k_smem_4x(blk, xq + 8 * n, xsc + 8 * nsc, xq + 9 * n, xsc + 9 * nsc, xq + 10 * n, xsc + 10 * nsc,
                    xq + 11 * n, xsc + 11 * nsc, b, lane, a[8], a[9], a[10], a[11]);
    acc_q4k_smem_4x(blk, xq + 12 * n, xsc + 12 * nsc, xq + 13 * n, xsc + 13 * nsc, xq + 14 * n, xsc + 14 * nsc,
                    xq + 15 * n, xsc + 15 * nsc, b, lane, a[12], a[13], a[14], a[15]);
}

__device__ __forceinline__ void emit16(float* Y, int m, int row, int add, int lane, float* a) {
#pragma unroll
    for (int t = 0; t < 16; ++t) a[t] = warp_sum(a[t]);
    if (lane != 0) return;
#pragma unroll
    for (int t = 0; t < 16; ++t) write_y(Y + t * m, row, a[t], add);
}

__global__ void __launch_bounds__(256, 2) gemv_q4k_soa_q8_t16_1row_pipe_k(const uint8_t* W, const int8_t* xq,
                                                                         const __half* xsc, float* Y, int m, int n,
                                                                         int add) {
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    const int warp = threadIdx.x / 32;
    extern __shared__ uint8_t smem_w[];
    uint8_t* wsm = smem_w + warp * (2 * kQ4KSoaBsz);
    float a[16] = {};
    const uint8_t* r0 =
        (row < m && nb > 0) ? (W + static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ4KSoaBsz) : nullptr;
    auto load_stage = [&](int st, int b) {
        cp_async_q4_blk(wsm + st * kQ4KSoaBsz, r0 ? r0 + static_cast<size_t>(b) * kQ4KSoaBsz : nullptr, lane);
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
        if (r0) acc16(wsm + stage * kQ4KSoaBsz, xq, xsc, n, nsc, b, lane, a);
        stage ^= 1;
    }
    if (row < m) emit16(Y, m, row, add, lane, a);
}

__global__ void __launch_bounds__(256, 2) gemv_q4k_soa_q8_t16_1row_dual_pipe_k(const uint8_t* W1, const uint8_t* W2,
                                                                              const int8_t* xq, const __half* xsc,
                                                                              float* Y1, float* Y2, int m, int n) {
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    const int warp = threadIdx.x / 32;
    extern __shared__ uint8_t smem_d[];
    uint8_t* wsm = smem_d + warp * (2 * kQ4KSoaBsz);
    auto run_w = [&](const uint8_t* W, float* Y) {
        float a[16] = {};
        const uint8_t* r0 =
            (row < m && nb > 0 && W) ? (W + static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ4KSoaBsz)
                                     : nullptr;
        auto load_stage = [&](int st, int b) {
            cp_async_q4_blk(wsm + st * kQ4KSoaBsz, r0 ? r0 + static_cast<size_t>(b) * kQ4KSoaBsz : nullptr, lane);
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
            if (r0) acc16(wsm + stage * kQ4KSoaBsz, xq, xsc, n, nsc, b, lane, a);
            stage ^= 1;
        }
        if (Y && row < m) emit16(Y, m, row, 0, lane, a);
    };
    run_w(W1, Y1);
    run_w(W2, Y2);
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q4k_q8_t16_1row_pipe(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n,
                                 int add) {
    if (!W || !xq || !xsc || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * kQ4KSoaBsz;
    gemv_q4k_soa_q8_t16_1row_pipe_k<<<pb, th, smem>>>(W, xq, xsc, Y, m, n, add);
}

void launch_q4k_q8_t16_1row_dual_pipe(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                                      float* Y1, float* Y2, int m, int n) {
    if (!W1 || !W2 || !xq || !xsc || !Y1 || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * kQ4KSoaBsz;
    gemv_q4k_soa_q8_t16_1row_dual_pipe_k<<<pb, th, smem>>>(W1, W2, xq, xsc, Y1, Y2, m, n);
}

} // namespace rapidllm::cuda_gemv
