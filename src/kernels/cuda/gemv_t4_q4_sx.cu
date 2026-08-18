// Isolated T=4 Q4 1-row / dual with X-only sx precomputed once per K-block.
// Same q/min dequant as gemv_t4_pipe.cu acc_q4k_smem_4x.
#include "rapidllm/kernels/gemv_t4_q4_sx.h"
#include "rapidllm/kernels/gemv_t4_pipe.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kQ4KSoaBsz = 160;
constexpr int kSxNbMax = 128;

__device__ int d_sx_t4[4 * kSxNbMax * 32];

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

__global__ void prep_sx_t4_k(const int8_t* xq, int n, int nb) {
    const int b = blockIdx.x;
    const int t = blockIdx.y;
    const int lane = threadIdx.x;
    if (b >= nb || t >= 4 || lane >= 32) return;
    const int group = lane >> 2;
    const int sub = lane & 3;
    const int off = t * n + b * 256 + group * 32 + sub * 8;
    const int u0 = *reinterpret_cast<const int*>(xq + off);
    const int u1 = *reinterpret_cast<const int*>(xq + off + 4);
    int sx = __dp4a(0x01010101, u0, 0);
    sx = __dp4a(0x01010101, u1, sx);
    d_sx_t4[(static_cast<int>(t) * kSxNbMax + b) * 32 + lane] = sx;
}

__device__ __forceinline__ void acc_q4k_sx_4x(const uint8_t* blk, const int8_t* xq0, const __half* sc0,
                                              const int8_t* xq1, const __half* sc1, const int8_t* xq2,
                                              const __half* sc2, const int8_t* xq3, const __half* sc3, int b,
                                              int lane, const int* sx0, const int* sx1, const int* sx2,
                                              const int* sx3, float& a0, float& a1, float& a2, float& a3) {
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
    const int sxv0 = sx0[lane];
    const int sxv1 = sx1[lane];
    const int sxv2 = sx2[lane];
    const int sxv3 = sx3[lane];
    const float xs0 = __half2float(sc0[b * 8 + group]);
    const float xs1 = __half2float(sc1[b * 8 + group]);
    const float xs2 = __half2float(sc2[b * 8 + group]);
    const float xs3 = __half2float(sc3[b * 8 + group]);
    a0 = fmaf(dsg * xs0, static_cast<float>(s0), a0);
    a0 = fmaf(-dmg * xs0, static_cast<float>(sxv0), a0);
    a1 = fmaf(dsg * xs1, static_cast<float>(s1), a1);
    a1 = fmaf(-dmg * xs1, static_cast<float>(sxv1), a1);
    a2 = fmaf(dsg * xs2, static_cast<float>(s2), a2);
    a2 = fmaf(-dmg * xs2, static_cast<float>(sxv2), a2);
    a3 = fmaf(dsg * xs3, static_cast<float>(s3), a3);
    a3 = fmaf(-dmg * xs3, static_cast<float>(sxv3), a3);
}

__global__ void __launch_bounds__(256, 4) gemv_q4k_q8_t4_1row_sx_k(const uint8_t* W, const int8_t* xq,
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
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
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
        const int* sx0 = d_sx_t4 + (0 * kSxNbMax + b) * 32;
        const int* sx1 = d_sx_t4 + (1 * kSxNbMax + b) * 32;
        const int* sx2 = d_sx_t4 + (2 * kSxNbMax + b) * 32;
        const int* sx3 = d_sx_t4 + (3 * kSxNbMax + b) * 32;
        if (r0)
            acc_q4k_sx_4x(wsm + stage * kQ4KSoaBsz, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc,
                          xq + 3 * n, xsc + 3 * nsc, b, lane, sx0, sx1, sx2, sx3, a0, a1, a2, a3);
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

__global__ void __launch_bounds__(256, 2) gemv_q4k_q8_t4_1row_dual_sx_k(const uint8_t* W1, const uint8_t* W2,
                                                                        const int8_t* xq, const __half* xsc,
                                                                        float* Y1, float* Y2, int m, int n) {
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
    for (int b = 0; b < nb; ++b) {
        if (b + 1 < nb) {
            load_stage(1 - stage, b + 1);
            cp_async_wait(1);
        } else {
            cp_async_wait(0);
        }
        __syncwarp();
        const uint8_t* s = wsm + stage * (2 * kQ4KSoaBsz);
        const int* sx0 = d_sx_t4 + (0 * kSxNbMax + b) * 32;
        const int* sx1 = d_sx_t4 + (1 * kSxNbMax + b) * 32;
        const int* sx2 = d_sx_t4 + (2 * kSxNbMax + b) * 32;
        const int* sx3 = d_sx_t4 + (3 * kSxNbMax + b) * 32;
        if (r1)
            acc_q4k_sx_4x(s, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n, xsc + 3 * nsc, b,
                          lane, sx0, sx1, sx2, sx3, a0, a1, a2, a3);
        if (r2)
            acc_q4k_sx_4x(s + kQ4KSoaBsz, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n,
                          xsc + 3 * nsc, b, lane, sx0, sx1, sx2, sx3, b0, b1, b2, b3);
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

void prep_sx(const int8_t* xq, int n) {
    const int nb = n / 256;
    if (nb <= 0 || nb > kSxNbMax) return;
    prep_sx_t4_k<<<dim3(nb, 4), 32>>>(xq, n, nb);
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q4k_q8_t4_1row_sx(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n,
                              int add) {
    if (!W || !xq || !xsc || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int nb = n / 256;
    if (nb > kSxNbMax) {
        launch_q4k_q8_t4_1row_pipe(W, xq, xsc, Y, m, n, add);
        return;
    }
    prep_sx(xq, n);
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * kQ4KSoaBsz;
    gemv_q4k_q8_t4_1row_sx_k<<<pb, th, smem>>>(W, xq, xsc, Y, m, n, add);
}

void launch_q4k_q8_t4_1row_dual_sx(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                                   float* Y1, float* Y2, int m, int n) {
    if (!W1 || !W2 || !xq || !xsc || !Y1 || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int nb = n / 256;
    if (nb > kSxNbMax) {
        launch_q4k_q8_t4_1row_dual_pipe(W1, W2, xq, xsc, Y1, Y2, m, n);
        return;
    }
    prep_sx(xq, n);
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * 2 * kQ4KSoaBsz;
    gemv_q4k_q8_t4_1row_dual_sx_k<<<pb, th, smem>>>(W1, W2, xq, xsc, Y1, Y2, m, n);
}

} // namespace rapidllm::cuda_gemv
