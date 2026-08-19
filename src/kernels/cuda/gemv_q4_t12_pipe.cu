// Isolated T=12 Q4 GEMV. cp.async W pipeline; acc is three T=4 groups so
// dequant matches gemv_t4_pipe acc_q4k_smem_4x / t12_1row acc_q4k_soa_q8_12x.
#include "rapidllm/kernels/gemv_q4_t12_pipe.h"

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

// Same math as gemv_t4_pipe acc_q4k_smem_4x.
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

__global__ void __launch_bounds__(256, 4) gemv_q4k_soa_q8_t12_1row_pipe_k(const uint8_t* W, const int8_t* xq,
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
    float a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
    float a8 = 0.f, a9 = 0.f, aa = 0.f, ab = 0.f;
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
        if (r0) {
            const uint8_t* blk = wsm + stage * kQ4KSoaBsz;
            acc_q4k_smem_4x(blk, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n,
                            xsc + 3 * nsc, b, lane, a0, a1, a2, a3);
            acc_q4k_smem_4x(blk, xq + 4 * n, xsc + 4 * nsc, xq + 5 * n, xsc + 5 * nsc, xq + 6 * n, xsc + 6 * nsc,
                            xq + 7 * n, xsc + 7 * nsc, b, lane, a4, a5, a6, a7);
            acc_q4k_smem_4x(blk, xq + 8 * n, xsc + 8 * nsc, xq + 9 * n, xsc + 9 * nsc, xq + 10 * n, xsc + 10 * nsc,
                            xq + 11 * n, xsc + 11 * nsc, b, lane, a8, a9, aa, ab);
        }
        stage ^= 1;
    }
    if (row >= m) return;
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    a2 = warp_sum(a2);
    a3 = warp_sum(a3);
    a4 = warp_sum(a4);
    a5 = warp_sum(a5);
    a6 = warp_sum(a6);
    a7 = warp_sum(a7);
    a8 = warp_sum(a8);
    a9 = warp_sum(a9);
    aa = warp_sum(aa);
    ab = warp_sum(ab);
    if (lane != 0) return;
    write_y(Y, row, a0, add);
    write_y(Y + m, row, a1, add);
    write_y(Y + 2 * m, row, a2, add);
    write_y(Y + 3 * m, row, a3, add);
    write_y(Y + 4 * m, row, a4, add);
    write_y(Y + 5 * m, row, a5, add);
    write_y(Y + 6 * m, row, a6, add);
    write_y(Y + 7 * m, row, a7, add);
    write_y(Y + 8 * m, row, a8, add);
    write_y(Y + 9 * m, row, a9, add);
    write_y(Y + 10 * m, row, aa, add);
    write_y(Y + 11 * m, row, ab, add);
}

__global__ void __launch_bounds__(256, 3) gemv_q4k_soa_q8_t12_1row_dual_pipe_k(const uint8_t* W1, const uint8_t* W2,
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
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f;
    float a6 = 0.f, a7 = 0.f, a8 = 0.f, a9 = 0.f, aa = 0.f, ab = 0.f;
    float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f, b4 = 0.f, b5 = 0.f;
    float b6 = 0.f, b7 = 0.f, b8 = 0.f, b9 = 0.f, ba = 0.f, bb = 0.f;
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
        if (r1) {
            acc_q4k_smem_4x(s, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n, xsc + 3 * nsc, b,
                            lane, a0, a1, a2, a3);
            acc_q4k_smem_4x(s, xq + 4 * n, xsc + 4 * nsc, xq + 5 * n, xsc + 5 * nsc, xq + 6 * n, xsc + 6 * nsc,
                            xq + 7 * n, xsc + 7 * nsc, b, lane, a4, a5, a6, a7);
            acc_q4k_smem_4x(s, xq + 8 * n, xsc + 8 * nsc, xq + 9 * n, xsc + 9 * nsc, xq + 10 * n, xsc + 10 * nsc,
                            xq + 11 * n, xsc + 11 * nsc, b, lane, a8, a9, aa, ab);
        }
        if (r2) {
            acc_q4k_smem_4x(s + kQ4KSoaBsz, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n,
                            xsc + 3 * nsc, b, lane, b0, b1, b2, b3);
            acc_q4k_smem_4x(s + kQ4KSoaBsz, xq + 4 * n, xsc + 4 * nsc, xq + 5 * n, xsc + 5 * nsc, xq + 6 * n,
                            xsc + 6 * nsc, xq + 7 * n, xsc + 7 * nsc, b, lane, b4, b5, b6, b7);
            acc_q4k_smem_4x(s + kQ4KSoaBsz, xq + 8 * n, xsc + 8 * nsc, xq + 9 * n, xsc + 9 * nsc, xq + 10 * n,
                            xsc + 10 * nsc, xq + 11 * n, xsc + 11 * nsc, b, lane, b8, b9, ba, bb);
        }
        stage ^= 1;
    }
    if (row >= m) return;
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    a2 = warp_sum(a2);
    a3 = warp_sum(a3);
    a4 = warp_sum(a4);
    a5 = warp_sum(a5);
    a6 = warp_sum(a6);
    a7 = warp_sum(a7);
    a8 = warp_sum(a8);
    a9 = warp_sum(a9);
    aa = warp_sum(aa);
    ab = warp_sum(ab);
    b0 = warp_sum(b0);
    b1 = warp_sum(b1);
    b2 = warp_sum(b2);
    b3 = warp_sum(b3);
    b4 = warp_sum(b4);
    b5 = warp_sum(b5);
    b6 = warp_sum(b6);
    b7 = warp_sum(b7);
    b8 = warp_sum(b8);
    b9 = warp_sum(b9);
    ba = warp_sum(ba);
    bb = warp_sum(bb);
    if (lane != 0) return;
    write_y(Y1, row, a0, 0);
    write_y(Y1 + m, row, a1, 0);
    write_y(Y1 + 2 * m, row, a2, 0);
    write_y(Y1 + 3 * m, row, a3, 0);
    write_y(Y1 + 4 * m, row, a4, 0);
    write_y(Y1 + 5 * m, row, a5, 0);
    write_y(Y1 + 6 * m, row, a6, 0);
    write_y(Y1 + 7 * m, row, a7, 0);
    write_y(Y1 + 8 * m, row, a8, 0);
    write_y(Y1 + 9 * m, row, a9, 0);
    write_y(Y1 + 10 * m, row, aa, 0);
    write_y(Y1 + 11 * m, row, ab, 0);
    if (Y2) {
        write_y(Y2, row, b0, 0);
        write_y(Y2 + m, row, b1, 0);
        write_y(Y2 + 2 * m, row, b2, 0);
        write_y(Y2 + 3 * m, row, b3, 0);
        write_y(Y2 + 4 * m, row, b4, 0);
        write_y(Y2 + 5 * m, row, b5, 0);
        write_y(Y2 + 6 * m, row, b6, 0);
        write_y(Y2 + 7 * m, row, b7, 0);
        write_y(Y2 + 8 * m, row, b8, 0);
        write_y(Y2 + 9 * m, row, b9, 0);
        write_y(Y2 + 10 * m, row, ba, 0);
        write_y(Y2 + 11 * m, row, bb, 0);
    }
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q4k_q8_t12_1row_pipe(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n,
                                 int add) {
    if (!W || !xq || !xsc || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * kQ4KSoaBsz;
    gemv_q4k_soa_q8_t12_1row_pipe_k<<<pb, th, smem>>>(W, xq, xsc, Y, m, n, add);
}

void launch_q4k_q8_t12_1row_dual_pipe(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                                      float* Y1, float* Y2, int m, int n) {
    if (!W1 || !W2 || !xq || !xsc || !Y1 || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * 2 * kQ4KSoaBsz;
    gemv_q4k_soa_q8_t12_1row_dual_pipe_k<<<pb, th, smem>>>(W1, W2, xq, xsc, Y1, Y2, m, n);
}

} // namespace rapidllm::cuda_gemv
