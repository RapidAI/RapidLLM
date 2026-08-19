// Isolated T=12 Q4 2-row GEMV. cp.async W pipeline; two output rows share the
// same xq tile (L1). Dequant matches gemv_t4_pipe acc_q4k_smem_4x.
#include "rapidllm/kernels/gemv_q4_t12_2row_pipe.h"

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

__device__ __forceinline__ void acc12(const uint8_t* blk, const int8_t* xq, const __half* xsc, int n, int nsc,
                                      int b, int lane, float& a0, float& a1, float& a2, float& a3, float& a4,
                                      float& a5, float& a6, float& a7, float& a8, float& a9, float& aa,
                                      float& ab) {
    acc_q4k_smem_4x(blk, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n, xsc + 3 * nsc, b,
                    lane, a0, a1, a2, a3);
    acc_q4k_smem_4x(blk, xq + 4 * n, xsc + 4 * nsc, xq + 5 * n, xsc + 5 * nsc, xq + 6 * n, xsc + 6 * nsc,
                    xq + 7 * n, xsc + 7 * nsc, b, lane, a4, a5, a6, a7);
    acc_q4k_smem_4x(blk, xq + 8 * n, xsc + 8 * nsc, xq + 9 * n, xsc + 9 * nsc, xq + 10 * n, xsc + 10 * nsc,
                    xq + 11 * n, xsc + 11 * nsc, b, lane, a8, a9, aa, ab);
}

__device__ __forceinline__ void emit12(float* Y, int m, int row, int add, int lane, float a0, float a1, float a2,
                                       float a3, float a4, float a5, float a6, float a7, float a8, float a9,
                                       float aa, float ab) {
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

__global__ void __launch_bounds__(256, 2) gemv_q4k_soa_q8_t12_2row_pipe_k(const uint8_t* W, const int8_t* xq,
                                                                         const __half* xsc, float* Y, int m, int n,
                                                                         int add) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    const int warp = threadIdx.x / 32;
    extern __shared__ uint8_t smem_w[];
    uint8_t* wsm = smem_w + warp * (2 * 2 * kQ4KSoaBsz);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f;
    float a6 = 0.f, a7 = 0.f, a8 = 0.f, a9 = 0.f, aa = 0.f, ab = 0.f;
    float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f, b4 = 0.f, b5 = 0.f;
    float b6 = 0.f, b7 = 0.f, b8 = 0.f, b9 = 0.f, ba = 0.f, bb = 0.f;
    const uint8_t* r0 =
        (row0 < m && nb > 0) ? (W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz) : nullptr;
    const uint8_t* r1 =
        (row1 < m && nb > 0) ? (W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz) : nullptr;
    auto load_stage = [&](int st, int b) {
        uint8_t* s = wsm + st * (2 * kQ4KSoaBsz);
        cp_async_q4_blk(s, r0 ? r0 + static_cast<size_t>(b) * kQ4KSoaBsz : nullptr, lane);
        cp_async_q4_blk(s + kQ4KSoaBsz, r1 ? r1 + static_cast<size_t>(b) * kQ4KSoaBsz : nullptr, lane);
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
        if (r0) acc12(s, xq, xsc, n, nsc, b, lane, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab);
        if (r1) acc12(s + kQ4KSoaBsz, xq, xsc, n, nsc, b, lane, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb);
        stage ^= 1;
    }
    if (row0 < m)
        emit12(Y, m, row0, add, lane, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab);
    if (row1 < m)
        emit12(Y, m, row1, add, lane, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb);
}

__global__ void __launch_bounds__(256, 2) gemv_q4k_soa_q8_t12_2row_dual_pipe_k(const uint8_t* W1, const uint8_t* W2,
                                                                              const int8_t* xq, const __half* xsc,
                                                                              float* Y1, float* Y2, int m, int n) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    const int warp = threadIdx.x / 32;
    extern __shared__ uint8_t smem_d[];
    uint8_t* wsm = smem_d + warp * (2 * 2 * kQ4KSoaBsz);
    auto run_w = [&](const uint8_t* W, float* Y) {
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f;
        float a6 = 0.f, a7 = 0.f, a8 = 0.f, a9 = 0.f, aa = 0.f, ab = 0.f;
        float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f, b4 = 0.f, b5 = 0.f;
        float b6 = 0.f, b7 = 0.f, b8 = 0.f, b9 = 0.f, ba = 0.f, bb = 0.f;
        const uint8_t* r0 =
            (row0 < m && nb > 0 && W) ? (W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz)
                                      : nullptr;
        const uint8_t* r1 =
            (row1 < m && nb > 0 && W) ? (W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz)
                                      : nullptr;
        auto load_stage = [&](int st, int b) {
            uint8_t* s = wsm + st * (2 * kQ4KSoaBsz);
            cp_async_q4_blk(s, r0 ? r0 + static_cast<size_t>(b) * kQ4KSoaBsz : nullptr, lane);
            cp_async_q4_blk(s + kQ4KSoaBsz, r1 ? r1 + static_cast<size_t>(b) * kQ4KSoaBsz : nullptr, lane);
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
            if (r0) acc12(s, xq, xsc, n, nsc, b, lane, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab);
            if (r1) acc12(s + kQ4KSoaBsz, xq, xsc, n, nsc, b, lane, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb);
            stage ^= 1;
        }
        if (Y && row0 < m)
            emit12(Y, m, row0, 0, lane, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab);
        if (Y && row1 < m)
            emit12(Y, m, row1, 0, lane, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb);
    };
    run_w(W1, Y1);
    run_w(W2, Y2);
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q4k_q8_t12_2row_pipe(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n,
                                 int add) {
    if (!W || !xq || !xsc || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int pairs = (m + 1) / 2;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (pairs + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * 2 * kQ4KSoaBsz;
    gemv_q4k_soa_q8_t12_2row_pipe_k<<<pb, th, smem>>>(W, xq, xsc, Y, m, n, add);
}

void launch_q4k_q8_t12_2row_dual_pipe(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                                      float* Y1, float* Y2, int m, int n) {
    if (!W1 || !W2 || !xq || !xsc || !Y1 || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int pairs = (m + 1) / 2;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (pairs + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * 2 * kQ4KSoaBsz;
    gemv_q4k_soa_q8_t12_2row_dual_pipe_k<<<pb, th, smem>>>(W1, W2, xq, xsc, Y1, Y2, m, n);
}

} // namespace rapidllm::cuda_gemv
