// Isolated T=4 Q4 2-row dual. Same dequant as gemv_t4_pipe.cu.
#include "rapidllm/kernels/gemv_t4_q4_2row.h"

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

// Same as gemv_t4_pipe.cu acc_q4k_smem_4x.
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

__global__ void __launch_bounds__(256, 2) gemv_q4k_soa_q8_t4_2row_dual_k(const uint8_t* W1, const uint8_t* W2,
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
    // 2 stages * 2 rows * 2 W * 160
    uint8_t* wsm = smem_d + warp * (2 * 2 * 2 * kQ4KSoaBsz);
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a03 = 0.f;
    float a10 = 0.f, a11 = 0.f, a12 = 0.f, a13 = 0.f;
    float b00 = 0.f, b01 = 0.f, b02 = 0.f, b03 = 0.f;
    float b10 = 0.f, b11 = 0.f, b12 = 0.f, b13 = 0.f;
    const uint8_t* r10 =
        (row0 < m && nb > 0) ? (W1 + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz) : nullptr;
    const uint8_t* r11 =
        (row1 < m && nb > 0) ? (W1 + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz) : nullptr;
    const uint8_t* r20 =
        (row0 < m && nb > 0) ? (W2 + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz) : nullptr;
    const uint8_t* r21 =
        (row1 < m && nb > 0) ? (W2 + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz) : nullptr;
    auto load_stage = [&](int st, int b) {
        uint8_t* s = wsm + st * (4 * kQ4KSoaBsz);
        cp_async_q4_blk(s + 0 * kQ4KSoaBsz, r10 ? r10 + static_cast<size_t>(b) * kQ4KSoaBsz : nullptr, lane);
        cp_async_q4_blk(s + 1 * kQ4KSoaBsz, r20 ? r20 + static_cast<size_t>(b) * kQ4KSoaBsz : nullptr, lane);
        cp_async_q4_blk(s + 2 * kQ4KSoaBsz, r11 ? r11 + static_cast<size_t>(b) * kQ4KSoaBsz : nullptr, lane);
        cp_async_q4_blk(s + 3 * kQ4KSoaBsz, r21 ? r21 + static_cast<size_t>(b) * kQ4KSoaBsz : nullptr, lane);
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
        const uint8_t* s = wsm + stage * (4 * kQ4KSoaBsz);
        if (r10)
            acc_q4k_smem_4x(s, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n, xsc + 3 * nsc, b,
                            lane, a00, a01, a02, a03);
        if (r20)
            acc_q4k_smem_4x(s + kQ4KSoaBsz, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n,
                            xsc + 3 * nsc, b, lane, b00, b01, b02, b03);
        if (r11)
            acc_q4k_smem_4x(s + 2 * kQ4KSoaBsz, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n,
                            xsc + 3 * nsc, b, lane, a10, a11, a12, a13);
        if (r21)
            acc_q4k_smem_4x(s + 3 * kQ4KSoaBsz, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n,
                            xsc + 3 * nsc, b, lane, b10, b11, b12, b13);
        stage ^= 1;
    }
    a00 = warp_sum(a00);
    a01 = warp_sum(a01);
    a02 = warp_sum(a02);
    a03 = warp_sum(a03);
    b00 = warp_sum(b00);
    b01 = warp_sum(b01);
    b02 = warp_sum(b02);
    b03 = warp_sum(b03);
    a10 = warp_sum(a10);
    a11 = warp_sum(a11);
    a12 = warp_sum(a12);
    a13 = warp_sum(a13);
    b10 = warp_sum(b10);
    b11 = warp_sum(b11);
    b12 = warp_sum(b12);
    b13 = warp_sum(b13);
    if (lane != 0) return;
    if (row0 < m) {
        write_y(Y1, row0, a00, 0);
        write_y(Y1 + m, row0, a01, 0);
        write_y(Y1 + 2 * m, row0, a02, 0);
        write_y(Y1 + 3 * m, row0, a03, 0);
        if (Y2) {
            write_y(Y2, row0, b00, 0);
            write_y(Y2 + m, row0, b01, 0);
            write_y(Y2 + 2 * m, row0, b02, 0);
            write_y(Y2 + 3 * m, row0, b03, 0);
        }
    }
    if (row1 < m) {
        write_y(Y1, row1, a10, 0);
        write_y(Y1 + m, row1, a11, 0);
        write_y(Y1 + 2 * m, row1, a12, 0);
        write_y(Y1 + 3 * m, row1, a13, 0);
        if (Y2) {
            write_y(Y2, row1, b10, 0);
            write_y(Y2 + m, row1, b11, 0);
            write_y(Y2 + 2 * m, row1, b12, 0);
            write_y(Y2 + 3 * m, row1, b13, 0);
        }
    }
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q4k_q8_t4_2row_dual(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                                float* Y1, float* Y2, int m, int n) {
    if (!W1 || !W2 || !xq || !xsc || !Y1 || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pairs = (m + 1) / 2;
    const int pb = (pairs + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * 4 * kQ4KSoaBsz;
    gemv_q4k_soa_q8_t4_2row_dual_k<<<pb, th, smem>>>(W1, W2, xq, xsc, Y1, Y2, m, n);
}

} // namespace rapidllm::cuda_gemv
