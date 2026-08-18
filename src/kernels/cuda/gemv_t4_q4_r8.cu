// Isolated T=4 Q4 8-row/warp. Same 4-lane group dp4a as gemv_t4_pipe acc_q4k_smem_4x.
#include "rapidllm/kernels/gemv_t4_q4_r8.h"
#include "rapidllm/kernels/gemv_t4_pipe.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kQ4KSoaBsz = 160;
constexpr int kRowsWarp = 8;

__device__ __forceinline__ float g4_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 1);
    v += __shfl_xor_sync(0xffffffff, v, 2);
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

__device__ __forceinline__ void acc_q4k_g4(const uint8_t* blk, const int8_t* xq0, const __half* sc0,
                                           const int8_t* xq1, const __half* sc1, const int8_t* xq2,
                                           const __half* sc2, const int8_t* xq3, const __half* sc3, int b, int g,
                                           int sub, float& a0, float& a1, float& a2, float& a3) {
    const int gpair = g >> 1;
    const int hi = g & 1;
    const __half* ds = reinterpret_cast<const __half*>(blk);
    const __half* dm = reinterpret_cast<const __half*>(blk + 16);
    const uint8_t* qs = blk + 32 + gpair * 32 + sub * 8;
    const uint2 q8 = *reinterpret_cast<const uint2*>(qs);
    const int q0 = static_cast<int>(hi ? ((q8.x >> 4) & 0x0f0f0f0f) : (q8.x & 0x0f0f0f0f));
    const int q1 = static_cast<int>(hi ? ((q8.y >> 4) & 0x0f0f0f0f) : (q8.y & 0x0f0f0f0f));
    const float dsg = __half2float(ds[g]);
    const float dmg = __half2float(dm[g]);
    const int off = b * 256 + g * 32 + sub * 8;
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
    const float xs0 = __half2float(sc0[b * 8 + g]);
    const float xs1 = __half2float(sc1[b * 8 + g]);
    const float xs2 = __half2float(sc2[b * 8 + g]);
    const float xs3 = __half2float(sc3[b * 8 + g]);
    a0 = fmaf(dsg * xs0, static_cast<float>(s0), a0);
    a0 = fmaf(-dmg * xs0, static_cast<float>(sx0), a0);
    a1 = fmaf(dsg * xs1, static_cast<float>(s1), a1);
    a1 = fmaf(-dmg * xs1, static_cast<float>(sx1), a1);
    a2 = fmaf(dsg * xs2, static_cast<float>(s2), a2);
    a2 = fmaf(-dmg * xs2, static_cast<float>(sx2), a2);
    a3 = fmaf(dsg * xs3, static_cast<float>(s3), a3);
    a3 = fmaf(-dmg * xs3, static_cast<float>(sx3), a3);
}

__global__ void __launch_bounds__(256, 2) gemv_q4k_soa_q8_t4_r8_k(const uint8_t* W, const int8_t* xq,
                                                                  const __half* xsc, float* Y, int m, int n,
                                                                  int add) {
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int rw = lane / 4;
    const int sub = lane & 3;
    const int row0 = (blockIdx.x * warps + warp) * kRowsWarp;
    const int row = row0 + rw;
    const int nb = n / 256;
    const int nsc = n >> 5;
    extern __shared__ uint8_t smem[];
    uint8_t* wsm = smem + warp * (2 * kRowsWarp * kQ4KSoaBsz);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    auto load_stage = [&](int st, int b) {
        uint8_t* dst = wsm + st * kRowsWarp * kQ4KSoaBsz;
        // 8 rows × 10×16B = 80 chunks, 32 lanes.
        for (int i = lane; i < kRowsWarp * 10; i += 32) {
            const int r = i / 10;
            const int c = i - r * 10;
            const int gr = row0 + r;
            const uint8_t* src =
                (gr < m && nb > 0) ? (W + static_cast<size_t>(gr) * static_cast<size_t>(nb) * kQ4KSoaBsz +
                                      static_cast<size_t>(b) * kQ4KSoaBsz)
                                   : nullptr;
            if (src) cp_async16(dst + r * kQ4KSoaBsz + c * 16, src + c * 16);
        }
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
        if (row < m) {
            const uint8_t* blk = wsm + stage * kRowsWarp * kQ4KSoaBsz + rw * kQ4KSoaBsz;
#pragma unroll
            for (int g = 0; g < 8; ++g)
                acc_q4k_g4(blk, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n, xsc + 3 * nsc, b,
                           g, sub, a0, a1, a2, a3);
        }
        stage ^= 1;
    }
    if (row >= m) return;
    a0 = g4_sum(a0);
    a1 = g4_sum(a1);
    a2 = g4_sum(a2);
    a3 = g4_sum(a3);
    if (sub != 0) return;
    write_y(Y, row, a0, add);
    write_y(Y + m, row, a1, add);
    write_y(Y + 2 * m, row, a2, add);
    write_y(Y + 3 * m, row, a3, add);
}

__global__ void __launch_bounds__(256, 2) gemv_q4k_soa_q8_t4_r8_dual_k(const uint8_t* W1, const uint8_t* W2,
                                                                       const int8_t* xq, const __half* xsc, float* Y1,
                                                                       float* Y2, int m, int n) {
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int rw = lane / 4;
    const int sub = lane & 3;
    const int row0 = (blockIdx.x * warps + warp) * kRowsWarp;
    const int row = row0 + rw;
    const int nb = n / 256;
    const int nsc = n >> 5;
    extern __shared__ uint8_t smem_d[];
    uint8_t* wsm = smem_d + warp * (2 * 2 * kRowsWarp * kQ4KSoaBsz);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
    auto load_stage = [&](int st, int b) {
        uint8_t* dst = wsm + st * (2 * kRowsWarp * kQ4KSoaBsz);
        for (int i = lane; i < kRowsWarp * 10; i += 32) {
            const int r = i / 10;
            const int c = i - r * 10;
            const int gr = row0 + r;
            const uint8_t* s1 =
                (gr < m && nb > 0) ? (W1 + static_cast<size_t>(gr) * static_cast<size_t>(nb) * kQ4KSoaBsz +
                                      static_cast<size_t>(b) * kQ4KSoaBsz)
                                   : nullptr;
            const uint8_t* s2 =
                (gr < m && nb > 0) ? (W2 + static_cast<size_t>(gr) * static_cast<size_t>(nb) * kQ4KSoaBsz +
                                      static_cast<size_t>(b) * kQ4KSoaBsz)
                                   : nullptr;
            if (s1) cp_async16(dst + r * kQ4KSoaBsz + c * 16, s1 + c * 16);
            if (s2) cp_async16(dst + kRowsWarp * kQ4KSoaBsz + r * kQ4KSoaBsz + c * 16, s2 + c * 16);
        }
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
        if (row < m) {
            const uint8_t* s = wsm + stage * (2 * kRowsWarp * kQ4KSoaBsz);
            const uint8_t* blk1 = s + rw * kQ4KSoaBsz;
            const uint8_t* blk2 = s + kRowsWarp * kQ4KSoaBsz + rw * kQ4KSoaBsz;
#pragma unroll
            for (int g = 0; g < 8; ++g) {
                acc_q4k_g4(blk1, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n, xsc + 3 * nsc, b,
                           g, sub, a0, a1, a2, a3);
                acc_q4k_g4(blk2, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc, xq + 3 * n, xsc + 3 * nsc, b,
                           g, sub, b0, b1, b2, b3);
            }
        }
        stage ^= 1;
    }
    if (row >= m) return;
    a0 = g4_sum(a0);
    a1 = g4_sum(a1);
    a2 = g4_sum(a2);
    a3 = g4_sum(a3);
    b0 = g4_sum(b0);
    b1 = g4_sum(b1);
    b2 = g4_sum(b2);
    b3 = g4_sum(b3);
    if (sub != 0) return;
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

void launch_q4k_q8_t4_r8(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n, int add) {
    if (!W || !xq || !xsc || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    if (m < 256) {
        launch_q4k_q8_t4_1row_pipe(W, xq, xsc, Y, m, n, add);
        return;
    }
    const int th = 256;
    const int tw = th / 32;
    const int rows_per_block = tw * kRowsWarp;
    const int pb = (m + rows_per_block - 1) / rows_per_block;
    const size_t smem = static_cast<size_t>(tw) * 2 * kRowsWarp * kQ4KSoaBsz;
    gemv_q4k_soa_q8_t4_r8_k<<<pb, th, smem>>>(W, xq, xsc, Y, m, n, add);
}

void launch_q4k_q8_t4_r8_dual(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc, float* Y1,
                              float* Y2, int m, int n) {
    if (!W1 || !W2 || !xq || !xsc || !Y1 || m <= 0 || n <= 0 || (n % 256) != 0) return;
    if (m < 256) {
        launch_q4k_q8_t4_1row_dual_pipe(W1, W2, xq, xsc, Y1, Y2, m, n);
        return;
    }
    const int th = 256;
    const int tw = th / 32;
    const int rows_per_block = tw * kRowsWarp;
    const int pb = (m + rows_per_block - 1) / rows_per_block;
    const size_t smem = static_cast<size_t>(tw) * 2 * 2 * kRowsWarp * kQ4KSoaBsz;
    gemv_q4k_soa_q8_t4_r8_dual_k<<<pb, th, smem>>>(W1, W2, xq, xsc, Y1, Y2, m, n);
}

} // namespace rapidllm::cuda_gemv
