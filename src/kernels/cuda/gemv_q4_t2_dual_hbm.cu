// Isolated T=2 Q4 dual HBM. Do not edit gemv_q6_t2.cu / gemv_q6_t2_1row.cu /
// gemv_q6_t2_hbm.cu / gemv_q6_t4.h. Same 2-row dequant as gemv_q4k_soa_q8_t2_dual_k.
#include "rapidllm/kernels/gemv_q4_t2_dual_hbm.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kQ4KSoaBsz = 160;
constexpr int kStages = 2;

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

// Same Q4 SOA integer dequant as gemv_q6_t2.cu acc_q4k_soa_q8_2x, one K-tile.
__device__ __forceinline__ void acc_q4k_soa_q8_2x_blk(const uint8_t* blk, const int8_t* xq0, const __half* sc0,
                                                      const int8_t* xq1, const __half* sc1, int lane, float& a0,
                                                      float& a1) {
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
    const int u0 = *reinterpret_cast<const int*>(xq0);
    const int u1 = *reinterpret_cast<const int*>(xq0 + 4);
    const int v0 = *reinterpret_cast<const int*>(xq1);
    const int v1 = *reinterpret_cast<const int*>(xq1 + 4);
    int s0 = __dp4a(q0, u0, 0);
    s0 = __dp4a(q1, u1, s0);
    int sx0 = __dp4a(0x01010101, u0, 0);
    sx0 = __dp4a(0x01010101, u1, sx0);
    int s1 = __dp4a(q0, v0, 0);
    s1 = __dp4a(q1, v1, s1);
    int sx1 = __dp4a(0x01010101, v0, 0);
    sx1 = __dp4a(0x01010101, v1, sx1);
    const float xs0 = __half2float(sc0[group]);
    const float xs1 = __half2float(sc1[group]);
    a0 = fmaf(dsg * xs0, static_cast<float>(s0), a0);
    a0 = fmaf(-dmg * xs0, static_cast<float>(sx0), a0);
    a1 = fmaf(dsg * xs1, static_cast<float>(s1), a1);
    a1 = fmaf(-dmg * xs1, static_cast<float>(sx1), a1);
}

__global__ void __launch_bounds__(256, 3) gemv_q4k_soa_q8_t2_dual_hbm_k(const uint8_t* W1, const uint8_t* W2,
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
    // 2 stages * 4 blocks (r0,s0,r1,s1) * 160
    extern __shared__ uint8_t smem_d[];
    uint8_t* wsm = smem_d + warp * (kStages * 4 * kQ4KSoaBsz);
    const int8_t* xq0 = xq;
    const int8_t* xq1 = xq + n;
    const __half* sc0 = xsc;
    const __half* sc1 = xsc + nsc;
    float a00 = 0.f, a01 = 0.f, b00 = 0.f, b01 = 0.f;
    float a10 = 0.f, a11 = 0.f, b10 = 0.f, b11 = 0.f;
    const uint8_t* r0 =
        (row0 < m && nb > 0) ? (W1 + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz) : nullptr;
    const uint8_t* s0 =
        (row0 < m && nb > 0) ? (W2 + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz) : nullptr;
    const uint8_t* r1 =
        (row1 < m && nb > 0) ? (W1 + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz) : nullptr;
    const uint8_t* s1 =
        (row1 < m && nb > 0) ? (W2 + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz) : nullptr;
    auto load_stage = [&](int st, int b) {
        uint8_t* base = wsm + st * 4 * kQ4KSoaBsz;
        const size_t off = static_cast<size_t>(b) * kQ4KSoaBsz;
        cp_async_q4_blk(base + 0 * kQ4KSoaBsz, r0 ? r0 + off : nullptr, lane);
        cp_async_q4_blk(base + 1 * kQ4KSoaBsz, s0 ? s0 + off : nullptr, lane);
        cp_async_q4_blk(base + 2 * kQ4KSoaBsz, r1 ? r1 + off : nullptr, lane);
        cp_async_q4_blk(base + 3 * kQ4KSoaBsz, s1 ? s1 + off : nullptr, lane);
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
        const uint8_t* base = wsm + stage * 4 * kQ4KSoaBsz;
        const int xoff = b * 256 + ((lane >> 2) * 32 + (lane & 3) * 8);
        const int8_t* u0 = xq0 + xoff;
        const int8_t* u1 = xq1 + xoff;
        const __half* c0 = sc0 + b * 8;
        const __half* c1 = sc1 + b * 8;
        if (r0) acc_q4k_soa_q8_2x_blk(base + 0 * kQ4KSoaBsz, u0, c0, u1, c1, lane, a00, a01);
        if (s0) acc_q4k_soa_q8_2x_blk(base + 1 * kQ4KSoaBsz, u0, c0, u1, c1, lane, b00, b01);
        if (r1) acc_q4k_soa_q8_2x_blk(base + 2 * kQ4KSoaBsz, u0, c0, u1, c1, lane, a10, a11);
        if (s1) acc_q4k_soa_q8_2x_blk(base + 3 * kQ4KSoaBsz, u0, c0, u1, c1, lane, b10, b11);
        stage ^= 1;
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

void launch_q4k_q8_t2_dual_hbm(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc, float* Y1,
                               float* Y2, int m, int n) {
    if (!W1 || !W2 || !xq || !xsc || !Y1 || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int pairs = (m + 1) / 2;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (pairs + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * kStages * 4 * kQ4KSoaBsz;
    gemv_q4k_soa_q8_t2_dual_hbm_k<<<pb, th, smem>>>(W1, W2, xq, xsc, Y1, Y2, m, n);
}

} // namespace rapidllm::cuda_gemv
