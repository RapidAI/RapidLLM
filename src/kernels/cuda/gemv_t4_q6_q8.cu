// Isolated T=4 Q6 × Q8 1-row. dp4a on packed q-32, same group map as
// acc_q6k_soa_q8. 1-row so 2-row float associativity cannot drift tokens.
#include "rapidllm/kernels/gemv_t4_q6_q8.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kQ6KSoaBsz = 224;

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

__device__ __forceinline__ void q6k_pack8(const uint8_t* ql, const uint8_t* qh, int qkind, int idx, int* q0,
                                          int* q1) {
    int8_t v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int qi = idx + i;
        int t;
        if (qkind == 0)
            t = (ql[qi] & 0xF) | ((qh[qi] & 3) << 4);
        else if (qkind == 1)
            t = (ql[32 + qi] & 0xF) | (((qh[qi] >> 2) & 3) << 4);
        else if (qkind == 2)
            t = (ql[qi] >> 4) | (((qh[qi] >> 4) & 3) << 4);
        else
            t = (ql[32 + qi] >> 4) | (((qh[qi] >> 6) & 3) << 4);
        v[i] = static_cast<int8_t>(t - 32);
    }
    *q0 = *reinterpret_cast<int*>(v);
    *q1 = *reinterpret_cast<int*>(v + 4);
}

__device__ __forceinline__ void acc_q6k_smem_q8_4x(const uint8_t* blk, const int8_t* xq0, const __half* sc0,
                                                   const int8_t* xq1, const __half* sc1, const int8_t* xq2,
                                                   const __half* sc2, const int8_t* xq3, const __half* sc3, int b,
                                                   int lane, float& a0, float& a1, float& a2, float& a3) {
    const int group = lane >> 1;
    const int sub = lane & 1;
    const int n128 = group >> 3;
    const int g8 = group & 7;
    const int qkind = g8 >> 1;
    const int idx = (g8 & 1) * 16 + sub * 8;
    const int xoff = group * 16 + sub * 8;
    const __half* ds = reinterpret_cast<const __half*>(blk);
    const uint8_t* ql = blk + 32 + n128 * 64;
    const uint8_t* qh = blk + 160 + n128 * 32;
    int q0, q1;
    q6k_pack8(ql, qh, qkind, idx, &q0, &q1);
    const float dsg = __half2float(ds[group]);
    const int off = b * 256 + xoff;
    const int u0 = *reinterpret_cast<const int*>(xq0 + off);
    const int u1 = *reinterpret_cast<const int*>(xq0 + off + 4);
    const int v0 = *reinterpret_cast<const int*>(xq1 + off);
    const int v1 = *reinterpret_cast<const int*>(xq1 + off + 4);
    const int w0 = *reinterpret_cast<const int*>(xq2 + off);
    const int w1 = *reinterpret_cast<const int*>(xq2 + off + 4);
    const int z0 = *reinterpret_cast<const int*>(xq3 + off);
    const int z1 = *reinterpret_cast<const int*>(xq3 + off + 4);
    int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#if __CUDA_ARCH__ >= 610
    s0 = __dp4a(q0, u0, 0);
    s0 = __dp4a(q1, u1, s0);
    s1 = __dp4a(q0, v0, 0);
    s1 = __dp4a(q1, v1, s1);
    s2 = __dp4a(q0, w0, 0);
    s2 = __dp4a(q1, w1, s2);
    s3 = __dp4a(q0, z0, 0);
    s3 = __dp4a(q1, z1, s3);
#else
    const int8_t* qq0 = reinterpret_cast<const int8_t*>(&q0);
    const int8_t* qq1 = reinterpret_cast<const int8_t*>(&q1);
    const int8_t* uu0 = reinterpret_cast<const int8_t*>(&u0);
    const int8_t* uu1 = reinterpret_cast<const int8_t*>(&u1);
    const int8_t* vv0 = reinterpret_cast<const int8_t*>(&v0);
    const int8_t* vv1 = reinterpret_cast<const int8_t*>(&v1);
    const int8_t* ww0 = reinterpret_cast<const int8_t*>(&w0);
    const int8_t* ww1 = reinterpret_cast<const int8_t*>(&w1);
    const int8_t* zz0 = reinterpret_cast<const int8_t*>(&z0);
    const int8_t* zz1 = reinterpret_cast<const int8_t*>(&z1);
    for (int k = 0; k < 4; ++k) {
        s0 += qq0[k] * uu0[k] + qq1[k] * uu1[k];
        s1 += qq0[k] * vv0[k] + qq1[k] * vv1[k];
        s2 += qq0[k] * ww0[k] + qq1[k] * ww1[k];
        s3 += qq0[k] * zz0[k] + qq1[k] * zz1[k];
    }
#endif
    const int sc = b * 8 + (xoff >> 5);
    const float xs0 = __half2float(sc0[sc]);
    const float xs1 = __half2float(sc1[sc]);
    const float xs2 = __half2float(sc2[sc]);
    const float xs3 = __half2float(sc3[sc]);
    a0 = fmaf(dsg * xs0, static_cast<float>(s0), a0);
    a1 = fmaf(dsg * xs1, static_cast<float>(s1), a1);
    a2 = fmaf(dsg * xs2, static_cast<float>(s2), a2);
    a3 = fmaf(dsg * xs3, static_cast<float>(s3), a3);
}

__global__ void __launch_bounds__(256, 3) gemv_q6k_soa_q8_t4_1row_k(const uint8_t* W, const int8_t* xq,
                                                                    const __half* xsc, float* Y, int m, int n,
                                                                    int add) {
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    const int warp = threadIdx.x / 32;
    extern __shared__ uint8_t smem_w[];
    uint8_t* wsm = smem_w + warp * (2 * kQ6KSoaBsz);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    const uint8_t* r0 =
        (row < m && nb > 0) ? (W + static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ6KSoaBsz) : nullptr;
    auto load_stage = [&](int st, int b) {
        cp_async_q6_blk(wsm + st * kQ6KSoaBsz, r0 ? r0 + static_cast<size_t>(b) * kQ6KSoaBsz : nullptr, lane);
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
        if (r0)
            acc_q6k_smem_q8_4x(wsm + stage * kQ6KSoaBsz, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc,
                               xq + 3 * n, xsc + 3 * nsc, b, lane, a0, a1, a2, a3);
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

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q6k_q8_t4_1row(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n,
                           int add) {
    if (!W || !xq || !xsc || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * kQ6KSoaBsz;
    gemv_q6k_soa_q8_t4_1row_k<<<pb, th, smem>>>(W, xq, xsc, Y, m, n, add);
}

} // namespace rapidllm::cuda_gemv
