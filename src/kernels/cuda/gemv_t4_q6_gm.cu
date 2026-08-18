// Isolated T=4 Q6 group-major GEMV. W is 288 B/superblock: int8 q[256] in
// weight order 0..255 (q-32) then half scales[16]. Same 16-weight groups as
// SOA ds[0..15]. 2 lanes/group × 8 int8 × __dp4a against ensure_xq Q8.
#include "rapidllm/kernels/gemv_t4_q6_gm.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kQ6KGmBsz = 288;

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

__device__ __forceinline__ void cp_async_gm_blk(uint8_t* dst, const uint8_t* src, int lane) {
    // 288 B = 18 × 16. Lanes 0..17 issue; 18..31 idle on the copy.
    if (src && lane < 18) cp_async16(dst + lane * 16, src + lane * 16);
}

__device__ __forceinline__ void acc_q6k_gm_q8_4x(const uint8_t* blk, const int8_t* xq0, const __half* sc0,
                                                 const int8_t* xq1, const __half* sc1, const int8_t* xq2,
                                                 const __half* sc2, const int8_t* xq3, const __half* sc3, int b,
                                                 int lane, float& a0, float& a1, float& a2, float& a3) {
    const int group = lane >> 1;
    const int sub = lane & 1;
    const int8_t* q8 = reinterpret_cast<const int8_t*>(blk) + group * 16 + sub * 8;
    const int q0 = *reinterpret_cast<const int*>(q8);
    const int q1 = *reinterpret_cast<const int*>(q8 + 4);
    const __half* ds = reinterpret_cast<const __half*>(blk + 256);
    const float dsg = __half2float(ds[group]);
    const int off = b * 256 + group * 16 + sub * 8;
    const int sc = b * 8 + (group >> 1);
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
    const float xs0 = __half2float(sc0[sc]);
    const float xs1 = __half2float(sc1[sc]);
    const float xs2 = __half2float(sc2[sc]);
    const float xs3 = __half2float(sc3[sc]);
    a0 = fmaf(dsg * xs0, static_cast<float>(s0), a0);
    a1 = fmaf(dsg * xs1, static_cast<float>(s1), a1);
    a2 = fmaf(dsg * xs2, static_cast<float>(s2), a2);
    a3 = fmaf(dsg * xs3, static_cast<float>(s3), a3);
}

__global__ void __launch_bounds__(256, 4) gemv_q6k_gm_q8_t4_k(const int8_t* Wgm, const int8_t* xq,
                                                              const __half* xsc, float* Y, int m, int n,
                                                              int add) {
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    const int warp = threadIdx.x / 32;
    extern __shared__ uint8_t smem_w[];
    uint8_t* wsm = smem_w + warp * (2 * kQ6KGmBsz);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    const uint8_t* r0 =
        (row < m && nb > 0)
            ? (reinterpret_cast<const uint8_t*>(Wgm) +
               static_cast<size_t>(row) * static_cast<size_t>(nb) * kQ6KGmBsz)
            : nullptr;
    auto load_stage = [&](int st, int b) {
        cp_async_gm_blk(wsm + st * kQ6KGmBsz,
                        r0 ? r0 + static_cast<size_t>(b) * kQ6KGmBsz : nullptr, lane);
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
            acc_q6k_gm_q8_4x(wsm + stage * kQ6KGmBsz, xq, xsc, xq + n, xsc + nsc, xq + 2 * n, xsc + 2 * nsc,
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

void launch_q6k_gm_q8_t4(const int8_t* Wgm, const int8_t* xq, const __half* xsc, float* Y, int m, int n,
                         int add) {
    if (!Wgm || !xq || !xsc || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw - 1) / tw;
    const size_t smem = static_cast<size_t>(tw) * 2 * kQ6KGmBsz;
    gemv_q6k_gm_q8_t4_k<<<pb, th, smem>>>(Wgm, xq, xsc, Y, m, n, add);
}

} // namespace rapidllm::cuda_gemv
