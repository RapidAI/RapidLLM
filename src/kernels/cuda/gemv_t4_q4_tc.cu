// Isolated T=4 Q4 16-row tensor-core GEMV. Dequant matches gemv_t4_pipe
// acc_q4k_smem_4x (nibble q, ds/dm, Q8 x). One m16n8k32.s8 per 32-wide group.
#include "rapidllm/kernels/gemv_t4_q4_tc.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kQ4KSoaBsz = 160;

__device__ __forceinline__ void write_y(float* y, int row, float acc, int add) {
    if (add) y[row] += acc;
    else y[row] = acc;
}

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void ldmatrix_x4(uint32_t o[4], const void* p) {
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t o[2], const void* p) {
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];"
                 : "=r"(o[0]), "=r"(o[1])
                 : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k32_s8(int32_t d[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void unpack_q4_group(int8_t* dst, const uint8_t* blk, int g) {
    const int gpair = g >> 1;
    const int hi = g & 1;
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        const uint2 q8 = *reinterpret_cast<const uint2*>(blk + 32 + gpair * 32 + sub * 8);
        const int q0 = static_cast<int>(hi ? ((q8.x >> 4) & 0x0f0f0f0f) : (q8.x & 0x0f0f0f0f));
        const int q1 = static_cast<int>(hi ? ((q8.y >> 4) & 0x0f0f0f0f) : (q8.y & 0x0f0f0f0f));
        dst[sub * 8 + 0] = static_cast<int8_t>(q0 & 255);
        dst[sub * 8 + 1] = static_cast<int8_t>((q0 >> 8) & 255);
        dst[sub * 8 + 2] = static_cast<int8_t>((q0 >> 16) & 255);
        dst[sub * 8 + 3] = static_cast<int8_t>((q0 >> 24) & 255);
        dst[sub * 8 + 4] = static_cast<int8_t>(q1 & 255);
        dst[sub * 8 + 5] = static_cast<int8_t>((q1 >> 8) & 255);
        dst[sub * 8 + 6] = static_cast<int8_t>((q1 >> 16) & 255);
        dst[sub * 8 + 7] = static_cast<int8_t>((q1 >> 24) & 255);
    }
}
#endif

__global__ void __launch_bounds__(256, 3) gemv_q4k_soa_q8_t4_tc_k(const uint8_t* W, const int8_t* xq,
                                                                  const __half* xsc, float* Y, int m, int n,
                                                                  int add) {
#if __CUDA_ARCH__ >= 800
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * warps + warp) * 16;
    const int nb = n / 256;
    const int nsc = n >> 5;
    extern __shared__ char raw[];
    // Per-warp A 16x32 int8, shared B 8x32, xsum[4], ds/dm[16].
    uint8_t* As = reinterpret_cast<uint8_t*>(raw);
    uint8_t* Bs = As + warps * 16 * 32;
    int32_t* xsum = reinterpret_cast<int32_t*>(Bs + 8 * 32);
    __half* dsm = reinterpret_cast<__half*>(xsum + 4);
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    uint8_t* Aw = As + warp * 16 * 32;
    for (int b = 0; b < nb; ++b) {
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int k0 = b * 256 + g * 32;
            if (lane < 16) {
                const int row = row0 + lane;
                if (row < m) {
                    const uint8_t* blk =
                        W + (static_cast<size_t>(row) * static_cast<size_t>(nb) + static_cast<size_t>(b)) *
                                kQ4KSoaBsz;
                    unpack_q4_group(reinterpret_cast<int8_t*>(Aw + lane * 32), blk, g);
                    dsm[warp * 32 + lane] = reinterpret_cast<const __half*>(blk)[g];
                    dsm[warp * 32 + 16 + lane] = reinterpret_cast<const __half*>(blk + 16)[g];
                } else {
#pragma unroll
                    for (int j = 0; j < 32; ++j) Aw[lane * 32 + j] = 0;
                    dsm[warp * 32 + lane] = __float2half(0.f);
                    dsm[warp * 32 + 16 + lane] = __float2half(0.f);
                }
            }
            if (warp == 0) {
                if (lane < 32) {
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        const int8_t v = xq[static_cast<size_t>(t) * n + k0 + lane];
                        Bs[t * 32 + lane] = static_cast<uint8_t>(v);
                    }
#pragma unroll
                    for (int t = 4; t < 8; ++t) Bs[t * 32 + lane] = 0;
                }
                if (lane < 4) {
                    int tot = 0;
#pragma unroll
                    for (int j = 0; j < 32; ++j) tot += static_cast<int>(static_cast<int8_t>(Bs[lane * 32 + j]));
                    xsum[lane] = tot;
                }
            }
            __syncthreads();
            uint32_t a[4], bv[2];
            ldmatrix_x4(a, Aw + (lane % 16) * 32 + (lane / 16) * 16);
            ldmatrix_x2(bv, Bs + (lane % 8) * 32 + (lane / 8) * 16);
            int32_t d[4] = {0, 0, 0, 0};
            mma_m16n8k32_s8(d, a, bv);
            const int r = lane / 4;
            const int pair = lane & 3;
            if (pair <= 1) {
                const int col = pair * 2;
                const float ds0 = __half2float(dsm[warp * 32 + r]);
                const float dm0 = __half2float(dsm[warp * 32 + 16 + r]);
                const float ds8 = __half2float(dsm[warp * 32 + r + 8]);
                const float dm8 = __half2float(dsm[warp * 32 + 16 + r + 8]);
                const int gidx = b * 8 + g;
                const float xs0 = __half2float(xsc[col * nsc + gidx]);
                const float xs1 = __half2float(xsc[(col + 1) * nsc + gidx]);
                const float sx0 = static_cast<float>(xsum[col]);
                const float sx1 = static_cast<float>(xsum[col + 1]);
                // pair0 writes acc0/acc1 (cols 0,1); pair1 writes acc2/acc3 (cols 2,3)
                // for the low row in even/odd... each thread only owns 2 cols of 2 rows.
                // Fold into the 4 named accs by column identity.
                if (pair == 0) {
                    acc0 = fmaf(ds0 * xs0, static_cast<float>(d[0]), acc0);
                    acc0 = fmaf(-dm0 * xs0, sx0, acc0);
                    acc1 = fmaf(ds0 * xs1, static_cast<float>(d[1]), acc1);
                    acc1 = fmaf(-dm0 * xs1, sx1, acc1);
                    acc2 = fmaf(ds8 * xs0, static_cast<float>(d[2]), acc2);
                    acc2 = fmaf(-dm8 * xs0, sx0, acc2);
                    acc3 = fmaf(ds8 * xs1, static_cast<float>(d[3]), acc3);
                    acc3 = fmaf(-dm8 * xs1, sx1, acc3);
                } else {
                    // cols 2,3: reuse acc names as (row r col2), (row r col3), (row r+8 col2), ...
                    acc0 = fmaf(ds0 * xs0, static_cast<float>(d[0]), acc0);
                    acc0 = fmaf(-dm0 * xs0, sx0, acc0);
                    acc1 = fmaf(ds0 * xs1, static_cast<float>(d[1]), acc1);
                    acc1 = fmaf(-dm0 * xs1, sx1, acc1);
                    acc2 = fmaf(ds8 * xs0, static_cast<float>(d[2]), acc2);
                    acc2 = fmaf(-dm8 * xs0, sx0, acc2);
                    acc3 = fmaf(ds8 * xs1, static_cast<float>(d[3]), acc3);
                    acc3 = fmaf(-dm8 * xs1, sx1, acc3);
                }
            }
            __syncthreads();
        }
    }
    if (row0 >= m) return;
    const int r = lane / 4;
    const int pair = lane & 3;
    if (pair > 1) return;
    const int col = pair * 2;
    const int rr0 = row0 + r;
    const int rr8 = row0 + r + 8;
    if (rr0 < m) {
        write_y(Y + col * m, rr0, acc0, add);
        write_y(Y + (col + 1) * m, rr0, acc1, add);
    }
    if (rr8 < m) {
        write_y(Y + col * m, rr8, acc2, add);
        write_y(Y + (col + 1) * m, rr8, acc3, add);
    }
#else
    (void)W;
    (void)xq;
    (void)xsc;
    (void)Y;
    (void)m;
    (void)n;
    (void)add;
#endif
}

} // namespace

namespace rapidllm::cuda_gemv {

void launch_q4k_q8_t4_1row_tc(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m, int n,
                              int add) {
    if (!W || !xq || !xsc || !Y || m <= 0 || n <= 0 || (n % 256) != 0) return;
    const int th = 256;
    const int tw = th / 32;
    const int pb = (m + tw * 16 - 1) / (tw * 16);
    // A: warps*16*32, B: 8*32, xsum: 4*4, ds/dm: warps*32*2
    const size_t smem = static_cast<size_t>(tw) * 16 * 32 + 8 * 32 + 16 + static_cast<size_t>(tw) * 32 * 2;
    gemv_q4k_soa_q8_t4_tc_k<<<pb, th, smem>>>(W, xq, xsc, Y, m, n, add);
}

void launch_q4k_q8_t4_1row_dual_tc(const uint8_t* W1, const uint8_t* W2, const int8_t* xq, const __half* xsc,
                                   float* Y1, float* Y2, int m, int n) {
    launch_q4k_q8_t4_1row_tc(W1, xq, xsc, Y1, m, n, 0);
    launch_q4k_q8_t4_1row_tc(W2, xq, xsc, Y2, m, n, 0);
}

} // namespace rapidllm::cuda_gemv
