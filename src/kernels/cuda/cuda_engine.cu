#include "rapidllm/runtime/cuda_engine.h"

#include "rapidllm/ir/model_desc.h"
#include "rapidllm/kernels/ops.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace rapidllm::cuda_gen {
namespace {

#define CUDA_CHECK(expr)                                                                                               \
    do {                                                                                                               \
        cudaError_t _e = (expr);                                                                                       \
        if (_e != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e));               \
    } while (0)

__device__ __forceinline__ float silu_d(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float sigmoid_d(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float softplus_d(float x) {
    return log1pf(expf(-fabsf(x))) + fmaxf(x, 0.f);
}

__device__ __forceinline__ float e4m3_to_f32(uint8_t b) {
    const int s = (b >> 7) & 1;
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    if (e == 0) {
        if (m == 0) return s ? -0.f : 0.f;
        const float v = ldexpf(static_cast<float>(m) / 8.f, 1 - 7);
        return s ? -v : v;
    }
    if (e == 15 && m == 7) return nanf("");
    const float v = ldexpf(1.f + static_cast<float>(m) / 8.f, e - 7);
    return s ? -v : v;
}

__device__ __forceinline__ float bf16_to_f32(uint16_t h) {
    return __uint_as_float(static_cast<uint32_t>(h) << 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

// GDN S is BF16 in DRAM (half the bytes of float), FP32 in smem for the
// delta-rule. Not the failed half-xs GEMV path — only the recurrent state.
__device__ __forceinline__ void gdn_s_load_f32(float* dst, const uint16_t* src, int n) {
    if ((n & 7) == 0) {
        for (int i = threadIdx.x; i < n / 8; i += blockDim.x) {
            const uint4 u = __ldg(reinterpret_cast<const uint4*>(src) + i);
            const uint16_t* h = reinterpret_cast<const uint16_t*>(&u);
            reinterpret_cast<float4*>(dst)[i * 2] =
                make_float4(bf16_to_f32(h[0]), bf16_to_f32(h[1]), bf16_to_f32(h[2]), bf16_to_f32(h[3]));
            reinterpret_cast<float4*>(dst)[i * 2 + 1] =
                make_float4(bf16_to_f32(h[4]), bf16_to_f32(h[5]), bf16_to_f32(h[6]), bf16_to_f32(h[7]));
        }
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = bf16_to_f32(src[i]);
    }
}

__device__ __forceinline__ void gdn_s_store_bf16(uint16_t* dst, const float* src, int n) {
    if ((n & 7) == 0) {
        for (int i = threadIdx.x; i < n / 8; i += blockDim.x) {
            const float4 a = reinterpret_cast<const float4*>(src)[i * 2];
            const float4 b = reinterpret_cast<const float4*>(src)[i * 2 + 1];
            uint16_t h[8] = {f32_to_bf16(a.x), f32_to_bf16(a.y), f32_to_bf16(a.z), f32_to_bf16(a.w),
                             f32_to_bf16(b.x), f32_to_bf16(b.y), f32_to_bf16(b.z), f32_to_bf16(b.w)};
            reinterpret_cast<uint4*>(dst)[i] = *reinterpret_cast<const uint4*>(h);
        }
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = f32_to_bf16(src[i]);
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

constexpr int kXsTile = 8192;
// 32 KiB dynamic smem keeps 2 blocks/SM on Ada. 48 KiB dropped occupancy and lost ~3%.
constexpr int kFp8XsCap = 8192;
// wd K=17408 → two 8704-col tiles (17×512). 34.8KB xs keeps 2-block occupancy;
// 12288/48KB was the failed 1-block path. One fewer tile sync than 8192+8192+1024.
constexpr int kWdXs = 8704;

__device__ __forceinline__ void write_y(float* y, int row, float acc, int add) {
    if (add) y[row] += acc;
    else y[row] = acc;
}

__global__ void gemv_f32_k(const float* W, const float* x, float* y, int m, int n, int add) {
    __shared__ float xs[kXsTile];
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    float acc = 0.f;
    for (int t0 = 0; t0 < n; t0 += kXsTile) {
        const int tn = n - t0 < kXsTile ? n - t0 : kXsTile;
        for (int i = threadIdx.x; i < tn; i += blockDim.x) xs[i] = x[t0 + i];
        __syncthreads();
        if (row < m) {
            const float* rowp = W + static_cast<size_t>(row) * n + t0;
            for (int j = lane; j < tn; j += 32) acc += rowp[j] * xs[j];
        }
        __syncthreads();
    }
    if (row < m) {
        acc = warp_sum(acc);
        if (lane == 0) write_y(y, row, acc, add);
    }
}

__device__ __forceinline__ float q8_dot32(const int8_t* q, const float* x) {
    float s = 0.f;
    const int4 a = __ldcs(reinterpret_cast<const int4*>(q));
    const int4 b = __ldcs(reinterpret_cast<const int4*>(q + 16));
    const signed char* qa = reinterpret_cast<const signed char*>(&a);
    const signed char* qb = reinterpret_cast<const signed char*>(&b);
#pragma unroll
    for (int k = 0; k < 16; ++k) s += static_cast<float>(qa[k]) * x[k];
#pragma unroll
    for (int k = 0; k < 16; ++k) s += static_cast<float>(qb[k]) * x[16 + k];
    return s;
}

__global__ void gemv_f16_k(const __half* W, const float* x, float* y, int m, int n, int add) {
    __shared__ float xs[kXsTile];
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    float acc = 0.f;
    for (int t0 = 0; t0 < n; t0 += kXsTile) {
        const int tn = n - t0 < kXsTile ? n - t0 : kXsTile;
        for (int i = threadIdx.x; i < tn; i += blockDim.x) xs[i] = x[t0 + i];
        __syncthreads();
        if (row < m) {
            const __half* rowp = W + static_cast<size_t>(row) * n + t0;
            int j = lane * 2;
            for (; j + 1 < tn; j += 64) {
                const float2 h = __half22float2(__ldg(reinterpret_cast<const __half2*>(rowp + j)));
                acc += h.x * xs[j] + h.y * xs[j + 1];
            }
            if ((tn & 1) && lane == 0) acc += __half2float(rowp[tn - 1]) * xs[tn - 1];
        }
        __syncthreads();
    }
    if (row < m) {
        acc = warp_sum(acc);
        if (lane == 0) write_y(y, row, acc, add);
    }
}

__global__ void gemv_bf16_k(const uint16_t* W, const float* x, float* y, int m, int n, int add) {
    __shared__ float xs[kXsTile];
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    float acc = 0.f;
    for (int t0 = 0; t0 < n; t0 += kXsTile) {
        const int tn = n - t0 < kXsTile ? n - t0 : kXsTile;
        for (int i = threadIdx.x; i < tn; i += blockDim.x) xs[i] = x[t0 + i];
        __syncthreads();
        if (row < m) {
            const uint16_t* rowp = W + static_cast<size_t>(row) * n + t0;
            int j = lane * 4;
            for (; j + 3 < tn; j += 128) {
                const uint2 p = __ldcs(reinterpret_cast<const uint2*>(rowp + j));
                acc += bf16_to_f32(static_cast<uint16_t>(p.x)) * xs[j] +
                       bf16_to_f32(static_cast<uint16_t>(p.x >> 16)) * xs[j + 1] +
                       bf16_to_f32(static_cast<uint16_t>(p.y)) * xs[j + 2] +
                       bf16_to_f32(static_cast<uint16_t>(p.y >> 16)) * xs[j + 3];
            }
            for (int r = (tn & ~3) + lane; r < tn; r += 32) acc += bf16_to_f32(rowp[r]) * xs[r];
        }
        if (t0 + kXsTile < n) __syncthreads();
    }
    if (row < m) {
        acc = warp_sum(acc);
        if (lane == 0) write_y(y, row, acc, add);
    }
}

// Two adjacent BF16 rows / warp. uint4 = 8 bf16, aimed at lm_head (248320 x 5120).
__global__ void gemv_bf16_2row_k(const uint16_t* W, const float* x, float* y, int m, int n, int add) {
    __shared__ float xs[kXsTile];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    float acc0 = 0.f, acc1 = 0.f;
    for (int t0 = 0; t0 < n; t0 += kXsTile) {
        const int tn = n - t0 < kXsTile ? n - t0 : kXsTile;
        for (int i = threadIdx.x; i < tn; i += blockDim.x) xs[i] = x[t0 + i];
        __syncthreads();
        if (row0 < m) {
            const uint16_t* p0 = W + static_cast<size_t>(row0) * n + t0;
            const uint16_t* p1 = row1 < m ? W + static_cast<size_t>(row1) * n + t0 : nullptr;
            int j = lane * 8;
            for (; j + 7 < tn; j += 256) {
                const uint4 u0 = __ldcs(reinterpret_cast<const uint4*>(p0 + j));
                acc0 += bf16_to_f32(static_cast<uint16_t>(u0.x)) * xs[j] +
                        bf16_to_f32(static_cast<uint16_t>(u0.x >> 16)) * xs[j + 1] +
                        bf16_to_f32(static_cast<uint16_t>(u0.y)) * xs[j + 2] +
                        bf16_to_f32(static_cast<uint16_t>(u0.y >> 16)) * xs[j + 3] +
                        bf16_to_f32(static_cast<uint16_t>(u0.z)) * xs[j + 4] +
                        bf16_to_f32(static_cast<uint16_t>(u0.z >> 16)) * xs[j + 5] +
                        bf16_to_f32(static_cast<uint16_t>(u0.w)) * xs[j + 6] +
                        bf16_to_f32(static_cast<uint16_t>(u0.w >> 16)) * xs[j + 7];
                if (p1) {
                    const uint4 u1 = __ldcs(reinterpret_cast<const uint4*>(p1 + j));
                    acc1 += bf16_to_f32(static_cast<uint16_t>(u1.x)) * xs[j] +
                            bf16_to_f32(static_cast<uint16_t>(u1.x >> 16)) * xs[j + 1] +
                            bf16_to_f32(static_cast<uint16_t>(u1.y)) * xs[j + 2] +
                            bf16_to_f32(static_cast<uint16_t>(u1.y >> 16)) * xs[j + 3] +
                            bf16_to_f32(static_cast<uint16_t>(u1.z)) * xs[j + 4] +
                            bf16_to_f32(static_cast<uint16_t>(u1.z >> 16)) * xs[j + 5] +
                            bf16_to_f32(static_cast<uint16_t>(u1.w)) * xs[j + 6] +
                            bf16_to_f32(static_cast<uint16_t>(u1.w >> 16)) * xs[j + 7];
                }
            }
            for (int r = (tn & ~7) + lane; r < tn; r += 32) {
                acc0 += bf16_to_f32(p0[r]) * xs[r];
                if (p1) acc1 += bf16_to_f32(p1[r]) * xs[r];
            }
        }
        if (t0 + kXsTile < n) __syncthreads();
    }
    if (row0 < m) {
        acc0 = warp_sum(acc0);
        if (lane == 0) write_y(y, row0, acc0, add);
    }
    if (row1 < m) {
        acc1 = warp_sum(acc1);
        if (lane == 0) write_y(y, row1, acc1, add);
    }
}

// Pad each 32-float Q8 block to 33 to avoid 32-way smem bank conflicts
// (threads in a warp read xs[lane*32 + k] which all hit bank k).
constexpr int kXsPad = 33;

__device__ __forceinline__ int xs_off(int b) { return b * kXsPad; }

// Q8 SoA: quants int8 [m, n], scales half [m, n/32]
__global__ void gemv_q8_soa_k(const int8_t* Q, const __half* scales, const float* x, float* y, int m,
                              int n, int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int tile = n < kXsTile ? n : kXsTile;
    const int nb = n / 32;
    float acc = 0.f;
    const int8_t* rowq = row < m ? Q + static_cast<size_t>(row) * n : nullptr;
    const __half* rows = row < m ? scales + static_cast<size_t>(row) * nb : nullptr;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        const int bn = tn / 32;
        for (int i = threadIdx.x; i < tn; i += blockDim.x) xs[(i >> 5) * kXsPad + (i & 31)] = x[t0 + i];
        __syncthreads();
        if (rowq) {
            const int b0 = t0 / 32;
            int b = lane;
            for (; b + 32 < bn; b += 64) {
                const float d0 = __half2float(__ldcs(rows + b0 + b));
                const float d1 = __half2float(__ldcs(rows + b0 + b + 32));
                acc += d0 * q8_dot32(rowq + t0 + b * 32, xs + xs_off(b));
                acc += d1 * q8_dot32(rowq + t0 + (b + 32) * 32, xs + xs_off(b + 32));
            }
            if (b < bn) {
                const float d = __half2float(__ldcs(rows + b0 + b));
                acc += d * q8_dot32(rowq + t0 + b * 32, xs + xs_off(b));
            }
        }
        __syncthreads();
    }
    if (row < m) {
        acc = warp_sum(acc);
        if (lane == 0) write_y(y, row, acc, add);
    }
}

// Batched Q8 GEMM: X [T, n], Y [T, m], stream each weight row once. T <= 4.
__global__ void gemm_q8_soa_k(const int8_t* Q, const __half* scales, const float* X, float* Y, int m,
                              int n, int T, int tile) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb = n / 32;
    const int tstride = (tile / 32) * kXsPad;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    const int8_t* rowq = row < m ? Q + static_cast<size_t>(row) * n : nullptr;
    const __half* rows = row < m ? scales + static_cast<size_t>(row) * nb : nullptr;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        for (int i = threadIdx.x; i < T * tn; i += blockDim.x) {
            const int t = i / tn;
            const int j = i - t * tn;
            xs[static_cast<size_t>(t) * tstride + (j >> 5) * kXsPad + (j & 31)] =
                X[static_cast<size_t>(t) * n + t0 + j];
        }
        __syncthreads();
        if (rowq) {
            const int bn = tn / 32;
            for (int b = lane; b < bn; b += 32) {
                const float d = __half2float(__ldcs(rows + t0 / 32 + b));
                const int8_t* q = rowq + t0 + b * 32;
                acc0 += d * q8_dot32(q, xs + xs_off(b));
                if (T > 1) acc1 += d * q8_dot32(q, xs + tstride + xs_off(b));
                if (T > 2) acc2 += d * q8_dot32(q, xs + 2 * tstride + xs_off(b));
                if (T > 3) acc3 += d * q8_dot32(q, xs + 3 * tstride + xs_off(b));
            }
        }
        __syncthreads();
    }
    if (row < m) {
        float a0 = warp_sum(acc0);
        if (lane == 0) Y[row] = a0;
        if (T > 1) {
            float a1 = warp_sum(acc1);
            if (lane == 0) Y[m + row] = a1;
        }
        if (T > 2) {
            float a2 = warp_sum(acc2);
            if (lane == 0) Y[2 * m + row] = a2;
        }
        if (T > 3) {
            float a3 = warp_sum(acc3);
            if (lane == 0) Y[3 * m + row] = a3;
        }
    }
}

__device__ __forceinline__ float fp8e4_to_f32(uint8_t b) {
#if __CUDA_ARCH__ >= 800
    const __half_raw hr = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3);
    __half h;
    *reinterpret_cast<unsigned short*>(&h) = hr.x;
    return __half2float(h);
#else
    return e4m3_to_f32(b);
#endif
}

__device__ __forceinline__ uint8_t f32_to_fp8e4(float f) {
#if __CUDA_ARCH__ >= 800
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3));
#else
    return 0;
#endif
}

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void mma_m16n8k16_f16(float d[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

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

__device__ __forceinline__ void mma_m16n8k32_e4m3(float d[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}
#endif

__global__ void zero_f32_k(float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 0.f;
}

// Native e4m3 tensor-core GEMV (T=1). W stays e4m3; x is quantized per 32-col tile (W8A8).
// grid.y = K-splits for occupancy on small M. splits==1 writes y; else atomicAdd.
__global__ void gemv_fp8_e4m3_mma_k(const uint8_t* W, const float* scale, const float* x, float* y, int m,
                                    int n) {
#if __CUDA_ARCH__ >= 890
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * warps + warp) * 16;
    extern __shared__ char raw[];
    uint8_t* As = reinterpret_cast<uint8_t*>(raw); // [warps][16*32]
    uint8_t* Bs = As + warps * 16 * 32;            // [8][32] col-major e4m3
    float* xf = reinterpret_cast<float*>(Bs + 8 * 32);
    float tot0 = 0.f, tot2 = 0.f;
    const int nb_n = (n + 127) / 128;
    const int bi = row0 / 128;
    const int splits = gridDim.y;
    int k_chunk = n;
    if (splits > 1) k_chunk = ((n / splits) + 127) & ~127;
    const int k0_base = blockIdx.y * k_chunk;
    const int k_end = k0_base + k_chunk < n ? k0_base + k_chunk : n;
    if (k0_base >= n) return;
    for (int k0 = k0_base; k0 < k_end; k0 += 32) {
        uint8_t* Aw = As + warp * 16 * 32;
        {
            const int rt = row0 / 16;
            const int kt = k0 / 32;
            const int nn = n / 32;
            const uint8_t* src = W + (static_cast<size_t>(rt) * nn + kt) * 512 + static_cast<size_t>(lane) * 16;
            *reinterpret_cast<uint4*>(Aw + lane * 16) = *reinterpret_cast<const uint4*>(src);
        }
        if (warp == 0) {
            const float xv = (k0 + lane < n) ? x[k0 + lane] : 0.f;
            xf[lane] = xv;
            float amax = warp_max(fabsf(xv));
            const float xs = fmaxf(amax, 1e-12f) / 448.f;
            if (lane == 0) xf[32] = xs; // stash scale after the 32 values — need extra slot
        }
        __syncthreads();
        const float xs = xf[32];
        if (warp == 0) {
            Bs[lane] = f32_to_fp8e4(xf[lane] / xs);
#pragma unroll
            for (int col = 1; col < 8; ++col) Bs[col * 32 + lane] = 0;
        }
        __syncthreads();
        uint32_t a[4], b[2];
        ldmatrix_x4(a, Aw + (lane % 16) * 32 + (lane / 16) * 16);
        ldmatrix_x2(b, Bs + (lane % 8) * 32 + (lane / 8) * 16);
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        mma_m16n8k32_e4m3(d, a, b);
        const float ws = (row0 < m) ? scale[bi * nb_n + k0 / 128] : 0.f;
        const float g = ws * xs;
        tot0 += d[0] * g;
        tot2 += d[2] * g;
        __syncthreads();
    }
    if (row0 >= m) return;
    const int r0 = lane / 4;
    if ((lane & 3) != 0) return;
    if (splits > 1) {
        if (row0 + r0 < m) atomicAdd(y + row0 + r0, tot0);
        if (row0 + r0 + 8 < m) atomicAdd(y + row0 + r0 + 8, tot2);
    } else {
        if (row0 + r0 < m) y[row0 + r0] = tot0;
        if (row0 + r0 + 8 < m) y[row0 + r0 + 8] = tot2;
    }
#else
    (void)W;
    (void)scale;
    (void)x;
    (void)y;
    (void)m;
    (void)n;
#endif
}

// Tensor-core block-scaled FP8 GEMM: W e4m3 [m,n] * scale[m/128,n/128], X f32 [T,n] -> Y f32 [T,m].
// Each warp owns 16 output rows. T<=8 uses the n8 MMA tile.
__global__ void gemm_fp8_tc_k(const uint8_t* W, const float* scale, const float* X, float* Y, int m, int n,
                              int T) {
#if __CUDA_ARCH__ >= 800
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * warps + warp) * 16;
    extern __shared__ char raw[];
    __half* As = reinterpret_cast<__half*>(raw); // [warps][16][16] row-major
    __half* Bs = As + warps * 16 * 16;           // [8][16] col-major = n-major
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const int nb_n = (n + 127) / 128;
    const int bi = row0 / 128;
    for (int k0 = 0; k0 < n; k0 += 16) {
        const float s = (row0 < m) ? scale[bi * nb_n + k0 / 128] : 0.f;
        __half* Aw = As + warp * 16 * 16;
        for (int i = lane; i < 16 * 16; i += 32) {
            const int r = i >> 4;
            const int c = i & 15;
            float v = 0.f;
            if (row0 + r < m && k0 + c < n) v = fp8e4_to_f32(W[static_cast<size_t>(row0 + r) * n + k0 + c]) * s;
            Aw[r * 16 + c] = __float2half(v);
        }
        if (warp == 0) {
            for (int i = lane; i < 16 * 8; i += 32) {
                const int r = i & 15; // k
                const int c = i >> 4; // n-col
                float v = 0.f;
                if (c < T && k0 + r < n) v = X[static_cast<size_t>(c) * n + k0 + r];
                Bs[c * 16 + r] = __float2half(v);
            }
        }
        __syncthreads();
        uint32_t a[4], b[2];
        ldmatrix_x4(a, Aw + (lane % 16) * 16 + (lane / 16) * 8);
        ldmatrix_x2(b, Bs + (lane % 8) * 16 + (lane / 8) * 8);
        mma_m16n8k16_f16(acc, a, b);
        __syncthreads();
    }
    if (row0 >= m) return;
    const int r0 = lane / 4;
    const int c0 = (lane % 4) * 2;
    if (c0 < T) {
        if (row0 + r0 < m) Y[static_cast<size_t>(c0) * m + row0 + r0] = acc[0];
        if (row0 + r0 + 8 < m) Y[static_cast<size_t>(c0) * m + row0 + r0 + 8] = acc[2];
    }
    if (c0 + 1 < T) {
        if (row0 + r0 < m) Y[static_cast<size_t>(c0 + 1) * m + row0 + r0] = acc[1];
        if (row0 + r0 + 8 < m) Y[static_cast<size_t>(c0 + 1) * m + row0 + r0 + 8] = acc[3];
    }
#else
    (void)W;
    (void)scale;
    (void)X;
    (void)Y;
    (void)m;
    (void)n;
    (void)T;
#endif
}

__device__ __forceinline__ void load_x_tile(float* xs, const float* x, int tn) {
    const int n4 = tn >> 2;
    for (int i = static_cast<int>(threadIdx.x); i < n4; i += static_cast<int>(blockDim.x))
        reinterpret_cast<float4*>(xs)[i] = reinterpret_cast<const float4*>(x)[i];
    for (int i = (n4 << 2) + static_cast<int>(threadIdx.x); i < tn; i += static_cast<int>(blockDim.x))
        xs[i] = x[i];
}

__device__ __forceinline__ float tile_ss_inv(const float* xs, int tn, int n, const float* d_ss, float eps) {
    if (d_ss) return rsqrtf((*d_ss) / static_cast<float>(n) + eps);
    float ss = 0.f;
    const int n4 = tn >> 2;
    for (int i = static_cast<int>(threadIdx.x); i < n4; i += static_cast<int>(blockDim.x)) {
        const float4 v = reinterpret_cast<const float4*>(xs)[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int i = (n4 << 2) + static_cast<int>(threadIdx.x); i < tn; i += static_cast<int>(blockDim.x))
        ss += xs[i] * xs[i];
    ss = warp_sum(ss);
    __shared__ float wbuf[16];
    if ((threadIdx.x & 31) == 0) wbuf[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.f;
        const int nw = blockDim.x >> 5;
        for (int i = 0; i < nw; ++i) tot += wbuf[i];
        wbuf[0] = rsqrtf(tot / static_cast<float>(n) + eps);
    }
    __syncthreads();
    return wbuf[0];
}

// Apply RMS to an already-loaded x tile: xs[i] *= (1+gamma[t0+i]) * rsqrt(ss/n+eps)
// d_ss == nullptr: reduce the tile (requires the full vector in xs, i.e. n == tn).
__device__ __forceinline__ void scale_x_rms(float* xs, const float* gamma, int t0, int tn, int n, const float* d_ss,
                                           float eps) {
    const float inv = tile_ss_inv(xs, tn, n, d_ss, eps);
    const int n4 = tn >> 2;
    for (int i = static_cast<int>(threadIdx.x); i < n4; i += static_cast<int>(blockDim.x)) {
        float4 v = reinterpret_cast<float4*>(xs)[i];
        float4 g = {1.f, 1.f, 1.f, 1.f};
        if (gamma) {
            const float4 gg = reinterpret_cast<const float4*>(gamma + t0)[i];
            g.x += gg.x;
            g.y += gg.y;
            g.z += gg.z;
            g.w += gg.w;
        }
        v.x *= inv * g.x;
        v.y *= inv * g.y;
        v.z *= inv * g.z;
        v.w *= inv * g.w;
        reinterpret_cast<float4*>(xs)[i] = v;
    }
    for (int i = (n4 << 2) + static_cast<int>(threadIdx.x); i < tn; i += static_cast<int>(blockDim.x)) {
        const float g = gamma ? (1.f + gamma[t0 + i]) : 1.f;
        xs[i] *= inv * g;
    }
}

// GDN out-proj: xs[i] *= gnorm[(t0+i)%gamma_n] * rsqrt(ss/n+eps) * silu(z[t0+i])
__device__ __forceinline__ void scale_x_gated_rms(float* xs, const float* gamma, const float* z, int t0, int tn,
                                                 int n, int gamma_n, const float* d_ss, float eps) {
    const float inv = tile_ss_inv(xs, tn, n, d_ss, eps);
    if (gamma_n <= 0) gamma_n = n;
    const int n4 = tn >> 2;
    for (int i = static_cast<int>(threadIdx.x); i < n4; i += static_cast<int>(blockDim.x)) {
        float4 v = reinterpret_cast<float4*>(xs)[i];
        const float4 zv = reinterpret_cast<const float4*>(z + t0)[i];
        const int i0 = t0 + (i << 2);
        const float g0 = gamma ? gamma[i0 % gamma_n] : 1.f;
        const float g1 = gamma ? gamma[(i0 + 1) % gamma_n] : 1.f;
        const float g2 = gamma ? gamma[(i0 + 2) % gamma_n] : 1.f;
        const float g3 = gamma ? gamma[(i0 + 3) % gamma_n] : 1.f;
        v.x *= g0 * inv * silu_d(zv.x);
        v.y *= g1 * inv * silu_d(zv.y);
        v.z *= g2 * inv * silu_d(zv.z);
        v.w *= g3 * inv * silu_d(zv.w);
        reinterpret_cast<float4*>(xs)[i] = v;
    }
    for (int i = (n4 << 2) + static_cast<int>(threadIdx.x); i < tn; i += static_cast<int>(blockDim.x)) {
        const float g = gamma ? gamma[(t0 + i) % gamma_n] : 1.f;
        xs[i] *= g * inv * silu_d(z[t0 + i]);
    }
}

// 512-col super-tile (4×128), lane-interleaved so one uint4 covers 4 K-blocks:
//   W[(sg * rows + row) * 512 + lane * 16 + t * 4]   sg=cb/4, t=cb%4
// Same matrix, same bytes as old 128-col K-major — not the failed wg|wu interleave.
// cols is padded to a multiple of 512 at pack time (27B dims already align).
int fp8_pack_cols(int cols) { return ((cols + 511) / 512) * 512; }

__device__ __forceinline__ const uint32_t* fp8_blk4(const uint8_t* W, int row, int rows, int cb, int lane) {
    return reinterpret_cast<const uint32_t*>(
        W + (static_cast<size_t>(cb >> 2) * rows + row) * 512 + (lane << 4) + ((cb & 3) << 2));
}

__device__ __forceinline__ uint4 fp8_ld16(const uint8_t* W, int row, int rows, int cb0, int lane) {
    return __ldcs(reinterpret_cast<const uint4*>(
        W + (static_cast<size_t>(cb0 >> 2) * rows + row) * 512 + (lane << 4)));
}

// Cache-all: do not mark evict-first. For wo after GDN L2 prefetch so
// later blocks still hit the prefetched lines. Not the failed L2::256B /
// evict_first PTX path.
__device__ __forceinline__ uint4 fp8_ld16_ca(const uint8_t* W, int row, int rows, int cb0, int lane) {
    return __ldg(reinterpret_cast<const uint4*>(
        W + (static_cast<size_t>(cb0 >> 2) * rows + row) * 512 + (lane << 4)));
}

void pack_fp8_kmajor_host(uint8_t* dst, const uint8_t* src, int rows, int cols) {
    const int nb = cols / 128;
    const int pc = fp8_pack_cols(cols);
    if (pc != cols) std::memset(dst, 0, static_cast<size_t>(rows) * static_cast<size_t>(pc));
    for (int r = 0; r < rows; ++r) {
        const uint8_t* s = src + static_cast<size_t>(r) * cols;
        for (int cb = 0; cb < nb; ++cb) {
            const int sg = cb >> 2;
            const int t = cb & 3;
            uint8_t* tile = dst + (static_cast<size_t>(sg) * rows + r) * 512;
            const uint8_t* blk = s + cb * 128;
            for (int lane = 0; lane < 32; ++lane)
                std::memcpy(tile + lane * 16 + t * 4, blk + lane * 4, 4);
        }
    }
}

__device__ __forceinline__ float fp8x4_dot(uint32_t p, const float* xs) {
#if __CUDA_ARCH__ >= 890
    const float4 xv = *reinterpret_cast<const float4*>(xs);
    unsigned h01, h23;
    const unsigned short lo = static_cast<unsigned short>(p);
    const unsigned short hi = static_cast<unsigned short>(p >> 16);
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h01) : "h"(lo));
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h23) : "h"(hi));
    const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h01));
    const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h23));
    return f0.x * xv.x + f0.y * xv.y + f1.x * xv.z + f1.y * xv.w;
#else
    return fp8e4_to_f32(static_cast<uint8_t>(p)) * xs[0] +
           fp8e4_to_f32(static_cast<uint8_t>(p >> 8)) * xs[1] +
           fp8e4_to_f32(static_cast<uint8_t>(p >> 16)) * xs[2] +
           fp8e4_to_f32(static_cast<uint8_t>(p >> 24)) * xs[3];
#endif
}

// One uint4 weight + float4 scale → 4 consecutive 128-col tiles (xb at tile0, lane*4).
__device__ __forceinline__ float fp8_dot16(uint4 w, float4 s, const float* xb) {
    return s.x * fp8x4_dot(w.x, xb) + s.y * fp8x4_dot(w.y, xb + 128) +
           s.z * fp8x4_dot(w.z, xb + 256) + s.w * fp8x4_dot(w.w, xb + 384);
}

// Consecutive even/odd rows share the same 128-row scale block.
// Aligned K (multiple of 512, one xs tile): no 128-col tail, no tile loop.
__device__ __forceinline__ void acc_fp8_2row_al(float& a0, float& a1, const uint8_t* W, const float* sc,
                                               int row0, int row1, int rows, const float* xs, int n, int lane,
                                               int live1, int b0 = 0) {
    const int nb = n >> 7;
#pragma unroll 2
    for (int b = 0; b < nb; b += 4) {
        const float* xb = xs + (b << 7) + (lane << 2);
        const float4 sv = __ldg(reinterpret_cast<const float4*>(sc + b0 + b));
        a0 += fp8_dot16(fp8_ld16(W, row0, rows, b0 + b, lane), sv, xb);
        if (live1) a1 += fp8_dot16(fp8_ld16(W, row1, rows, b0 + b, lane), sv, xb);
    }
}

__device__ __forceinline__ void acc_fp8_2row_al_ca(float& a0, float& a1, const uint8_t* W, const float* sc,
                                                  int row0, int row1, int rows, const float* xs, int n, int lane,
                                                  int live1) {
    const int nb = n >> 7;
#pragma unroll 2
    for (int b = 0; b < nb; b += 4) {
        const float* xb = xs + (b << 7) + (lane << 2);
        const float4 sv = __ldg(reinterpret_cast<const float4*>(sc + b));
        a0 += fp8_dot16(fp8_ld16_ca(W, row0, rows, b, lane), sv, xb);
        if (live1) a1 += fp8_dot16(fp8_ld16_ca(W, row1, rows, b, lane), sv, xb);
    }
}

__global__ void __launch_bounds__(512, 2)
gemv_fp8_k(const uint8_t* W, const float* scale, const float* x, float* y, int m, int n, int block, int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int tile = n <= kFp8XsCap ? n : kFp8XsCap;
    const int nb_n = (n + block - 1) / block;
    const int bi = row / block;
    float acc = 0.f;
    const float* sc = (row < m && scale) ? scale + bi * nb_n : nullptr;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        load_x_tile(xs, x + t0, tn);
        __syncthreads();
        if (row < m && sc) {
            const int b0 = t0 / block;
            const int nb = tn / block;
            int b = 0;
            for (; b + 3 < nb; b += 4) {
                const float4 sv = __ldg(reinterpret_cast<const float4*>(sc + b0 + b));
                acc += fp8_dot16(fp8_ld16(W, row, m, b0 + b, lane), sv, xs + b * block + lane * 4);
            }
            for (; b < nb; ++b) {
                const float s0 = sc[b0 + b];
                const uint32_t p0 = __ldcs(fp8_blk4(W, row, m, b0 + b, lane));
                acc += s0 * fp8x4_dot(p0, xs + b * block + lane * 4);
            }
        }
        if (t0 + tile < n) __syncthreads();
    }
    if (row < m) {
        acc = warp_sum(acc);
        if (lane == 0) write_y(y, row, acc, add);
    }
}

// Two adjacent rows per warp — extra ILP on the same x tile.
__global__ void __launch_bounds__(512, 2)
gemv_fp8_2row_k(const uint8_t* W, const float* scale, const float* x, float* y, int m, int n, int block,
                int add, float* acc_ss, const float* x_gamma, const float* d_ss, float rms_eps,
                const float* x_sig, float* y_split, int split_at, const float* x_silu, int gnorm_n) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int tile = n <= kFp8XsCap ? n : kFp8XsCap;
    const int nb_n = (n + block - 1) / block;
    float acc0 = 0.f, acc1 = 0.f;
    const float* sc0 = (row0 < m && scale) ? scale + (row0 / block) * nb_n : nullptr;
    const float* sc1 = (row1 < m && scale) ? scale + (row1 / block) * nb_n : nullptr;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        load_x_tile(xs, x + t0, tn);
        if (x_silu && (d_ss || tn == n)) {
            __syncthreads();
            scale_x_gated_rms(xs, x_gamma, x_silu, t0, tn, n, gnorm_n, d_ss, rms_eps);
        } else if (x_gamma && (d_ss || tn == n)) {
            __syncthreads();
            scale_x_rms(xs, x_gamma, t0, tn, n, d_ss, rms_eps);
        }
        if (x_sig) {
            __syncthreads();
            for (int i = static_cast<int>(threadIdx.x); i < tn; i += static_cast<int>(blockDim.x))
                xs[i] *= sigmoid_d(x_sig[t0 + i]);
        }
        __syncthreads();
        if (row0 < m && sc0) {
            const int b0 = t0 / block;
            const int nb = tn / block;
            int b = 0;
            for (; b + 3 < nb; b += 4) {
                const float* xb = xs + b * block + lane * 4;
                acc0 += fp8_dot16(fp8_ld16(W, row0, m, b0 + b, lane),
                                  __ldg(reinterpret_cast<const float4*>(sc0 + b0 + b)), xb);
                if (row1 < m && sc1)
                    acc1 += fp8_dot16(fp8_ld16(W, row1, m, b0 + b, lane),
                                      __ldg(reinterpret_cast<const float4*>(sc1 + b0 + b)), xb);
            }
            for (; b < nb; ++b) {
                acc0 += sc0[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(W, row0, m, b0 + b, lane)),
                                                xs + b * block + lane * 4);
                if (row1 < m && sc1)
                    acc1 += sc1[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(W, row1, m, b0 + b, lane)),
                                                    xs + b * block + lane * 4);
            }
        }
        if (t0 + tile < n) __syncthreads();
    }
    if (row0 < m) {
        acc0 = warp_sum(acc0);
        if (lane == 0) {
            if (y_split && split_at > 0 && row0 >= split_at) y_split[row0 - split_at] = acc0;
            else write_y(y, row0, acc0, add);
        }
    }
    if (row1 < m) {
        acc1 = warp_sum(acc1);
        if (lane == 0) {
            if (y_split && split_at > 0 && row1 >= split_at) y_split[row1 - split_at] = acc1;
            else write_y(y, row1, acc1, add);
        }
    }
    if (acc_ss && lane == 0) {
        float s = 0.f;
        if (row0 < m) {
            const float v = y[row0];
            s += v * v;
        }
        if (row1 < m) {
            const float v = y[row1];
            s += v * v;
        }
        if (s != 0.f) atomicAdd(acc_ss, s);
    }
}

// Residual down-proj (wd): no RMS/gate/split. Same 2-row math, fewer regs than gemv_fp8_2row_k.
__global__ void __launch_bounds__(512, 2)
gemv_fp8_2row_add_k(const uint8_t* W, const float* scale, const float* x, float* y, int m, int n, int block) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int tile = n <= kFp8XsCap ? n : kFp8XsCap;
    const int nb_n = (n + block - 1) / block;
    float acc0 = 0.f, acc1 = 0.f;
    const float* sc0 = (row0 < m && scale) ? scale + (row0 / block) * nb_n : nullptr;
    const float* sc1 = (row1 < m && scale) ? scale + (row1 / block) * nb_n : nullptr;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        load_x_tile(xs, x + t0, tn);
        __syncthreads();
        if (row0 < m && sc0) {
            const int b0 = t0 / block;
            const int nb = tn / block;
            int b = 0;
            for (; b + 3 < nb; b += 4) {
                const float* xb = xs + b * block + lane * 4;
                acc0 += fp8_dot16(fp8_ld16(W, row0, m, b0 + b, lane),
                                  __ldg(reinterpret_cast<const float4*>(sc0 + b0 + b)), xb);
                if (row1 < m && sc1)
                    acc1 += fp8_dot16(fp8_ld16(W, row1, m, b0 + b, lane),
                                      __ldg(reinterpret_cast<const float4*>(sc1 + b0 + b)), xb);
            }
            for (; b < nb; ++b) {
                acc0 += sc0[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(W, row0, m, b0 + b, lane)),
                                                xs + b * block + lane * 4);
                if (row1 < m && sc1)
                    acc1 += sc1[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(W, row1, m, b0 + b, lane)),
                                                    xs + b * block + lane * 4);
            }
        }
        if (t0 + tile < n) __syncthreads();
    }
    if (row0 < m) {
        acc0 = warp_sum(acc0);
        if (lane == 0) write_y(y, row0, acc0, 1);
    }
    if (row1 < m) {
        acc1 = warp_sum(acc1);
        if (lane == 0) write_y(y, row1, acc1, 1);
    }
}

// Single-tile 2-row (K multiple of 512, K <= 8192). Hot path for wo/wq/lm_head.
__global__ void __launch_bounds__(512, 2)
gemv_fp8_2row_st_k(const uint8_t* W, const float* scale, const float* x, float* y, int m, int n, int block,
                   int add, float* acc_ss, const float* x_gamma, const float* d_ss, float rms_eps,
                   const float* x_sig, float* y_split, int split_at, const float* x_silu, int gnorm_n,
                   int use_ca) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb_n = n / block;
    float acc0 = 0.f, acc1 = 0.f;
    const float* sc0 = (row0 < m && scale) ? scale + (row0 / block) * nb_n : nullptr;
    load_x_tile(xs, x, n);
    if (x_silu) {
        __syncthreads();
        scale_x_gated_rms(xs, x_gamma, x_silu, 0, n, n, gnorm_n, d_ss, rms_eps);
    } else if (x_gamma) {
        __syncthreads();
        scale_x_rms(xs, x_gamma, 0, n, n, d_ss, rms_eps);
    }
    if (x_sig) {
        __syncthreads();
        for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x))
            xs[i] *= sigmoid_d(x_sig[i]);
    }
    __syncthreads();
    if (row0 < m && sc0) {
        if (use_ca)
            acc_fp8_2row_al_ca(acc0, acc1, W, sc0, row0, row1, m, xs, n, lane, (row1 < m) ? 1 : 0);
        else
            acc_fp8_2row_al(acc0, acc1, W, sc0, row0, row1, m, xs, n, lane, (row1 < m) ? 1 : 0);
    }
    if (row0 < m) {
        acc0 = warp_sum(acc0);
        if (lane == 0) {
            if (y_split && split_at > 0 && row0 >= split_at) y_split[row0 - split_at] = acc0;
            else write_y(y, row0, acc0, add);
        }
    }
    if (row1 < m) {
        acc1 = warp_sum(acc1);
        if (lane == 0) {
            if (y_split && split_at > 0 && row1 >= split_at) y_split[row1 - split_at] = acc1;
            else write_y(y, row1, acc1, add);
        }
    }
    if (acc_ss && lane == 0) {
        float s = 0.f;
        if (row0 < m) {
            const float v = y[row0];
            s += v * v;
        }
        if (row1 < m) {
            const float v = y[row1];
            s += v * v;
        }
        if (s != 0.f) atomicAdd(acc_ss, s);
    }
}

__global__ void __launch_bounds__(512, 2)
gemv_fp8_2row_add_st_k(const uint8_t* W, const float* scale, const float* x, float* y, int m, int n, int block) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb_n = n / block;
    float acc0 = 0.f, acc1 = 0.f;
    const float* sc0 = (row0 < m && scale) ? scale + (row0 / block) * nb_n : nullptr;
    load_x_tile(xs, x, n);
    __syncthreads();
    if (row0 < m && sc0)
        acc_fp8_2row_al(acc0, acc1, W, sc0, row0, row1, m, xs, n, lane, (row1 < m) ? 1 : 0);
    if (row0 < m) {
        acc0 = warp_sum(acc0);
        if (lane == 0) write_y(y, row0, acc0, 1);
    }
    if (row1 < m) {
        acc1 = warp_sum(acc1);
        if (lane == 0) write_y(y, row1, acc1, 1);
    }
}

// wd (K=17408): two 8704-col tiles when n==2*kWdXs; else 8192+8192+1024. No RMS/gate.
__global__ void __launch_bounds__(512, 2)
gemv_fp8_2row_add_wd_k(const uint8_t* W, const float* scale, const float* x, float* y, int m, int n, int block) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb_n = (n + block - 1) / block;
    float acc0 = 0.f, acc1 = 0.f;
    const float* sc0 = (row0 < m && scale) ? scale + (row0 / block) * nb_n : nullptr;
    const int live1 = (row1 < m) ? 1 : 0;
    if (n == 2 * kWdXs) {
        load_x_tile(xs, x, kWdXs);
        __syncthreads();
        if (row0 < m && sc0) acc_fp8_2row_al(acc0, acc1, W, sc0, row0, row1, m, xs, kWdXs, lane, live1, 0);
        __syncthreads();
        load_x_tile(xs, x + kWdXs, kWdXs);
        __syncthreads();
        if (row0 < m && sc0)
            acc_fp8_2row_al(acc0, acc1, W, sc0, row0, row1, m, xs, kWdXs, lane, live1, kWdXs / 128);
    } else {
        load_x_tile(xs, x, kFp8XsCap);
        __syncthreads();
        if (row0 < m && sc0) acc_fp8_2row_al(acc0, acc1, W, sc0, row0, row1, m, xs, kFp8XsCap, lane, live1, 0);
        __syncthreads();
        load_x_tile(xs, x + kFp8XsCap, kFp8XsCap);
        __syncthreads();
        if (row0 < m && sc0) acc_fp8_2row_al(acc0, acc1, W, sc0, row0, row1, m, xs, kFp8XsCap, lane, live1, 64);
        const int rem = n - 2 * kFp8XsCap;
        if (rem > 0) {
            __syncthreads();
            load_x_tile(xs, x + 2 * kFp8XsCap, rem);
            __syncthreads();
            if (row0 < m && sc0) acc_fp8_2row_al(acc0, acc1, W, sc0, row0, row1, m, xs, rem, lane, live1, 128);
        }
    }
    if (row0 < m) {
        acc0 = warp_sum(acc0);
        if (lane == 0) write_y(y, row0, acc0, 1);
    }
    if (row1 < m) {
        acc1 = warp_sum(acc1);
        if (lane == 0) write_y(y, row1, acc1, 1);
    }
}

// Two same-shape FP8 GEMVs sharing x (mlp.gate + mlp.up).
// No launch_bounds: two weight streams need extra regs; let the compiler pick occupancy.
__global__ void
gemv_fp8_dual_k(const uint8_t* W1, const float* s1, const uint8_t* W2, const float* s2, const float* x,
                float* y1, float* y2, int m, int n, int block, int fuse_swiglu, const float* x_gamma,
                const float* d_ss, float rms_eps) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int tile = n <= kFp8XsCap ? n : kFp8XsCap;
    const int nb_n = (n + block - 1) / block;
    const int bi = row / block;
    float acc1 = 0.f, acc2 = 0.f;
    const float* sc1 = (row < m && s1) ? s1 + bi * nb_n : nullptr;
    const float* sc2 = (row < m && s2) ? s2 + bi * nb_n : nullptr;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        for (int i = threadIdx.x; i < tn; i += blockDim.x) xs[i] = x[t0 + i];
        if (x_gamma && (d_ss || tn == n)) {
            __syncthreads();
            scale_x_rms(xs, x_gamma, t0, tn, n, d_ss, rms_eps);
        }
        __syncthreads();
        if (row < m && sc1 && sc2) {
            const int b0 = t0 / block;
            const int nb = tn / block;
            int b = 0;
            for (; b + 3 < nb; b += 4) {
                const float* xb = xs + b * block + lane * 4;
                acc1 += fp8_dot16(fp8_ld16(W1, row, m, b0 + b, lane),
                                  __ldg(reinterpret_cast<const float4*>(sc1 + b0 + b)), xb);
                acc2 += fp8_dot16(fp8_ld16(W2, row, m, b0 + b, lane),
                                  __ldg(reinterpret_cast<const float4*>(sc2 + b0 + b)), xb);
            }
            for (; b < nb; ++b) {
                const uint32_t p1 = __ldcs(fp8_blk4(W1, row, m, b0 + b, lane));
                const uint32_t p2 = __ldcs(fp8_blk4(W2, row, m, b0 + b, lane));
                const float* xb = xs + b * block + lane * 4;
                acc1 += sc1[b0 + b] * fp8x4_dot(p1, xb);
                acc2 += sc2[b0 + b] * fp8x4_dot(p2, xb);
            }
        }
        if (t0 + tile < n) __syncthreads();
    }
    if (row < m) {
        acc1 = warp_sum(acc1);
        acc2 = warp_sum(acc2);
        if (lane == 0) {
            if (fuse_swiglu) y1[row] = silu_d(acc1) * acc2;
            else {
                y1[row] = acc1;
                y2[row] = acc2;
            }
        }
    }
}

// 2-row dual: same x tile, two weight matrices, 2 rows/warp (wg+wu hot path).
__global__ void __launch_bounds__(512, 2)
gemv_fp8_2row_dual_k(const uint8_t* W1, const float* s1, const uint8_t* W2, const float* s2, const float* x,
                     float* y1, float* y2, int m, int n, int block, int fuse_swiglu, const float* x_gamma,
                     const float* d_ss, float rms_eps) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int tile = n <= kFp8XsCap ? n : kFp8XsCap;
    const int nb_n = (n + block - 1) / block;
    float a10 = 0.f, a11 = 0.f, a20 = 0.f, a21 = 0.f;
    const float* sc10 = (row0 < m && s1) ? s1 + (row0 / block) * nb_n : nullptr;
    const float* sc11 = (row1 < m && s1) ? s1 + (row1 / block) * nb_n : nullptr;
    const float* sc20 = (row0 < m && s2) ? s2 + (row0 / block) * nb_n : nullptr;
    const float* sc21 = (row1 < m && s2) ? s2 + (row1 / block) * nb_n : nullptr;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        load_x_tile(xs, x + t0, tn);
        if (x_gamma && (d_ss || tn == n)) {
            __syncthreads();
            scale_x_rms(xs, x_gamma, t0, tn, n, d_ss, rms_eps);
        }
        __syncthreads();
        if (row0 < m && sc10) {
            const int b0 = t0 / block;
            const int nb = tn / block;
            int b = 0;
            for (; b + 3 < nb; b += 4) {
                const float* xb = xs + b * block + lane * 4;
                a10 += fp8_dot16(fp8_ld16(W1, row0, m, b0 + b, lane),
                                 __ldg(reinterpret_cast<const float4*>(sc10 + b0 + b)), xb);
                if (sc20)
                    a20 += fp8_dot16(fp8_ld16(W2, row0, m, b0 + b, lane),
                                     __ldg(reinterpret_cast<const float4*>(sc20 + b0 + b)), xb);
                if (row1 < m && sc11)
                    a11 += fp8_dot16(fp8_ld16(W1, row1, m, b0 + b, lane),
                                     __ldg(reinterpret_cast<const float4*>(sc11 + b0 + b)), xb);
                if (row1 < m && sc21)
                    a21 += fp8_dot16(fp8_ld16(W2, row1, m, b0 + b, lane),
                                     __ldg(reinterpret_cast<const float4*>(sc21 + b0 + b)), xb);
            }
            for (; b < nb; ++b) {
                const float* xb = xs + b * block + lane * 4;
                a10 += sc10[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(W1, row0, m, b0 + b, lane)), xb);
                if (sc20)
                    a20 += sc20[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(W2, row0, m, b0 + b, lane)), xb);
                if (row1 < m && sc11)
                    a11 += sc11[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(W1, row1, m, b0 + b, lane)), xb);
                if (row1 < m && sc21)
                    a21 += sc21[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(W2, row1, m, b0 + b, lane)), xb);
            }
        }
        if (t0 + tile < n) __syncthreads();
    }
    if (row0 < m) {
        a10 = warp_sum(a10);
        a20 = warp_sum(a20);
        if (lane == 0) {
            if (fuse_swiglu) y1[row0] = silu_d(a10) * a20;
            else {
                y1[row0] = a10;
                y2[row0] = a20;
            }
        }
    }
    if (row1 < m) {
        a11 = warp_sum(a11);
        a21 = warp_sum(a21);
        if (lane == 0) {
            if (fuse_swiglu) y1[row1] = silu_d(a11) * a21;
            else {
                y1[row1] = a11;
                y2[row1] = a21;
            }
        }
    }
}

__global__ void __launch_bounds__(512, 2)
gemv_fp8_2row_dual_st_k(const uint8_t* W1, const float* s1, const uint8_t* W2, const float* s2, const float* x,
                        float* y1, float* y2, int m, int n, int block, int fuse_swiglu, const float* x_gamma,
                        const float* d_ss, float rms_eps) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb_n = n / block;
    float a10 = 0.f, a11 = 0.f, a20 = 0.f, a21 = 0.f;
    const float* sc10 = (row0 < m && s1) ? s1 + (row0 / block) * nb_n : nullptr;
    const float* sc20 = (row0 < m && s2) ? s2 + (row0 / block) * nb_n : nullptr;
    load_x_tile(xs, x, n);
    if (x_gamma) {
        __syncthreads();
        scale_x_rms(xs, x_gamma, 0, n, n, d_ss, rms_eps);
    }
    __syncthreads();
    const int live1 = (row1 < m) ? 1 : 0;
    if (sc10) acc_fp8_2row_al(a10, a11, W1, sc10, row0, row1, m, xs, n, lane, live1);
    if (sc20) acc_fp8_2row_al(a20, a21, W2, sc20, row0, row1, m, xs, n, lane, live1);
    if (row0 < m) {
        a10 = warp_sum(a10);
        a20 = warp_sum(a20);
        if (lane == 0) {
            if (fuse_swiglu) y1[row0] = silu_d(a10) * a20;
            else {
                y1[row0] = a10;
                y2[row0] = a20;
            }
        }
    }
    if (row1 < m) {
        a11 = warp_sum(a11);
        a21 = warp_sum(a21);
        if (lane == 0) {
            if (fuse_swiglu) y1[row1] = silu_d(a11) * a21;
            else {
                y1[row1] = a11;
                y2[row1] = a21;
            }
        }
    }
}

// Tiny leftover dual (GDN wa/wb, ~48 rows). Isolated so the big pair loop stays 4 accs.
// reuse_xs: caller already left the (RMS-scaled) full vector in xs (n <= tile).
__device__ __noinline__ void gemv_fp8_leftover_dual(const uint8_t* W3, const float* s3, float* y3, int m3,
                                                    const uint8_t* W4, const float* s4, float* y4, int m4,
                                                    const float* x, int n, int block, const float* x_gamma,
                                                    const float* d_ss, float rms_eps, int reuse_xs) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int tile = n <= kFp8XsCap ? n : kFp8XsCap;
    const int nb_n = (n + block - 1) / block;
    const int mmax = m3 > m4 ? m3 : m4;
    float a3[4] = {0.f, 0.f, 0.f, 0.f};
    float a4[4] = {0.f, 0.f, 0.f, 0.f};
    int mine[4] = {-1, -1, -1, -1};
    int nmine = 0;
    for (int row = warp; row < mmax && nmine < 4; row += warps) mine[nmine++] = row;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        if (!reuse_xs) {
            load_x_tile(xs, x + t0, tn);
            if (x_gamma && (d_ss || tn == n)) {
                __syncthreads();
                scale_x_rms(xs, x_gamma, t0, tn, n, d_ss, rms_eps);
            }
            __syncthreads();
        }
        const int b0 = t0 / block;
        const int nb = tn / block;
        for (int r = 0; r < nmine; ++r) {
            const int row = mine[r];
            const float* sc3 = (row < m3 && s3) ? s3 + (row / block) * nb_n : nullptr;
            const float* sc4 = (row < m4 && s4) ? s4 + (row / block) * nb_n : nullptr;
            for (int b = 0; b < nb; ++b) {
                const float* xb = xs + b * block + lane * 4;
                if (sc3) a3[r] += sc3[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(W3, row, m3, b0 + b, lane)), xb);
                if (sc4) a4[r] += sc4[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(W4, row, m4, b0 + b, lane)), xb);
            }
        }
        if (t0 + tile < n) __syncthreads();
    }
    for (int r = 0; r < nmine; ++r) {
        const int row = mine[r];
        if (row < m3) {
            a3[r] = warp_sum(a3[r]);
            if (lane == 0) y3[row] = a3[r];
        }
        if (row < m4) {
            a4[r] = warp_sum(a4[r]);
            if (lane == 0) y4[row] = a4[r];
        }
    }
}

// Two FP8 GEMVs, same K, possibly different M (GDN wqkv + wz). One x tile.
// Optional W3/W4 (wa/wb) computed in block 0 only so the main loop stays 4 accs.
__global__ void __launch_bounds__(512, 2)
gemv_fp8_2row_pair_k(const uint8_t* W1, const float* s1, const uint8_t* W2, const float* s2, const float* x,
                     float* y1, float* y2, int m1, int m2, int n, int block, const uint8_t* W3, const float* s3,
                     float* y3, int m3, const uint8_t* W4, const float* s4, float* y4, int m4,
                     const float* x_gamma, const float* d_ss, float rms_eps) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int tile = n <= kFp8XsCap ? n : kFp8XsCap;
    const int nb_n = (n + block - 1) / block;
    const int mmax = m1 > m2 ? m1 : m2;
    float a10 = 0.f, a11 = 0.f, a20 = 0.f, a21 = 0.f;
    const float* sc10 = (row0 < m1 && s1) ? s1 + (row0 / block) * nb_n : nullptr;
    const float* sc11 = (row1 < m1 && s1) ? s1 + (row1 / block) * nb_n : nullptr;
    const float* sc20 = (row0 < m2 && s2) ? s2 + (row0 / block) * nb_n : nullptr;
    const float* sc21 = (row1 < m2 && s2) ? s2 + (row1 / block) * nb_n : nullptr;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        load_x_tile(xs, x + t0, tn);
        if (x_gamma && (d_ss || tn == n)) {
            __syncthreads();
            scale_x_rms(xs, x_gamma, t0, tn, n, d_ss, rms_eps);
        }
        __syncthreads();
        const int b0 = t0 / block;
        const int nb = tn / block;
        int b = 0;
        for (; b + 3 < nb; b += 4) {
            const float* xb = xs + b * block + lane * 4;
            if (sc10)
                a10 += fp8_dot16(fp8_ld16(W1, row0, m1, b0 + b, lane),
                                 __ldg(reinterpret_cast<const float4*>(sc10 + b0 + b)), xb);
            if (sc20)
                a20 += fp8_dot16(fp8_ld16(W2, row0, m2, b0 + b, lane),
                                 __ldg(reinterpret_cast<const float4*>(sc20 + b0 + b)), xb);
            if (sc11)
                a11 += fp8_dot16(fp8_ld16(W1, row1, m1, b0 + b, lane),
                                 __ldg(reinterpret_cast<const float4*>(sc11 + b0 + b)), xb);
            if (sc21)
                a21 += fp8_dot16(fp8_ld16(W2, row1, m2, b0 + b, lane),
                                 __ldg(reinterpret_cast<const float4*>(sc21 + b0 + b)), xb);
        }
        for (; b < nb; ++b) {
            const float* xb = xs + b * block + lane * 4;
            if (sc10) a10 += sc10[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(W1, row0, m1, b0 + b, lane)), xb);
            if (sc20) a20 += sc20[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(W2, row0, m2, b0 + b, lane)), xb);
            if (sc11) a11 += sc11[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(W1, row1, m1, b0 + b, lane)), xb);
            if (sc21) a21 += sc21[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(W2, row1, m2, b0 + b, lane)), xb);
        }
        if (t0 + tile < n) __syncthreads();
    }
    (void)mmax;
    if (row0 < m1) {
        a10 = warp_sum(a10);
        if (lane == 0) y1[row0] = a10;
    }
    if (row0 < m2) {
        a20 = warp_sum(a20);
        if (lane == 0) y2[row0] = a20;
    }
    if (row1 < m1) {
        a11 = warp_sum(a11);
        if (lane == 0) y1[row1] = a11;
    }
    if (row1 < m2) {
        a21 = warp_sum(a21);
        if (lane == 0) y2[row1] = a21;
    }
    if (W3 && m3 > 0 && blockIdx.x == 0) {
        __syncthreads();
        gemv_fp8_leftover_dual(W3, s3, y3, m3, W4, s4, y4, m4, x, n, block, x_gamma, d_ss, rms_eps,
                               n <= tile);
    }
}

__global__ void __launch_bounds__(512, 2)
gemv_fp8_2row_pair_st_k(const uint8_t* W1, const float* s1, const uint8_t* W2, const float* s2, const float* x,
                        float* y1, float* y2, int m1, int m2, int n, int block, const uint8_t* W3, const float* s3,
                        float* y3, int m3, const uint8_t* W4, const float* s4, float* y4, int m4,
                        const float* x_gamma, const float* d_ss, float rms_eps) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb_n = n / block;
    float a10 = 0.f, a11 = 0.f, a20 = 0.f, a21 = 0.f;
    const float* sc10 = (row0 < m1 && s1) ? s1 + (row0 / block) * nb_n : nullptr;
    const float* sc20 = (row0 < m2 && s2) ? s2 + (row0 / block) * nb_n : nullptr;
    load_x_tile(xs, x, n);
    if (x_gamma) {
        __syncthreads();
        scale_x_rms(xs, x_gamma, 0, n, n, d_ss, rms_eps);
    }
    __syncthreads();
    if (sc10) acc_fp8_2row_al(a10, a11, W1, sc10, row0, row1, m1, xs, n, lane, (row1 < m1) ? 1 : 0);
    if (sc20) acc_fp8_2row_al(a20, a21, W2, sc20, row0, row1, m2, xs, n, lane, (row1 < m2) ? 1 : 0);
    if (row0 < m1) {
        a10 = warp_sum(a10);
        if (lane == 0) y1[row0] = a10;
    }
    if (row0 < m2) {
        a20 = warp_sum(a20);
        if (lane == 0) y2[row0] = a20;
    }
    if (row1 < m1) {
        a11 = warp_sum(a11);
        if (lane == 0) y1[row1] = a11;
    }
    if (row1 < m2) {
        a21 = warp_sum(a21);
        if (lane == 0) y2[row1] = a21;
    }
    if (W3 && m3 > 0 && blockIdx.x == 0) {
        __syncthreads();
        gemv_fp8_leftover_dual(W3, s3, y3, m3, W4, s4, y4, m4, x, n, block, x_gamma, d_ss, rms_eps, 1);
    }
}

// Attn in-proj: Wq (split q|gate) + Wk + Wv, one x tile + optional fused RMS.
__global__ void __launch_bounds__(512, 2)
gemv_fp8_2row_attn_in_k(const uint8_t* Wq, const float* sq, const uint8_t* Wk, const float* sk,
                        const uint8_t* Wv, const float* sv, const float* x, float* q, float* gate,
                        float* k, float* v, int mq, int mk, int mv, int n, int block, int split_at,
                        const float* x_gamma, const float* d_ss, float rms_eps) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int tile = n <= kFp8XsCap ? n : kFp8XsCap;
    const int nb_n = (n + block - 1) / block;
    float aq0 = 0.f, aq1 = 0.f, ak0 = 0.f, ak1 = 0.f, av0 = 0.f, av1 = 0.f;
    const float* scq0 = (row0 < mq && sq) ? sq + (row0 / block) * nb_n : nullptr;
    const float* scq1 = (row1 < mq && sq) ? sq + (row1 / block) * nb_n : nullptr;
    const float* sck0 = (row0 < mk && sk) ? sk + (row0 / block) * nb_n : nullptr;
    const float* sck1 = (row1 < mk && sk) ? sk + (row1 / block) * nb_n : nullptr;
    const float* scv0 = (row0 < mv && sv) ? sv + (row0 / block) * nb_n : nullptr;
    const float* scv1 = (row1 < mv && sv) ? sv + (row1 / block) * nb_n : nullptr;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        load_x_tile(xs, x + t0, tn);
        if (x_gamma && (d_ss || tn == n)) {
            __syncthreads();
            scale_x_rms(xs, x_gamma, t0, tn, n, d_ss, rms_eps);
        }
        __syncthreads();
        const int b0 = t0 / block;
        const int nb = tn / block;
        for (int b = 0; b < nb; ++b) {
            const float* xb = xs + b * block + lane * 4;
            if (scq0) aq0 += scq0[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(Wq, row0, mq, b0 + b, lane)), xb);
            if (scq1) aq1 += scq1[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(Wq, row1, mq, b0 + b, lane)), xb);
            if (sck0) ak0 += sck0[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(Wk, row0, mk, b0 + b, lane)), xb);
            if (sck1) ak1 += sck1[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(Wk, row1, mk, b0 + b, lane)), xb);
            if (scv0) av0 += scv0[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(Wv, row0, mv, b0 + b, lane)), xb);
            if (scv1) av1 += scv1[b0 + b] * fp8x4_dot(__ldcs(fp8_blk4(Wv, row1, mv, b0 + b, lane)), xb);
        }
        if (t0 + tile < n) __syncthreads();
    }
    if (row0 < mq) {
        aq0 = warp_sum(aq0);
        if (lane == 0) {
            if (split_at > 0 && row0 >= split_at) gate[row0 - split_at] = aq0;
            else q[row0] = aq0;
        }
    }
    if (row1 < mq) {
        aq1 = warp_sum(aq1);
        if (lane == 0) {
            if (split_at > 0 && row1 >= split_at) gate[row1 - split_at] = aq1;
            else q[row1] = aq1;
        }
    }
    if (row0 < mk) {
        ak0 = warp_sum(ak0);
        if (lane == 0) k[row0] = ak0;
    }
    if (row1 < mk) {
        ak1 = warp_sum(ak1);
        if (lane == 0) k[row1] = ak1;
    }
    if (row0 < mv) {
        av0 = warp_sum(av0);
        if (lane == 0) v[row0] = av0;
    }
    if (row1 < mv) {
        av1 = warp_sum(av1);
        if (lane == 0) v[row1] = av1;
    }
}
__global__ void gemm_fp8_hw_k(const uint8_t* W, const float* scale, const float* X, float* Y, int m, int n,
                              int T, int tile, int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb_n = (n + 127) / 128;
    const int bi = row / 128;
    const int tstride = tile;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        for (int i = threadIdx.x; i < T * tn; i += blockDim.x) {
            const int t = i / tn;
            const int j = i - t * tn;
            xs[t * tstride + j] = X[static_cast<size_t>(t) * n + t0 + j];
        }
        __syncthreads();
        if (row < m && scale) {
            const int b0 = t0 / 128;
            const int nb = tn / 128;
            int b = 0;
            for (; b + 3 < nb; b += 4) {
                const float4 sv = __ldg(reinterpret_cast<const float4*>(scale + bi * nb_n + b0 + b));
                const uint4 p = fp8_ld16(W, row, m, b0 + b, lane);
                const int o0 = b * 128 + lane * 4;
                acc0 += fp8_dot16(p, sv, xs + o0);
                if (T > 1) acc1 += fp8_dot16(p, sv, xs + tstride + o0);
                if (T > 2) acc2 += fp8_dot16(p, sv, xs + 2 * tstride + o0);
                if (T > 3) acc3 += fp8_dot16(p, sv, xs + 3 * tstride + o0);
            }
            for (; b < nb; ++b) {
                const float s = scale[bi * nb_n + b0 + b];
                const uint32_t p = __ldcs(fp8_blk4(W, row, m, b0 + b, lane));
                const int o = b * 128 + lane * 4;
                acc0 += s * fp8x4_dot(p, xs + o);
                if (T > 1) acc1 += s * fp8x4_dot(p, xs + tstride + o);
                if (T > 2) acc2 += s * fp8x4_dot(p, xs + 2 * tstride + o);
                if (T > 3) acc3 += s * fp8x4_dot(p, xs + 3 * tstride + o);
            }
        }
        __syncthreads();
    }
    if (row < m) {
        float a0 = warp_sum(acc0);
        if (lane == 0) write_y(Y, row, a0, add);
        if (T > 1) {
            float a1 = warp_sum(acc1);
            if (lane == 0) write_y(Y + m, row, a1, add);
        }
        if (T > 2) {
            float a2 = warp_sum(acc2);
            if (lane == 0) write_y(Y + 2 * m, row, a2, add);
        }
        if (T > 3) {
            float a3 = warp_sum(acc3);
            if (lane == 0) write_y(Y + 3 * m, row, a3, add);
        }
    }
}

// T=3 bakeoff prefill: no T-branches, float4 x load, same 16-warp 1-row grid as T=1 GEMV.
__global__ void __launch_bounds__(512, 2)
gemm_fp8_t3_k(const uint8_t* W, const float* scale, const float* X, float* Y, int m, int n, int tile,
              int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb_n = (n + 127) / 128;
    const int bi = row / 128;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        load_x_tile(xs, X + t0, tn);
        load_x_tile(xs + tile, X + n + t0, tn);
        load_x_tile(xs + 2 * tile, X + 2 * n + t0, tn);
        __syncthreads();
        if (row < m && scale) {
            const int b0 = t0 / 128;
            const int nb = tn / 128;
            int b = 0;
            for (; b + 3 < nb; b += 4) {
                const float4 sv = __ldg(reinterpret_cast<const float4*>(scale + bi * nb_n + b0 + b));
                const uint4 p = fp8_ld16(W, row, m, b0 + b, lane);
                const int o0 = b * 128 + lane * 4;
                acc0 += fp8_dot16(p, sv, xs + o0);
                acc1 += fp8_dot16(p, sv, xs + tile + o0);
                acc2 += fp8_dot16(p, sv, xs + 2 * tile + o0);
            }
            for (; b < nb; ++b) {
                const float s = __ldg(scale + bi * nb_n + b0 + b);
                const uint32_t p = __ldcs(fp8_blk4(W, row, m, b0 + b, lane));
                const int o = b * 128 + lane * 4;
                acc0 += s * fp8x4_dot(p, xs + o);
                acc1 += s * fp8x4_dot(p, xs + tile + o);
                acc2 += s * fp8x4_dot(p, xs + 2 * tile + o);
            }
        }
        __syncthreads();
    }
    if (row < m) {
        float a0 = warp_sum(acc0);
        if (lane == 0) write_y(Y, row, a0, add);
        float a1 = warp_sum(acc1);
        if (lane == 0) write_y(Y + m, row, a1, add);
        float a2 = warp_sum(acc2);
        if (lane == 0) write_y(Y + 2 * m, row, a2, add);
    }
}

// T=3 dual: one x stream, two FP8 mats (mlp.gate+up or GDN wa+wb).
__global__ void gemm_fp8_t3_dual_k(const uint8_t* W1, const float* s1, const uint8_t* W2, const float* s2,
                                   const float* X, float* Y1, float* Y2, int m, int n, int tile,
                                   int fuse_swiglu) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb_n = (n + 127) / 128;
    const int bi = row / 128;
    float a10 = 0.f, a11 = 0.f, a12 = 0.f, a20 = 0.f, a21 = 0.f, a22 = 0.f;
    const float* sc1 = (row < m && s1) ? s1 + bi * nb_n : nullptr;
    const float* sc2 = (row < m && s2) ? s2 + bi * nb_n : nullptr;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        load_x_tile(xs, X + t0, tn);
        load_x_tile(xs + tile, X + n + t0, tn);
        load_x_tile(xs + 2 * tile, X + 2 * n + t0, tn);
        __syncthreads();
        if (row < m && sc1 && sc2) {
            const int b0 = t0 / 128;
            const int nb = tn / 128;
            for (int b = 0; b < nb; ++b) {
                const float u = __ldg(sc1 + b0 + b);
                const float v = __ldg(sc2 + b0 + b);
                const uint32_t p1 = __ldcs(fp8_blk4(W1, row, m, b0 + b, lane));
                const uint32_t p2 = __ldcs(fp8_blk4(W2, row, m, b0 + b, lane));
                const int o = b * 128 + lane * 4;
                a10 += u * fp8x4_dot(p1, xs + o);
                a11 += u * fp8x4_dot(p1, xs + tile + o);
                a12 += u * fp8x4_dot(p1, xs + 2 * tile + o);
                a20 += v * fp8x4_dot(p2, xs + o);
                a21 += v * fp8x4_dot(p2, xs + tile + o);
                a22 += v * fp8x4_dot(p2, xs + 2 * tile + o);
            }
        }
        __syncthreads();
    }
    if (row < m) {
        a10 = warp_sum(a10);
        a20 = warp_sum(a20);
        a11 = warp_sum(a11);
        a21 = warp_sum(a21);
        a12 = warp_sum(a12);
        a22 = warp_sum(a22);
        if (lane == 0) {
            if (fuse_swiglu) {
                Y1[row] = silu_d(a10) * a20;
                Y1[row + m] = silu_d(a11) * a21;
                Y1[row + 2 * m] = silu_d(a12) * a22;
            } else {
                Y1[row] = a10;
                Y2[row] = a20;
                Y1[row + m] = a11;
                Y2[row + m] = a21;
                Y1[row + 2 * m] = a12;
                Y2[row + 2 * m] = a22;
            }
        }
    }
}

__global__ void __launch_bounds__(512, 2)
gemm_fp8_t4_k(const uint8_t* W, const float* scale, const float* X, float* Y, int m, int n, int tile,
              int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb_n = (n + 127) / 128;
    const int bi = row / 128;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        load_x_tile(xs, X + t0, tn);
        load_x_tile(xs + tile, X + n + t0, tn);
        load_x_tile(xs + 2 * tile, X + 2 * n + t0, tn);
        load_x_tile(xs + 3 * tile, X + 3 * n + t0, tn);
        __syncthreads();
        if (row < m && scale) {
            const int b0 = t0 / 128;
            const int nb = tn / 128;
            int b = 0;
            for (; b + 3 < nb; b += 4) {
                const float4 sv = __ldg(reinterpret_cast<const float4*>(scale + bi * nb_n + b0 + b));
                const uint4 p = fp8_ld16(W, row, m, b0 + b, lane);
                const int o0 = b * 128 + lane * 4;
                acc0 += fp8_dot16(p, sv, xs + o0);
                acc1 += fp8_dot16(p, sv, xs + tile + o0);
                acc2 += fp8_dot16(p, sv, xs + 2 * tile + o0);
                acc3 += fp8_dot16(p, sv, xs + 3 * tile + o0);
            }
            for (; b < nb; ++b) {
                const float s = __ldg(scale + bi * nb_n + b0 + b);
                const uint32_t p = __ldcs(fp8_blk4(W, row, m, b0 + b, lane));
                const int o = b * 128 + lane * 4;
                acc0 += s * fp8x4_dot(p, xs + o);
                acc1 += s * fp8x4_dot(p, xs + tile + o);
                acc2 += s * fp8x4_dot(p, xs + 2 * tile + o);
                acc3 += s * fp8x4_dot(p, xs + 3 * tile + o);
            }
        }
        __syncthreads();
    }
    if (row < m) {
        float a0 = warp_sum(acc0);
        if (lane == 0) write_y(Y, row, a0, add);
        float a1 = warp_sum(acc1);
        if (lane == 0) write_y(Y + m, row, a1, add);
        float a2 = warp_sum(acc2);
        if (lane == 0) write_y(Y + 2 * m, row, a2, add);
        float a3 = warp_sum(acc3);
        if (lane == 0) write_y(Y + 3 * m, row, a3, add);
    }
}

__global__ void gemm_fp8_2row_k(const uint8_t* W, const float* scale, const float* X, float* Y, int m, int n,
                                int T, int tile, int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb_n = (n + 127) / 128;
    const int tstride = tile;
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a03 = 0.f;
    float a10 = 0.f, a11 = 0.f, a12 = 0.f, a13 = 0.f;
    const uint8_t* r0 = row0 < m ? W + static_cast<size_t>(row0) * n : nullptr;
    const uint8_t* r1 = row1 < m ? W + static_cast<size_t>(row1) * n : nullptr;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        for (int i = threadIdx.x; i < T * tn; i += blockDim.x) {
            const int t = i / tn;
            const int j = i - t * tn;
            xs[t * tstride + j] = X[static_cast<size_t>(t) * n + t0 + j];
        }
        __syncthreads();
        const int b0 = t0 / 128;
        const int nb = tn / 128;
        if (r0) {
            const int bi0 = row0 / 128;
            for (int b = 0; b < nb; ++b) {
                const float s = scale[bi0 * nb_n + b0 + b];
                const uint32_t p = __ldcs(fp8_blk4(W, row0, m, b0 + b, lane));
                const int o = b * 128 + lane * 4;
                a00 += s * fp8x4_dot(p, xs + o);
                if (T > 1) a01 += s * fp8x4_dot(p, xs + tstride + o);
                if (T > 2) a02 += s * fp8x4_dot(p, xs + 2 * tstride + o);
                if (T > 3) a03 += s * fp8x4_dot(p, xs + 3 * tstride + o);
            }
        }
        if (r1) {
            const int bi1 = row1 / 128;
            for (int b = 0; b < nb; ++b) {
                const float s = scale[bi1 * nb_n + b0 + b];
                const uint32_t p = __ldcs(fp8_blk4(W, row1, m, b0 + b, lane));
                const int o = b * 128 + lane * 4;
                a10 += s * fp8x4_dot(p, xs + o);
                if (T > 1) a11 += s * fp8x4_dot(p, xs + tstride + o);
                if (T > 2) a12 += s * fp8x4_dot(p, xs + 2 * tstride + o);
                if (T > 3) a13 += s * fp8x4_dot(p, xs + 3 * tstride + o);
            }
        }
        __syncthreads();
    }
    if (row0 < m) {
        float v0 = warp_sum(a00);
        if (lane == 0) write_y(Y, row0, v0, add);
        if (T > 1) {
            float v1 = warp_sum(a01);
            if (lane == 0) write_y(Y + m, row0, v1, add);
        }
        if (T > 2) {
            float v2 = warp_sum(a02);
            if (lane == 0) write_y(Y + 2 * m, row0, v2, add);
        }
        if (T > 3) {
            float v3 = warp_sum(a03);
            if (lane == 0) write_y(Y + 3 * m, row0, v3, add);
        }
    }
    if (row1 < m) {
        float v0 = warp_sum(a10);
        if (lane == 0) write_y(Y, row1, v0, add);
        if (T > 1) {
            float v1 = warp_sum(a11);
            if (lane == 0) write_y(Y + m, row1, v1, add);
        }
        if (T > 2) {
            float v2 = warp_sum(a12);
            if (lane == 0) write_y(Y + 2 * m, row1, v2, add);
        }
        if (T > 3) {
            float v3 = warp_sum(a13);
            if (lane == 0) write_y(Y + 3 * m, row1, v3, add);
        }
    }
}

__global__ void rmsnorm_k(const float* x, const float* gamma, float* y, int n, float eps) {
    __shared__ float buf[256];
    float ss = 0.f;
    const int n4 = n >> 2;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = reinterpret_cast<const float4*>(x)[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int i = (n4 << 2) + threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    buf[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(buf[0] / static_cast<float>(n) + eps);
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = reinterpret_cast<const float4*>(x)[i];
        float4 g = {1.f, 1.f, 1.f, 1.f};
        if (gamma) {
            const float4 gg = reinterpret_cast<const float4*>(gamma)[i];
            g.x += gg.x;
            g.y += gg.y;
            g.z += gg.z;
            g.w += gg.w;
        }
        float4 o;
        o.x = v.x * inv * g.x;
        o.y = v.y * inv * g.y;
        o.z = v.z * inv * g.z;
        o.w = v.w * inv * g.w;
        reinterpret_cast<float4*>(y)[i] = o;
    }
    for (int i = (n4 << 2) + threadIdx.x; i < n; i += blockDim.x) {
        const float g = gamma ? (1.f + gamma[i]) : 1.f;
        y[i] = x[i] * inv * g;
    }
}

// One-block write of sum(x^2). No atomics. Next GEMV applies RMS while loading x.
__global__ void reduce_ss_k(const float* x, float* d_ss, int n) {
    __shared__ float buf[256];
    float ss = 0.f;
    const int n4 = n >> 2;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = reinterpret_cast<const float4*>(x)[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int i = (n4 << 2) + threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    buf[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) *d_ss = buf[0];
}

// RMS using a precomputed sum-of-squares (from residual GEMV atomics).
__global__ void apply_rms_ss_k(const float* x, const float* gamma, const float* d_ss, float* y, int n, float eps) {
    const float inv = rsqrtf((*d_ss) / static_cast<float>(n) + eps);
    const int n4 = n >> 2;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = reinterpret_cast<const float4*>(x)[i];
        float4 g = {1.f, 1.f, 1.f, 1.f};
        if (gamma) {
            const float4 gg = reinterpret_cast<const float4*>(gamma)[i];
            g.x += gg.x;
            g.y += gg.y;
            g.z += gg.z;
            g.w += gg.w;
        }
        float4 o;
        o.x = v.x * inv * g.x;
        o.y = v.y * inv * g.y;
        o.z = v.z * inv * g.z;
        o.w = v.w * inv * g.w;
        reinterpret_cast<float4*>(y)[i] = o;
    }
    for (int i = (n4 << 2) + threadIdx.x; i < n; i += blockDim.x) {
        const float g = gamma ? (1.f + gamma[i]) : 1.f;
        y[i] = x[i] * inv * g;
    }
}

__global__ void add_res_k(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ void swiglu_k(const float* gate, const float* up, float* y, int n) {
    const int i4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i4 + 3 < n) {
        const float4 g = *reinterpret_cast<const float4*>(gate + i4);
        const float4 u = *reinterpret_cast<const float4*>(up + i4);
        float4 o;
        o.x = silu_d(g.x) * u.x;
        o.y = silu_d(g.y) * u.y;
        o.z = silu_d(g.z) * u.z;
        o.w = silu_d(g.w) * u.w;
        *reinterpret_cast<float4*>(y + i4) = o;
        return;
    }
    for (int i = i4; i < n; ++i) y[i] = silu_d(gate[i]) * up[i];
}

__global__ void gated_rms_k(const float* x, const float* z, const float* gamma, float* y, int n, float eps,
                            int gamma_n) {
    __shared__ float buf[256];
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    buf[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(buf[0] / static_cast<float>(n) + eps);
    if (gamma_n <= 0) gamma_n = n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float g = gamma ? gamma[i % gamma_n] : 1.f;
        y[i] = g * (x[i] * inv) * silu_d(z[i]);
    }
}

__device__ __forceinline__ float conv1d_mix(const float* x_t, const float* w, const float* state, int c, int k) {
    const float* st = state + c * k;
    const float* wc = w + c * k;
    if (k == 4) return silu_d(wc[0] * st[1] + wc[1] * st[2] + wc[2] * st[3] + wc[3] * x_t[c]);
    float acc = 0.f;
    for (int p = 0; p < k - 1; ++p) acc += wc[p] * st[p + 1];
    acc += wc[k - 1] * x_t[c];
    return silu_d(acc);
}

__device__ __forceinline__ void conv1d_push(const float* x_t, float* state, int c, int k) {
    float* st = state + c * k;
    if (k == 4) {
        st[0] = st[1];
        st[1] = st[2];
        st[2] = st[3];
        st[3] = x_t[c];
        return;
    }
    for (int p = 0; p < k - 1; ++p) st[p] = st[p + 1];
    st[k - 1] = x_t[c];
}

__global__ void conv1d_upd_k(const float* x_t, const float* w, float* state, float* y, int dim, int k) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= dim) return;
    y[c] = conv1d_mix(x_t, w, state, c, k);
    conv1d_push(x_t, state, c, k);
}

__global__ void expand_qkv_k(const float* mix, float* qh, float* kh, float* vh, float* beta, float* glog,
                             const float* aa, const float* bb, const float* A_log, const float* dt_bias, int nk,
                             int nv, int dk, int dv) {
    const int h = blockIdx.x;
    if (h >= nv) return;
    const int qdim = nk * dk;
    const int rep = nv / nk;
    const int src = h / rep;
    const float* Q = mix;
    const float* K = mix + qdim;
    const float* V = mix + 2 * qdim;
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        qh[h * dk + i] = Q[src * dk + i];
        kh[h * dk + i] = K[src * dk + i];
    }
    for (int i = threadIdx.x; i < dv; i += blockDim.x) vh[h * dv + i] = V[h * dv + i];
    if (threadIdx.x == 0) {
        beta[h] = sigmoid_d(bb[h]);
        glog[h] = -expf(A_log[h]) * softplus_d(aa[h] + dt_bias[h]);
    }
}

__global__ void delta_step_k(float* S, const float* q_in, const float* k_in, const float* v, const float* beta,
                             const float* g_log, float* o, int n_v, int dk, int dv, float eps_l2) {
    const int h = blockIdx.x;
    if (h >= n_v) return;
    extern __shared__ float sm[];
    float* q = sm;
    float* k = sm + dk;
    float* kv = sm + 2 * dk;
    float* delta = sm + 2 * dk + dv;

    float qss = 0.f, kss = 0.f;
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        const float qi = q_in[h * dk + i];
        const float ki = k_in[h * dk + i];
        q[i] = qi;
        k[i] = ki;
        qss += qi * qi;
        kss += ki * ki;
    }
    __shared__ float qbuf[128], kbuf[128];
    qbuf[threadIdx.x] = qss;
    kbuf[threadIdx.x] = kss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            qbuf[threadIdx.x] += qbuf[threadIdx.x + s];
            kbuf[threadIdx.x] += kbuf[threadIdx.x + s];
        }
        __syncthreads();
    }
    __shared__ float qinv, kinv;
    if (threadIdx.x == 0) {
        qinv = rsqrtf(qbuf[0] + eps_l2) * rsqrtf(static_cast<float>(dk));
        kinv = rsqrtf(kbuf[0] + eps_l2);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        q[i] *= qinv;
        k[i] *= kinv;
    }
    __syncthreads();

    float* Sh = S + static_cast<size_t>(h) * dk * dv;
    const float g = expf(g_log[h]);
    for (int t = threadIdx.x; t < dk * dv; t += blockDim.x) Sh[t] *= g;
    __syncthreads();

    for (int j = threadIdx.x; j < dv; j += blockDim.x) kv[j] = 0.f;
    __syncthreads();
    for (int i = 0; i < dk; ++i) {
        const float ki = k[i];
        const float* srow = Sh + i * dv;
        for (int j = threadIdx.x; j < dv; j += blockDim.x) kv[j] += srow[j] * ki;
    }
    __syncthreads();
    const float b = beta[h];
    for (int j = threadIdx.x; j < dv; j += blockDim.x) delta[j] = (v[h * dv + j] - kv[j]) * b;
    __syncthreads();
    for (int i = 0; i < dk; ++i) {
        float* srow = Sh + i * dv;
        const float ki = k[i];
        for (int j = threadIdx.x; j < dv; j += blockDim.x) srow[j] += ki * delta[j];
    }
    __syncthreads();
    for (int j = threadIdx.x; j < dv; j += blockDim.x) o[h * dv + j] = 0.f;
    __syncthreads();
    for (int i = 0; i < dk; ++i) {
        const float qi = q[i];
        const float* srow = Sh + i * dv;
        for (int j = threadIdx.x; j < dv; j += blockDim.x) o[h * dv + j] += srow[j] * qi;
    }
}

// One launch per GDN layer: expand + delta-rule for all T tokens (S is recurrent).
// When dk*dv <= 16384 the state lives in smem (one load/store of S, fused 2-pass).
__global__ void gdn_prefill_steps_k(const float* mix_seq, const float* aa_seq, const float* bb_seq, uint16_t* S,
                                    const float* A_log, const float* dt_bias, float* o_seq, int T, int nk,
                                    int nv, int dk, int dv, int qkv_dim, float eps_l2, const float* qkv_raw,
                                    const float* conv_w, float* conv_st, int conv_k) {
    const int h = blockIdx.x;
    if (h >= nv || dk <= 0 || dv <= 0) return;
    extern __shared__ float sm[];
    float* q = sm;
    float* k = sm + dk;
    const bool s_smem = (dk * dv <= 16384);
    float* Shs = sm + 2 * dk;
    const int qdim = nk * dk;
    const int rep = nv / nk;
    const int src = h / rep;
    const bool fuse_conv = (T == 1 && qkv_raw && conv_w && conv_st && conv_k > 0);
    uint16_t* Sh = S + static_cast<size_t>(h) * dk * dv;
    __shared__ float qbuf[128], kbuf[128], qinv, kinv, beta_h, glog_h;

    if (s_smem) {
        gdn_s_load_f32(Shs, Sh, dk * dv);
        __syncthreads();
    }
    float* Sp = s_smem ? Shs : nullptr;

    for (int t = 0; t < T; ++t) {
        const float* mix = mix_seq ? mix_seq + static_cast<size_t>(t) * qkv_dim : nullptr;
        const float* aa = aa_seq + static_cast<size_t>(t) * nv;
        const float* bb = bb_seq + static_cast<size_t>(t) * nv;
        const float* Q = mix;
        const float* K = mix ? mix + qdim : nullptr;
        const float* V = mix ? mix + 2 * qdim : nullptr;
        float qss = 0.f, kss = 0.f;
        for (int i = threadIdx.x; i < dk; i += blockDim.x) {
            const float qi = fuse_conv ? conv1d_mix(qkv_raw, conv_w, conv_st, src * dk + i, conv_k)
                                       : Q[src * dk + i];
            const float ki = fuse_conv ? conv1d_mix(qkv_raw, conv_w, conv_st, qdim + src * dk + i, conv_k)
                                       : K[src * dk + i];
            q[i] = qi;
            k[i] = ki;
            qss += qi * qi;
            kss += ki * ki;
        }
        qss = warp_sum(qss);
        kss = warp_sum(kss);
        if ((threadIdx.x & 31) == 0) {
            qbuf[threadIdx.x >> 5] = qss;
            kbuf[threadIdx.x >> 5] = kss;
        }
        if (threadIdx.x == 0) {
            beta_h = sigmoid_d(bb[h]);
            glog_h = -expf(A_log[h]) * softplus_d(aa[h] + dt_bias[h]);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float qt = 0.f, kt = 0.f;
            const int nw = blockDim.x >> 5;
            for (int i = 0; i < nw; ++i) {
                qt += qbuf[i];
                kt += kbuf[i];
            }
            qinv = rsqrtf(qt + eps_l2) * rsqrtf(static_cast<float>(dk));
            kinv = rsqrtf(kt + eps_l2);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < dk; i += blockDim.x) {
            q[i] *= qinv;
            k[i] *= kinv;
        }
        __syncthreads();
        const float g = expf(glog_h);
        const float b = beta_h;
        // float4 only when every thread owns a vector; otherwise 128 threads on dv=128
        // would leave 96 idle (old j4 < dv/4 mapping). Global S is BF16; smem is FP32.
        if (s_smem && (dv & 3) == 0 && blockDim.x <= dv / 4) {
            for (int j4 = threadIdx.x; j4 < dv / 4; j4 += blockDim.x) {
                const int j = j4 * 4;
                float kv0 = 0.f, kv1 = 0.f, kv2 = 0.f, kv3 = 0.f;
#pragma unroll 8
                for (int i = 0; i < dk; ++i) {
                    float4 s = *reinterpret_cast<float4*>(Sp + i * dv + j);
                    s.x *= g;
                    s.y *= g;
                    s.z *= g;
                    s.w *= g;
                    *reinterpret_cast<float4*>(Sp + i * dv + j) = s;
                    const float ki = k[i];
                    kv0 += s.x * ki;
                    kv1 += s.y * ki;
                    kv2 += s.z * ki;
                    kv3 += s.w * ki;
                }
                float4 del;
                if (fuse_conv) {
                    const int vc = 2 * qdim + h * dv + j;
                    del.x = (conv1d_mix(qkv_raw, conv_w, conv_st, vc, conv_k) - kv0) * b;
                    del.y = (conv1d_mix(qkv_raw, conv_w, conv_st, vc + 1, conv_k) - kv1) * b;
                    del.z = (conv1d_mix(qkv_raw, conv_w, conv_st, vc + 2, conv_k) - kv2) * b;
                    del.w = (conv1d_mix(qkv_raw, conv_w, conv_st, vc + 3, conv_k) - kv3) * b;
                } else {
                    const float* Vh = V + h * dv + j;
                    del.x = (Vh[0] - kv0) * b;
                    del.y = (Vh[1] - kv1) * b;
                    del.z = (Vh[2] - kv2) * b;
                    del.w = (Vh[3] - kv3) * b;
                }
                float oj0 = 0.f, oj1 = 0.f, oj2 = 0.f, oj3 = 0.f;
#pragma unroll 8
                for (int i = 0; i < dk; ++i) {
                    float4 s = *reinterpret_cast<float4*>(Sp + i * dv + j);
                    const float ki = k[i];
                    s.x += ki * del.x;
                    s.y += ki * del.y;
                    s.z += ki * del.z;
                    s.w += ki * del.w;
                    *reinterpret_cast<float4*>(Sp + i * dv + j) = s;
                    const float qi = q[i];
                    oj0 += s.x * qi;
                    oj1 += s.y * qi;
                    oj2 += s.z * qi;
                    oj3 += s.w * qi;
                }
                float4 oj = {oj0, oj1, oj2, oj3};
                *reinterpret_cast<float4*>(o_seq + static_cast<size_t>(t) * nv * dv + h * dv + j) = oj;
            }
        } else if (s_smem) {
            for (int j = threadIdx.x; j < dv; j += blockDim.x) {
                float kvj = 0.f;
#pragma unroll 8
                for (int i = 0; i < dk; ++i) {
                    const float s = Sp[i * dv + j] * g;
                    Sp[i * dv + j] = s;
                    kvj += s * k[i];
                }
                const float vj = fuse_conv ? conv1d_mix(qkv_raw, conv_w, conv_st, 2 * qdim + h * dv + j, conv_k)
                                           : V[h * dv + j];
                const float del = (vj - kvj) * b;
                float oj = 0.f;
#pragma unroll 8
                for (int i = 0; i < dk; ++i) {
                    const float s = Sp[i * dv + j] + k[i] * del;
                    Sp[i * dv + j] = s;
                    oj += s * q[i];
                }
                o_seq[static_cast<size_t>(t) * nv * dv + h * dv + j] = oj;
            }
        } else {
            for (int j = threadIdx.x; j < dv; j += blockDim.x) {
                float kvj = 0.f;
#pragma unroll 8
                for (int i = 0; i < dk; ++i) {
                    const float s = bf16_to_f32(Sh[i * dv + j]) * g;
                    Sh[i * dv + j] = f32_to_bf16(s);
                    kvj += s * k[i];
                }
                const float vj = fuse_conv ? conv1d_mix(qkv_raw, conv_w, conv_st, 2 * qdim + h * dv + j, conv_k)
                                           : V[h * dv + j];
                const float del = (vj - kvj) * b;
                float oj = 0.f;
#pragma unroll 8
                for (int i = 0; i < dk; ++i) {
                    const float s = bf16_to_f32(Sh[i * dv + j]) + k[i] * del;
                    Sh[i * dv + j] = f32_to_bf16(s);
                    oj += s * q[i];
                }
                o_seq[static_cast<size_t>(t) * nv * dv + h * dv + j] = oj;
            }
        }
        __syncthreads();
    }
    if (fuse_conv) {
        if (h % rep == 0) {
            for (int i = threadIdx.x; i < dk; i += blockDim.x) {
                conv1d_push(qkv_raw, conv_st, src * dk + i, conv_k);
                conv1d_push(qkv_raw, conv_st, qdim + src * dk + i, conv_k);
            }
        }
        for (int i = threadIdx.x; i < dv; i += blockDim.x)
            conv1d_push(qkv_raw, conv_st, 2 * qdim + h * dv + i, conv_k);
    }
    if (s_smem) gdn_s_store_bf16(Sh, Shs, dk * dv);
}

// T=1 decode GDN: dk=dv=128, conv_k=4. Same math as gdn_prefill_steps_k, no T/mix branches.
// pf/pf_bytes: fire-and-forget L2 prefetch of the next GEMV (usually wo, ~21MB).
// Issued after q/k are in smem so the dk·dv loops overlap DRAM fill. No 2nd stream, no join.
// 256 threads: same 128-col compute (tx<128), extra warp helps S BF16
// copy + wo prefetch. Occupancy still 1 block/SM (65KB S). Not the failed
// dv-float4 path that idled 96 of 128 on compute. DRAM S is BF16 (32KB/head).
__global__ void __launch_bounds__(256, 1) gdn_decode_t1_k(const float* aa, const float* bb, uint16_t* S,
                                                          const float* A_log, const float* dt_bias, float* o,
                                                          int nk, int nv, const float* qkv_raw,
                                                          const float* conv_w, float* conv_st,
                                                          const uint8_t* pf, int pf_bytes) {
    constexpr int dk = 128, dv = 128;
    const int h = blockIdx.x;
    if (h >= nv) return;
    extern __shared__ float sm[];
    float* q = sm;
    float* k = sm + dk;
    float* Sp = sm + 2 * dk;
    uint16_t* Sh = S + static_cast<size_t>(h) * dk * dv;
    const int qdim = nk * dk;
    const int rep = nv / nk;
    const int src = h / rep;
    __shared__ float qbuf[8], kbuf[8], qinv, kinv, beta_h, glog_h;

    gdn_s_load_f32(Sp, Sh, dk * dv);

    float qss = 0.f, kss = 0.f;
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        const float qi = conv1d_mix(qkv_raw, conv_w, conv_st, src * dk + i, 4);
        const float ki = conv1d_mix(qkv_raw, conv_w, conv_st, qdim + src * dk + i, 4);
        q[i] = qi;
        k[i] = ki;
        qss += qi * qi;
        kss += ki * ki;
    }
    qss = warp_sum(qss);
    kss = warp_sum(kss);
    if ((threadIdx.x & 31) == 0) {
        qbuf[threadIdx.x >> 5] = qss;
        kbuf[threadIdx.x >> 5] = kss;
    }
    if (threadIdx.x == 0) {
        beta_h = sigmoid_d(bb[h]);
        glog_h = -expf(A_log[h]) * softplus_d(aa[h] + dt_bias[h]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int nw = blockDim.x >> 5;
        float qt = 0.f, kt = 0.f;
        for (int w = 0; w < nw; ++w) {
            qt += qbuf[w];
            kt += kbuf[w];
        }
        qinv = rsqrtf(qt + 1e-6f) * rsqrtf(static_cast<float>(dk));
        kinv = rsqrtf(kt + 1e-6f);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        q[i] *= qinv;
        k[i] *= kinv;
    }
    __syncthreads();
    if (pf && pf_bytes >= 128) {
        const int nch = pf_bytes >> 7;
        const int tid = h * blockDim.x + threadIdx.x;
        const int nthr = nv * blockDim.x;
        for (int c = tid; c < nch; c += nthr) {
            const char* addr = reinterpret_cast<const char*>(pf) + (static_cast<size_t>(c) << 7);
            asm volatile("prefetch.global.L2 [%0];" ::"l"(addr));
        }
    }
    const float g = expf(glog_h);
    const float b = beta_h;
    const int j = threadIdx.x;
    // Compute stays 128-wide (one thread per dv). Extra 128 threads only
    // accelerate S copy / prefetch above.
    if (j < dv) {
        float kvj = 0.f;
#pragma unroll 4
        for (int i = 0; i < dk; i += 4) {
            const float s0 = Sp[(i + 0) * dv + j] * g;
            const float s1 = Sp[(i + 1) * dv + j] * g;
            const float s2 = Sp[(i + 2) * dv + j] * g;
            const float s3 = Sp[(i + 3) * dv + j] * g;
            Sp[(i + 0) * dv + j] = s0;
            Sp[(i + 1) * dv + j] = s1;
            Sp[(i + 2) * dv + j] = s2;
            Sp[(i + 3) * dv + j] = s3;
            kvj += s0 * k[i] + s1 * k[i + 1] + s2 * k[i + 2] + s3 * k[i + 3];
        }
        const float vj = conv1d_mix(qkv_raw, conv_w, conv_st, 2 * qdim + h * dv + j, 4);
        const float del = (vj - kvj) * b;
        float oj = 0.f;
#pragma unroll 4
        for (int i = 0; i < dk; i += 4) {
            const float s0 = Sp[(i + 0) * dv + j] + k[i] * del;
            const float s1 = Sp[(i + 1) * dv + j] + k[i + 1] * del;
            const float s2 = Sp[(i + 2) * dv + j] + k[i + 2] * del;
            const float s3 = Sp[(i + 3) * dv + j] + k[i + 3] * del;
            Sp[(i + 0) * dv + j] = s0;
            Sp[(i + 1) * dv + j] = s1;
            Sp[(i + 2) * dv + j] = s2;
            Sp[(i + 3) * dv + j] = s3;
            oj += s0 * q[i] + s1 * q[i + 1] + s2 * q[i + 2] + s3 * q[i + 3];
        }
        o[h * dv + j] = oj;
        conv1d_push(qkv_raw, conv_st, 2 * qdim + h * dv + j, 4);
    }
    __syncthreads();
    if (h % rep == 0) {
        for (int i = threadIdx.x; i < dk; i += blockDim.x) {
            conv1d_push(qkv_raw, conv_st, src * dk + i, 4);
            conv1d_push(qkv_raw, conv_st, qdim + src * dk + i, 4);
        }
    }
    gdn_s_store_bf16(Sh, Sp, dk * dv);
}

__global__ void gated_rms_batch_k(const float* x, const float* z, const float* gamma, float* y, int n, int T,
                                  float eps, int gamma_n) {
    const int t = blockIdx.x;
    if (t >= T) return;
    __shared__ float buf[256];
    const float* xt = x + static_cast<size_t>(t) * n;
    const float* zt = z + static_cast<size_t>(t) * n;
    float* yt = y + static_cast<size_t>(t) * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xt[i] * xt[i];
    buf[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(buf[0] / static_cast<float>(n) + eps);
    if (gamma_n <= 0) gamma_n = n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float g = gamma ? gamma[i % gamma_n] : 1.f;
        yt[i] = g * (xt[i] * inv) * silu_d(zt[i]);
    }
}

__global__ void split_qg_k(const float* qg, float* q, float* gate, int qn) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < qn) {
        q[i] = qg[i];
        gate[i] = qg[qn + i];
    }
}

__global__ void head_rms_k(float* x, const float* gamma, int n_heads, int hd, float eps) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    float* v = x + h * hd;
    float ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += v[i] * v[i];
    ss = warp_sum(ss);
    __shared__ float inv;
    if (threadIdx.x == 0) inv = rsqrtf(ss / static_cast<float>(hd) + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        const float g = gamma ? (1.f + gamma[i]) : 1.f;
        v[i] *= inv * g;
    }
}

__global__ void rope_k(float* q, float* k, int n_q, int n_kv, int head_dim, int rotary_dim, const int* pos,
                       float theta) {
    const int t = blockIdx.x; // 0 = q, 1 = k
    const int n_heads = t == 0 ? n_q : n_kv;
    const int h = blockIdx.y;
    if (h >= n_heads) return;
    float* v = (t == 0 ? q : k) + h * head_dim;
    const float p = static_cast<float>(*pos);
    const int pairs = rotary_dim / 2;
    for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
        const float freq = 1.f / powf(theta, static_cast<float>(i) / static_cast<float>(pairs));
        const float ang = p * freq;
        const float c = cosf(ang), s = sinf(ang);
        const float x0 = v[2 * i], x1 = v[2 * i + 1];
        v[2 * i] = x0 * c - x1 * s;
        v[2 * i + 1] = x0 * s + x1 * c;
    }
}

__global__ void store_kv_k(float* cache, const float* src, const int* pos, int kn) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kn) cache[static_cast<size_t>(*pos) * kn + i] = src[i];
}

// T=1: head RMS + RoPE on Q and K, then store K/V. Grid = max(n_q, n_kv), 32 threads.
__global__ void qk_prep_store_k(float* q, float* k, const float* v, const float* q_norm, const float* k_norm,
                                float* k_cache, float* v_cache, int n_q, int n_kv, int hd, int rotary,
                                const int* pos, float theta, float eps) {
    const int h = blockIdx.x;
    const int p = *pos;
    const int kn = n_kv * hd;
    const int pairs = rotary / 2;
    const float pf = static_cast<float>(p);
    if (h < n_q) {
        float* vh = q + h * hd;
        float ss = 0.f;
        for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += vh[i] * vh[i];
        ss = warp_sum(ss);
        __shared__ float invq;
        if (threadIdx.x == 0) invq = rsqrtf(ss / static_cast<float>(hd) + eps);
        __syncthreads();
        for (int i = threadIdx.x; i < hd; i += blockDim.x) {
            const float g = q_norm ? (1.f + q_norm[i]) : 1.f;
            vh[i] *= invq * g;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
            const float freq = 1.f / powf(theta, static_cast<float>(i) / static_cast<float>(pairs));
            const float ang = pf * freq;
            const float c = cosf(ang), s = sinf(ang);
            const float x0 = vh[2 * i], x1 = vh[2 * i + 1];
            vh[2 * i] = x0 * c - x1 * s;
            vh[2 * i + 1] = x0 * s + x1 * c;
        }
    }
    if (h < n_kv) {
        float* vh = k + h * hd;
        float ss = 0.f;
        for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += vh[i] * vh[i];
        ss = warp_sum(ss);
        __shared__ float invk;
        if (threadIdx.x == 0) invk = rsqrtf(ss / static_cast<float>(hd) + eps);
        __syncthreads();
        for (int i = threadIdx.x; i < hd; i += blockDim.x) {
            const float g = k_norm ? (1.f + k_norm[i]) : 1.f;
            vh[i] *= invk * g;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
            const float freq = 1.f / powf(theta, static_cast<float>(i) / static_cast<float>(pairs));
            const float ang = pf * freq;
            const float c = cosf(ang), s = sinf(ang);
            const float x0 = vh[2 * i], x1 = vh[2 * i + 1];
            vh[2 * i] = x0 * c - x1 * s;
            vh[2 * i + 1] = x0 * s + x1 * c;
        }
        __syncthreads();
        float* kc = k_cache + static_cast<size_t>(p) * kn + h * hd;
        float* vc = v_cache + static_cast<size_t>(p) * kn + h * hd;
        const float* vs = v + h * hd;
        for (int i = threadIdx.x; i < hd; i += blockDim.x) {
            kc[i] = vh[i];
            vc[i] = vs[i];
        }
    }
}

__global__ void attn_decode_k(const float* q, const float* k_cache, const float* v_cache, float* o, const int* pos,
                              int n_q, int n_kv, int head_dim) {
    const int hq = blockIdx.x;
    if (hq >= n_q) return;
    const int T = *pos + 1;
    const int rep = n_q / n_kv;
    const int hkv = hq / rep;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    extern __shared__ float sm[];
    float* scores = sm;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        float dot = 0.f;
        const float* qh = q + hq * head_dim;
        const float* kh = k_cache + static_cast<size_t>(t) * n_kv * head_dim + hkv * head_dim;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kh[d];
        scores[t] = dot * scale;
    }
    __syncthreads();
    __shared__ float mx, sumv;
    if (threadIdx.x == 0) {
        float mm = -1e30f;
        for (int t = 0; t < T; ++t) mm = fmaxf(mm, scores[t]);
        mx = mm;
        float s = 0.f;
        for (int t = 0; t < T; ++t) {
            scores[t] = expf(scores[t] - mx);
            s += scores[t];
        }
        sumv = s > 0.f ? s : 1.f;
    }
    __syncthreads();
    float* oh = o + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < T; ++t) {
            const float* vh = v_cache + static_cast<size_t>(t) * n_kv * head_dim + hkv * head_dim;
            acc += (scores[t] / sumv) * vh[d];
        }
        oh[d] = acc;
    }
}

// T=1: per-q-head RMS+RoPE, GQA replica writes K/V into cache from a local k copy (no
// in-place k race), then decode attn. Grid = n_q, 128 threads (4-warp RMS reduce).
// Not the failed 128-thread + in-kernel RoPE table (that used warp_sum as block sum).
// smem = ctx + hd floats.
__global__ void qk_attn_decode_k(float* q, const float* k, const float* v, const float* q_norm,
                                 const float* k_norm, float* k_cache, float* v_cache, float* o,
                                 const int* pos, int n_q, int n_kv, int hd, int rotary, float theta,
                                 float eps, int ctx) {
    const int hq = blockIdx.x;
    if (hq >= n_q || n_kv <= 0 || hd <= 0 || ctx <= 0) return;
    const int p = *pos;
    const int T = p + 1;
    const int kn = n_kv * hd;
    const int pairs = rotary / 2;
    const float pf = static_cast<float>(p);
    const int rep = n_q / n_kv;
    const int hkv = hq / rep;
    extern __shared__ float sm[];
    float* scores = sm;
    float* khloc = sm + (ctx > 8192 ? 0 : ctx);

    float* qh = q + hq * hd;
    __shared__ float inv, wss[4];
    const bool f4 = (hd == 128 && (ctx & 3) == 0);
    float ss = 0.f;
    if (f4) {
        const int i4 = threadIdx.x;
        if (i4 < 32) {
            const float4 v = *reinterpret_cast<const float4*>(qh + (i4 << 2));
            ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += qh[i] * qh[i];
    }
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) wss[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = wss[0];
        const int nw = blockDim.x >> 5;
        for (int w = 1; w < nw && w < 4; ++w) tot += wss[w];
        inv = rsqrtf(tot / static_cast<float>(hd) + eps);
    }
    __syncthreads();
    if (f4) {
        const int i4 = threadIdx.x;
        if (i4 < 32) {
            float4 v = *reinterpret_cast<float4*>(qh + (i4 << 2));
            float4 g = {0.f, 0.f, 0.f, 0.f};
            if (q_norm) g = *reinterpret_cast<const float4*>(q_norm + (i4 << 2));
            v.x *= inv * (1.f + g.x);
            v.y *= inv * (1.f + g.y);
            v.z *= inv * (1.f + g.z);
            v.w *= inv * (1.f + g.w);
            *reinterpret_cast<float4*>(qh + (i4 << 2)) = v;
        }
    } else {
        for (int i = threadIdx.x; i < hd; i += blockDim.x) {
            const float g = q_norm ? (1.f + q_norm[i]) : 1.f;
            qh[i] *= inv * g;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
        const float freq = 1.f / powf(theta, static_cast<float>(i) / static_cast<float>(pairs));
        const float ang = pf * freq;
        const float c = cosf(ang), s = sinf(ang);
        const float x0 = qh[2 * i], x1 = qh[2 * i + 1];
        qh[2 * i] = x0 * c - x1 * s;
        qh[2 * i + 1] = x0 * s + x1 * c;
    }

    const float* ksrc = k + hkv * hd;
    if (f4) {
        const int i4 = threadIdx.x;
        if (i4 < 32)
            reinterpret_cast<float4*>(khloc)[i4] = reinterpret_cast<const float4*>(ksrc)[i4];
    } else {
        for (int i = threadIdx.x; i < hd; i += blockDim.x) khloc[i] = ksrc[i];
    }
    __syncthreads();
    ss = 0.f;
    if (f4) {
        const int i4 = threadIdx.x;
        if (i4 < 32) {
            const float4 v = reinterpret_cast<const float4*>(khloc)[i4];
            ss = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += khloc[i] * khloc[i];
    }
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) wss[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = wss[0];
        const int nw = blockDim.x >> 5;
        for (int w = 1; w < nw && w < 4; ++w) tot += wss[w];
        inv = rsqrtf(tot / static_cast<float>(hd) + eps);
    }
    __syncthreads();
    if (f4) {
        const int i4 = threadIdx.x;
        if (i4 < 32) {
            float4 v = reinterpret_cast<float4*>(khloc)[i4];
            float4 g = {0.f, 0.f, 0.f, 0.f};
            if (k_norm) g = *reinterpret_cast<const float4*>(k_norm + (i4 << 2));
            v.x *= inv * (1.f + g.x);
            v.y *= inv * (1.f + g.y);
            v.z *= inv * (1.f + g.z);
            v.w *= inv * (1.f + g.w);
            reinterpret_cast<float4*>(khloc)[i4] = v;
        }
    } else {
        for (int i = threadIdx.x; i < hd; i += blockDim.x) {
            const float g = k_norm ? (1.f + k_norm[i]) : 1.f;
            khloc[i] *= inv * g;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
        const float freq = 1.f / powf(theta, static_cast<float>(i) / static_cast<float>(pairs));
        const float ang = pf * freq;
        const float c = cosf(ang), s = sinf(ang);
        const float x0 = khloc[2 * i], x1 = khloc[2 * i + 1];
        khloc[2 * i] = x0 * c - x1 * s;
        khloc[2 * i + 1] = x0 * s + x1 * c;
    }
    __syncthreads();
    float* kc = k_cache + static_cast<size_t>(p) * kn + hkv * hd;
    float* vc = v_cache + static_cast<size_t>(p) * kn + hkv * hd;
    const float* vs = v + hkv * hd;
    if (f4) {
        const int i4 = threadIdx.x;
        if (i4 < 32) {
            reinterpret_cast<float4*>(kc)[i4] = reinterpret_cast<const float4*>(khloc)[i4];
            reinterpret_cast<float4*>(vc)[i4] = reinterpret_cast<const float4*>(vs)[i4];
        }
    } else {
        for (int i = threadIdx.x; i < hd; i += blockDim.x) {
            kc[i] = khloc[i];
            vc[i] = vs[i];
        }
    }
    __syncthreads();

    const float scale = rsqrtf(static_cast<float>(hd));
    const int Tend = T < ctx ? T : ctx;
    float* oh = o + hq * hd;
    if (ctx > 8192) {
        // Online softmax: scores do not fit in smem at 128k/200k.
        float acc = 0.f;
        for (int t = 0; t < Tend; ++t) {
            const float* kh = k_cache + static_cast<size_t>(t) * kn + hkv * hd;
            float dot = 0.f;
            if (f4) {
                const int i4 = threadIdx.x;
                if (i4 < 32) {
                    const float4 qv = reinterpret_cast<const float4*>(qh)[i4];
                    const float4 kv = reinterpret_cast<const float4*>(kh)[i4];
                    dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
                }
            } else {
                for (int d = threadIdx.x; d < hd; d += blockDim.x) dot += qh[d] * kh[d];
            }
            dot = warp_sum(dot);
            if ((threadIdx.x & 31) == 0) wss[threadIdx.x >> 5] = dot;
            __syncthreads();
            if (threadIdx.x == 0) {
                float tot = wss[0];
                const int nw = blockDim.x >> 5;
                for (int w = 1; w < nw && w < 4; ++w) tot += wss[w];
                const float s = tot * scale;
                float& om = wss[0];
                float& ol = wss[1];
                float& aa = wss[2];
                float& pp = wss[3];
                if (t == 0) {
                    om = -1e30f;
                    ol = 0.f;
                }
                const float m2 = fmaxf(om, s);
                aa = expf(om - m2);
                pp = expf(s - m2);
                ol = ol * aa + pp;
                om = m2;
            }
            __syncthreads();
            const float aa = wss[2];
            const float pp = wss[3];
            const float* vh = v_cache + static_cast<size_t>(t) * kn + hkv * hd;
            if (threadIdx.x < hd) acc = acc * aa + pp * vh[threadIdx.x];
            __syncthreads();
        }
        if (threadIdx.x < hd) {
            const float den = wss[1] > 0.f ? wss[1] : 1.f;
            oh[threadIdx.x] = acc / den;
        }
    } else {
        for (int t = threadIdx.x; t < Tend; t += blockDim.x) {
            float dot = 0.f;
            const float* kh = k_cache + static_cast<size_t>(t) * kn + hkv * hd;
            if (f4) {
#pragma unroll
                for (int d = 0; d < 32; ++d) {
                    const float4 qv = reinterpret_cast<const float4*>(qh)[d];
                    const float4 kv = reinterpret_cast<const float4*>(kh)[d];
                    dot += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
                }
            } else {
                for (int d = 0; d < hd; ++d) dot += qh[d] * kh[d];
            }
            scores[t] = dot * scale;
        }
        __syncthreads();
        __shared__ float mx, sumv;
        if (threadIdx.x == 0) {
            float mm = -1e30f;
            for (int t = 0; t < Tend; ++t) mm = fmaxf(mm, scores[t]);
            mx = mm;
            float s = 0.f;
            for (int t = 0; t < Tend; ++t) {
                scores[t] = expf(scores[t] - mx);
                s += scores[t];
            }
            sumv = s > 0.f ? s : 1.f;
        }
        __syncthreads();
        for (int d = threadIdx.x; d < hd; d += blockDim.x) {
            float acc = 0.f;
            for (int t = 0; t < Tend; ++t) {
                const float* vh = v_cache + static_cast<size_t>(t) * kn + hkv * hd;
                acc += (scores[t] / sumv) * vh[d];
            }
            oh[d] = acc;
        }
    }
}

__global__ void apply_gate_k(float* o, const float* gate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] *= sigmoid_d(gate[i]);
}

__global__ void embed_f32_k(const float* E, const int* tok, float* x, int H) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H) x[i] = E[static_cast<size_t>(*tok) * H + i];
}

__global__ void embed_f16_k(const __half* E, const int* tok, float* x, int H) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H) x[i] = __half2float(E[static_cast<size_t>(*tok) * H + i]);
}

__global__ void embed_bf16_k(const uint16_t* E, const int* tok, float* x, int H) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H) x[i] = bf16_to_f32(E[static_cast<size_t>(*tok) * H + i]);
}

__global__ void inc_pos_k(int* pos) {
    if (threadIdx.x == 0 && blockIdx.x == 0) ++(*pos);
}

__global__ void set_pos_k(int* pos, int v) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *pos = v;
}

__global__ void argmax_partial_k(const float* x, int n, float* blk_max, int* blk_idx) {
    const int tid = threadIdx.x;
    const int i0 = blockIdx.x * blockDim.x;
    float best = -1e30f;
    int bi = 0;
    for (int i = i0 + tid; i < n && i < i0 + blockDim.x; i += 1) {
        if (x[i] > best) {
            best = x[i];
            bi = i;
        }
    }
    // one element per thread in this block
    const int i = i0 + tid;
    if (i < n) {
        best = x[i];
        bi = i;
    } else {
        best = -1e30f;
        bi = 0;
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float o = __shfl_down_sync(0xffffffff, best, off);
        const int oi = __shfl_down_sync(0xffffffff, bi, off);
        if (o > best) {
            best = o;
            bi = oi;
        }
    }
    __shared__ float wm[32];
    __shared__ int wi[32];
    if ((tid & 31) == 0) {
        wm[tid / 32] = best;
        wi[tid / 32] = bi;
    }
    __syncthreads();
    if (tid < 32) {
        float b = (tid < (blockDim.x + 31) / 32) ? wm[tid] : -1e30f;
        int ii = (tid < (blockDim.x + 31) / 32) ? wi[tid] : 0;
        for (int off = 16; off > 0; off >>= 1) {
            const float o = __shfl_down_sync(0xffffffff, b, off);
            const int oi = __shfl_down_sync(0xffffffff, ii, off);
            if (o > b) {
                b = o;
                ii = oi;
            }
        }
        if (tid == 0) {
            blk_max[blockIdx.x] = b;
            blk_idx[blockIdx.x] = ii;
        }
    }
}

__global__ void rmsnorm_batch_k(const float* x, const float* gamma, float* y, int n, int T, float eps) {
    const int t = blockIdx.x;
    if (t >= T) return;
    __shared__ float buf[256];
    const float* xt = x + static_cast<size_t>(t) * n;
    float* yt = y + static_cast<size_t>(t) * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xt[i] * xt[i];
    buf[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(buf[0] / static_cast<float>(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float g = gamma ? (1.f + gamma[i]) : 1.f;
        yt[i] = xt[i] * inv * g;
    }
}

__global__ void add_res_n_k(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ void swiglu_n_k(const float* gate, const float* up, float* y, int n) {
    const int i4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i4 + 3 < n) {
        const float4 g = *reinterpret_cast<const float4*>(gate + i4);
        const float4 u = *reinterpret_cast<const float4*>(up + i4);
        float4 o;
        o.x = silu_d(g.x) * u.x;
        o.y = silu_d(g.y) * u.y;
        o.z = silu_d(g.z) * u.z;
        o.w = silu_d(g.w) * u.w;
        *reinterpret_cast<float4*>(y + i4) = o;
        return;
    }
    for (int i = i4; i < n; ++i) y[i] = silu_d(gate[i]) * up[i];
}

__global__ void embed_f32_batch_k(const float* E, const int* toks, float* x, int H, int T) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T && i < H) x[static_cast<size_t>(t) * H + i] = E[static_cast<size_t>(toks[t]) * H + i];
}

__global__ void embed_f16_batch_k(const __half* E, const int* toks, float* x, int H, int T) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T && i < H) x[static_cast<size_t>(t) * H + i] = __half2float(E[static_cast<size_t>(toks[t]) * H + i]);
}

__global__ void embed_bf16_batch_k(const uint16_t* E, const int* toks, float* x, int H, int T) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T && i < H) x[static_cast<size_t>(t) * H + i] = bf16_to_f32(E[static_cast<size_t>(toks[t]) * H + i]);
}

__global__ void f32_to_f16_k(const float* x, __half* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

__global__ void f32_to_bf16_k(const float* x, uint16_t* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = static_cast<uint16_t>(__float_as_uint(x[i]) >> 16);
}

__global__ void split_qg_batch_k(const float* qg, float* q, float* gate, int qn, int T) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T && i < qn) {
        q[static_cast<size_t>(t) * qn + i] = qg[static_cast<size_t>(t) * qn * 2 + i];
        gate[static_cast<size_t>(t) * qn + i] = qg[static_cast<size_t>(t) * qn * 2 + qn + i];
    }
}

__global__ void head_rms_batch_k(float* x, const float* gamma, int n_heads, int hd, int T, float eps) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    if (h >= n_heads || t >= T) return;
    float* v = x + (static_cast<size_t>(t) * n_heads + h) * hd;
    float ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += v[i] * v[i];
    ss = warp_sum(ss);
    __shared__ float inv;
    if (threadIdx.x == 0) inv = rsqrtf(ss / static_cast<float>(hd) + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        const float g = gamma ? (1.f + gamma[i]) : 1.f;
        v[i] *= inv * g;
    }
}

__global__ void rope_batch_k(float* q, float* k, int n_q, int n_kv, int head_dim, int rotary_dim, int pos0,
                             int T, float theta) {
    const int which = blockIdx.x; // 0 = q, 1 = k
    const int h = blockIdx.y;
    const int t = blockIdx.z;
    if (t >= T) return;
    const int n_heads = which == 0 ? n_q : n_kv;
    if (h >= n_heads) return;
    float* v = (which == 0 ? q : k) + (static_cast<size_t>(t) * n_heads + h) * head_dim;
    const float p = static_cast<float>(pos0 + t);
    const int pairs = rotary_dim / 2;
    for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
        const float freq = 1.f / powf(theta, static_cast<float>(i) / static_cast<float>(pairs));
        const float ang = p * freq;
        const float c = cosf(ang), s = sinf(ang);
        const float x0 = v[2 * i], x1 = v[2 * i + 1];
        v[2 * i] = x0 * c - x1 * s;
        v[2 * i + 1] = x0 * s + x1 * c;
    }
}

__global__ void store_kv_batch_k(float* cache, const float* src, int pos0, int T, int kn) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T && i < kn) cache[static_cast<size_t>(pos0 + t) * kn + i] = src[static_cast<size_t>(t) * kn + i];
}

__global__ void attn_prefill_k(const float* q, const float* k_cache, const float* v_cache, float* o, int pos0,
                               int T, int n_q, int n_kv, int head_dim) {
    const int hq = blockIdx.x;
    const int t = blockIdx.y;
    if (hq >= n_q || t >= T) return;
    const int pos = pos0 + t;
    const int Tend = pos + 1;
    const int rep = n_q / n_kv;
    const int hkv = hq / rep;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float* qh = q + (static_cast<size_t>(t) * n_q + hq) * head_dim;
    float* oh = o + (static_cast<size_t>(t) * n_q + hq) * head_dim;
    if (Tend > 8192) {
        __shared__ float wss[4];
        float acc = 0.f;
        for (int s = 0; s < Tend; ++s) {
            const float* kh = k_cache + static_cast<size_t>(s) * n_kv * head_dim + hkv * head_dim;
            float dot = 0.f;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) dot += qh[d] * kh[d];
            dot = warp_sum(dot);
            if ((threadIdx.x & 31) == 0) wss[threadIdx.x >> 5] = dot;
            __syncthreads();
            if (threadIdx.x == 0) {
                float tot = wss[0];
                const int nw = blockDim.x >> 5;
                for (int w = 1; w < nw && w < 4; ++w) tot += wss[w];
                const float sc = tot * scale;
                if (s == 0) {
                    wss[0] = -1e30f;
                    wss[1] = 0.f;
                }
                const float m2 = fmaxf(wss[0], sc);
                wss[2] = expf(wss[0] - m2);
                wss[3] = expf(sc - m2);
                wss[1] = wss[1] * wss[2] + wss[3];
                wss[0] = m2;
            }
            __syncthreads();
            const float* vh = v_cache + static_cast<size_t>(s) * n_kv * head_dim + hkv * head_dim;
            if (threadIdx.x < head_dim) acc = acc * wss[2] + wss[3] * vh[threadIdx.x];
            __syncthreads();
        }
        if (threadIdx.x < head_dim) oh[threadIdx.x] = acc / (wss[1] > 0.f ? wss[1] : 1.f);
        return;
    }
    extern __shared__ float scores[];
    for (int s = threadIdx.x; s < Tend; s += blockDim.x) {
        float dot = 0.f;
        const float* kh = k_cache + static_cast<size_t>(s) * n_kv * head_dim + hkv * head_dim;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kh[d];
        scores[s] = dot * scale;
    }
    __syncthreads();
    __shared__ float mx, sumv;
    if (threadIdx.x == 0) {
        float mm = -1e30f;
        for (int s = 0; s < Tend; ++s) mm = fmaxf(mm, scores[s]);
        mx = mm;
        float z = 0.f;
        for (int s = 0; s < Tend; ++s) {
            scores[s] = expf(scores[s] - mx);
            z += scores[s];
        }
        sumv = z > 0.f ? z : 1.f;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < Tend; ++s) {
            const float* vh = v_cache + static_cast<size_t>(s) * n_kv * head_dim + hkv * head_dim;
            acc += (scores[s] / sumv) * vh[d];
        }
        oh[d] = acc;
    }
}

__global__ void apply_gate_n_k(float* o, const float* gate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] *= sigmoid_d(gate[i]);
}

__global__ void conv1d_prefill_k(const float* x, const float* w, float* y, float* state, int seq, int dim,
                                 int k) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= dim) return;
    for (int t = 0; t < seq; ++t) {
        float acc = 0.f;
        for (int p = 0; p < k; ++p) {
            const int src = t - (k - 1 - p);
            const float xv = src >= 0 ? x[static_cast<size_t>(src) * dim + c] : 0.f;
            acc += w[c * k + p] * xv;
        }
        y[static_cast<size_t>(t) * dim + c] = silu_d(acc);
    }
    float* st = state + c * k;
    for (int p = 0; p < k; ++p) {
        const int src = seq - k + p;
        st[p] = src >= 0 ? x[static_cast<size_t>(src) * dim + c] : 0.f;
    }
}

__global__ void argmax_final_k(const float* blk_max, const int* blk_idx, int nblk, int* out, int* pos,
                               int* next_tok, int* gen_out, int* gen_n) {
    float best = -1e30f;
    int bi = 0;
    for (int i = threadIdx.x; i < nblk; i += blockDim.x) {
        if (blk_max[i] > best) {
            best = blk_max[i];
            bi = blk_idx[i];
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float o = __shfl_down_sync(0xffffffff, best, off);
        const int oi = __shfl_down_sync(0xffffffff, bi, off);
        if (o > best) {
            best = o;
            bi = oi;
        }
    }
    if (threadIdx.x == 0) {
        *out = bi;
        if (next_tok) *next_tok = bi;
        if (pos) ++*pos;
        if (gen_out && gen_n) {
            const int i = *gen_n;
            gen_out[i] = bi;
            *gen_n = i + 1;
        }
    }
}

// T independent argmaxes: grid (nblk, T). blk_max/idx laid out [T][nblk].
__global__ void argmax_partial_rows_k(const float* x, int n, int stride, float* blk_max, int* blk_idx) {
    const int t = blockIdx.y;
    const int tid = threadIdx.x;
    const int i0 = blockIdx.x * blockDim.x;
    const float* row = x + static_cast<size_t>(t) * stride;
    const int i = i0 + tid;
    float best = (i < n) ? row[i] : -1e30f;
    int bi = (i < n) ? i : 0;
    for (int off = 16; off > 0; off >>= 1) {
        const float o = __shfl_down_sync(0xffffffff, best, off);
        const int oi = __shfl_down_sync(0xffffffff, bi, off);
        if (o > best) {
            best = o;
            bi = oi;
        }
    }
    __shared__ float wm[32];
    __shared__ int wi[32];
    if ((tid & 31) == 0) {
        wm[tid / 32] = best;
        wi[tid / 32] = bi;
    }
    __syncthreads();
    if (tid < 32) {
        float b = (tid < (blockDim.x + 31) / 32) ? wm[tid] : -1e30f;
        int ii = (tid < (blockDim.x + 31) / 32) ? wi[tid] : 0;
        for (int off = 16; off > 0; off >>= 1) {
            const float o = __shfl_down_sync(0xffffffff, b, off);
            const int oi = __shfl_down_sync(0xffffffff, ii, off);
            if (o > b) {
                b = o;
                ii = oi;
            }
        }
        if (tid == 0) {
            const int slot = t * gridDim.x + blockIdx.x;
            blk_max[slot] = b;
            blk_idx[slot] = ii;
        }
    }
}

__global__ void argmax_final_rows_k(const float* blk_max, const int* blk_idx, int nblk, int T, int* out) {
    const int t = threadIdx.x;
    if (t >= T) return;
    float best = -1e30f;
    int bi = 0;
    const float* bm = blk_max + t * nblk;
    const int* bi_p = blk_idx + t * nblk;
    for (int i = 0; i < nblk; ++i) {
        if (bm[i] > best) {
            best = bm[i];
            bi = bi_p[i];
        }
    }
    out[t] = bi;
}

__global__ void rope_batch_dp_k(float* q, float* k, int n_q, int n_kv, int head_dim, int rotary_dim,
                                const int* dpos, int T, float theta) {
    const int which = blockIdx.x;
    const int h = blockIdx.y;
    const int t = blockIdx.z;
    if (t >= T) return;
    const int n_heads = which == 0 ? n_q : n_kv;
    if (h >= n_heads) return;
    float* v = (which == 0 ? q : k) + (static_cast<size_t>(t) * n_heads + h) * head_dim;
    const float p = static_cast<float>(*dpos + t);
    const int pairs = rotary_dim / 2;
    for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
        const float freq = 1.f / powf(theta, static_cast<float>(i) / static_cast<float>(pairs));
        const float ang = p * freq;
        const float c = cosf(ang), s = sinf(ang);
        const float x0 = v[2 * i], x1 = v[2 * i + 1];
        v[2 * i] = x0 * c - x1 * s;
        v[2 * i + 1] = x0 * s + x1 * c;
    }
}

__global__ void store_kv_batch_dp_k(float* cache, const float* src, const int* dpos, int T, int kn) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T && i < kn) cache[static_cast<size_t>(*dpos + t) * kn + i] = src[static_cast<size_t>(t) * kn + i];
}

__global__ void attn_prefill_dp_k(const float* q, const float* k_cache, const float* v_cache, float* o,
                                  const int* dpos, int T, int n_q, int n_kv, int head_dim) {
    const int hq = blockIdx.x;
    const int t = blockIdx.y;
    if (hq >= n_q || t >= T) return;
    const int pos = *dpos + t;
    const int Tend = pos + 1;
    const int rep = n_q / n_kv;
    const int hkv = hq / rep;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    extern __shared__ float scores[];
    const float* qh = q + (static_cast<size_t>(t) * n_q + hq) * head_dim;
    for (int s = threadIdx.x; s < Tend; s += blockDim.x) {
        float dot = 0.f;
        const float* kh = k_cache + static_cast<size_t>(s) * n_kv * head_dim + hkv * head_dim;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kh[d];
        scores[s] = dot * scale;
    }
    __syncthreads();
    __shared__ float mx, sumv;
    if (threadIdx.x == 0) {
        float mm = -1e30f;
        for (int s = 0; s < Tend; ++s) mm = fmaxf(mm, scores[s]);
        mx = mm;
        float z = 0.f;
        for (int s = 0; s < Tend; ++s) {
            scores[s] = expf(scores[s] - mx);
            z += scores[s];
        }
        sumv = z > 0.f ? z : 1.f;
    }
    __syncthreads();
    float* oh = o + (static_cast<size_t>(t) * n_q + hq) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < Tend; ++s) {
            const float* vh = v_cache + static_cast<size_t>(s) * n_kv * head_dim + hkv * head_dim;
            acc += (scores[s] / sumv) * vh[d];
        }
        oh[d] = acc;
    }
}

struct GpuW {
    const uint8_t* data = nullptr;
    const float* scale = nullptr;
    QuantKind q = QuantKind::F32;
    int rows = 0, cols = 0;
    bool fp8_tiled = false; // 16x32 e4m3 tiles for MMA
    bool fp8_kmajor = false; // 512-col lane-interleaved: W[(sg*rows+row)*512 + lane*16 + t*4]
};

cublasHandle_t g_blas = nullptr;
__half* g_xf16 = nullptr;
int g_xf16_n = 0;
bool g_fp8_tc = true;
bool g_fp8_e4m3_mma = false;

bool launch_cublas_f16(const GpuW& w, const float* X, float* Y, int T);
void launch_fp8_tc(const GpuW& w, const float* X, float* Y, int T);
void launch_gemm_fp8(const GpuW& w, const float* X, float* Y, int T, int add);
void launch_fp8_e4m3_mma(const GpuW& w, const float* x, float* y);
void launch_linear(const GpuW& w, const float* X, float* Y, int T, int add = 0);

int fp8_xs_tile(int n) { return n <= kFp8XsCap ? n : kFp8XsCap; }
bool fp8_single_tile(int n) { return n > 0 && n <= kFp8XsCap && (n & 511) == 0; }

void gemv_grid(const GpuW& w, int& blocks, int& threads) {
    const int warps = w.rows >= 2048 ? 16 : 8;
    threads = warps * 32;
    blocks = (w.rows + warps - 1) / warps;
}

void launch_gemv(const GpuW& w, const float* x, float* y, int add = 0, float* acc_ss = nullptr,
                 const float* x_gamma = nullptr, const float* d_ss = nullptr, float rms_eps = 0.f,
                 const float* x_sig = nullptr, float* y_split = nullptr, int split_at = 0,
                 const float* x_silu = nullptr, int gnorm_n = 0, int use_ca = 0) {
    if (!w.data || w.rows <= 0 || w.cols <= 0) return;
    int blocks = 0, threads = 0;
    gemv_grid(w, blocks, threads);
    switch (w.q) {
    case QuantKind::F32:
        gemv_f32_k<<<blocks, threads>>>(reinterpret_cast<const float*>(w.data), x, y, w.rows, w.cols, add);
        break;
    case QuantKind::F16:
        gemv_f16_k<<<blocks, threads>>>(reinterpret_cast<const __half*>(w.data), x, y, w.rows, w.cols, add);
        break;
    case QuantKind::BF16:
        if (w.rows >= 4096) {
            const int pairs = (w.rows + 1) / 2;
            const int warps = threads / 32;
            const int pblocks = (pairs + warps - 1) / warps;
            gemv_bf16_2row_k<<<pblocks, threads>>>(reinterpret_cast<const uint16_t*>(w.data), x, y, w.rows,
                                                   w.cols, add);
        } else {
            gemv_bf16_k<<<blocks, threads>>>(reinterpret_cast<const uint16_t*>(w.data), x, y, w.rows, w.cols, add);
        }
        break;
    case QuantKind::Q8_0: {
        const int tile = w.cols < kXsTile ? w.cols : kXsTile;
        const int smem_n = (tile / 32) * kXsPad;
        gemv_q8_soa_k<<<blocks, threads, sizeof(float) * smem_n>>>(reinterpret_cast<const int8_t*>(w.data),
                                                                   reinterpret_cast<const __half*>(w.scale), x, y,
                                                                   w.rows, w.cols, add);
        break;
    }
    case QuantKind::FP8_E4M3_B128: {
        if (g_fp8_e4m3_mma && w.rows >= 16 && w.cols >= 32 && !add) {
            launch_fp8_e4m3_mma(w, x, y);
            break;
        }
        const int tile = fp8_xs_tile(w.cols);
        if (w.rows >= 4096) {
            const int pairs = (w.rows + 1) / 2;
            const int warps = threads / 32;
            const int pblocks = (pairs + warps - 1) / warps;
            const bool st = fp8_single_tile(w.cols);
            if (add && !acc_ss && !x_gamma && !x_sig && !y_split && !x_silu) {
                if (st)
                    gemv_fp8_2row_add_st_k<<<pblocks, threads, sizeof(float) * tile>>>(w.data, w.scale, x, y,
                                                                                      w.rows, w.cols, 128);
                else if (w.cols > kFp8XsCap) {
                    const int wd_sm = (w.cols == 2 * kWdXs) ? kWdXs : tile;
                    gemv_fp8_2row_add_wd_k<<<pblocks, threads, sizeof(float) * wd_sm>>>(w.data, w.scale, x, y,
                                                                                       w.rows, w.cols, 128);
                }
                else
                    gemv_fp8_2row_add_k<<<pblocks, threads, sizeof(float) * tile>>>(w.data, w.scale, x, y, w.rows,
                                                                                   w.cols, 128);
            } else if (st)
                gemv_fp8_2row_st_k<<<pblocks, threads, sizeof(float) * tile>>>(
                    w.data, w.scale, x, y, w.rows, w.cols, 128, add, acc_ss, x_gamma, d_ss, rms_eps, x_sig,
                    y_split, split_at, x_silu, gnorm_n, use_ca);
            else
                gemv_fp8_2row_k<<<pblocks, threads, sizeof(float) * tile>>>(
                    w.data, w.scale, x, y, w.rows, w.cols, 128, add, acc_ss, x_gamma, d_ss, rms_eps, x_sig,
                    y_split, split_at, x_silu, gnorm_n);
        } else {
            gemv_fp8_k<<<blocks, threads, sizeof(float) * tile>>>(w.data, w.scale, x, y, w.rows, w.cols, 128, add);
        }
        break;
    }
    default:
        throw std::runtime_error("CUDA GEMV unsupported quant");
    }
}

void launch_gemv_dual(const GpuW& w1, const GpuW& w2, const float* x, float* y1, float* y2,
                     int fuse_swiglu = 0, const float* x_gamma = nullptr, const float* d_ss = nullptr,
                     float rms_eps = 0.f) {
    if (w1.q == QuantKind::FP8_E4M3_B128 && w2.q == QuantKind::FP8_E4M3_B128 && w1.rows == w2.rows &&
        w1.cols == w2.cols && w1.data && w2.data && w1.rows > 0 && w1.cols > 0) {
        int blocks = 0, threads = 0;
        gemv_grid(w1, blocks, threads);
        const int tile = fp8_xs_tile(w1.cols);
        if (w1.rows >= 4096) {
            const int pairs = (w1.rows + 1) / 2;
            const int warps = threads / 32;
            const int pblocks = (pairs + warps - 1) / warps;
            if (fp8_single_tile(w1.cols))
                gemv_fp8_2row_dual_st_k<<<pblocks, threads, sizeof(float) * tile>>>(
                    w1.data, w1.scale, w2.data, w2.scale, x, y1, y2, w1.rows, w1.cols, 128, fuse_swiglu, x_gamma,
                    d_ss, rms_eps);
            else
                gemv_fp8_2row_dual_k<<<pblocks, threads, sizeof(float) * tile>>>(
                    w1.data, w1.scale, w2.data, w2.scale, x, y1, y2, w1.rows, w1.cols, 128, fuse_swiglu, x_gamma,
                    d_ss, rms_eps);
        } else {
            gemv_fp8_dual_k<<<blocks, threads, sizeof(float) * tile>>>(
                w1.data, w1.scale, w2.data, w2.scale, x, y1, y2, w1.rows, w1.cols, 128, fuse_swiglu, x_gamma,
                d_ss, rms_eps);
        }
        return;
    }
    launch_gemv(w1, x, y1, 0, nullptr, x_gamma, d_ss, rms_eps);
    launch_gemv(w2, x, y2, 0, nullptr, x_gamma, d_ss, rms_eps);
    if (fuse_swiglu && y1 && y2)
        swiglu_k<<<(((w1.rows + 3) / 4) + 255) / 256, 256>>>(y1, y2, y1, w1.rows);
}

void launch_gemv_pair(const GpuW& w1, const GpuW& w2, const float* x, float* y1, float* y2,
                     const GpuW* w3 = nullptr, float* y3 = nullptr, const GpuW* w4 = nullptr,
                     float* y4 = nullptr, const float* x_gamma = nullptr, const float* d_ss = nullptr,
                     float rms_eps = 0.f) {
    if (w1.q == QuantKind::FP8_E4M3_B128 && w2.q == QuantKind::FP8_E4M3_B128 && w1.cols == w2.cols &&
        w1.data && w2.data && w1.cols > 0 && w1.rows > 0 && w2.rows > 0 &&
        std::max(w1.rows, w2.rows) >= 4096) {
        const int mmax = std::max(w1.rows, w2.rows);
        GpuW dummy = w1;
        dummy.rows = mmax;
        int blocks = 0, threads = 0;
        gemv_grid(dummy, blocks, threads);
        const int tile = fp8_xs_tile(w1.cols);
        const int pairs = (mmax + 1) / 2;
        const int warps = threads / 32;
        const int pblocks = (pairs + warps - 1) / warps;
        const bool extra = w3 && w4 && y3 && y4 && w3->data && w4->data && w3->rows > 0 && w4->rows > 0 &&
                           w3->q == QuantKind::FP8_E4M3_B128 && w4->q == QuantKind::FP8_E4M3_B128 &&
                           w3->cols == w1.cols && w4->cols == w1.cols;
        if (fp8_single_tile(w1.cols))
            gemv_fp8_2row_pair_st_k<<<pblocks, threads, sizeof(float) * tile>>>(
                w1.data, w1.scale, w2.data, w2.scale, x, y1, y2, w1.rows, w2.rows, w1.cols, 128,
                extra ? w3->data : nullptr, extra ? w3->scale : nullptr, extra ? y3 : nullptr,
                extra ? w3->rows : 0, extra ? w4->data : nullptr, extra ? w4->scale : nullptr,
                extra ? y4 : nullptr, extra ? w4->rows : 0, x_gamma, d_ss, rms_eps);
        else
            gemv_fp8_2row_pair_k<<<pblocks, threads, sizeof(float) * tile>>>(
                w1.data, w1.scale, w2.data, w2.scale, x, y1, y2, w1.rows, w2.rows, w1.cols, 128,
                extra ? w3->data : nullptr, extra ? w3->scale : nullptr, extra ? y3 : nullptr,
                extra ? w3->rows : 0, extra ? w4->data : nullptr, extra ? w4->scale : nullptr,
                extra ? y4 : nullptr, extra ? w4->rows : 0, x_gamma, d_ss, rms_eps);
        return;
    }
    launch_gemv(w1, x, y1, 0, nullptr, x_gamma, d_ss, rms_eps);
    launch_gemv(w2, x, y2, 0, nullptr, x_gamma, d_ss, rms_eps);
    if (w3 && y3) launch_gemv(*w3, x, y3, 0, nullptr, x_gamma, d_ss, rms_eps);
    if (w4 && y4) launch_gemv(*w4, x, y4, 0, nullptr, x_gamma, d_ss, rms_eps);
}

void launch_gemv_attn_in(const GpuW& wq, const GpuW& wk, const GpuW& wv, const float* x, float* q,
                         float* gate, float* k, float* v, int split_at, const float* x_gamma = nullptr,
                         const float* d_ss = nullptr, float rms_eps = 0.f) {
    if (wq.q == QuantKind::FP8_E4M3_B128 && wk.q == QuantKind::FP8_E4M3_B128 &&
        wv.q == QuantKind::FP8_E4M3_B128 && wq.cols == wk.cols && wk.cols == wv.cols && wq.data &&
        wk.data && wv.data && wq.rows >= 4096 && split_at > 0 && wq.rows == split_at * 2) {
        GpuW dummy = wq;
        int blocks = 0, threads = 0;
        gemv_grid(dummy, blocks, threads);
        const int tile = fp8_xs_tile(wq.cols);
        const int pairs = (wq.rows + 1) / 2;
        const int warps = threads / 32;
        const int pblocks = (pairs + warps - 1) / warps;
        gemv_fp8_2row_attn_in_k<<<pblocks, threads, sizeof(float) * tile>>>(
            wq.data, wq.scale, wk.data, wk.scale, wv.data, wv.scale, x, q, gate, k, v, wq.rows, wk.rows,
            wv.rows, wq.cols, 128, split_at, x_gamma, d_ss, rms_eps);
        return;
    }
    launch_gemv(wq, x, q, 0, nullptr, x_gamma, d_ss, rms_eps, nullptr, gate, split_at);
    launch_gemv_dual(wk, wv, x, k, v, 0, x_gamma, d_ss, rms_eps);
}

bool launch_cublas_f16(const GpuW& w, const float* X, float* Y, int T) {
    if (!g_blas || !g_xf16 || T <= 0) return false;
    const int n = w.cols * T;
    if (n > g_xf16_n) return false;
    const float alpha = 1.f, beta = 0.f;
    if (w.q == QuantKind::F16) {
        f32_to_f16_k<<<(n + 255) / 256, 256>>>(X, g_xf16, n);
        return cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, w.rows, T, w.cols, &alpha, w.data, CUDA_R_16F,
                            w.cols, g_xf16, CUDA_R_16F, w.cols, &beta, Y, CUDA_R_32F, w.rows, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS;
    }
    if (w.q == QuantKind::BF16) return false;
    return false;
}

int gdn_dyn_smem(int dk, int dv) {
    if (dk > 0 && dv > 0 && dk * dv <= 16384)
        return (2 * dk + dk * dv) * static_cast<int>(sizeof(float));
    return (2 * dk + 2 * dv) * static_cast<int>(sizeof(float));
}

int gdn_threads(int dv) {
    if (dv >= 32 && dv <= 256) return dv;
    return 128;
}

void launch_gdn(const float* mix, const float* aa, const float* bb, uint16_t* S, const float* A_log,
                const float* dt_bias, float* o, int T, int nk, int nv, int dk, int dv, int qkv_dim,
                const float* qkv_raw = nullptr, const float* conv_w = nullptr, float* conv_st = nullptr,
                int conv_k = 0, const uint8_t* pf = nullptr, int pf_bytes = 0) {
    const int sm = gdn_dyn_smem(dk, dv);
    if (T == 1 && dk == 128 && dv == 128 && conv_k == 4 && qkv_raw && conv_w && conv_st && nv > 0)
        gdn_decode_t1_k<<<nv, 256, sm>>>(aa, bb, S, A_log, dt_bias, o, nk, nv, qkv_raw, conv_w, conv_st, pf,
                                         pf_bytes);
    else
        gdn_prefill_steps_k<<<nv, gdn_threads(dv), sm>>>(mix, aa, bb, S, A_log, dt_bias, o, T, nk, nv, dk, dv,
                                                         qkv_dim, 1e-6f, qkv_raw, conv_w, conv_st, conv_k);
}

int gemm_fp8_tile(int T, int n) {
    (void)n;
    int tile = 4096;
    while (T > 0 && static_cast<size_t>(T) * tile * sizeof(float) > 48 * 1024 && tile > 256) tile /= 2;
    return tile;
}

void launch_gemm_fp8(const GpuW& w, const float* X, float* Y, int T, int add) {
    int blocks = 0, threads = 0;
    gemv_grid(w, blocks, threads);
    const int tile = gemm_fp8_tile(T, w.cols);
    const size_t smem = static_cast<size_t>(T) * tile * sizeof(float);
    if (T == 3) {
        gemm_fp8_t3_k<<<blocks, threads, smem>>>(w.data, w.scale, X, Y, w.rows, w.cols, tile, add);
        return;
    }
    if (T == 4) {
        gemm_fp8_t4_k<<<blocks, threads, smem>>>(w.data, w.scale, X, Y, w.rows, w.cols, tile, add);
        return;
    }
    gemm_fp8_hw_k<<<blocks, threads, smem>>>(w.data, w.scale, X, Y, w.rows, w.cols, T, tile, add);
}

void launch_gemm_fp8_dual(const GpuW& w1, const GpuW& w2, const float* X, float* Y1, float* Y2, int T,
                          int fuse_swiglu) {
    if (T == 3 && w1.q == QuantKind::FP8_E4M3_B128 && w2.q == QuantKind::FP8_E4M3_B128 && w1.data &&
        w2.data && w1.rows == w2.rows && w1.cols == w2.cols && w1.rows > 0 && w1.cols > 0) {
        int blocks = 0, threads = 0;
        gemv_grid(w1, blocks, threads);
        const int tile = gemm_fp8_tile(3, w1.cols);
        const size_t smem = static_cast<size_t>(3) * tile * sizeof(float);
        gemm_fp8_t3_dual_k<<<blocks, threads, smem>>>(w1.data, w1.scale, w2.data, w2.scale, X, Y1, Y2, w1.rows,
                                                      w1.cols, tile, fuse_swiglu);
        return;
    }
    launch_linear(w1, X, Y1, T);
    launch_linear(w2, X, Y2, T);
    if (fuse_swiglu && Y1 && Y2)
        swiglu_n_k<<<(((w1.rows * T + 3) / 4) + 255) / 256, 256>>>(Y1, Y2, Y1, w1.rows * T);
}

void launch_fp8_e4m3_mma(const GpuW& w, const float* x, float* y) {
    const int warps = 16;
    const int threads = warps * 32;
    const int row_blocks = (w.rows + warps * 16 - 1) / (warps * 16);
    const int splits = 1;
    const size_t smem = static_cast<size_t>(warps) * 16 * 32 + 8 * 32 + 33 * sizeof(float);
    if (splits > 1) {
        zero_f32_k<<<(w.rows + 255) / 256, 256>>>(y, w.rows);
        dim3 grid(row_blocks, splits);
        gemv_fp8_e4m3_mma_k<<<grid, threads, smem>>>(w.data, w.scale, x, y, w.rows, w.cols);
    } else {
        gemv_fp8_e4m3_mma_k<<<row_blocks, threads, smem>>>(w.data, w.scale, x, y, w.rows, w.cols);
    }
}

void launch_fp8_tc(const GpuW& w, const float* X, float* Y, int T) {
    const int warps = 8;
    const int threads = warps * 32;
    const int blocks = (w.rows + warps * 16 - 1) / (warps * 16);
    const size_t smem = static_cast<size_t>(warps) * 16 * 16 * sizeof(__half) + 8 * 16 * sizeof(__half);
    gemm_fp8_tc_k<<<blocks, threads, smem>>>(w.data, w.scale, X, Y, w.rows, w.cols, T);
}

bool fp8_tc_selftest() {
    const int m = 64, n = 256;
    std::vector<uint8_t> W(static_cast<size_t>(m) * n);
    std::vector<float> scale((m / 128 + 1) * (n / 128), 0.05f);
    std::vector<float> x(n), y_cpu(m), y_gpu(m);
    for (int i = 0; i < m * n; ++i) W[static_cast<size_t>(i)] = static_cast<uint8_t>((i * 17 + 3) & 0x7f);
    for (int i = 0; i < n; ++i) x[i] = 0.01f * static_cast<float>((i % 13) - 6);
    ops::gemv_fp8(W.data(), scale.data(), x.data(), y_cpu.data(), m, n, 128);
    std::vector<uint8_t> tiled(static_cast<size_t>(m) * n);
    const int nt = n / 32;
    for (int r = 0; r < m; ++r) {
        const int rt = r / 16, ri = r % 16;
        for (int kt = 0; kt < nt; ++kt)
            std::memcpy(tiled.data() + (static_cast<size_t>(rt) * nt + kt) * 512 + ri * 32,
                        W.data() + static_cast<size_t>(r) * n + kt * 32, 32);
    }
    uint8_t* dW = nullptr;
    float *dS = nullptr, *dX = nullptr, *dY = nullptr;
    const size_t pack_bytes = static_cast<size_t>(m) * static_cast<size_t>(fp8_pack_cols(n));
    const size_t dw_bytes = pack_bytes > W.size() ? pack_bytes : W.size();
    if (cudaMalloc(&dW, dw_bytes) != cudaSuccess) return false;
    if (cudaMalloc(&dS, scale.size() * 4) != cudaSuccess) {
        cudaFree(dW);
        return false;
    }
    if (cudaMalloc(&dX, n * 4) != cudaSuccess) {
        cudaFree(dW);
        cudaFree(dS);
        return false;
    }
    if (cudaMalloc(&dY, m * 4) != cudaSuccess) {
        cudaFree(dW);
        cudaFree(dS);
        cudaFree(dX);
        return false;
    }
    cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dS, scale.data(), scale.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dX, x.data(), n * 4, cudaMemcpyHostToDevice);
    GpuW tw;
    tw.data = dW;
    tw.scale = dS;
    tw.q = QuantKind::FP8_E4M3_B128;
    tw.rows = m;
    tw.cols = n;
    launch_fp8_tc(tw, dX, dY, 1);
    cudaMemcpy(y_gpu.data(), dY, m * 4, cudaMemcpyDeviceToHost);
    float maxe_f16 = 0.f;
    for (int i = 0; i < m; ++i) maxe_f16 = std::max(maxe_f16, std::fabs(y_cpu[i] - y_gpu[i]));
    const bool f16_ok = maxe_f16 < 0.08f && cudaGetLastError() == cudaSuccess;

    cudaMemcpy(dW, tiled.data(), tiled.size(), cudaMemcpyHostToDevice);
    tw.fp8_tiled = true;
    cudaMemset(dY, 0, m * 4);
    launch_fp8_e4m3_mma(tw, dX, dY);
    cudaMemcpy(y_gpu.data(), dY, m * 4, cudaMemcpyDeviceToHost);
    float maxe_e4 = 0.f, maxa = 0.f;
    for (int i = 0; i < m; ++i) {
        maxe_e4 = std::max(maxe_e4, std::fabs(y_cpu[i] - y_gpu[i]));
        maxa = std::max(maxa, std::fabs(y_cpu[i]));
    }
    const bool e4_ok = maxe_e4 < std::max(0.15f * maxa, 0.08f) && cudaGetLastError() == cudaSuccess;
    // Native e4m3 MMA is correct but ~2x slower than hw-cvt GEMV on this Ada GEMV
    // workload (16 rows/warp vs 1 row/warp bandwidth). Keep it off the T=1 hot path.
    g_fp8_e4m3_mma = false;
    (void)e4_ok;

    std::vector<uint8_t> kmaj(pack_bytes);
    pack_fp8_kmajor_host(kmaj.data(), W.data(), m, n);
    cudaMemcpy(dW, kmaj.data(), kmaj.size(), cudaMemcpyHostToDevice);
    tw.fp8_tiled = false;
    tw.fp8_kmajor = true;
    cudaMemset(dY, 0, m * 4);
    launch_gemv(tw, dX, dY, 0);
    cudaMemcpy(y_gpu.data(), dY, m * 4, cudaMemcpyDeviceToHost);
    float maxe_km = 0.f;
    for (int i = 0; i < m; ++i) maxe_km = std::max(maxe_km, std::fabs(y_cpu[i] - y_gpu[i]));
    const bool km_ok = maxe_km < 0.08f && cudaGetLastError() == cudaSuccess;
    std::fprintf(stderr, "fp8_mma_f16_err=%.4f ok=%d  e4m3_err=%.4f ok=%d  kmajor_err=%.4f ok=%d\n", maxe_f16,
                 f16_ok ? 1 : 0, maxe_e4, e4_ok ? 1 : 0, maxe_km, km_ok ? 1 : 0);
    cudaFree(dW);
    cudaFree(dS);
    cudaFree(dX);
    cudaFree(dY);
    if (!km_ok) throw std::runtime_error("fp8 kmajor GEMV selftest failed");
    return f16_ok;
}

void launch_gemm_q8(const GpuW& w, const float* X, float* Y, int T) {
    int blocks = 0, threads = 0;
    gemv_grid(w, blocks, threads);
    int tile = 2048;
    while (T > 0 && static_cast<size_t>(T) * (tile / 32) * kXsPad * sizeof(float) > 24 * 1024 && tile > 256)
        tile /= 2;
    const size_t smem = static_cast<size_t>(T) * (tile / 32) * kXsPad * sizeof(float);
    gemm_q8_soa_k<<<blocks, threads, smem>>>(reinterpret_cast<const int8_t*>(w.data),
                                             reinterpret_cast<const __half*>(w.scale), X, Y, w.rows, w.cols,
                                             T, tile);
}

void launch_linear(const GpuW& w, const float* X, float* Y, int T, int add) {
    if (!w.data || w.rows <= 0 || w.cols <= 0 || T <= 0) return;
    if (!add && (w.q == QuantKind::F16 || w.q == QuantKind::BF16) && launch_cublas_f16(w, X, Y, T)) return;
    if (w.q == QuantKind::FP8_E4M3_B128 && T > 1) {
        if (!add && w.fp8_tiled && g_fp8_e4m3_mma) {
            for (int t = 0; t < T; ++t)
                launch_fp8_e4m3_mma(w, X + static_cast<size_t>(t) * w.cols, Y + static_cast<size_t>(t) * w.rows);
            return;
        }
        int t = 0;
        while (t < T) {
            const int tt = T - t > 4 ? 4 : T - t;
            launch_gemm_fp8(w, X + static_cast<size_t>(t) * w.cols, Y + static_cast<size_t>(t) * w.rows, tt, add);
            t += tt;
        }
        return;
    }
    if (!add && w.q == QuantKind::Q8_0 && T > 1) {
        int t = 0;
        while (t < T) {
            const int tt = T - t > 4 ? 4 : T - t;
            if (tt == 1)
                launch_gemv(w, X + static_cast<size_t>(t) * w.cols, Y + static_cast<size_t>(t) * w.rows);
            else
                launch_gemm_q8(w, X + static_cast<size_t>(t) * w.cols, Y + static_cast<size_t>(t) * w.rows, tt);
            t += tt;
        }
        return;
    }
    if (T == 1) {
        launch_gemv(w, X, Y, add);
        return;
    }
    for (int t = 0; t < T; ++t)
        launch_gemv(w, X + static_cast<size_t>(t) * w.cols, Y + static_cast<size_t>(t) * w.rows, add);
}

void launch_rms(const float* x, const float* g, float* y, int n, float eps) {
    const int th = n >= 1024 ? 256 : 128;
    rmsnorm_k<<<1, th>>>(x, g, y, n, eps);
}

void launch_reduce_ss(const float* x, float* d_ss, int n) {
    const int th = n >= 1024 ? 256 : 128;
    reduce_ss_k<<<1, th>>>(x, d_ss, n);
}

void launch_rms_ss(const float* x, const float* g, const float* d_ss, float* y, int n, float eps) {
    const int th = n >= 1024 ? 256 : 128;
    apply_rms_ss_k<<<1, th>>>(x, g, d_ss, y, n, eps);
}

void launch_add(float* x, const float* y, int n) {
    add_res_k<<<(n + 255) / 256, 256>>>(x, y, n);
}

struct GpuLayer {
    LayerKind kind = LayerKind::GatedDeltaNet;
    int slot = 0;
    GpuW wqkv, wz, wa, wb, wo;
    GpuW wq, wk, wv, wo_a;
    GpuW wg, wu, wd;
    const float* attn_norm = nullptr;
    const float* ffn_norm = nullptr;
    const float* A_log = nullptr;
    const float* dt_bias = nullptr;
    const float* conv_w = nullptr;
    const float* gnorm = nullptr;
    const float* q_norm = nullptr;
    const float* k_norm = nullptr;
    int nk = 0, nv = 0, dk = 0, dv = 0, conv_k = 0, gnorm_n = 0;
    int nq = 0, nkv = 0, hd = 0, rotary = 0;
    int inter = 0;
    float eps = 1e-6f;
    float theta = 1e7f;
};

class EngineImpl final : public Engine {
public:
    EngineImpl(WeightStore& store, int ctx) : store_(&store), ctx_(ctx) {
        if (cublasCreate(&g_blas) != CUBLAS_STATUS_SUCCESS) g_blas = nullptr;
        if (g_blas) {
            cublasSetMathMode(g_blas, CUBLAS_TENSOR_OP_MATH);
            cublasSetStream(g_blas, cudaStreamPerThread);
        }
        g_fp8_tc = fp8_tc_selftest();
        std::fprintf(stderr, "fp8_tensor_core=%d e4m3_mma=%d\n", g_fp8_tc ? 1 : 0, g_fp8_e4m3_mma ? 1 : 0);
        {
            const int xs_bytes = kFp8XsCap * static_cast<int>(sizeof(float));
            const cudaError_t ae =
                cudaFuncSetAttribute(gdn_prefill_steps_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 1024);
            if (ae != cudaSuccess) cudaGetLastError();
            const cudaError_t a2 =
                cudaFuncSetAttribute(gdn_decode_t1_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 1024);
            if (a2 != cudaSuccess) cudaGetLastError();
            const cudaError_t be =
                cudaFuncSetAttribute(gemm_fp8_t3_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 1024);
            if (be != cudaSuccess) cudaGetLastError();
            const cudaError_t ce =
                cudaFuncSetAttribute(gemm_fp8_t3_dual_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 1024);
            if (ce != cudaSuccess) cudaGetLastError();
            cudaError_t xe;
            xe = cudaFuncSetAttribute(gemv_fp8_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_fp8_2row_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_fp8_2row_add_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_fp8_dual_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_fp8_2row_dual_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_fp8_2row_pair_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_fp8_2row_st_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_fp8_2row_add_st_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_fp8_2row_add_wd_k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      kWdXs * static_cast<int>(sizeof(float)));
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_fp8_2row_dual_st_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_fp8_2row_pair_st_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_fp8_2row_attn_in_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
        }
        if (cudaStreamCreateWithFlags(&bak_stream_, cudaStreamNonBlocking) != cudaSuccess) bak_stream_ = nullptr;
        build();
    }
    ~EngineImpl() override {
        release();
        g_xf16 = nullptr;
        g_xf16_n = 0;
        if (g_blas) {
            cublasDestroy(g_blas);
            g_blas = nullptr;
        }
    }

    void prefill(const int32_t* ids, int n) override {
        CUDA_CHECK(cudaMemset(d_S_, 0, s_bytes_));
        CUDA_CHECK(cudaMemset(d_conv_, 0, conv_bytes_));
        CUDA_CHECK(cudaMemset(d_kcache_, 0, kv_bytes_));
        CUDA_CHECK(cudaMemset(d_vcache_, 0, kv_bytes_));
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_pos_, &zero, 4, cudaMemcpyHostToDevice));
        pos_ = 0;
        if (n >= 2 && n <= kPfGraphMax && pf_graph_execs_[n]) {
            CUDA_CHECK(cudaMemcpy(d_toks_, ids, sizeof(int) * n, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaGraphLaunch(pf_graph_execs_[n], cudaStreamPerThread));
            pos_ = n;
            if (n > 0) CUDA_CHECK(cudaMemcpy(d_tok_, ids + (n - 1), 4, cudaMemcpyHostToDevice));
        } else if (n <= 1 || pf_cap_ <= 1) {
            for (int t = 0; t < n; ++t) decode_token(ids[t]);
        } else {
            launch_prefill(ids, n);
            maybe_capture_prefill(n);
        }
        if (!graph_exec_) {
            CUDA_CHECK(cudaDeviceSynchronize());
            maybe_capture();
        }
        finish_logits();
        CUDA_CHECK(cudaMemcpy(d_tok_, d_best_, 4, cudaMemcpyDeviceToDevice));
    }

    void decode_token(int32_t token) override {
        if (h_tok_pin_) {
            *h_tok_pin_ = token;
            CUDA_CHECK(cudaMemcpyAsync(d_tok_, h_tok_pin_, 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
        } else {
            CUDA_CHECK(cudaMemcpy(d_tok_, &token, 4, cudaMemcpyHostToDevice));
        }
        if (graph_exec_ && pos_ > 0) {
            CUDA_CHECK(cudaGraphLaunch(graph_exec_, cudaStreamPerThread));
        } else {
            launch_decode();
        }
        finish_logits();
        ++pos_;
    }

    void decode_steps(int n) override {
        if (n <= 0) return;
        if (n > kGenCap) {
            for (int i = 0; i < n; ++i) decode_token(greedy());
            return;
        }
        CUDA_CHECK(cudaMemsetAsync(d_gen_n_, 0, 4, cudaStreamPerThread));
        for (int i = 0; i < n; ++i) {
            if (graph_exec_ && pos_ > 0)
                CUDA_CHECK(cudaGraphLaunch(graph_exec_, cudaStreamPerThread));
            else
                launch_decode();
            ++pos_;
        }
        finish_logits();
    }

    void copy_gen_tokens(int32_t* host, int n) override {
        if (!host || n <= 0) return;
        const int take = n > kGenCap ? kGenCap : n;
        CUDA_CHECK(cudaMemcpy(host, d_gen_out_, sizeof(int) * take, cudaMemcpyDeviceToHost));
    }

    void copy_logits(float* host) const override {
        if (h_pin_) {
            CUDA_CHECK(cudaMemcpyAsync(h_pin_, d_logits_, sizeof(float) * vocab_, cudaMemcpyDeviceToHost,
                                       cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
            std::memcpy(host, h_pin_, sizeof(float) * vocab_);
        } else {
            CUDA_CHECK(cudaMemcpy(host, d_logits_, sizeof(float) * vocab_, cudaMemcpyDeviceToHost));
        }
    }

    int32_t greedy() const override { return last_tok_; }
    int pos() const override { return pos_; }
    int vocab() const override { return vocab_; }
    int hidden() const override { return hidden_; }
    int max_batch() const override { return max_batch_; }

    void decode_tokens(const int32_t* tokens, const int* poss, int B) override {
        if (B <= 0) return;
        if (B == 1) {
            CUDA_CHECK(cudaMemcpy(d_pos_, poss, 4, cudaMemcpyHostToDevice));
            pos_ = poss[0];
            decode_token(tokens[0]);
            return;
        }
        if (B > max_batch_) throw std::runtime_error("decode_tokens B exceeds max_batch");
        CUDA_CHECK(cudaMemcpy(d_toks_, tokens, sizeof(int) * B, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos_b_, poss, sizeof(int) * B, cudaMemcpyHostToDevice));
        launch_decode_batch(B);
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }

    void copy_logits_n(float* host, int B) const override {
        if (B <= 0) return;
        const int n = std::min(B, max_batch_);
        CUDA_CHECK(cudaMemcpy(host, d_logits_, sizeof(float) * static_cast<size_t>(vocab_) * n,
                              cudaMemcpyDeviceToHost));
    }

    void replicate_slot0(int n_slots) override {
        if (n_slots <= 1) return;
        n_slots = std::min(n_slots, max_batch_);
        const size_t s1 = s_bytes_ / static_cast<size_t>(max_batch_);
        const size_t c1 = conv_bytes_ / static_cast<size_t>(max_batch_);
        const size_t k1 = kv_bytes_ / static_cast<size_t>(max_batch_);
        for (int b = 1; b < n_slots; ++b) {
            if (s1) CUDA_CHECK(cudaMemcpy(reinterpret_cast<uint8_t*>(d_S_) + s1 * b, d_S_, s1, cudaMemcpyDeviceToDevice));
            if (c1)
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<uint8_t*>(d_conv_) + c1 * b, d_conv_, c1,
                                      cudaMemcpyDeviceToDevice));
            if (k1) {
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<uint8_t*>(d_kcache_) + k1 * b, d_kcache_, k1,
                                      cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<uint8_t*>(d_vcache_) + k1 * b, d_vcache_, k1,
                                      cudaMemcpyDeviceToDevice));
            }
        }
    }

    int spec_verify(const int32_t* toks, int T, int32_t* preds) override {
        if (T <= 0 || !toks) return 0;
        if (T == 1) {
            decode_token(toks[0]);
            if (preds) preds[0] = last_tok_;
            return 1;
        }
        T = std::min(T, std::min(pf_cap_, logit_rows_));
        const int pos0 = pos_;
        if (bak_stream_) CUDA_CHECK(cudaStreamSynchronize(bak_stream_));
        if (bak_stream_) {
            if (s_bytes_)
                CUDA_CHECK(cudaMemcpyAsync(d_S_bak_, d_S_, s_bytes_, cudaMemcpyDeviceToDevice, bak_stream_));
            if (conv_bytes_)
                CUDA_CHECK(cudaMemcpyAsync(d_conv_bak_, d_conv_, conv_bytes_, cudaMemcpyDeviceToDevice, bak_stream_));
        } else {
            if (s_bytes_) CUDA_CHECK(cudaMemcpy(d_S_bak_, d_S_, s_bytes_, cudaMemcpyDeviceToDevice));
            if (conv_bytes_) CUDA_CHECK(cudaMemcpy(d_conv_bak_, d_conv_, conv_bytes_, cudaMemcpyDeviceToDevice));
        }
        CUDA_CHECK(cudaMemcpy(d_toks_, toks, sizeof(int) * T, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos_, &pos0, 4, cudaMemcpyHostToDevice));
        if (T == 4 && spec_graph_exec_) {
            CUDA_CHECK(cudaGraphLaunch(spec_graph_exec_, cudaStreamPerThread));
        } else {
            launch_spec_chunk(T);
        }
        int hpred[8];
        CUDA_CHECK(cudaMemcpy(hpred, d_best_n_, sizeof(int) * T, cudaMemcpyDeviceToHost));
        if (preds) {
            for (int t = 0; t < T; ++t) preds[t] = hpred[t];
        }
        int k = 1;
        while (k < T && hpred[k - 1] == toks[k]) ++k;
        if (k < T) {
            if (bak_stream_) CUDA_CHECK(cudaStreamSynchronize(bak_stream_));
            if (s_bytes_) CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
            if (conv_bytes_) CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_toks_, toks, sizeof(int) * k, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_pos_, &pos0, 4, cudaMemcpyHostToDevice));
            launch_spec_chunk(k);
        }
        last_tok_ = hpred[k - 1];
        pos_ = pos0 + k;
        set_pos_k<<<1, 1>>>(d_pos_, pos_);
        CUDA_CHECK(cudaMemcpy(d_h_, d_h_seq_ + static_cast<size_t>(k - 1) * hidden_, sizeof(float) * hidden_,
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_tok_, toks + (k - 1), 4, cudaMemcpyHostToDevice));
        return k;
    }

private:
    const TensorDesc& must(const std::string& ir) const {
        const TensorDesc* t = store_->table().find(ir);
        if (!t) throw std::runtime_error("cuda missing weight " + ir);
        return *t;
    }

    void* alloc(size_t n) {
        void* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, n ? n : 4));
        allocs_.push_back(p);
        return p;
    }

    GpuW upload_w(const TensorDesc& t, int rows, int cols) {
        GpuW w;
        w.q = t.quant;
        w.rows = rows;
        w.cols = cols;
        if (t.data.empty()) throw std::runtime_error("empty tensor " + t.ir_name);
        if (t.quant == QuantKind::Q8_0) {
            if (cols <= 0 || (cols & 31) != 0)
                throw std::runtime_error("Q8_0 cols must be multiple of 32: " + t.ir_name);
            const int nb = cols / 32;
            const size_t qn = static_cast<size_t>(rows) * static_cast<size_t>(cols);
            const size_t sn = static_cast<size_t>(rows) * static_cast<size_t>(nb);
            std::vector<int8_t> quants(qn);
            std::vector<uint16_t> scales(sn);
            const uint8_t* src = t.data.data();
            for (int r = 0; r < rows; ++r) {
                const uint8_t* srow = src + static_cast<size_t>(r) * static_cast<size_t>(nb) * 34;
                int8_t* qrow = quants.data() + static_cast<size_t>(r) * cols;
                uint16_t* drow = scales.data() + static_cast<size_t>(r) * nb;
                for (int b = 0; b < nb; ++b) {
                    const uint8_t* blk = srow + static_cast<size_t>(b) * 34;
                    drow[b] = static_cast<uint16_t>(blk[0] | (static_cast<uint16_t>(blk[1]) << 8));
                    std::memcpy(qrow + b * 32, blk + 2, 32);
                }
            }
            void* pq = alloc(qn);
            void* ps = alloc(sn * sizeof(uint16_t));
            CUDA_CHECK(cudaMemcpy(pq, quants.data(), qn, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ps, scales.data(), sn * sizeof(uint16_t), cudaMemcpyHostToDevice));
            w.data = static_cast<const uint8_t*>(pq);
            w.scale = static_cast<const float*>(ps);
            return w;
        }
        if (t.quant == QuantKind::Q6_K) {
            if (cols <= 0 || (cols % 256) != 0 || (cols % 128) != 0)
                throw std::runtime_error("Q6_K cols must be multiple of 256: " + t.ir_name);
            constexpr int bsz = 256 / 2 + 256 / 4 + 256 / 16 + 2;
            const int nblk = cols / 256;
            const size_t rowb = static_cast<size_t>(nblk) * static_cast<size_t>(bsz);
            const int br = 128, bc = 128;
            const int nb_r = (rows + br - 1) / br;
            const int nb_c = cols / bc;
            std::vector<uint8_t> q(static_cast<size_t>(rows) * static_cast<size_t>(cols));
            std::vector<float> sc(static_cast<size_t>(nb_r) * static_cast<size_t>(nb_c));
            std::vector<float> block(static_cast<size_t>(br) * static_cast<size_t>(cols));
            for (int bi = 0; bi < nb_r; ++bi) {
                const int r0 = bi * br;
                const int r1 = r0 + br < rows ? r0 + br : rows;
                const int rh = r1 - r0;
                for (int r = 0; r < rh; ++r)
                    ops::dequant_q6_k(t.data.data() + static_cast<size_t>(r0 + r) * rowb,
                                      block.data() + static_cast<size_t>(r) * cols, cols);
                for (int bj = 0; bj < nb_c; ++bj) {
                    const int c0 = bj * bc;
                    float amax = 0.f;
                    for (int r = 0; r < rh; ++r) {
                        const float* row = block.data() + static_cast<size_t>(r) * cols + c0;
                        for (int c = 0; c < bc; ++c) amax = std::max(amax, std::fabs(row[c]));
                    }
                    const float s = amax > 0.f ? amax / 448.f : 1.f;
                    sc[static_cast<size_t>(bi) * nb_c + bj] = s;
                    const float inv = 1.f / s;
                    for (int r = 0; r < rh; ++r) {
                        const float* row = block.data() + static_cast<size_t>(r) * cols + c0;
                        uint8_t* dst = q.data() + static_cast<size_t>(r0 + r) * cols + c0;
                        for (int c = 0; c < bc; ++c) dst[c] = ops::f32_to_e4m3(row[c] * inv);
                    }
                }
            }
            std::vector<uint8_t> packed(static_cast<size_t>(rows) * static_cast<size_t>(fp8_pack_cols(cols)));
            pack_fp8_kmajor_host(packed.data(), q.data(), rows, cols);
            void* pq = alloc(packed.size());
            void* ps = alloc(sc.size() * sizeof(float));
            CUDA_CHECK(cudaMemcpy(pq, packed.data(), packed.size(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ps, sc.data(), sc.size() * sizeof(float), cudaMemcpyHostToDevice));
            w.q = QuantKind::FP8_E4M3_B128;
            w.data = static_cast<const uint8_t*>(pq);
            w.scale = static_cast<const float*>(ps);
            w.fp8_kmajor = true;
            return w;
        }
        if (t.quant == QuantKind::FP8_E4M3_B128 && rows > 0 && cols > 0) {
            if ((cols % 128) != 0)
                throw std::runtime_error("FP8 cols must be multiple of 128: " + t.ir_name);
            std::vector<uint8_t> packed(static_cast<size_t>(rows) * static_cast<size_t>(fp8_pack_cols(cols)));
            pack_fp8_kmajor_host(packed.data(), t.data.data(), rows, cols);
            void* p = alloc(packed.size());
            CUDA_CHECK(cudaMemcpy(p, packed.data(), packed.size(), cudaMemcpyHostToDevice));
            w.data = static_cast<const uint8_t*>(p);
            w.fp8_kmajor = true;
            if (!t.scale.empty()) {
                void* s = alloc(t.scale.size() * sizeof(float));
                CUDA_CHECK(cudaMemcpy(s, t.scale.data(), t.scale.size() * sizeof(float), cudaMemcpyHostToDevice));
                w.scale = static_cast<const float*>(s);
            }
            return w;
        }
        void* p = alloc(t.data.size());
        CUDA_CHECK(cudaMemcpy(p, t.data.data(), t.data.size(), cudaMemcpyHostToDevice));
        w.data = static_cast<const uint8_t*>(p);
        if (!t.scale.empty()) {
            void* s = alloc(t.scale.size() * sizeof(float));
            CUDA_CHECK(cudaMemcpy(s, t.scale.data(), t.scale.size() * sizeof(float), cudaMemcpyHostToDevice));
            w.scale = static_cast<const float*>(s);
        }
        return w;
    }

    // Re-quantize a large BF16 matrix to block-scaled e4m3 so T=1 hits the FP8 GEMV path.
    GpuW upload_bf16_as_fp8(const TensorDesc& t, int rows, int cols) {
        if (t.quant != QuantKind::BF16 || rows <= 0 || cols <= 0 || (cols % 128) != 0)
            return upload_w(t, rows, cols);
        if (t.data.size() < static_cast<size_t>(rows) * static_cast<size_t>(cols) * 2)
            return upload_w(t, rows, cols);
        const uint16_t* src = reinterpret_cast<const uint16_t*>(t.data.data());
        const int br = 128, bc = 128;
        const int nb_r = (rows + br - 1) / br;
        const int nb_c = cols / bc;
        std::vector<uint8_t> q(static_cast<size_t>(rows) * static_cast<size_t>(cols));
        std::vector<float> sc(static_cast<size_t>(nb_r) * static_cast<size_t>(nb_c));
        for (int bi = 0; bi < nb_r; ++bi) {
            const int r0 = bi * br;
            const int r1 = r0 + br < rows ? r0 + br : rows;
            for (int bj = 0; bj < nb_c; ++bj) {
                const int c0 = bj * bc;
                float amax = 0.f;
                for (int r = r0; r < r1; ++r) {
                    const uint16_t* row = src + static_cast<size_t>(r) * cols + c0;
                    for (int c = 0; c < bc; ++c) {
                        uint32_t u = static_cast<uint32_t>(row[c]) << 16;
                        float v;
                        std::memcpy(&v, &u, 4);
                        amax = std::max(amax, std::fabs(v));
                    }
                }
                const float s = amax > 0.f ? amax / 448.f : 1.f;
                sc[static_cast<size_t>(bi) * nb_c + bj] = s;
                const float inv = 1.f / s;
                for (int r = r0; r < r1; ++r) {
                    const uint16_t* row = src + static_cast<size_t>(r) * cols + c0;
                    uint8_t* dst = q.data() + static_cast<size_t>(r) * cols + c0;
                    for (int c = 0; c < bc; ++c) {
                        uint32_t u = static_cast<uint32_t>(row[c]) << 16;
                        float v;
                        std::memcpy(&v, &u, 4);
                        dst[c] = ops::f32_to_e4m3(v * inv);
                    }
                }
            }
        }
        std::vector<uint8_t> packed(static_cast<size_t>(rows) * static_cast<size_t>(fp8_pack_cols(cols)));
        pack_fp8_kmajor_host(packed.data(), q.data(), rows, cols);
        GpuW w;
        w.q = QuantKind::FP8_E4M3_B128;
        w.rows = rows;
        w.cols = cols;
        w.fp8_kmajor = true;
        void* pq = alloc(packed.size());
        void* ps = alloc(sc.size() * sizeof(float));
        CUDA_CHECK(cudaMemcpy(pq, packed.data(), packed.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ps, sc.data(), sc.size() * sizeof(float), cudaMemcpyHostToDevice));
        w.data = static_cast<const uint8_t*>(pq);
        w.scale = static_cast<const float*>(ps);
        std::fprintf(stderr, "lm_head_fp8=1 kmajor=1 rows=%d cols=%d\n", rows, cols);
        return w;
    }

    const float* upload_f32(const TensorDesc& t) {
        if (t.quant != QuantKind::F32) throw std::runtime_error("expected F32 " + t.ir_name);
        void* p = alloc(t.data.size());
        CUDA_CHECK(cudaMemcpy(p, t.data.data(), t.data.size(), cudaMemcpyHostToDevice));
        return static_cast<const float*>(p);
    }

    void embed_launch() {
        const int th = 256;
        const int bl = (hidden_ + th - 1) / th;
        switch (embed_q_) {
        case QuantKind::F32:
            embed_f32_k<<<bl, th>>>(reinterpret_cast<const float*>(d_embed_), d_tok_, d_h_, hidden_);
            break;
        case QuantKind::F16:
            embed_f16_k<<<bl, th>>>(reinterpret_cast<const __half*>(d_embed_), d_tok_, d_h_, hidden_);
            break;
        case QuantKind::BF16:
            embed_bf16_k<<<bl, th>>>(reinterpret_cast<const uint16_t*>(d_embed_), d_tok_, d_h_, hidden_);
            break;
        default:
            throw std::runtime_error("unsupported embed quant on CUDA");
        }
    }

    void embed_batch(int T) {
        const int th = 256;
        const int bl = (hidden_ + th - 1) / th;
        dim3 grid(bl, T);
        switch (embed_q_) {
        case QuantKind::F32:
            embed_f32_batch_k<<<grid, th>>>(reinterpret_cast<const float*>(d_embed_), d_toks_, d_h_seq_, hidden_, T);
            break;
        case QuantKind::F16:
            embed_f16_batch_k<<<grid, th>>>(reinterpret_cast<const __half*>(d_embed_), d_toks_, d_h_seq_, hidden_, T);
            break;
        case QuantKind::BF16:
            embed_bf16_batch_k<<<grid, th>>>(reinterpret_cast<const uint16_t*>(d_embed_), d_toks_, d_h_seq_, hidden_,
                                             T);
            break;
        default:
            throw std::runtime_error("unsupported embed quant on CUDA");
        }
    }

    void launch_rms_batch(const float* x, const float* g, float* y, int n, int T, float eps) {
        const int th = n >= 1024 ? 256 : 128;
        rmsnorm_batch_k<<<T, th>>>(x, g, y, n, T, eps);
    }

    void launch_prefill_chunk(int pos0, int T) {
        const ModelDesc& m = store_->model();
        embed_batch(T);
        for (GpuLayer& L : layers_) {
            launch_rms_batch(d_h_seq_, L.attn_norm, d_xn_seq_, hidden_, T, L.eps);
            if (L.kind == LayerKind::GatedDeltaNet) {
                const int qdim = L.nk * L.dk;
                const int qkv_dim = qdim * 2 + L.nv * L.dv;
                const int zdim = L.nv * L.dv;
                launch_linear(L.wqkv, d_xn_seq_, d_qkv_seq_, T);
                launch_linear(L.wz, d_xn_seq_, d_z_seq_, T);
                launch_linear(L.wa, d_xn_seq_, d_aa_seq_, T);
                launch_linear(L.wb, d_xn_seq_, d_bb_seq_, T);
                float* conv_st = d_conv_ + static_cast<size_t>(L.slot) * qkv_dim * L.conv_k;
                if (pos0 == 0) {
                    conv1d_prefill_k<<<(qkv_dim + 127) / 128, 128>>>(d_qkv_seq_, L.conv_w, d_mix_seq_, conv_st, T,
                                                                     qkv_dim, L.conv_k);
                } else {
                    for (int t = 0; t < T; ++t) {
                        conv1d_upd_k<<<(qkv_dim + 127) / 128, 128>>>(d_qkv_seq_ + static_cast<size_t>(t) * qkv_dim,
                                                                     L.conv_w, conv_st,
                                                                     d_mix_seq_ + static_cast<size_t>(t) * qkv_dim,
                                                                     qkv_dim, L.conv_k);
                    }
                }
                uint16_t* S = d_S_ + static_cast<size_t>(L.slot) * L.nv * L.dk * L.dv;
                launch_gdn(d_mix_seq_, d_aa_seq_, d_bb_seq_, S, L.A_log, L.dt_bias, d_o_seq_, T, L.nk, L.nv,
                           L.dk, L.dv, qkv_dim);
                gated_rms_batch_k<<<T, 256>>>(d_o_seq_, d_z_seq_, L.gnorm, d_og_seq_, zdim, T, 1e-6f, L.gnorm_n);
                launch_linear(L.wo, d_og_seq_, d_h_seq_, T, 1);
            } else {
                const int qn = L.nq * L.hd;
                const int kn = L.nkv * L.hd;
                launch_linear(L.wq, d_xn_seq_, d_qg_seq_, T);
                launch_linear(L.wk, d_xn_seq_, d_k_seq_, T);
                launch_linear(L.wv, d_xn_seq_, d_v_seq_, T);
                dim3 sg((qn + 255) / 256, T);
                split_qg_batch_k<<<sg, 256>>>(d_qg_seq_, d_q_seq_, d_gate_seq_, qn, T);
                dim3 hg(L.nq, T);
                head_rms_batch_k<<<hg, 32>>>(d_q_seq_, L.q_norm, L.nq, L.hd, T, L.eps);
                dim3 hkg(L.nkv, T);
                head_rms_batch_k<<<hkg, 32>>>(d_k_seq_, L.k_norm, L.nkv, L.hd, T, L.eps);
                dim3 rg(2, std::max(L.nq, L.nkv), T);
                rope_batch_k<<<rg, 32>>>(d_q_seq_, d_k_seq_, L.nq, L.nkv, L.hd, L.rotary, pos0, T, L.theta);
                float* kc = d_kcache_ + static_cast<size_t>(L.slot) * ctx_ * kn;
                float* vc = d_vcache_ + static_cast<size_t>(L.slot) * ctx_ * kn;
                dim3 kg((kn + 127) / 128, T);
                store_kv_batch_k<<<kg, 128>>>(kc, d_k_seq_, pos0, T, kn);
                store_kv_batch_k<<<kg, 128>>>(vc, d_v_seq_, pos0, T, kn);
                const int tend = pos0 + T;
                const size_t sm = tend > 8192 ? 0 : static_cast<size_t>(tend) * sizeof(float);
                dim3 ag(L.nq, T);
                attn_prefill_k<<<ag, tend > 8192 ? 128 : 32, sm>>>(d_q_seq_, kc, vc, d_o_seq_, pos0, T, L.nq,
                                                                   L.nkv, L.hd);
                apply_gate_n_k<<<(qn * T + 255) / 256, 256>>>(d_o_seq_, d_gate_seq_, qn * T);
                launch_linear(L.wo_a, d_o_seq_, d_h_seq_, T, 1);
            }
            launch_rms_batch(d_h_seq_, L.ffn_norm, d_xn_seq_, hidden_, T, m.rms_eps);
            launch_linear(L.wg, d_xn_seq_, d_gate_mlp_seq_, T);
            launch_linear(L.wu, d_xn_seq_, d_up_seq_, T);
            swiglu_n_k<<<(((L.inter * T + 3) / 4) + 255) / 256, 256>>>(d_gate_mlp_seq_, d_up_seq_, d_gate_mlp_seq_,
                                                                        L.inter * T);
            launch_linear(L.wd, d_gate_mlp_seq_, d_h_seq_, T, 1);
        }
    }

    void launch_argmax(int* inc_pos = nullptr, int self_feed = 0) {
        const int th = 256;
        const int nblk = (vocab_ + th - 1) / th;
        if (nblk <= 0) return;
        argmax_partial_k<<<nblk, th>>>(d_logits_, vocab_, d_amax_, d_aidx_);
        argmax_final_k<<<1, 32>>>(d_amax_, d_aidx_, nblk, d_best_, inc_pos, self_feed ? d_tok_ : nullptr,
                                  self_feed ? d_gen_out_ : nullptr, self_feed ? d_gen_n_ : nullptr);
    }

    void launch_argmax_rows(int T) {
        const int th = 256;
        const int nblk = (vocab_ + th - 1) / th;
        if (nblk <= 0 || T <= 0) return;
        dim3 grid(nblk, T);
        argmax_partial_rows_k<<<grid, th>>>(d_logits_, vocab_, vocab_, d_amax_, d_aidx_);
        argmax_final_rows_k<<<1, 32>>>(d_amax_, d_aidx_, nblk, T, d_best_n_);
    }

    // Mid-decode T-token chunk: pos from d_pos_, always conv-upd, attn smem = ctx_.
    void launch_spec_chunk(int T) {
        const ModelDesc& m = store_->model();
        embed_batch(T);
        for (GpuLayer& L : layers_) {
            launch_rms_batch(d_h_seq_, L.attn_norm, d_xn_seq_, hidden_, T, L.eps);
            if (L.kind == LayerKind::GatedDeltaNet) {
                const int qdim = L.nk * L.dk;
                const int qkv_dim = qdim * 2 + L.nv * L.dv;
                const int zdim = L.nv * L.dv;
                launch_linear(L.wqkv, d_xn_seq_, d_qkv_seq_, T);
                launch_linear(L.wz, d_xn_seq_, d_z_seq_, T);
                launch_linear(L.wa, d_xn_seq_, d_aa_seq_, T);
                launch_linear(L.wb, d_xn_seq_, d_bb_seq_, T);
                float* conv_st = d_conv_ + static_cast<size_t>(L.slot) * qkv_dim * L.conv_k;
                for (int t = 0; t < T; ++t) {
                    conv1d_upd_k<<<(qkv_dim + 127) / 128, 128>>>(d_qkv_seq_ + static_cast<size_t>(t) * qkv_dim,
                                                                 L.conv_w, conv_st,
                                                                 d_mix_seq_ + static_cast<size_t>(t) * qkv_dim,
                                                                 qkv_dim, L.conv_k);
                }
                uint16_t* S = d_S_ + static_cast<size_t>(L.slot) * L.nv * L.dk * L.dv;
                launch_gdn(d_mix_seq_, d_aa_seq_, d_bb_seq_, S, L.A_log, L.dt_bias, d_o_seq_, T, L.nk, L.nv,
                           L.dk, L.dv, qkv_dim);
                gated_rms_batch_k<<<T, 256>>>(d_o_seq_, d_z_seq_, L.gnorm, d_og_seq_, zdim, T, 1e-6f, L.gnorm_n);
                launch_linear(L.wo, d_og_seq_, d_h_seq_, T, 1);
            } else {
                const int qn = L.nq * L.hd;
                const int kn = L.nkv * L.hd;
                launch_linear(L.wq, d_xn_seq_, d_qg_seq_, T);
                launch_linear(L.wk, d_xn_seq_, d_k_seq_, T);
                launch_linear(L.wv, d_xn_seq_, d_v_seq_, T);
                dim3 sg((qn + 255) / 256, T);
                split_qg_batch_k<<<sg, 256>>>(d_qg_seq_, d_q_seq_, d_gate_seq_, qn, T);
                dim3 hg(L.nq, T);
                head_rms_batch_k<<<hg, 32>>>(d_q_seq_, L.q_norm, L.nq, L.hd, T, L.eps);
                dim3 hkg(L.nkv, T);
                head_rms_batch_k<<<hkg, 32>>>(d_k_seq_, L.k_norm, L.nkv, L.hd, T, L.eps);
                dim3 rg(2, std::max(L.nq, L.nkv), T);
                rope_batch_dp_k<<<rg, 32>>>(d_q_seq_, d_k_seq_, L.nq, L.nkv, L.hd, L.rotary, d_pos_, T, L.theta);
                float* kc = d_kcache_ + static_cast<size_t>(L.slot) * ctx_ * kn;
                float* vc = d_vcache_ + static_cast<size_t>(L.slot) * ctx_ * kn;
                dim3 kg((kn + 127) / 128, T);
                store_kv_batch_dp_k<<<kg, 128>>>(kc, d_k_seq_, d_pos_, T, kn);
                store_kv_batch_dp_k<<<kg, 128>>>(vc, d_v_seq_, d_pos_, T, kn);
                const size_t sm = ctx_ > 8192 ? 0 : static_cast<size_t>(ctx_) * sizeof(float);
                dim3 ag(L.nq, T);
                attn_prefill_dp_k<<<ag, ctx_ > 8192 ? 128 : 32, sm>>>(d_q_seq_, kc, vc, d_o_seq_, d_pos_, T, L.nq,
                                                                      L.nkv, L.hd);
                apply_gate_n_k<<<(qn * T + 255) / 256, 256>>>(d_o_seq_, d_gate_seq_, qn * T);
                launch_linear(L.wo_a, d_o_seq_, d_h_seq_, T, 1);
            }
            launch_rms_batch(d_h_seq_, L.ffn_norm, d_xn_seq_, hidden_, T, m.rms_eps);
            launch_linear(L.wg, d_xn_seq_, d_gate_mlp_seq_, T);
            launch_linear(L.wu, d_xn_seq_, d_up_seq_, T);
            swiglu_n_k<<<(((L.inter * T + 3) / 4) + 255) / 256, 256>>>(d_gate_mlp_seq_, d_up_seq_, d_gate_mlp_seq_,
                                                                        L.inter * T);
            launch_linear(L.wd, d_gate_mlp_seq_, d_h_seq_, T, 1);
        }
        launch_rms_batch(d_h_seq_, d_final_norm_, d_xn_seq_, hidden_, T, store_->model().rms_eps);
        launch_linear(lm_head_, d_xn_seq_, d_logits_, T);
        launch_argmax_rows(T);
    }

    void launch_prefill_tail(int last_T, int n) {
        if (last_T > 0) {
            launch_rms(d_h_seq_ + static_cast<size_t>(last_T - 1) * hidden_, d_final_norm_, d_xn_, hidden_,
                       store_->model().rms_eps);
            launch_linear(lm_head_, d_xn_, d_logits_, 1);
            launch_argmax();
        }
        set_pos_k<<<1, 1>>>(d_pos_, n);
    }

    void launch_prefill(const int32_t* ids, int n) {
        int pos0 = 0;
        int last_T = 0;
        while (pos0 < n) {
            const int T = std::min(pf_cap_, n - pos0);
            CUDA_CHECK(cudaMemcpy(d_toks_, ids + pos0, sizeof(int) * T, cudaMemcpyHostToDevice));
            launch_prefill_chunk(pos0, T);
            pos0 += T;
            last_T = T;
        }
        launch_prefill_tail(last_T, n);
        pos_ = n;
        if (n > 0) CUDA_CHECK(cudaMemcpy(d_tok_, ids + (n - 1), 4, cudaMemcpyHostToDevice));
    }

    void launch_decode() {
        embed_launch();
        const ModelDesc& m = store_->model();
        for (GpuLayer& L : layers_) {
            const bool gdn = L.kind == LayerKind::GatedDeltaNet;
            const bool xrms_in =
                d_ss_ &&
                (gdn ? (L.wqkv.q == QuantKind::FP8_E4M3_B128 && L.wz.q == QuantKind::FP8_E4M3_B128 &&
                        L.wa.q == QuantKind::FP8_E4M3_B128 && L.wb.q == QuantKind::FP8_E4M3_B128 &&
                        L.wqkv.rows >= 4096 && L.wz.rows >= 4096 && L.wa.rows > 0 && L.wb.rows > 0 &&
                        L.wa.cols == L.wqkv.cols && L.wb.cols == L.wqkv.cols)
                     : (L.wq.q == QuantKind::FP8_E4M3_B128 && L.wk.q == QuantKind::FP8_E4M3_B128 &&
                        L.wv.q == QuantKind::FP8_E4M3_B128 && L.wq.rows >= 4096 && L.wk.rows > 0 &&
                        L.wv.rows == L.wk.rows && L.wk.cols == L.wq.cols));
            if (!xrms_in) launch_rms(d_h_, L.attn_norm, d_xn_, hidden_, L.eps);
            const float* xin = xrms_in ? d_h_ : d_xn_;
            const float* xg = xrms_in ? L.attn_norm : nullptr;
            const float* xss = nullptr;
            const float xeps = xrms_in ? L.eps : 0.f;
            if (gdn) {
                const int qdim = L.nk * L.dk;
                const int qkv_dim = qdim * 2 + L.nv * L.dv;
                const int zdim = L.nv * L.dv;
                const bool leftover_ok = L.wa.q == QuantKind::FP8_E4M3_B128 && L.wb.q == QuantKind::FP8_E4M3_B128 &&
                                         L.wa.rows > 0 && L.wb.rows > 0 && L.wa.rows <= 64 && L.wb.rows <= 64 &&
                                         L.wa.cols == L.wqkv.cols && L.wb.cols == L.wqkv.cols;
                launch_gemv_pair(L.wqkv, L.wz, xin, d_qkv_, d_z_, leftover_ok ? &L.wa : nullptr,
                                 leftover_ok ? d_aa_ : nullptr, leftover_ok ? &L.wb : nullptr,
                                 leftover_ok ? d_bb_ : nullptr, xg, xss, xeps);
                if (!leftover_ok) launch_gemv_dual(L.wa, L.wb, xin, d_aa_, d_bb_, 0, xg, xss, xeps);
                float* conv_st = d_conv_ + static_cast<size_t>(L.slot) * qkv_dim * L.conv_k;
                uint16_t* S = d_S_ + static_cast<size_t>(L.slot) * L.nv * L.dk * L.dv;
                const uint8_t* wo_pf = (L.wo.q == QuantKind::FP8_E4M3_B128 && L.wo.data && L.wo.rows > 0 &&
                                        L.wo.cols > 0)
                                           ? L.wo.data
                                           : nullptr;
                const int wo_pf_bytes =
                    wo_pf ? L.wo.rows * fp8_pack_cols(L.wo.cols) : 0;
                launch_gdn(nullptr, d_aa_, d_bb_, S, L.A_log, L.dt_bias, d_o_, 1, L.nk, L.nv, L.dk, L.dv,
                           qkv_dim, d_qkv_, L.conv_w, conv_st, L.conv_k, wo_pf, wo_pf_bytes);
                const bool wo_grms = d_ss_ && L.wo.q == QuantKind::FP8_E4M3_B128 && L.wo.rows >= 4096 &&
                                     L.wo.cols == zdim;
                if (wo_grms) {
                    launch_gemv(L.wo, d_o_, d_h_, 1, nullptr, L.gnorm, nullptr, 1e-6f, nullptr, nullptr, 0,
                                d_z_, L.gnorm_n, 1);
                } else {
                    gated_rms_k<<<1, 256>>>(d_o_, d_z_, L.gnorm, d_og_, zdim, 1e-6f, L.gnorm_n);
                    launch_gemv(L.wo, d_og_, d_h_, 1, nullptr, nullptr, nullptr, 0.f, nullptr, nullptr, 0,
                                nullptr, 0, 1);
                }
            } else {
                const int qn = L.nq * L.hd;
                const int kn = L.nkv * L.hd;
                if (L.wq.q == QuantKind::FP8_E4M3_B128 && L.wq.rows >= 4096 && L.wq.rows == qn * 2)
                    launch_gemv(L.wq, xin, d_q_, 0, nullptr, xg, xss, xeps, nullptr, d_gate_, qn);
                else {
                    launch_gemv(L.wq, xin, d_qg_, 0, nullptr, xg, xss, xeps);
                    split_qg_k<<<(qn + 255) / 256, 256>>>(d_qg_, d_q_, d_gate_, qn);
                }
                launch_gemv_dual(L.wk, L.wv, xin, d_k_, d_vtmp_, 0, xg, xss, xeps);
                float* kc = d_kcache_ + static_cast<size_t>(L.slot) * ctx_ * kn;
                float* vc = d_vcache_ + static_cast<size_t>(L.slot) * ctx_ * kn;
                const size_t sm = ctx_ > 8192
                                      ? static_cast<size_t>(L.hd) * sizeof(float)
                                      : (static_cast<size_t>(ctx_) + static_cast<size_t>(L.hd)) * sizeof(float);
                qk_attn_decode_k<<<L.nq, 128, sm>>>(d_q_, d_k_, d_vtmp_, L.q_norm, L.k_norm, kc, vc, d_o_,
                                                   d_pos_, L.nq, L.nkv, L.hd, L.rotary, L.theta, L.eps, ctx_);
                const bool gate_x = L.wo_a.q == QuantKind::FP8_E4M3_B128 && L.wo_a.rows >= 4096;
                if (gate_x)
                    launch_gemv(L.wo_a, d_o_, d_h_, 1, nullptr, nullptr, nullptr, 0.f, d_gate_);
                else {
                    apply_gate_k<<<(qn + 255) / 256, 256>>>(d_o_, d_gate_, qn);
                    launch_gemv(L.wo_a, d_o_, d_h_, 1);
                }
            }
            const bool mlp_xrms = d_ss_ && L.wg.q == QuantKind::FP8_E4M3_B128 && L.wg.rows >= 4096;
            if (mlp_xrms) {
                launch_gemv_dual(L.wg, L.wu, d_h_, d_gate_mlp_, d_up_, 1, L.ffn_norm, nullptr, m.rms_eps);
            } else {
                launch_rms(d_h_, L.ffn_norm, d_xn_, hidden_, m.rms_eps);
                launch_gemv_dual(L.wg, L.wu, d_xn_, d_gate_mlp_, d_up_, 1);
            }
            launch_gemv(L.wd, d_gate_mlp_, d_h_, 1);
        }
        const bool lh_xrms = d_ss_ && lm_head_.q == QuantKind::FP8_E4M3_B128 && lm_head_.rows >= 4096 &&
                             lm_head_.cols == hidden_;
        if (lh_xrms) {
            launch_gemv(lm_head_, d_h_, d_logits_, 0, nullptr, d_final_norm_, nullptr,
                        store_->model().rms_eps);
        } else {
            launch_rms(d_h_, d_final_norm_, d_xn_, hidden_, store_->model().rms_eps);
            launch_gemv(lm_head_, d_xn_, d_logits_);
        }
        launch_argmax(d_pos_, 1);
    }

    void launch_decode_batch(int B) {
        embed_batch(B);
        const ModelDesc& m = store_->model();
        for (GpuLayer& L : layers_) {
            launch_rms_batch(d_h_seq_, L.attn_norm, d_xn_seq_, hidden_, B, L.eps);
            if (L.kind == LayerKind::GatedDeltaNet) {
                const int qdim = L.nk * L.dk;
                const int qkv_dim = qdim * 2 + L.nv * L.dv;
                const int zdim = L.nv * L.dv;
                launch_linear(L.wqkv, d_xn_seq_, d_qkv_seq_, B);
                launch_linear(L.wz, d_xn_seq_, d_z_seq_, B);
                launch_linear(L.wa, d_xn_seq_, d_aa_seq_, B);
                launch_linear(L.wb, d_xn_seq_, d_bb_seq_, B);
                for (int b = 0; b < B; ++b) {
                    float* conv_st = d_conv_ + (static_cast<size_t>(b) * n_delta_ + L.slot) * qkv_dim * L.conv_k;
                    conv1d_upd_k<<<(qkv_dim + 127) / 128, 128>>>(d_qkv_seq_ + static_cast<size_t>(b) * qkv_dim,
                                                                 L.conv_w, conv_st,
                                                                 d_mix_seq_ + static_cast<size_t>(b) * qkv_dim,
                                                                 qkv_dim, L.conv_k);
                    uint16_t* S = d_S_ + (static_cast<size_t>(b) * n_delta_ + L.slot) * L.nv * L.dk * L.dv;
                    launch_gdn(d_mix_seq_ + static_cast<size_t>(b) * qkv_dim,
                               d_aa_seq_ + static_cast<size_t>(b) * L.nv, d_bb_seq_ + static_cast<size_t>(b) * L.nv,
                               S, L.A_log, L.dt_bias, d_o_seq_ + static_cast<size_t>(b) * zdim, 1, L.nk, L.nv,
                               L.dk, L.dv, qkv_dim);
                }
                gated_rms_batch_k<<<B, 256>>>(d_o_seq_, d_z_seq_, L.gnorm, d_og_seq_, zdim, B, 1e-6f, L.gnorm_n);
                launch_linear(L.wo, d_og_seq_, d_h_seq_, B, 1);
            } else {
                const int qn = L.nq * L.hd;
                const int kn = L.nkv * L.hd;
                launch_linear(L.wq, d_xn_seq_, d_qg_seq_, B);
                launch_linear(L.wk, d_xn_seq_, d_k_seq_, B);
                launch_linear(L.wv, d_xn_seq_, d_v_seq_, B);
                dim3 sg((qn + 255) / 256, B);
                split_qg_batch_k<<<sg, 256>>>(d_qg_seq_, d_q_seq_, d_gate_seq_, qn, B);
                dim3 hg(L.nq, B);
                head_rms_batch_k<<<hg, 32>>>(d_q_seq_, L.q_norm, L.nq, L.hd, B, L.eps);
                dim3 hkg(L.nkv, B);
                head_rms_batch_k<<<hkg, 32>>>(d_k_seq_, L.k_norm, L.nkv, L.hd, B, L.eps);
                for (int b = 0; b < B; ++b) {
                    CUDA_CHECK(cudaMemcpyAsync(d_pos_, d_pos_b_ + b, 4, cudaMemcpyDeviceToDevice,
                                               cudaStreamPerThread));
                    dim3 rg(2, std::max(L.nq, L.nkv));
                    rope_k<<<rg, 32>>>(d_q_seq_ + static_cast<size_t>(b) * qn, d_k_seq_ + static_cast<size_t>(b) * kn,
                                       L.nq, L.nkv, L.hd, L.rotary, d_pos_, L.theta);
                    float* kc = d_kcache_ + (static_cast<size_t>(b) * n_attn_ + L.slot) * ctx_ * kn;
                    float* vc = d_vcache_ + (static_cast<size_t>(b) * n_attn_ + L.slot) * ctx_ * kn;
                    store_kv_k<<<(kn + 127) / 128, 128>>>(kc, d_k_seq_ + static_cast<size_t>(b) * kn, d_pos_, kn);
                    store_kv_k<<<(kn + 127) / 128, 128>>>(vc, d_v_seq_ + static_cast<size_t>(b) * kn, d_pos_, kn);
                    const size_t sm = static_cast<size_t>(ctx_) * sizeof(float);
                    attn_decode_k<<<L.nq, 32, sm>>>(d_q_seq_ + static_cast<size_t>(b) * qn, kc, vc,
                                                    d_o_seq_ + static_cast<size_t>(b) * qn, d_pos_, L.nq, L.nkv,
                                                    L.hd);
                }
                apply_gate_n_k<<<(qn * B + 255) / 256, 256>>>(d_o_seq_, d_gate_seq_, qn * B);
                launch_linear(L.wo_a, d_o_seq_, d_h_seq_, B, 1);
            }
            launch_rms_batch(d_h_seq_, L.ffn_norm, d_xn_seq_, hidden_, B, m.rms_eps);
            launch_linear(L.wg, d_xn_seq_, d_gate_mlp_seq_, B);
            launch_linear(L.wu, d_xn_seq_, d_up_seq_, B);
            swiglu_n_k<<<(((L.inter * B + 3) / 4) + 255) / 256, 256>>>(d_gate_mlp_seq_, d_up_seq_, d_gate_mlp_seq_,
                                                                        L.inter * B);
            launch_linear(L.wd, d_gate_mlp_seq_, d_h_seq_, B, 1);
        }
        launch_rms_batch(d_h_seq_, d_final_norm_, d_xn_seq_, hidden_, B, store_->model().rms_eps);
        launch_linear(lm_head_, d_xn_seq_, d_logits_, B);
        for (int b = 0; b < B; ++b) inc_pos_k<<<1, 1>>>(d_pos_b_ + b);
    }

    void finish_logits() {
        if (h_best_pin_) {
            CUDA_CHECK(cudaMemcpyAsync(h_best_pin_, d_best_, 4, cudaMemcpyDeviceToHost, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
            last_tok_ = *h_best_pin_;
        } else {
            CUDA_CHECK(cudaMemcpy(&last_tok_, d_best_, 4, cudaMemcpyDeviceToHost));
        }
    }

    void abort_stream_capture() {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(cudaStreamPerThread, &st) != cudaSuccess) return;
        if (st == cudaStreamCaptureStatusNone) return;
        cudaGraph_t g = nullptr;
        cudaStreamEndCapture(cudaStreamPerThread, &g);
        if (g) cudaGraphDestroy(g);
    }

    bool instantiate_graph(cudaGraph_t g, cudaGraphExec_t* exec, const char* tag, int n) {
        const cudaError_t e = cudaGraphInstantiate(exec, g, nullptr, nullptr, 0);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "%s_inst n=%d err=%s\n", tag, n, cudaGetErrorString(e));
            if (g) cudaGraphDestroy(g);
            *exec = nullptr;
            return false;
        }
        return true;
    }

    // Pin GDN S in L2 (reused every token). miss=Normal — not the failed
    // miss=Streaming + persist d_h_ path. Leave ~1/4 L2 for wo prefetch.
    void persist_gdn_s() {
        if (!d_S_ || s_bytes_ < (1u << 20)) return;
        int max_persist = 0, l2 = 0;
        if (cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize, 0) != cudaSuccess) return;
        if (cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0) != cudaSuccess) return;
        if (max_persist <= 0 || l2 <= 0) return;
        size_t want = s_bytes_;
        const size_t leave = static_cast<size_t>(l2) / 4;
        if (static_cast<size_t>(l2) > leave && want + leave > static_cast<size_t>(l2))
            want = static_cast<size_t>(l2) - leave;
        if (want > static_cast<size_t>(max_persist)) want = static_cast<size_t>(max_persist);
        if (want < (1u << 20)) return;
        // 0x06 = cudaLimitPersistingL2CacheMaxSize (CUDA 11+). Some host
        // headers used by this nvcc don't export the enumerator name.
        if (cudaDeviceSetLimit(static_cast<cudaLimit>(0x06), want) != cudaSuccess) {
            cudaGetLastError();
            return;
        }
        cudaCtxResetPersistingL2Cache();
        cudaAccessPolicyWindow win{};
        win.base_ptr = d_S_;
        win.num_bytes = want;
        win.hitRatio = 1.f;
        win.hitProp = cudaAccessPropertyPersisting;
        win.missProp = cudaAccessPropertyNormal;
        cudaStreamAttrValue val{};
        val.accessPolicyWindow = win;
        if (cudaStreamSetAttribute(cudaStreamPerThread, cudaStreamAttributeAccessPolicyWindow, &val) != cudaSuccess)
            cudaGetLastError();
    }

    void maybe_capture() {
        if (graph_exec_) return;
        abort_stream_capture();
        cudaError_t e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "decode_capture_begin err=%s\n", cudaGetErrorString(e));
            return;
        }
        launch_decode();
        cudaGraph_t g = nullptr;
        e = cudaStreamEndCapture(cudaStreamPerThread, &g);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "decode_capture_end err=%s\n", cudaGetErrorString(e));
            abort_stream_capture();
            return;
        }
        if (!instantiate_graph(g, &graph_exec_, "decode_capture", 0)) return;
        graph_ = g;
        cudaGraphUpload(graph_exec_, cudaStreamPerThread);
        std::fprintf(stderr, "decode_cuda_graph=1\n");
    }

    void maybe_capture_prefill(int n) {
        if (n < 2 || n > kPfGraphMax || pf_graph_execs_[n]) return;
        abort_stream_capture();
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "pf_capture_sync n=%d err=%s\n", n, cudaGetErrorString(e));
            cudaGetLastError();
        }
        abort_stream_capture();
        e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "pf_capture_begin n=%d err=%s\n", n, cudaGetErrorString(e));
            return;
        }
        launch_prefill_chunk(0, n);
        launch_prefill_tail(n, n);
        cudaGraph_t g = nullptr;
        e = cudaStreamEndCapture(cudaStreamPerThread, &g);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "pf_capture_end n=%d err=%s\n", n, cudaGetErrorString(e));
            abort_stream_capture();
            return;
        }
        if (!instantiate_graph(g, &pf_graph_execs_[n], "pf_capture", n)) return;
        pf_graphs_[n] = g;
        cudaGraphUpload(pf_graph_execs_[n], cudaStreamPerThread);
        std::fprintf(stderr, "prefill_cuda_graph n=%d\n", n);
    }

    void maybe_capture_spec() {
        if (spec_graph_exec_ || pf_cap_ < 4) return;
        abort_stream_capture();
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) cudaGetLastError();
        abort_stream_capture();
        e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "spec_capture_begin err=%s\n", cudaGetErrorString(e));
            return;
        }
        launch_spec_chunk(4);
        cudaGraph_t g = nullptr;
        e = cudaStreamEndCapture(cudaStreamPerThread, &g);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "spec_capture_end err=%s\n", cudaGetErrorString(e));
            abort_stream_capture();
            return;
        }
        if (!instantiate_graph(g, &spec_graph_exec_, "spec_capture", 4)) return;
        spec_graph_ = g;
        cudaGraphUpload(spec_graph_exec_, cudaStreamPerThread);
        std::fprintf(stderr, "spec_cuda_graph n=4\n");
    }

    int32_t greedy_host() const {
        int best = 0;
        float mv = h_logits_[0];
        for (int i = 1; i < vocab_; ++i) {
            if (h_logits_[i] > mv) {
                mv = h_logits_[i];
                best = i;
            }
        }
        return best;
    }

    void build() {
        const ModelDesc& m = store_->model();
        hidden_ = m.hidden;
        vocab_ = m.vocab;
        const TensorDesc& emb = must("embed");
        embed_q_ = emb.quant;
        if (emb.quant == QuantKind::Q6_K || emb.quant == QuantKind::Q8_0) {
            const int er = emb.shape[0] > 0 ? static_cast<int>(emb.shape[0]) : vocab_;
            const int ec = emb.shape[1] > 0 ? static_cast<int>(emb.shape[1]) : hidden_;
            std::vector<float> ef(static_cast<size_t>(er) * static_cast<size_t>(ec));
            if (emb.quant == QuantKind::Q6_K) {
                constexpr int bsz = 210;
                const int nblk = ec / 256;
                const size_t rowb = static_cast<size_t>(nblk) * bsz;
                for (int r = 0; r < er; ++r)
                    ops::dequant_q6_k(emb.data.data() + static_cast<size_t>(r) * rowb,
                                      ef.data() + static_cast<size_t>(r) * ec, ec);
            } else {
                for (int r = 0; r < er; ++r)
                    ops::dequant_q8_0(emb.data.data() + static_cast<size_t>(r) * (ec / 32) * 34,
                                      ef.data() + static_cast<size_t>(r) * ec, ec);
            }
            d_embed_ = alloc(ef.size() * sizeof(float));
            CUDA_CHECK(cudaMemcpy(const_cast<void*>(d_embed_), ef.data(), ef.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            embed_q_ = QuantKind::F32;
        } else {
            d_embed_ = alloc(emb.data.size());
            CUDA_CHECK(cudaMemcpy(const_cast<void*>(d_embed_), emb.data.data(), emb.data.size(), cudaMemcpyHostToDevice));
        }
        const TensorDesc& lh = must("lm_head");
        int lh_rows = lh.shape[0] > 0 ? static_cast<int>(lh.shape[0]) : vocab_;
        int lh_cols = lh.shape[1] > 0 ? static_cast<int>(lh.shape[1]) : hidden_;
        const char* keep_bf16 = std::getenv("RAPIDLLM_BF16_LMHEAD");
        if (lh.quant == QuantKind::BF16 && lh_rows >= 4096 && (lh_cols % 128) == 0 &&
            !(keep_bf16 && keep_bf16[0] == '1'))
            lm_head_ = upload_bf16_as_fp8(lh, lh_rows, lh_cols);
        else
            lm_head_ = upload_w(lh, lh_rows, lh_cols);
        d_final_norm_ = upload_f32(must("final_norm"));

        int n_delta = 0, n_attn = 0;
        int max_qkv = 0, max_z = 0, max_nv = 0, max_dk = 0, max_dv = 0, max_inter = 0, max_qn = 0, max_kn = 0;
        for (int i = 0; i < m.n_layers; ++i) {
            const LayerDesc& Ld = m.layers[static_cast<size_t>(i)];
            GpuLayer L;
            L.kind = Ld.kind;
            L.eps = Ld.rms_eps;
            L.theta = m.rope.theta;
            const std::string p = "layers[" + std::to_string(i) + "].";
            L.attn_norm = upload_f32(must(p + "attn_norm"));
            L.ffn_norm = upload_f32(must(p + "ffn_norm"));
            const TensorDesc& wd = must(p + "mlp.down");
            L.inter = wd.shape[1] > 0 ? static_cast<int>(wd.shape[1]) : m.intermediate;
            L.wg = upload_w(must(p + "mlp.gate"), L.inter, hidden_);
            L.wu = upload_w(must(p + "mlp.up"), L.inter, hidden_);
            L.wd = upload_w(wd, hidden_, L.inter);
            max_inter = std::max(max_inter, L.inter);
            if (Ld.kind == LayerKind::GatedDeltaNet) {
                L.slot = n_delta++;
                L.nk = Ld.delta.n_k_heads;
                L.nv = Ld.delta.n_v_heads;
                L.dk = Ld.delta.k_dim;
                L.dv = Ld.delta.v_dim;
                L.conv_k = Ld.delta.conv_k;
                const int qkv_dim = L.nk * L.dk * 2 + L.nv * L.dv;
                const int zdim = L.nv * L.dv;
                L.wqkv = upload_w(must(p + "delta.gemm.in_proj_qkv"), qkv_dim, hidden_);
                L.wz = upload_w(must(p + "delta.gemm.in_proj_z"), zdim, hidden_);
                L.wa = upload_w(must(p + "delta.leftover.in_proj_a"), L.nv, hidden_);
                L.wb = upload_w(must(p + "delta.leftover.in_proj_b"), L.nv, hidden_);
                L.wo = upload_w(must(p + "delta.gemm.out_proj"), hidden_, zdim);
                L.A_log = upload_f32(must(p + "delta.leftover.a_log"));
                L.dt_bias = upload_f32(must(p + "delta.leftover.dt_bias"));
                L.conv_w = upload_f32(must(p + "delta.leftover.conv1d"));
                const TensorDesc& gn = must(p + "delta.leftover.norm");
                L.gnorm = upload_f32(gn);
                L.gnorm_n = gn.shape[0] > 0 ? static_cast<int>(gn.shape[0]) : 0;
                max_qkv = std::max(max_qkv, qkv_dim);
                max_z = std::max(max_z, zdim);
                max_nv = std::max(max_nv, L.nv);
                max_dk = std::max(max_dk, L.dk);
                max_dv = std::max(max_dv, L.dv);
            } else {
                L.slot = n_attn++;
                L.nq = Ld.attn.n_q;
                L.nkv = Ld.attn.n_kv;
                L.hd = Ld.attn.head_dim;
                L.rotary = std::max(2, static_cast<int>(L.hd * m.rope.partial_factor));
                const int qg = L.nq * L.hd * 2;
                const int kn = L.nkv * L.hd;
                L.wq = upload_w(must(p + "attn.wq"), qg, hidden_);
                L.wk = upload_w(must(p + "attn.wk"), kn, hidden_);
                L.wv = upload_w(must(p + "attn.wv"), kn, hidden_);
                L.wo_a = upload_w(must(p + "attn.wo"), hidden_, L.nq * L.hd);
                L.q_norm = upload_f32(must(p + "attn.q_norm"));
                L.k_norm = upload_f32(must(p + "attn.k_norm"));
                max_qn = std::max(max_qn, L.nq * L.hd);
                max_kn = std::max(max_kn, kn);
                max_z = std::max(max_z, L.nq * L.hd);
            }
            layers_.push_back(L);
        }
        n_delta_ = n_delta;
        n_attn_ = n_attn;
        max_qkv_ = max_qkv;
        max_z_ = max_z;
        max_nv_ = max_nv;
        max_inter_ = max_inter;
        max_qn_ = max_qn;
        max_kn_ = max_kn;
        // Short ctx keeps T<=32 so existing n=2..4 prefill graphs stay valid.
        // Long ctx uses T<=256 so 128k/200k prefill is not 4k serial chunks.
        pf_cap_ = std::max(1, std::min(ctx_, ctx_ > 4096 ? 256 : 32));
        max_batch_ = 1;
        if (const char* e = std::getenv("RAPIDLLM_MAX_BATCH")) {
            const int v = std::atoi(e);
            if (v > 1) max_batch_ = std::min(v, pf_cap_);
        }

        d_h_ = static_cast<float*>(alloc(sizeof(float) * hidden_));
        d_ss_ = static_cast<float*>(alloc(sizeof(float)));
        CUDA_CHECK(cudaMemset(d_ss_, 0, sizeof(float)));
        d_xn_ = static_cast<float*>(alloc(sizeof(float) * hidden_));
        d_y_ = static_cast<float*>(alloc(sizeof(float) * hidden_));
        d_qkv_ = static_cast<float*>(alloc(sizeof(float) * std::max(max_qkv, 1)));
        d_mix_ = static_cast<float*>(alloc(sizeof(float) * std::max(max_qkv, 1)));
        d_z_ = static_cast<float*>(alloc(sizeof(float) * std::max(max_z, 1)));
        d_aa_ = static_cast<float*>(alloc(sizeof(float) * std::max(max_nv, 1)));
        d_bb_ = static_cast<float*>(alloc(sizeof(float) * std::max(max_nv, 1)));
        d_qh_ = static_cast<float*>(alloc(sizeof(float) * std::max(max_nv * max_dk, 1)));
        d_kh_ = static_cast<float*>(alloc(sizeof(float) * std::max(max_nv * max_dk, 1)));
        d_vh_ = static_cast<float*>(alloc(sizeof(float) * std::max(max_nv * max_dv, 1)));
        d_beta_ = static_cast<float*>(alloc(sizeof(float) * std::max(max_nv, 1)));
        d_glog_ = static_cast<float*>(alloc(sizeof(float) * std::max(max_nv, 1)));
        d_o_ = static_cast<float*>(alloc(sizeof(float) * std::max(std::max(max_z, max_qn), 1)));
        d_og_ = static_cast<float*>(alloc(sizeof(float) * std::max(max_z, 1)));
        d_qg_ = static_cast<float*>(alloc(sizeof(float) * std::max(max_qn * 2, 1)));
        d_q_ = static_cast<float*>(alloc(sizeof(float) * std::max(max_qn, 1)));
        d_gate_ = static_cast<float*>(alloc(sizeof(float) * std::max(max_qn, 1)));
        d_k_ = static_cast<float*>(alloc(sizeof(float) * std::max(max_kn, 1)));
        d_vtmp_ = static_cast<float*>(alloc(sizeof(float) * std::max(max_kn, 1)));
        d_gate_mlp_ = static_cast<float*>(alloc(sizeof(float) * std::max(max_inter, 1)));
        d_up_ = static_cast<float*>(alloc(sizeof(float) * std::max(max_inter, 1)));
        logit_rows_ = std::max(max_batch_, 8);
        d_logits_ = static_cast<float*>(alloc(sizeof(float) * static_cast<size_t>(vocab_) * logit_rows_));
        d_tok_ = static_cast<int*>(alloc(4));
        d_pos_ = static_cast<int*>(alloc(4));
        d_pos_b_ = static_cast<int*>(alloc(sizeof(int) * std::max(max_batch_, logit_rows_)));
        d_best_ = static_cast<int*>(alloc(4));
        d_best_n_ = static_cast<int*>(alloc(sizeof(int) * logit_rows_));
        d_gen_out_ = static_cast<int*>(alloc(sizeof(int) * kGenCap));
        d_gen_n_ = static_cast<int*>(alloc(4));
        CUDA_CHECK(cudaMemset(d_gen_n_, 0, 4));
        const int nblk = (vocab_ + 255) / 256;
        d_amax_ = static_cast<float*>(alloc(sizeof(float) * nblk * logit_rows_));
        d_aidx_ = static_cast<int*>(alloc(sizeof(int) * nblk * logit_rows_));

        s_bytes_ = sizeof(uint16_t) * static_cast<size_t>(max_batch_) * std::max(n_delta, 1) * std::max(max_nv, 1) *
                   std::max(max_dk, 1) * std::max(max_dv, 1);
        conv_bytes_ = sizeof(float) * static_cast<size_t>(max_batch_) * std::max(n_delta, 1) * std::max(max_qkv, 1) * 4;
        const int kn_max = std::max(max_kn, 1);
        kv_bytes_ = sizeof(float) * static_cast<size_t>(max_batch_) * std::max(n_attn, 1) * ctx_ * kn_max;
        d_S_ = static_cast<uint16_t*>(alloc(s_bytes_));
        d_conv_ = static_cast<float*>(alloc(conv_bytes_));
        d_S_bak_ = static_cast<uint16_t*>(alloc(s_bytes_));
        d_conv_bak_ = static_cast<float*>(alloc(conv_bytes_));
        d_kcache_ = static_cast<float*>(alloc(kv_bytes_));
        d_vcache_ = static_cast<float*>(alloc(kv_bytes_));

        d_h_seq_ = static_cast<float*>(alloc(sizeof(float) * pf_cap_ * hidden_));
        d_xn_seq_ = static_cast<float*>(alloc(sizeof(float) * pf_cap_ * hidden_));
        d_y_seq_ = static_cast<float*>(alloc(sizeof(float) * pf_cap_ * hidden_));
        d_qkv_seq_ = static_cast<float*>(alloc(sizeof(float) * pf_cap_ * std::max(max_qkv, 1)));
        d_mix_seq_ = static_cast<float*>(alloc(sizeof(float) * pf_cap_ * std::max(max_qkv, 1)));
        d_z_seq_ = static_cast<float*>(alloc(sizeof(float) * pf_cap_ * std::max(max_z, 1)));
        d_aa_seq_ = static_cast<float*>(alloc(sizeof(float) * pf_cap_ * std::max(max_nv, 1)));
        d_bb_seq_ = static_cast<float*>(alloc(sizeof(float) * pf_cap_ * std::max(max_nv, 1)));
        d_og_seq_ = static_cast<float*>(alloc(sizeof(float) * pf_cap_ * std::max(max_z, 1)));
        d_qg_seq_ = static_cast<float*>(alloc(sizeof(float) * pf_cap_ * std::max(max_qn * 2, 1)));
        d_q_seq_ = static_cast<float*>(alloc(sizeof(float) * pf_cap_ * std::max(max_qn, 1)));
        d_gate_seq_ = static_cast<float*>(alloc(sizeof(float) * pf_cap_ * std::max(max_qn, 1)));
        d_k_seq_ = static_cast<float*>(alloc(sizeof(float) * pf_cap_ * std::max(max_kn, 1)));
        d_v_seq_ = static_cast<float*>(alloc(sizeof(float) * pf_cap_ * std::max(max_kn, 1)));
        d_o_seq_ = static_cast<float*>(alloc(sizeof(float) * pf_cap_ * std::max(std::max(max_z, max_qn), 1)));
        d_gate_mlp_seq_ = static_cast<float*>(alloc(sizeof(float) * pf_cap_ * std::max(max_inter, 1)));
        d_up_seq_ = static_cast<float*>(alloc(sizeof(float) * pf_cap_ * std::max(max_inter, 1)));
        d_toks_ = static_cast<int*>(alloc(sizeof(int) * pf_cap_));
        const int xf16_n = pf_cap_ * std::max({hidden_, std::max(max_inter, 1), std::max(max_qkv, 1),
                                               std::max(max_z, 1), std::max(max_qn * 2, 1), std::max(max_kn, 1)});
        d_xf16_ = static_cast<__half*>(alloc(sizeof(__half) * xf16_n));
        g_xf16 = d_xf16_;
        g_xf16_n = xf16_n;

        h_logits_.assign(static_cast<size_t>(vocab_), 0.f);
        h_pin_ = nullptr;
        if (cudaMallocHost(reinterpret_cast<void**>(&h_pin_), sizeof(float) * static_cast<size_t>(vocab_)) !=
            cudaSuccess)
            h_pin_ = nullptr;
        h_best_pin_ = nullptr;
        if (cudaMallocHost(reinterpret_cast<void**>(&h_best_pin_), sizeof(int)) != cudaSuccess) h_best_pin_ = nullptr;
        h_tok_pin_ = nullptr;
        if (cudaMallocHost(reinterpret_cast<void**>(&h_tok_pin_), sizeof(int)) != cudaSuccess) h_tok_pin_ = nullptr;
        CUDA_CHECK(cudaMemset(d_S_, 0, s_bytes_));
        CUDA_CHECK(cudaMemset(d_conv_, 0, conv_bytes_));
        CUDA_CHECK(cudaMemset(d_kcache_, 0, kv_bytes_));
        CUDA_CHECK(cudaMemset(d_vcache_, 0, kv_bytes_));
        persist_gdn_s();
        {
            int zero = 0;
            CUDA_CHECK(cudaMemcpy(d_pos_, &zero, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_tok_, &zero, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
            maybe_capture();
            for (int t = 2; t <= 4; ++t) maybe_capture_prefill(t);
            maybe_capture_spec();
        }
        for (auto& [_, t] : store_->table().tensors) {
            t.data.clear();
            t.data.shrink_to_fit();
            t.scale.clear();
            t.scale.shrink_to_fit();
        }
    }

    void release() {
        if (h_pin_) {
            cudaFreeHost(h_pin_);
            h_pin_ = nullptr;
        }
        if (h_best_pin_) {
            cudaFreeHost(h_best_pin_);
            h_best_pin_ = nullptr;
        }
        if (h_tok_pin_) {
            cudaFreeHost(h_tok_pin_);
            h_tok_pin_ = nullptr;
        }
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
        if (graph_) cudaGraphDestroy(graph_);
        if (spec_graph_exec_) cudaGraphExecDestroy(spec_graph_exec_);
        if (spec_graph_) cudaGraphDestroy(spec_graph_);
        if (bak_stream_) {
            cudaStreamDestroy(bak_stream_);
            bak_stream_ = nullptr;
        }
        for (int i = 0; i <= kPfGraphMax; ++i) {
            if (pf_graph_execs_[i]) cudaGraphExecDestroy(pf_graph_execs_[i]);
            if (pf_graphs_[i]) cudaGraphDestroy(pf_graphs_[i]);
        }
        for (void* p : allocs_) cudaFree(p);
        allocs_.clear();
    }

    WeightStore* store_ = nullptr;
    int ctx_ = 0, hidden_ = 0, vocab_ = 0, pos_ = 0, last_tok_ = 0;
    int n_delta_ = 0, n_attn_ = 0;
    int pf_cap_ = 1;
    int max_batch_ = 1;
    int logit_rows_ = 1;
    int max_qkv_ = 0, max_z_ = 0, max_nv_ = 0, max_inter_ = 0, max_qn_ = 0, max_kn_ = 0;
    QuantKind embed_q_ = QuantKind::F16;
    const void* d_embed_ = nullptr;
    GpuW lm_head_{};
    const float* d_final_norm_ = nullptr;
    std::vector<GpuLayer> layers_;
    std::vector<void*> allocs_;
    float *d_h_ = nullptr, *d_xn_ = nullptr, *d_y_ = nullptr, *d_ss_ = nullptr;
    float *d_qkv_ = nullptr, *d_mix_ = nullptr, *d_z_ = nullptr, *d_aa_ = nullptr, *d_bb_ = nullptr;
    float *d_qh_ = nullptr, *d_kh_ = nullptr, *d_vh_ = nullptr, *d_beta_ = nullptr, *d_glog_ = nullptr;
    float *d_o_ = nullptr, *d_og_ = nullptr;
    float *d_qg_ = nullptr, *d_q_ = nullptr, *d_gate_ = nullptr, *d_k_ = nullptr, *d_vtmp_ = nullptr;
    float *d_gate_mlp_ = nullptr, *d_up_ = nullptr, *d_logits_ = nullptr;
    uint16_t *d_S_ = nullptr, *d_S_bak_ = nullptr;
    float *d_conv_ = nullptr, *d_kcache_ = nullptr, *d_vcache_ = nullptr;
    float *d_conv_bak_ = nullptr;
    float *d_amax_ = nullptr;
    int *d_tok_ = nullptr, *d_pos_ = nullptr, *d_pos_b_ = nullptr, *d_best_ = nullptr, *d_best_n_ = nullptr,
        *d_aidx_ = nullptr, *d_gen_out_ = nullptr, *d_gen_n_ = nullptr;
    float *d_h_seq_ = nullptr, *d_xn_seq_ = nullptr, *d_y_seq_ = nullptr;
    float *d_qkv_seq_ = nullptr, *d_mix_seq_ = nullptr, *d_z_seq_ = nullptr;
    float *d_aa_seq_ = nullptr, *d_bb_seq_ = nullptr, *d_og_seq_ = nullptr;
    float *d_qg_seq_ = nullptr, *d_q_seq_ = nullptr, *d_gate_seq_ = nullptr;
    float *d_k_seq_ = nullptr, *d_v_seq_ = nullptr, *d_o_seq_ = nullptr;
    float *d_gate_mlp_seq_ = nullptr, *d_up_seq_ = nullptr;
    int* d_toks_ = nullptr;
    __half* d_xf16_ = nullptr;
    size_t s_bytes_ = 0, conv_bytes_ = 0, kv_bytes_ = 0;
    std::vector<float> h_logits_;
    mutable float* h_pin_ = nullptr;
    int* h_best_pin_ = nullptr;
    int* h_tok_pin_ = nullptr;
    static constexpr int kPfGraphMax = 8;
    static constexpr int kGenCap = 64;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    cudaGraph_t spec_graph_ = nullptr;
    cudaGraphExec_t spec_graph_exec_ = nullptr;
    cudaStream_t bak_stream_ = nullptr;
    cudaGraph_t pf_graphs_[kPfGraphMax + 1] = {};
    cudaGraphExec_t pf_graph_execs_[kPfGraphMax + 1] = {};
};

} // namespace

bool available() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return false;
    return n > 0;
}

std::unique_ptr<Engine> Engine::create(WeightStore& store, int ctx) {
    if (!available()) return nullptr;
    return std::make_unique<EngineImpl>(store, ctx);
}

} // namespace rapidllm::cuda_gen
