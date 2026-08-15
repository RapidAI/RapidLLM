#include "rapidllm/runtime/cuda_engine.h"

#include "rapidllm/ir/model_desc.h"
#include "rapidllm/kernels/ops.h"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <library_types.h>
#ifndef CUDA_R_8F_E4M3
#define CUDA_R_8F_E4M3 ((cudaDataType_t)28)
#endif

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

// Butterfly reduction — every lane holds the sum (attn softmax needs this).
__device__ __forceinline__ float warp_sum_all(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

__device__ __forceinline__ float warp_max_all(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
    return v;
}

// Asymmetric KV (BoFan llama.cpp-MTP-TurboQuant recipe):
// K = q8 (score precision), V = FWHT-128 + 3-bit Lloyd-Max (memory).
constexpr int kTqBlk = 128;
constexpr int kTq3B = 48;
constexpr float kTqInvSqrt = 0.08838834764831845f; // 1/sqrt(128)

__device__ __forceinline__ float tq3_cent(int i) {
    // 3-bit Lloyd-Max for N(0,1); scale is ||x||/sqrt(d) after orthonormal WHT.
    const float c[8] = {-2.152f, -1.344f, -0.756f, -0.245f, 0.245f, 0.756f, 1.344f, 2.152f};
    return c[i & 7];
}

// All threads in the block must call this (tid>=128 only hit the barriers).
__device__ void fwht128_sync(float* x, int tid) {
    for (int stride = 1; stride < kTqBlk; stride <<= 1) {
        float a = 0.f, b = 0.f;
        int j = 0;
        if (tid < kTqBlk) {
            j = tid ^ stride;
            a = x[tid];
            b = x[j];
        }
        __syncthreads();
        if (tid < kTqBlk && tid < j) {
            x[tid] = a + b;
            x[j] = a - b;
        }
        __syncthreads();
    }
    if (tid < kTqBlk) x[tid] *= kTqInvSqrt;
}

__device__ __forceinline__ int tq3_nn(float v) {
    int bi = 0;
    float bd = fabsf(v - tq3_cent(0));
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        const float d = fabsf(v - tq3_cent(i));
        if (d < bd) {
            bd = d;
            bi = i;
        }
    }
    return bi;
}

__device__ void pack_tq3_16(uint8_t* dst, const int* idx, int tid) {
    if (tid >= 16) return;
    const int i = tid * 8;
    uint32_t w = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) w |= static_cast<uint32_t>(idx[i + j] & 7) << (3 * j);
    dst[tid * 3 + 0] = static_cast<uint8_t>(w);
    dst[tid * 3 + 1] = static_cast<uint8_t>(w >> 8);
    dst[tid * 3 + 2] = static_cast<uint8_t>(w >> 16);
}

__device__ void unpack_tq3_tid(const uint8_t* src, int tid, float scale, float* out) {
    const int g = tid >> 3;
    const int j = tid & 7;
    const uint32_t w = static_cast<uint32_t>(src[g * 3]) | (static_cast<uint32_t>(src[g * 3 + 1]) << 8) |
                       (static_cast<uint32_t>(src[g * 3 + 2]) << 16);
    out[tid] = tq3_cent(static_cast<int>((w >> (3 * j)) & 7u)) * scale;
}

// Two independent FWHT-128 in one 256-thread block (tid 0..127 / 128..255).
// In-warp stages use shuffles; only stride 32/64 touch smem. ~4 barriers vs 28.
__device__ void fwht128x2_shuf(float* x0, float* x1, int tid) {
    const int lane = tid & 31;
    const int lid = tid & 127;
    float* x = (tid < 128) ? x0 : x1;
    float v = x[lid];
#pragma unroll
    for (int s = 1; s < 32; s <<= 1) {
        const float o = __shfl_xor_sync(0xffffffff, v, s);
        v = ((lane & s) == 0) ? (v + o) : (o - v);
    }
    x[lid] = v;
    __syncthreads();
#pragma unroll
    for (int s = 32; s <= 64; s <<= 1) {
        const int j = lid ^ s;
        const float a = x[lid];
        const float b = x[j];
        __syncthreads();
        if (lid < j) {
            x[lid] = a + b;
            x[j] = a - b;
        }
        __syncthreads();
    }
    x[lid] *= kTqInvSqrt;
}

__device__ void dequant_v_tq3_256(float* dst, const uint8_t* v_qs, const __half* v_sc, size_t tok_h, int tid) {
    const int half = tid >> 7;
    const int lid = tid & 127;
    const float vscl = __half2float(v_sc[tok_h * 2 + half]);
    unpack_tq3_tid(v_qs + (tok_h * 2 + half) * kTq3B, lid, vscl, dst + half * 128);
    fwht128x2_shuf(dst, dst + 128, tid);
}

// Warp-striped flash decode for hd=256 F16 KV. No scores[T] smem, so --ctx>8k
// (and the 262k F16 window) no longer walks T with a block barrier per token.
__device__ void flash_attn_decode_f16_hd256(const float* qh, const __half* k_cache, const __half* v_cache,
                                            float* oh, int Tend, int kn, int hkv, float scale) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int d0 = lane * 8;
    float qv[8], acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        qv[i] = qh[d0 + i];
        acc[i] = 0.f;
    }
    float mi = -1e30f, li = 0.f;
    for (int t = warp; t < Tend; t += 8) {
        const __half* kh = k_cache + static_cast<size_t>(t) * kn + hkv * 256 + d0;
        const uint4 kb = __ldg(reinterpret_cast<const uint4*>(kh));
        const __half* kp = reinterpret_cast<const __half*>(&kb);
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(kp));
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(kp + 2));
        const float2 c = __half22float2(*reinterpret_cast<const __half2*>(kp + 4));
        const float2 d = __half22float2(*reinterpret_cast<const __half2*>(kp + 6));
        float dot = qv[0] * a.x + qv[1] * a.y + qv[2] * b.x + qv[3] * b.y + qv[4] * c.x + qv[5] * c.y +
                    qv[6] * d.x + qv[7] * d.y;
        dot = warp_sum_all(dot) * scale;
        const float m2 = fmaxf(mi, dot);
        const float alpha = __expf(mi - m2);
        const float w = __expf(dot - m2);
        li = li * alpha + w;
        const __half* vh = v_cache + static_cast<size_t>(t) * kn + hkv * 256 + d0;
        const uint4 vb = __ldg(reinterpret_cast<const uint4*>(vh));
        const __half* vp = reinterpret_cast<const __half*>(&vb);
        const float2 va = __half22float2(*reinterpret_cast<const __half2*>(vp));
        const float2 vb2 = __half22float2(*reinterpret_cast<const __half2*>(vp + 2));
        const float2 vc = __half22float2(*reinterpret_cast<const __half2*>(vp + 4));
        const float2 vd = __half22float2(*reinterpret_cast<const __half2*>(vp + 6));
        acc[0] = acc[0] * alpha + w * va.x;
        acc[1] = acc[1] * alpha + w * va.y;
        acc[2] = acc[2] * alpha + w * vb2.x;
        acc[3] = acc[3] * alpha + w * vb2.y;
        acc[4] = acc[4] * alpha + w * vc.x;
        acc[5] = acc[5] * alpha + w * vc.y;
        acc[6] = acc[6] * alpha + w * vd.x;
        acc[7] = acc[7] * alpha + w * vd.y;
        mi = m2;
    }
    __shared__ float wmi[8], wli[8];
    __shared__ float wacc[8][32][8];
    if (lane == 0) {
        wmi[warp] = mi;
        wli[warp] = li;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) wacc[warp][lane][i] = acc[i];
    __syncthreads();
    if (warp == 0) {
        float m = -1e30f;
#pragma unroll
        for (int w = 0; w < 8; ++w) m = fmaxf(m, wmi[w]);
        float l = 0.f;
        float a[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) a[i] = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const float al = __expf(wmi[w] - m);
            l += wli[w] * al;
#pragma unroll
            for (int i = 0; i < 8; ++i) a[i] += wacc[w][lane][i] * al;
        }
        const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) oh[d0 + i] = a[i] * inv;
    }
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

__device__ __forceinline__ void load_x_tile(float* xs, const float* x, int tn) {
    const int n4 = tn >> 2;
    for (int i = static_cast<int>(threadIdx.x); i < n4; i += static_cast<int>(blockDim.x))
        reinterpret_cast<float4*>(xs)[i] = reinterpret_cast<const float4*>(x)[i];
    for (int i = (n4 << 2) + static_cast<int>(threadIdx.x); i < tn; i += static_cast<int>(blockDim.x))
        xs[i] = x[i];
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

// Two adjacent Q8 rows / warp — same SoA layout as gemv_q8_soa_k.
__global__ void gemv_q8_soa_2row_k(const int8_t* Q, const __half* scales, const float* x, float* y, int m,
                                   int n, int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int tile = n < kXsTile ? n : kXsTile;
    const int nb = n / 32;
    float acc0 = 0.f, acc1 = 0.f;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        const int bn = tn / 32;
        for (int i = threadIdx.x; i < tn; i += blockDim.x) xs[(i >> 5) * kXsPad + (i & 31)] = x[t0 + i];
        __syncthreads();
        if (row0 < m) {
            const int8_t* q0 = Q + static_cast<size_t>(row0) * n + t0;
            const __half* s0 = scales + static_cast<size_t>(row0) * nb + t0 / 32;
            const int8_t* q1 = row1 < m ? Q + static_cast<size_t>(row1) * n + t0 : nullptr;
            const __half* s1 = row1 < m ? scales + static_cast<size_t>(row1) * nb + t0 / 32 : nullptr;
            int b = lane;
            for (; b + 32 < bn; b += 64) {
                acc0 += __half2float(__ldcs(s0 + b)) * q8_dot32(q0 + b * 32, xs + xs_off(b));
                acc0 += __half2float(__ldcs(s0 + b + 32)) * q8_dot32(q0 + (b + 32) * 32, xs + xs_off(b + 32));
                if (q1) {
                    acc1 += __half2float(__ldcs(s1 + b)) * q8_dot32(q1 + b * 32, xs + xs_off(b));
                    acc1 += __half2float(__ldcs(s1 + b + 32)) * q8_dot32(q1 + (b + 32) * 32, xs + xs_off(b + 32));
                }
            }
            if (b < bn) {
                acc0 += __half2float(__ldcs(s0 + b)) * q8_dot32(q0 + b * 32, xs + xs_off(b));
                if (q1) acc1 += __half2float(__ldcs(s1 + b)) * q8_dot32(q1 + b * 32, xs + xs_off(b));
            }
        }
        __syncthreads();
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

// GGUF Q4_K / Q5_K / Q6_K superblock = 256 weights. Packed row-major, same as CPU dequant.
constexpr int kQ4KBsz = 144;
constexpr int kQ5KBsz = 176;
constexpr int kQ6KBsz = 210;
// GPU SoA: pre-expanded scales, 4/5/6-bit quants still packed (not FP8).
constexpr int kQ4KSoaBsz = 160; // half dscale[8] + half dmin[8] + qs[128]
constexpr int kQ5KSoaBsz = 192; // half dscale[8] + half dmin[8] + qh[32] + ql[128]
constexpr int kQ6KSoaBsz = 224; // half dscale[16] + ql[128] + qh[64]

__device__ __forceinline__ void q4k_scale_min_d(const uint8_t* sc, int j, int& s, int& mn) {
    if (j < 4) {
        s = sc[j] & 63;
        mn = sc[j + 4] & 63;
    } else {
        s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        mn = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
}

__device__ __forceinline__ float acc_q4k_row(const uint8_t* row, const float* xs, int nb, int lane) {
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ4KBsz;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const float minv = __half2float(*reinterpret_cast<const __half*>(blk + 2));
        const uint8_t* sc = blk + 4;
        const uint8_t* q = blk + 16;
        const float* xb = xs + b * 256;
        const uint32_t qpack = (static_cast<uint32_t>(q[lane]) << 0) |
                               (static_cast<uint32_t>(q[32 + lane]) << 8) |
                               (static_cast<uint32_t>(q[64 + lane]) << 16) |
                               (static_cast<uint32_t>(q[96 + lane]) << 24);
#pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            int s0, m0, s1, m1;
            q4k_scale_min_d(sc, grp * 2, s0, m0);
            q4k_scale_min_d(sc, grp * 2 + 1, s1, m1);
            const uint8_t qq = static_cast<uint8_t>(qpack >> (grp * 8));
            acc += (d * static_cast<float>(s0) * static_cast<float>(qq & 15) - minv * static_cast<float>(m0)) *
                   xb[lane];
            acc += (d * static_cast<float>(s1) * static_cast<float>(qq >> 4) - minv * static_cast<float>(m1)) *
                   xb[32 + lane];
            xb += 64;
        }
    }
    return acc;
}

__device__ __forceinline__ float acc_q5k_row(const uint8_t* row, const float* xs, int nb, int lane) {
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ5KBsz;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const float minv = __half2float(*reinterpret_cast<const __half*>(blk + 2));
        const uint8_t* sc = blk + 4;
        const uint8_t* qh = blk + 16;
        const uint8_t* ql = blk + 48;
        const float* xb = xs + b * 256;
        uint8_t u1 = 1, u2 = 2;
#pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            int s0, m0, s1, m1;
            q4k_scale_min_d(sc, grp * 2, s0, m0);
            q4k_scale_min_d(sc, grp * 2 + 1, s1, m1);
            const int qlo = ql[lane];
            const int qhi = qh[lane];
            const int v0 = (qlo & 15) + ((qhi & u1) ? 16 : 0);
            const int v1 = (qlo >> 4) + ((qhi & u2) ? 16 : 0);
            acc += (d * static_cast<float>(s0) * static_cast<float>(v0) - minv * static_cast<float>(m0)) * xb[lane];
            acc += (d * static_cast<float>(s1) * static_cast<float>(v1) - minv * static_cast<float>(m1)) *
                   xb[32 + lane];
            ql += 32;
            xb += 64;
            u1 = static_cast<uint8_t>(u1 << 2);
            u2 = static_cast<uint8_t>(u2 << 2);
        }
    }
    return acc;
}

__device__ __forceinline__ float acc_q6k_row(const uint8_t* row, const float* xs, int nb, int lane) {
    float acc = 0.f;
    const int is = lane / 16;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ6KBsz;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
        const float d = __half2float(*reinterpret_cast<const __half*>(blk + 208));
        const float* xb = xs + b * 256;
#pragma unroll
        for (int n128 = 0; n128 < 2; ++n128) {
            const int q1 = static_cast<int>((ql[lane] & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32;
            const int q2 = static_cast<int>((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32;
            const int q3 = static_cast<int>((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32;
            const int q4 = static_cast<int>((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4)) - 32;
            acc += d * static_cast<float>(sc[is]) * static_cast<float>(q1) * xb[lane];
            acc += d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2) * xb[32 + lane];
            acc += d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3) * xb[64 + lane];
            acc += d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4) * xb[96 + lane];
            ql += 64;
            qh += 32;
            sc += 8;
            xb += 128;
        }
    }
    return acc;
}

__device__ __forceinline__ float acc_q4k_soa_row(const uint8_t* row, const float* xs, int nb, int lane) {
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ4KSoaBsz;
        const __half* ds = reinterpret_cast<const __half*>(blk);
        const __half* dm = reinterpret_cast<const __half*>(blk + 16);
        const uint8_t* q = blk + 32;
        const float* xb = xs + b * 256;
        const uint32_t qpack = (static_cast<uint32_t>(__ldcs(q + lane))) |
                               (static_cast<uint32_t>(__ldcs(q + 32 + lane)) << 8) |
                               (static_cast<uint32_t>(__ldcs(q + 64 + lane)) << 16) |
                               (static_cast<uint32_t>(__ldcs(q + 96 + lane)) << 24);
#pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const uint8_t qq = static_cast<uint8_t>(qpack >> (grp * 8));
            acc = fmaf(__half2float(ds[grp * 2]) * static_cast<float>(qq & 15) - __half2float(dm[grp * 2]),
                       xb[lane], acc);
            acc = fmaf(__half2float(ds[grp * 2 + 1]) * static_cast<float>(qq >> 4) - __half2float(dm[grp * 2 + 1]),
                       xb[32 + lane], acc);
            xb += 64;
        }
    }
    return acc;
}

__device__ __forceinline__ float acc_q5k_soa_row(const uint8_t* row, const float* xs, int nb, int lane) {
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ5KSoaBsz;
        const __half* ds = reinterpret_cast<const __half*>(blk);
        const __half* dm = reinterpret_cast<const __half*>(blk + 16);
        const uint8_t* qh = blk + 32;
        const uint8_t* ql = blk + 64;
        const float* xb = xs + b * 256;
        uint8_t u1 = 1, u2 = 2;
#pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const int qlo = ql[lane];
            const int qhi = qh[lane];
            const int v0 = (qlo & 15) + ((qhi & u1) ? 16 : 0);
            const int v1 = (qlo >> 4) + ((qhi & u2) ? 16 : 0);
            acc = fmaf(__half2float(ds[grp * 2]) * static_cast<float>(v0) - __half2float(dm[grp * 2]), xb[lane],
                       acc);
            acc = fmaf(__half2float(ds[grp * 2 + 1]) * static_cast<float>(v1) - __half2float(dm[grp * 2 + 1]),
                       xb[32 + lane], acc);
            ql += 32;
            xb += 64;
            u1 = static_cast<uint8_t>(u1 << 2);
            u2 = static_cast<uint8_t>(u2 << 2);
        }
    }
    return acc;
}

__device__ __forceinline__ float acc_q6k_soa_row(const uint8_t* row, const float* xs, int nb, int lane) {
    float acc = 0.f;
    const int is = lane / 16;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ6KSoaBsz;
        const __half* ds = reinterpret_cast<const __half*>(blk);
        const uint8_t* ql = blk + 32;
        const uint8_t* qh = blk + 160;
        const float* xb = xs + b * 256;
#pragma unroll
        for (int n128 = 0; n128 < 2; ++n128) {
            const int q1 = static_cast<int>((ql[lane] & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32;
            const int q2 = static_cast<int>((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32;
            const int q3 = static_cast<int>((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32;
            const int q4 = static_cast<int>((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4)) - 32;
            acc = fmaf(__half2float(ds[is]) * static_cast<float>(q1), xb[lane], acc);
            acc = fmaf(__half2float(ds[is + 2]) * static_cast<float>(q2), xb[32 + lane], acc);
            acc = fmaf(__half2float(ds[is + 4]) * static_cast<float>(q3), xb[64 + lane], acc);
            acc = fmaf(__half2float(ds[is + 6]) * static_cast<float>(q4), xb[96 + lane], acc);
            ql += 64;
            qh += 32;
            ds += 8;
            xb += 128;
        }
    }
    return acc;
}

template <int Bsz>
__global__ void gemv_qk_k(const uint8_t* W, const float* x, float* y, int m, int n, int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int tile = n < kXsTile ? n : kXsTile;
    const int nb_all = n / 256;
    float acc = 0.f;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        const int nb = tn / 256;
        load_x_tile(xs, x + t0, tn);
        __syncthreads();
        if (row < m && nb > 0) {
            const uint8_t* roww = W + (static_cast<size_t>(row) * nb_all + static_cast<size_t>(t0 / 256)) * Bsz;
            if (Bsz == kQ4KSoaBsz) acc += acc_q4k_soa_row(roww, xs, nb, lane);
            else if (Bsz == kQ5KSoaBsz) acc += acc_q5k_soa_row(roww, xs, nb, lane);
            else if (Bsz == kQ6KSoaBsz) acc += acc_q6k_soa_row(roww, xs, nb, lane);
            else if (Bsz == kQ4KBsz) acc += acc_q4k_row(roww, xs, nb, lane);
            else if (Bsz == kQ5KBsz) acc += acc_q5k_row(roww, xs, nb, lane);
            else acc += acc_q6k_row(roww, xs, nb, lane);
        }
        __syncthreads();
    }
    if (row < m) {
        acc = warp_sum(acc);
        if (lane == 0) write_y(y, row, acc, add);
    }
}

template <int Bsz>
__global__ void gemv_qk_2row_k(const uint8_t* W, const float* x, float* y, int m, int n, int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int tile = n < kXsTile ? n : kXsTile;
    const int nb_all = n / 256;
    float acc0 = 0.f, acc1 = 0.f;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        const int nb = tn / 256;
        load_x_tile(xs, x + t0, tn);
        __syncthreads();
        if (row0 < m && nb > 0) {
            const uint8_t* r0 = W + (static_cast<size_t>(row0) * nb_all + static_cast<size_t>(t0 / 256)) * Bsz;
            if (Bsz == kQ4KSoaBsz) acc0 += acc_q4k_soa_row(r0, xs, nb, lane);
            else if (Bsz == kQ5KSoaBsz) acc0 += acc_q5k_soa_row(r0, xs, nb, lane);
            else if (Bsz == kQ6KSoaBsz) acc0 += acc_q6k_soa_row(r0, xs, nb, lane);
            else if (Bsz == kQ4KBsz) acc0 += acc_q4k_row(r0, xs, nb, lane);
            else if (Bsz == kQ5KBsz) acc0 += acc_q5k_row(r0, xs, nb, lane);
            else acc0 += acc_q6k_row(r0, xs, nb, lane);
            if (row1 < m) {
                const uint8_t* r1 = W + (static_cast<size_t>(row1) * nb_all + static_cast<size_t>(t0 / 256)) * Bsz;
                if (Bsz == kQ4KSoaBsz) acc1 += acc_q4k_soa_row(r1, xs, nb, lane);
                else if (Bsz == kQ5KSoaBsz) acc1 += acc_q5k_soa_row(r1, xs, nb, lane);
                else if (Bsz == kQ6KSoaBsz) acc1 += acc_q6k_soa_row(r1, xs, nb, lane);
                else if (Bsz == kQ4KBsz) acc1 += acc_q4k_row(r1, xs, nb, lane);
                else if (Bsz == kQ5KBsz) acc1 += acc_q5k_row(r1, xs, nb, lane);
                else acc1 += acc_q6k_row(r1, xs, nb, lane);
            }
        }
        __syncthreads();
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

// Prefill Q4/Q5/Q6: reuse each superblock across TT tokens. TT <= 16, tile=256.
template <int Bsz>
__global__ void gemm_qk_t_k(const uint8_t* W, const float* X, float* Y, int m, int n, int T, int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    constexpr int tile = 256;
    const int nb_all = n / 256;
    float acc[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) acc[t] = 0.f;
    for (int t0 = 0; t0 < n; t0 += tile) {
        for (int i = threadIdx.x; i < T * tile; i += blockDim.x) {
            const int t = i / tile;
            const int j = i - t * tile;
            xs[i] = X[static_cast<size_t>(t) * n + t0 + j];
        }
        __syncthreads();
        if (row < m) {
            const uint8_t* roww =
                W + (static_cast<size_t>(row) * nb_all + static_cast<size_t>(t0 / 256)) * Bsz;
            for (int t = 0; t < T; ++t) {
                if (Bsz == kQ4KSoaBsz) acc[t] += acc_q4k_soa_row(roww, xs + t * tile, 1, lane);
                else if (Bsz == kQ5KSoaBsz) acc[t] += acc_q5k_soa_row(roww, xs + t * tile, 1, lane);
                else if (Bsz == kQ6KSoaBsz) acc[t] += acc_q6k_soa_row(roww, xs + t * tile, 1, lane);
                else if (Bsz == kQ4KBsz) acc[t] += acc_q4k_row(roww, xs + t * tile, 1, lane);
                else if (Bsz == kQ5KBsz) acc[t] += acc_q5k_row(roww, xs + t * tile, 1, lane);
                else acc[t] += acc_q6k_row(roww, xs + t * tile, 1, lane);
            }
        }
        __syncthreads();
    }
    if (row < m) {
        for (int t = 0; t < T; ++t) {
            float a = warp_sum(acc[t]);
            if (lane == 0) write_y(Y + static_cast<size_t>(t) * m, row, a, add);
        }
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
    float acc4 = 0.f, acc5 = 0.f, acc6 = 0.f, acc7 = 0.f;
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
                if (T > 4) acc4 += fp8_dot16(p, sv, xs + 4 * tstride + o0);
                if (T > 5) acc5 += fp8_dot16(p, sv, xs + 5 * tstride + o0);
                if (T > 6) acc6 += fp8_dot16(p, sv, xs + 6 * tstride + o0);
                if (T > 7) acc7 += fp8_dot16(p, sv, xs + 7 * tstride + o0);
            }
            for (; b < nb; ++b) {
                const float s = scale[bi * nb_n + b0 + b];
                const uint32_t p = __ldcs(fp8_blk4(W, row, m, b0 + b, lane));
                const int o = b * 128 + lane * 4;
                acc0 += s * fp8x4_dot(p, xs + o);
                if (T > 1) acc1 += s * fp8x4_dot(p, xs + tstride + o);
                if (T > 2) acc2 += s * fp8x4_dot(p, xs + 2 * tstride + o);
                if (T > 3) acc3 += s * fp8x4_dot(p, xs + 3 * tstride + o);
                if (T > 4) acc4 += s * fp8x4_dot(p, xs + 4 * tstride + o);
                if (T > 5) acc5 += s * fp8x4_dot(p, xs + 5 * tstride + o);
                if (T > 6) acc6 += s * fp8x4_dot(p, xs + 6 * tstride + o);
                if (T > 7) acc7 += s * fp8x4_dot(p, xs + 7 * tstride + o);
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
        if (T > 4) {
            float a4 = warp_sum(acc4);
            if (lane == 0) write_y(Y + 4 * m, row, a4, add);
        }
        if (T > 5) {
            float a5 = warp_sum(acc5);
            if (lane == 0) write_y(Y + 5 * m, row, a5, add);
        }
        if (T > 6) {
            float a6 = warp_sum(acc6);
            if (lane == 0) write_y(Y + 6 * m, row, a6, add);
        }
        if (T > 7) {
            float a7 = warp_sum(acc7);
            if (lane == 0) write_y(Y + 7 * m, row, a7, add);
        }
    }
}

// Prefill GEMM: one weight pass over a T-tile (8/16/32). launch_linear used to split
// T into 4-token chunks, so a 256-token prefill reread all 28 GiB weights 64 times.
template <int NT>
__global__ void gemm_fp8_mt_k(const uint8_t* W, const float* scale, const float* X, float* Y, int m, int n,
                              int T, int tile, int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb_n = (n + 127) / 128;
    const int bi = row / 128;
    const int tstride = tile;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    const int tt = T < NT ? T : NT;
    for (int k0 = 0; k0 < n; k0 += tile) {
        const int tn = n - k0 < tile ? n - k0 : tile;
        for (int i = threadIdx.x; i < tt * tn; i += blockDim.x) {
            const int t = i / tn;
            const int j = i - t * tn;
            xs[t * tstride + j] = X[static_cast<size_t>(t) * n + k0 + j];
        }
        __syncthreads();
        if (row < m && scale) {
            const int b0 = k0 / 128;
            const int nb = tn / 128;
            int b = 0;
            for (; b + 3 < nb; b += 4) {
                const float4 sv = __ldg(reinterpret_cast<const float4*>(scale + bi * nb_n + b0 + b));
                const uint4 p = fp8_ld16(W, row, m, b0 + b, lane);
                const int o0 = b * 128 + lane * 4;
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    if (t < tt) acc[t] += fp8_dot16(p, sv, xs + t * tstride + o0);
                }
            }
            for (; b < nb; ++b) {
                const float s = scale[bi * nb_n + b0 + b];
                const uint32_t p = __ldcs(fp8_blk4(W, row, m, b0 + b, lane));
                const int o = b * 128 + lane * 4;
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    if (t < tt) acc[t] += s * fp8x4_dot(p, xs + t * tstride + o);
                }
            }
        }
        __syncthreads();
    }
    if (row < m) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t < tt) {
                const float a = warp_sum(acc[t]);
                if (lane == 0) write_y(Y + static_cast<size_t>(t) * m, row, a, add);
            }
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

// Walk T tokens in one launch (the T-loop of conv1d_upd_k was ~1k launches
// per GDN layer on the second 1024-tile of a 2048-fill).
__device__ __forceinline__ void conv1d_k4_one(const float* x, const float* w, float* state, float* y, int T,
                                             int dim, int c) {
    float* st = state + c * 4;
    const float* wc = w + c * 4;
    float s0 = st[0], s1 = st[1], s2 = st[2], s3 = st[3];
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    float xt = x[c];
    for (int t = 0; t < T; ++t) {
        const float nxt = (t + 1 < T) ? x[static_cast<size_t>(t + 1) * dim + c] : 0.f;
        y[static_cast<size_t>(t) * dim + c] = silu_d(w0 * s1 + w1 * s2 + w2 * s3 + w3 * xt);
        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = xt;
        xt = nxt;
    }
    st[0] = s0;
    st[1] = s1;
    st[2] = s2;
    st[3] = s3;
}

__global__ void conv1d_upd_seq_k(const float* x, const float* w, float* state, float* y, int T, int dim, int k) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (k == 4) {
        const int c = tid * 2;
        if (c >= dim) return;
        if (c + 1 >= dim) {
            conv1d_k4_one(x, w, state, y, T, dim, c);
            return;
        }
        float* sta = state + c * 4;
        float* stb = state + (c + 1) * 4;
        const float* wa = w + c * 4;
        const float* wb = w + (c + 1) * 4;
        float a0 = sta[0], a1 = sta[1], a2 = sta[2], a3 = sta[3];
        float b0 = stb[0], b1 = stb[1], b2 = stb[2], b3 = stb[3];
        const float u0 = wa[0], u1 = wa[1], u2 = wa[2], u3 = wa[3];
        const float v0 = wb[0], v1 = wb[1], v2 = wb[2], v3 = wb[3];
        float xa = x[c], xb = x[c + 1];
        for (int t = 0; t < T; ++t) {
            const size_t base = static_cast<size_t>(t) * dim;
            y[base + c] = silu_d(u0 * a1 + u1 * a2 + u2 * a3 + u3 * xa);
            y[base + c + 1] = silu_d(v0 * b1 + v1 * b2 + v2 * b3 + v3 * xb);
            a0 = a1;
            a1 = a2;
            a2 = a3;
            a3 = xa;
            b0 = b1;
            b1 = b2;
            b2 = b3;
            b3 = xb;
            if (t + 1 < T) {
                xa = x[base + dim + c];
                xb = x[base + dim + c + 1];
            }
        }
        sta[0] = a0;
        sta[1] = a1;
        sta[2] = a2;
        sta[3] = a3;
        stb[0] = b0;
        stb[1] = b1;
        stb[2] = b2;
        stb[3] = b3;
        return;
    }
    const int c = tid;
    if (c >= dim) return;
    for (int t = 0; t < T; ++t) {
        y[static_cast<size_t>(t) * dim + c] = conv1d_mix(x + static_cast<size_t>(t) * dim, w, state, c, k);
        conv1d_push(x + static_cast<size_t>(t) * dim, state, c, k);
    }
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
            const int nw = blockDim.x >= 32 ? (blockDim.x >> 5) : 1;
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

// Split each head's 128-col S across 8 blocks (384 warps). Each column is
// owned by a lane pair that splits the 128-row DK loop (xor-16 reduce).
// In-place RMS-norm of Q/K in mix (once per token/head). split_k used to redo
// this in each of 8 j-tiles — 8× Q/K traffic + 8× rsqrt on the 2048-fill.
__global__ void gdn_norm_qk_mix_k(float* mix, int T, int nk, int qkv_dim) {
    constexpr int DK = 128;
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    if (h >= nk || t >= T) return;
    const int lane = threadIdx.x;
    float* Q = mix + static_cast<size_t>(t) * qkv_dim + h * DK;
    float* K = Q + nk * DK;
    float qss = Q[lane] * Q[lane] + Q[lane + 32] * Q[lane + 32] + Q[lane + 64] * Q[lane + 64] +
                Q[lane + 96] * Q[lane + 96];
    float kss = K[lane] * K[lane] + K[lane + 32] * K[lane + 32] + K[lane + 64] * K[lane + 64] +
                K[lane + 96] * K[lane + 96];
    qss = warp_sum_all(qss);
    kss = warp_sum_all(kss);
    const float qinv = rsqrtf(qss + 1e-6f) * 0.08838834764f;
    const float kinv = rsqrtf(kss + 1e-6f);
    Q[lane] *= qinv;
    Q[lane + 32] *= qinv;
    Q[lane + 64] *= qinv;
    Q[lane + 96] *= qinv;
    K[lane] *= kinv;
    K[lane + 32] *= kinv;
    K[lane + 64] *= kinv;
    K[lane + 96] *= kinv;
}

// Pack g=exp(-exp(A)*softplus(a+dt)) and beta=sigmoid(b) once per (t,head).
// Overwrites aa/bb; split_k only reads them.
__global__ void gdn_pack_ab_k(float* aa, float* bb, const float* A_log, const float* dt_bias, int T, int nv) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = T * nv;
    if (i >= n) return;
    const int h = i - (i / nv) * nv;
    aa[i] = expf(-expf(A_log[h]) * softplus_d(aa[i] + dt_bias[h]));
    bb[i] = sigmoid_d(bb[i]);
}

// Two warps share one Q/K load (4 j-tiles instead of 8). Same math as the
// 1-warp split; only the j-column ownership changes.
__global__ void gdn_prefill_split2_k(const float* mix_seq, const float* aa_seq, const float* bb_seq, uint16_t* S,
                                     const float* A_log, const float* dt_bias, float* o_seq, int T, int nk,
                                     int nv, int qkv_dim, int h0) {
    (void)A_log;
    (void)dt_bias;
    constexpr int DK = 128, DV = 128, JT = 16, HALF = DK / 2;
    const int h = blockIdx.x + h0;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int j0 = blockIdx.y * 32 + warp * JT;
    if (h >= nv || j0 >= DV) return;
    const int j = lane & (JT - 1);
    const int half = lane >> 4;
    const int i0 = half * HALF;
    extern __shared__ float sm[];
    float* q0 = sm;
    float* k0 = sm + DK;
    uint16_t* Sh = S + static_cast<size_t>(h) * DK * DV;
    float scol[HALF];
#pragma unroll
    for (int i = 0; i < HALF; ++i) scol[i] = bf16_to_f32(Sh[(i0 + i) * DV + j0 + j]);
    const int qdim = nk * DK;
    const int src_h = h / (nv / nk);
    for (int t = 0; t < T; ++t) {
        const float* mix = mix_seq + static_cast<size_t>(t) * qkv_dim;
        const float* Q = mix + src_h * DK;
        const float* K = mix + qdim + src_h * DK;
        const float* V = mix + 2 * qdim;
        for (int i = threadIdx.x * 4; i < DK; i += 256) {
            *reinterpret_cast<float4*>(q0 + i) = *reinterpret_cast<const float4*>(Q + i);
            *reinterpret_cast<float4*>(k0 + i) = *reinterpret_cast<const float4*>(K + i);
        }
        __syncthreads();
        const float b = bb_seq[static_cast<size_t>(t) * nv + h];
        const float g = aa_seq[static_cast<size_t>(t) * nv + h];
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < HALF; i += 4) {
            scol[i + 0] *= g;
            scol[i + 1] *= g;
            scol[i + 2] *= g;
            scol[i + 3] *= g;
            kv += scol[i + 0] * k0[i0 + i] + scol[i + 1] * k0[i0 + i + 1] + scol[i + 2] * k0[i0 + i + 2] +
                  scol[i + 3] * k0[i0 + i + 3];
        }
        kv += __shfl_xor_sync(0xffffffff, kv, 16);
        const float del = (V[h * DV + j0 + j] - kv) * b;
        float oj = 0.f;
#pragma unroll
        for (int i = 0; i < HALF; i += 4) {
            scol[i + 0] += k0[i0 + i] * del;
            scol[i + 1] += k0[i0 + i + 1] * del;
            scol[i + 2] += k0[i0 + i + 2] * del;
            scol[i + 3] += k0[i0 + i + 3] * del;
            oj += scol[i + 0] * q0[i0 + i] + scol[i + 1] * q0[i0 + i + 1] + scol[i + 2] * q0[i0 + i + 2] +
                  scol[i + 3] * q0[i0 + i + 3];
        }
        oj += __shfl_xor_sync(0xffffffff, oj, 16);
        if (half == 0) o_seq[static_cast<size_t>(t) * nv * DV + h * DV + j0 + j] = oj;
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < HALF; ++i) Sh[(i0 + i) * DV + j0 + j] = f32_to_bf16(scol[i]);
}

// mix Q/K already L2-normed; aa/bb already packed to (g, beta).
__global__ void gdn_prefill_split_k(const float* mix_seq, const float* aa_seq, const float* bb_seq, uint16_t* S,
                                    const float* A_log, const float* dt_bias, float* o_seq, int T, int nk,
                                    int nv, int qkv_dim, int h0) {
    (void)A_log;
    (void)dt_bias;
    constexpr int DK = 128, DV = 128, NY = 8, JT = DV / NY, HALF = DK / 2;
    const int h = blockIdx.x + h0;
    const int j0 = blockIdx.y * JT;
    if (h >= nv || j0 >= DV) return;
    const int lane = threadIdx.x;
    const int j = lane & (JT - 1);
    const int half = lane >> 4;
    const int i0 = half * HALF;
    extern __shared__ float sm[];
    float* q0 = sm;
    float* k0 = sm + DK;
    uint16_t* Sh = S + static_cast<size_t>(h) * DK * DV;
    float scol[HALF];
#pragma unroll
    for (int i = 0; i < HALF; ++i) scol[i] = bf16_to_f32(Sh[(i0 + i) * DV + j0 + j]);
    const int qdim = nk * DK;
    const int src_h = h / (nv / nk);
    auto load_qk = [&](int t, float* qd, float* kd) {
        const float* mix = mix_seq + static_cast<size_t>(t) * qkv_dim;
        const float* Q = mix + src_h * DK;
        const float* K = mix + qdim + src_h * DK;
        *reinterpret_cast<float4*>(qd + lane * 4) = *reinterpret_cast<const float4*>(Q + lane * 4);
        *reinterpret_cast<float4*>(kd + lane * 4) = *reinterpret_cast<const float4*>(K + lane * 4);
    };
    for (int t = 0; t < T; ++t) {
        load_qk(t, q0, k0);
        __syncwarp();
        const float* mix = mix_seq + static_cast<size_t>(t) * qkv_dim;
        const float* V = mix + 2 * qdim;
        const float b = bb_seq[static_cast<size_t>(t) * nv + h];
        const float g = aa_seq[static_cast<size_t>(t) * nv + h];
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < HALF; i += 4) {
            scol[i + 0] *= g;
            scol[i + 1] *= g;
            scol[i + 2] *= g;
            scol[i + 3] *= g;
            kv += scol[i + 0] * k0[i0 + i] + scol[i + 1] * k0[i0 + i + 1] + scol[i + 2] * k0[i0 + i + 2] +
                  scol[i + 3] * k0[i0 + i + 3];
        }
        kv += __shfl_xor_sync(0xffffffff, kv, 16);
        const float del = (V[h * DV + j0 + j] - kv) * b;
        float oj = 0.f;
#pragma unroll
        for (int i = 0; i < HALF; i += 4) {
            scol[i + 0] += k0[i0 + i] * del;
            scol[i + 1] += k0[i0 + i + 1] * del;
            scol[i + 2] += k0[i0 + i + 2] * del;
            scol[i + 3] += k0[i0 + i + 3] * del;
            oj += scol[i + 0] * q0[i0 + i] + scol[i + 1] * q0[i0 + i + 1] + scol[i + 2] * q0[i0 + i + 2] +
                  scol[i + 3] * q0[i0 + i + 3];
        }
        oj += __shfl_xor_sync(0xffffffff, oj, 16);
        if (half == 0) o_seq[static_cast<size_t>(t) * nv * DV + h * DV + j0 + j] = oj;
    }
#pragma unroll
    for (int i = 0; i < HALF; ++i) Sh[(i0 + i) * DV + j0 + j] = f32_to_bf16(scol[i]);
}

// One warp per (head, query). Each lane holds 8 of 256 dims. No block smem.
__global__ void attn_prefill_warp_k(const float* q, const float* k_cache, const float* v_cache, float* o,
                                    int pos0, int T, int n_q, int n_kv, int head_dim) {
    const int hq = blockIdx.x;
    const int t = blockIdx.y;
    if (hq >= n_q || t >= T) return;
    const int pos = pos0 + t;
    const int Tend = pos + 1;
    const int hkv = hq / (n_q / n_kv);
    const int lane = threadIdx.x;
    const int d0 = lane * 8;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float* qh = q + (static_cast<size_t>(t) * n_q + hq) * head_dim + d0;
    float q0 = qh[0], q1 = qh[1], q2 = qh[2], q3 = qh[3], q4 = qh[4], q5 = qh[5], q6 = qh[6], q7 = qh[7];
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    float mi = -1e30f, li = 0.f;
    const int kn = n_kv * head_dim;
    for (int s = 0; s < Tend; ++s) {
        const float* kh = k_cache + static_cast<size_t>(s) * kn + hkv * head_dim + d0;
        float dot = q0 * kh[0] + q1 * kh[1] + q2 * kh[2] + q3 * kh[3] + q4 * kh[4] + q5 * kh[5] + q6 * kh[6] +
                    q7 * kh[7];
        dot = warp_sum(dot) * scale;
        const float m2 = fmaxf(mi, dot);
        const float alpha = expf(mi - m2);
        const float w = expf(dot - m2);
        li = li * alpha + w;
        const float* vh = v_cache + static_cast<size_t>(s) * kn + hkv * head_dim + d0;
        a0 = a0 * alpha + w * vh[0];
        a1 = a1 * alpha + w * vh[1];
        a2 = a2 * alpha + w * vh[2];
        a3 = a3 * alpha + w * vh[3];
        a4 = a4 * alpha + w * vh[4];
        a5 = a5 * alpha + w * vh[5];
        a6 = a6 * alpha + w * vh[6];
        a7 = a7 * alpha + w * vh[7];
        mi = m2;
    }
    const float inv = li > 0.f ? 1.f / li : 0.f;
    float* oh = o + (static_cast<size_t>(t) * n_q + hq) * head_dim + d0;
    oh[0] = a0 * inv;
    oh[1] = a1 * inv;
    oh[2] = a2 * inv;
    oh[3] = a3 * inv;
    oh[4] = a4 * inv;
    oh[5] = a5 * inv;
    oh[6] = a6 * inv;
    oh[7] = a7 * inv;
}

// QT queries share each K/V load (cuts attn DRAM ~QT× on long prefill).
template <int QT>
__global__ void attn_prefill_qtile_k(const float* q, const float* k_cache, const float* v_cache, float* o,
                                     int pos0, int T, int n_q, int n_kv, int head_dim) {
    const int hq = blockIdx.x;
    const int tb = blockIdx.y * QT;
    if (hq >= n_q || tb >= T) return;
    const int nt = T - tb < QT ? T - tb : QT;
    const int hkv = hq / (n_q / n_kv);
    const int lane = threadIdx.x;
    const int d0 = lane * 8;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int kn = n_kv * head_dim;
    float qv[QT][8];
    float acc[QT][8];
    float mi[QT], li[QT];
#pragma unroll
    for (int t = 0; t < QT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[t][i] = 0.f;
        mi[t] = -1e30f;
        li[t] = 0.f;
        if (t < nt) {
            const float* qh = q + (static_cast<size_t>(tb + t) * n_q + hq) * head_dim + d0;
#pragma unroll
            for (int i = 0; i < 8; ++i) qv[t][i] = qh[i];
        }
    }
    const int tend_max = pos0 + tb + nt;
    for (int s = 0; s < tend_max; ++s) {
        const float* kh = k_cache + static_cast<size_t>(s) * kn + hkv * head_dim + d0;
        const float4 kA = *reinterpret_cast<const float4*>(kh);
        const float4 kB = *reinterpret_cast<const float4*>(kh + 4);
        const float k0 = kA.x, k1 = kA.y, k2 = kA.z, k3 = kA.w, k4 = kB.x, k5 = kB.y, k6 = kB.z, k7 = kB.w;
        const float* vh = v_cache + static_cast<size_t>(s) * kn + hkv * head_dim + d0;
        const float4 vA = *reinterpret_cast<const float4*>(vh);
        const float4 vB = *reinterpret_cast<const float4*>(vh + 4);
        const float v0 = vA.x, v1 = vA.y, v2 = vA.z, v3 = vA.w, v4 = vB.x, v5 = vB.y, v6 = vB.z, v7 = vB.w;
#pragma unroll
        for (int t = 0; t < QT; ++t) {
            if (t >= nt || s > pos0 + tb + t) continue;
            float dot = qv[t][0] * k0 + qv[t][1] * k1 + qv[t][2] * k2 + qv[t][3] * k3 + qv[t][4] * k4 +
                        qv[t][5] * k5 + qv[t][6] * k6 + qv[t][7] * k7;
            dot = warp_sum(dot) * scale;
            const float m2 = fmaxf(mi[t], dot);
            const float alpha = expf(mi[t] - m2);
            const float w = expf(dot - m2);
            li[t] = li[t] * alpha + w;
            acc[t][0] = acc[t][0] * alpha + w * v0;
            acc[t][1] = acc[t][1] * alpha + w * v1;
            acc[t][2] = acc[t][2] * alpha + w * v2;
            acc[t][3] = acc[t][3] * alpha + w * v3;
            acc[t][4] = acc[t][4] * alpha + w * v4;
            acc[t][5] = acc[t][5] * alpha + w * v5;
            acc[t][6] = acc[t][6] * alpha + w * v6;
            acc[t][7] = acc[t][7] * alpha + w * v7;
            mi[t] = m2;
        }
    }
#pragma unroll
    for (int t = 0; t < QT; ++t) {
        if (t >= nt) break;
        const float inv = li[t] > 0.f ? 1.f / li[t] : 0.f;
        float* oh = o + (static_cast<size_t>(tb + t) * n_q + hq) * head_dim + d0;
#pragma unroll
        for (int i = 0; i < 8; ++i) oh[i] = acc[t][i] * inv;
    }
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
    if ((n & 3) == 0) {
        for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
            const float4 v = reinterpret_cast<const float4*>(xt)[i];
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xt[i] * xt[i];
    }
    buf[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(buf[0] / static_cast<float>(n) + eps);
    if (gamma_n <= 0) gamma_n = n;
    if ((n & 3) == 0 && gamma_n == n) {
        for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
            const float4 xv = reinterpret_cast<const float4*>(xt)[i];
            const float4 zv = reinterpret_cast<const float4*>(zt)[i];
            const float4 gv = gamma ? reinterpret_cast<const float4*>(gamma)[i] : make_float4(1.f, 1.f, 1.f, 1.f);
            float4 o;
            o.x = gv.x * (xv.x * inv) * silu_d(zv.x);
            o.y = gv.y * (xv.y * inv) * silu_d(zv.y);
            o.z = gv.z * (xv.z * inv) * silu_d(zv.z);
            o.w = gv.w * (xv.w * inv) * silu_d(zv.w);
            reinterpret_cast<float4*>(yt)[i] = o;
        }
        return;
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float g = gamma ? gamma[i % gamma_n] : 1.f;
        yt[i] = g * (xt[i] * inv) * silu_d(zt[i]);
    }
}

__global__ void gated_rms_xe_batch_k(const float* x, const float* z, const float* gamma, float* y, uint8_t* xe,
                                     int n, int T, float eps, int gamma_n) {
    const int t = blockIdx.x;
    if (t >= T) return;
    __shared__ float buf[256];
    const float* xt = x + static_cast<size_t>(t) * n;
    const float* zt = z + static_cast<size_t>(t) * n;
    float* yt = y + static_cast<size_t>(t) * n;
    uint8_t* yet = xe + static_cast<size_t>(t) * n;
    float ss = 0.f;
    if ((n & 3) == 0) {
        for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
            const float4 v = reinterpret_cast<const float4*>(xt)[i];
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xt[i] * xt[i];
    }
    buf[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(buf[0] / static_cast<float>(n) + eps);
    if (gamma_n <= 0) gamma_n = n;
    if ((n & 3) == 0 && gamma_n == n) {
        for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
            const float4 xv = reinterpret_cast<const float4*>(xt)[i];
            const float4 zv = reinterpret_cast<const float4*>(zt)[i];
            const float4 gv = gamma ? reinterpret_cast<const float4*>(gamma)[i] : make_float4(1.f, 1.f, 1.f, 1.f);
            float4 o;
            o.x = gv.x * (xv.x * inv) * silu_d(zv.x);
            o.y = gv.y * (xv.y * inv) * silu_d(zv.y);
            o.z = gv.z * (xv.z * inv) * silu_d(zv.z);
            o.w = gv.w * (xv.w * inv) * silu_d(zv.w);
            reinterpret_cast<float4*>(yt)[i] = o;
            const int i0 = i << 2;
            yet[i0 + 0] = f32_to_fp8e4(o.x);
            yet[i0 + 1] = f32_to_fp8e4(o.y);
            yet[i0 + 2] = f32_to_fp8e4(o.z);
            yet[i0 + 3] = f32_to_fp8e4(o.w);
        }
        return;
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float g = gamma ? gamma[i % gamma_n] : 1.f;
        const float o = g * (xt[i] * inv) * silu_d(zt[i]);
        yt[i] = o;
        yet[i] = f32_to_fp8e4(o);
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
                                 float eps, int ctx, int kv_f16, int8_t* k_q8, __half* k_sc, uint8_t* v_qs,
                                 __half* v_sc) {
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
    const float* vs = v + hkv * hd;
    // F16 store first: the tq pack below reuses khloc for V's FWHT.
    if (kv_f16 == 1) {
        __half* kc = reinterpret_cast<__half*>(k_cache) + static_cast<size_t>(p) * kn + hkv * hd;
        __half* vc = reinterpret_cast<__half*>(v_cache) + static_cast<size_t>(p) * kn + hkv * hd;
        for (int i = threadIdx.x; i < hd; i += blockDim.x) {
            kc[i] = __float2half(khloc[i]);
            vc[i] = __float2half(vs[i]);
        }
        __syncthreads();
    } else if (kv_f16 == 0) {
        float* kc = k_cache + static_cast<size_t>(p) * kn + hkv * hd;
        float* vc = v_cache + static_cast<size_t>(p) * kn + hkv * hd;
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
    }
    // Persist compact KV whenever buffers are bound, including the F16
    // window (pos<kv_win_). Otherwise the first pos>=win attend reads
    // uninitialized tq for tokens 0..win-1. Only the GQA group leader
    // writes — 6 Q heads share one KV head.
    if (k_q8 && k_sc && v_qs && v_sc && hd == 256 && (hq % rep == 0)) {
        float amax = 0.f;
        for (int i = threadIdx.x; i < hd; i += blockDim.x) amax = fmaxf(amax, fabsf(khloc[i]));
        amax = warp_max_all(amax);
        if ((threadIdx.x & 31) == 0) wss[threadIdx.x >> 5] = amax;
        __syncthreads();
        if (threadIdx.x == 0) {
            float m = wss[0];
            const int nw = blockDim.x >> 5;
            for (int w = 1; w < nw && w < 4; ++w) m = fmaxf(m, wss[w]);
            wss[0] = m / 127.f + 1e-8f;
            k_sc[static_cast<size_t>(p) * n_kv + hkv] = __float2half(wss[0]);
        }
        __syncthreads();
        const float ks = wss[0];
        int8_t* kq = k_q8 + (static_cast<size_t>(p) * n_kv + hkv) * hd;
        for (int i = threadIdx.x; i < hd; i += blockDim.x)
            kq[i] = static_cast<int8_t>(lrintf(fminf(fmaxf(khloc[i] / ks, -127.f), 127.f)));
        const int nblk = hd / kTqBlk;
        const int tid = threadIdx.x;
        for (int b = 0; b < nblk; ++b) {
            if (tid < kTqBlk) khloc[tid] = vs[b * kTqBlk + tid];
            __syncthreads();
            fwht128_sync(khloc, tid);
            float ss = (tid < kTqBlk) ? khloc[tid] * khloc[tid] : 0.f;
            if (tid < kTqBlk) ss = warp_sum_all(ss);
            if (tid < kTqBlk && (tid & 31) == 0) wss[tid >> 5] = ss;
            __syncthreads();
            if (tid == 0) {
                const float rms =
                    sqrtf((wss[0] + wss[1] + wss[2] + wss[3]) / static_cast<float>(kTqBlk)) + 1e-8f;
                wss[0] = rms;
                v_sc[(static_cast<size_t>(p) * n_kv + hkv) * nblk + b] = __float2half(rms);
            }
            __syncthreads();
            __shared__ int tidx[kTqBlk];
            if (tid < kTqBlk) tidx[tid] = tq3_nn(khloc[tid] / wss[0]);
            __syncthreads();
            pack_tq3_16(v_qs + ((static_cast<size_t>(p) * n_kv + hkv) * nblk + b) * kTq3B, tidx, tid);
            __syncthreads();
        }
    }
    __syncthreads();

    const float scale = rsqrtf(static_cast<float>(hd));
    const int Tend = T < ctx ? T : ctx;
    float* oh = o + hq * hd;
    if (kv_f16 == 3) {
        return;
    }
    if (kv_f16 == 1 && hd == 256 && ctx > 8192) {
        flash_attn_decode_f16_hd256(qh, reinterpret_cast<const __half*>(k_cache),
                                    reinterpret_cast<const __half*>(v_cache), oh, Tend, kn, hkv, scale);
        return;
    }
    if (ctx > 8192 || kv_f16 == 2) {
        // Online softmax: scores do not fit in smem at 128k/200k.
        float acc = 0.f;
        for (int t = 0; t < Tend; ++t) {
            float dot = 0.f;
            if (kv_f16 == 2 && k_q8 && k_sc) {
                const float ks = __half2float(k_sc[static_cast<size_t>(t) * n_kv + hkv]);
                const int8_t* kq = k_q8 + (static_cast<size_t>(t) * n_kv + hkv) * hd;
                for (int d = threadIdx.x; d < hd; d += blockDim.x)
                    dot += qh[d] * (ks * static_cast<float>(kq[d]));
            } else if (kv_f16) {
                const __half* kh = reinterpret_cast<const __half*>(k_cache) + static_cast<size_t>(t) * kn +
                                   hkv * hd;
                for (int d = threadIdx.x; d < hd; d += blockDim.x) dot += qh[d] * __half2float(kh[d]);
            } else if (f4) {
                const float* kh = k_cache + static_cast<size_t>(t) * kn + hkv * hd;
                const int i4 = threadIdx.x;
                if (i4 < 32) {
                    const float4 qv = reinterpret_cast<const float4*>(qh)[i4];
                    const float4 kv = reinterpret_cast<const float4*>(kh)[i4];
                    dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
                }
            } else {
                const float* kh = k_cache + static_cast<size_t>(t) * kn + hkv * hd;
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
            if (kv_f16 == 2 && v_qs && v_sc && hd == 256) {
                const int nblk = hd / kTqBlk;
                const int tid = threadIdx.x;
                float vv = 0.f;
                for (int bb = 0; bb < nblk; ++bb) {
                    const float vscl =
                        __half2float(v_sc[(static_cast<size_t>(t) * n_kv + hkv) * nblk + bb]);
                    const uint8_t* src =
                        v_qs + ((static_cast<size_t>(t) * n_kv + hkv) * nblk + bb) * kTq3B;
                    if (tid < kTqBlk) unpack_tq3_tid(src, tid, vscl, khloc);
                    __syncthreads();
                    fwht128_sync(khloc, tid);
                    if (tid >= bb * kTqBlk && tid < (bb + 1) * kTqBlk) vv = khloc[tid - bb * kTqBlk];
                    __syncthreads();
                }
                if (tid < hd) acc = acc * aa + pp * vv;
            } else if (kv_f16) {
                const __half* vh = reinterpret_cast<const __half*>(v_cache) + static_cast<size_t>(t) * kn +
                                   hkv * hd;
                if (threadIdx.x < hd) acc = acc * aa + pp * __half2float(vh[threadIdx.x]);
            } else {
                const float* vh = v_cache + static_cast<size_t>(t) * kn + hkv * hd;
                if (threadIdx.x < hd) acc = acc * aa + pp * vh[threadIdx.x];
            }
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
    if ((n & 3) == 0) {
        for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
            const float4 v = reinterpret_cast<const float4*>(xt)[i];
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xt[i] * xt[i];
    }
    buf[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(buf[0] / static_cast<float>(n) + eps);
    if ((n & 3) == 0) {
        for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
            const float4 v = reinterpret_cast<const float4*>(xt)[i];
            const float4 g =
                gamma ? reinterpret_cast<const float4*>(gamma)[i] : make_float4(0.f, 0.f, 0.f, 0.f);
            float4 o;
            o.x = v.x * inv * (1.f + g.x);
            o.y = v.y * inv * (1.f + g.y);
            o.z = v.z * inv * (1.f + g.z);
            o.w = v.w * inv * (1.f + g.w);
            reinterpret_cast<float4*>(yt)[i] = o;
        }
        return;
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float g = gamma ? (1.f + gamma[i]) : 1.f;
        yt[i] = xt[i] * inv * g;
    }
}

__global__ void rmsnorm_xe_batch_k(const float* x, const float* gamma, float* y, uint8_t* xe, int n, int T,
                                   float eps) {
    const int t = blockIdx.x;
    if (t >= T) return;
    __shared__ float buf[256];
    const float* xt = x + static_cast<size_t>(t) * n;
    float* yt = y + static_cast<size_t>(t) * n;
    uint8_t* yet = xe + static_cast<size_t>(t) * n;
    float ss = 0.f;
    if ((n & 3) == 0) {
        for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
            const float4 v = reinterpret_cast<const float4*>(xt)[i];
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xt[i] * xt[i];
    }
    buf[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(buf[0] / static_cast<float>(n) + eps);
    if ((n & 3) == 0) {
        for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
            const float4 v = reinterpret_cast<const float4*>(xt)[i];
            const float4 g =
                gamma ? reinterpret_cast<const float4*>(gamma)[i] : make_float4(0.f, 0.f, 0.f, 0.f);
            float4 o;
            o.x = v.x * inv * (1.f + g.x);
            o.y = v.y * inv * (1.f + g.y);
            o.z = v.z * inv * (1.f + g.z);
            o.w = v.w * inv * (1.f + g.w);
            reinterpret_cast<float4*>(yt)[i] = o;
            const int i0 = i << 2;
            yet[i0 + 0] = f32_to_fp8e4(o.x);
            yet[i0 + 1] = f32_to_fp8e4(o.y);
            yet[i0 + 2] = f32_to_fp8e4(o.z);
            yet[i0 + 3] = f32_to_fp8e4(o.w);
        }
        return;
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float g = gamma ? (1.f + gamma[i]) : 1.f;
        const float o = xt[i] * inv * g;
        yt[i] = o;
        yet[i] = f32_to_fp8e4(o);
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

// One 128×128 scale tile per block. 1024 threads → 4 rows/warp. FP8→half via HW cvt.
__global__ void dequant_fp8_kmaj_row_k(const uint8_t* W, const float* scale, __half* out, int m, int n) {
    const int row0 = blockIdx.x * 128;
    const int cb = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    const int nb_n = n >> 7;
    if (cb >= nb_n || !scale) return;
    const __half hs = __float2half(scale[(row0 >> 7) * nb_n + cb]);
    const int k0 = cb * 128 + lane * 4;
    for (int r = warp; r < 128; r += nwarps) {
        const int row = row0 + r;
        if (row >= m) continue;
        const uint32_t p = __ldcs(fp8_blk4(W, row, m, cb, lane));
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&p);
        __half* dst = out + static_cast<size_t>(row) * n + k0;
#if __CUDA_ARCH__ >= 800
        __half_raw r0 = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b[0]), __NV_E4M3);
        __half_raw r1 = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b[1]), __NV_E4M3);
        __half_raw r2 = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b[2]), __NV_E4M3);
        __half_raw r3 = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b[3]), __NV_E4M3);
        __half h0, h1, h2, h3;
        *reinterpret_cast<unsigned short*>(&h0) = r0.x;
        *reinterpret_cast<unsigned short*>(&h1) = r1.x;
        *reinterpret_cast<unsigned short*>(&h2) = r2.x;
        *reinterpret_cast<unsigned short*>(&h3) = r3.x;
        dst[0] = __hmul(h0, hs);
        dst[1] = __hmul(h1, hs);
        dst[2] = __hmul(h2, hs);
        dst[3] = __hmul(h3, hs);
#else
        dst[0] = __float2half(fp8e4_to_f32(b[0]) * __half2float(hs));
        dst[1] = __float2half(fp8e4_to_f32(b[1]) * __half2float(hs));
        dst[2] = __float2half(fp8e4_to_f32(b[2]) * __half2float(hs));
        dst[3] = __float2half(fp8e4_to_f32(b[3]) * __half2float(hs));
#endif
    }
}

// k-major e4 + block scale → row-major e4 (scale absorbed). Prefill-only;
// T=1 GEMV keeps the original k-major + scale path.
__global__ void unpack_fp8_kmaj_e4_k(const uint8_t* W, const float* scale, uint8_t* out, int m, int n) {
    const int row0 = blockIdx.x * 128;
    const int cb = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    const int nb_n = n >> 7;
    if (cb >= nb_n || !scale) return;
    const float s = scale[(row0 >> 7) * nb_n + cb];
    const int k0 = cb * 128 + lane * 4;
    for (int r = warp; r < 128; r += nwarps) {
        const int row = row0 + r;
        if (row >= m) continue;
        const uint32_t p = __ldcs(fp8_blk4(W, row, m, cb, lane));
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&p);
        uint8_t* dst = out + static_cast<size_t>(row) * n + k0;
        dst[0] = f32_to_fp8e4(fp8e4_to_f32(b[0]) * s);
        dst[1] = f32_to_fp8e4(fp8e4_to_f32(b[1]) * s);
        dst[2] = f32_to_fp8e4(fp8e4_to_f32(b[2]) * s);
        dst[3] = f32_to_fp8e4(fp8e4_to_f32(b[3]) * s);
    }
}

__global__ void absorb_fp8_scale_k(uint8_t* q, const float* sc, int m, int n) {
    // 1D over (row, col/4): lm_head rows (248320) exceed gridDim.y=65535.
    const int n4 = n >> 2;
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t tot = static_cast<size_t>(m) * static_cast<size_t>(n4);
    if (i >= tot || !sc) return;
    const int r = static_cast<int>(i / static_cast<size_t>(n4));
    const int c = static_cast<int>(i - static_cast<size_t>(r) * static_cast<size_t>(n4)) << 2;
    const float s = sc[(r >> 7) * (n >> 7) + (c >> 7)];
    uint8_t* p = q + static_cast<size_t>(r) * n + c;
    p[0] = f32_to_fp8e4(fp8e4_to_f32(p[0]) * s);
    p[1] = f32_to_fp8e4(fp8e4_to_f32(p[1]) * s);
    p[2] = f32_to_fp8e4(fp8e4_to_f32(p[2]) * s);
    p[3] = f32_to_fp8e4(fp8e4_to_f32(p[3]) * s);
}

__device__ __forceinline__ float e4_dot16_rm(uint4 p, const float* x) {
    return fp8x4_dot(p.x, x) + fp8x4_dot(p.y, x + 4) + fp8x4_dot(p.z, x + 8) + fp8x4_dot(p.w, x + 12);
}

// Row-major e4 (scales already absorbed). T=1 path; k-major GEMV kernels unchanged.
__global__ void __launch_bounds__(512, 2) gemv_fp8_rm_2row_k(const uint8_t* W, const float* x, float* y, int m,
                                                             int n, int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int tile = n <= kFp8XsCap ? n : kFp8XsCap;
    float acc0 = 0.f, acc1 = 0.f;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        load_x_tile(xs, x + t0, tn);
        __syncthreads();
        if (row0 < m) {
            const uint8_t* w0 = W + static_cast<size_t>(row0) * n + t0;
            const uint8_t* w1 = (row1 < m) ? W + static_cast<size_t>(row1) * n + t0 : nullptr;
            for (int j = lane * 16; j + 15 < tn; j += 512) {
                acc0 += e4_dot16_rm(*reinterpret_cast<const uint4*>(w0 + j), xs + j);
                if (w1) acc1 += e4_dot16_rm(*reinterpret_cast<const uint4*>(w1 + j), xs + j);
            }
        }
        if (t0 + tile < n) __syncthreads();
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

// One warp / output row. Used for skinny leftovers (m=48) and medium K/V
// (m=1024) so those matrices never pay a full-matrix FP8→FP16 dequant + cuBLAS
// setup (~1 ms each, ~150 ms of 256-fill linear).
template <int TN>
__global__ void gemm_fp8_row_k(const uint8_t* W, const float* scale, const float* X, float* Y, int m,
                               int n, int T, int add) {
    constexpr int KN = 512;
    extern __shared__ float xs[];
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (row >= m || !scale) return;
    const int nb_n = n >> 7;
    const int bi = row >> 7;
    for (int t0 = 0; t0 < T; t0 += TN) {
        const int tt = T - t0 < TN ? T - t0 : TN;
        float acc[TN];
#pragma unroll
        for (int t = 0; t < TN; ++t) acc[t] = 0.f;
        for (int k0 = 0; k0 < n; k0 += KN) {
            const int kn = n - k0 < KN ? n - k0 : KN;
            for (int i = lane; i < tt * kn; i += 32) {
                const int t = i / kn;
                const int c = i - t * kn;
                xs[t * KN + c] = X[static_cast<size_t>(t0 + t) * n + k0 + c];
            }
            __syncwarp();
            if (kn == KN) {
                const int b0 = k0 >> 7;
                const float4 sv = __ldg(reinterpret_cast<const float4*>(scale + bi * nb_n + b0));
                const uint4 p = fp8_ld16(W, row, m, b0, lane);
#pragma unroll
                for (int t = 0; t < TN; ++t) {
                    if (t < tt) acc[t] += fp8_dot16(p, sv, xs + t * KN + (lane << 2));
                }
            } else {
                const int nb = kn >> 7;
                for (int b = 0; b < nb; ++b) {
                    const float s = scale[bi * nb_n + (k0 >> 7) + b];
                    const uint32_t p = __ldcs(fp8_blk4(W, row, m, (k0 >> 7) + b, lane));
#pragma unroll
                    for (int t = 0; t < TN; ++t) {
                        if (t < tt) acc[t] += s * fp8x4_dot(p, xs + t * KN + b * 128 + (lane << 2));
                    }
                }
            }
            __syncwarp();
        }
#pragma unroll
        for (int t = 0; t < TN; ++t) {
            if (t < tt) {
                const float a = warp_sum(acc[t]);
                if (lane == 0) write_y(Y + static_cast<size_t>(t0 + t) * m, row, a, add);
            }
        }
    }
}

// k-major FP8 → smem FP16 tile (128×128, padded ld) × X panel. ldmatrix
// reads the tiles directly — no per-k16 A/B staging copies (those made the
// first shared-TC attempt 3× slower than dequant+cuBLAS). Full W never
// lands in HBM as FP16.
template <int BN>
__global__ void __launch_bounds__(256, 2) gemm_fp8_shared_tc_k(const uint8_t* W, const float* scale,
                                                              const float* X, float* Y, int m, int n,
                                                              int T, int add) {
#if __CUDA_ARCH__ >= 800
    constexpr int BM = 128, BK = 128, WLD = 136, XLD = 136, NG = BN / 8;
    const int row_base = blockIdx.x * BM;
    const int t0 = blockIdx.y * BN;
    const int tt = T - t0 < BN ? T - t0 : BN;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wrow = row_base + warp * 16;
    extern __shared__ char raw[];
    __half* Wf = reinterpret_cast<__half*>(raw);
    __half* Xf = Wf + BM * WLD;
    float acc[NG][4];
#pragma unroll
    for (int g = 0; g < NG; ++g) acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
    const int nb_n = n >> 7;
    const int bi = row_base >> 7;
    for (int k0 = 0; k0 < n; k0 += BK) {
        const int cb = k0 >> 7;
        const float s = (row_base < m && scale) ? scale[bi * nb_n + cb] : 0.f;
        const __half hs = __float2half(s);
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const int row = wrow + r;
            const uint32_t p = (row < m) ? *fp8_blk4(W, row, m, cb, lane) : 0u;
            const uint8_t* b = reinterpret_cast<const uint8_t*>(&p);
            __half* dst = Wf + (warp * 16 + r) * WLD + (lane << 2);
#if __CUDA_ARCH__ >= 890
            __half_raw z0 = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b[0]), __NV_E4M3);
            __half_raw z1 = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b[1]), __NV_E4M3);
            __half_raw z2 = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b[2]), __NV_E4M3);
            __half_raw z3 = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b[3]), __NV_E4M3);
            __half h0, h1, h2, h3;
            *reinterpret_cast<unsigned short*>(&h0) = z0.x;
            *reinterpret_cast<unsigned short*>(&h1) = z1.x;
            *reinterpret_cast<unsigned short*>(&h2) = z2.x;
            *reinterpret_cast<unsigned short*>(&h3) = z3.x;
            dst[0] = __hmul(h0, hs);
            dst[1] = __hmul(h1, hs);
            dst[2] = __hmul(h2, hs);
            dst[3] = __hmul(h3, hs);
#else
            dst[0] = __float2half(fp8e4_to_f32(b[0]) * s);
            dst[1] = __float2half(fp8e4_to_f32(b[1]) * s);
            dst[2] = __float2half(fp8e4_to_f32(b[2]) * s);
            dst[3] = __float2half(fp8e4_to_f32(b[3]) * s);
#endif
        }
        for (int i = threadIdx.x * 4; i < tt * BK; i += 1024) {
            const int t = i >> 7;
            const int c = i & 127;
            const float4 v = *reinterpret_cast<const float4*>(X + static_cast<size_t>(t0 + t) * n + k0 + c);
            Xf[t * XLD + c + 0] = __float2half(v.x);
            Xf[t * XLD + c + 1] = __float2half(v.y);
            Xf[t * XLD + c + 2] = __float2half(v.z);
            Xf[t * XLD + c + 3] = __float2half(v.w);
        }
        for (int t = tt; t < BN; ++t) {
            for (int c = threadIdx.x; c < BK; c += 256) Xf[t * XLD + c] = __half(0);
        }
        __syncthreads();
#pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            uint32_t a[4];
            ldmatrix_x4(a, Wf + (warp * 16 + (lane & 15)) * WLD + kt * 16 + ((lane >> 4) << 3));
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                if (g * 8 >= tt) continue;
                uint32_t bv[2];
                ldmatrix_x2(bv, Xf + (g * 8 + (lane & 7)) * XLD + kt * 16 + ((lane >> 3) << 3));
                mma_m16n8k16_f16(acc[g], a, bv);
            }
        }
        __syncthreads();
    }
    if (wrow < m) {
        const int r0 = lane >> 2;
        const int c0 = (lane & 3) << 1;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const int tok = g * 8 + c0;
            if (tok < tt) {
                if (wrow + r0 < m) write_y(Y + static_cast<size_t>(t0 + tok) * m, wrow + r0, acc[g][0], add);
                if (wrow + r0 + 8 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok) * m, wrow + r0 + 8, acc[g][2], add);
            }
            if (tok + 1 < tt) {
                if (wrow + r0 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok + 1) * m, wrow + r0, acc[g][1], add);
                if (wrow + r0 + 8 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok + 1) * m, wrow + r0 + 8, acc[g][3], add);
            }
        }
    }
#else
    (void)W;
    (void)scale;
    (void)X;
    (void)Y;
    (void)m;
    (void)n;
    (void)T;
    (void)add;
#endif
}

// Native e4m3 tensor-core GEMM: k-major W is only reshuffled into a padded
// smem tile (still e4m3). X panel is quantized to e4m3. Block scale is
// applied on the accumulator. No HBM FP16 weight table. SM89+.
template <int BN>
__global__ void __launch_bounds__(256, 4) gemm_fp8_kmaj_e4_k(const uint8_t* W, const float* scale,
                                                            const float* X, float* Y, int m, int n, int T,
                                                            int add) {
#if __CUDA_ARCH__ >= 890
    constexpr int BM = 128, BK = 128, WLD = 144, XLD = 144, NG = BN / 8;
    const int row_base = blockIdx.x * BM;
    const int t0 = blockIdx.y * BN;
    const int tt = T - t0 < BN ? T - t0 : BN;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wrow = row_base + warp * 16;
    extern __shared__ char raw[];
    uint8_t* Wf = reinterpret_cast<uint8_t*>(raw);
    uint8_t* Xf = Wf + BM * WLD;
    float acc[NG][4];
#pragma unroll
    for (int g = 0; g < NG; ++g) acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
    const int nb_n = n >> 7;
    const int bi = row_base >> 7;
    for (int k0 = 0; k0 < n; k0 += BK) {
            const int cb = k0 >> 7;
            const float s = (row_base < m && scale) ? scale[bi * nb_n + cb] : 0.f;
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                const int row = wrow + r;
                const uint32_t p = (row < m) ? *fp8_blk4(W, row, m, cb, lane) : 0u;
                *reinterpret_cast<uint32_t*>(Wf + (warp * 16 + r) * WLD + (lane << 2)) = p;
            }
            for (int i = threadIdx.x * 4; i < tt * BK; i += 1024) {
                const int t = i >> 7;
                const int c = i & 127;
                const float4 v = *reinterpret_cast<const float4*>(X + static_cast<size_t>(t0 + t) * n + k0 + c);
                Xf[t * XLD + c + 0] = f32_to_fp8e4(v.x);
                Xf[t * XLD + c + 1] = f32_to_fp8e4(v.y);
                Xf[t * XLD + c + 2] = f32_to_fp8e4(v.z);
                Xf[t * XLD + c + 3] = f32_to_fp8e4(v.w);
            }
            for (int t = tt; t < BN; ++t) {
                for (int c = threadIdx.x; c < BK; c += 256) Xf[t * XLD + c] = 0;
            }
            __syncthreads();
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                uint32_t a[4];
                ldmatrix_x4(a, Wf + (warp * 16 + (lane & 15)) * WLD + kt * 32 + ((lane >> 4) << 4));
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    if (g * 8 >= tt) continue;
                    uint32_t bv[2];
                    ldmatrix_x2(bv, Xf + (g * 8 + (lane & 7)) * XLD + kt * 32 + ((lane >> 3) << 4));
                    float d[4] = {0.f, 0.f, 0.f, 0.f};
                    mma_m16n8k32_e4m3(d, a, bv);
                    acc[g][0] += d[0] * s;
                    acc[g][1] += d[1] * s;
                    acc[g][2] += d[2] * s;
                    acc[g][3] += d[3] * s;
                }
            }
            __syncthreads();
        }
        if (wrow < m) {
            const int r0 = lane >> 2;
            const int c0 = (lane & 3) << 1;
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                const int tok = g * 8 + c0;
                if (tok < tt) {
                    if (wrow + r0 < m) write_y(Y + static_cast<size_t>(t0 + tok) * m, wrow + r0, acc[g][0], add);
                    if (wrow + r0 + 8 < m)
                        write_y(Y + static_cast<size_t>(t0 + tok) * m, wrow + r0 + 8, acc[g][2], add);
                }
                if (tok + 1 < tt) {
                    if (wrow + r0 < m)
                        write_y(Y + static_cast<size_t>(t0 + tok + 1) * m, wrow + r0, acc[g][1], add);
                    if (wrow + r0 + 8 < m)
                        write_y(Y + static_cast<size_t>(t0 + tok + 1) * m, wrow + r0 + 8, acc[g][3], add);
                }
            }
        }
#else
    (void)W;
    (void)scale;
    (void)X;
    (void)Y;
    (void)m;
    (void)n;
    (void)T;
    (void)add;
#endif
}

// 512 threads / 16 warps: two warps share 16 rows and split a 128-token
// panel (2 W passes at T=256 vs 4). SM89+.
__global__ void __launch_bounds__(512, 2) gemm_fp8_kmaj_e4_bn128_k(const uint8_t* W, const float* scale,
                                                                  const float* X, float* Y, int m, int n,
                                                                  int T, int add) {
#if __CUDA_ARCH__ >= 890
    constexpr int BM = 128, BN = 128, BK = 128, WLD = 144, XLD = 144, NG = 8;
    const int row_base = blockIdx.x * BM;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wr = warp >> 1;
    const int wn = warp & 1;
    const int wrow = row_base + wr * 16;
    const int g0 = wn * NG;
    extern __shared__ char raw[];
    uint8_t* Wf = reinterpret_cast<uint8_t*>(raw);
    uint8_t* Xf = Wf + BM * WLD;
    const int nb_n = n >> 7;
    const int bi = row_base >> 7;
    const int tile0 = gridDim.y > 1 ? blockIdx.y : 0;
    const int ntile = (T + BN - 1) / BN;
    const int tile1 = gridDim.y > 1 ? blockIdx.y + 1 : ntile;
    for (int tile = tile0; tile < tile1; ++tile) {
    const int t0 = tile * BN;
    const int tt = T - t0 < BN ? T - t0 : BN;
    float acc[NG][4];
#pragma unroll
    for (int g = 0; g < NG; ++g) acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
    for (int k0 = 0; k0 < n; k0 += BK) {
        const int cb = k0 >> 7;
        const float s = (row_base < m && scale) ? scale[bi * nb_n + cb] : 0.f;
        if (wn == 0) {
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                const int row = wrow + r;
                const uint32_t p = (row < m) ? *fp8_blk4(W, row, m, cb, lane) : 0u;
                *reinterpret_cast<uint32_t*>(Wf + (wr * 16 + r) * WLD + (lane << 2)) = p;
            }
        }
        for (int i = threadIdx.x * 4; i < tt * BK; i += 2048) {
            const int t = i >> 7;
            const int c = i & 127;
            const float4 v = *reinterpret_cast<const float4*>(X + static_cast<size_t>(t0 + t) * n + k0 + c);
            Xf[t * XLD + c + 0] = f32_to_fp8e4(v.x);
            Xf[t * XLD + c + 1] = f32_to_fp8e4(v.y);
            Xf[t * XLD + c + 2] = f32_to_fp8e4(v.z);
            Xf[t * XLD + c + 3] = f32_to_fp8e4(v.w);
        }
        for (int t = tt; t < BN; ++t) {
            for (int c = threadIdx.x; c < BK; c += 512) Xf[t * XLD + c] = 0;
        }
        __syncthreads();
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint32_t a[4];
            ldmatrix_x4(a, Wf + (wr * 16 + (lane & 15)) * WLD + kt * 32 + ((lane >> 4) << 4));
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                const int gg = g0 + g;
                if (gg * 8 >= tt) continue;
                uint32_t bv[2];
                ldmatrix_x2(bv, Xf + (gg * 8 + (lane & 7)) * XLD + kt * 32 + ((lane >> 3) << 4));
                float d[4] = {0.f, 0.f, 0.f, 0.f};
                mma_m16n8k32_e4m3(d, a, bv);
                acc[g][0] += d[0] * s;
                acc[g][1] += d[1] * s;
                acc[g][2] += d[2] * s;
                acc[g][3] += d[3] * s;
            }
        }
        __syncthreads();
    }
    if (wrow < m) {
        const int r0 = lane >> 2;
        const int c0 = (lane & 3) << 1;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const int tok = (g0 + g) * 8 + c0;
            if (tok < tt) {
                if (wrow + r0 < m) write_y(Y + static_cast<size_t>(t0 + tok) * m, wrow + r0, acc[g][0], add);
                if (wrow + r0 + 8 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok) * m, wrow + r0 + 8, acc[g][2], add);
            }
            if (tok + 1 < tt) {
                if (wrow + r0 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok + 1) * m, wrow + r0, acc[g][1], add);
                if (wrow + r0 + 8 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok + 1) * m, wrow + r0 + 8, acc[g][3], add);
            }
        }
    }
    if (tile + 1 < tile1) __syncthreads();
    }
#else
    (void)W;
    (void)scale;
    (void)X;
    (void)Y;
    (void)m;
    (void)n;
    (void)T;
    (void)add;
#endif
}

// 1024 threads / 32 warps: four warps share 16 rows and split a 256-token
// panel. 2D grid (row, T/256) — each CTA owns one T panel, so T=256 is one
// W pass and T=1024 is four parallel panels (not the failed 1D inner-T scan).
__global__ void __launch_bounds__(1024, 1) gemm_fp8_kmaj_e4_bn256_k(const uint8_t* W, const float* scale,
                                                                   const float* X, float* Y, int m, int n,
                                                                   int T, int add) {
#if __CUDA_ARCH__ >= 890
    constexpr int BM = 128, BN = 256, BK = 128, WLD = 144, XLD = 144, NG = 8;
    const int row_base = blockIdx.x * BM;
    const int t0 = blockIdx.y * BN;
    const int tt = T - t0 < BN ? T - t0 : BN;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wr = warp >> 2;
    const int wt = warp & 3;
    const int wrow = row_base + wr * 16;
    const int g0 = wt * NG;
    extern __shared__ char raw[];
    uint8_t* Wf = reinterpret_cast<uint8_t*>(raw);
    uint8_t* Xf = Wf + BM * WLD;
    float acc[NG][4];
#pragma unroll
    for (int g = 0; g < NG; ++g) acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
    const int nb_n = n >> 7;
    const int bi = row_base >> 7;
    for (int k0 = 0; k0 < n; k0 += BK) {
        const int cb = k0 >> 7;
        const float s = (row_base < m && scale) ? scale[bi * nb_n + cb] : 0.f;
        if (wt == 0) {
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                const int row = wrow + r;
                const uint32_t p = (row < m) ? *fp8_blk4(W, row, m, cb, lane) : 0u;
                *reinterpret_cast<uint32_t*>(Wf + (wr * 16 + r) * WLD + (lane << 2)) = p;
            }
        }
        for (int i = threadIdx.x * 4; i < tt * BK; i += 4096) {
            const int t = i >> 7;
            const int c = i & 127;
            const float4 v = *reinterpret_cast<const float4*>(X + static_cast<size_t>(t0 + t) * n + k0 + c);
            Xf[t * XLD + c + 0] = f32_to_fp8e4(v.x);
            Xf[t * XLD + c + 1] = f32_to_fp8e4(v.y);
            Xf[t * XLD + c + 2] = f32_to_fp8e4(v.z);
            Xf[t * XLD + c + 3] = f32_to_fp8e4(v.w);
        }
        for (int t = tt; t < BN; ++t) {
            for (int c = threadIdx.x; c < BK; c += 1024) Xf[t * XLD + c] = 0;
        }
        __syncthreads();
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint32_t a[4];
            ldmatrix_x4(a, Wf + (wr * 16 + (lane & 15)) * WLD + kt * 32 + ((lane >> 4) << 4));
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                const int gg = g0 + g;
                if (gg * 8 >= tt) continue;
                uint32_t bv[2];
                ldmatrix_x2(bv, Xf + (gg * 8 + (lane & 7)) * XLD + kt * 32 + ((lane >> 3) << 4));
                float d[4] = {0.f, 0.f, 0.f, 0.f};
                mma_m16n8k32_e4m3(d, a, bv);
                acc[g][0] += d[0] * s;
                acc[g][1] += d[1] * s;
                acc[g][2] += d[2] * s;
                acc[g][3] += d[3] * s;
            }
        }
        __syncthreads();
    }
    if (wrow < m) {
        const int r0 = lane >> 2;
        const int c0 = (lane & 3) << 1;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const int tok = (g0 + g) * 8 + c0;
            if (tok < tt) {
                if (wrow + r0 < m) write_y(Y + static_cast<size_t>(t0 + tok) * m, wrow + r0, acc[g][0], add);
                if (wrow + r0 + 8 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok) * m, wrow + r0 + 8, acc[g][2], add);
            }
            if (tok + 1 < tt) {
                if (wrow + r0 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok + 1) * m, wrow + r0, acc[g][1], add);
                if (wrow + r0 + 8 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok + 1) * m, wrow + r0 + 8, acc[g][3], add);
            }
        }
    }
#else
    (void)W;
    (void)scale;
    (void)X;
    (void)Y;
    (void)m;
    (void)n;
    (void)T;
    (void)add;
#endif
}

// 512-col W super-tile: one fp8_ld16 (uint4) feeds four 128-col MMA K-steps.
// 2D T-split (BN=128), not a 1D T-scan and not T=256/BN=256.
__global__ void __launch_bounds__(512, 1) gemm_fp8_kmaj_e4_bk512_k(const uint8_t* W, const float* scale,
                                                                  const float* X, float* Y, int m, int n,
                                                                  int T, int add, const uint8_t* Xe) {
#if __CUDA_ARCH__ >= 890
    constexpr int BM = 128, BN = 128, BK = 128, WLD = 528, XLD = 144, NG = 8;
    const int row_base = blockIdx.x * BM;
    const int t0 = blockIdx.y * BN;
    const int tt = T - t0 < BN ? T - t0 : BN;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wr = warp >> 1;
    const int wn = warp & 1;
    const int wrow = row_base + wr * 16;
    const int g0 = wn * NG;
    extern __shared__ char raw[];
    uint8_t* Wf = reinterpret_cast<uint8_t*>(raw);
    uint8_t* Xf = Wf + BM * WLD;
    float acc[NG][4];
#pragma unroll
    for (int g = 0; g < NG; ++g) acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
    const int nb_n = n >> 7;
    const int bi = row_base >> 7;
    for (int k0 = 0; k0 < n; k0 += 512) {
        const int cb0 = k0 >> 7;
        if (wn == 0) {
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                const int row = wrow + r;
                const uint4 p = (row < m) ? fp8_ld16(W, row, m, cb0, lane) : make_uint4(0, 0, 0, 0);
                uint8_t* dst = Wf + (wr * 16 + r) * WLD + (lane << 2);
                *reinterpret_cast<uint32_t*>(dst + 0) = p.x;
                *reinterpret_cast<uint32_t*>(dst + 128) = p.y;
                *reinterpret_cast<uint32_t*>(dst + 256) = p.z;
                *reinterpret_cast<uint32_t*>(dst + 384) = p.w;
            }
        }
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
            const int kk = k0 + kb * BK;
            if (kk >= n) break;
            const float s = (row_base < m && scale) ? scale[bi * nb_n + cb0 + kb] : 0.f;
            if (Xe) {
                for (int i = threadIdx.x * 16; i < tt * BK; i += 8192) {
                    const int t = i >> 7;
                    const int c = i & 127;
                    uint4 v;
                    const unsigned long long addr = reinterpret_cast<unsigned long long>(
                        Xe + static_cast<size_t>(t0 + t) * n + kk + c);
                    asm volatile("ld.global.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                                 : "l"(addr));
                    *reinterpret_cast<uint4*>(Xf + t * XLD + c) = v;
                }
            } else {
                for (int i = threadIdx.x * 4; i < tt * BK; i += 2048) {
                    const int t = i >> 7;
                    const int c = i & 127;
                    const float4 v =
                        *reinterpret_cast<const float4*>(X + static_cast<size_t>(t0 + t) * n + kk + c);
                    Xf[t * XLD + c + 0] = f32_to_fp8e4(v.x);
                    Xf[t * XLD + c + 1] = f32_to_fp8e4(v.y);
                    Xf[t * XLD + c + 2] = f32_to_fp8e4(v.z);
                    Xf[t * XLD + c + 3] = f32_to_fp8e4(v.w);
                }
            }
            if (tt < BN) {
                for (int t = tt; t < BN; ++t) {
                    for (int c = threadIdx.x; c < BK; c += 512) Xf[t * XLD + c] = 0;
                }
            }
            __syncthreads();
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                uint32_t a[4];
                ldmatrix_x4(a, Wf + (wr * 16 + (lane & 15)) * WLD + kb * 128 + kt * 32 + ((lane >> 4) << 4));
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    const int gg = g0 + g;
                    if (gg * 8 >= tt) continue;
                    uint32_t bv[2];
                    ldmatrix_x2(bv, Xf + (gg * 8 + (lane & 7)) * XLD + kt * 32 + ((lane >> 3) << 4));
                    float d[4] = {0.f, 0.f, 0.f, 0.f};
                    mma_m16n8k32_e4m3(d, a, bv);
                    acc[g][0] += d[0] * s;
                    acc[g][1] += d[1] * s;
                    acc[g][2] += d[2] * s;
                    acc[g][3] += d[3] * s;
                }
            }
            __syncthreads();
        }
    }
    if (wrow < m) {
        const int r0 = lane >> 2;
        const int c0 = (lane & 3) << 1;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const int tok = (g0 + g) * 8 + c0;
            if (tok < tt) {
                if (wrow + r0 < m) write_y(Y + static_cast<size_t>(t0 + tok) * m, wrow + r0, acc[g][0], add);
                if (wrow + r0 + 8 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok) * m, wrow + r0 + 8, acc[g][2], add);
            }
            if (tok + 1 < tt) {
                if (wrow + r0 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok + 1) * m, wrow + r0, acc[g][1], add);
                if (wrow + r0 + 8 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok + 1) * m, wrow + r0 + 8, acc[g][3], add);
            }
        }
    }
#else
    (void)W;
    (void)scale;
    (void)X;
    (void)Y;
    (void)m;
    (void)n;
    (void)T;
    (void)add;
    (void)Xe;
#endif
}

// Xe: two 128-col X tiles in smem, one sync per two K. WLD=512. Same 2D BN=128.
__global__ void __launch_bounds__(512, 1) gemm_fp8_kmaj_e4_bk512x_k(const uint8_t* W, const float* scale,
                                                                   const uint8_t* Xe, float* Y, int m, int n,
                                                                   int T, int add) {
#if __CUDA_ARCH__ >= 890
    constexpr int BM = 128, BN = 128, BK = 128, WLD = 512, XLD = 256, NG = 8;
    const int row_base = blockIdx.x * BM;
    const int t0 = blockIdx.y * BN;
    const int tt = T - t0 < BN ? T - t0 : BN;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wr = warp >> 1;
    const int wn = warp & 1;
    const int wrow = row_base + wr * 16;
    const int g0 = wn * NG;
    extern __shared__ char raw[];
    uint8_t* Wf = reinterpret_cast<uint8_t*>(raw);
    uint8_t* Xf = Wf + BM * WLD;
    float acc[NG][4];
#pragma unroll
    for (int g = 0; g < NG; ++g) acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
    const int nb_n = n >> 7;
    const int bi = row_base >> 7;
    for (int k0 = 0; k0 < n; k0 += 512) {
        const int cb0 = k0 >> 7;
        if (wn == 0) {
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                const int row = wrow + r;
                const uint4 p = (row < m) ? fp8_ld16(W, row, m, cb0, lane) : make_uint4(0, 0, 0, 0);
                uint8_t* dst = Wf + (wr * 16 + r) * WLD + (lane << 2);
                *reinterpret_cast<uint32_t*>(dst + 0) = p.x;
                *reinterpret_cast<uint32_t*>(dst + 128) = p.y;
                *reinterpret_cast<uint32_t*>(dst + 256) = p.z;
                *reinterpret_cast<uint32_t*>(dst + 384) = p.w;
            }
        }
#pragma unroll
        for (int kp = 0; kp < 4; kp += 2) {
            const int kk = k0 + kp * BK;
            if (kk >= n) break;
            const int nkb = (kk + BK < n) ? 2 : 1;
            for (int i = threadIdx.x * 16; i < tt * BK * nkb; i += 8192) {
                const int t = i / (BK * nkb);
                const int c = i - t * BK * nkb;
                *reinterpret_cast<uint4*>(Xf + t * XLD + c) =
                    *reinterpret_cast<const uint4*>(Xe + static_cast<size_t>(t0 + t) * n + kk + c);
            }
            if (tt < BN) {
                for (int t = tt; t < BN; ++t)
                    for (int c = threadIdx.x; c < BK * nkb; c += 512) Xf[t * XLD + c] = 0;
            }
            __syncthreads();
#pragma unroll
            for (int kb = 0; kb < 2; ++kb) {
                if (kb >= nkb) break;
                const float s = (row_base < m && scale) ? scale[bi * nb_n + cb0 + kp + kb] : 0.f;
#pragma unroll
                for (int kt = 0; kt < 4; ++kt) {
                    uint32_t a[4];
                    ldmatrix_x4(a, Wf + (wr * 16 + (lane & 15)) * WLD + (kp + kb) * 128 + kt * 32 +
                                       ((lane >> 4) << 4));
#pragma unroll
                    for (int g = 0; g < NG; ++g) {
                        const int gg = g0 + g;
                        if (gg * 8 >= tt) continue;
                        uint32_t bv[2];
                        ldmatrix_x2(bv, Xf + (gg * 8 + (lane & 7)) * XLD + kb * 128 + kt * 32 +
                                            ((lane >> 3) << 4));
                        float d[4] = {0.f, 0.f, 0.f, 0.f};
                        mma_m16n8k32_e4m3(d, a, bv);
                        acc[g][0] += d[0] * s;
                        acc[g][1] += d[1] * s;
                        acc[g][2] += d[2] * s;
                        acc[g][3] += d[3] * s;
                    }
                }
            }
            __syncthreads();
        }
    }
    if (wrow < m) {
        const int r0 = lane >> 2;
        const int c0 = (lane & 3) << 1;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const int tok = (g0 + g) * 8 + c0;
            if (tok < tt) {
                if (wrow + r0 < m) write_y(Y + static_cast<size_t>(t0 + tok) * m, wrow + r0, acc[g][0], add);
                if (wrow + r0 + 8 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok) * m, wrow + r0 + 8, acc[g][2], add);
            }
            if (tok + 1 < tt) {
                if (wrow + r0 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok + 1) * m, wrow + r0, acc[g][1], add);
                if (wrow + r0 + 8 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok + 1) * m, wrow + r0 + 8, acc[g][3], add);
            }
        }
    }
#else
    (void)W;
    (void)scale;
    (void)Xe;
    (void)Y;
    (void)m;
    (void)n;
    (void)T;
    (void)add;
#endif
}

// Occupancy 2: 512 threads, 48KB (W 128x256 + X 128x128). uint4 load, two
// K-blocks in smem. Doubles threads/SM vs bk512. Not 1D T-scan / BN=256@T=256.
__global__ void __launch_bounds__(512, 2) gemm_fp8_kmaj_e4_o2_k(const uint8_t* W, const float* scale,
                                                               const float* X, float* Y, int m, int n, int T,
                                                               int add, const uint8_t* Xe) {
#if __CUDA_ARCH__ >= 890
    constexpr int BM = 128, BN = 128, BK = 128, WLD = 256, XLD = 128, NG = 8;
    const int row_base = blockIdx.x * BM;
    const int t0 = blockIdx.y * BN;
    const int tt = T - t0 < BN ? T - t0 : BN;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wr = warp >> 1;
    const int wn = warp & 1;
    const int wrow = row_base + wr * 16;
    const int g0 = wn * NG;
    extern __shared__ char raw[];
    uint8_t* Wf = reinterpret_cast<uint8_t*>(raw);
    uint8_t* Xf = Wf + BM * WLD;
    float acc[NG][4];
#pragma unroll
    for (int g = 0; g < NG; ++g) acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
    const int nb_n = n >> 7;
    const int bi = row_base >> 7;
    for (int k0 = 0; k0 < n; k0 += 256) {
        const int cb0 = (k0 >> 7) & ~3;
        const int pair = (k0 >> 7) & 2;
        if (wn == 0) {
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                const int row = wrow + r;
                const uint4 p = (row < m) ? fp8_ld16(W, row, m, cb0, lane) : make_uint4(0, 0, 0, 0);
                uint8_t* dst = Wf + (wr * 16 + r) * WLD + (lane << 2);
                *reinterpret_cast<uint32_t*>(dst + 0) = pair ? p.z : p.x;
                *reinterpret_cast<uint32_t*>(dst + 128) = pair ? p.w : p.y;
            }
        }
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
            const int kk = k0 + kb * BK;
            if (kk >= n) break;
            const float s = (row_base < m && scale) ? scale[bi * nb_n + (k0 >> 7) + kb] : 0.f;
            if (Xe) {
                for (int i = threadIdx.x * 16; i < tt * BK; i += 8192) {
                    const int t = i >> 7;
                    const int c = i & 127;
                    *reinterpret_cast<uint4*>(Xf + t * XLD + c) =
                        *reinterpret_cast<const uint4*>(Xe + static_cast<size_t>(t0 + t) * n + kk + c);
                }
            } else {
                for (int i = threadIdx.x * 4; i < tt * BK; i += 2048) {
                    const int t = i >> 7;
                    const int c = i & 127;
                    const float4 v =
                        *reinterpret_cast<const float4*>(X + static_cast<size_t>(t0 + t) * n + kk + c);
                    Xf[t * XLD + c + 0] = f32_to_fp8e4(v.x);
                    Xf[t * XLD + c + 1] = f32_to_fp8e4(v.y);
                    Xf[t * XLD + c + 2] = f32_to_fp8e4(v.z);
                    Xf[t * XLD + c + 3] = f32_to_fp8e4(v.w);
                }
            }
            if (tt < BN) {
                for (int t = tt; t < BN; ++t)
                    for (int c = threadIdx.x; c < BK; c += 512) Xf[t * XLD + c] = 0;
            }
            __syncthreads();
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                uint32_t a[4];
                ldmatrix_x4(a, Wf + (wr * 16 + (lane & 15)) * WLD + kb * 128 + kt * 32 + ((lane >> 4) << 4));
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    const int gg = g0 + g;
                    if (gg * 8 >= tt) continue;
                    uint32_t bv[2];
                    ldmatrix_x2(bv, Xf + (gg * 8 + (lane & 7)) * XLD + kt * 32 + ((lane >> 3) << 4));
                    float d[4] = {0.f, 0.f, 0.f, 0.f};
                    mma_m16n8k32_e4m3(d, a, bv);
                    acc[g][0] += d[0] * s;
                    acc[g][1] += d[1] * s;
                    acc[g][2] += d[2] * s;
                    acc[g][3] += d[3] * s;
                }
            }
            __syncthreads();
        }
    }
    if (wrow < m) {
        const int r0 = lane >> 2;
        const int c0 = (lane & 3) << 1;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const int tok = (g0 + g) * 8 + c0;
            if (tok < tt) {
                if (wrow + r0 < m) write_y(Y + static_cast<size_t>(t0 + tok) * m, wrow + r0, acc[g][0], add);
                if (wrow + r0 + 8 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok) * m, wrow + r0 + 8, acc[g][2], add);
            }
            if (tok + 1 < tt) {
                if (wrow + r0 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok + 1) * m, wrow + r0, acc[g][1], add);
                if (wrow + r0 + 8 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok + 1) * m, wrow + r0 + 8, acc[g][3], add);
            }
        }
    }
#else
    (void)W;
    (void)scale;
    (void)X;
    (void)Y;
    (void)m;
    (void)n;
    (void)T;
    (void)add;
    (void)Xe;
#endif
}

// Higher occupancy: BM=64 BN=64, 256 threads, uint4 512-col W (~42KB, 2/SM).
// 2D T-split. Not 1D T-scan / T=256 BN=256.
__global__ void __launch_bounds__(256, 2) gemm_fp8_kmaj_e4_occ_k(const uint8_t* W, const float* scale,
                                                                const float* X, float* Y, int m, int n, int T,
                                                                int add, const uint8_t* Xe) {
#if __CUDA_ARCH__ >= 890
    constexpr int BM = 64, BN = 64, BK = 128, WLD = 528, XLD = 144, NG = 4;
    const int row_base = blockIdx.x * BM;
    const int t0 = blockIdx.y * BN;
    const int tt = T - t0 < BN ? T - t0 : BN;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wr = warp >> 1;
    const int wn = warp & 1;
    const int wrow = row_base + wr * 16;
    const int g0 = wn * NG;
    extern __shared__ char raw[];
    uint8_t* Wf = reinterpret_cast<uint8_t*>(raw);
    uint8_t* Xf = Wf + BM * WLD;
    float acc[NG][4];
#pragma unroll
    for (int g = 0; g < NG; ++g) acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
    const int nb_n = n >> 7;
    const int bi = row_base >> 7;
    for (int k0 = 0; k0 < n; k0 += 512) {
        const int cb0 = k0 >> 7;
        if (wn == 0) {
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                const int row = wrow + r;
                const uint4 p = (row < m) ? fp8_ld16(W, row, m, cb0, lane) : make_uint4(0, 0, 0, 0);
                uint8_t* dst = Wf + (wr * 16 + r) * WLD + (lane << 2);
                *reinterpret_cast<uint32_t*>(dst + 0) = p.x;
                *reinterpret_cast<uint32_t*>(dst + 128) = p.y;
                *reinterpret_cast<uint32_t*>(dst + 256) = p.z;
                *reinterpret_cast<uint32_t*>(dst + 384) = p.w;
            }
        }
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
            const int kk = k0 + kb * BK;
            if (kk >= n) break;
            const float s = (row_base < m && scale) ? scale[bi * nb_n + cb0 + kb] : 0.f;
            if (Xe) {
                for (int i = threadIdx.x * 16; i < tt * BK; i += 4096) {
                    const int t = i >> 7;
                    const int c = i & 127;
                    *reinterpret_cast<uint4*>(Xf + t * XLD + c) =
                        *reinterpret_cast<const uint4*>(Xe + static_cast<size_t>(t0 + t) * n + kk + c);
                }
            } else {
                for (int i = threadIdx.x * 4; i < tt * BK; i += 1024) {
                    const int t = i >> 7;
                    const int c = i & 127;
                    const float4 v =
                        *reinterpret_cast<const float4*>(X + static_cast<size_t>(t0 + t) * n + kk + c);
                    Xf[t * XLD + c + 0] = f32_to_fp8e4(v.x);
                    Xf[t * XLD + c + 1] = f32_to_fp8e4(v.y);
                    Xf[t * XLD + c + 2] = f32_to_fp8e4(v.z);
                    Xf[t * XLD + c + 3] = f32_to_fp8e4(v.w);
                }
            }
            if (tt < BN) {
                for (int t = tt; t < BN; ++t)
                    for (int c = threadIdx.x; c < BK; c += 256) Xf[t * XLD + c] = 0;
            }
            __syncthreads();
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                uint32_t a[4];
                ldmatrix_x4(a, Wf + (wr * 16 + (lane & 15)) * WLD + kb * 128 + kt * 32 + ((lane >> 4) << 4));
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    const int gg = g0 + g;
                    if (gg * 8 >= tt) continue;
                    uint32_t bv[2];
                    ldmatrix_x2(bv, Xf + (gg * 8 + (lane & 7)) * XLD + kt * 32 + ((lane >> 3) << 4));
                    float d[4] = {0.f, 0.f, 0.f, 0.f};
                    mma_m16n8k32_e4m3(d, a, bv);
                    acc[g][0] += d[0] * s;
                    acc[g][1] += d[1] * s;
                    acc[g][2] += d[2] * s;
                    acc[g][3] += d[3] * s;
                }
            }
            __syncthreads();
        }
    }
    if (wrow < m) {
        const int r0 = lane >> 2;
        const int c0 = (lane & 3) << 1;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const int tok = (g0 + g) * 8 + c0;
            if (tok < tt) {
                if (wrow + r0 < m) write_y(Y + static_cast<size_t>(t0 + tok) * m, wrow + r0, acc[g][0], add);
                if (wrow + r0 + 8 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok) * m, wrow + r0 + 8, acc[g][2], add);
            }
            if (tok + 1 < tt) {
                if (wrow + r0 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok + 1) * m, wrow + r0, acc[g][1], add);
                if (wrow + r0 + 8 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok + 1) * m, wrow + r0 + 8, acc[g][3], add);
            }
        }
    }
#else
    (void)W;
    (void)scale;
    (void)X;
    (void)Y;
    (void)m;
    (void)n;
    (void)T;
    (void)add;
    (void)Xe;
#endif
}

// Xe path: WLD=512 + two X panels. cp.async next X while MMA current.
// Not a 1D T-scan; still 2D BN=128.
__global__ void __launch_bounds__(512, 1) gemm_fp8_kmaj_e4_bk512p_k(const uint8_t* W, const float* scale,
                                                                   const uint8_t* Xe, float* Y, int m, int n,
                                                                   int T, int add) {
#if __CUDA_ARCH__ >= 890
    constexpr int BM = 128, BN = 128, BK = 128, WLD = 512, XLD = 128, NG = 8;
    const int row_base = blockIdx.x * BM;
    const int t0 = blockIdx.y * BN;
    const int tt = T - t0 < BN ? T - t0 : BN;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wr = warp >> 1;
    const int wn = warp & 1;
    const int wrow = row_base + wr * 16;
    const int g0 = wn * NG;
    extern __shared__ char raw[];
    uint8_t* Wf = reinterpret_cast<uint8_t*>(raw);
    uint8_t* Xf0 = Wf + BM * WLD;
    uint8_t* Xf1 = Xf0 + BN * XLD;
    float acc[NG][4];
#pragma unroll
    for (int g = 0; g < NG; ++g) acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
    const int nb_n = n >> 7;
    const int bi = row_base >> 7;
    auto cp_x = [&](uint8_t* Xf, int kk) {
        for (int i = threadIdx.x * 16; i < tt * BK; i += 8192) {
            const int t = i >> 7;
            const int c = i & 127;
            unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(Xf + t * XLD + c));
            const void* src = Xe + static_cast<size_t>(t0 + t) * n + kk + c;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
        }
        asm volatile("cp.async.commit_group;");
    };
    for (int k0 = 0; k0 < n; k0 += 512) {
        const int cb0 = k0 >> 7;
        if (wn == 0) {
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                const int row = wrow + r;
                const uint4 p = (row < m) ? fp8_ld16(W, row, m, cb0, lane) : make_uint4(0, 0, 0, 0);
                uint8_t* dst = Wf + (wr * 16 + r) * WLD + (lane << 2);
                *reinterpret_cast<uint32_t*>(dst + 0) = p.x;
                *reinterpret_cast<uint32_t*>(dst + 128) = p.y;
                *reinterpret_cast<uint32_t*>(dst + 256) = p.z;
                *reinterpret_cast<uint32_t*>(dst + 384) = p.w;
            }
        }
        cp_x(Xf0, k0);
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
            const int kk = k0 + kb * BK;
            if (kk >= n) break;
            uint8_t* Xcur = (kb & 1) ? Xf1 : Xf0;
            uint8_t* Xnxt = (kb & 1) ? Xf0 : Xf1;
            if (kb + 1 < 4 && kk + BK < n) cp_x(Xnxt, kk + BK);
            const float s = (row_base < m && scale) ? scale[bi * nb_n + cb0 + kb] : 0.f;
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                uint32_t a[4];
                ldmatrix_x4(a, Wf + (wr * 16 + (lane & 15)) * WLD + kb * 128 + kt * 32 + ((lane >> 4) << 4));
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    const int gg = g0 + g;
                    if (gg * 8 >= tt) continue;
                    uint32_t bv[2];
                    ldmatrix_x2(bv, Xcur + (gg * 8 + (lane & 7)) * XLD + kt * 32 + ((lane >> 3) << 4));
                    float d[4] = {0.f, 0.f, 0.f, 0.f};
                    mma_m16n8k32_e4m3(d, a, bv);
                    acc[g][0] += d[0] * s;
                    acc[g][1] += d[1] * s;
                    acc[g][2] += d[2] * s;
                    acc[g][3] += d[3] * s;
                }
            }
            if (kb + 1 < 4 && kk + BK < n) {
                asm volatile("cp.async.wait_group 0;");
                __syncthreads();
            }
        }
        __syncthreads();
    }
    if (wrow < m) {
        const int r0 = lane >> 2;
        const int c0 = (lane & 3) << 1;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const int tok = (g0 + g) * 8 + c0;
            if (tok < tt) {
                if (wrow + r0 < m) write_y(Y + static_cast<size_t>(t0 + tok) * m, wrow + r0, acc[g][0], add);
                if (wrow + r0 + 8 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok) * m, wrow + r0 + 8, acc[g][2], add);
            }
            if (tok + 1 < tt) {
                if (wrow + r0 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok + 1) * m, wrow + r0, acc[g][1], add);
                if (wrow + r0 + 8 < m)
                    write_y(Y + static_cast<size_t>(t0 + tok + 1) * m, wrow + r0 + 8, acc[g][3], add);
            }
        }
    }
#else
    (void)W;
    (void)scale;
    (void)Xe;
    (void)Y;
    (void)m;
    (void)n;
    (void)T;
    (void)add;
#endif
}

// Same X, two k-major FP8 weights. One X quant per K-tile, two W reshuffles.
template <int BN>
__global__ void __launch_bounds__(256, 3) gemm_fp8_kmaj_e4_dual_k(
    const uint8_t* W1, const float* S1, const uint8_t* W2, const float* S2, const float* X, float* Y1,
    float* Y2, int m1, int m2, int n, int T) {
#if __CUDA_ARCH__ >= 890
    constexpr int BM = 128, BK = 128, WLD = 144, XLD = 144, NG = BN / 8;
    const int row_base = blockIdx.x * BM;
    const int t0 = blockIdx.y * BN;
    const int tt = T - t0 < BN ? T - t0 : BN;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int m = m1 > m2 ? m1 : m2;
    extern __shared__ char raw[];
    uint8_t* Wf1 = reinterpret_cast<uint8_t*>(raw);
    uint8_t* Wf2 = Wf1 + BM * WLD;
    uint8_t* Xf = Wf2 + BM * WLD;
    float a1[NG][4], a2[NG][4];
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        a1[g][0] = a1[g][1] = a1[g][2] = a1[g][3] = 0.f;
        a2[g][0] = a2[g][1] = a2[g][2] = a2[g][3] = 0.f;
    }
    const int nb_n = n >> 7;
    const int bi = row_base >> 7;
    for (int k0 = 0; k0 < n; k0 += BK) {
        const int cb = k0 >> 7;
        const float s1 = (row_base < m1 && S1) ? S1[bi * nb_n + cb] : 0.f;
        const float s2 = (row_base < m2 && S2) ? S2[bi * nb_n + cb] : 0.f;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const int row = row_base + warp * 16 + r;
            const uint32_t p1 = (row < m1) ? *fp8_blk4(W1, row, m1, cb, lane) : 0u;
            const uint32_t p2 = (row < m2) ? *fp8_blk4(W2, row, m2, cb, lane) : 0u;
            *reinterpret_cast<uint32_t*>(Wf1 + (warp * 16 + r) * WLD + (lane << 2)) = p1;
            *reinterpret_cast<uint32_t*>(Wf2 + (warp * 16 + r) * WLD + (lane << 2)) = p2;
        }
        for (int i = threadIdx.x * 4; i < tt * BK; i += 1024) {
            const int t = i >> 7;
            const int c = i & 127;
            const float4 v = *reinterpret_cast<const float4*>(X + static_cast<size_t>(t0 + t) * n + k0 + c);
            Xf[t * XLD + c + 0] = f32_to_fp8e4(v.x);
            Xf[t * XLD + c + 1] = f32_to_fp8e4(v.y);
            Xf[t * XLD + c + 2] = f32_to_fp8e4(v.z);
            Xf[t * XLD + c + 3] = f32_to_fp8e4(v.w);
        }
        for (int t = tt; t < BN; ++t) {
            for (int c = threadIdx.x; c < BK; c += 256) Xf[t * XLD + c] = 0;
        }
        __syncthreads();
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint32_t A1[4], A2[4];
            ldmatrix_x4(A1, Wf1 + (warp * 16 + (lane & 15)) * WLD + kt * 32 + ((lane >> 4) << 4));
            ldmatrix_x4(A2, Wf2 + (warp * 16 + (lane & 15)) * WLD + kt * 32 + ((lane >> 4) << 4));
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                if (g * 8 >= tt) continue;
                uint32_t bv[2];
                ldmatrix_x2(bv, Xf + (g * 8 + (lane & 7)) * XLD + kt * 32 + ((lane >> 3) << 4));
                float d1[4] = {0.f, 0.f, 0.f, 0.f}, d2[4] = {0.f, 0.f, 0.f, 0.f};
                mma_m16n8k32_e4m3(d1, A1, bv);
                mma_m16n8k32_e4m3(d2, A2, bv);
                a1[g][0] += d1[0] * s1;
                a1[g][1] += d1[1] * s1;
                a1[g][2] += d1[2] * s1;
                a1[g][3] += d1[3] * s1;
                a2[g][0] += d2[0] * s2;
                a2[g][1] += d2[1] * s2;
                a2[g][2] += d2[2] * s2;
                a2[g][3] += d2[3] * s2;
            }
        }
        __syncthreads();
    }
    const int r0 = lane >> 2;
    const int c0 = (lane & 3) << 1;
    const int wrow = row_base + warp * 16;
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        const int tok = g * 8 + c0;
        if (tok < tt) {
            if (wrow + r0 < m1) Y1[static_cast<size_t>(t0 + tok) * m1 + wrow + r0] = a1[g][0];
            if (wrow + r0 + 8 < m1) Y1[static_cast<size_t>(t0 + tok) * m1 + wrow + r0 + 8] = a1[g][2];
            if (wrow + r0 < m2) Y2[static_cast<size_t>(t0 + tok) * m2 + wrow + r0] = a2[g][0];
            if (wrow + r0 + 8 < m2) Y2[static_cast<size_t>(t0 + tok) * m2 + wrow + r0 + 8] = a2[g][2];
        }
        if (tok + 1 < tt) {
            if (wrow + r0 < m1) Y1[static_cast<size_t>(t0 + tok + 1) * m1 + wrow + r0] = a1[g][1];
            if (wrow + r0 + 8 < m1) Y1[static_cast<size_t>(t0 + tok + 1) * m1 + wrow + r0 + 8] = a1[g][3];
            if (wrow + r0 < m2) Y2[static_cast<size_t>(t0 + tok + 1) * m2 + wrow + r0] = a2[g][1];
            if (wrow + r0 + 8 < m2) Y2[static_cast<size_t>(t0 + tok + 1) * m2 + wrow + r0 + 8] = a2[g][3];
        }
    }
    (void)m;
#else
    (void)W1;
    (void)S1;
    (void)W2;
    (void)S2;
    (void)X;
    (void)Y1;
    (void)Y2;
    (void)m1;
    (void)m2;
    (void)n;
    (void)T;
#endif
}

// K-outer / T-inner: one W tile stays in smem while all tokens of this
// TN-panel are accumulated into Ys. T=256 → 2 W passes instead of 4.
template <int TN>
__global__ void __launch_bounds__(128, 2) gemm_fp8_e4_yst_k(const uint8_t* W, const float* scale,
                                                           const float* X, float* Y, int m, int n, int T,
                                                           int add) {
#if __CUDA_ARCH__ >= 890
    constexpr int BM = 64, BK = 128, WLD = 144, XLD = 144, NG = 8;
    const int row_base = blockIdx.x * BM;
    const int tbase = blockIdx.y * TN;
    const int tt = T - tbase < TN ? T - tbase : TN;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= 4) return;
    extern __shared__ char raw[];
    float* Ys = reinterpret_cast<float*>(raw);
    uint8_t* Wf = reinterpret_cast<uint8_t*>(Ys + TN * BM);
    uint8_t* Xf = Wf + BM * WLD;
    for (int i = threadIdx.x; i < tt * BM; i += 128) Ys[i] = 0.f;
    __syncthreads();
    const int nb_n = n >> 7;
    const int bi = row_base >> 7;
    const int wrow = row_base + warp * 16;
    for (int k0 = 0; k0 < n; k0 += BK) {
        const int cb = k0 >> 7;
        const float s = (row_base < m && scale) ? scale[bi * nb_n + cb] : 0.f;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const int row = wrow + r;
            const uint32_t p = (row < m) ? *fp8_blk4(W, row, m, cb, lane) : 0u;
            *reinterpret_cast<uint32_t*>(Wf + (warp * 16 + r) * WLD + (lane << 2)) = p;
        }
        for (int t0 = 0; t0 < tt; t0 += 64) {
            const int ntok = tt - t0 < 64 ? tt - t0 : 64;
            for (int i = threadIdx.x * 4; i < ntok * BK; i += 512) {
                const int t = i >> 7;
                const int c = i & 127;
                const float4 v =
                    *reinterpret_cast<const float4*>(X + static_cast<size_t>(tbase + t0 + t) * n + k0 + c);
                Xf[t * XLD + c + 0] = f32_to_fp8e4(v.x);
                Xf[t * XLD + c + 1] = f32_to_fp8e4(v.y);
                Xf[t * XLD + c + 2] = f32_to_fp8e4(v.z);
                Xf[t * XLD + c + 3] = f32_to_fp8e4(v.w);
            }
            for (int t = ntok; t < 64; ++t) {
                for (int c = threadIdx.x; c < BK; c += 128) Xf[t * XLD + c] = 0;
            }
            __syncthreads();
            float acc[NG][4];
#pragma unroll
            for (int g = 0; g < NG; ++g) acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                uint32_t a[4];
                ldmatrix_x4(a, Wf + (warp * 16 + (lane & 15)) * WLD + kt * 32 + ((lane >> 4) << 4));
#pragma unroll
                for (int g = 0; g < NG; ++g) {
                    if (g * 8 >= ntok) continue;
                    uint32_t bv[2];
                    ldmatrix_x2(bv, Xf + (g * 8 + (lane & 7)) * XLD + kt * 32 + ((lane >> 3) << 4));
                    mma_m16n8k32_e4m3(acc[g], a, bv);
                }
            }
            const int r0 = lane >> 2;
            const int c0 = (lane & 3) << 1;
#pragma unroll
            for (int g = 0; g < NG; ++g) {
                const int tok = g * 8 + c0;
                if (tok < ntok) {
                    if (wrow + r0 < m) Ys[(t0 + tok) * BM + warp * 16 + r0] += acc[g][0] * s;
                    if (wrow + r0 + 8 < m) Ys[(t0 + tok) * BM + warp * 16 + r0 + 8] += acc[g][2] * s;
                }
                if (tok + 1 < ntok) {
                    if (wrow + r0 < m) Ys[(t0 + tok + 1) * BM + warp * 16 + r0] += acc[g][1] * s;
                    if (wrow + r0 + 8 < m) Ys[(t0 + tok + 1) * BM + warp * 16 + r0 + 8] += acc[g][3] * s;
                }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < tt * BM; i += 128) {
        const int t = i / BM;
        const int r = i - t * BM;
        const int row = row_base + r;
        if (row < m) write_y(Y + static_cast<size_t>(tbase + t) * m, row, Ys[i], add);
    }
#else
    (void)W;
    (void)scale;
    (void)X;
    (void)Y;
    (void)m;
    (void)n;
    (void)T;
    (void)add;
#endif
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

__global__ void kv_h2f_n_k(const __half* x, float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(x[i]);
}

__global__ void store_kv_batch_h_k(__half* cache, const float* src, int pos0, int T, int kn) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T && i < kn)
        cache[static_cast<size_t>(pos0 + t) * kn + i] = __float2half(src[static_cast<size_t>(t) * kn + i]);
}

// One block per (token, KV head). 256 threads cover hd=256.
__global__ void store_k_q8_th_k(int8_t* qs, __half* sc, const float* src, int pos0, int T, int nkv, int hd) {
    const int t = blockIdx.y;
    const int h = blockIdx.x;
    if (t >= T || h >= nkv) return;
    const float* v = src + (static_cast<size_t>(t) * nkv + h) * hd;
    float amax = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) amax = fmaxf(amax, fabsf(v[i]));
    amax = warp_max_all(amax);
    __shared__ float smx[8];
    if ((threadIdx.x & 31) == 0) smx[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = smx[0];
        const int nw = (blockDim.x + 31) >> 5;
        for (int w = 1; w < nw && w < 8; ++w) m = fmaxf(m, smx[w]);
        smx[0] = m / 127.f + 1e-8f;
        sc[static_cast<size_t>(pos0 + t) * nkv + h] = __float2half(smx[0]);
    }
    __syncthreads();
    const float s = smx[0];
    int8_t* q = qs + (static_cast<size_t>(pos0 + t) * nkv + h) * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        const float u = fminf(fmaxf(v[i] / s, -127.f), 127.f);
        q[i] = static_cast<int8_t>(lrintf(u));
    }
}

// grid (nblk, nkv, T), 128 threads. nblk = hd/128.
__global__ void store_v_tq3_th_k(uint8_t* qs, __half* sc, const float* src, int pos0, int T, int nkv, int hd) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int t = blockIdx.z;
    const int nblk = hd / kTqBlk;
    if (t >= T || h >= nkv || b >= nblk) return;
    const int tid = threadIdx.x;
    if (tid >= kTqBlk) return;
    extern __shared__ float sm[];
    float* x = sm;
    const float* v = src + (static_cast<size_t>(t) * nkv + h) * hd + b * kTqBlk;
    x[tid] = v[tid];
    __syncthreads();
    fwht128_sync(x, tid);
    float ss = x[tid] * x[tid];
    ss = warp_sum_all(ss);
    __shared__ float srms;
    __shared__ float wss[4];
    if ((tid & 31) == 0) wss[tid >> 5] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = wss[0] + wss[1] + wss[2] + wss[3];
        srms = sqrtf(tot / static_cast<float>(kTqBlk)) + 1e-8f;
        sc[(static_cast<size_t>(pos0 + t) * nkv + h) * nblk + b] = __float2half(srms);
    }
    __syncthreads();
    const float inv = 1.f / srms;
    __shared__ int idx[kTqBlk];
    idx[tid] = tq3_nn(x[tid] * inv);
    __syncthreads();
    uint8_t* dst = qs + ((static_cast<size_t>(pos0 + t) * nkv + h) * nblk + b) * kTq3B;
    pack_tq3_16(dst, idx, tid);
}

__global__ void dequant_k_q8_h_k(__half* dst, const int8_t* qs, const __half* sc, int pos0, int T, int nkv,
                                 int hd) {
    const int t = blockIdx.y;
    const int h = blockIdx.x;
    if (t >= T || h >= nkv) return;
    const float s = __half2float(sc[static_cast<size_t>(pos0 + t) * nkv + h]);
    const int8_t* q = qs + (static_cast<size_t>(pos0 + t) * nkv + h) * hd;
    __half* o = dst + (static_cast<size_t>(t) * nkv + h) * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) o[i] = __float2half(s * static_cast<float>(q[i]));
}

__global__ void dequant_v_tq3_h_k(__half* dst, const uint8_t* qs, const __half* sc, int pos0, int T, int nkv,
                                  int hd) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int t = blockIdx.z;
    const int nblk = hd / kTqBlk;
    if (t >= T || h >= nkv || b >= nblk) return;
    const int tid = threadIdx.x;
    if (tid >= kTqBlk) return;
    extern __shared__ float sm[];
    float* x = sm;
    const float scale = __half2float(sc[(static_cast<size_t>(pos0 + t) * nkv + h) * nblk + b]);
    const uint8_t* src = qs + ((static_cast<size_t>(pos0 + t) * nkv + h) * nblk + b) * kTq3B;
    unpack_tq3_tid(src, tid, scale, x);
    __syncthreads();
    fwht128_sync(x, tid);
    dst[(static_cast<size_t>(t) * nkv + h) * hd + b * kTqBlk + tid] = __float2half(x[tid]);
}

// Prefill attend from compact q8-K / tq3-V. Used when tend > F16 window.
__global__ void attn_prefill_tq_k(const float* q, float* o, int pos0, int T, int n_q, int n_kv, int hd,
                                  const int8_t* k_q8, const __half* k_sc, const uint8_t* v_qs,
                                  const __half* v_sc) {
    const int hq = blockIdx.x;
    const int tloc = blockIdx.y;
    if (hq >= n_q || tloc >= T || hd != 256 || !k_q8 || !k_sc || !v_qs || !v_sc || n_kv <= 0) return;
    const int hkv = hq / (n_q / n_kv);
    const int Tend = pos0 + tloc + 1;
    const int tid = threadIdx.x;
    const float qv = tid < hd ? q[(static_cast<size_t>(tloc) * n_q + hq) * hd + tid] : 0.f;
    const float scale = rsqrtf(static_cast<float>(hd));
    extern __shared__ float khloc[];
    __shared__ float wss[8];
    float acc = 0.f;
    for (int t = 0; t < Tend; ++t) {
        float dot = 0.f;
        const float ks = __half2float(k_sc[static_cast<size_t>(t) * n_kv + hkv]);
        const int8_t* kq = k_q8 + (static_cast<size_t>(t) * n_kv + hkv) * hd;
        if (tid < hd) dot = qv * (ks * static_cast<float>(kq[tid]));
        dot = warp_sum(dot);
        if ((tid & 31) == 0) wss[tid >> 5] = dot;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.f;
            const int nw = blockDim.x >> 5;
            for (int w = 0; w < nw && w < 8; ++w) tot += wss[w];
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
        const int nblk = hd / kTqBlk;
        float vv = 0.f;
        for (int bb = 0; bb < nblk; ++bb) {
            const float vscl = __half2float(v_sc[(static_cast<size_t>(t) * n_kv + hkv) * nblk + bb]);
            const uint8_t* src = v_qs + ((static_cast<size_t>(t) * n_kv + hkv) * nblk + bb) * kTq3B;
            if (tid < kTqBlk) unpack_tq3_tid(src, tid, vscl, khloc);
            __syncthreads();
            fwht128_sync(khloc, tid);
            if (tid >= bb * kTqBlk && tid < (bb + 1) * kTqBlk) vv = khloc[tid - bb * kTqBlk];
            __syncthreads();
        }
        if (tid < hd) acc = acc * aa + pp * vv;
        __syncthreads();
    }
    if (tid < hd) {
        const float den = wss[1] > 0.f ? wss[1] : 1.f;
        o[(static_cast<size_t>(tloc) * n_q + hq) * hd + tid] = acc / den;
    }
}

// One block per KV head: dequant tq3-V once, attend all Q heads in the GQA group.
__global__ void qk_attn_decode_tq_gqa_k(const float* q, float* o, const int* pos, int n_q, int n_kv, int hd,
                                        const int8_t* k_q8, const __half* k_sc, const uint8_t* v_qs,
                                        const __half* v_sc) {
    const int hkv = blockIdx.x;
    if (hkv >= n_kv || hd != 256 || n_kv <= 0 || !pos || !k_q8 || !k_sc || !v_qs || !v_sc) return;
    const int Tend = *pos + 1;
    const int rep = n_q / n_kv;
    if (rep < 1 || rep > 8) return;
    const int tid = threadIdx.x;
    const int hq0 = hkv * rep;
    const float scale = rsqrtf(256.f);
    __shared__ float smV[256];
    float qv[8], acc[8];
#pragma unroll
    for (int qi = 0; qi < 8; ++qi) {
        acc[qi] = 0.f;
        qv[qi] = (qi < rep && tid < hd) ? q[(hq0 + qi) * hd + tid] : 0.f;
    }
    __shared__ float red[8][8];
    __shared__ float smi[8], sli[8], saa[8], spp[8];
    if (tid < 8) {
        smi[tid] = -1e30f;
        sli[tid] = 0.f;
    }
    __syncthreads();
    for (int t = 0; t < Tend; ++t) {
        dequant_v_tq3_256(smV, v_qs, v_sc, static_cast<size_t>(t) * n_kv + hkv, tid);
        const float ks = __half2float(k_sc[static_cast<size_t>(t) * n_kv + hkv]);
        const int8_t* kq = k_q8 + (static_cast<size_t>(t) * n_kv + hkv) * hd;
        const float kv = (tid < hd) ? ks * static_cast<float>(kq[tid]) : 0.f;
        float dots[8];
#pragma unroll
        for (int qi = 0; qi < 8; ++qi) {
            float d = (qi < rep) ? qv[qi] * kv : 0.f;
            dots[qi] = warp_sum(d);
        }
        if ((tid & 31) == 0) {
#pragma unroll
            for (int qi = 0; qi < 8; ++qi) red[tid >> 5][qi] = dots[qi];
        }
        __syncthreads();
        if (tid == 0) {
#pragma unroll
            for (int qi = 0; qi < 8; ++qi) {
                if (qi >= rep) continue;
                float tot = 0.f;
#pragma unroll
                for (int w = 0; w < 8; ++w) tot += red[w][qi];
                const float s = tot * scale;
                const float m2 = fmaxf(smi[qi], s);
                saa[qi] = __expf(smi[qi] - m2);
                spp[qi] = __expf(s - m2);
                sli[qi] = sli[qi] * saa[qi] + spp[qi];
                smi[qi] = m2;
            }
        }
        __syncthreads();
        const float vv = smV[tid];
#pragma unroll
        for (int qi = 0; qi < 8; ++qi) {
            if (qi < rep) acc[qi] = acc[qi] * saa[qi] + spp[qi] * vv;
        }
        __syncthreads();
    }
    if (tid < hd) {
#pragma unroll
        for (int qi = 0; qi < 8; ++qi) {
            if (qi >= rep) break;
            const float den = sli[qi] > 0.f ? sli[qi] : 1.f;
            o[(hq0 + qi) * hd + tid] = acc[qi] / den;
        }
    }
}

__global__ void attn_prefill_tq_gqa_k(const float* q, float* o, int pos0, int T, int n_q, int n_kv, int hd,
                                      const int8_t* k_q8, const __half* k_sc, const uint8_t* v_qs,
                                      const __half* v_sc) {
    const int hkv = blockIdx.x;
    const int tloc = blockIdx.y;
    if (hkv >= n_kv || tloc >= T || hd != 256 || n_kv <= 0 || !k_q8 || !k_sc || !v_qs || !v_sc) return;
    const int Tend = pos0 + tloc + 1;
    const int rep = n_q / n_kv;
    if (rep < 1 || rep > 8) return;
    const int tid = threadIdx.x;
    const int hq0 = hkv * rep;
    const float scale = rsqrtf(256.f);
    __shared__ float smV[256];
    float qv[8], acc[8];
#pragma unroll
    for (int qi = 0; qi < 8; ++qi) {
        acc[qi] = 0.f;
        qv[qi] = (qi < rep && tid < hd)
                     ? q[(static_cast<size_t>(tloc) * n_q + hq0 + qi) * hd + tid]
                     : 0.f;
    }
    __shared__ float red[8][8];
    __shared__ float smi[8], sli[8], saa[8], spp[8];
    if (tid < 8) {
        smi[tid] = -1e30f;
        sli[tid] = 0.f;
    }
    __syncthreads();
    for (int t = 0; t < Tend; ++t) {
        dequant_v_tq3_256(smV, v_qs, v_sc, static_cast<size_t>(t) * n_kv + hkv, tid);
        const float ks = __half2float(k_sc[static_cast<size_t>(t) * n_kv + hkv]);
        const int8_t* kq = k_q8 + (static_cast<size_t>(t) * n_kv + hkv) * hd;
        const float kv = (tid < hd) ? ks * static_cast<float>(kq[tid]) : 0.f;
        float dots[8];
#pragma unroll
        for (int qi = 0; qi < 8; ++qi) {
            float d = (qi < rep) ? qv[qi] * kv : 0.f;
            dots[qi] = warp_sum(d);
        }
        if ((tid & 31) == 0) {
#pragma unroll
            for (int qi = 0; qi < 8; ++qi) red[tid >> 5][qi] = dots[qi];
        }
        __syncthreads();
        if (tid == 0) {
#pragma unroll
            for (int qi = 0; qi < 8; ++qi) {
                if (qi >= rep) continue;
                float tot = 0.f;
#pragma unroll
                for (int w = 0; w < 8; ++w) tot += red[w][qi];
                const float s = tot * scale;
                const float m2 = fmaxf(smi[qi], s);
                saa[qi] = __expf(smi[qi] - m2);
                spp[qi] = __expf(s - m2);
                sli[qi] = sli[qi] * saa[qi] + spp[qi];
                smi[qi] = m2;
            }
        }
        __syncthreads();
        const float vv = smV[tid];
#pragma unroll
        for (int qi = 0; qi < 8; ++qi) {
            if (qi < rep) acc[qi] = acc[qi] * saa[qi] + spp[qi] * vv;
        }
        __syncthreads();
    }
    if (tid < hd) {
#pragma unroll
        for (int qi = 0; qi < 8; ++qi) {
            if (qi >= rep) break;
            const float den = sli[qi] > 0.f ? sli[qi] : 1.f;
            o[(static_cast<size_t>(tloc) * n_q + hq0 + qi) * hd + tid] = acc[qi] / den;
        }
    }
}

template <int QT>
__global__ void attn_prefill_qtile_h_k(const float* q, const __half* k_cache, const __half* v_cache, float* o,
                                       int pos0, int T, int n_q, int n_kv, int head_dim) {
    const int hq = blockIdx.x;
    const int tb = blockIdx.y * QT;
    if (hq >= n_q || tb >= T) return;
    const int nt = T - tb < QT ? T - tb : QT;
    const int hkv = hq / (n_q / n_kv);
    const int lane = threadIdx.x;
    const int d0 = lane * 8;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int kn = n_kv * head_dim;
    float qv[QT][8];
    float acc[QT][8];
    float mi[QT], li[QT];
#pragma unroll
    for (int t = 0; t < QT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[t][i] = 0.f;
        mi[t] = -1e30f;
        li[t] = 0.f;
        if (t < nt) {
            const float* qh = q + (static_cast<size_t>(tb + t) * n_q + hq) * head_dim + d0;
#pragma unroll
            for (int i = 0; i < 8; ++i) qv[t][i] = qh[i];
        }
    }
    const int tend_max = pos0 + tb + nt;
    for (int s = 0; s < tend_max; ++s) {
        const uint4 kb = __ldg(reinterpret_cast<const uint4*>(k_cache + static_cast<size_t>(s) * kn +
                                                              hkv * head_dim + d0));
        const uint4 vb = __ldg(reinterpret_cast<const uint4*>(v_cache + static_cast<size_t>(s) * kn +
                                                              hkv * head_dim + d0));
        const __half* kh = reinterpret_cast<const __half*>(&kb);
        const __half* vh = reinterpret_cast<const __half*>(&vb);
        const float2 ka = __half22float2(*reinterpret_cast<const __half2*>(kh));
        const float2 kb2 = __half22float2(*reinterpret_cast<const __half2*>(kh + 2));
        const float2 kc = __half22float2(*reinterpret_cast<const __half2*>(kh + 4));
        const float2 kd = __half22float2(*reinterpret_cast<const __half2*>(kh + 6));
        const float2 va = __half22float2(*reinterpret_cast<const __half2*>(vh));
        const float2 vb2 = __half22float2(*reinterpret_cast<const __half2*>(vh + 2));
        const float2 vc = __half22float2(*reinterpret_cast<const __half2*>(vh + 4));
        const float2 vd = __half22float2(*reinterpret_cast<const __half2*>(vh + 6));
        const float k0 = ka.x, k1 = ka.y, k2 = kb2.x, k3 = kb2.y, k4 = kc.x, k5 = kc.y, k6 = kd.x, k7 = kd.y;
        const float v0 = va.x, v1 = va.y, v2 = vb2.x, v3 = vb2.y, v4 = vc.x, v5 = vc.y, v6 = vd.x, v7 = vd.y;
#pragma unroll
        for (int t = 0; t < QT; ++t) {
            if (t >= nt || s > pos0 + tb + t) continue;
            float dot = qv[t][0] * k0 + qv[t][1] * k1 + qv[t][2] * k2 + qv[t][3] * k3 + qv[t][4] * k4 +
                        qv[t][5] * k5 + qv[t][6] * k6 + qv[t][7] * k7;
            dot = warp_sum(dot) * scale;
            const float m2 = fmaxf(mi[t], dot);
            const float alpha = expf(mi[t] - m2);
            const float w = expf(dot - m2);
            li[t] = li[t] * alpha + w;
            acc[t][0] = acc[t][0] * alpha + w * v0;
            acc[t][1] = acc[t][1] * alpha + w * v1;
            acc[t][2] = acc[t][2] * alpha + w * v2;
            acc[t][3] = acc[t][3] * alpha + w * v3;
            acc[t][4] = acc[t][4] * alpha + w * v4;
            acc[t][5] = acc[t][5] * alpha + w * v5;
            acc[t][6] = acc[t][6] * alpha + w * v6;
            acc[t][7] = acc[t][7] * alpha + w * v7;
            mi[t] = m2;
        }
    }
#pragma unroll
    for (int t = 0; t < QT; ++t) {
        if (t >= nt) break;
        const float inv = li[t] > 0.f ? 1.f / li[t] : 0.f;
        float* oh = o + (static_cast<size_t>(tb + t) * n_q + hq) * head_dim + d0;
#pragma unroll
        for (int i = 0; i < 8; ++i) oh[i] = acc[t][i] * inv;
    }
}

// GQA + software-pipelined KV. Two independent dots in flight hide warp_sum
// latency. F16 cache loads only — not the failed cublas F16 QK path.
template <int QT>
__global__ void __launch_bounds__(192, 2)
attn_prefill_gqa_h_k(const float* q, const __half* k_cache, const __half* v_cache, float* o, int pos0, int T,
                     int n_q, int n_kv, int head_dim) {
    const int hkv = blockIdx.x;
    const int tb = blockIdx.y * QT;
    if (hkv >= n_kv || tb >= T) return;
    const int rep = n_q / n_kv;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= rep) return;
    const int hq = hkv * rep + warp;
    const int nt = T - tb < QT ? T - tb : QT;
    const int d0 = lane * 8;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int kn = n_kv * head_dim;
    float qv[QT][8];
    float acc[QT][8];
    float mi[QT], li[QT];
#pragma unroll
    for (int t = 0; t < QT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[t][i] = 0.f;
        mi[t] = -1e30f;
        li[t] = 0.f;
        if (t < nt) {
            const float* qh = q + (static_cast<size_t>(tb + t) * n_q + hq) * head_dim + d0;
#pragma unroll
            for (int i = 0; i < 8; ++i) qv[t][i] = qh[i];
        }
    }
    const int tend_max = pos0 + tb + nt;
    for (int s = 0; s < tend_max; ++s) {
        const uint4 kb = __ldg(reinterpret_cast<const uint4*>(k_cache + static_cast<size_t>(s) * kn +
                                                              hkv * head_dim + d0));
        const uint4 vb = __ldg(reinterpret_cast<const uint4*>(v_cache + static_cast<size_t>(s) * kn +
                                                              hkv * head_dim + d0));
        const __half* kh = reinterpret_cast<const __half*>(&kb);
        const __half* vh = reinterpret_cast<const __half*>(&vb);
        const float2 ka = __half22float2(*reinterpret_cast<const __half2*>(kh));
        const float2 kb2 = __half22float2(*reinterpret_cast<const __half2*>(kh + 2));
        const float2 kc = __half22float2(*reinterpret_cast<const __half2*>(kh + 4));
        const float2 kd = __half22float2(*reinterpret_cast<const __half2*>(kh + 6));
        const float2 va = __half22float2(*reinterpret_cast<const __half2*>(vh));
        const float2 vb2 = __half22float2(*reinterpret_cast<const __half2*>(vh + 2));
        const float2 vc = __half22float2(*reinterpret_cast<const __half2*>(vh + 4));
        const float2 vd = __half22float2(*reinterpret_cast<const __half2*>(vh + 6));
        const float k0 = ka.x, k1 = ka.y, k2 = kb2.x, k3 = kb2.y, k4 = kc.x, k5 = kc.y, k6 = kd.x, k7 = kd.y;
        const float v0 = va.x, v1 = va.y, v2 = vb2.x, v3 = vb2.y, v4 = vc.x, v5 = vc.y, v6 = vd.x, v7 = vd.y;
#pragma unroll
        for (int t = 0; t < QT; ++t) {
            if (t >= nt || s > pos0 + tb + t) continue;
            float dot = qv[t][0] * k0 + qv[t][1] * k1 + qv[t][2] * k2 + qv[t][3] * k3 + qv[t][4] * k4 +
                        qv[t][5] * k5 + qv[t][6] * k6 + qv[t][7] * k7;
            dot = warp_sum_all(dot) * scale;
            const float m2 = fmaxf(mi[t], dot);
            const float alpha = expf(mi[t] - m2);
            const float w = expf(dot - m2);
            li[t] = li[t] * alpha + w;
            acc[t][0] = acc[t][0] * alpha + w * v0;
            acc[t][1] = acc[t][1] * alpha + w * v1;
            acc[t][2] = acc[t][2] * alpha + w * v2;
            acc[t][3] = acc[t][3] * alpha + w * v3;
            acc[t][4] = acc[t][4] * alpha + w * v4;
            acc[t][5] = acc[t][5] * alpha + w * v5;
            acc[t][6] = acc[t][6] * alpha + w * v6;
            acc[t][7] = acc[t][7] * alpha + w * v7;
            mi[t] = m2;
        }
    }
#pragma unroll
    for (int t = 0; t < QT; ++t) {
        if (t >= nt) break;
        const float inv = li[t] > 0.f ? 1.f / li[t] : 0.f;
        float* oh = o + (static_cast<size_t>(tb + t) * n_q + hq) * head_dim + d0;
#pragma unroll
        for (int i = 0; i < 8; ++i) oh[i] = acc[t][i] * inv;
    }
}

template <int QT>
__global__ void __launch_bounds__(192, 2)
attn_prefill_gqa_k(const float* q, const float* k_cache, const float* v_cache, float* o, int pos0, int T,
                   int n_q, int n_kv, int head_dim) {
    const int hkv = blockIdx.x;
    const int tb = blockIdx.y * QT;
    if (hkv >= n_kv || tb >= T) return;
    const int rep = n_q / n_kv;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= rep) return;
    const int hq = hkv * rep + warp;
    const int nt = T - tb < QT ? T - tb : QT;
    const int d0 = lane * 8;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int kn = n_kv * head_dim;
    float qv[QT][8];
    float acc[QT][8];
    float mi[QT], li[QT];
#pragma unroll
    for (int t = 0; t < QT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[t][i] = 0.f;
        mi[t] = -1e30f;
        li[t] = 0.f;
        if (t < nt) {
            const float* qh = q + (static_cast<size_t>(tb + t) * n_q + hq) * head_dim + d0;
#pragma unroll
            for (int i = 0; i < 8; ++i) qv[t][i] = qh[i];
        }
    }
    const int tend_max = pos0 + tb + nt;
    for (int s = 0; s < tend_max; ++s) {
        const float* kh = k_cache + static_cast<size_t>(s) * kn + hkv * head_dim + d0;
        const float4 kA = *reinterpret_cast<const float4*>(kh);
        const float4 kB = *reinterpret_cast<const float4*>(kh + 4);
        const float* vh = v_cache + static_cast<size_t>(s) * kn + hkv * head_dim + d0;
        const float4 vA = *reinterpret_cast<const float4*>(vh);
        const float4 vB = *reinterpret_cast<const float4*>(vh + 4);
#pragma unroll
        for (int t = 0; t < QT; ++t) {
            if (t >= nt || s > pos0 + tb + t) continue;
            float dot = qv[t][0] * kA.x + qv[t][1] * kA.y + qv[t][2] * kA.z + qv[t][3] * kA.w +
                        qv[t][4] * kB.x + qv[t][5] * kB.y + qv[t][6] * kB.z + qv[t][7] * kB.w;
            dot = warp_sum_all(dot) * scale;
            const float m2 = fmaxf(mi[t], dot);
            const float alpha = expf(mi[t] - m2);
            const float w = expf(dot - m2);
            li[t] = li[t] * alpha + w;
            acc[t][0] = acc[t][0] * alpha + w * vA.x;
            acc[t][1] = acc[t][1] * alpha + w * vA.y;
            acc[t][2] = acc[t][2] * alpha + w * vA.z;
            acc[t][3] = acc[t][3] * alpha + w * vA.w;
            acc[t][4] = acc[t][4] * alpha + w * vB.x;
            acc[t][5] = acc[t][5] * alpha + w * vB.y;
            acc[t][6] = acc[t][6] * alpha + w * vB.z;
            acc[t][7] = acc[t][7] * alpha + w * vB.w;
            mi[t] = m2;
        }
    }
#pragma unroll
    for (int t = 0; t < QT; ++t) {
        if (t >= nt) break;
        const float inv = li[t] > 0.f ? 1.f / li[t] : 0.f;
        float* oh = o + (static_cast<size_t>(tb + t) * n_q + hq) * head_dim + d0;
#pragma unroll
        for (int i = 0; i < 8; ++i) oh[i] = acc[t][i] * inv;
    }
}

__global__ void attn_pack_q_h_k(const float* q, __half* out, int T, int t0, int n_q, int hd, int hkv, int rep) {
    const int n = T * rep * hd;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int d = i - (i / hd) * hd;
    const int tr = i / hd;
    const int t = tr / rep;
    const int r = tr - t * rep;
    const float v = q[(static_cast<size_t>(t0 + t) * n_q + hkv * rep + r) * hd + d];
    out[i] = __float2half(fminf(fmaxf(v, -65504.f), 65504.f));
}

__global__ void attn_pack_q_f_k(const float* q, float* out, int T, int t0, int n_q, int hd, int hkv, int rep) {
    const int n = T * rep * hd;
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    const int d = i - (i / hd) * hd;
    const int tr = i / hd;
    const int t = tr / rep;
    const int r = tr - t * rep;
    const float* src = q + (static_cast<size_t>(t0 + t) * n_q + hkv * rep + r) * hd + d;
    if (d + 3 < hd)
        *reinterpret_cast<float4*>(out + i) = *reinterpret_cast<const float4*>(src);
    else
        for (int k = 0; k < 4 && i + k < n; ++k) out[i + k] = src[k];
}

__global__ void attn_pack_q_f_all_k(const float* q, float* out, int T, int t0, int n_q, int hd, int n_kv,
                                   int rep) {
    const int rows = T * rep;
    const int n = n_kv * rows * hd;
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    const int d = i - (i / hd) * hd;
    const int tr = (i / hd) % rows;
    const int hkv = (i / hd) / rows;
    const int t = tr / rep;
    const int r = tr - t * rep;
    const float* src = q + (static_cast<size_t>(t0 + t) * n_q + hkv * rep + r) * hd + d;
    if (d + 3 < hd)
        *reinterpret_cast<float4*>(out + i) = *reinterpret_cast<const float4*>(src);
    else
        for (int k = 0; k < 4 && i + k < n; ++k) out[i + k] = src[k];
}

__global__ void attn_pack_k_h_k(const __half* cache, __half* out, int tend, int n_kv, int hd, int hkv) {
    const int n = tend * hd;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int d = i - (i / hd) * hd;
    const int s = i / hd;
    out[i] = cache[(static_cast<size_t>(s) * n_kv + hkv) * hd + d];
}

__global__ void attn_pack_kv_f_k(const __half* cache, float* out, int tend, int n_kv, int hd, int hkv) {
    const int n = tend * hd;
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    const int d = i - (i / hd) * hd;
    const int s = i / hd;
    const __half* src = cache + (static_cast<size_t>(s) * n_kv + hkv) * hd + d;
    if (d + 3 < hd) {
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(src));
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(src + 2));
        *reinterpret_cast<float4*>(out + i) = make_float4(a.x, a.y, b.x, b.y);
    } else {
        for (int k = 0; k < 4 && i + k < n; ++k) out[i + k] = __half2float(src[k]);
    }
}

__global__ void attn_pack_kv_f_all_k(const __half* cache, float* out, int tend, int n_kv, int hd) {
    const int n = n_kv * tend * hd;
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    const int d = i - (i / hd) * hd;
    const int s = (i / hd) % tend;
    const int hkv = (i / hd) / tend;
    const __half* src = cache + (static_cast<size_t>(s) * n_kv + hkv) * hd + d;
    if (d + 3 < hd) {
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(src));
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(src + 2));
        *reinterpret_cast<float4*>(out + i) = make_float4(a.x, a.y, b.x, b.y);
    } else {
        for (int k = 0; k < 4 && i + k < n; ++k) out[i + k] = __half2float(src[k]);
    }
}

__global__ void attn_unpack_o_k(const float* in, float* o, int T, int t0, int n_q, int hd, int hkv, int rep) {
    const int n = T * rep * hd;
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    const int d = i - (i / hd) * hd;
    const int tr = i / hd;
    const int t = tr / rep;
    const int r = tr - t * rep;
    float* dst = o + (static_cast<size_t>(t0 + t) * n_q + hkv * rep + r) * hd + d;
    if (d + 3 < hd)
        *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(in + i);
    else
        for (int k = 0; k < 4 && i + k < n; ++k) dst[k] = in[i + k];
}

__global__ void attn_unpack_o_all_k(const float* in, float* o, int T, int t0, int n_q, int hd, int n_kv, int rep) {
    const int rows = T * rep;
    const int n = n_kv * rows * hd;
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    const int d = i - (i / hd) * hd;
    const int tr = (i / hd) % rows;
    const int hkv = (i / hd) / rows;
    const int t = tr / rep;
    const int r = tr - t * rep;
    float* dst = o + (static_cast<size_t>(t0 + t) * n_q + hkv * rep + r) * hd + d;
    if (d + 3 < hd)
        *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(in + i);
    else
        for (int k = 0; k < 4 && i + k < n; ++k) dst[k] = in[i + k];
}

__global__ void attn_softmax_causal_k(float* s, int rows, int cols, int pos0, int rep, int qrows) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int local = qrows > 0 ? row - (row / qrows) * qrows : row;
    const int t = local / rep;
    int valid = pos0 + t + 1;
    if (valid > cols) valid = cols;
    if (valid <= 0) return;
    float* rowp = s + static_cast<size_t>(row) * cols;
    float mx = -1e30f;
    for (int i = threadIdx.x; i < valid; i += blockDim.x) mx = fmaxf(mx, rowp[i]);
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
    __shared__ float red[8];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = red[0];
        const int nw = (blockDim.x + 31) >> 5;
        for (int w = 1; w < nw && w < 8; ++w) m = fmaxf(m, red[w]);
        red[0] = m;
    }
    __syncthreads();
    mx = red[0];
    float sum = 0.f;
    for (int i = threadIdx.x; i < valid; i += blockDim.x) {
        const float e = expf(rowp[i] - mx);
        rowp[i] = e;
        sum += e;
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    if (lane == 0) red[wid] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float z = red[0];
        const int nw = (blockDim.x + 31) >> 5;
        for (int w = 1; w < nw && w < 8; ++w) z += red[w];
        red[0] = z > 0.f ? z : 1.f;
    }
    __syncthreads();
    const float inv = 1.f / red[0];
    for (int i = threadIdx.x; i < valid; i += blockDim.x) rowp[i] *= inv;
    for (int i = valid + threadIdx.x; i < cols; i += blockDim.x) rowp[i] = 0.f;
}

extern cublasHandle_t g_blas;
bool launch_attn_gqa_gemm(const float* q, const __half* kc, const __half* vc, float* o, float* scores,
                          size_t scores_f, float* scratch, size_t scratch_f, int pos0, int T, int n_q,
                          int n_kv, int hd) {
    if (!g_blas || !q || !kc || !vc || !o || !scores || !scratch || T <= 0 || hd != 256) return false;
    if (n_kv <= 0 || n_q % n_kv != 0) return false;
    const int rep = n_q / n_kv;
    const int tend = pos0 + T;
    if (rep < 2 || tend <= 0) return false;
    int ts = T;
    while (ts > 0 &&
           static_cast<size_t>(n_kv) * static_cast<size_t>(ts) * static_cast<size_t>(rep) *
                   static_cast<size_t>(tend) >
               scores_f)
        ts /= 2;
    if (ts < 1) return false;
    const size_t kn1 = static_cast<size_t>(tend) * hd;
    const size_t kn = kn1 * n_kv;
    const size_t qn_ts = static_cast<size_t>(ts) * rep * hd * n_kv;
    const size_t need = qn_ts + 16 + kn + 16 + kn + 16 + qn_ts + 16;
    if (need > scratch_f) return false;
    float* qf = scratch;
    float* kf = scratch + qn_ts + 16;
    float* vf = kf + kn + 16;
    float* of = vf + kn + 16;
    const float alpha = 0.0625f;
    const float beta = 0.f;
    const float one = 1.f;
    const float zero = 0.f;
    cublasSetStream(g_blas, cudaStreamPerThread);
    const int kbl = (static_cast<int>(kn) / 4 + 255) / 256;
    attn_pack_kv_f_all_k<<<kbl, 256>>>(kc, kf, tend, n_kv, hd);
    attn_pack_kv_f_all_k<<<kbl, 256>>>(vc, vf, tend, n_kv, hd);
    const int sth = tend >= 256 ? 256 : 128;
    const long long stride_k = static_cast<long long>(kn1);
    const bool batched = n_kv > 1;
    for (int t0 = 0; t0 < T; t0 += ts) {
        const int nt = T - t0 < ts ? T - t0 : ts;
        const int rows = nt * rep;
        const int qn = rows * hd * n_kv;
        const int qbl = (qn / 4 + 255) / 256;
        attn_pack_q_f_all_k<<<qbl, 256>>>(q, qf, nt, t0, n_q, hd, n_kv, rep);
        const long long stride_q = static_cast<long long>(rows) * hd;
        const long long stride_s = static_cast<long long>(rows) * tend;
        bool ok = false;
        if (batched &&
            cublasGemmStridedBatchedEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, tend, rows, hd, &alpha, kf,
                                       CUDA_R_32F, hd, stride_k, qf, CUDA_R_32F, hd, stride_q, &beta, scores,
                                       CUDA_R_32F, tend, stride_s, n_kv, CUBLAS_COMPUTE_32F_FAST_TF32,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS)
            ok = true;
        if (!ok) {
            for (int h = 0; h < n_kv; ++h) {
                if (cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, tend, rows, hd, &alpha, kf + h * kn1,
                                 CUDA_R_32F, hd, qf + h * stride_q, CUDA_R_32F, hd, &beta,
                                 scores + h * stride_s, CUDA_R_32F, tend, CUBLAS_COMPUTE_32F_FAST_TF32,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS &&
                    cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, tend, rows, hd, &alpha, kf + h * kn1, hd,
                                qf + h * stride_q, hd, &beta, scores + h * stride_s, tend) !=
                        CUBLAS_STATUS_SUCCESS)
                    return false;
            }
        }
        attn_softmax_causal_k<<<n_kv * rows, sth>>>(scores, n_kv * rows, tend, pos0 + t0, rep, rows);
        ok = false;
        if (batched &&
            cublasGemmStridedBatchedEx(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, hd, rows, tend, &one, vf, CUDA_R_32F,
                                       hd, stride_k, scores, CUDA_R_32F, tend, stride_s, &zero, of,
                                       CUDA_R_32F, hd, stride_q, n_kv, CUBLAS_COMPUTE_32F_FAST_TF32,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS)
            ok = true;
        if (!ok) {
            for (int h = 0; h < n_kv; ++h) {
                if (cublasGemmEx(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, hd, rows, tend, &one, vf + h * kn1,
                                 CUDA_R_32F, hd, scores + h * stride_s, CUDA_R_32F, tend, &zero,
                                 of + h * stride_q, CUDA_R_32F, hd, CUBLAS_COMPUTE_32F_FAST_TF32,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS &&
                    cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, hd, rows, tend, &one, vf + h * kn1, hd,
                                scores + h * stride_s, tend, &zero, of + h * stride_q, hd) !=
                        CUBLAS_STATUS_SUCCESS)
                    return false;
            }
        }
        attn_unpack_o_all_k<<<qbl, 256>>>(of, o, nt, t0, n_q, hd, n_kv, rep);
    }
    return true;
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
        int d = 0;
        for (; d + 3 < head_dim; d += 4) {
            const float4 q4 = *reinterpret_cast<const float4*>(qh + d);
            const float4 k4 = *reinterpret_cast<const float4*>(kh + d);
            dot += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
        }
        for (; d < head_dim; ++d) dot += qh[d] * kh[d];
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

__device__ __forceinline__ void conv1d_prefill_k4_one(const float* x, const float* w, float* y, float* state,
                                                     int seq, int dim, int c) {
    float* st = state + c * 4;
    const float* wc = w + c * 4;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    float xt = seq > 0 ? x[c] : 0.f;
    for (int t = 0; t < seq; ++t) {
        const float nxt = (t + 1 < seq) ? x[static_cast<size_t>(t + 1) * dim + c] : 0.f;
        y[static_cast<size_t>(t) * dim + c] = silu_d(w0 * s1 + w1 * s2 + w2 * s3 + w3 * xt);
        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = xt;
        xt = nxt;
    }
    st[0] = s0;
    st[1] = s1;
    st[2] = s2;
    st[3] = s3;
}

__global__ void conv1d_prefill_k(const float* x, const float* w, float* y, float* state, int seq, int dim,
                                 int k) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (k == 4) {
        const int c = tid * 2;
        if (c >= dim) return;
        if (c + 1 >= dim) {
            conv1d_prefill_k4_one(x, w, y, state, seq, dim, c);
            return;
        }
        float* sta = state + c * 4;
        float* stb = state + (c + 1) * 4;
        const float* wa = w + c * 4;
        const float* wb = w + (c + 1) * 4;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
        const float u0 = wa[0], u1 = wa[1], u2 = wa[2], u3 = wa[3];
        const float v0 = wb[0], v1 = wb[1], v2 = wb[2], v3 = wb[3];
        float xa = seq > 0 ? x[c] : 0.f, xb = seq > 0 ? x[c + 1] : 0.f;
        for (int t = 0; t < seq; ++t) {
            const size_t base = static_cast<size_t>(t) * dim;
            y[base + c] = silu_d(u0 * a1 + u1 * a2 + u2 * a3 + u3 * xa);
            y[base + c + 1] = silu_d(v0 * b1 + v1 * b2 + v2 * b3 + v3 * xb);
            a0 = a1;
            a1 = a2;
            a2 = a3;
            a3 = xa;
            b0 = b1;
            b1 = b2;
            b2 = b3;
            b3 = xb;
            if (t + 1 < seq) {
                xa = x[base + dim + c];
                xb = x[base + dim + c + 1];
            }
        }
        sta[0] = a0;
        sta[1] = a1;
        sta[2] = a2;
        sta[3] = a3;
        stb[0] = b0;
        stb[1] = b1;
        stb[2] = b2;
        stb[3] = b3;
        return;
    }
    const int c = tid;
    if (c >= dim) return;
    float* st = state + c * k;
    const float* wc = w + c * k;
    for (int t = 0; t < seq; ++t) {
        float acc = 0.f;
        for (int p = 0; p < k; ++p) {
            const int src = t - (k - 1 - p);
            const float xv = src >= 0 ? x[static_cast<size_t>(src) * dim + c] : 0.f;
            acc += wc[p] * xv;
        }
        y[static_cast<size_t>(t) * dim + c] = silu_d(acc);
    }
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
    bool fp8_rowmaj = false; // row-major e4, scales absorbed — cublasLt prefill
    bool qk_soa = false;    // K-quant scales pre-expanded, quants still packed
};

cublasHandle_t g_blas = nullptr;
__half* g_xf16 = nullptr;
int g_xf16_n = 0;
__half* g_wf16 = nullptr;
size_t g_wf16_n = 0;
const float* g_xf16_src = nullptr;
int g_xf16_src_n = 0;
cudaStream_t g_dq_stream = nullptr;
cudaEvent_t g_dq_done[2] = {};
cudaEvent_t g_gemm_done[2] = {};
int g_wf_slot = 0;
bool g_fp8_tc = true;
bool g_fp8_e4m3_mma = false;
bool g_fp8_shared_tc = false;
bool g_fp8_e4_tc = false;
bool g_fp8_e4_bn256 = false;
bool g_fp8_e4_bk512 = false;
bool g_fp8_e4_pipe = false;
bool g_fp8_e4_occ = false;
bool g_fp8_e4_o2 = false;
bool g_cublas_fp8_e4 = false;
cublasLtHandle_t g_lt = nullptr;
cublasLtHandle_t g_lt2 = nullptr;
// Persistent cublasLt descriptors: create/destroy per GEMM breaks T=256 graph
// capture (dangling desc) and adds host setup that previously lost to bk512.
struct LtPack {
    int m = 0, n = 0, k = 0, add = 0;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    cublasLtMatmulAlgo_t algo{};
    bool have_algo = false;
    size_t ws = 0;
};
constexpr int kLtMax = 64;
LtPack g_lt_cache[kLtMax];
int g_lt_n = 0;
void* g_lt_ws = nullptr;
void* g_lt_ws2 = nullptr;
size_t g_lt_ws_bytes = 0;
// File-scope const in .cu is a device symbol; cublasLt alpha/beta must be host.
struct LtHostScalar {
    float alpha = 1.f;
    float beta0 = 0.f;
    float beta1 = 1.f;
};
LtHostScalar g_lt_hs;
int g_lt_ok = 0, g_lt_fail = 0;

void lt_cache_clear() {
    for (int i = 0; i < g_lt_n; ++i) {
        if (g_lt_cache[i].la) cublasLtMatrixLayoutDestroy(g_lt_cache[i].la);
        if (g_lt_cache[i].lb) cublasLtMatrixLayoutDestroy(g_lt_cache[i].lb);
        if (g_lt_cache[i].lc) cublasLtMatrixLayoutDestroy(g_lt_cache[i].lc);
        if (g_lt_cache[i].desc) cublasLtMatmulDescDestroy(g_lt_cache[i].desc);
        g_lt_cache[i] = LtPack{};
    }
    g_lt_n = 0;
    if (g_lt_ws) {
        cudaFree(g_lt_ws);
        g_lt_ws = nullptr;
        g_lt_ws_bytes = 0;
    }
    if (g_lt_ws2) {
        cudaFree(g_lt_ws2);
        g_lt_ws2 = nullptr;
    }
}

LtPack* lt_get(int m, int n, int k, int add) {
    for (int i = 0; i < g_lt_n; ++i) {
        if (g_lt_cache[i].m == m && g_lt_cache[i].n == n && g_lt_cache[i].k == k && g_lt_cache[i].add == add)
            return &g_lt_cache[i];
    }
    if (g_lt_n >= kLtMax || !g_lt) return nullptr;
    LtPack& p = g_lt_cache[g_lt_n];
    p = LtPack{};
    p.m = m;
    p.n = n;
    p.k = k;
    p.add = add;
    if (cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return nullptr;
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    if (cublasLtMatrixLayoutCreate(&p.la, CUDA_R_8F_E4M3, k, m, k) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_8F_E4M3, k, n, k) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_32F, m, n, m) != CUBLAS_STATUS_SUCCESS) {
        if (p.la) cublasLtMatrixLayoutDestroy(p.la);
        if (p.lb) cublasLtMatrixLayoutDestroy(p.lb);
        if (p.lc) cublasLtMatrixLayoutDestroy(p.lc);
        if (p.desc) cublasLtMatmulDescDestroy(p.desc);
        p = LtPack{};
        return nullptr;
    }
    if (!g_lt_ws) {
        g_lt_ws_bytes = 64ull * 1024ull * 1024ull;
        if (cudaMalloc(&g_lt_ws, g_lt_ws_bytes) != cudaSuccess) {
            g_lt_ws = nullptr;
            g_lt_ws_bytes = 0;
            cudaGetLastError();
        }
    }
    if (!g_lt_ws2 && g_lt_ws_bytes) {
        if (cudaMalloc(&g_lt_ws2, g_lt_ws_bytes) != cudaSuccess) {
            g_lt_ws2 = nullptr;
            cudaGetLastError();
        }
    }
    cublasLtMatmulPreference_t pref = nullptr;
    if (cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_bytes,
                                             sizeof(g_lt_ws_bytes));
        cublasLtMatmulHeuristicResult_t heur{};
        int nr = 0;
        if (cublasLtMatmulAlgoGetHeuristic(g_lt, p.desc, p.la, p.lb, p.lc, p.lc, pref, 1, &heur, &nr) ==
                CUBLAS_STATUS_SUCCESS &&
            nr > 0 && heur.state == CUBLAS_STATUS_SUCCESS) {
            p.algo = heur.algo;
            p.have_algo = true;
            p.ws = heur.workspaceSize;
            if (p.ws > g_lt_ws_bytes) {
                p.have_algo = false;
                p.ws = 0;
            }
            std::fprintf(stderr, "lt_heur m=%d n=%d k=%d add=%d algo=%d ws=%zu\n", m, n, k, add,
                         p.have_algo ? 1 : 0, p.ws);
        } else {
            std::fprintf(stderr, "lt_heur_fail m=%d n=%d k=%d add=%d nr=%d\n", m, n, k, add, nr);
        }
        cublasLtMatmulPreferenceDestroy(pref);
    }
    ++g_lt_n;
    return &p;
}

void use_xe(const float* X, int T, int n);
extern const uint8_t* g_xe;
void lt_tune(const GpuW& w, const float* X, float* Y, int T, int add) {
    if (!g_lt || !w.data || T < 256 || w.rows < 128 || w.cols < 128) return;
    LtPack* pack = nullptr;
    for (int i = 0; i < g_lt_n; ++i)
        if (g_lt_cache[i].m == w.rows && g_lt_cache[i].n == T && g_lt_cache[i].k == w.cols &&
            g_lt_cache[i].add == add)
            pack = &g_lt_cache[i];
    if (!pack || !pack->desc) return;
    use_xe(X, T, w.cols);
    if (!g_xe) return;
    cublasLtMatmulPreference_t pref = nullptr;
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) return;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt_ws_bytes,
                                         sizeof(g_lt_ws_bytes));
    cublasLtMatmulHeuristicResult_t heurs[8];
    int nr = 0;
    const cublasStatus_t hs =
        cublasLtMatmulAlgoGetHeuristic(g_lt, pack->desc, pack->la, pack->lb, pack->lc, pack->lc, pref, 8,
                                       heurs, &nr);
    cublasLtMatmulPreferenceDestroy(pref);
    if (hs != CUBLAS_STATUS_SUCCESS || nr <= 0) return;
    const float* beta = add ? &g_lt_hs.beta1 : &g_lt_hs.beta0;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float best = 1e30f;
    int bi = -1;
    for (int i = 0; i < nr; ++i) {
        if (heurs[i].state != CUBLAS_STATUS_SUCCESS || heurs[i].workspaceSize > g_lt_ws_bytes) continue;
        void* ws = heurs[i].workspaceSize ? g_lt_ws : nullptr;
        cublasLtMatmul(g_lt, pack->desc, &g_lt_hs.alpha, w.data, pack->la, g_xe, pack->lb, beta, Y, pack->lc,
                       Y, pack->lc, &heurs[i].algo, ws, heurs[i].workspaceSize, cudaStreamPerThread);
        cudaEventRecord(e0, cudaStreamPerThread);
        if (cublasLtMatmul(g_lt, pack->desc, &g_lt_hs.alpha, w.data, pack->la, g_xe, pack->lb, beta, Y,
                           pack->lc, Y, pack->lc, &heurs[i].algo, ws, heurs[i].workspaceSize,
                           cudaStreamPerThread) != CUBLAS_STATUS_SUCCESS)
            continue;
        cudaEventRecord(e1, cudaStreamPerThread);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) {
            best = ms;
            bi = i;
        }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (bi >= 0) {
        pack->algo = heurs[bi].algo;
        pack->have_algo = true;
        pack->ws = heurs[bi].workspaceSize;
        std::fprintf(stderr, "lt_tune m=%d n=%d k=%d add=%d pick=%d/%d ms=%.3f ws=%zu\n", w.rows, T, w.cols,
                     add, bi, nr, best, pack->ws);
    }
}

cudaStream_t g_launch_stream = nullptr;
cudaStream_t g_bak_stream = nullptr;
cudaEvent_t g_pair_ev = nullptr;
uint8_t* g_xe_buf = nullptr;
const uint8_t* g_xe = nullptr;
int g_xe_n = 0, g_xe_T = 0, g_xe_cap = 0;

cudaStream_t gemm_stream() { return g_launch_stream ? g_launch_stream : cudaStreamPerThread; }

__global__ void f32_to_e4_n_k(const float* x, uint8_t* y, int n) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
    if (i + 3 < n) {
        const float4 v = *reinterpret_cast<const float4*>(x + i);
        y[i + 0] = f32_to_fp8e4(v.x);
        y[i + 1] = f32_to_fp8e4(v.y);
        y[i + 2] = f32_to_fp8e4(v.z);
        y[i + 3] = f32_to_fp8e4(v.w);
    } else {
        for (int k = 0; k < 4 && i + k < n; ++k) y[i + k] = f32_to_fp8e4(x[i + k]);
    }
}

void use_xe(const float* X, int T, int n) {
    if (!g_xe_buf || !X || T < 16 || n <= 0 || T * n > g_xe_cap) {
        g_xe = nullptr;
        g_xe_n = 0;
        g_xe_T = 0;
        return;
    }
    const int nn = T * n;
    f32_to_e4_n_k<<<(nn + 1023) / 1024, 256, 0, gemm_stream()>>>(X, g_xe_buf, nn);
    g_xe = g_xe_buf;
    g_xe_n = n;
    g_xe_T = T;
}

__global__ void swiglu_to_e4_k(const float* gate, const float* up, uint8_t* xe, int n) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
    if (i + 3 < n) {
        const float4 g = *reinterpret_cast<const float4*>(gate + i);
        const float4 u = *reinterpret_cast<const float4*>(up + i);
        xe[i + 0] = f32_to_fp8e4(silu_d(g.x) * u.x);
        xe[i + 1] = f32_to_fp8e4(silu_d(g.y) * u.y);
        xe[i + 2] = f32_to_fp8e4(silu_d(g.z) * u.z);
        xe[i + 3] = f32_to_fp8e4(silu_d(g.w) * u.w);
    } else {
        for (int k = 0; k < 4 && i + k < n; ++k) xe[i + k] = f32_to_fp8e4(silu_d(gate[i + k]) * up[i + k]);
    }
}

void use_swiglu_xe(const float* gate, const float* up, int T, int n) {
    if (!g_xe_buf || !gate || !up || T < 16 || n <= 0 || T * n > g_xe_cap) {
        g_xe = nullptr;
        g_xe_n = 0;
        g_xe_T = 0;
        return;
    }
    const int nn = T * n;
    swiglu_to_e4_k<<<(nn + 1023) / 1024, 256, 0, gemm_stream()>>>(gate, up, g_xe_buf, nn);
    g_xe = g_xe_buf;
    g_xe_n = n;
    g_xe_T = T;
}

void launch_linear(const GpuW& w, const float* X, float* Y, int T, int add = 0);

void launch_linear_pair(const GpuW& a, const GpuW& b, const float* X, float* Ya, float* Yb, int T) {
    // Dual-stream cublasLt on T=1024/2048 lost to HBM contention vs sequential
    // (2048-fill 7.68 vs 8.02). Keep sequential; leftover∥conv uses bak separately.
    launch_linear(a, X, Ya, T);
    launch_linear(b, X, Yb, T);
}

bool launch_cublas_f16(const GpuW& w, const float* X, float* Y, int T);
void launch_fp8_tc(const GpuW& w, const float* X, float* Y, int T);
void launch_gemm_fp8(const GpuW& w, const float* X, float* Y, int T, int add);
void launch_fp8_e4m3_mma(const GpuW& w, const float* x, float* y);
bool launch_fp8_shared_tc(const GpuW& w, const float* X, float* Y, int T, int add);
bool launch_fp8_e4_tc(const GpuW& w, const float* X, float* Y, int T, int add);
void launch_linear(const GpuW& w, const float* X, float* Y, int T, int add);
void launch_linear_pair(const GpuW& a, const GpuW& b, const float* X, float* Ya, float* Yb, int T);

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
        if (w.rows >= 4096) {
            const int pairs = (w.rows + 1) / 2;
            const int warps = threads / 32;
            const int pblocks = (pairs + warps - 1) / warps;
            gemv_q8_soa_2row_k<<<pblocks, threads, sizeof(float) * smem_n>>>(
                reinterpret_cast<const int8_t*>(w.data), reinterpret_cast<const __half*>(w.scale), x, y, w.rows,
                w.cols, add);
        } else {
            gemv_q8_soa_k<<<blocks, threads, sizeof(float) * smem_n>>>(reinterpret_cast<const int8_t*>(w.data),
                                                                       reinterpret_cast<const __half*>(w.scale), x,
                                                                       y, w.rows, w.cols, add);
        }
        break;
    }
    case QuantKind::Q4_K:
    case QuantKind::Q5_K:
    case QuantKind::Q6_K: {
        if (w.cols <= 0 || (w.cols % 256) != 0)
            throw std::runtime_error("CUDA K-quant GEMV cols must be multiple of 256");
        const int tile = w.cols < kXsTile ? w.cols : kXsTile;
        const size_t smem = sizeof(float) * static_cast<size_t>(tile);
        // One row / warp: k-quant dequant is compute-bound; 2-row doubles that work.
        if (w.q == QuantKind::Q4_K) {
            if (w.qk_soa) gemv_qk_k<kQ4KSoaBsz><<<blocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
            else gemv_qk_k<kQ4KBsz><<<blocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
        } else if (w.q == QuantKind::Q5_K) {
            if (w.qk_soa) gemv_qk_k<kQ5KSoaBsz><<<blocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
            else gemv_qk_k<kQ5KBsz><<<blocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
        } else {
            if (w.qk_soa) gemv_qk_k<kQ6KSoaBsz><<<blocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
            else gemv_qk_k<kQ6KBsz><<<blocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
        }
        break;
    }
    case QuantKind::FP8_E4M3_B128: {
        if (w.fp8_rowmaj && w.data && w.rows > 0 && w.cols > 0) {
            const int tile = fp8_xs_tile(w.cols);
            const int warps = threads / 32;
            const int pairs = (w.rows + 1) / 2;
            const int pblocks = (pairs + warps - 1) / warps;
            gemv_fp8_rm_2row_k<<<pblocks, threads, sizeof(float) * tile>>>(w.data, x, y, w.rows, w.cols, add);
            break;
        }
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
        w1.cols == w2.cols && w1.data && w2.data && w1.rows > 0 && w1.cols > 0 && w1.fp8_kmajor &&
        w2.fp8_kmajor) {
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
        w1.data && w2.data && w1.cols > 0 && w1.rows > 0 && w2.rows > 0 && w1.fp8_kmajor && w2.fp8_kmajor &&
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
        wk.data && wv.data && wq.rows >= 4096 && split_at > 0 && wq.rows == split_at * 2 &&
        wq.fp8_kmajor && wk.fp8_kmajor && wv.fp8_kmajor) {
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

// k-major FP8 × X[T,n] → Y[T,m]. One 128-col W tile is converted in smem, then
// 8×(k16 MMA) reuse it across a 64-token panel. No full-matrix FP16 workspace.
__global__ void gemm_fp8_kmaj_tc_k(const uint8_t* W, const float* scale, const float* X, float* Y, int m,
                                   int n, int T, int add) {
#if __CUDA_ARCH__ >= 800
    constexpr int TS = 64, KN = 128;
    const int warps = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * warps + warp) * 16;
    extern __shared__ char raw[];
    __half* Xs = reinterpret_cast<__half*>(raw);          // [TS][KN]
    __half* Ws = Xs + TS * KN;                           // [warps][16][KN]
    __half* As = Ws + warps * 16 * KN;                   // [warps][16*16] ldmatrix A
    __half* Bs = As + warps * 256;                       // [8][16] ldmatrix B
    const int nb_n = (n + 127) >> 7;
    for (int t0 = 0; t0 < T; t0 += TS) {
        const int tt = T - t0 < TS ? T - t0 : TS;
        float acc[8][4];
#pragma unroll
        for (int g = 0; g < 8; ++g) acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
        for (int k0 = 0; k0 < n; k0 += KN) {
            const int cb = k0 >> 7;
            for (int i = threadIdx.x; i < tt * KN; i += blockDim.x) {
                const int t = i / KN;
                const int c = i - t * KN;
                Xs[t * KN + c] = __float2half(X[static_cast<size_t>(t0 + t) * n + k0 + c]);
            }
            __half* Ww = Ws + warp * 16 * KN;
            const float s = (row0 < m && scale) ? scale[(row0 >> 7) * nb_n + cb] : 0.f;
            for (int r = 0; r < 16; ++r) {
                const int row = row0 + r;
                const uint32_t p = (row < m) ? *fp8_blk4(W, row, m, cb, lane) : 0;
                const uint8_t* b = reinterpret_cast<const uint8_t*>(&p);
                __half* dst = Ww + r * KN + lane * 4;
                dst[0] = __float2half(fp8e4_to_f32(b[0]) * s);
                dst[1] = __float2half(fp8e4_to_f32(b[1]) * s);
                dst[2] = __float2half(fp8e4_to_f32(b[2]) * s);
                dst[3] = __float2half(fp8e4_to_f32(b[3]) * s);
            }
            __syncthreads();
            __half* Aw = As + warp * 256;
            for (int kt = 0; kt < 8; ++kt) {
                for (int i = lane; i < 256; i += 32) {
                    const int r = i >> 4;
                    const int c = i & 15;
                    Aw[r * 16 + c] = Ww[r * KN + kt * 16 + c];
                }
                for (int g = 0; g < 8; ++g) {
                    if (g * 8 >= tt) break;
                    if (warp == 0) {
                        for (int i = lane; i < 128; i += 32) {
                            const int tk = i >> 4;
                            const int ck = i & 15;
                            Bs[tk * 16 + ck] = Xs[(g * 8 + tk) * KN + kt * 16 + ck];
                        }
                    }
                    __syncthreads();
                    uint32_t a[4], b[2];
                    ldmatrix_x4(a, Aw + (lane % 16) * 16 + (lane / 16) * 8);
                    ldmatrix_x2(b, Bs + (lane % 8) * 16 + (lane / 8) * 8);
                    mma_m16n8k16_f16(acc[g], a, b);
                    __syncthreads();
                }
            }
        }
        if (row0 < m) {
            const int r0 = lane / 4;
            const int c0 = (lane % 4) * 2;
            for (int g = 0; g < 8; ++g) {
                const int tok0 = g * 8 + c0;
                if (tok0 < tt) {
                    if (row0 + r0 < m) write_y(Y + static_cast<size_t>(t0 + tok0) * m, row0 + r0, acc[g][0], add);
                    if (row0 + r0 + 8 < m)
                        write_y(Y + static_cast<size_t>(t0 + tok0) * m, row0 + r0 + 8, acc[g][2], add);
                }
                if (tok0 + 1 < tt) {
                    if (row0 + r0 < m)
                        write_y(Y + static_cast<size_t>(t0 + tok0 + 1) * m, row0 + r0, acc[g][1], add);
                    if (row0 + r0 + 8 < m)
                        write_y(Y + static_cast<size_t>(t0 + tok0 + 1) * m, row0 + r0 + 8, acc[g][3], add);
                }
            }
        }
        __syncthreads();
    }
#else
    (void)W;
    (void)scale;
    (void)X;
    (void)Y;
    (void)m;
    (void)n;
    (void)T;
    (void)add;
#endif
}

// One warp / (row, T-tile). Reuses each k-major W vector across TN tokens.
// The BM=128 e4 dual path is ~15 ms/layer on m=48; this is the leftover prefill.
template <int TN>
__global__ void gemm_fp8_kmaj_skinny_grid_k(const uint8_t* W, const float* scale, const float* X, float* Y,
                                            int m, int n, int T, int add) {
    const int row = blockIdx.x;
    const int t0 = blockIdx.y * TN;
    const int lane = threadIdx.x & 31;
    if (row >= m || t0 >= T || !scale) return;
    const int tt = T - t0 < TN ? T - t0 : TN;
    const int nb_n = n >> 7;
    const int bi = row >> 7;
    float acc[TN];
#pragma unroll
    for (int t = 0; t < TN; ++t) acc[t] = 0.f;
    for (int b = 0; b < nb_n; ++b) {
        const float s = scale[bi * nb_n + b];
        const uint32_t p = __ldcs(fp8_blk4(W, row, m, b, lane));
#pragma unroll
        for (int t = 0; t < TN; ++t) {
            if (t < tt)
                acc[t] += s * fp8x4_dot(p, X + static_cast<size_t>(t0 + t) * n + (b << 7) + (lane << 2));
        }
    }
#pragma unroll
    for (int t = 0; t < TN; ++t) {
        if (t < tt) {
            const float a = warp_sum(acc[t]);
            if (lane == 0) write_y(Y + static_cast<size_t>(t0 + t) * m, row, a, add);
        }
    }
}

int g_skinny_n = 0;

bool launch_fp8_skinny(const GpuW& w, const float* X, float* Y, int T, int add) {
    if (w.q != QuantKind::FP8_E4M3_B128 || !w.fp8_kmajor || !w.data || !w.scale) return false;
    if (w.rows <= 0 || w.rows >= 128 || T < 2 || (w.cols & 127) != 0) return false;
    constexpr int TN = 8;
    dim3 grid(w.rows, (T + TN - 1) / TN);
    gemm_fp8_kmaj_skinny_grid_k<TN><<<grid, 32>>>(w.data, w.scale, X, Y, w.rows, w.cols, T, add);
    ++g_skinny_n;
    return true;
}

bool launch_fp8_e4_tc(const GpuW& w, const float* X, float* Y, int T, int add) {
    if (w.rows < 128 && launch_fp8_skinny(w, X, Y, T, add)) return true;
    if (!g_fp8_e4_tc || T < 16 || w.rows < 16 || !w.fp8_kmajor) return false;
    if (w.q != QuantKind::FP8_E4M3_B128 || !w.data || !w.scale) return false;
    if ((w.cols & 127) != 0) return false;
    constexpr int BM = 128, WLD = 144, XLD = 144;
    // Skinny leftover (m=48): stay on 256-thread BN=32/64. bn128/bn256 would
    // launch 512–1024 threads for one 48-row tile (most warps idle on W).
    cudaStream_t st = gemm_stream();
    if (w.rows >= 128 && T >= 128 && g_fp8_e4_bk512) {
        constexpr int WLD512 = 528;
        dim3 grid((w.rows + BM - 1) / BM, (T + 127) / 128);
        const size_t smem = static_cast<size_t>(BM) * WLD512 + static_cast<size_t>(128) * XLD;
        const uint8_t* Xe = (g_xe && g_xe_n == w.cols && g_xe_T == T) ? g_xe : nullptr;
        gemm_fp8_kmaj_e4_bk512_k<<<grid, 512, smem, st>>>(w.data, w.scale, X, Y, w.rows, w.cols, T, add, Xe);
        return true;
    }
    if (w.rows >= 128 && T > 256 && g_fp8_e4_bn256) {
        dim3 grid((w.rows + BM - 1) / BM, (T + 255) / 256);
        const size_t smem = static_cast<size_t>(BM) * WLD + static_cast<size_t>(256) * XLD;
        gemm_fp8_kmaj_e4_bn256_k<<<grid, 1024, smem, st>>>(w.data, w.scale, X, Y, w.rows, w.cols, T, add);
        return true;
    }
    if (w.rows >= 128 && T >= 128) {
        dim3 grid((w.rows + BM - 1) / BM, (T + 127) / 128);
        const size_t smem = static_cast<size_t>(BM) * WLD + static_cast<size_t>(128) * XLD;
        gemm_fp8_kmaj_e4_bn128_k<<<grid, 512, smem, st>>>(w.data, w.scale, X, Y, w.rows, w.cols, T, add);
        return true;
    }
    const int BN = T >= 64 ? 64 : 32;
    dim3 grid((w.rows + BM - 1) / BM, (T + BN - 1) / BN);
    const size_t smem = static_cast<size_t>(BM) * WLD + static_cast<size_t>(BN) * XLD;
    if (BN == 64)
        gemm_fp8_kmaj_e4_k<64><<<grid, 256, smem, st>>>(w.data, w.scale, X, Y, w.rows, w.cols, T, add);
    else
        gemm_fp8_kmaj_e4_k<32><<<grid, 256, smem, st>>>(w.data, w.scale, X, Y, w.rows, w.cols, T, add);
    return true;
}

bool launch_fp8_shared_tc(const GpuW& w, const float* X, float* Y, int T, int add) {
    if (!g_fp8_shared_tc || T < 16 || T > 256 || w.rows < 128 || !w.fp8_kmajor) return false;
    if (w.q != QuantKind::FP8_E4M3_B128 || !w.data || !w.scale) return false;
    if ((w.cols & 127) != 0) return false;
    constexpr int BM = 128, WLD = 136, XLD = 136;
    auto go = [&](int BN) {
        dim3 grid((w.rows + BM - 1) / BM, (T + BN - 1) / BN);
        const size_t smem = (static_cast<size_t>(BM) * WLD + static_cast<size_t>(BN) * XLD) * sizeof(__half);
        if (BN == 64)
            gemm_fp8_shared_tc_k<64><<<grid, 256, smem>>>(w.data, w.scale, X, Y, w.rows, w.cols, T, add);
        else
            gemm_fp8_shared_tc_k<32><<<grid, 256, smem>>>(w.data, w.scale, X, Y, w.rows, w.cols, T, add);
    };
    go(T >= 64 ? 64 : 32);
    return true;
}

bool launch_cublas_fp8(const GpuW& w, const float* X, float* Y, int T, int add) {
    if (!g_blas || !g_xf16 || !g_wf16 || T < 16) return false;
    if (w.q != QuantKind::FP8_E4M3_B128 || !w.fp8_kmajor || !w.data || !w.scale) return false;
    if (w.rows < 16 || w.cols < 128 || (w.cols & 127) != 0) return false;
    const size_t one = g_wf16_n / 2;
    const size_t wn = static_cast<size_t>(w.rows) * static_cast<size_t>(w.cols);
    if (one == 0 || wn > one) return false;
    if (w.cols * T > g_xf16_n) return false;
    if (!g_dq_stream) {
        cudaStreamCreateWithFlags(&g_dq_stream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&g_dq_done[0], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&g_dq_done[1], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&g_gemm_done[0], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&g_gemm_done[1], cudaEventDisableTiming);
        cudaEventRecord(g_gemm_done[0], cudaStreamPerThread);
        cudaEventRecord(g_gemm_done[1], cudaStreamPerThread);
    }
    const int slot = g_wf_slot;
    g_wf_slot = 1 - g_wf_slot;
    __half* wbuf = g_wf16 + slot * one;
    cudaStreamWaitEvent(g_dq_stream, g_gemm_done[slot], 0);
    dim3 grid((w.rows + 127) / 128, w.cols / 128);
    dequant_fp8_kmaj_row_k<<<grid, 256, 0, g_dq_stream>>>(w.data, w.scale, wbuf, w.rows, w.cols);
    cudaEventRecord(g_dq_done[slot], g_dq_stream);
    cudaStreamWaitEvent(cudaStreamPerThread, g_dq_done[slot], 0);
    f32_to_f16_k<<<(w.cols * T + 255) / 256, 256>>>(X, g_xf16, w.cols * T);
    const float alpha = 1.f, beta = add ? 1.f : 0.f;
    const bool ok = cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, w.rows, T, w.cols, &alpha, wbuf, CUDA_R_16F,
                                 w.cols, g_xf16, CUDA_R_16F, w.cols, &beta, Y, CUDA_R_32F, w.rows,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS;
    cudaEventRecord(g_gemm_done[slot], cudaStreamPerThread);
    return ok;
}

// Persistent row-major e4 (scales absorbed at load). No per-GEMM unpack. T=1 untouched.
bool launch_cublas_fp8_e4(const GpuW& w, const float* X, float* Y, int T, int add) {
    cublasLtHandle_t lt = (g_launch_stream && g_launch_stream == g_bak_stream && g_lt2) ? g_lt2 : g_lt;
    if (!g_cublas_fp8_e4 || !lt || T < 16 || w.rows < 128) return false;
    if (w.q != QuantKind::FP8_E4M3_B128 || !w.data || !w.fp8_rowmaj) return false;
    if (w.cols < 128 || (w.cols & 127) != 0) return false;
    LtPack* pack = lt_get(w.rows, T, w.cols, add ? 1 : 0);
    if (!pack || !pack->desc) return false;
    const uint8_t* Xe = (g_xe && g_xe_n == w.cols && g_xe_T == T) ? g_xe : nullptr;
    uint8_t* xbuf = nullptr;
    if (!Xe) {
        if (!g_xf16 || w.cols * T > g_xf16_n) return false;
        xbuf = reinterpret_cast<uint8_t*>(g_xf16);
        f32_to_e4_n_k<<<(w.cols * T + 1023) / 1024, 256, 0, gemm_stream()>>>(X, xbuf, w.cols * T);
        Xe = xbuf;
    }
    const float* beta = add ? &g_lt_hs.beta1 : &g_lt_hs.beta0;
    void* ws = g_lt_ws;
    if (g_launch_stream && g_launch_stream == g_bak_stream && g_lt_ws2) ws = g_lt_ws2;
    const cublasStatus_t st = cublasLtMatmul(
        lt, pack->desc, &g_lt_hs.alpha, w.data, pack->la, Xe, pack->lb, beta, Y, pack->lc, Y, pack->lc,
        pack->have_algo ? &pack->algo : nullptr, pack->ws ? ws : nullptr, pack->ws, gemm_stream());
    if (st == CUBLAS_STATUS_SUCCESS) {
        ++g_lt_ok;
        return true;
    }
    ++g_lt_fail;
    return false;
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
    else if (T > 1 && dk == 128 && dv == 128 && mix) {
        gdn_norm_qk_mix_k<<<dim3(nk, T), 32>>>(const_cast<float*>(mix), T, nk, qkv_dim);
        gdn_pack_ab_k<<<(T * nv + 255) / 256, 256>>>(const_cast<float*>(aa), const_cast<float*>(bb), A_log,
                                                     dt_bias, T, nv);
        const int sm_split = (2 * 128) * static_cast<int>(sizeof(float));
        gdn_prefill_split2_k<<<dim3(nv, 4), 64, sm_split>>>(mix, aa, bb, S, A_log, dt_bias, o, T, nk, nv,
                                                            qkv_dim, 0);
    } else {
        int th = gdn_threads(dv);
        // float4 S path needs th<=dv/4, but warp_sum + nw=th/32 need a full warp.
        if (T > 1 && dk * dv <= 16384 && dv >= 128 && (dv % 4) == 0) th = dv / 4;
        gdn_prefill_steps_k<<<nv, th, sm>>>(mix, aa, bb, S, A_log, dt_bias, o, T, nk, nv, dk, dv, qkv_dim,
                                            1e-6f, qkv_raw, conv_w, conv_st, conv_k);
    }
}

int gemm_fp8_tile(int T, int n) {
    (void)n;
    int tile = 4096;
    while (T > 0 && static_cast<size_t>(T) * tile * sizeof(float) > 48 * 1024 && tile > 256) tile /= 2;
    return tile;
}

// One weight pass: K-outer 128-col tiles, T processed in TS-row smem panels.
// Y is accumulated in HBM so we never hold T accumulators in registers.
template <int TS>
__global__ void gemm_fp8_stream_k(const uint8_t* W, const float* scale, const float* X, float* Y, int m,
                                  int n, int T, int add) {
    constexpr int KN = 128;
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb_n = (n + 127) / 128;
    const int bi = row / 128;
    if (row < m && !add) {
        for (int t = 0; t < T; ++t) Y[static_cast<size_t>(t) * m + row] = 0.f;
    }
    __syncthreads();
    for (int t0 = 0; t0 < T; t0 += TS) {
        const int tt = T - t0 < TS ? T - t0 : TS;
        for (int k0 = 0; k0 < n; k0 += KN) {
            const int kn = n - k0 < KN ? n - k0 : KN;
            for (int i = threadIdx.x; i < tt * kn; i += blockDim.x) {
                const int t = i / kn;
                const int j = i - t * kn;
                xs[t * KN + j] = X[static_cast<size_t>(t0 + t) * n + k0 + j];
            }
            __syncthreads();
            if (row < m && scale && kn == KN) {
                const int b = k0 / 128;
                const float s = scale[bi * nb_n + b];
                const uint32_t p = __ldcs(fp8_blk4(W, row, m, b, lane));
                for (int t = 0; t < tt; ++t) {
                    float a = s * fp8x4_dot(p, xs + t * KN + lane * 4);
                    a = warp_sum(a);
                    if (lane == 0) Y[static_cast<size_t>(t0 + t) * m + row] += a;
                }
            }
            __syncthreads();
        }
    }
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
    if (T >= 2 && w1.rows < 128 && w2.rows < 128 && launch_fp8_skinny(w1, X, Y1, T, 0) &&
        launch_fp8_skinny(w2, X, Y2, T, 0)) {
        if (fuse_swiglu && Y1 && Y2)
            swiglu_n_k<<<(((w1.rows * T + 3) / 4) + 255) / 256, 256>>>(Y1, Y2, Y1, w1.rows * T);
        return;
    }
    if (T >= 128 && w1.rows >= 128 && w2.rows >= 128) {
        launch_linear_pair(w1, w2, X, Y1, Y2, T);
        if (fuse_swiglu && Y1 && Y2)
            swiglu_n_k<<<(((w1.rows * T + 3) / 4) + 255) / 256, 256>>>(Y1, Y2, Y1, w1.rows * T);
        return;
    }
    // T>=128: two BN=128 singles beat dual BN=64 (4 W passes, 256 threads).
    // Dual stays for leftover (m<128) and for 16<=T<128.
    if (g_fp8_e4_tc && T >= 16 && w1.fp8_kmajor && w2.fp8_kmajor &&
        w1.q == QuantKind::FP8_E4M3_B128 && w2.q == QuantKind::FP8_E4M3_B128 && w1.data && w2.data &&
        w1.scale && w2.scale && w1.cols == w2.cols && w1.cols > 0 && (w1.cols & 127) == 0 &&
        w1.rows >= 16 && w2.rows >= 16 && (T < 128 || w1.rows < 128)) {
        constexpr int BM = 128, WLD = 144, XLD = 144;
        const int BN = T >= 64 ? 64 : 32;
        const int m = w1.rows > w2.rows ? w1.rows : w2.rows;
        dim3 grid((m + BM - 1) / BM, (T + BN - 1) / BN);
        const size_t smem = static_cast<size_t>(2 * BM) * WLD + static_cast<size_t>(BN) * XLD;
        if (BN == 64)
            gemm_fp8_kmaj_e4_dual_k<64><<<grid, 256, smem>>>(w1.data, w1.scale, w2.data, w2.scale, X, Y1, Y2,
                                                             w1.rows, w2.rows, w1.cols, T);
        else
            gemm_fp8_kmaj_e4_dual_k<32><<<grid, 256, smem>>>(w1.data, w1.scale, w2.data, w2.scale, X, Y1, Y2,
                                                             w1.rows, w2.rows, w1.cols, T);
        if (fuse_swiglu && Y1 && Y2)
            swiglu_n_k<<<(((w1.rows * T + 3) / 4) + 255) / 256, 256>>>(Y1, Y2, Y1, w1.rows * T);
        return;
    }
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

    // Shared-tile TC: m=128 n=256 T=128 (covers BN=128 e4 path).
    {
        const int sm = 128, sn = 256, sT = 128;
        std::vector<uint8_t> sW(static_cast<size_t>(sm) * sn);
        std::vector<float> ssc((sm / 128) * (sn / 128), 0.05f);
        std::vector<float> sX(static_cast<size_t>(sT) * sn), sYcpu(static_cast<size_t>(sT) * sm),
            sYgpu(static_cast<size_t>(sT) * sm);
        for (size_t i = 0; i < sW.size(); ++i) sW[i] = static_cast<uint8_t>((i * 13 + 7) & 0x7f);
        for (size_t i = 0; i < sX.size(); ++i) sX[i] = 0.01f * static_cast<float>((static_cast<int>(i) % 11) - 5);
        for (int t = 0; t < sT; ++t)
            ops::gemv_fp8(sW.data(), ssc.data(), sX.data() + static_cast<size_t>(t) * sn,
                          sYcpu.data() + static_cast<size_t>(t) * sm, sm, sn, 128);
        std::vector<uint8_t> sK(static_cast<size_t>(sm) * fp8_pack_cols(sn));
        pack_fp8_kmajor_host(sK.data(), sW.data(), sm, sn);
        uint8_t* dWs = nullptr;
        float *dSs = nullptr, *dXs = nullptr, *dYs = nullptr;
        bool ok_alloc = cudaMalloc(&dWs, sK.size()) == cudaSuccess &&
                        cudaMalloc(&dSs, ssc.size() * 4) == cudaSuccess &&
                        cudaMalloc(&dXs, sX.size() * 4) == cudaSuccess &&
                        cudaMalloc(&dYs, sYgpu.size() * 4) == cudaSuccess;
        if (ok_alloc) {
            cudaMemcpy(dWs, sK.data(), sK.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(dSs, ssc.data(), ssc.size() * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dXs, sX.data(), sX.size() * 4, cudaMemcpyHostToDevice);
            cudaMemset(dYs, 0, sYgpu.size() * 4);
            GpuW sw;
            sw.data = dWs;
            sw.scale = dSs;
            sw.q = QuantKind::FP8_E4M3_B128;
            sw.rows = sm;
            sw.cols = sn;
            sw.fp8_kmajor = true;
            cudaGetLastError();
            g_fp8_shared_tc = true;
            const bool launched = launch_fp8_shared_tc(sw, dXs, dYs, sT, 0);
            cudaDeviceSynchronize();
            cudaMemcpy(sYgpu.data(), dYs, sYgpu.size() * 4, cudaMemcpyDeviceToHost);
            float maxe = 0.f;
            for (size_t i = 0; i < sYcpu.size(); ++i) maxe = std::max(maxe, std::fabs(sYcpu[i] - sYgpu[i]));
            const bool tc_ok = launched && maxe < 0.12f && cudaGetLastError() == cudaSuccess;
            g_fp8_shared_tc = tc_ok;
            std::fprintf(stderr, "fp8_shared_tc_err=%.4f ok=%d used=%d\n", maxe, tc_ok ? 1 : 0,
                         g_fp8_shared_tc ? 1 : 0);

            cudaMemset(dYs, 0, sYgpu.size() * 4);
            cudaGetLastError();
            g_fp8_e4_tc = true;
            const bool e4_launch = launch_fp8_e4_tc(sw, dXs, dYs, sT, 0);
            cudaDeviceSynchronize();
            cudaMemcpy(sYgpu.data(), dYs, sYgpu.size() * 4, cudaMemcpyDeviceToHost);
            float maxe4 = 0.f, maxa4 = 0.f;
            for (size_t i = 0; i < sYcpu.size(); ++i) {
                maxe4 = std::max(maxe4, std::fabs(sYcpu[i] - sYgpu[i]));
                maxa4 = std::max(maxa4, std::fabs(sYcpu[i]));
            }
            g_fp8_e4_tc = e4_launch && maxe4 < std::max(0.20f * maxa4, 0.15f) &&
                          cudaGetLastError() == cudaSuccess;
            std::fprintf(stderr, "fp8_e4_tc_err=%.4f maxa=%.4f ok=%d used=%d\n", maxe4, maxa4,
                         g_fp8_e4_tc ? 1 : 0, g_fp8_e4_tc ? 1 : 0);
            if (g_fp8_e4_tc && g_blas && sm >= 128) {
                uint8_t *dWr = nullptr, *dXe = nullptr;
                if (cudaMalloc(&dWr, static_cast<size_t>(sm) * sn) == cudaSuccess &&
                    cudaMalloc(&dXe, static_cast<size_t>(sT) * sn) == cudaSuccess) {
                    dim3 ug((sm + 127) / 128, sn / 128);
                    unpack_fp8_kmaj_e4_k<<<ug, 256>>>(dWs, dSs, dWr, sm, sn);
                    f32_to_e4_n_k<<<(sT * sn + 1023) / 1024, 256>>>(dXs, dXe, sT * sn);
                    cudaMemset(dYs, 0, sYgpu.size() * 4);
                    cudaGetLastError();
                    const float al = 1.f, be = 0.f;
                    cublasStatus_t cst = CUBLAS_STATUS_NOT_INITIALIZED;
                    if (g_lt) {
                        cublasLtMatmulDesc_t desc = nullptr;
                        cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
                        if (cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) ==
                            CUBLAS_STATUS_SUCCESS) {
                            cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
                            cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
                            cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
                            cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, sn, sm, sn);
                            cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, sn, sT, sn);
                            cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, sm, sT, sm);
                            cst = cublasLtMatmul(g_lt, desc, &al, dWr, la, dXe, lb, &be, dYs, lc, dYs, lc,
                                                 nullptr, nullptr, 0, cudaStreamPerThread);
                            if (la) cublasLtMatrixLayoutDestroy(la);
                            if (lb) cublasLtMatrixLayoutDestroy(lb);
                            if (lc) cublasLtMatrixLayoutDestroy(lc);
                            cublasLtMatmulDescDestroy(desc);
                        }
                    }
                    cudaDeviceSynchronize();
                    cudaMemcpy(sYgpu.data(), dYs, sYgpu.size() * 4, cudaMemcpyDeviceToHost);
                    float ec = 0.f, ac = 0.f;
                    for (size_t i = 0; i < sYcpu.size(); ++i) {
                        ec = std::max(ec, std::fabs(sYcpu[i] - sYgpu[i]));
                        ac = std::max(ac, std::fabs(sYcpu[i]));
                    }
                    g_cublas_fp8_e4 = cst == CUBLAS_STATUS_SUCCESS &&
                                      ec < std::max(0.25f * ac, 0.20f) &&
                                      cudaGetLastError() == cudaSuccess;
                    std::fprintf(stderr, "cublas_fp8_e4_err=%.4f maxa=%.4f st=%d ok=%d\n", ec, ac,
                                 static_cast<int>(cst), g_cublas_fp8_e4 ? 1 : 0);
                }
                if (dWr) cudaFree(dWr);
                if (dXe) cudaFree(dXe);
            }
            if (g_fp8_e4_tc) {
                cudaMemset(dYs, 0, sYgpu.size() * 4);
                cudaGetLastError();
                g_fp8_e4_bk512 = true;
                const bool lbk = launch_fp8_e4_tc(sw, dXs, dYs, sT, 0);
                cudaDeviceSynchronize();
                cudaMemcpy(sYgpu.data(), dYs, sYgpu.size() * 4, cudaMemcpyDeviceToHost);
                float ebk = 0.f, abk = 0.f;
                for (size_t i = 0; i < sYcpu.size(); ++i) {
                    ebk = std::max(ebk, std::fabs(sYcpu[i] - sYgpu[i]));
                    abk = std::max(abk, std::fabs(sYcpu[i]));
                }
                g_fp8_e4_bk512 = lbk && ebk < std::max(0.20f * abk, 0.15f) &&
                                 cudaGetLastError() == cudaSuccess;
                std::fprintf(stderr, "fp8_e4_bk512_err=%.4f maxa=%.4f ok=%d\n", ebk, abk,
                             g_fp8_e4_bk512 ? 1 : 0);
                cudaMemset(dYs, 0, sYgpu.size() * 4);
                cudaGetLastError();
                g_fp8_e4_occ = true;
                const bool locc = launch_fp8_e4_tc(sw, dXs, dYs, sT, 0);
                cudaDeviceSynchronize();
                cudaMemcpy(sYgpu.data(), dYs, sYgpu.size() * 4, cudaMemcpyDeviceToHost);
                float eocc = 0.f, aocc = 0.f;
                for (size_t i = 0; i < sYcpu.size(); ++i) {
                    eocc = std::max(eocc, std::fabs(sYcpu[i] - sYgpu[i]));
                    aocc = std::max(aocc, std::fabs(sYcpu[i]));
                }
                g_fp8_e4_occ = locc && eocc < std::max(0.20f * aocc, 0.15f) &&
                               cudaGetLastError() == cudaSuccess;
                std::fprintf(stderr, "fp8_e4_occ_err=%.4f maxa=%.4f ok=%d\n", eocc, aocc,
                             g_fp8_e4_occ ? 1 : 0);
                cudaMemset(dYs, 0, sYgpu.size() * 4);
                cudaGetLastError();
                g_fp8_e4_o2 = true;
                const bool lo2 = launch_fp8_e4_tc(sw, dXs, dYs, sT, 0);
                cudaDeviceSynchronize();
                cudaMemcpy(sYgpu.data(), dYs, sYgpu.size() * 4, cudaMemcpyDeviceToHost);
                float eo2e = 0.f, ao2 = 0.f;
                for (size_t i = 0; i < sYcpu.size(); ++i) {
                    eo2e = std::max(eo2e, std::fabs(sYcpu[i] - sYgpu[i]));
                    ao2 = std::max(ao2, std::fabs(sYcpu[i]));
                }
                g_fp8_e4_o2 = lo2 && eo2e < std::max(0.20f * ao2, 0.15f) &&
                              cudaGetLastError() == cudaSuccess;
                std::fprintf(stderr, "fp8_e4_o2_err=%.4f maxa=%.4f ok=%d\n", eo2e, ao2,
                             g_fp8_e4_o2 ? 1 : 0);
                if (g_fp8_e4_bk512) {
                    const int t2 = 256;
                    std::vector<float> x2(static_cast<size_t>(t2) * sn), y2c(static_cast<size_t>(t2) * sm),
                        y2g(static_cast<size_t>(t2) * sm);
                    for (size_t i = 0; i < x2.size(); ++i)
                        x2[i] = 0.01f * static_cast<float>((static_cast<int>(i) % 11) - 5);
                    for (int t = 0; t < t2; ++t)
                        ops::gemv_fp8(sW.data(), ssc.data(), x2.data() + static_cast<size_t>(t) * sn,
                                      y2c.data() + static_cast<size_t>(t) * sm, sm, sn, 128);
                    float *dX3 = nullptr, *dY3 = nullptr;
                    if (cudaMalloc(&dX3, x2.size() * 4) == cudaSuccess &&
                        cudaMalloc(&dY3, y2g.size() * 4) == cudaSuccess) {
                        cudaMemcpy(dX3, x2.data(), x2.size() * 4, cudaMemcpyHostToDevice);
                        cudaMemset(dY3, 0, y2g.size() * 4);
                        cudaGetLastError();
                        const bool l2 = launch_fp8_e4_tc(sw, dX3, dY3, t2, 0);
                        cudaDeviceSynchronize();
                        cudaMemcpy(y2g.data(), dY3, y2g.size() * 4, cudaMemcpyDeviceToHost);
                        float e2 = 0.f, a2 = 0.f;
                        for (size_t i = 0; i < y2c.size(); ++i) {
                            e2 = std::max(e2, std::fabs(y2c[i] - y2g[i]));
                            a2 = std::max(a2, std::fabs(y2c[i]));
                        }
                        const bool ok2 = l2 && e2 < std::max(0.20f * a2, 0.15f) &&
                                         cudaGetLastError() == cudaSuccess;
                        std::fprintf(stderr, "fp8_e4_bk512_t256_err=%.4f maxa=%.4f ok=%d\n", e2, a2,
                                     ok2 ? 1 : 0);
                        if (!ok2) g_fp8_e4_bk512 = false;
                    }
                    if (dX3) cudaFree(dX3);
                    if (dY3) cudaFree(dY3);
                    if (g_fp8_e4_bk512) {
                        std::vector<uint8_t> xe(static_cast<size_t>(sT) * sn);
                        for (size_t i = 0; i < xe.size(); ++i) xe[i] = ops::f32_to_e4m3(sX[i]);
                        uint8_t* dXe = nullptr;
                        if (cudaMalloc(&dXe, xe.size()) == cudaSuccess) {
                            cudaMemcpy(dXe, xe.data(), xe.size(), cudaMemcpyHostToDevice);
                            cudaMemset(dYs, 0, sYgpu.size() * 4);
                            cudaGetLastError();
                            g_xe = dXe;
                            g_xe_n = sn;
                            g_xe_T = sT;
                            g_fp8_e4_pipe = true;
                            const bool lp = launch_fp8_e4_tc(sw, dXs, dYs, sT, 0);
                            cudaDeviceSynchronize();
                            cudaMemcpy(sYgpu.data(), dYs, sYgpu.size() * 4, cudaMemcpyDeviceToHost);
                            float ep = 0.f, ap = 0.f;
                            for (size_t i = 0; i < sYcpu.size(); ++i) {
                                ep = std::max(ep, std::fabs(sYcpu[i] - sYgpu[i]));
                                ap = std::max(ap, std::fabs(sYcpu[i]));
                            }
                            g_fp8_e4_pipe = lp && ep < std::max(0.20f * ap, 0.15f) &&
                                            cudaGetLastError() == cudaSuccess;
                            std::fprintf(stderr, "fp8_e4_pipe_err=%.4f maxa=%.4f ok=%d\n", ep, ap,
                                         g_fp8_e4_pipe ? 1 : 0);
                            g_xe = nullptr;
                            g_xe_n = 0;
                            g_xe_T = 0;
                        }
                        if (dXe) cudaFree(dXe);
                    }
                }
            }
            if (g_fp8_e4_tc) {
                const int t256 = 512;
                std::vector<float> x256(static_cast<size_t>(t256) * sn), y256c(static_cast<size_t>(t256) * sm),
                    y256g(static_cast<size_t>(t256) * sm);
                for (size_t i = 0; i < x256.size(); ++i)
                    x256[i] = 0.01f * static_cast<float>((static_cast<int>(i) % 11) - 5);
                for (int t = 0; t < t256; ++t)
                    ops::gemv_fp8(sW.data(), ssc.data(), x256.data() + static_cast<size_t>(t) * sn,
                                  y256c.data() + static_cast<size_t>(t) * sm, sm, sn, 128);
                float *dX2 = nullptr, *dY2 = nullptr;
                if (cudaMalloc(&dX2, x256.size() * 4) == cudaSuccess &&
                    cudaMalloc(&dY2, y256g.size() * 4) == cudaSuccess) {
                    cudaMemcpy(dX2, x256.data(), x256.size() * 4, cudaMemcpyHostToDevice);
                    cudaMemset(dY2, 0, y256g.size() * 4);
                    cudaGetLastError();
                    g_fp8_e4_bn256 = true;
                    const bool l256 = launch_fp8_e4_tc(sw, dX2, dY2, t256, 0);
                    cudaDeviceSynchronize();
                    cudaMemcpy(y256g.data(), dY2, y256g.size() * 4, cudaMemcpyDeviceToHost);
                    float e256 = 0.f, a256 = 0.f;
                    for (size_t i = 0; i < y256c.size(); ++i) {
                        e256 = std::max(e256, std::fabs(y256c[i] - y256g[i]));
                        a256 = std::max(a256, std::fabs(y256c[i]));
                    }
                    g_fp8_e4_bn256 = l256 && e256 < std::max(0.20f * a256, 0.15f) &&
                                     cudaGetLastError() == cudaSuccess;
                    std::fprintf(stderr, "fp8_e4_bn256_err=%.4f maxa=%.4f ok=%d\n", e256, a256,
                                 g_fp8_e4_bn256 ? 1 : 0);
                }
                if (dX2) cudaFree(dX2);
                if (dY2) cudaFree(dY2);
            }
        } else {
            g_fp8_shared_tc = false;
            g_fp8_e4_tc = false;
            g_fp8_e4_bn256 = false;
            g_fp8_e4_bk512 = false;
            g_fp8_e4_occ = false;
            g_fp8_e4_o2 = false;
        }
        if (dWs) cudaFree(dWs);
        if (dSs) cudaFree(dSs);
        if (dXs) cudaFree(dXs);
        if (dYs) cudaFree(dYs);
    }
    return f16_ok;
}

static void host_q4k_scale_min(const uint8_t* sc, int j, int& s, int& mn) {
    if (j < 4) {
        s = sc[j] & 63;
        mn = sc[j + 4] & 63;
    } else {
        s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        mn = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
}

// Pre-expand K-quant scales; keep 4/5/6-bit quants packed (not FP8).
static void pack_q4k_soa(uint8_t* dst, const uint8_t* src, int rows, int cols) {
    const int nb = cols / 256;
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = src + (static_cast<size_t>(r) * nb + b) * kQ4KBsz;
            uint8_t* out = dst + (static_cast<size_t>(r) * nb + b) * kQ4KSoaBsz;
            __half dh, dm;
            std::memcpy(&dh, blk, 2);
            std::memcpy(&dm, blk + 2, 2);
            const float d = __half2float(dh);
            const float minv = __half2float(dm);
            const uint8_t* sc = blk + 4;
            __half* ds = reinterpret_cast<__half*>(out);
            __half* dmin = reinterpret_cast<__half*>(out + 16);
            for (int j = 0; j < 8; ++j) {
                int s = 0, mn = 0;
                host_q4k_scale_min(sc, j, s, mn);
                ds[j] = __float2half(d * static_cast<float>(s));
                dmin[j] = __float2half(minv * static_cast<float>(mn));
            }
            std::memcpy(out + 32, blk + 16, 128);
        }
    }
}

static void pack_q5k_soa(uint8_t* dst, const uint8_t* src, int rows, int cols) {
    const int nb = cols / 256;
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = src + (static_cast<size_t>(r) * nb + b) * kQ5KBsz;
            uint8_t* out = dst + (static_cast<size_t>(r) * nb + b) * kQ5KSoaBsz;
            __half dh, dm;
            std::memcpy(&dh, blk, 2);
            std::memcpy(&dm, blk + 2, 2);
            const float d = __half2float(dh);
            const float minv = __half2float(dm);
            const uint8_t* sc = blk + 4;
            __half* ds = reinterpret_cast<__half*>(out);
            __half* dmin = reinterpret_cast<__half*>(out + 16);
            for (int j = 0; j < 8; ++j) {
                int s = 0, mn = 0;
                host_q4k_scale_min(sc, j, s, mn);
                ds[j] = __float2half(d * static_cast<float>(s));
                dmin[j] = __float2half(minv * static_cast<float>(mn));
            }
            std::memcpy(out + 32, blk + 16, 32);  // qh
            std::memcpy(out + 64, blk + 48, 128); // ql
        }
    }
}

static void pack_q6k_soa(uint8_t* dst, const uint8_t* src, int rows, int cols) {
    const int nb = cols / 256;
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = src + (static_cast<size_t>(r) * nb + b) * kQ6KBsz;
            uint8_t* out = dst + (static_cast<size_t>(r) * nb + b) * kQ6KSoaBsz;
            const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
            __half dh;
            std::memcpy(&dh, blk + 208, 2);
            const float d = __half2float(dh);
            __half* ds = reinterpret_cast<__half*>(out);
            for (int j = 0; j < 16; ++j) ds[j] = __float2half(d * static_cast<float>(sc[j]));
            std::memcpy(out + 32, blk, 128);       // ql
            std::memcpy(out + 160, blk + 128, 64); // qh
        }
    }
}

void kquant_gemv_selftest() {
    const int m = 64, n = 512;
    std::vector<float> x(n), y_cpu(m), y_gpu(m);
    for (int i = 0; i < n; ++i) x[static_cast<size_t>(i)] = 0.02f * static_cast<float>((i % 17) - 8);
    auto run = [&](QuantKind q, int bsz, auto cpu_gemv, const char* name) {
        const size_t bytes = static_cast<size_t>(m) * (n / 256) * static_cast<size_t>(bsz);
        std::vector<uint8_t> W(bytes);
        for (size_t i = 0; i < bytes; ++i) W[i] = static_cast<uint8_t>((i * 17 + 11) & 0xff);
        // Force finite superblock scales so random bytes cannot become Inf/NaN.
        const int nb = n / 256;
        for (int r = 0; r < m; ++r) {
            for (int b = 0; b < nb; ++b) {
                uint8_t* blk = W.data() + (static_cast<size_t>(r) * nb + b) * static_cast<size_t>(bsz);
                const __half dh = __float2half(0.05f);
                const __half dm = __float2half(0.01f);
                if (q == QuantKind::Q6_K) {
                    std::memcpy(blk + 208, &dh, 2);
                    for (int s = 0; s < 16; ++s) blk[192 + s] = static_cast<uint8_t>(static_cast<int8_t>((s & 7) - 3));
                } else {
                    std::memcpy(blk, &dh, 2);
                    std::memcpy(blk + 2, &dm, 2);
                }
            }
        }
        cpu_gemv(W.data(), x.data(), y_cpu.data(), m, n);
        const int soa_bsz = q == QuantKind::Q4_K ? kQ4KSoaBsz : (q == QuantKind::Q5_K ? kQ5KSoaBsz : kQ6KSoaBsz);
        const size_t soa_bytes = static_cast<size_t>(m) * (n / 256) * static_cast<size_t>(soa_bsz);
        std::vector<uint8_t> Soa(soa_bytes);
        if (q == QuantKind::Q4_K) pack_q4k_soa(Soa.data(), W.data(), m, n);
        else if (q == QuantKind::Q5_K) pack_q5k_soa(Soa.data(), W.data(), m, n);
        else pack_q6k_soa(Soa.data(), W.data(), m, n);
        uint8_t* dW = nullptr;
        float *dX = nullptr, *dY = nullptr;
        CUDA_CHECK(cudaMalloc(&dW, soa_bytes));
        CUDA_CHECK(cudaMalloc(&dX, sizeof(float) * n));
        CUDA_CHECK(cudaMalloc(&dY, sizeof(float) * m));
        CUDA_CHECK(cudaMemcpy(dW, Soa.data(), soa_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dX, x.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dY, 0, sizeof(float) * m));
        GpuW tw;
        tw.data = dW;
        tw.q = q;
        tw.rows = m;
        tw.cols = n;
        tw.qk_soa = true;
        launch_gemv(tw, dX, dY, 0);
        CUDA_CHECK(cudaMemcpy(y_gpu.data(), dY, sizeof(float) * m, cudaMemcpyDeviceToHost));
        float maxe = 0.f, maxa = 0.f;
        for (int i = 0; i < m; ++i) {
            maxe = std::max(maxe, std::fabs(y_cpu[i] - y_gpu[i]));
            maxa = std::max(maxa, std::fabs(y_cpu[i]));
        }
        cudaFree(dW);
        cudaFree(dX);
        cudaFree(dY);
        const bool ok = maxe < std::max(0.02f * maxa, 1e-3f) && cudaGetLastError() == cudaSuccess;
        std::fprintf(stderr, "native_%s_gemv_err=%.5f maxa=%.4f ok=%d\n", name, maxe, maxa, ok ? 1 : 0);
        if (!ok) throw std::runtime_error(std::string("native ") + name + " GEMV selftest failed");
    };
    run(QuantKind::Q4_K, 144, ops::gemv_q4_k, "q4k");
    run(QuantKind::Q5_K, 176, ops::gemv_q5_k, "q5k");
    run(QuantKind::Q6_K, 210, ops::gemv_q6_k, "q6k");
}

// Store compact KV for T tokens then attend the last token via tq (kv_mode=2).
// Drives the pos>=win handoff that 262k short benches never reach.
void tq_kv_selftest() {
    const int T = 8, n_q = 2, n_kv = 1, hd = 256, kn = n_kv * hd;
    const int nblk = hd / kTqBlk;
    std::vector<float> hK(static_cast<size_t>(T) * kn), hV(static_cast<size_t>(T) * kn),
        hQ(static_cast<size_t>(n_q) * hd), hOf(static_cast<size_t>(n_q) * hd),
        hOt(static_cast<size_t>(n_q) * hd);
    for (int t = 0; t < T; ++t) {
        for (int d = 0; d < hd; ++d) {
            hK[static_cast<size_t>(t) * kn + d] = 0.02f * static_cast<float>(((t * 17 + d) % 13) - 6);
            hV[static_cast<size_t>(t) * kn + d] = 0.03f * static_cast<float>(((t * 11 + d) % 9) - 4);
        }
    }
    for (int d = 0; d < n_q * hd; ++d) hQ[static_cast<size_t>(d)] = hK[static_cast<size_t>(T - 1) * kn + (d % hd)];
    float *dK = nullptr, *dV = nullptr, *dQf = nullptr, *dQt = nullptr, *dOf = nullptr, *dOt = nullptr;
    __half *dKf = nullptr, *dVf = nullptr, *dKsc = nullptr, *dVsc = nullptr;
    int8_t* dKq = nullptr;
    uint8_t* dVq = nullptr;
    int* dPos = nullptr;
    const size_t tok_h = static_cast<size_t>(T) * n_kv;
    CUDA_CHECK(cudaMalloc(&dK, hK.size() * 4));
    CUDA_CHECK(cudaMalloc(&dV, hV.size() * 4));
    CUDA_CHECK(cudaMalloc(&dQf, hQ.size() * 4));
    CUDA_CHECK(cudaMalloc(&dQt, hQ.size() * 4));
    CUDA_CHECK(cudaMalloc(&dOf, hOf.size() * 4));
    CUDA_CHECK(cudaMalloc(&dOt, hOt.size() * 4));
    CUDA_CHECK(cudaMalloc(&dKf, hK.size() * 2));
    CUDA_CHECK(cudaMalloc(&dVf, hV.size() * 2));
    CUDA_CHECK(cudaMalloc(&dKq, tok_h * hd));
    CUDA_CHECK(cudaMalloc(&dKsc, tok_h * 2));
    CUDA_CHECK(cudaMalloc(&dVq, tok_h * nblk * kTq3B));
    CUDA_CHECK(cudaMalloc(&dVsc, tok_h * nblk * 2));
    CUDA_CHECK(cudaMalloc(&dPos, 4));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), hK.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), hV.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dQf, hQ.data(), hQ.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dQt, hQ.data(), hQ.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dKq, 0, tok_h * hd));
    CUDA_CHECK(cudaMemset(dVq, 0, tok_h * nblk * kTq3B));
    const int pos = T - 1;
    CUDA_CHECK(cudaMemcpy(dPos, &pos, 4, cudaMemcpyHostToDevice));
    store_kv_batch_h_k<<<dim3((kn + 127) / 128, T), 128>>>(dKf, dK, 0, T, kn);
    store_kv_batch_h_k<<<dim3((kn + 127) / 128, T), 128>>>(dVf, dV, 0, T, kn);
    store_k_q8_th_k<<<dim3(n_kv, T), 256>>>(dKq, dKsc, dK, 0, T, n_kv, hd);
    store_v_tq3_th_k<<<dim3(nblk, n_kv, T), 128, kTqBlk * sizeof(float)>>>(dVq, dVsc, dV, 0, T, n_kv, hd);
    const size_t sm_on = static_cast<size_t>(hd) * sizeof(float);
    qk_attn_decode_k<<<n_q, 256, sm_on>>>(dQf, dK + static_cast<size_t>(pos) * kn, dV + static_cast<size_t>(pos) * kn,
                                          nullptr, nullptr, reinterpret_cast<float*>(dKf),
                                          reinterpret_cast<float*>(dVf), dOf, dPos, n_q, n_kv, hd, 0, 1e7f,
                                          1e-6f, 16384, 1, nullptr, nullptr, nullptr, nullptr);
    qk_attn_decode_k<<<n_q, 256, sm_on>>>(
        dQt, dK + static_cast<size_t>(pos) * kn, dV + static_cast<size_t>(pos) * kn, nullptr, nullptr,
        reinterpret_cast<float*>(dKf), reinterpret_cast<float*>(dVf), dOt, dPos, n_q, n_kv, hd, 0, 1e7f, 1e-6f,
        16384, 3, dKq, dKsc, dVq, dVsc);
    qk_attn_decode_tq_gqa_k<<<n_kv, 256>>>(dQt, dOt, dPos, n_q, n_kv, hd, dKq, dKsc, dVq, dVsc);
    CUDA_CHECK(cudaMemcpy(hOf.data(), dOf, hOf.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hOt.data(), dOt, hOt.size() * 4, cudaMemcpyDeviceToHost));
    float dot = 0.f, na = 0.f, nb = 0.f, maxa = 0.f;
    bool finite = true;
    for (size_t i = 0; i < hOf.size(); ++i) {
        finite = finite && std::isfinite(hOf[i]) && std::isfinite(hOt[i]);
        dot += hOf[i] * hOt[i];
        na += hOf[i] * hOf[i];
        nb += hOt[i] * hOt[i];
        maxa = std::max(maxa, std::fabs(hOf[i]));
    }
    const float cos = (na > 0.f && nb > 0.f) ? dot / std::sqrt(na * nb) : 0.f;
    __half *dKd = nullptr, *dVd = nullptr;
    CUDA_CHECK(cudaMalloc(&dKd, hK.size() * 2));
    CUDA_CHECK(cudaMalloc(&dVd, hV.size() * 2));
    dequant_k_q8_h_k<<<dim3(n_kv, T), 256>>>(dKd, dKq, dKsc, 0, T, n_kv, hd);
    dequant_v_tq3_h_k<<<dim3(nblk, n_kv, T), 128, kTqBlk * sizeof(float)>>>(dVd, dVq, dVsc, 0, T, n_kv, hd);
    std::vector<__half> hKd(hK.size());
    CUDA_CHECK(cudaMemcpy(hKd.data(), dKd, hK.size() * 2, cudaMemcpyDeviceToHost));
    float kmaxe = 0.f, kmaxa = 0.f;
    // Skip the last token: decode re-stores tq at pos after RMS.
    for (int t = 0; t < T - 1; ++t) {
        for (int d = 0; d < hd; ++d) {
            const size_t i = static_cast<size_t>(t) * kn + d;
            const float dk = __half2float(hKd[i]);
            kmaxe = std::max(kmaxe, std::fabs(dk - hK[i]));
            kmaxa = std::max(kmaxa, std::fabs(hK[i]));
        }
    }
    float *dOp = nullptr;
    CUDA_CHECK(cudaMalloc(&dOp, static_cast<size_t>(T) * n_q * hd * 4));
    attn_prefill_tq_gqa_k<<<dim3(n_kv, T), 256>>>(dQf, dOp, 0, T, n_q, n_kv, hd, dKq, dKsc, dVq, dVsc);
    std::vector<float> hOp(static_cast<size_t>(T) * n_q * hd);
    CUDA_CHECK(cudaMemcpy(hOp.data(), dOp, hOp.size() * 4, cudaMemcpyDeviceToHost));
    bool pfin = true;
    for (float v : hOp) pfin = pfin && std::isfinite(v);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dQf);
    cudaFree(dQt);
    cudaFree(dOf);
    cudaFree(dOt);
    cudaFree(dKf);
    cudaFree(dVf);
    cudaFree(dKq);
    cudaFree(dKsc);
    cudaFree(dVq);
    cudaFree(dVsc);
    cudaFree(dPos);
    cudaFree(dKd);
    cudaFree(dVd);
    cudaFree(dOp);
    const bool ok = finite && pfin && cos > 0.65f && kmaxe < std::max(0.05f * kmaxa, 1e-3f) &&
                    cudaGetLastError() == cudaSuccess;
    std::fprintf(stderr, "tq_attend_cos=%.4f tq_prefill_finite=%d k_q8_err=%.5f maxa=%.4f ok=%d\n", cos,
                 pfin ? 1 : 0, kmaxe, maxa, ok ? 1 : 0);
    if (!ok) throw std::runtime_error("tq store+attend selftest failed");
}

void launch_gemm_qk(const GpuW& w, const float* X, float* Y, int T, int add) {
    int blocks = 0, threads = 0;
    gemv_grid(w, blocks, threads);
    const size_t smem = sizeof(float) * static_cast<size_t>(T) * 256;
    if (w.q == QuantKind::Q4_K) {
        if (w.qk_soa) gemm_qk_t_k<kQ4KSoaBsz><<<blocks, threads, smem>>>(w.data, X, Y, w.rows, w.cols, T, add);
        else gemm_qk_t_k<kQ4KBsz><<<blocks, threads, smem>>>(w.data, X, Y, w.rows, w.cols, T, add);
    } else if (w.q == QuantKind::Q5_K) {
        if (w.qk_soa) gemm_qk_t_k<kQ5KSoaBsz><<<blocks, threads, smem>>>(w.data, X, Y, w.rows, w.cols, T, add);
        else gemm_qk_t_k<kQ5KBsz><<<blocks, threads, smem>>>(w.data, X, Y, w.rows, w.cols, T, add);
    } else {
        if (w.qk_soa) gemm_qk_t_k<kQ6KSoaBsz><<<blocks, threads, smem>>>(w.data, X, Y, w.rows, w.cols, T, add);
        else gemm_qk_t_k<kQ6KBsz><<<blocks, threads, smem>>>(w.data, X, Y, w.rows, w.cols, T, add);
    }
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

template <int TN>
__global__ void gemm_f16_row_k(const __half* W, const float* X, float* Y, int m, int n, int T, int add) {
    constexpr int KN = 128;
    extern __shared__ float xs[];
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (row >= m) return;
    const __half* wr = W + static_cast<size_t>(row) * n;
    for (int t0 = 0; t0 < T; t0 += TN) {
        const int tt = T - t0 < TN ? T - t0 : TN;
        float acc[TN];
#pragma unroll
        for (int t = 0; t < TN; ++t) acc[t] = 0.f;
        for (int k0 = 0; k0 < n; k0 += KN) {
            const int kn = n - k0 < KN ? n - k0 : KN;
            for (int i = lane; i < tt * kn; i += 32) {
                const int t = i / kn;
                const int c = i - t * kn;
                xs[t * KN + c] = X[static_cast<size_t>(t0 + t) * n + k0 + c];
            }
            __syncwarp();
#pragma unroll
            for (int t = 0; t < TN; ++t) {
                if (t >= tt) continue;
                float a = 0.f;
                for (int j = lane * 2; j + 1 < kn; j += 64) {
                    const float2 h = __half22float2(__ldg(reinterpret_cast<const __half2*>(wr + k0 + j)));
                    a += h.x * xs[t * KN + j] + h.y * xs[t * KN + j + 1];
                }
                acc[t] += a;
            }
            __syncwarp();
        }
#pragma unroll
        for (int t = 0; t < TN; ++t) {
            if (t < tt) {
                const float a = warp_sum(acc[t]);
                if (lane == 0) write_y(Y + static_cast<size_t>(t0 + t) * m, row, a, add);
            }
        }
    }
}

bool launch_cublas_f32(const GpuW& w, const float* X, float* Y, int T, int add) {
    if (!g_blas || T < 16 || w.q != QuantKind::F32 || !w.data) return false;
    const float* beta = add ? &g_lt_hs.beta1 : &g_lt_hs.beta0;
    cublasSetStream(g_blas, gemm_stream());
    return cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, w.rows, T, w.cols, &g_lt_hs.alpha,
                       reinterpret_cast<const float*>(w.data), w.cols, X, w.cols, beta, Y, w.rows) ==
           CUBLAS_STATUS_SUCCESS;
}

void launch_linear(const GpuW& w, const float* X, float* Y, int T, int add) {
    if (!w.data || w.rows <= 0 || w.cols <= 0 || T <= 0) return;
    if (w.q == QuantKind::F32 || w.q == QuantKind::F16 || w.q == QuantKind::BF16) {
        if (T >= 16 && w.q == QuantKind::F32 && launch_cublas_f32(w, X, Y, T, add)) return;
        if (!add && (w.q == QuantKind::F16 || w.q == QuantKind::BF16) && launch_cublas_f16(w, X, Y, T)) return;
        if (T == 1) {
            launch_gemv(w, X, Y, add);
            return;
        }
        for (int t = 0; t < T; ++t)
            launch_gemv(w, X + static_cast<size_t>(t) * w.cols, Y + static_cast<size_t>(t) * w.rows, add);
        return;
    }
    if (w.q == QuantKind::FP8_E4M3_B128 && T > 1) {
        // Row-major (scales absorbed): cublasLt on persistent W. T<16 stays
        // per-token rm GEMV so T=2/3/4 graphs never see cublasLt setup.
        if (w.fp8_rowmaj) {
            if (T >= 16 && launch_cublas_fp8_e4(w, X, Y, T, add)) return;
            for (int t = 0; t < T; ++t)
                launch_gemv(w, X + static_cast<size_t>(t) * w.cols, Y + static_cast<size_t>(t) * w.rows, add);
            return;
        }
        if (T >= 128 && launch_cublas_fp8_e4(w, X, Y, T, add)) return;
        if (launch_fp8_e4_tc(w, X, Y, T, add)) return;
        if (launch_fp8_shared_tc(w, X, Y, T, add)) return;
        if (launch_cublas_fp8(w, X, Y, T, add)) return;
        if (!add && w.fp8_tiled && g_fp8_e4m3_mma) {
            for (int t = 0; t < T; ++t)
                launch_fp8_e4m3_mma(w, X + static_cast<size_t>(t) * w.cols, Y + static_cast<size_t>(t) * w.rows);
            return;
        }
        int t = 0;
        while (t < T) {
            const int rem = T - t;
            const int tt = rem >= 8 ? 8 : (rem > 4 ? 4 : rem);
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
    if ((w.q == QuantKind::Q4_K || w.q == QuantKind::Q5_K || w.q == QuantKind::Q6_K) && T > 1) {
        int t = 0;
        while (t < T) {
            const int tt = T - t > 16 ? 16 : T - t;
            if (tt == 1)
                launch_gemv(w, X + static_cast<size_t>(t) * w.cols, Y + static_cast<size_t>(t) * w.rows, add);
            else
                launch_gemm_qk(w, X + static_cast<size_t>(t) * w.cols, Y + static_cast<size_t>(t) * w.rows, tt,
                               add);
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
        if (cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) g_lt = nullptr;
        if (cublasLtCreate(&g_lt2) != CUBLAS_STATUS_SUCCESS) g_lt2 = nullptr;
        cudaFuncSetCacheConfig(gdn_prefill_split_k, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(gdn_prefill_split2_k, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(attn_prefill_qtile_k<8>, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(attn_prefill_tq_k, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(attn_prefill_tq_gqa_k, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(qk_attn_decode_tq_gqa_k, cudaFuncCachePreferL1);
        if (g_blas) {
            cublasSetMathMode(g_blas, CUBLAS_TENSOR_OP_MATH);
            cublasSetStream(g_blas, cudaStreamPerThread);
        }
        {
            const cudaError_t s32 = cudaFuncSetAttribute(
                gemm_fp8_shared_tc_k<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
            if (s32 != cudaSuccess) cudaGetLastError();
            const cudaError_t s64 = cudaFuncSetAttribute(
                gemm_fp8_shared_tc_k<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
            if (s64 != cudaSuccess) cudaGetLastError();
            const cudaError_t e32 = cudaFuncSetAttribute(
                gemm_fp8_kmaj_e4_k<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);
            if (e32 != cudaSuccess) cudaGetLastError();
            const cudaError_t e64 = cudaFuncSetAttribute(
                gemm_fp8_kmaj_e4_k<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);
            if (e64 != cudaSuccess) cudaGetLastError();
            const cudaError_t e128 = cudaFuncSetAttribute(
                gemm_fp8_kmaj_e4_bn128_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);
            if (e128 != cudaSuccess) cudaGetLastError();
            const cudaError_t e256 = cudaFuncSetAttribute(
                gemm_fp8_kmaj_e4_bn256_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
            if (e256 != cudaSuccess) cudaGetLastError();
            const cudaError_t ebk = cudaFuncSetAttribute(
                gemm_fp8_kmaj_e4_bk512_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
            if (ebk != cudaSuccess) cudaGetLastError();
            const cudaError_t ebx = cudaFuncSetAttribute(
                gemm_fp8_kmaj_e4_bk512x_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
            if (ebx != cudaSuccess) cudaGetLastError();
            const cudaError_t eocc = cudaFuncSetAttribute(
                gemm_fp8_kmaj_e4_occ_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);
            if (eocc != cudaSuccess) cudaGetLastError();
            const cudaError_t eo2 = cudaFuncSetAttribute(
                gemm_fp8_kmaj_e4_o2_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);
            if (eo2 != cudaSuccess) cudaGetLastError();
            const cudaError_t ebp = cudaFuncSetAttribute(
                gemm_fp8_kmaj_e4_bk512p_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
            if (ebp != cudaSuccess) cudaGetLastError();
            const cudaError_t d32 = cudaFuncSetAttribute(
                gemm_fp8_kmaj_e4_dual_k<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
            if (d32 != cudaSuccess) cudaGetLastError();
            const cudaError_t d64 = cudaFuncSetAttribute(
                gemm_fp8_kmaj_e4_dual_k<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
            if (d64 != cudaSuccess) cudaGetLastError();
            const cudaError_t ys = cudaFuncSetAttribute(
                gemm_fp8_e4_yst_k<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
            if (ys != cudaSuccess) cudaGetLastError();
        }
        g_fp8_tc = fp8_tc_selftest();
        kquant_gemv_selftest();
        tq_kv_selftest();
        std::fprintf(stderr, "fp8_tensor_core=%d e4_tc=%d e4_bk512=%d cublas_fp8_e4=%d\n",
                     g_fp8_tc ? 1 : 0, g_fp8_e4_tc ? 1 : 0, g_fp8_e4_bk512 ? 1 : 0,
                     g_cublas_fp8_e4 ? 1 : 0);
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
            const cudaError_t mt8 =
                cudaFuncSetAttribute(gemm_fp8_mt_k<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 1024);
            if (mt8 != cudaSuccess) cudaGetLastError();
            const cudaError_t mt16 =
                cudaFuncSetAttribute(gemm_fp8_mt_k<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 1024);
            if (mt16 != cudaSuccess) cudaGetLastError();
            const cudaError_t mt32 =
                cudaFuncSetAttribute(gemm_fp8_mt_k<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 1024);
            if (mt32 != cudaSuccess) cudaGetLastError();
            const cudaError_t st64 =
                cudaFuncSetAttribute(gemm_fp8_stream_k<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 1024);
            if (st64 != cudaSuccess) cudaGetLastError();
            const cudaError_t tck =
                cudaFuncSetAttribute(gemm_fp8_kmaj_tc_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 1024);
            if (tck != cudaSuccess) cudaGetLastError();
            cudaError_t xe;
            xe = cudaFuncSetAttribute(gemv_fp8_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_fp8_2row_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_fp8_rm_2row_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
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
            xe = cudaFuncSetAttribute(gemv_q8_soa_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_q8_soa_2row_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_k<kQ4KBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_2row_k<kQ4KBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_k<kQ5KBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_2row_k<kQ5KBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_k<kQ6KBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_2row_k<kQ6KBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_k<kQ4KSoaBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_k<kQ5KSoaBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_k<kQ6KSoaBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
        }
        if (cudaStreamCreateWithFlags(&bak_stream_, cudaStreamNonBlocking) != cudaSuccess) bak_stream_ = nullptr;
        g_bak_stream = bak_stream_;
        build();
    }
    ~EngineImpl() override {
        release();
        g_xf16 = nullptr;
        g_xf16_n = 0;
        g_xf16_src = nullptr;
        g_xf16_src_n = 0;
        g_wf16 = nullptr;
        g_wf16_n = 0;
        if (g_blas) {
            cublasDestroy(g_blas);
            g_blas = nullptr;
        }
        if (g_lt2) {
            cublasLtDestroy(g_lt2);
            g_lt2 = nullptr;
        }
        if (g_lt) {
            cublasLtDestroy(g_lt);
            g_lt = nullptr;
        }
    }

    void set_vision_override(const float* host, int n, int offset) override {
        vis_n_ = 0;
        vis_off_ = -1;
        if (!host || n <= 0 || offset < 0) return;
        const size_t bytes = sizeof(float) * static_cast<size_t>(n) * hidden_;
        if (bytes > vis_bytes_) {
            d_vis_ = static_cast<float*>(alloc(bytes));
            vis_bytes_ = bytes;
        }
        CUDA_CHECK(cudaMemcpy(d_vis_, host, bytes, cudaMemcpyHostToDevice));
        vis_n_ = n;
        vis_off_ = offset;
    }

    void prefill(const int32_t* ids, int n) override {
        CUDA_CHECK(cudaMemset(d_S_, 0, s_bytes_));
        CUDA_CHECK(cudaMemset(d_conv_, 0, conv_bytes_));
        if (ctx_ <= 1024 && kv_bytes_ > 0) {
            CUDA_CHECK(cudaMemset(d_kcache_, 0, kv_bytes_));
            CUDA_CHECK(cudaMemset(d_vcache_, 0, kv_bytes_));
        }
        // Compact persist is write-before-read per slot; a 262k memset is seconds of tax.
        // Attn only reads KV slots this prefill / later decode writes.
        // A full-window K/V memset is ~20 GiB at ctx=163k and is not needed.
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_pos_, &zero, 4, cudaMemcpyHostToDevice));
        pos_ = 0;
        const bool use_vis = vis_n_ > 0 && vis_off_ >= 0;
        if (!use_vis && n == 256 && pf256_exec_) {
            CUDA_CHECK(cudaMemcpy(d_toks_, ids, sizeof(int) * n, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaGraphLaunch(pf256_exec_, cudaStreamPerThread));
            pos_ = n;
            if (n > 0) CUDA_CHECK(cudaMemcpy(d_tok_, ids + (n - 1), 4, cudaMemcpyHostToDevice));
            snap_last_residual(n);
        } else if (!use_vis && n >= 2 && n <= kPfGraphMax && pf_graph_execs_[n]) {
            CUDA_CHECK(cudaMemcpy(d_toks_, ids, sizeof(int) * n, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaGraphLaunch(pf_graph_execs_[n], cudaStreamPerThread));
            pos_ = n;
            if (n > 0) CUDA_CHECK(cudaMemcpy(d_tok_, ids + (n - 1), 4, cudaMemcpyHostToDevice));
            snap_last_residual(n);
        } else if (!use_vis && (n <= 1 || pf_cap_ <= 1)) {
            for (int t = 0; t < n; ++t) decode_token(ids[t]);
        } else {
            launch_prefill(ids, n);
            if (!use_vis && !skip_pf_graph_) maybe_capture_prefill(n);
        }
        if (!graph_exec_ && vocab_ > 256) {
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
        if (graph_exec_ && pos_ > 0 && !(kv_tq_ && pos_ >= kv_win_)) {
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
            if (graph_exec_ && pos_ > 0 && !(kv_tq_ && pos_ >= kv_win_))
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

    void copy_last_hidden(float* host) override {
        if (!host || hidden_ <= 0 || !d_h_ || !d_final_norm_) return;
        launch_rms(d_h_, d_final_norm_, d_xn_, hidden_, store_->model().rms_eps);
        CUDA_CHECK(cudaMemcpy(host, d_xn_, sizeof(float) * static_cast<size_t>(hidden_), cudaMemcpyDeviceToHost));
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
        const int nblk = std::max(1, max_hd_ / kTqBlk);
        const size_t tq_tok = static_cast<size_t>(n_attn_) * ctx_ * static_cast<size_t>(std::max(max_nkv_, 1));
        const size_t kq1 = tq_tok * static_cast<size_t>(std::max(max_hd_, 1));
        const size_t ksc1 = tq_tok * sizeof(__half);
        const size_t vq1 = tq_tok * static_cast<size_t>(nblk) * kTq3B;
        const size_t vsc1 = tq_tok * static_cast<size_t>(nblk) * sizeof(__half);
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
            if (kv_tq_ && d_k_q8_ && d_k_sc_ && d_v_qs_ && d_v_sc_) {
                CUDA_CHECK(cudaMemcpy(d_k_q8_ + kq1 * b, d_k_q8_, kq1, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<uint8_t*>(d_k_sc_) + ksc1 * b, d_k_sc_, ksc1,
                                      cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_v_qs_ + vq1 * b, d_v_qs_, vq1, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<uint8_t*>(d_v_sc_) + vsc1 * b, d_v_sc_, vsc1,
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
            if (kv_bytes_ && d_k_bak_ && d_v_bak_) {
                CUDA_CHECK(cudaMemcpyAsync(d_k_bak_, d_kcache_, kv_bytes_, cudaMemcpyDeviceToDevice, bak_stream_));
                CUDA_CHECK(cudaMemcpyAsync(d_v_bak_, d_vcache_, kv_bytes_, cudaMemcpyDeviceToDevice, bak_stream_));
            }
        } else {
            if (s_bytes_) CUDA_CHECK(cudaMemcpy(d_S_bak_, d_S_, s_bytes_, cudaMemcpyDeviceToDevice));
            if (conv_bytes_) CUDA_CHECK(cudaMemcpy(d_conv_bak_, d_conv_, conv_bytes_, cudaMemcpyDeviceToDevice));
            if (kv_bytes_ && d_k_bak_ && d_v_bak_) {
                CUDA_CHECK(cudaMemcpy(d_k_bak_, d_kcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_v_bak_, d_vcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
            }
        }
        if (bak_stream_) CUDA_CHECK(cudaStreamSynchronize(bak_stream_));
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
            if (kv_bytes_ && d_k_bak_ && d_v_bak_) {
                CUDA_CHECK(cudaMemcpy(d_kcache_, d_k_bak_, kv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_vcache_, d_v_bak_, kv_bytes_, cudaMemcpyDeviceToDevice));
            }
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
        // Skinny leftover (m=48 A/B) stays BF16 on official FP8 checkpoints.
        // Prefill T=256/1024 was 256× F32 GEMV (~188/750 ms). Promote to F32
        // once so cublasSgemm can run; T=1 leftover k-major kernels unused here.
        if (t.quant == QuantKind::BF16 && rows > 0 && rows < 128 && cols > 0 &&
            t.data.size() >= static_cast<size_t>(rows) * static_cast<size_t>(cols) * 2) {
            std::vector<float> f(static_cast<size_t>(rows) * static_cast<size_t>(cols));
            const uint16_t* src = reinterpret_cast<const uint16_t*>(t.data.data());
            for (size_t i = 0; i < f.size(); ++i) {
                uint32_t u = static_cast<uint32_t>(src[i]) << 16;
                std::memcpy(&f[i], &u, 4);
            }
            void* p = alloc(f.size() * sizeof(float));
            CUDA_CHECK(cudaMemcpy(p, f.data(), f.size() * sizeof(float), cudaMemcpyHostToDevice));
            w.q = QuantKind::F32;
            w.data = static_cast<const uint8_t*>(p);
            return w;
        }
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
            static int q8once = 0;
            if (!q8once) {
                q8once = 1;
                std::fprintf(stderr, "native_q8=1 (packed SoA GEMV)\n");
            }
            return w;
        }
        if (t.quant == QuantKind::Q4_K || t.quant == QuantKind::Q5_K || t.quant == QuantKind::Q6_K) {
            if (cols <= 0 || (cols % 256) != 0)
                throw std::runtime_error("K-quant cols must be multiple of 256: " + t.ir_name);
            const int bsz = t.quant == QuantKind::Q4_K ? 144 : (t.quant == QuantKind::Q5_K ? 176 : 210);
            const size_t need = static_cast<size_t>(rows) * static_cast<size_t>(cols / 256) * static_cast<size_t>(bsz);
            if (t.data.size() < need)
                throw std::runtime_error("K-quant tensor short: " + t.ir_name);
            // Default: keep packed K-quants and run native GEMV. Requant-to-FP8
            // is opt-in via RAPIDLLM_REQUANT_KQUANT=1.
            const char* requant = std::getenv("RAPIDLLM_REQUANT_KQUANT");
            if (!(requant && requant[0] == '1')) {
                const int soa_bsz = t.quant == QuantKind::Q4_K ? kQ4KSoaBsz
                                                               : (t.quant == QuantKind::Q5_K ? kQ5KSoaBsz : kQ6KSoaBsz);
                const size_t soa_n = static_cast<size_t>(rows) * static_cast<size_t>(cols / 256) *
                                     static_cast<size_t>(soa_bsz);
                std::vector<uint8_t> soa(soa_n);
                if (t.quant == QuantKind::Q4_K) pack_q4k_soa(soa.data(), t.data.data(), rows, cols);
                else if (t.quant == QuantKind::Q5_K) pack_q5k_soa(soa.data(), t.data.data(), rows, cols);
                else pack_q6k_soa(soa.data(), t.data.data(), rows, cols);
                void* p = alloc(soa_n);
                CUDA_CHECK(cudaMemcpy(p, soa.data(), soa_n, cudaMemcpyHostToDevice));
                w.q = t.quant;
                w.data = static_cast<const uint8_t*>(p);
                w.scale = nullptr;
                w.qk_soa = true;
                static int seen[16] = {};
                const int qi = static_cast<int>(t.quant);
                if (qi >= 0 && qi < 16 && !seen[qi]) {
                    seen[qi] = 1;
                    const char* nm = t.quant == QuantKind::Q4_K ? "q4k"
                                     : t.quant == QuantKind::Q5_K ? "q5k"
                                                                  : "q6k";
                    std::fprintf(stderr, "native_kquant=1 %s soa=1 (no fp8 requant)\n", nm);
                }
                return w;
            }
            static int ronce = 0;
            if (!ronce) {
                ronce = 1;
                std::fprintf(stderr, "requant_fp8=1 q=%d (RAPIDLLM_REQUANT_KQUANT=1)\n", static_cast<int>(t.quant));
            }
            const int nblk = cols / 256;
            const size_t rowb = static_cast<size_t>(nblk) * static_cast<size_t>(bsz);
            const int br = 128, bc = 128;
            const int nb_r = (rows + br - 1) / br;
            const int nb_c = cols / bc;
            std::vector<uint8_t> q(static_cast<size_t>(rows) * static_cast<size_t>(cols));
            std::vector<float> sc(static_cast<size_t>(nb_r) * static_cast<size_t>(nb_c));
            std::vector<float> block(static_cast<size_t>(br) * static_cast<size_t>(cols));
            auto deq_row = [&](const uint8_t* src, float* dst) {
                if (t.quant == QuantKind::Q4_K) ops::dequant_q4_k(src, dst, cols);
                else if (t.quant == QuantKind::Q5_K) ops::dequant_q5_k(src, dst, cols);
                else ops::dequant_q6_k(src, dst, cols);
            };
            for (int bi = 0; bi < nb_r; ++bi) {
                const int r0 = bi * br;
                const int r1 = r0 + br < rows ? r0 + br : rows;
                const int rh = r1 - r0;
                for (int r = 0; r < rh; ++r)
                    deq_row(t.data.data() + static_cast<size_t>(r0 + r) * rowb,
                            block.data() + static_cast<size_t>(r) * cols);
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
            if (rows >= 128 && !t.scale.empty()) {
                const size_t nbytes = static_cast<size_t>(rows) * static_cast<size_t>(cols);
                void* p = alloc(nbytes);
                CUDA_CHECK(cudaMemcpy(p, t.data.data(), nbytes, cudaMemcpyHostToDevice));
                void* s = alloc(t.scale.size() * sizeof(float));
                CUDA_CHECK(cudaMemcpy(s, t.scale.data(), t.scale.size() * sizeof(float), cudaMemcpyHostToDevice));
                const size_t n4 = static_cast<size_t>(rows) * static_cast<size_t>(cols / 4);
                absorb_fp8_scale_k<<<static_cast<unsigned>((n4 + 255) / 256), 256>>>(
                    static_cast<uint8_t*>(p), static_cast<const float*>(s), rows, cols);
                CUDA_CHECK(cudaDeviceSynchronize());
                w.data = static_cast<const uint8_t*>(p);
                w.scale = static_cast<const float*>(s);
                w.fp8_kmajor = false;
                w.fp8_rowmaj = true;
                return w;
            }
            // Leftover (m=48) stays k-major so T=1 fused leftover GEMV is unchanged.
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

    void launch_rms_xe(const float* x, const float* g, float* y, int n, int T, float eps) {
        if (!g_xe_buf || T < 16 || n <= 0 || T * n > g_xe_cap) {
            launch_rms_batch(x, g, y, n, T, eps);
            use_xe(y, T, n);
            return;
        }
        const int th = n >= 1024 ? 256 : 128;
        rmsnorm_xe_batch_k<<<T, th>>>(x, g, y, g_xe_buf, n, T, eps);
        g_xe = g_xe_buf;
        g_xe_n = n;
        g_xe_T = T;
    }

    void apply_vision_chunk(int pos0, int T) {
        if (vis_n_ <= 0 || vis_off_ < 0 || !d_vis_) return;
        const int a = std::max(pos0, vis_off_);
        const int b = std::min(pos0 + T, vis_off_ + vis_n_);
        if (b <= a) return;
        CUDA_CHECK(cudaMemcpy(d_h_seq_ + static_cast<size_t>(a - pos0) * hidden_,
                              d_vis_ + static_cast<size_t>(a - vis_off_) * hidden_,
                              sizeof(float) * static_cast<size_t>(b - a) * hidden_, cudaMemcpyDeviceToDevice));
    }

    int kv_stride() const { return kv_tq_ ? kv_win_ : ctx_; }

    int8_t* k_q8_slot(int slot, int b = 0) {
        return d_k_q8_ + (static_cast<size_t>(b) * n_attn_ + slot) * ctx_ * max_nkv_ * max_hd_;
    }
    __half* k_sc_slot(int slot, int b = 0) {
        return d_k_sc_ + (static_cast<size_t>(b) * n_attn_ + slot) * ctx_ * max_nkv_;
    }
    uint8_t* v_qs_slot(int slot, int b = 0) {
        const int nblk = std::max(1, max_hd_ / kTqBlk);
        return d_v_qs_ + (static_cast<size_t>(b) * n_attn_ + slot) * ctx_ * max_nkv_ * nblk * kTq3B;
    }
    __half* v_sc_slot(int slot, int b = 0) {
        const int nblk = std::max(1, max_hd_ / kTqBlk);
        return d_v_sc_ + (static_cast<size_t>(b) * n_attn_ + slot) * ctx_ * max_nkv_ * nblk;
    }

    void store_tq_kv(const GpuLayer& L, const float* ksrc, const float* vsrc, int pos0, int T) {
        if (!kv_tq_ || L.hd != 256 || T <= 0) return;
        store_k_q8_th_k<<<dim3(L.nkv, T), 256>>>(k_q8_slot(L.slot), k_sc_slot(L.slot), ksrc, pos0, T, L.nkv,
                                                 L.hd);
        const int nblk = L.hd / kTqBlk;
        store_v_tq3_th_k<<<dim3(nblk, L.nkv, T), 128, kTqBlk * sizeof(float)>>>(
            v_qs_slot(L.slot), v_sc_slot(L.slot), vsrc, pos0, T, L.nkv, L.hd);
    }

    void dequant_tq_to_h(__half* kc, __half* vc, const GpuLayer& L, int tend) {
        if (!kv_tq_ || L.hd != 256 || tend <= 0) return;
        dequant_k_q8_h_k<<<dim3(L.nkv, tend), 256>>>(kc, k_q8_slot(L.slot), k_sc_slot(L.slot), 0, tend, L.nkv,
                                                     L.hd);
        const int nblk = L.hd / kTqBlk;
        dequant_v_tq3_h_k<<<dim3(nblk, L.nkv, tend), 128, kTqBlk * sizeof(float)>>>(
            vc, v_qs_slot(L.slot), v_sc_slot(L.slot), 0, tend, L.nkv, L.hd);
    }

    void launch_prefill_chunk(int pos0, int T) {
        const ModelDesc& m = store_->model();
        embed_batch(T);
        apply_vision_chunk(pos0, T);
        const bool prof = !capturing_ && std::getenv("RAPIDLLM_PF_PROFILE") &&
                          std::getenv("RAPIDLLM_PF_PROFILE")[0] == '1';
        cudaEvent_t ev0 = nullptr, ev1 = nullptr;
        float ms_lin = 0, ms_left = 0, ms_conv = 0, ms_gdn = 0, ms_attn = 0, ms_mlp = 0;
        const int lt0 = g_lt_ok, lf0 = g_lt_fail, sk0 = g_skinny_n;
        if (prof) {
            cudaEventCreate(&ev0);
            cudaEventCreate(&ev1);
        }
        auto mark = [&](cudaEvent_t e) {
            if (prof) cudaEventRecord(e, cudaStreamPerThread);
        };
        auto acc = [&](float& dst) {
            if (!prof) return;
            cudaEventSynchronize(ev1);
            float ms = 0;
            cudaEventElapsedTime(&ms, ev0, ev1);
            dst += ms;
        };
        for (GpuLayer& L : layers_) {
            launch_rms_xe(d_h_seq_, L.attn_norm, d_xn_seq_, hidden_, T, L.eps);
            if (L.kind == LayerKind::GatedDeltaNet) {
                const int qdim = L.nk * L.dk;
                const int qkv_dim = qdim * 2 + L.nv * L.dv;
                const int zdim = L.nv * L.dv;
                mark(ev0);
                launch_linear_pair(L.wqkv, L.wz, d_xn_seq_, d_qkv_seq_, d_z_seq_, T);
                mark(ev1);
                acc(ms_lin);
                float* conv_st = d_conv_ + static_cast<size_t>(L.slot) * qkv_dim * L.conv_k;
                const int cth = 32;
                const int cbl = (qkv_dim + 63) / 64;
                const bool ov_left = T >= 256 && g_bak_stream && !capturing_;
                mark(ev0);
                if (ov_left) {
                    g_launch_stream = g_bak_stream;
                    launch_gemm_fp8_dual(L.wa, L.wb, d_xn_seq_, d_aa_seq_, d_bb_seq_, T, 0);
                    g_launch_stream = nullptr;
                    if (pos0 == 0)
                        conv1d_prefill_k<<<cbl, cth>>>(d_qkv_seq_, L.conv_w, d_mix_seq_, conv_st, T, qkv_dim,
                                                       L.conv_k);
                    else
                        conv1d_upd_seq_k<<<cbl, cth>>>(d_qkv_seq_, L.conv_w, conv_st, d_mix_seq_, T, qkv_dim,
                                                       L.conv_k);
                    if (!g_pair_ev) cudaEventCreateWithFlags(&g_pair_ev, cudaEventDisableTiming);
                    cudaEventRecord(g_pair_ev, g_bak_stream);
                    cudaStreamWaitEvent(cudaStreamPerThread, g_pair_ev, 0);
                    if (g_blas) cublasSetStream(g_blas, cudaStreamPerThread);
                    mark(ev1);
                    acc(ms_left);
                    acc(ms_conv);
                } else {
                    launch_gemm_fp8_dual(L.wa, L.wb, d_xn_seq_, d_aa_seq_, d_bb_seq_, T, 0);
                    if (g_blas) cublasSetStream(g_blas, cudaStreamPerThread);
                    mark(ev1);
                    acc(ms_left);
                    mark(ev0);
                    if (pos0 == 0)
                        conv1d_prefill_k<<<cbl, cth>>>(d_qkv_seq_, L.conv_w, d_mix_seq_, conv_st, T, qkv_dim,
                                                       L.conv_k);
                    else
                        conv1d_upd_seq_k<<<cbl, cth>>>(d_qkv_seq_, L.conv_w, conv_st, d_mix_seq_, T, qkv_dim,
                                                       L.conv_k);
                    mark(ev1);
                    acc(ms_conv);
                }
                mark(ev0);
                uint16_t* S = d_S_ + static_cast<size_t>(L.slot) * L.nv * L.dk * L.dv;
                launch_gdn(d_mix_seq_, d_aa_seq_, d_bb_seq_, S, L.A_log, L.dt_bias, d_o_seq_, T, L.nk, L.nv,
                           L.dk, L.dv, qkv_dim);
                if (T >= 16 && g_xe_buf && T * zdim <= g_xe_cap) {
                    gated_rms_xe_batch_k<<<T, 256>>>(d_o_seq_, d_z_seq_, L.gnorm, d_og_seq_, g_xe_buf, zdim, T,
                                                     1e-6f, L.gnorm_n);
                    g_xe = g_xe_buf;
                    g_xe_n = zdim;
                    g_xe_T = T;
                } else {
                    gated_rms_batch_k<<<T, 256>>>(d_o_seq_, d_z_seq_, L.gnorm, d_og_seq_, zdim, T, 1e-6f,
                                                  L.gnorm_n);
                    use_xe(d_og_seq_, T, zdim);
                }
                mark(ev1);
                acc(ms_gdn);
                mark(ev0);
                launch_linear(L.wo, d_og_seq_, d_h_seq_, T, 1);
                mark(ev1);
                acc(ms_lin);
            } else {
                const int qn = L.nq * L.hd;
                const int kn = L.nkv * L.hd;
                mark(ev0);
                launch_linear_pair(L.wq, L.wk, d_xn_seq_, d_qg_seq_, d_k_seq_, T);
                launch_linear(L.wv, d_xn_seq_, d_v_seq_, T);
                mark(ev1);
                acc(ms_lin);
                dim3 sg((qn + 255) / 256, T);
                split_qg_batch_k<<<sg, 256>>>(d_qg_seq_, d_q_seq_, d_gate_seq_, qn, T);
                dim3 hg(L.nq, T);
                head_rms_batch_k<<<hg, 32>>>(d_q_seq_, L.q_norm, L.nq, L.hd, T, L.eps);
                dim3 hkg(L.nkv, T);
                head_rms_batch_k<<<hkg, 32>>>(d_k_seq_, L.k_norm, L.nkv, L.hd, T, L.eps);
                dim3 rg(2, std::max(L.nq, L.nkv), T);
                rope_batch_k<<<rg, 32>>>(d_q_seq_, d_k_seq_, L.nq, L.nkv, L.hd, L.rotary, pos0, T, L.theta);
                dim3 kg((kn + 127) / 128, T);
                mark(ev0);
                const int tend = pos0 + T;
                dim3 ag(L.nq, T);
                const int rep = (L.nkv > 0 && L.nq % L.nkv == 0) ? L.nq / L.nkv : 0;
                const bool gqa = L.hd == 256 && T >= 32 && rep >= 2 && rep <= 8;
                if (kv_f16_ && L.hd == 256) {
                    __half* kc = reinterpret_cast<__half*>(d_kcache_) +
                                 static_cast<size_t>(L.slot) * kv_stride() * kn;
                    __half* vc = reinterpret_cast<__half*>(d_vcache_) +
                                 static_cast<size_t>(L.slot) * kv_stride() * kn;
                    if (tend <= kv_win_) {
                        store_kv_batch_h_k<<<kg, 128>>>(kc, d_k_seq_, pos0, T, kn);
                        store_kv_batch_h_k<<<kg, 128>>>(vc, d_v_seq_, pos0, T, kn);
                    }
                    if (kv_tq_) store_tq_kv(L, d_k_seq_, d_v_seq_, pos0, T);
                    const size_t sc_cap = static_cast<size_t>(pf_cap_) * static_cast<size_t>(std::max(L.inter, 1));
                    if (kv_tq_ && tend > kv_win_ && L.hd == 256) {
                        attn_prefill_tq_gqa_k<<<dim3(L.nkv, T), 256>>>(
                            d_q_seq_, d_o_seq_, pos0, T, L.nq, L.nkv, L.hd, k_q8_slot(L.slot),
                            k_sc_slot(L.slot), v_qs_slot(L.slot), v_sc_slot(L.slot));
                    } else if (gqa && T >= 64 &&
                        launch_attn_gqa_gemm(d_q_seq_, kc, vc, d_o_seq_, d_gate_mlp_seq_, sc_cap, d_up_seq_,
                                             sc_cap, pos0, T, L.nq, L.nkv, L.hd)) {
                    } else if (gqa) {
                        const int th = 32 * rep;
                        if (T >= 64)
                            attn_prefill_gqa_h_k<4><<<dim3(L.nkv, (T + 3) / 4), th>>>(
                                d_q_seq_, kc, vc, d_o_seq_, pos0, T, L.nq, L.nkv, L.hd);
                        else
                            attn_prefill_gqa_h_k<2><<<dim3(L.nkv, (T + 1) / 2), th>>>(
                                d_q_seq_, kc, vc, d_o_seq_, pos0, T, L.nq, L.nkv, L.hd);
                    } else if (T >= 128)
                        attn_prefill_qtile_h_k<8><<<dim3(L.nq, (T + 7) / 8), 32>>>(
                            d_q_seq_, kc, vc, d_o_seq_, pos0, T, L.nq, L.nkv, L.hd);
                    else if (T >= 32)
                        attn_prefill_qtile_h_k<4><<<dim3(L.nq, (T + 3) / 4), 32>>>(
                            d_q_seq_, kc, vc, d_o_seq_, pos0, T, L.nq, L.nkv, L.hd);
                    else if (tend >= 64)
                        attn_prefill_warp_k<<<ag, 32>>>(d_q_seq_, reinterpret_cast<float*>(kc),
                                                        reinterpret_cast<float*>(vc), d_o_seq_, pos0, T, L.nq,
                                                        L.nkv, L.hd);
                    else {
                        const size_t sm = tend > 8192 ? 0 : static_cast<size_t>(tend) * sizeof(float);
                        attn_prefill_k<<<ag, tend > 8192 ? 128 : 32, sm>>>(
                            d_q_seq_, reinterpret_cast<float*>(kc), reinterpret_cast<float*>(vc), d_o_seq_, pos0,
                            T, L.nq, L.nkv, L.hd);
                    }
                } else {
                float* kc = d_kcache_ + static_cast<size_t>(L.slot) * ctx_ * kn;
                float* vc = d_vcache_ + static_cast<size_t>(L.slot) * ctx_ * kn;
                store_kv_batch_k<<<kg, 128>>>(kc, d_k_seq_, pos0, T, kn);
                store_kv_batch_k<<<kg, 128>>>(vc, d_v_seq_, pos0, T, kn);
                if (gqa) {
                    const int th = 32 * rep;
                    if (T >= 64)
                        attn_prefill_gqa_k<4><<<dim3(L.nkv, (T + 3) / 4), th>>>(d_q_seq_, kc, vc, d_o_seq_,
                                                                                pos0, T, L.nq, L.nkv, L.hd);
                    else
                        attn_prefill_gqa_k<2><<<dim3(L.nkv, (T + 1) / 2), th>>>(d_q_seq_, kc, vc, d_o_seq_,
                                                                                pos0, T, L.nq, L.nkv, L.hd);
                } else if (L.hd == 256 && T >= 128)
                    attn_prefill_qtile_k<8><<<dim3(L.nq, (T + 7) / 8), 32>>>(d_q_seq_, kc, vc, d_o_seq_, pos0, T,
                                                                             L.nq, L.nkv, L.hd);
                else if (L.hd == 256 && T >= 32)
                    attn_prefill_qtile_k<4><<<dim3(L.nq, (T + 3) / 4), 32>>>(d_q_seq_, kc, vc, d_o_seq_, pos0, T,
                                                                             L.nq, L.nkv, L.hd);
                else if (L.hd == 256 && tend >= 64)
                    attn_prefill_warp_k<<<ag, 32>>>(d_q_seq_, kc, vc, d_o_seq_, pos0, T, L.nq, L.nkv, L.hd);
                else {
                    const size_t sm = tend > 8192 ? 0 : static_cast<size_t>(tend) * sizeof(float);
                    attn_prefill_k<<<ag, tend > 8192 ? 128 : 32, sm>>>(d_q_seq_, kc, vc, d_o_seq_, pos0, T, L.nq,
                                                                       L.nkv, L.hd);
                }
                }
                apply_gate_n_k<<<(qn * T + 255) / 256, 256>>>(d_o_seq_, d_gate_seq_, qn * T);
                mark(ev1);
                acc(ms_attn);
                mark(ev0);
                use_xe(d_o_seq_, T, qn);
                launch_linear(L.wo_a, d_o_seq_, d_h_seq_, T, 1);
                mark(ev1);
                acc(ms_lin);
            }
            mark(ev0);
            launch_rms_xe(d_h_seq_, L.ffn_norm, d_xn_seq_, hidden_, T, m.rms_eps);
            if (T >= 16) {
                launch_gemm_fp8_dual(L.wg, L.wu, d_xn_seq_, d_gate_mlp_seq_, d_up_seq_, T, 0);
                use_swiglu_xe(d_gate_mlp_seq_, d_up_seq_, T, L.inter);
            } else {
                launch_gemm_fp8_dual(L.wg, L.wu, d_xn_seq_, d_gate_mlp_seq_, d_up_seq_, T, 1);
            }
            launch_linear(L.wd, d_gate_mlp_seq_, d_h_seq_, T, 1);
            mark(ev1);
            acc(ms_mlp);
        }
        if (prof) {
            std::fprintf(stderr,
                         "pf_prof pos0=%d T=%d lin=%.1fms left=%.1fms conv=%.1fms gdn=%.1fms attn=%.1fms mlp=%.1fms "
                         "lt=%d/%d skinny=%d\n",
                         pos0, T, ms_lin, ms_left, ms_conv, ms_gdn, ms_attn, ms_mlp, g_lt_ok - lt0,
                         g_lt_fail - lf0, g_skinny_n - sk0);
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);
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
            use_xe(d_xn_seq_, T, hidden_);
            if (L.kind == LayerKind::GatedDeltaNet) {
                const int qdim = L.nk * L.dk;
                const int qkv_dim = qdim * 2 + L.nv * L.dv;
                const int zdim = L.nv * L.dv;
                launch_linear_pair(L.wqkv, L.wz, d_xn_seq_, d_qkv_seq_, d_z_seq_, T);
                launch_gemm_fp8_dual(L.wa, L.wb, d_xn_seq_, d_aa_seq_, d_bb_seq_, T, 0);
                float* conv_st = d_conv_ + static_cast<size_t>(L.slot) * qkv_dim * L.conv_k;
                conv1d_upd_seq_k<<<(qkv_dim + 63) / 64, 32>>>(d_qkv_seq_, L.conv_w, conv_st, d_mix_seq_, T,
                                                              qkv_dim, L.conv_k);
                uint16_t* S = d_S_ + static_cast<size_t>(L.slot) * L.nv * L.dk * L.dv;
                launch_gdn(d_mix_seq_, d_aa_seq_, d_bb_seq_, S, L.A_log, L.dt_bias, d_o_seq_, T, L.nk, L.nv,
                           L.dk, L.dv, qkv_dim);
                gated_rms_batch_k<<<T, 256>>>(d_o_seq_, d_z_seq_, L.gnorm, d_og_seq_, zdim, T, 1e-6f, L.gnorm_n);
                use_xe(d_og_seq_, T, zdim);
                launch_linear(L.wo, d_og_seq_, d_h_seq_, T, 1);
            } else {
                const int qn = L.nq * L.hd;
                const int kn = L.nkv * L.hd;
                launch_linear_pair(L.wq, L.wk, d_xn_seq_, d_qg_seq_, d_k_seq_, T);
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
                use_xe(d_o_seq_, T, qn);
                launch_linear(L.wo_a, d_o_seq_, d_h_seq_, T, 1);
            }
            launch_rms_batch(d_h_seq_, L.ffn_norm, d_xn_seq_, hidden_, T, m.rms_eps);
            use_xe(d_xn_seq_, T, hidden_);
            if (T >= 16) {
                launch_gemm_fp8_dual(L.wg, L.wu, d_xn_seq_, d_gate_mlp_seq_, d_up_seq_, T, 0);
                use_swiglu_xe(d_gate_mlp_seq_, d_up_seq_, T, L.inter);
            } else {
                launch_gemm_fp8_dual(L.wg, L.wu, d_xn_seq_, d_gate_mlp_seq_, d_up_seq_, T, 1);
            }
            launch_linear(L.wd, d_gate_mlp_seq_, d_h_seq_, T, 1);
        }
        launch_rms_batch(d_h_seq_, d_final_norm_, d_xn_seq_, hidden_, T, store_->model().rms_eps);
        use_xe(d_xn_seq_, T, hidden_);
        launch_linear(lm_head_, d_xn_seq_, d_logits_, T);
        launch_argmax_rows(T);
    }

    void snap_last_residual(int last_T) {
        if (last_T > 0 && d_h_ && d_h_seq_)
            CUDA_CHECK(cudaMemcpy(d_h_, d_h_seq_ + static_cast<size_t>(last_T - 1) * hidden_,
                                  sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
    }

    void launch_prefill_tail(int last_T, int n) {
        if (last_T > 0) {
            snap_last_residual(last_T);
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
            const int slot = (T == 1024 && pos0 == 0) ? 0 : (T == 1024 && pos0 == 1024) ? 1 : -1;
            if (slot >= 0 && pf1024_exec_[slot]) {
                CUDA_CHECK(cudaGraphLaunch(pf1024_exec_[slot], cudaStreamPerThread));
            } else {
                launch_prefill_chunk(pos0, T);
            }
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
                        L.wqkv.fp8_kmajor && L.wz.fp8_kmajor && L.wqkv.rows >= 4096 && L.wz.rows >= 4096 &&
                        L.wa.rows > 0 && L.wb.rows > 0 && L.wa.cols == L.wqkv.cols && L.wb.cols == L.wqkv.cols)
                     : (L.wq.q == QuantKind::FP8_E4M3_B128 && L.wk.q == QuantKind::FP8_E4M3_B128 &&
                        L.wv.q == QuantKind::FP8_E4M3_B128 && L.wq.fp8_kmajor && L.wq.rows >= 4096 &&
                        L.wk.rows > 0 && L.wv.rows == L.wk.rows && L.wk.cols == L.wq.cols));
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
                                         L.wa.cols == L.wqkv.cols && L.wb.cols == L.wqkv.cols &&
                                         L.wqkv.fp8_kmajor && L.wz.fp8_kmajor;
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
                    wo_pf ? (L.wo.fp8_rowmaj ? L.wo.rows * L.wo.cols : L.wo.rows * fp8_pack_cols(L.wo.cols))
                          : 0;
                launch_gdn(nullptr, d_aa_, d_bb_, S, L.A_log, L.dt_bias, d_o_, 1, L.nk, L.nv, L.dk, L.dv,
                           qkv_dim, d_qkv_, L.conv_w, conv_st, L.conv_k, wo_pf, wo_pf_bytes);
                const bool wo_grms = d_ss_ && L.wo.q == QuantKind::FP8_E4M3_B128 && L.wo.fp8_kmajor &&
                                     L.wo.rows >= 4096 && L.wo.cols == zdim;
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
                if (L.wq.q == QuantKind::FP8_E4M3_B128 && L.wq.fp8_kmajor && L.wq.rows >= 4096 &&
                    L.wq.rows == qn * 2)
                    launch_gemv(L.wq, xin, d_q_, 0, nullptr, xg, xss, xeps, nullptr, d_gate_, qn);
                else {
                    launch_gemv(L.wq, xin, d_qg_, 0, nullptr, xg, xss, xeps);
                    split_qg_k<<<(qn + 255) / 256, 256>>>(d_qg_, d_q_, d_gate_, qn);
                }
                launch_gemv_dual(L.wk, L.wv, xin, d_k_, d_vtmp_, 0, xg, xss, xeps);
                float* kc =
                    kv_f16_ && L.hd == 256
                        ? reinterpret_cast<float*>(reinterpret_cast<__half*>(d_kcache_) +
                                                   static_cast<size_t>(L.slot) * kv_stride() * kn)
                        : d_kcache_ + static_cast<size_t>(L.slot) * kv_stride() * kn;
                float* vc =
                    kv_f16_ && L.hd == 256
                        ? reinterpret_cast<float*>(reinterpret_cast<__half*>(d_vcache_) +
                                                   static_cast<size_t>(L.slot) * kv_stride() * kn)
                        : d_vcache_ + static_cast<size_t>(L.slot) * kv_stride() * kn;
                const int use_win = kv_tq_ && pos_ < kv_win_;
                // 16384 forces the online-softmax path (1 KiB smem) used at --ctx 163840.
                // Passing kv_win_=8192 would allocate 33 KiB scores and drop occupancy.
                const int kctx = use_win ? 16384 : ctx_;
                const size_t sm = kctx > 8192 ? static_cast<size_t>(L.hd) * sizeof(float)
                                              : (static_cast<size_t>(kctx) + static_cast<size_t>(L.hd)) *
                                                    sizeof(float);
                const int kv_mode =
                    (kv_tq_ && L.hd == 256 && !use_win) ? 3 : (kv_f16_ && L.hd == 256 ? 1 : 0);
                qk_attn_decode_k<<<L.nq, L.hd >= 256 ? 256 : 128, sm>>>(
                    d_q_, d_k_, d_vtmp_, L.q_norm, L.k_norm, kc, vc, d_o_, d_pos_, L.nq, L.nkv, L.hd,
                    L.rotary, L.theta, L.eps, kctx, kv_mode, kv_tq_ ? k_q8_slot(L.slot) : nullptr,
                    kv_tq_ ? k_sc_slot(L.slot) : nullptr, kv_tq_ ? v_qs_slot(L.slot) : nullptr,
                    kv_tq_ ? v_sc_slot(L.slot) : nullptr);
                if (kv_mode == 3)
                    qk_attn_decode_tq_gqa_k<<<L.nkv, 256>>>(d_q_, d_o_, d_pos_, L.nq, L.nkv, L.hd,
                                                            k_q8_slot(L.slot), k_sc_slot(L.slot),
                                                            v_qs_slot(L.slot), v_sc_slot(L.slot));
                const bool gate_x = L.wo_a.q == QuantKind::FP8_E4M3_B128 && L.wo_a.fp8_kmajor && L.wo_a.rows >= 4096;
                if (gate_x)
                    launch_gemv(L.wo_a, d_o_, d_h_, 1, nullptr, nullptr, nullptr, 0.f, d_gate_);
                else {
                    apply_gate_k<<<(qn + 255) / 256, 256>>>(d_o_, d_gate_, qn);
                    launch_gemv(L.wo_a, d_o_, d_h_, 1);
                }
            }
            const bool mlp_xrms = d_ss_ && L.wg.q == QuantKind::FP8_E4M3_B128 && L.wg.fp8_kmajor && L.wg.rows >= 4096;
            if (mlp_xrms) {
                launch_gemv_dual(L.wg, L.wu, d_h_, d_gate_mlp_, d_up_, 1, L.ffn_norm, nullptr, m.rms_eps);
            } else {
                launch_rms(d_h_, L.ffn_norm, d_xn_, hidden_, m.rms_eps);
                launch_gemv_dual(L.wg, L.wu, d_xn_, d_gate_mlp_, d_up_, 1);
            }
            launch_gemv(L.wd, d_gate_mlp_, d_h_, 1);
        }
        const bool lh_xrms = d_ss_ && lm_head_.q == QuantKind::FP8_E4M3_B128 && lm_head_.fp8_kmajor &&
                             lm_head_.rows >= 4096 && lm_head_.cols == hidden_;
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
        int hpos0 = 0;
        CUDA_CHECK(cudaMemcpy(&hpos0, d_pos_b_, 4, cudaMemcpyDeviceToHost));
        for (GpuLayer& L : layers_) {
            launch_rms_batch(d_h_seq_, L.attn_norm, d_xn_seq_, hidden_, B, L.eps);
            if (L.kind == LayerKind::GatedDeltaNet) {
                const int qdim = L.nk * L.dk;
                const int qkv_dim = qdim * 2 + L.nv * L.dv;
                const int zdim = L.nv * L.dv;
                launch_linear(L.wqkv, d_xn_seq_, d_qkv_seq_, B);
                launch_linear(L.wz, d_xn_seq_, d_z_seq_, B);
                launch_gemm_fp8_dual(L.wa, L.wb, d_xn_seq_, d_aa_seq_, d_bb_seq_, B, 0);
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
                const int use_win = kv_tq_ && hpos0 < kv_win_;
                const int kctx = use_win ? 16384 : (kv_tq_ ? ctx_ : ctx_);
                const size_t sm = kctx > 8192 ? static_cast<size_t>(L.hd) * sizeof(float)
                                              : (static_cast<size_t>(kctx) + static_cast<size_t>(L.hd)) *
                                                    sizeof(float);
                const int kv_mode =
                    (kv_tq_ && L.hd == 256 && !use_win) ? 3 : (kv_f16_ && L.hd == 256 ? 1 : 0);
                const int ath = L.hd >= 256 ? 256 : 128;
                for (int b = 0; b < B; ++b) {
                    float* kc =
                        kv_f16_ && L.hd == 256
                            ? reinterpret_cast<float*>(reinterpret_cast<__half*>(d_kcache_) +
                                                       (static_cast<size_t>(b) * n_attn_ + L.slot) *
                                                           kv_stride() * kn)
                            : d_kcache_ + (static_cast<size_t>(b) * n_attn_ + L.slot) * kv_stride() * kn;
                    float* vc =
                        kv_f16_ && L.hd == 256
                            ? reinterpret_cast<float*>(reinterpret_cast<__half*>(d_vcache_) +
                                                       (static_cast<size_t>(b) * n_attn_ + L.slot) *
                                                           kv_stride() * kn)
                            : d_vcache_ + (static_cast<size_t>(b) * n_attn_ + L.slot) * kv_stride() * kn;
                    qk_attn_decode_k<<<L.nq, ath, sm>>>(
                        d_q_seq_ + static_cast<size_t>(b) * qn, d_k_seq_ + static_cast<size_t>(b) * kn,
                        d_v_seq_ + static_cast<size_t>(b) * kn, L.q_norm, L.k_norm, kc, vc,
                        d_o_seq_ + static_cast<size_t>(b) * qn, d_pos_b_ + b, L.nq, L.nkv, L.hd, L.rotary,
                        L.theta, L.eps, kctx, kv_mode, kv_tq_ ? k_q8_slot(L.slot, b) : nullptr,
                        kv_tq_ ? k_sc_slot(L.slot, b) : nullptr, kv_tq_ ? v_qs_slot(L.slot, b) : nullptr,
                        kv_tq_ ? v_sc_slot(L.slot, b) : nullptr);
                    if (kv_mode == 3)
                        qk_attn_decode_tq_gqa_k<<<L.nkv, 256>>>(
                            d_q_seq_ + static_cast<size_t>(b) * qn, d_o_seq_ + static_cast<size_t>(b) * qn,
                            d_pos_b_ + b, L.nq, L.nkv, L.hd, k_q8_slot(L.slot, b), k_sc_slot(L.slot, b),
                            v_qs_slot(L.slot, b), v_sc_slot(L.slot, b));
                }
                apply_gate_n_k<<<(qn * B + 255) / 256, 256>>>(d_o_seq_, d_gate_seq_, qn * B);
                launch_linear(L.wo_a, d_o_seq_, d_h_seq_, B, 1);
            }
            launch_rms_batch(d_h_seq_, L.ffn_norm, d_xn_seq_, hidden_, B, m.rms_eps);
            launch_gemm_fp8_dual(L.wg, L.wu, d_xn_seq_, d_gate_mlp_seq_, d_up_seq_, B, 1);
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
        if (vocab_ <= 256 && d_logits_) {
            float h8[8] = {};
            const int n = vocab_ < 8 ? vocab_ : 8;
            cudaMemcpy(h8, d_logits_, sizeof(float) * n, cudaMemcpyDeviceToHost);
            const cudaError_t err = cudaGetLastError();
            float hall[48];
            const int nv = vocab_ < 48 ? vocab_ : 48;
            cudaMemcpy(hall, d_logits_, sizeof(float) * nv, cudaMemcpyDeviceToHost);
            int am = 0;
            float mv = hall[0];
            for (int i = 1; i < nv; ++i)
                if (hall[i] > mv) {
                    mv = hall[i];
                    am = i;
                }
            std::fprintf(stderr, "tiny_logits tok=%d argmax=%d max=%.4f err=%s", last_tok_, am, mv,
                         err == cudaSuccess ? "ok" : cudaGetErrorString(err));
            for (int i = 0; i < n; ++i) std::fprintf(stderr, " %.4f", hall[i]);
            std::fprintf(stderr, " |44=%.4f\n", nv > 44 ? hall[44] : 0.f);
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

    void maybe_capture_pf_tile(int pos0, int T) {
        const int slot = (T == 1024 && pos0 == 0) ? 0 : (T == 1024 && pos0 == 1024) ? 1 : -1;
        if (slot < 0 || pf1024_exec_[slot] || pf_cap_ < 1024) return;
        abort_stream_capture();
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) cudaGetLastError();
        capturing_ = true;
        e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
        if (e != cudaSuccess) {
            capturing_ = false;
            std::fprintf(stderr, "pf1024_capture_begin pos0=%d err=%s\n", pos0, cudaGetErrorString(e));
            return;
        }
        launch_prefill_chunk(pos0, T);
        cudaGraph_t g = nullptr;
        e = cudaStreamEndCapture(cudaStreamPerThread, &g);
        capturing_ = false;
        if (e != cudaSuccess) {
            std::fprintf(stderr, "pf1024_capture_end pos0=%d err=%s\n", pos0, cudaGetErrorString(e));
            abort_stream_capture();
            return;
        }
        if (!instantiate_graph(g, &pf1024_exec_[slot], "pf1024_capture", pos0)) return;
        pf1024_graph_[slot] = g;
        cudaGraphUpload(pf1024_exec_[slot], cudaStreamPerThread);
        std::fprintf(stderr, "prefill_cuda_graph T=1024 pos0=%d\n", pos0);
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

    void persist_l2(void* p, size_t nbytes) {
        if (!p || nbytes == 0) return;
        cudaAccessPolicyWindow win{};
        win.base_ptr = p;
        win.num_bytes = nbytes > (16ull << 20) ? (16ull << 20) : nbytes;
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
        if (n == 256) {
            if (pf256_exec_ || pf_cap_ < 256) return;
        } else if (n < 2 || n > kPfGraphMax || pf_graph_execs_[n]) {
            return;
        }
        abort_stream_capture();
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "pf_capture_sync n=%d err=%s\n", n, cudaGetErrorString(e));
            cudaGetLastError();
        }
        abort_stream_capture();
        capturing_ = true;
        e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
        if (e != cudaSuccess) {
            capturing_ = false;
            std::fprintf(stderr, "pf_capture_begin n=%d err=%s\n", n, cudaGetErrorString(e));
            return;
        }
        launch_prefill_chunk(0, n);
        launch_prefill_tail(n, n);
        cudaGraph_t g = nullptr;
        e = cudaStreamEndCapture(cudaStreamPerThread, &g);
        capturing_ = false;
        if (e != cudaSuccess) {
            std::fprintf(stderr, "pf_capture_end n=%d err=%s\n", n, cudaGetErrorString(e));
            abort_stream_capture();
            return;
        }
        if (n == 256) {
            if (!instantiate_graph(g, &pf256_exec_, "pf_capture", n)) return;
            pf256_graph_ = g;
            cudaGraphUpload(pf256_exec_, cudaStreamPerThread);
        } else {
            if (!instantiate_graph(g, &pf_graph_execs_[n], "pf_capture", n)) return;
            pf_graphs_[n] = g;
            cudaGraphUpload(pf_graph_execs_[n], cudaStreamPerThread);
        }
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
        if (emb.quant == QuantKind::Q4_K || emb.quant == QuantKind::Q5_K || emb.quant == QuantKind::Q6_K ||
            emb.quant == QuantKind::Q8_0) {
            const int er = emb.shape[0] > 0 ? static_cast<int>(emb.shape[0]) : vocab_;
            const int ec = emb.shape[1] > 0 ? static_cast<int>(emb.shape[1]) : hidden_;
            std::vector<float> ef(static_cast<size_t>(er) * static_cast<size_t>(ec));
            if (emb.quant == QuantKind::Q8_0) {
                for (int r = 0; r < er; ++r)
                    ops::dequant_q8_0(emb.data.data() + static_cast<size_t>(r) * (ec / 32) * 34,
                                      ef.data() + static_cast<size_t>(r) * ec, ec);
            } else {
                const int bsz = emb.quant == QuantKind::Q4_K ? 144 : (emb.quant == QuantKind::Q5_K ? 176 : 210);
                const int nblk = ec / 256;
                const size_t rowb = static_cast<size_t>(nblk) * static_cast<size_t>(bsz);
                for (int r = 0; r < er; ++r) {
                    const uint8_t* src = emb.data.data() + static_cast<size_t>(r) * rowb;
                    float* dst = ef.data() + static_cast<size_t>(r) * ec;
                    if (emb.quant == QuantKind::Q4_K) ops::dequant_q4_k(src, dst, ec);
                    else if (emb.quant == QuantKind::Q5_K) ops::dequant_q5_k(src, dst, ec);
                    else ops::dequant_q6_k(src, dst, ec);
                }
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

        kv_f16_ = false;
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
                if (L.hd == 256) kv_f16_ = true;
                max_nkv_ = std::max(max_nkv_, L.nkv);
                max_hd_ = std::max(max_hd_, L.hd);
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
        // 1024-token prefill chunks: one weight pass per chunk after mt GEMM.
        // 256 was 4x more chunks (and used to be 64x more GEMM passes at T=4).
        pf_cap_ = std::max(1, std::min(ctx_, ctx_ > 4096 ? 1024 : 32));
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
        kv_tq_ = false;
        if (kv_f16_ && max_hd_ == 256 && max_nkv_ > 0) {
            const char* etq = std::getenv("RAPIDLLM_KV_TQ");
            if (etq && etq[0] == '0')
                kv_tq_ = false;
            else if (etq && etq[0] == '1')
                kv_tq_ = true;
            else
                // F16 KV OOMs this 48 GB box above 163840. Compact persist is
                // required for --ctx 262144; compute still uses an F16 window.
                kv_tq_ = ctx_ > 163840;
        }
        kv_win_ = ctx_;
        if (kv_tq_) kv_win_ = std::min(ctx_, std::max(8192, pf_cap_ * 2));
        const size_t f16_full = sizeof(__half) * static_cast<size_t>(max_batch_) * std::max(n_attn, 1) *
                                static_cast<size_t>(ctx_) * kn_max;
        kv_bytes_ = (kv_f16_ ? sizeof(__half) : sizeof(float)) *
                    static_cast<size_t>(max_batch_) * std::max(n_attn, 1) * kv_win_ * kn_max;
        d_S_ = static_cast<uint16_t*>(alloc(s_bytes_));
        d_conv_ = static_cast<float*>(alloc(conv_bytes_));
        d_S_bak_ = static_cast<uint16_t*>(alloc(s_bytes_));
        d_conv_bak_ = static_cast<float*>(alloc(conv_bytes_));
        d_kcache_ = static_cast<float*>(alloc(kv_bytes_));
        d_k_bak_ = kv_bytes_ ? static_cast<float*>(alloc(kv_bytes_)) : nullptr;
        d_v_bak_ = kv_bytes_ ? static_cast<float*>(alloc(kv_bytes_)) : nullptr;
        kvf_n_ = 0;
        d_kvf_ = nullptr;
        if (kv_f16_) {
            kvf_n_ = 4096 * kn_max * 2;
            d_kvf_ = static_cast<float*>(alloc(sizeof(float) * static_cast<size_t>(kvf_n_)));
        }
        d_vcache_ = static_cast<float*>(alloc(kv_bytes_));
        d_k_q8_ = nullptr;
        d_k_sc_ = nullptr;
        d_v_qs_ = nullptr;
        d_v_sc_ = nullptr;
        if (kv_tq_) {
            const int nblk = max_hd_ / kTqBlk;
            const size_t tok_h = static_cast<size_t>(max_batch_) * std::max(n_attn, 1) * ctx_ *
                                 static_cast<size_t>(max_nkv_);
            d_k_q8_ = static_cast<int8_t*>(alloc(tok_h * static_cast<size_t>(max_hd_)));
            d_k_sc_ = static_cast<__half*>(alloc(tok_h * sizeof(__half)));
            d_v_qs_ = static_cast<uint8_t*>(alloc(tok_h * static_cast<size_t>(std::max(nblk, 1)) * kTq3B));
            d_v_sc_ = static_cast<__half*>(alloc(tok_h * static_cast<size_t>(std::max(nblk, 1)) * sizeof(__half)));
            CUDA_CHECK(cudaMemset(d_k_q8_, 0, tok_h * static_cast<size_t>(max_hd_)));
            CUDA_CHECK(cudaMemset(d_k_sc_, 0, tok_h * sizeof(__half)));
            CUDA_CHECK(cudaMemset(d_v_qs_, 0, tok_h * static_cast<size_t>(std::max(nblk, 1)) * kTq3B));
            CUDA_CHECK(cudaMemset(d_v_sc_, 0, tok_h * static_cast<size_t>(std::max(nblk, 1)) * sizeof(__half)));
            const size_t tq_bytes = tok_h * static_cast<size_t>(max_hd_) + tok_h * 2 +
                                    tok_h * static_cast<size_t>(std::max(nblk, 1)) * (kTq3B + 2);
            std::fprintf(stderr,
                         "kv_tq=1 ctk=q8 ctv=tq3 ctx=%d win=%d persist=%zuMiB scratch=%zuMiB f16_full=%zuMiB "
                         "save=%.0f%%\n",
                         ctx_, kv_win_, tq_bytes / (1024 * 1024), kv_bytes_ * 2 / (1024 * 1024),
                         f16_full * 2 / (1024 * 1024),
                         f16_full > 0 ? 100.0 * (1.0 - double(tq_bytes + kv_bytes_ * 2) / double(f16_full * 2))
                                      : 0.0);
        }
        {
            size_t free_b = 0, tot_b = 0;
            if (cudaMemGetInfo(&free_b, &tot_b) == cudaSuccess)
                std::fprintf(stderr, "cuda_mem used=%zuMiB free=%zuMiB total=%zuMiB max_batch=%d\n",
                             (tot_b - free_b) / (1024 * 1024), free_b / (1024 * 1024), tot_b / (1024 * 1024),
                             max_batch_);
        }

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
        g_xe_cap = xf16_n;
        d_xe_seq_ = static_cast<uint8_t*>(alloc(static_cast<size_t>(g_xe_cap)));
        g_xe_buf = d_xe_seq_;
        g_xe = nullptr;
        g_xe_n = 0;
        g_xe_T = 0;
        {
            size_t persist = 16 * 1024 * 1024;
            if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist) != cudaSuccess) cudaGetLastError();
            cudaStreamAttrValue pol;
            std::memset(&pol, 0, sizeof(pol));
            pol.accessPolicyWindow.base_ptr = g_xe_buf;
            pol.accessPolicyWindow.num_bytes = static_cast<size_t>(std::min(g_xe_cap, static_cast<int>(persist)));
            pol.accessPolicyWindow.hitRatio = 1.f;
            pol.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            pol.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            if (cudaStreamSetAttribute(cudaStreamPerThread, cudaStreamAttributeAccessPolicyWindow, &pol) !=
                cudaSuccess)
                cudaGetLastError();
        }
        {
            const size_t kmax = static_cast<size_t>(
                std::max({hidden_, std::max(max_inter, 1), std::max(max_qkv, 1), std::max(max_qn * 2, 1)}));
            const size_t nmax = static_cast<size_t>(
                std::max({hidden_, std::max(max_inter, 1), std::max(max_qkv, 1), std::max(max_qn * 2, 1)}));
            g_wf16_n = 2 * kmax * nmax;
            d_wf16_ = static_cast<__half*>(alloc(sizeof(__half) * g_wf16_n));
            g_wf16 = d_wf16_;
        }

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
            // Populate cublasLt desc/algo cache and workspace before graph capture.
            int nrm = 0;
            auto cached = [](int m, int n, int k, int add) {
                for (int i = 0; i < g_lt_n; ++i)
                    if (g_lt_cache[i].m == m && g_lt_cache[i].n == n && g_lt_cache[i].k == k &&
                        g_lt_cache[i].add == add)
                        return true;
                return false;
            };
            auto warm = [&](const GpuW& w, int T) {
                if (!w.fp8_rowmaj || T < 16) return;
                float* X = w.cols > hidden_ ? d_gate_mlp_seq_ : d_xn_seq_;
                float* Y = w.rows > hidden_ ? d_up_seq_ : d_h_seq_;
                if (!X || !Y) return;
                if (!cached(w.rows, T, w.cols, 0)) {
                    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
                    cudaEventCreate(&ev0);
                    cudaEventCreate(&ev1);
                    cudaEventRecord(ev0);
                    const bool ok = launch_cublas_fp8_e4(w, X, Y, T, 0);
                    cudaEventRecord(ev1);
                    cudaEventSynchronize(ev1);
                    float ms = 0;
                    cudaEventElapsedTime(&ms, ev0, ev1);
                    cudaEventDestroy(ev0);
                    cudaEventDestroy(ev1);
                    std::fprintf(stderr, "lt_warm m=%d n=%d k=%d ok=%d ms=%.2f\n", w.rows, T, w.cols, ok ? 1 : 0,
                                 ms);
                    if (ok) {
                        ++nrm;
                        if (T >= 256) lt_tune(w, X, Y, T, 0);
                    }
                }
                if (!cached(w.rows, T, w.cols, 1)) {
                    launch_cublas_fp8_e4(w, X, Y, T, 1);
                    if (T >= 256) lt_tune(w, X, Y, T, 1);
                }
            };
            const int ts[2] = {std::min(256, pf_cap_), pf_cap_};
            for (int ti = 0; ti < 2; ++ti) {
                const int T = ts[ti];
                if (T < 16) continue;
                if (ti == 1 && T == ts[0]) continue;
                for (const GpuLayer& L : layers_) {
                    warm(L.wqkv, T);
                    warm(L.wz, T);
                    warm(L.wo, T);
                    warm(L.wq, T);
                    warm(L.wk, T);
                    warm(L.wv, T);
                    warm(L.wo_a, T);
                    warm(L.wg, T);
                    warm(L.wu, T);
                    warm(L.wd, T);
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            int n_rm = 0, n_km = 0;
            auto tally = [&](const GpuW& w) {
                if (w.fp8_rowmaj) ++n_rm;
                if (w.fp8_kmajor) ++n_km;
            };
            for (const GpuLayer& L : layers_) {
                tally(L.wqkv);
                tally(L.wz);
                tally(L.wa);
                tally(L.wb);
                tally(L.wo);
                tally(L.wq);
                tally(L.wk);
                tally(L.wv);
                tally(L.wo_a);
                tally(L.wg);
                tally(L.wu);
                tally(L.wd);
            }
            tally(lm_head_);
            std::fprintf(stderr, "cublas_fp8_e4_warm=%d cache=%d ws=%zu fp8_rm=%d fp8_km=%d\n", nrm, g_lt_n,
                         g_lt_ws_bytes, n_rm, n_km);
            if (!layers_.empty()) {
                const GpuW& w0 = layers_[0].wqkv.rows ? layers_[0].wqkv : layers_[0].wq;
                const GpuW& wa0 = layers_[0].wa;
                std::fprintf(stderr, "w0 q=%d rows=%d cols=%d rm=%d km=%d wa q=%d rows=%d\n",
                             static_cast<int>(w0.q), w0.rows, w0.cols, w0.fp8_rowmaj ? 1 : 0,
                             w0.fp8_kmajor ? 1 : 0, static_cast<int>(wa0.q), wa0.rows);
            }
        }
        {
            int zero = 0;
            CUDA_CHECK(cudaMemcpy(d_pos_, &zero, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_tok_, &zero, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
            if (vocab_ > 256) maybe_capture();
            skip_pf_graph_ = vocab_ <= 256;
            for (const GpuLayer& L : layers_) {
                if (L.wqkv.fp8_rowmaj || L.wg.fp8_rowmaj || L.wq.fp8_rowmaj || L.wo.fp8_rowmaj) {
                    skip_pf_graph_ = true;
                    break;
                }
            }
            if (lm_head_.fp8_rowmaj) skip_pf_graph_ = true;
            if (!skip_pf_graph_) {
                for (int t = 2; t <= 4; ++t) maybe_capture_prefill(t);
            }
            if (vocab_ > 256) maybe_capture_spec();
            if (pf_cap_ >= 1024 && vocab_ > 256) {
                maybe_capture_pf_tile(0, 1024);
                maybe_capture_pf_tile(1024, 1024);
            }
        }
        {
            size_t free_b = 0, tot_b = 0;
            if (cudaMemGetInfo(&free_b, &tot_b) == cudaSuccess)
                std::fprintf(stderr, "cuda_mem_ready used=%zuMiB free=%zuMiB max_batch=%d\n",
                             (tot_b - free_b) / (1024 * 1024), free_b / (1024 * 1024), max_batch_);
        }
        for (auto& [_, t] : store_->table().tensors) {
            // Host MTP draft needs embed + lm_head + mtp.* after GPU upload.
            if (t.ir_name == "embed" || t.ir_name == "lm_head" || t.ir_name.rfind("mtp.", 0) == 0)
                continue;
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
        if (pf256_exec_) cudaGraphExecDestroy(pf256_exec_);
        if (pf256_graph_) cudaGraphDestroy(pf256_graph_);
        for (int i = 0; i < 2; ++i) {
            if (pf1024_exec_[i]) cudaGraphExecDestroy(pf1024_exec_[i]);
            if (pf1024_graph_[i]) cudaGraphDestroy(pf1024_graph_[i]);
        }
        if (bak_stream_) {
            cudaStreamDestroy(bak_stream_);
            bak_stream_ = nullptr;
            g_bak_stream = nullptr;
        }
        g_xe_buf = nullptr;
        g_xe = nullptr;
        g_xe_n = 0;
        g_xe_T = 0;
        g_xe_cap = 0;
        lt_cache_clear();
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
    bool skip_pf_graph_ = false;
    bool kv_f16_ = false;
    bool kv_tq_ = false;
    int kv_win_ = 1;
    int max_nkv_ = 0, max_hd_ = 0;
    int8_t* d_k_q8_ = nullptr;
    __half* d_k_sc_ = nullptr;
    uint8_t* d_v_qs_ = nullptr;
    __half* d_v_sc_ = nullptr;
    float* d_kvf_ = nullptr;
    int kvf_n_ = 0;
    int max_batch_ = 1;
    int logit_rows_ = 1;
    int max_qkv_ = 0, max_z_ = 0, max_nv_ = 0, max_inter_ = 0, max_qn_ = 0, max_kn_ = 0;
    QuantKind embed_q_ = QuantKind::F16;
    const void* d_embed_ = nullptr;
    float* d_vis_ = nullptr;
    size_t vis_bytes_ = 0;
    int vis_n_ = 0;
    int vis_off_ = -1;
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
    float *d_k_bak_ = nullptr, *d_v_bak_ = nullptr;
    float *d_conv_bak_ = nullptr;
    float *d_amax_ = nullptr;
    int *d_tok_ = nullptr, *d_pos_ = nullptr, *d_pos_b_ = nullptr, *d_best_ = nullptr, *d_best_n_ = nullptr,
        *d_aidx_ = nullptr, *d_gen_out_ = nullptr, *d_gen_n_ = nullptr;
    float *d_h_seq_ = nullptr, *d_xn_seq_ = nullptr, *d_y_seq_ = nullptr;
    uint8_t* d_xe_seq_ = nullptr;
    float *d_qkv_seq_ = nullptr, *d_mix_seq_ = nullptr, *d_z_seq_ = nullptr;
    float *d_aa_seq_ = nullptr, *d_bb_seq_ = nullptr, *d_og_seq_ = nullptr;
    float *d_qg_seq_ = nullptr, *d_q_seq_ = nullptr, *d_gate_seq_ = nullptr;
    float *d_k_seq_ = nullptr, *d_v_seq_ = nullptr, *d_o_seq_ = nullptr;
    float *d_gate_mlp_seq_ = nullptr, *d_up_seq_ = nullptr;
    int* d_toks_ = nullptr;
    __half* d_xf16_ = nullptr;
    __half* d_wf16_ = nullptr;
    size_t s_bytes_ = 0, conv_bytes_ = 0, kv_bytes_ = 0;
    std::vector<float> h_logits_;
    mutable float* h_pin_ = nullptr;
    int* h_best_pin_ = nullptr;
    int* h_tok_pin_ = nullptr;
    static constexpr int kPfGraphMax = 8;
    static constexpr int kGenCap = 64;
    bool capturing_ = false;
    cudaGraph_t pf256_graph_ = nullptr;
    cudaGraphExec_t pf256_exec_ = nullptr;
    cudaGraph_t pf1024_graph_[2] = {};
    cudaGraphExec_t pf1024_exec_[2] = {};
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
