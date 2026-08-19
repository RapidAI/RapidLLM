#include "rapidllm/runtime/cuda_engine.h"
#include "rapidllm/runtime/mtp_live_draft.h"

#include "rapidllm/ir/model_desc.h"
#include "rapidllm/kernels/flashinfer_attn.h"
#include "rapidllm/kernels/gemv_q6_t4.h"
#include "rapidllm/kernels/gemv_t4_1row.h"
#include "rapidllm/kernels/gemv_t4_pipe.h"
#include "rapidllm/kernels/gemv_t4_q6_ku.h"
#include "rapidllm/kernels/gemv_t4_q6_t12tile.h"
#include "rapidllm/kernels/gemv_t4_q6_xs.h"
#include "rapidllm/kernels/gemv_t4_occ.h"
#include "rapidllm/kernels/gemv_t4_q6_pipe.h"
#include "rapidllm/kernels/gemv_t4_q8.h"
#include "rapidllm/kernels/gemv_t4_q6_2row.h"
#include "rapidllm/kernels/gemv_t4_q4_2row.h"
#include "rapidllm/kernels/gemv_t4_q4_sx.h"
#include "rapidllm/kernels/gemv_t4_q4_p3.h"
#include "rapidllm/kernels/gemv_t4_u2.h"
#include "rapidllm/kernels/gemv_t4_q4_r8.h"
#include "rapidllm/kernels/gemv_t4_q6_q8.h"
#include "rapidllm/kernels/gemv_t4_q4_x2.h"
#include "rapidllm/kernels/gemv_t4_q6_ho.h"
#include "rapidllm/kernels/gemv_t4_q6q4.h"
#include "rapidllm/kernels/gemv_t4_q4_tc.h"
#include "rapidllm/kernels/gemv_t4_q6_i8.h"
#include "rapidllm/kernels/gemv_t4_q4_p2.h"
#include "rapidllm/kernels/gemv_t4_q4_xs.h"
#include "rapidllm/kernels/gemv_t4_ur2.h"
#include "rapidllm/kernels/gemv_t4_q4_mmq.h"
#include "rapidllm/kernels/gemv_t4_q6_gm.h"
#include "rapidllm/kernels/gemv_t4_sk.h"
#include "rapidllm/kernels/gemv_t4_q6_ilp.h"
#include "rapidllm/kernels/gemv_q6_t1_2row.h"
#include "rapidllm/kernels/gemv_q6_t12_smemx.h"
#include "rapidllm/kernels/gemv_q4_t12_pipe.h"
#include "rapidllm/kernels/gemv_q4_t12_2row_pipe.h"
#include "rapidllm/kernels/gemv_q4_t16_pipe.h"
#include "rapidllm/kernels/gemv_q6_t16_1row.h"
#include "rapidllm/kernels/gemv_q4_t2_dual_1row.h"
#include "rapidllm/kernels/gemv_q6_t2_2row.h"
#include "rapidllm/kernels/gemv_q6_t2_splitk.h"
#include "rapidllm/kernels/mtp_attn_decode.h"
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
#include <utility>
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

// 0: HF grouped V (src = h / (nv/nk)). 1: llama.cpp tiled V (src = h % nk).
__constant__ int c_gdn_v_tiled = 0;
void set_gdn_v_tiled(int v) { CUDA_CHECK(cudaMemcpyToSymbol(c_gdn_v_tiled, &v, sizeof(int))); }

__device__ __forceinline__ int gdn_src_h(int h, int nk, int nv) {
    return c_gdn_v_tiled ? (h % nk) : (h / (nv / nk));
}

// One v-head per k-head owns the Q/K conv push (r==0).
__device__ __forceinline__ bool gdn_qk_conv_owner(int h, int nk, int nv) {
    return c_gdn_v_tiled ? (h < nk) : ((h % (nv / nk)) == 0);
}

// hd>=64: official Qwen rotate-half on first `rotary` dims (pair i with i+rotary/2).
// else: pair-interleaved (tiny fixture contract).
__device__ __forceinline__ void rope_apply(float* v, int rotary, float pos, float theta, int hd) {
    const int pairs = rotary / 2;
    if (pairs <= 0) return;
    if (hd >= 64) {
        for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
            const float freq = 1.f / powf(theta, static_cast<float>(i) / static_cast<float>(pairs));
            const float ang = pos * freq;
            const float c = cosf(ang), s = sinf(ang);
            const float x0 = v[i], x1 = v[i + pairs];
            v[i] = x0 * c - x1 * s;
            v[i + pairs] = x0 * s + x1 * c;
        }
    } else {
        for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
            const float freq = 1.f / powf(theta, static_cast<float>(i) / static_cast<float>(pairs));
            const float ang = pos * freq;
            const float c = cosf(ang), s = sinf(ang);
            const float x0 = v[2 * i], x1 = v[2 * i + 1];
            v[2 * i] = x0 * c - x1 * s;
            v[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}
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

// Host-set: RAPIDLLM_NO_FLASH=1 falls back to smem-score / non-FI prefill.
__device__ int d_use_flash = 1;

bool flash_attn_on() {
    const char* e = std::getenv("RAPIDLLM_NO_FLASH");
    return !(e && e[0] == '1');
}

void set_flash_attn(int on) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_use_flash, &on, sizeof(on)));
}

size_t decode_attn_smem(int kctx, int hd, int kv_f16) {
    if (flash_attn_on() && kv_f16 == 1 && hd == 256)
        return sizeof(float) * static_cast<size_t>(std::max(hd, 1));
    return kctx > 8192 ? sizeof(float) * static_cast<size_t>(hd)
                       : sizeof(float) * (static_cast<size_t>(kctx) + static_cast<size_t>(hd));
}

bool flash_gqa_ok(int n_q, int n_kv, int hd, int kv_f16) {
    if (!flash_attn_on() || kv_f16 != 1 || hd != 256 || n_kv <= 0 || n_q <= 0) return false;
    if (n_q % n_kv != 0) return false;
    const int g = n_q / n_kv;
    return g >= 2 && g <= 8;
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

__device__ __forceinline__ float bf16_dot8(uint4 u, const float* x, int j) {
    return bf16_to_f32(static_cast<uint16_t>(u.x)) * x[j] +
           bf16_to_f32(static_cast<uint16_t>(u.x >> 16)) * x[j + 1] +
           bf16_to_f32(static_cast<uint16_t>(u.y)) * x[j + 2] +
           bf16_to_f32(static_cast<uint16_t>(u.y >> 16)) * x[j + 3] +
           bf16_to_f32(static_cast<uint16_t>(u.z)) * x[j + 4] +
           bf16_to_f32(static_cast<uint16_t>(u.z >> 16)) * x[j + 5] +
           bf16_to_f32(static_cast<uint16_t>(u.w)) * x[j + 6] +
           bf16_to_f32(static_cast<uint16_t>(u.w >> 16)) * x[j + 7];
}

// T=2 BF16: one W stream, both X tokens. Beats 2x T=1 GEMV and cublas n=2.
__global__ void gemv_bf16_t2_k(const uint16_t* W, const float* X, float* Y, int m, int n, int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int tile = kXsTile;
    float a00 = 0.f, a01 = 0.f, a10 = 0.f, a11 = 0.f;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        for (int i = threadIdx.x; i < tn; i += blockDim.x) {
            xs[i] = X[t0 + i];
            xs[tile + i] = X[n + t0 + i];
        }
        __syncthreads();
        if (row0 < m) {
            const uint16_t* p0 = W + static_cast<size_t>(row0) * n + t0;
            const uint16_t* p1 = row1 < m ? W + static_cast<size_t>(row1) * n + t0 : nullptr;
            int j = lane * 8;
            for (; j + 7 < tn; j += 256) {
                const uint4 u0 = __ldcs(reinterpret_cast<const uint4*>(p0 + j));
                a00 += bf16_dot8(u0, xs, j);
                a01 += bf16_dot8(u0, xs + tile, j);
                if (p1) {
                    const uint4 u1 = __ldcs(reinterpret_cast<const uint4*>(p1 + j));
                    a10 += bf16_dot8(u1, xs, j);
                    a11 += bf16_dot8(u1, xs + tile, j);
                }
            }
            for (int r = (tn & ~7) + lane; r < tn; r += 32) {
                const float w0 = bf16_to_f32(p0[r]);
                a00 += w0 * xs[r];
                a01 += w0 * xs[tile + r];
                if (p1) {
                    const float w1 = bf16_to_f32(p1[r]);
                    a10 += w1 * xs[r];
                    a11 += w1 * xs[tile + r];
                }
            }
        }
        if (t0 + tile < n) __syncthreads();
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        if (lane == 0) write_y(Y, row0, a00, add);
        a01 = warp_sum(a01);
        if (lane == 0) write_y(Y + m, row0, a01, add);
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        if (lane == 0) write_y(Y, row1, a10, add);
        a11 = warp_sum(a11);
        if (lane == 0) write_y(Y + m, row1, a11, add);
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
constexpr int kQ6KGmBsz = 288;  // int8 q[256] (q-32, weight order) + half scales[16]

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

__global__ void quant_x_q8_k(const float* x, int8_t* xq, __half* xsc, int n) {
    const int blk = blockIdx.x * blockDim.x + threadIdx.x;
    const int nblk = n >> 5;
    if (blk >= nblk) return;
    const float* xb = x + (blk << 5);
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(xb[i]));
    const float s = amax * (1.f / 127.f);
    const float inv = s > 0.f ? 1.f / s : 0.f;
    xsc[blk] = __float2half(s);
    int8_t* qb = xq + (blk << 5);
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        int q = __float2int_rn(xb[i] * inv);
        q = q > 127 ? 127 : (q < -127 ? -127 : q);
        qb[i] = static_cast<int8_t>(q);
    }
}

// T rows of n; xq/xsc laid out [t][n] / [t][n/32]. xsum[t][n/32] is the
// int8 group sum (exact fold of the four lane-sx8 terms in Q4 T=4).
__global__ void quant_x_q8_n_k(const float* x, int8_t* xq, __half* xsc, int32_t* xsum, int n, int T) {
    const int nblk = n >> 5;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = nblk > 0 ? idx / nblk : 0;
    const int blk = nblk > 0 ? idx - t * nblk : 0;
    if (t >= T || blk >= nblk) return;
    const float* xb = x + static_cast<size_t>(t) * n + (blk << 5);
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(xb[i]));
    const float s = amax * (1.f / 127.f);
    const float inv = s > 0.f ? 1.f / s : 0.f;
    xsc[t * nblk + blk] = __float2half(s);
    int8_t* qb = xq + static_cast<size_t>(t) * n + (blk << 5);
    int sum = 0;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        int q = __float2int_rn(xb[i] * inv);
        q = q > 127 ? 127 : (q < -127 ? -127 : q);
        qb[i] = static_cast<int8_t>(q);
        sum += q;
    }
    if (xsum) xsum[t * nblk + blk] = sum;
}

// Integer Q4_K × Q8: 4 lanes / 32-wide group, 8 weights / lane, real __dp4a.
// Same dequant as acc_q4k_soa_row: (ds*q - dm) * (xs*xq).
__device__ __forceinline__ float acc_q4k_soa_q8(const uint8_t* row, const int8_t* xq, const __half* xsc, int nb,
                                                int lane) {
    const int group = lane >> 2;
    const int sub = lane & 3;
    const int gpair = group >> 1;
    const int hi = group & 1;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ4KSoaBsz;
        const __half* ds = reinterpret_cast<const __half*>(blk);
        const __half* dm = reinterpret_cast<const __half*>(blk + 16);
        const uint8_t* qs = blk + 32 + gpair * 32 + sub * 8;
        const int8_t* xb = xq + b * 256 + group * 32 + sub * 8;
        const uint2 q8 = *reinterpret_cast<const uint2*>(qs);
        const int q0 = static_cast<int>(hi ? ((q8.x >> 4) & 0x0f0f0f0f) : (q8.x & 0x0f0f0f0f));
        const int q1 = static_cast<int>(hi ? ((q8.y >> 4) & 0x0f0f0f0f) : (q8.y & 0x0f0f0f0f));
        const int x0 = *reinterpret_cast<const int*>(xb);
        const int x1 = *reinterpret_cast<const int*>(xb + 4);
        int sumi = 0, sumx = 0;
#if __CUDA_ARCH__ >= 610
        sumi = __dp4a(q0, x0, 0);
        sumi = __dp4a(q1, x1, sumi);
        sumx = __dp4a(0x01010101, x0, 0);
        sumx = __dp4a(0x01010101, x1, sumx);
#else
        const int8_t* qq = reinterpret_cast<const int8_t*>(&q0);
        const int8_t* xx = reinterpret_cast<const int8_t*>(&x0);
        for (int k = 0; k < 4; ++k) {
            sumi += qq[k] * xx[k];
            sumx += xx[k];
        }
        qq = reinterpret_cast<const int8_t*>(&q1);
        xx = reinterpret_cast<const int8_t*>(&x1);
        for (int k = 0; k < 4; ++k) {
            sumi += qq[k] * xx[k];
            sumx += xx[k];
        }
#endif
        const float xs = __half2float(xsc[b * 8 + group]);
        acc = fmaf(__half2float(ds[group]) * xs, static_cast<float>(sumi), acc);
        acc = fmaf(-__half2float(dm[group]) * xs, static_cast<float>(sumx), acc);
    }
    return acc;
}

// One W stream, two Q8 x vectors (T=2).
__device__ __forceinline__ void acc_q4k_soa_q8_2x(const uint8_t* row, const int8_t* xq0, const __half* sc0,
                                                  const int8_t* xq1, const __half* sc1, int nb, int lane,
                                                  float& a0, float& a1) {
    const int group = lane >> 2;
    const int sub = lane & 3;
    const int gpair = group >> 1;
    const int hi = group & 1;
#pragma unroll 2
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ4KSoaBsz;
        const __half* ds = reinterpret_cast<const __half*>(blk);
        const __half* dm = reinterpret_cast<const __half*>(blk + 16);
        const uint8_t* qs = blk + 32 + gpair * 32 + sub * 8;
        const uint2 q8 = *reinterpret_cast<const uint2*>(qs);
        const int q0 = static_cast<int>(hi ? ((q8.x >> 4) & 0x0f0f0f0f) : (q8.x & 0x0f0f0f0f));
        const int q1 = static_cast<int>(hi ? ((q8.y >> 4) & 0x0f0f0f0f) : (q8.y & 0x0f0f0f0f));
        const float dsg = __half2float(ds[group]);
        const float dmg = __half2float(dm[group]);
        const int8_t* xb0 = xq0 + b * 256 + group * 32 + sub * 8;
        const int8_t* xb1 = xq1 + b * 256 + group * 32 + sub * 8;
        const int u0 = *reinterpret_cast<const int*>(xb0);
        const int u1 = *reinterpret_cast<const int*>(xb0 + 4);
        const int v0 = *reinterpret_cast<const int*>(xb1);
        const int v1 = *reinterpret_cast<const int*>(xb1 + 4);
        int s0 = 0, sx0 = 0, s1 = 0, sx1 = 0;
#if __CUDA_ARCH__ >= 610
        s0 = __dp4a(q0, u0, 0);
        s0 = __dp4a(q1, u1, s0);
        sx0 = __dp4a(0x01010101, u0, 0);
        sx0 = __dp4a(0x01010101, u1, sx0);
        s1 = __dp4a(q0, v0, 0);
        s1 = __dp4a(q1, v1, s1);
        sx1 = __dp4a(0x01010101, v0, 0);
        sx1 = __dp4a(0x01010101, v1, sx1);
#else
        const int8_t* qq0 = reinterpret_cast<const int8_t*>(&q0);
        const int8_t* qq1 = reinterpret_cast<const int8_t*>(&q1);
        const int8_t* uu0 = reinterpret_cast<const int8_t*>(&u0);
        const int8_t* uu1 = reinterpret_cast<const int8_t*>(&u1);
        const int8_t* vv0 = reinterpret_cast<const int8_t*>(&v0);
        const int8_t* vv1 = reinterpret_cast<const int8_t*>(&v1);
        for (int k = 0; k < 4; ++k) {
            s0 += qq0[k] * uu0[k];
            s0 += qq1[k] * uu1[k];
            sx0 += uu0[k] + uu1[k];
            s1 += qq0[k] * vv0[k];
            s1 += qq1[k] * vv1[k];
            sx1 += vv0[k] + vv1[k];
        }
#endif
        const float xs0 = __half2float(sc0[b * 8 + group]);
        const float xs1 = __half2float(sc1[b * 8 + group]);
        a0 = fmaf(dsg * xs0, static_cast<float>(s0), a0);
        a0 = fmaf(-dmg * xs0, static_cast<float>(sx0), a0);
        a1 = fmaf(dsg * xs1, static_cast<float>(s1), a1);
        a1 = fmaf(-dmg * xs1, static_cast<float>(sx1), a1);
    }
}

// One W stream, three Q8 x vectors (T=3 spec).
__device__ __forceinline__ void acc_q4k_soa_q8_3x(const uint8_t* row, const int8_t* xq0, const __half* sc0,
                                                  const int8_t* xq1, const __half* sc1, const int8_t* xq2,
                                                  const __half* sc2, int nb, int lane, float& a0, float& a1,
                                                  float& a2) {
    const int group = lane >> 2;
    const int sub = lane & 3;
    const int gpair = group >> 1;
    const int hi = group & 1;
#pragma unroll 2
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ4KSoaBsz;
        const __half* ds = reinterpret_cast<const __half*>(blk);
        const __half* dm = reinterpret_cast<const __half*>(blk + 16);
        const uint8_t* qs = blk + 32 + gpair * 32 + sub * 8;
        const uint2 q8 = __ldcs(reinterpret_cast<const uint2*>(qs));
        const int q0 = static_cast<int>(hi ? ((q8.x >> 4) & 0x0f0f0f0f) : (q8.x & 0x0f0f0f0f));
        const int q1 = static_cast<int>(hi ? ((q8.y >> 4) & 0x0f0f0f0f) : (q8.y & 0x0f0f0f0f));
        const float dsg = __half2float(__ldg(ds + group));
        const float dmg = __half2float(__ldg(dm + group));
        const int off = b * 256 + group * 32 + sub * 8;
        const int u0 = *reinterpret_cast<const int*>(xq0 + off);
        const int u1 = *reinterpret_cast<const int*>(xq0 + off + 4);
        const int v0 = *reinterpret_cast<const int*>(xq1 + off);
        const int v1 = *reinterpret_cast<const int*>(xq1 + off + 4);
        const int w0 = *reinterpret_cast<const int*>(xq2 + off);
        const int w1 = *reinterpret_cast<const int*>(xq2 + off + 4);
        int s0 = 0, sx0 = 0, s1 = 0, sx1 = 0, s2 = 0, sx2 = 0;
#if __CUDA_ARCH__ >= 610
        s0 = __dp4a(q0, u0, 0);
        s0 = __dp4a(q1, u1, s0);
        sx0 = __dp4a(0x01010101, u0, 0);
        sx0 = __dp4a(0x01010101, u1, sx0);
        s1 = __dp4a(q0, v0, 0);
        s1 = __dp4a(q1, v1, s1);
        sx1 = __dp4a(0x01010101, v0, 0);
        sx1 = __dp4a(0x01010101, v1, sx1);
        s2 = __dp4a(q0, w0, 0);
        s2 = __dp4a(q1, w1, s2);
        sx2 = __dp4a(0x01010101, w0, 0);
        sx2 = __dp4a(0x01010101, w1, sx2);
#else
        const int8_t* qq0 = reinterpret_cast<const int8_t*>(&q0);
        const int8_t* qq1 = reinterpret_cast<const int8_t*>(&q1);
        const int8_t* uu0 = reinterpret_cast<const int8_t*>(&u0);
        const int8_t* uu1 = reinterpret_cast<const int8_t*>(&u1);
        const int8_t* vv0 = reinterpret_cast<const int8_t*>(&v0);
        const int8_t* vv1 = reinterpret_cast<const int8_t*>(&v1);
        const int8_t* ww0 = reinterpret_cast<const int8_t*>(&w0);
        const int8_t* ww1 = reinterpret_cast<const int8_t*>(&w1);
        for (int k = 0; k < 4; ++k) {
            s0 += qq0[k] * uu0[k];
            s0 += qq1[k] * uu1[k];
            sx0 += uu0[k] + uu1[k];
            s1 += qq0[k] * vv0[k];
            s1 += qq1[k] * vv1[k];
            sx1 += vv0[k] + vv1[k];
            s2 += qq0[k] * ww0[k];
            s2 += qq1[k] * ww1[k];
            sx2 += ww0[k] + ww1[k];
        }
#endif
        const float xs0 = __half2float(sc0[b * 8 + group]);
        const float xs1 = __half2float(sc1[b * 8 + group]);
        const float xs2 = __half2float(sc2[b * 8 + group]);
        a0 = fmaf(dsg * xs0, static_cast<float>(s0), a0);
        a0 = fmaf(-dmg * xs0, static_cast<float>(sx0), a0);
        a1 = fmaf(dsg * xs1, static_cast<float>(s1), a1);
        a1 = fmaf(-dmg * xs1, static_cast<float>(sx1), a1);
        a2 = fmaf(dsg * xs2, static_cast<float>(s2), a2);
        a2 = fmaf(-dmg * xs2, static_cast<float>(sx2), a2);
    }
}

// One W stream, four Q8 x vectors (T=4 spec).
__device__ __forceinline__ void acc_q4k_soa_q8_4x(const uint8_t* row, const int8_t* xq0, const __half* sc0,
                                                  const int8_t* xq1, const __half* sc1, const int8_t* xq2,
                                                  const __half* sc2, const int8_t* xq3, const __half* sc3, int nb,
                                                  int lane, float& a0, float& a1, float& a2, float& a3) {
    const int group = lane >> 2;
    const int sub = lane & 3;
    const int gpair = group >> 1;
    const int hi = group & 1;
#pragma unroll 2
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ4KSoaBsz;
        const __half* ds = reinterpret_cast<const __half*>(blk);
        const __half* dm = reinterpret_cast<const __half*>(blk + 16);
        const uint8_t* qs = blk + 32 + gpair * 32 + sub * 8;
        const uint2 q8 = __ldcs(reinterpret_cast<const uint2*>(qs));
        const int q0 = static_cast<int>(hi ? ((q8.x >> 4) & 0x0f0f0f0f) : (q8.x & 0x0f0f0f0f));
        const int q1 = static_cast<int>(hi ? ((q8.y >> 4) & 0x0f0f0f0f) : (q8.y & 0x0f0f0f0f));
        const float dsg = __half2float(__ldg(ds + group));
        const float dmg = __half2float(__ldg(dm + group));
        const int off = b * 256 + group * 32 + sub * 8;
        const int u0 = *reinterpret_cast<const int*>(xq0 + off);
        const int u1 = *reinterpret_cast<const int*>(xq0 + off + 4);
        const int v0 = *reinterpret_cast<const int*>(xq1 + off);
        const int v1 = *reinterpret_cast<const int*>(xq1 + off + 4);
        const int w0i = *reinterpret_cast<const int*>(xq2 + off);
        const int w1i = *reinterpret_cast<const int*>(xq2 + off + 4);
        const int z0 = *reinterpret_cast<const int*>(xq3 + off);
        const int z1 = *reinterpret_cast<const int*>(xq3 + off + 4);
        int s0 = 0, sx0 = 0, s1 = 0, sx1 = 0, s2 = 0, sx2 = 0, s3 = 0, sx3 = 0;
#if __CUDA_ARCH__ >= 610
        s0 = __dp4a(q0, u0, 0);
        s0 = __dp4a(q1, u1, s0);
        sx0 = __dp4a(0x01010101, u0, 0);
        sx0 = __dp4a(0x01010101, u1, sx0);
        s1 = __dp4a(q0, v0, 0);
        s1 = __dp4a(q1, v1, s1);
        sx1 = __dp4a(0x01010101, v0, 0);
        sx1 = __dp4a(0x01010101, v1, sx1);
        s2 = __dp4a(q0, w0i, 0);
        s2 = __dp4a(q1, w1i, s2);
        sx2 = __dp4a(0x01010101, w0i, 0);
        sx2 = __dp4a(0x01010101, w1i, sx2);
        s3 = __dp4a(q0, z0, 0);
        s3 = __dp4a(q1, z1, s3);
        sx3 = __dp4a(0x01010101, z0, 0);
        sx3 = __dp4a(0x01010101, z1, sx3);
#else
        const int8_t* qq0 = reinterpret_cast<const int8_t*>(&q0);
        const int8_t* qq1 = reinterpret_cast<const int8_t*>(&q1);
        const int8_t* uu0 = reinterpret_cast<const int8_t*>(&u0);
        const int8_t* uu1 = reinterpret_cast<const int8_t*>(&u1);
        const int8_t* vv0 = reinterpret_cast<const int8_t*>(&v0);
        const int8_t* vv1 = reinterpret_cast<const int8_t*>(&v1);
        const int8_t* ww0 = reinterpret_cast<const int8_t*>(&w0i);
        const int8_t* ww1 = reinterpret_cast<const int8_t*>(&w1i);
        const int8_t* zz0 = reinterpret_cast<const int8_t*>(&z0);
        const int8_t* zz1 = reinterpret_cast<const int8_t*>(&z1);
        for (int k = 0; k < 4; ++k) {
            s0 += qq0[k] * uu0[k] + qq1[k] * uu1[k];
            sx0 += uu0[k] + uu1[k];
            s1 += qq0[k] * vv0[k] + qq1[k] * vv1[k];
            sx1 += vv0[k] + vv1[k];
            s2 += qq0[k] * ww0[k] + qq1[k] * ww1[k];
            sx2 += ww0[k] + ww1[k];
            s3 += qq0[k] * zz0[k] + qq1[k] * zz1[k];
            sx3 += zz0[k] + zz1[k];
        }
#endif
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
}

// Q6_K × Q8: 16 groups of 16, 2 lanes/group, 8 weights/lane, __dp4a.
// Matches acc_q6k_soa_row dequant (q-32)*ds.
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

__device__ __forceinline__ float acc_q6k_soa_q8(const uint8_t* row, const int8_t* xq, const __half* xsc, int nb,
                                                int lane) {
    const int group = lane >> 1;
    const int sub = lane & 1;
    const int n128 = group >> 3;
    const int g8 = group & 7;
    const int qkind = g8 >> 1;
    const int idx = (g8 & 1) * 16 + sub * 8;
    const int xoff = group * 16 + sub * 8;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ6KSoaBsz;
        const __half* ds = reinterpret_cast<const __half*>(blk);
        const uint8_t* ql = blk + 32 + n128 * 64;
        const uint8_t* qh = blk + 160 + n128 * 32;
        int q0, q1;
        q6k_pack8(ql, qh, qkind, idx, &q0, &q1);
        const int8_t* xb = xq + b * 256 + xoff;
        const int u0 = *reinterpret_cast<const int*>(xb);
        const int u1 = *reinterpret_cast<const int*>(xb + 4);
        int sumi = 0;
#if __CUDA_ARCH__ >= 610
        sumi = __dp4a(q0, u0, 0);
        sumi = __dp4a(q1, u1, sumi);
#else
        const int8_t* qq0 = reinterpret_cast<const int8_t*>(&q0);
        const int8_t* qq1 = reinterpret_cast<const int8_t*>(&q1);
        const int8_t* xx0 = reinterpret_cast<const int8_t*>(&u0);
        const int8_t* xx1 = reinterpret_cast<const int8_t*>(&u1);
        for (int k = 0; k < 4; ++k) sumi += qq0[k] * xx0[k] + qq1[k] * xx1[k];
#endif
        const float xs = __half2float(xsc[b * 8 + (xoff >> 5)]);
        acc = fmaf(__half2float(ds[group]) * xs, static_cast<float>(sumi), acc);
    }
    return acc;
}

__device__ __forceinline__ void acc_q6k_soa_q8_2x(const uint8_t* row, const int8_t* xq0, const __half* sc0,
                                                  const int8_t* xq1, const __half* sc1, int nb, int lane,
                                                  float& a0, float& a1) {
    const int group = lane >> 1;
    const int sub = lane & 1;
    const int n128 = group >> 3;
    const int g8 = group & 7;
    const int qkind = g8 >> 1;
    const int idx = (g8 & 1) * 16 + sub * 8;
    const int xoff = group * 16 + sub * 8;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ6KSoaBsz;
        const __half* ds = reinterpret_cast<const __half*>(blk);
        const uint8_t* ql = blk + 32 + n128 * 64;
        const uint8_t* qh = blk + 160 + n128 * 32;
        int q0, q1;
        q6k_pack8(ql, qh, qkind, idx, &q0, &q1);
        const float dsg = __half2float(ds[group]);
        const int8_t* xb0 = xq0 + b * 256 + xoff;
        const int8_t* xb1 = xq1 + b * 256 + xoff;
        const int u0 = *reinterpret_cast<const int*>(xb0);
        const int u1 = *reinterpret_cast<const int*>(xb0 + 4);
        const int v0 = *reinterpret_cast<const int*>(xb1);
        const int v1 = *reinterpret_cast<const int*>(xb1 + 4);
        int s0 = 0, s1 = 0;
#if __CUDA_ARCH__ >= 610
        s0 = __dp4a(q0, u0, 0);
        s0 = __dp4a(q1, u1, s0);
        s1 = __dp4a(q0, v0, 0);
        s1 = __dp4a(q1, v1, s1);
#else
        const int8_t* qq0 = reinterpret_cast<const int8_t*>(&q0);
        const int8_t* qq1 = reinterpret_cast<const int8_t*>(&q1);
        const int8_t* uu0 = reinterpret_cast<const int8_t*>(&u0);
        const int8_t* uu1 = reinterpret_cast<const int8_t*>(&u1);
        const int8_t* vv0 = reinterpret_cast<const int8_t*>(&v0);
        const int8_t* vv1 = reinterpret_cast<const int8_t*>(&v1);
        for (int k = 0; k < 4; ++k) {
            s0 += qq0[k] * uu0[k] + qq1[k] * uu1[k];
            s1 += qq0[k] * vv0[k] + qq1[k] * vv1[k];
        }
#endif
        const float xs0 = __half2float(sc0[b * 8 + (xoff >> 5)]);
        const float xs1 = __half2float(sc1[b * 8 + (xoff >> 5)]);
        a0 = fmaf(dsg * xs0, static_cast<float>(s0), a0);
        a1 = fmaf(dsg * xs1, static_cast<float>(s1), a1);
    }
}

__device__ __forceinline__ void acc_q6k_soa_q8_3x(const uint8_t* row, const int8_t* xq0, const __half* sc0,
                                                  const int8_t* xq1, const __half* sc1, const int8_t* xq2,
                                                  const __half* sc2, int nb, int lane, float& a0, float& a1,
                                                  float& a2) {
    const int group = lane >> 1;
    const int sub = lane & 1;
    const int n128 = group >> 3;
    const int g8 = group & 7;
    const int qkind = g8 >> 1;
    const int idx = (g8 & 1) * 16 + sub * 8;
    const int xoff = group * 16 + sub * 8;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ6KSoaBsz;
        const __half* ds = reinterpret_cast<const __half*>(blk);
        const uint8_t* ql = blk + 32 + n128 * 64;
        const uint8_t* qh = blk + 160 + n128 * 32;
        int q0, q1;
        q6k_pack8(ql, qh, qkind, idx, &q0, &q1);
        const float dsg = __half2float(__ldg(ds + group));
        const int off = b * 256 + xoff;
        const int u0 = *reinterpret_cast<const int*>(xq0 + off);
        const int u1 = *reinterpret_cast<const int*>(xq0 + off + 4);
        const int v0 = *reinterpret_cast<const int*>(xq1 + off);
        const int v1 = *reinterpret_cast<const int*>(xq1 + off + 4);
        const int w0 = *reinterpret_cast<const int*>(xq2 + off);
        const int w1 = *reinterpret_cast<const int*>(xq2 + off + 4);
        int s0 = 0, s1 = 0, s2 = 0;
#if __CUDA_ARCH__ >= 610
        s0 = __dp4a(q0, u0, 0);
        s0 = __dp4a(q1, u1, s0);
        s1 = __dp4a(q0, v0, 0);
        s1 = __dp4a(q1, v1, s1);
        s2 = __dp4a(q0, w0, 0);
        s2 = __dp4a(q1, w1, s2);
#else
        const int8_t* qq0 = reinterpret_cast<const int8_t*>(&q0);
        const int8_t* qq1 = reinterpret_cast<const int8_t*>(&q1);
        const int8_t* uu0 = reinterpret_cast<const int8_t*>(&u0);
        const int8_t* uu1 = reinterpret_cast<const int8_t*>(&u1);
        const int8_t* vv0 = reinterpret_cast<const int8_t*>(&v0);
        const int8_t* vv1 = reinterpret_cast<const int8_t*>(&v1);
        const int8_t* ww0 = reinterpret_cast<const int8_t*>(&w0);
        const int8_t* ww1 = reinterpret_cast<const int8_t*>(&w1);
        for (int k = 0; k < 4; ++k) {
            s0 += qq0[k] * uu0[k] + qq1[k] * uu1[k];
            s1 += qq0[k] * vv0[k] + qq1[k] * vv1[k];
            s2 += qq0[k] * ww0[k] + qq1[k] * ww1[k];
        }
#endif
        const float xs0 = __half2float(sc0[b * 8 + (xoff >> 5)]);
        const float xs1 = __half2float(sc1[b * 8 + (xoff >> 5)]);
        const float xs2 = __half2float(sc2[b * 8 + (xoff >> 5)]);
        a0 = fmaf(dsg * xs0, static_cast<float>(s0), a0);
        a1 = fmaf(dsg * xs1, static_cast<float>(s1), a1);
        a2 = fmaf(dsg * xs2, static_cast<float>(s2), a2);
    }
}

__global__ void gemv_q6k_soa_q8_2row_k(const uint8_t* W, const int8_t* xq, const __half* xsc, float* y, int m, int n,
                                       int add) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    float acc0 = 0.f, acc1 = 0.f;
    if (row0 < m && nb > 0) {
        const uint8_t* r0 = W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ6KSoaBsz;
        acc0 = acc_q6k_soa_q8(r0, xq, xsc, nb, lane);
        if (row1 < m) {
            const uint8_t* r1 = W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ6KSoaBsz;
            acc1 = acc_q6k_soa_q8(r1, xq, xsc, nb, lane);
        }
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

__global__ void gemv_q6k_soa_q8_t2_2row_k(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m,
                                          int n, int add) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int8_t* xq0 = xq;
    const int8_t* xq1 = xq + n;
    const __half* sc0 = xsc;
    const __half* sc1 = xsc + (n >> 5);
    float a00 = 0.f, a01 = 0.f, a10 = 0.f, a11 = 0.f;
    if (row0 < m && nb > 0) {
        const uint8_t* r0 = W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ6KSoaBsz;
        acc_q6k_soa_q8_2x(r0, xq0, sc0, xq1, sc1, nb, lane, a00, a01);
        if (row1 < m) {
            const uint8_t* r1 = W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ6KSoaBsz;
            acc_q6k_soa_q8_2x(r1, xq0, sc0, xq1, sc1, nb, lane, a10, a11);
        }
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        a01 = warp_sum(a01);
        if (lane == 0) {
            write_y(Y, row0, a00, add);
            write_y(Y + m, row0, a01, add);
        }
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        a11 = warp_sum(a11);
        if (lane == 0) {
            write_y(Y, row1, a10, add);
            write_y(Y + m, row1, a11, add);
        }
    }
}

__global__ void __launch_bounds__(256, 2) gemv_q6k_soa_q8_t3_2row_k(const uint8_t* W, const int8_t* xq,
                                                                    const __half* xsc, float* Y, int m, int n,
                                                                    int add) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    const int8_t* xq0 = xq;
    const int8_t* xq1 = xq + n;
    const int8_t* xq2 = xq + 2 * n;
    const __half* sc0 = xsc;
    const __half* sc1 = xsc + nsc;
    const __half* sc2 = xsc + 2 * nsc;
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a10 = 0.f, a11 = 0.f, a12 = 0.f;
    if (row0 < m && nb > 0) {
        const uint8_t* r0 = W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ6KSoaBsz;
        acc_q6k_soa_q8_3x(r0, xq0, sc0, xq1, sc1, xq2, sc2, nb, lane, a00, a01, a02);
        if (row1 < m) {
            const uint8_t* r1 = W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ6KSoaBsz;
            acc_q6k_soa_q8_3x(r1, xq0, sc0, xq1, sc1, xq2, sc2, nb, lane, a10, a11, a12);
        }
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        a01 = warp_sum(a01);
        a02 = warp_sum(a02);
        if (lane == 0) {
            write_y(Y, row0, a00, add);
            write_y(Y + m, row0, a01, add);
            write_y(Y + 2 * m, row0, a02, add);
        }
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        a11 = warp_sum(a11);
        a12 = warp_sum(a12);
        if (lane == 0) {
            write_y(Y, row1, a10, add);
            write_y(Y + m, row1, a11, add);
            write_y(Y + 2 * m, row1, a12, add);
        }
    }
}

__global__ void gemv_q4k_soa_q8_2row_k(const uint8_t* W, const int8_t* xq, const __half* xsc, float* y, int m, int n,
                                       int add) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    float acc0 = 0.f, acc1 = 0.f;
    if (row0 < m && nb > 0) {
        const uint8_t* r0 = W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        acc0 = acc_q4k_soa_q8(r0, xq, xsc, nb, lane);
        if (row1 < m) {
            const uint8_t* r1 = W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz;
            acc1 = acc_q4k_soa_q8(r1, xq, xsc, nb, lane);
        }
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

// T=2 spec verify: one W read, two Q8 x vectors, two rows / warp. No float-x smem.
__global__ void gemv_q4k_soa_q8_t2_2row_k(const uint8_t* W, const int8_t* xq, const __half* xsc, float* Y, int m,
                                          int n, int add) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int8_t* xq0 = xq;
    const int8_t* xq1 = xq + n;
    const __half* sc0 = xsc;
    const __half* sc1 = xsc + (n >> 5);
    float a00 = 0.f, a01 = 0.f, a10 = 0.f, a11 = 0.f;
    if (row0 < m && nb > 0) {
        const uint8_t* r0 = W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        acc_q4k_soa_q8_2x(r0, xq0, sc0, xq1, sc1, nb, lane, a00, a01);
        if (row1 < m) {
            const uint8_t* r1 = W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz;
            acc_q4k_soa_q8_2x(r1, xq0, sc0, xq1, sc1, nb, lane, a10, a11);
        }
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        a01 = warp_sum(a01);
        if (lane == 0) {
            write_y(Y, row0, a00, add);
            write_y(Y + m, row0, a01, add);
        }
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        a11 = warp_sum(a11);
        if (lane == 0) {
            write_y(Y, row1, a10, add);
            write_y(Y + m, row1, a11, add);
        }
    }
}

// T=3 spec: one W read, three Q8 x, two rows / warp.
__global__ void __launch_bounds__(256, 2) gemv_q4k_soa_q8_t3_2row_k(const uint8_t* W, const int8_t* xq,
                                                                    const __half* xsc, float* Y, int m,
                                          int n, int add) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    const int8_t* xq0 = xq;
    const int8_t* xq1 = xq + n;
    const int8_t* xq2 = xq + 2 * n;
    const __half* sc0 = xsc;
    const __half* sc1 = xsc + nsc;
    const __half* sc2 = xsc + 2 * nsc;
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a10 = 0.f, a11 = 0.f, a12 = 0.f;
    if (row0 < m && nb > 0) {
        const uint8_t* r0 = W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        const uint8_t* r1 = (row1 < m) ? (W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz)
                                       : nullptr;
        // One block of each row in flight — better HBM overlap than two full-row passes.
        for (int b0 = 0; b0 < nb; ++b0) {
            acc_q4k_soa_q8_3x(r0 + static_cast<size_t>(b0) * kQ4KSoaBsz, xq0 + b0 * 256, sc0 + b0 * 8,
                              xq1 + b0 * 256, sc1 + b0 * 8, xq2 + b0 * 256, sc2 + b0 * 8, 1, lane, a00, a01,
                              a02);
            if (r1)
                acc_q4k_soa_q8_3x(r1 + static_cast<size_t>(b0) * kQ4KSoaBsz, xq0 + b0 * 256, sc0 + b0 * 8,
                                  xq1 + b0 * 256, sc1 + b0 * 8, xq2 + b0 * 256, sc2 + b0 * 8, 1, lane, a10,
                                  a11, a12);
        }
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        a01 = warp_sum(a01);
        a02 = warp_sum(a02);
        if (lane == 0) {
            write_y(Y, row0, a00, add);
            write_y(Y + m, row0, a01, add);
            write_y(Y + 2 * m, row0, a02, add);
        }
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        a11 = warp_sum(a11);
        a12 = warp_sum(a12);
        if (lane == 0) {
            write_y(Y, row1, a10, add);
            write_y(Y + m, row1, a11, add);
            write_y(Y + 2 * m, row1, a12, add);
        }
    }
}

// wg+wu T=3: one x, two Q4 W streams, optional SwiGLU.
__global__ void __launch_bounds__(256, 2) gemv_q4k_soa_q8_t3_dual_k(const uint8_t* W1, const uint8_t* W2,
                                                                    const int8_t* xq, const __half* xsc, float* Y1,
                                                                    float* Y2, int m, int n, int fuse_swiglu) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    const int8_t* xq0 = xq;
    const int8_t* xq1 = xq + n;
    const int8_t* xq2 = xq + 2 * n;
    const __half* sc0 = xsc;
    const __half* sc1 = xsc + nsc;
    const __half* sc2 = xsc + 2 * nsc;
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, b00 = 0.f, b01 = 0.f, b02 = 0.f;
    float a10 = 0.f, a11 = 0.f, a12 = 0.f, b10 = 0.f, b11 = 0.f, b12 = 0.f;
    if (row0 < m && nb > 0) {
        const uint8_t* r0 = W1 + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        const uint8_t* s0 = W2 + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        const uint8_t* r1 = (row1 < m) ? (W1 + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz)
                                       : nullptr;
        const uint8_t* s1 = (row1 < m) ? (W2 + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz)
                                       : nullptr;
        for (int b0 = 0; b0 < nb; ++b0) {
            acc_q4k_soa_q8_3x(r0 + static_cast<size_t>(b0) * kQ4KSoaBsz, xq0 + b0 * 256, sc0 + b0 * 8,
                              xq1 + b0 * 256, sc1 + b0 * 8, xq2 + b0 * 256, sc2 + b0 * 8, 1, lane, a00, a01,
                              a02);
            acc_q4k_soa_q8_3x(s0 + static_cast<size_t>(b0) * kQ4KSoaBsz, xq0 + b0 * 256, sc0 + b0 * 8,
                              xq1 + b0 * 256, sc1 + b0 * 8, xq2 + b0 * 256, sc2 + b0 * 8, 1, lane, b00, b01,
                              b02);
            if (r1)
                acc_q4k_soa_q8_3x(r1 + static_cast<size_t>(b0) * kQ4KSoaBsz, xq0 + b0 * 256, sc0 + b0 * 8,
                                  xq1 + b0 * 256, sc1 + b0 * 8, xq2 + b0 * 256, sc2 + b0 * 8, 1, lane, a10,
                                  a11, a12);
            if (s1)
                acc_q4k_soa_q8_3x(s1 + static_cast<size_t>(b0) * kQ4KSoaBsz, xq0 + b0 * 256, sc0 + b0 * 8,
                                  xq1 + b0 * 256, sc1 + b0 * 8, xq2 + b0 * 256, sc2 + b0 * 8, 1, lane, b10,
                                  b11, b12);
        }
    }
    auto emit = [&](float g, float u, float* yg, float* yu, int row) {
        g = warp_sum(g);
        u = warp_sum(u);
        if (lane != 0) return;
        if (fuse_swiglu) write_y(yg, row, silu_d(g) * u, 0);
        else {
            write_y(yg, row, g, 0);
            if (yu) write_y(yu, row, u, 0);
        }
    };
    if (row0 < m) {
        emit(a00, b00, Y1, Y2, row0);
        emit(a01, b01, Y1 + m, Y2 ? Y2 + m : nullptr, row0);
        emit(a02, b02, Y1 + 2 * m, Y2 ? Y2 + 2 * m : nullptr, row0);
    }
    if (row1 < m) {
        emit(a10, b10, Y1, Y2, row1);
        emit(a11, b11, Y1 + m, Y2 ? Y2 + m : nullptr, row1);
        emit(a12, b12, Y1 + 2 * m, Y2 ? Y2 + 2 * m : nullptr, row1);
    }
}

// T=4 spec: one W read, four Q8 x, two rows / warp.
__global__ void __launch_bounds__(256, 2) gemv_q4k_soa_q8_t4_2row_k(const uint8_t* W, const int8_t* xq,
                                                                    const __half* xsc, float* Y, int m, int n,
                                                                    int add) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int nsc = n >> 5;
    const int8_t* xq0 = xq;
    const int8_t* xq1 = xq + n;
    const int8_t* xq2 = xq + 2 * n;
    const int8_t* xq3 = xq + 3 * n;
    const __half* sc0 = xsc;
    const __half* sc1 = xsc + nsc;
    const __half* sc2 = xsc + 2 * nsc;
    const __half* sc3 = xsc + 3 * nsc;
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a03 = 0.f, a10 = 0.f, a11 = 0.f, a12 = 0.f, a13 = 0.f;
    if (row0 < m && nb > 0) {
        const uint8_t* r0 = W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ4KSoaBsz;
        acc_q4k_soa_q8_4x(r0, xq0, sc0, xq1, sc1, xq2, sc2, xq3, sc3, nb, lane, a00, a01, a02, a03);
        if (row1 < m) {
            const uint8_t* r1 = W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ4KSoaBsz;
            acc_q4k_soa_q8_4x(r1, xq0, sc0, xq1, sc1, xq2, sc2, xq3, sc3, nb, lane, a10, a11, a12, a13);
        }
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        a01 = warp_sum(a01);
        a02 = warp_sum(a02);
        a03 = warp_sum(a03);
        if (lane == 0) {
            write_y(Y, row0, a00, add);
            write_y(Y + m, row0, a01, add);
            write_y(Y + 2 * m, row0, a02, add);
            write_y(Y + 3 * m, row0, a03, add);
        }
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        a11 = warp_sum(a11);
        a12 = warp_sum(a12);
        a13 = warp_sum(a13);
        if (lane == 0) {
            write_y(Y, row1, a10, add);
            write_y(Y + m, row1, a11, add);
            write_y(Y + 2 * m, row1, a12, add);
            write_y(Y + 3 * m, row1, a13, add);
        }
    }
}

// T=2, 4 rows / warp: x loaded once per Q4 block, 4 W rows. 256 thr / 3 blocks/SM.
__global__ void __launch_bounds__(256, 3) gemv_q4k_soa_q8_t2_4row_k(const uint8_t* W, const int8_t* xq,
                                                                    const __half* xsc, float* Y, int m, int n,
                                                                    int add) {
    const int warps = blockDim.x / 32;
    const int pack = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pack * 4;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    if (row0 >= m || nb <= 0) return;
    const int group = lane >> 2;
    const int sub = lane & 3;
    const int gpair = group >> 1;
    const int hi = group & 1;
    const int8_t* xq0 = xq;
    const int8_t* xq1 = xq + n;
    const __half* sc0 = xsc;
    const __half* sc1 = xsc + (n >> 5);
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const uint8_t* r[4];
#pragma unroll
    for (int rr = 0; rr < 4; ++rr)
        r[rr] = (row0 + rr < m) ? (W + static_cast<size_t>(row0 + rr) * static_cast<size_t>(nb) * kQ4KSoaBsz)
                                : nullptr;
    for (int b = 0; b < nb; ++b) {
        const int8_t* xb0 = xq0 + b * 256 + group * 32 + sub * 8;
        const int8_t* xb1 = xq1 + b * 256 + group * 32 + sub * 8;
        const int u0 = *reinterpret_cast<const int*>(xb0);
        const int u1 = *reinterpret_cast<const int*>(xb0 + 4);
        const int v0 = *reinterpret_cast<const int*>(xb1);
        const int v1 = *reinterpret_cast<const int*>(xb1 + 4);
        const float xs0 = __half2float(sc0[b * 8 + group]);
        const float xs1 = __half2float(sc1[b * 8 + group]);
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
            if (!r[rr]) continue;
            const uint8_t* blk = r[rr] + static_cast<size_t>(b) * kQ4KSoaBsz;
            const __half* ds = reinterpret_cast<const __half*>(blk);
            const __half* dm = reinterpret_cast<const __half*>(blk + 16);
            const uint8_t* qs = blk + 32 + gpair * 32 + sub * 8;
            const uint2 q8 = *reinterpret_cast<const uint2*>(qs);
            const int q0 = static_cast<int>(hi ? ((q8.x >> 4) & 0x0f0f0f0f) : (q8.x & 0x0f0f0f0f));
            const int q1 = static_cast<int>(hi ? ((q8.y >> 4) & 0x0f0f0f0f) : (q8.y & 0x0f0f0f0f));
            int s0 = 0, sx0 = 0, s1 = 0, sx1 = 0;
#if __CUDA_ARCH__ >= 610
            s0 = __dp4a(q0, u0, 0);
            s0 = __dp4a(q1, u1, s0);
            sx0 = __dp4a(0x01010101, u0, 0);
            sx0 = __dp4a(0x01010101, u1, sx0);
            s1 = __dp4a(q0, v0, 0);
            s1 = __dp4a(q1, v1, s1);
            sx1 = __dp4a(0x01010101, v0, 0);
            sx1 = __dp4a(0x01010101, v1, sx1);
#else
            const int8_t* qq0 = reinterpret_cast<const int8_t*>(&q0);
            const int8_t* qq1 = reinterpret_cast<const int8_t*>(&q1);
            const int8_t* uu0 = reinterpret_cast<const int8_t*>(&u0);
            const int8_t* uu1 = reinterpret_cast<const int8_t*>(&u1);
            const int8_t* vv0 = reinterpret_cast<const int8_t*>(&v0);
            const int8_t* vv1 = reinterpret_cast<const int8_t*>(&v1);
            for (int k = 0; k < 4; ++k) {
                s0 += qq0[k] * uu0[k];
                s0 += qq1[k] * uu1[k];
                sx0 += uu0[k] + uu1[k];
                s1 += qq0[k] * vv0[k];
                s1 += qq1[k] * vv1[k];
                sx1 += vv0[k] + vv1[k];
            }
#endif
            const float dsg = __half2float(ds[group]);
            const float dmg = __half2float(dm[group]);
            acc[rr * 2 + 0] = fmaf(dsg * xs0, static_cast<float>(s0), acc[rr * 2 + 0]);
            acc[rr * 2 + 0] = fmaf(-dmg * xs0, static_cast<float>(sx0), acc[rr * 2 + 0]);
            acc[rr * 2 + 1] = fmaf(dsg * xs1, static_cast<float>(s1), acc[rr * 2 + 1]);
            acc[rr * 2 + 1] = fmaf(-dmg * xs1, static_cast<float>(sx1), acc[rr * 2 + 1]);
        }
    }
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        const int row = row0 + rr;
        if (row >= m) break;
        const float t0 = warp_sum(acc[rr * 2 + 0]);
        const float t1 = warp_sum(acc[rr * 2 + 1]);
        if (lane == 0) {
            write_y(Y, row, t0, add);
            write_y(Y + m, row, t1, add);
        }
    }
}

// One W stream, two x vectors. x1 may be global (T=2 second token).
__device__ __forceinline__ void acc_q4k_soa_2x(const uint8_t* row, const float* x0, const float* x1, int nb,
                                               int lane, float& a0, float& a1) {
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ4KSoaBsz;
        const __half* ds = reinterpret_cast<const __half*>(blk);
        const __half* dm = reinterpret_cast<const __half*>(blk + 16);
        const uint8_t* q = blk + 32;
        const float* b0 = x0 + b * 256;
        const float* b1 = x1 + b * 256;
        const uint32_t qpack = (static_cast<uint32_t>(__ldcs(q + lane))) |
                               (static_cast<uint32_t>(__ldcs(q + 32 + lane)) << 8) |
                               (static_cast<uint32_t>(__ldcs(q + 64 + lane)) << 16) |
                               (static_cast<uint32_t>(__ldcs(q + 96 + lane)) << 24);
#pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const uint8_t qq = static_cast<uint8_t>(qpack >> (grp * 8));
            const float w0 = __half2float(ds[grp * 2]) * static_cast<float>(qq & 15) - __half2float(dm[grp * 2]);
            const float w1 = __half2float(ds[grp * 2 + 1]) * static_cast<float>(qq >> 4) - __half2float(dm[grp * 2 + 1]);
            a0 = fmaf(w0, __ldg(b0 + lane), a0);
            a0 = fmaf(w1, __ldg(b0 + 32 + lane), a0);
            a1 = fmaf(w0, __ldg(b1 + lane), a1);
            a1 = fmaf(w1, __ldg(b1 + 32 + lane), a1);
            b0 += 64;
            b1 += 64;
        }
    }
}

__device__ __forceinline__ void acc_q6k_soa_2x(const uint8_t* row, const float* x0, const float* x1, int nb,
                                               int lane, float& a0, float& a1) {
    const int is = lane / 16;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ6KSoaBsz;
        const __half* ds = reinterpret_cast<const __half*>(blk);
        const uint8_t* ql = blk + 32;
        const uint8_t* qh = blk + 160;
        const float* b0 = x0 + b * 256;
        const float* b1 = x1 + b * 256;
#pragma unroll
        for (int n128 = 0; n128 < 2; ++n128) {
            const int q1 = static_cast<int>((ql[lane] & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32;
            const int q2 = static_cast<int>((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32;
            const int q3 = static_cast<int>((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32;
            const int q4 = static_cast<int>((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4)) - 32;
            const float s1 = __half2float(ds[is]) * static_cast<float>(q1);
            const float s2 = __half2float(ds[is + 2]) * static_cast<float>(q2);
            const float s3 = __half2float(ds[is + 4]) * static_cast<float>(q3);
            const float s4 = __half2float(ds[is + 6]) * static_cast<float>(q4);
            a0 = fmaf(s1, __ldg(b0 + lane), a0);
            a0 = fmaf(s2, __ldg(b0 + 32 + lane), a0);
            a0 = fmaf(s3, __ldg(b0 + 64 + lane), a0);
            a0 = fmaf(s4, __ldg(b0 + 96 + lane), a0);
            a1 = fmaf(s1, __ldg(b1 + lane), a1);
            a1 = fmaf(s2, __ldg(b1 + 32 + lane), a1);
            a1 = fmaf(s3, __ldg(b1 + 64 + lane), a1);
            a1 = fmaf(s4, __ldg(b1 + 96 + lane), a1);
            ql += 64;
            qh += 32;
            ds += 8;
            b0 += 128;
            b1 += 128;
        }
    }
}

__device__ __forceinline__ void acc_q4k_soa_3x(const uint8_t* row, const float* x0, const float* x1, const float* x2,
                                               int nb, int lane, float& a0, float& a1, float& a2) {
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ4KSoaBsz;
        const __half* ds = reinterpret_cast<const __half*>(blk);
        const __half* dm = reinterpret_cast<const __half*>(blk + 16);
        const uint8_t* q = blk + 32;
        const float* b0 = x0 + b * 256;
        const float* b1 = x1 + b * 256;
        const float* b2 = x2 + b * 256;
        const uint32_t qpack = (static_cast<uint32_t>(__ldcs(q + lane))) |
                               (static_cast<uint32_t>(__ldcs(q + 32 + lane)) << 8) |
                               (static_cast<uint32_t>(__ldcs(q + 64 + lane)) << 16) |
                               (static_cast<uint32_t>(__ldcs(q + 96 + lane)) << 24);
#pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const uint8_t qq = static_cast<uint8_t>(qpack >> (grp * 8));
            const float w0 = __half2float(ds[grp * 2]) * static_cast<float>(qq & 15) - __half2float(dm[grp * 2]);
            const float w1 = __half2float(ds[grp * 2 + 1]) * static_cast<float>(qq >> 4) - __half2float(dm[grp * 2 + 1]);
            a0 = fmaf(w0, __ldg(b0 + lane), a0);
            a0 = fmaf(w1, __ldg(b0 + 32 + lane), a0);
            a1 = fmaf(w0, __ldg(b1 + lane), a1);
            a1 = fmaf(w1, __ldg(b1 + 32 + lane), a1);
            a2 = fmaf(w0, __ldg(b2 + lane), a2);
            a2 = fmaf(w1, __ldg(b2 + 32 + lane), a2);
            b0 += 64;
            b1 += 64;
            b2 += 64;
        }
    }
}

__device__ __forceinline__ void acc_q6k_soa_3x(const uint8_t* row, const float* x0, const float* x1, const float* x2,
                                               int nb, int lane, float& a0, float& a1, float& a2) {
    const int is = lane / 16;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ6KSoaBsz;
        const __half* ds = reinterpret_cast<const __half*>(blk);
        const uint8_t* ql = blk + 32;
        const uint8_t* qh = blk + 160;
        const float* b0 = x0 + b * 256;
        const float* b1 = x1 + b * 256;
        const float* b2 = x2 + b * 256;
#pragma unroll
        for (int n128 = 0; n128 < 2; ++n128) {
            const int q1 = static_cast<int>((ql[lane] & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32;
            const int q2 = static_cast<int>((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32;
            const int q3 = static_cast<int>((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32;
            const int q4 = static_cast<int>((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4)) - 32;
            const float s1 = __half2float(ds[is]) * static_cast<float>(q1);
            const float s2 = __half2float(ds[is + 2]) * static_cast<float>(q2);
            const float s3 = __half2float(ds[is + 4]) * static_cast<float>(q3);
            const float s4 = __half2float(ds[is + 6]) * static_cast<float>(q4);
            a0 = fmaf(s1, __ldg(b0 + lane), a0);
            a0 = fmaf(s2, __ldg(b0 + 32 + lane), a0);
            a0 = fmaf(s3, __ldg(b0 + 64 + lane), a0);
            a0 = fmaf(s4, __ldg(b0 + 96 + lane), a0);
            a1 = fmaf(s1, __ldg(b1 + lane), a1);
            a1 = fmaf(s2, __ldg(b1 + 32 + lane), a1);
            a1 = fmaf(s3, __ldg(b1 + 64 + lane), a1);
            a1 = fmaf(s4, __ldg(b1 + 96 + lane), a1);
            a2 = fmaf(s1, __ldg(b2 + lane), a2);
            a2 = fmaf(s2, __ldg(b2 + 32 + lane), a2);
            a2 = fmaf(s3, __ldg(b2 + 64 + lane), a2);
            a2 = fmaf(s4, __ldg(b2 + 96 + lane), a2);
            ql += 64;
            qh += 32;
            ds += 8;
            b0 += 128;
            b1 += 128;
            b2 += 128;
        }
    }
}

// T=3 Q6 float: two rows interleaved so both W streams stay in flight.
__global__ void __launch_bounds__(256, 2) gemv_q6k_soa_f32_t3_2row_k(const uint8_t* W, const float* X, float* Y,
                                                                     int m, int n, int add) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int is = lane / 16;
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a10 = 0.f, a11 = 0.f, a12 = 0.f;
    const uint8_t* r0 = (row0 < m && nb > 0)
                            ? (W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ6KSoaBsz)
                            : nullptr;
    const uint8_t* r1 = (row1 < m && nb > 0)
                            ? (W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ6KSoaBsz)
                            : nullptr;
    const float* x0 = X;
    const float* x1 = X + n;
    const float* x2 = X + 2 * n;
    for (int b = 0; b < nb; ++b) {
        const float* b0 = x0 + b * 256;
        const float* b1 = x1 + b * 256;
        const float* b2 = x2 + b * 256;
        const float x0a = __ldg(b0 + lane), x0b = __ldg(b0 + 32 + lane);
        const float x0c = __ldg(b0 + 64 + lane), x0d = __ldg(b0 + 96 + lane);
        const float x1a = __ldg(b1 + lane), x1b = __ldg(b1 + 32 + lane);
        const float x1c = __ldg(b1 + 64 + lane), x1d = __ldg(b1 + 96 + lane);
        const float x2a = __ldg(b2 + lane), x2b = __ldg(b2 + 32 + lane);
        const float x2c = __ldg(b2 + 64 + lane), x2d = __ldg(b2 + 96 + lane);
        auto acc_row = [&](const uint8_t* row, float& a0, float& a1, float& a2) {
            const uint8_t* blk = row + static_cast<size_t>(b) * kQ6KSoaBsz;
            const __half* ds = reinterpret_cast<const __half*>(blk);
            const uint8_t* ql = blk + 32;
            const uint8_t* qh = blk + 160;
#pragma unroll
            for (int n128 = 0; n128 < 2; ++n128) {
                const uint8_t qlo = __ldcs(ql + lane);
                const uint8_t qhi = __ldcs(qh + lane);
                const uint8_t qlo2 = __ldcs(ql + 32 + lane);
                const int q1 = static_cast<int>((qlo & 0xF) | (((qhi >> 0) & 3) << 4)) - 32;
                const int q2 = static_cast<int>((qlo2 & 0xF) | (((qhi >> 2) & 3) << 4)) - 32;
                const int q3 = static_cast<int>((qlo >> 4) | (((qhi >> 4) & 3) << 4)) - 32;
                const int q4 = static_cast<int>((qlo2 >> 4) | (((qhi >> 6) & 3) << 4)) - 32;
                const float s1 = __half2float(__ldg(ds + is)) * static_cast<float>(q1);
                const float s2 = __half2float(__ldg(ds + is + 2)) * static_cast<float>(q2);
                const float s3 = __half2float(__ldg(ds + is + 4)) * static_cast<float>(q3);
                const float s4 = __half2float(__ldg(ds + is + 6)) * static_cast<float>(q4);
                if (n128 == 0) {
                    a0 = fmaf(s1, x0a, a0);
                    a0 = fmaf(s2, x0b, a0);
                    a0 = fmaf(s3, x0c, a0);
                    a0 = fmaf(s4, x0d, a0);
                    a1 = fmaf(s1, x1a, a1);
                    a1 = fmaf(s2, x1b, a1);
                    a1 = fmaf(s3, x1c, a1);
                    a1 = fmaf(s4, x1d, a1);
                    a2 = fmaf(s1, x2a, a2);
                    a2 = fmaf(s2, x2b, a2);
                    a2 = fmaf(s3, x2c, a2);
                    a2 = fmaf(s4, x2d, a2);
                } else {
                    a0 = fmaf(s1, __ldg(b0 + 128 + lane), a0);
                    a0 = fmaf(s2, __ldg(b0 + 160 + lane), a0);
                    a0 = fmaf(s3, __ldg(b0 + 192 + lane), a0);
                    a0 = fmaf(s4, __ldg(b0 + 224 + lane), a0);
                    a1 = fmaf(s1, __ldg(b1 + 128 + lane), a1);
                    a1 = fmaf(s2, __ldg(b1 + 160 + lane), a1);
                    a1 = fmaf(s3, __ldg(b1 + 192 + lane), a1);
                    a1 = fmaf(s4, __ldg(b1 + 224 + lane), a1);
                    a2 = fmaf(s1, __ldg(b2 + 128 + lane), a2);
                    a2 = fmaf(s2, __ldg(b2 + 160 + lane), a2);
                    a2 = fmaf(s3, __ldg(b2 + 192 + lane), a2);
                    a2 = fmaf(s4, __ldg(b2 + 224 + lane), a2);
                }
                ql += 64;
                qh += 32;
                ds += 8;
            }
        };
        if (r0) acc_row(r0, a00, a01, a02);
        if (r1) acc_row(r1, a10, a11, a12);
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        a01 = warp_sum(a01);
        a02 = warp_sum(a02);
        if (lane == 0) {
            write_y(Y, row0, a00, add);
            write_y(Y + m, row0, a01, add);
            write_y(Y + 2 * m, row0, a02, add);
        }
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        a11 = warp_sum(a11);
        a12 = warp_sum(a12);
        if (lane == 0) {
            write_y(Y, row1, a10, add);
            write_y(Y + m, row1, a11, add);
            write_y(Y + 2 * m, row1, a12, add);
        }
    }
}

// Keep this unused sibling in the TU. Removing it changed nvcc codegen of
// gemv_q6k_soa_f32_t3_2row_k enough to flip Q4 decode onto 4 5 0 31 46474.
__global__ void __launch_bounds__(256, 2) gemv_q6k_soa_f32_t3_dual_k(const uint8_t* W1, const uint8_t* W2,
                                                                     const float* X, float* Y1, float* Y2,
                                                                     int m1, int m2, int n) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int is = lane / 16;
    const int mmax = m1 > m2 ? m1 : m2;
    if (row0 >= mmax || nb <= 0) return;
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, b00 = 0.f, b01 = 0.f, b02 = 0.f;
    float a10 = 0.f, a11 = 0.f, a12 = 0.f, b10 = 0.f, b11 = 0.f, b12 = 0.f;
    const uint8_t* r0a = (row0 < m1) ? (W1 + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ6KSoaBsz)
                                     : nullptr;
    const uint8_t* r0b = (row0 < m2) ? (W2 + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ6KSoaBsz)
                                     : nullptr;
    const uint8_t* r1a = (row1 < m1) ? (W1 + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ6KSoaBsz)
                                     : nullptr;
    const uint8_t* r1b = (row1 < m2) ? (W2 + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ6KSoaBsz)
                                     : nullptr;
    const float* x0 = X;
    const float* x1 = X + n;
    const float* x2 = X + 2 * n;
    for (int b = 0; b < nb; ++b) {
        const float* p0 = x0 + b * 256;
        const float* p1 = x1 + b * 256;
        const float* p2 = x2 + b * 256;
        const float x0a = __ldg(p0 + lane), x0b = __ldg(p0 + 32 + lane);
        const float x0c = __ldg(p0 + 64 + lane), x0d = __ldg(p0 + 96 + lane);
        const float x1a = __ldg(p1 + lane), x1b = __ldg(p1 + 32 + lane);
        const float x1c = __ldg(p1 + 64 + lane), x1d = __ldg(p1 + 96 + lane);
        const float x2a = __ldg(p2 + lane), x2b = __ldg(p2 + 32 + lane);
        const float x2c = __ldg(p2 + 64 + lane), x2d = __ldg(p2 + 96 + lane);
        auto acc_row = [&](const uint8_t* row, float& a0, float& a1, float& a2) {
            const uint8_t* blk = row + static_cast<size_t>(b) * kQ6KSoaBsz;
            const __half* ds = reinterpret_cast<const __half*>(blk);
            const uint8_t* ql = blk + 32;
            const uint8_t* qh = blk + 160;
#pragma unroll
            for (int n128 = 0; n128 < 2; ++n128) {
                const uint8_t qlo = __ldcs(ql + lane);
                const uint8_t qhi = __ldcs(qh + lane);
                const uint8_t qlo2 = __ldcs(ql + 32 + lane);
                const int q1 = static_cast<int>((qlo & 0xF) | (((qhi >> 0) & 3) << 4)) - 32;
                const int q2 = static_cast<int>((qlo2 & 0xF) | (((qhi >> 2) & 3) << 4)) - 32;
                const int q3 = static_cast<int>((qlo >> 4) | (((qhi >> 4) & 3) << 4)) - 32;
                const int q4 = static_cast<int>((qlo2 >> 4) | (((qhi >> 6) & 3) << 4)) - 32;
                const float s1 = __half2float(__ldg(ds + is)) * static_cast<float>(q1);
                const float s2 = __half2float(__ldg(ds + is + 2)) * static_cast<float>(q2);
                const float s3 = __half2float(__ldg(ds + is + 4)) * static_cast<float>(q3);
                const float s4 = __half2float(__ldg(ds + is + 6)) * static_cast<float>(q4);
                if (n128 == 0) {
                    a0 = fmaf(s1, x0a, a0);
                    a0 = fmaf(s2, x0b, a0);
                    a0 = fmaf(s3, x0c, a0);
                    a0 = fmaf(s4, x0d, a0);
                    a1 = fmaf(s1, x1a, a1);
                    a1 = fmaf(s2, x1b, a1);
                    a1 = fmaf(s3, x1c, a1);
                    a1 = fmaf(s4, x1d, a1);
                    a2 = fmaf(s1, x2a, a2);
                    a2 = fmaf(s2, x2b, a2);
                    a2 = fmaf(s3, x2c, a2);
                    a2 = fmaf(s4, x2d, a2);
                } else {
                    a0 = fmaf(s1, __ldg(p0 + 128 + lane), a0);
                    a0 = fmaf(s2, __ldg(p0 + 160 + lane), a0);
                    a0 = fmaf(s3, __ldg(p0 + 192 + lane), a0);
                    a0 = fmaf(s4, __ldg(p0 + 224 + lane), a0);
                    a1 = fmaf(s1, __ldg(p1 + 128 + lane), a1);
                    a1 = fmaf(s2, __ldg(p1 + 160 + lane), a1);
                    a1 = fmaf(s3, __ldg(p1 + 192 + lane), a1);
                    a1 = fmaf(s4, __ldg(p1 + 224 + lane), a1);
                    a2 = fmaf(s1, __ldg(p2 + 128 + lane), a2);
                    a2 = fmaf(s2, __ldg(p2 + 160 + lane), a2);
                    a2 = fmaf(s3, __ldg(p2 + 192 + lane), a2);
                    a2 = fmaf(s4, __ldg(p2 + 224 + lane), a2);
                }
                ql += 64;
                qh += 32;
                ds += 8;
            }
        };
        if (r0a) acc_row(r0a, a00, a01, a02);
        if (r0b) acc_row(r0b, b00, b01, b02);
        if (r1a) acc_row(r1a, a10, a11, a12);
        if (r1b) acc_row(r1b, b10, b11, b12);
    }
    auto emit3 = [&](float a0, float a1, float a2, float* Y, int m, int row) {
        a0 = warp_sum(a0);
        a1 = warp_sum(a1);
        a2 = warp_sum(a2);
        if (lane != 0) return;
        write_y(Y, row, a0, 0);
        write_y(Y + m, row, a1, 0);
        write_y(Y + 2 * m, row, a2, 0);
    };
    if (row0 < m1) emit3(a00, a01, a02, Y1, m1, row0);
    if (row0 < m2) emit3(b00, b01, b02, Y2, m2, row0);
    if (row1 < m1) emit3(a10, a11, a12, Y1, m1, row1);
    if (row1 < m2) emit3(b10, b11, b12, Y2, m2, row1);
}

__device__ __forceinline__ void acc_q6k_soa_4x(const uint8_t* row, const float* x0, const float* x1, const float* x2,
                                               const float* x3, int nb, int lane, float& a0, float& a1, float& a2,
                                               float& a3);

__global__ void __launch_bounds__(256, 2) gemv_q6k_soa_f32_t4_2row_k(const uint8_t* W, const float* X, float* Y,
                                                                     int m, int n, int add) {
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int nb = n / 256;
    const int is = lane / 16;
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a03 = 0.f, a10 = 0.f, a11 = 0.f, a12 = 0.f, a13 = 0.f;
    const uint8_t* r0 = (row0 < m && nb > 0)
                            ? (W + static_cast<size_t>(row0) * static_cast<size_t>(nb) * kQ6KSoaBsz)
                            : nullptr;
    const uint8_t* r1 = (row1 < m && nb > 0)
                            ? (W + static_cast<size_t>(row1) * static_cast<size_t>(nb) * kQ6KSoaBsz)
                            : nullptr;
    const float* x0 = X;
    const float* x1 = X + n;
    const float* x2 = X + 2 * n;
    const float* x3 = X + 3 * n;
    for (int b = 0; b < nb; ++b) {
        if (r0)
            acc_q6k_soa_4x(r0 + static_cast<size_t>(b) * kQ6KSoaBsz, x0 + b * 256, x1 + b * 256, x2 + b * 256,
                           x3 + b * 256, 1, lane, a00, a01, a02, a03);
        if (r1)
            acc_q6k_soa_4x(r1 + static_cast<size_t>(b) * kQ6KSoaBsz, x0 + b * 256, x1 + b * 256, x2 + b * 256,
                           x3 + b * 256, 1, lane, a10, a11, a12, a13);
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        a01 = warp_sum(a01);
        a02 = warp_sum(a02);
        a03 = warp_sum(a03);
        if (lane == 0) {
            write_y(Y, row0, a00, add);
            write_y(Y + m, row0, a01, add);
            write_y(Y + 2 * m, row0, a02, add);
            write_y(Y + 3 * m, row0, a03, add);
        }
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        a11 = warp_sum(a11);
        a12 = warp_sum(a12);
        a13 = warp_sum(a13);
        if (lane == 0) {
            write_y(Y, row1, a10, add);
            write_y(Y + m, row1, a11, add);
            write_y(Y + 2 * m, row1, a12, add);
            write_y(Y + 3 * m, row1, a13, add);
        }
    }
}

__device__ __forceinline__ void acc_q4k_soa_4x(const uint8_t* row, const float* x0, const float* x1, const float* x2,
                                               const float* x3, int nb, int lane, float& a0, float& a1, float& a2,
                                               float& a3) {
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ4KSoaBsz;
        const __half* ds = reinterpret_cast<const __half*>(blk);
        const __half* dm = reinterpret_cast<const __half*>(blk + 16);
        const uint8_t* q = blk + 32;
        const float* b0 = x0 + b * 256;
        const float* b1 = x1 + b * 256;
        const float* b2 = x2 + b * 256;
        const float* b3 = x3 + b * 256;
        const uint32_t qpack = (static_cast<uint32_t>(__ldcs(q + lane))) |
                               (static_cast<uint32_t>(__ldcs(q + 32 + lane)) << 8) |
                               (static_cast<uint32_t>(__ldcs(q + 64 + lane)) << 16) |
                               (static_cast<uint32_t>(__ldcs(q + 96 + lane)) << 24);
#pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            const uint8_t qq = static_cast<uint8_t>(qpack >> (grp * 8));
            const float w0 = __half2float(ds[grp * 2]) * static_cast<float>(qq & 15) - __half2float(dm[grp * 2]);
            const float w1 = __half2float(ds[grp * 2 + 1]) * static_cast<float>(qq >> 4) - __half2float(dm[grp * 2 + 1]);
            a0 = fmaf(w0, __ldg(b0 + lane), a0);
            a0 = fmaf(w1, __ldg(b0 + 32 + lane), a0);
            a1 = fmaf(w0, __ldg(b1 + lane), a1);
            a1 = fmaf(w1, __ldg(b1 + 32 + lane), a1);
            a2 = fmaf(w0, __ldg(b2 + lane), a2);
            a2 = fmaf(w1, __ldg(b2 + 32 + lane), a2);
            a3 = fmaf(w0, __ldg(b3 + lane), a3);
            a3 = fmaf(w1, __ldg(b3 + 32 + lane), a3);
            b0 += 64;
            b1 += 64;
            b2 += 64;
            b3 += 64;
        }
    }
}

__device__ __forceinline__ void acc_q6k_soa_4x(const uint8_t* row, const float* x0, const float* x1, const float* x2,
                                               const float* x3, int nb, int lane, float& a0, float& a1, float& a2,
                                               float& a3) {
    const int is = lane / 16;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + static_cast<size_t>(b) * kQ6KSoaBsz;
        const __half* ds = reinterpret_cast<const __half*>(blk);
        const uint8_t* ql = blk + 32;
        const uint8_t* qh = blk + 160;
        const float* b0 = x0 + b * 256;
        const float* b1 = x1 + b * 256;
        const float* b2 = x2 + b * 256;
        const float* b3 = x3 + b * 256;
#pragma unroll
        for (int n128 = 0; n128 < 2; ++n128) {
            const int q1 = static_cast<int>((ql[lane] & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32;
            const int q2 = static_cast<int>((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32;
            const int q3 = static_cast<int>((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32;
            const int q4 = static_cast<int>((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4)) - 32;
            const float s1 = __half2float(ds[is]) * static_cast<float>(q1);
            const float s2 = __half2float(ds[is + 2]) * static_cast<float>(q2);
            const float s3 = __half2float(ds[is + 4]) * static_cast<float>(q3);
            const float s4 = __half2float(ds[is + 6]) * static_cast<float>(q4);
            a0 = fmaf(s1, __ldg(b0 + lane), a0);
            a0 = fmaf(s2, __ldg(b0 + 32 + lane), a0);
            a0 = fmaf(s3, __ldg(b0 + 64 + lane), a0);
            a0 = fmaf(s4, __ldg(b0 + 96 + lane), a0);
            a1 = fmaf(s1, __ldg(b1 + lane), a1);
            a1 = fmaf(s2, __ldg(b1 + 32 + lane), a1);
            a1 = fmaf(s3, __ldg(b1 + 64 + lane), a1);
            a1 = fmaf(s4, __ldg(b1 + 96 + lane), a1);
            a2 = fmaf(s1, __ldg(b2 + lane), a2);
            a2 = fmaf(s2, __ldg(b2 + 32 + lane), a2);
            a2 = fmaf(s3, __ldg(b2 + 64 + lane), a2);
            a2 = fmaf(s4, __ldg(b2 + 96 + lane), a2);
            a3 = fmaf(s1, __ldg(b3 + lane), a3);
            a3 = fmaf(s2, __ldg(b3 + 32 + lane), a3);
            a3 = fmaf(s3, __ldg(b3 + 64 + lane), a3);
            a3 = fmaf(s4, __ldg(b3 + 96 + lane), a3);
            ql += 64;
            qh += 32;
            ds += 8;
            b0 += 128;
            b1 += 128;
            b2 += 128;
            b3 += 128;
        }
    }
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

// T=2 spec verify: one W read, two x vectors, two rows per warp.
// x0 in smem at T=1 tile (32 KiB, 2-block/SM). x1 via __ldg so 5120/6144
// are one pass — the old 256-col dual-smem tile paid ~20 syncs per GEMV.
template <int Bsz>
__global__ void gemv_qk_t2_2row_k(const uint8_t* W, const float* X, float* Y, int m, int n, int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int tile = n < kXsTile ? n : kXsTile;
    const int nb_all = n / 256;
    float a00 = 0.f, a01 = 0.f, a10 = 0.f, a11 = 0.f;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        const int nb = tn / 256;
        // SOA 2x paths ldg both x; skip the 20–32 KiB tile so more blocks fit.
        const bool soa2 = (Bsz == kQ4KSoaBsz || Bsz == kQ6KSoaBsz);
        if (!soa2) {
            load_x_tile(xs, X + t0, tn);
            __syncthreads();
        }
        const float* x0p = soa2 ? (X + t0) : xs;
        const float* x1 = X + n + t0;
        if (row0 < m && nb > 0) {
            const uint8_t* r0 = W + (static_cast<size_t>(row0) * nb_all + static_cast<size_t>(t0 / 256)) * Bsz;
            if (Bsz == kQ4KSoaBsz)
                acc_q4k_soa_2x(r0, x0p, x1, nb, lane, a00, a01);
            else if (Bsz == kQ6KSoaBsz)
                acc_q6k_soa_2x(r0, x0p, x1, nb, lane, a00, a01);
            else if (Bsz == kQ4KBsz) {
                a00 += acc_q4k_row(r0, xs, nb, lane);
                a01 += acc_q4k_row(r0, x1, nb, lane);
            } else if (Bsz == kQ6KBsz) {
                a00 += acc_q6k_row(r0, xs, nb, lane);
                a01 += acc_q6k_row(r0, x1, nb, lane);
            } else if (Bsz == kQ5KBsz) {
                a00 += acc_q5k_row(r0, xs, nb, lane);
                a01 += acc_q5k_row(r0, x1, nb, lane);
            } else {
                a00 += acc_q5k_soa_row(r0, xs, nb, lane);
                a01 += acc_q5k_soa_row(r0, x1, nb, lane);
            }
            if (row1 < m) {
                const uint8_t* r1 = W + (static_cast<size_t>(row1) * nb_all + static_cast<size_t>(t0 / 256)) * Bsz;
                if (Bsz == kQ4KSoaBsz)
                    acc_q4k_soa_2x(r1, x0p, x1, nb, lane, a10, a11);
                else if (Bsz == kQ6KSoaBsz)
                    acc_q6k_soa_2x(r1, x0p, x1, nb, lane, a10, a11);
                else if (Bsz == kQ4KBsz) {
                    a10 += acc_q4k_row(r1, xs, nb, lane);
                    a11 += acc_q4k_row(r1, x1, nb, lane);
                } else if (Bsz == kQ6KBsz) {
                    a10 += acc_q6k_row(r1, xs, nb, lane);
                    a11 += acc_q6k_row(r1, x1, nb, lane);
                } else if (Bsz == kQ5KBsz) {
                    a10 += acc_q5k_row(r1, xs, nb, lane);
                    a11 += acc_q5k_row(r1, x1, nb, lane);
                } else {
                    a10 += acc_q5k_soa_row(r1, xs, nb, lane);
                    a11 += acc_q5k_soa_row(r1, x1, nb, lane);
                }
            }
        }
        if (t0 + tile < n) __syncthreads();
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        a01 = warp_sum(a01);
        if (lane == 0) {
            write_y(Y, row0, a00, add);
            write_y(Y + m, row0, a01, add);
        }
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        a11 = warp_sum(a11);
        if (lane == 0) {
            write_y(Y, row1, a10, add);
            write_y(Y + m, row1, a11, add);
        }
    }
}

// T=3 prefill: x0 in smem, x1/x2 from L2, one W stream, two rows / warp.
template <int Bsz>
__global__ void gemv_qk_t3_2row_k(const uint8_t* W, const float* X, float* Y, int m, int n, int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int tile = n < kXsTile ? n : kXsTile;
    const int nb_all = n / 256;
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a10 = 0.f, a11 = 0.f, a12 = 0.f;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        const int nb = tn / 256;
        const bool soa3 = (Bsz == kQ4KSoaBsz || Bsz == kQ6KSoaBsz);
        if (!soa3) {
            load_x_tile(xs, X + t0, tn);
            __syncthreads();
        }
        const float* x0p = soa3 ? (X + t0) : xs;
        const float* x1 = X + n + t0;
        const float* x2 = X + 2 * n + t0;
        if (row0 < m && nb > 0) {
            const uint8_t* r0 = W + (static_cast<size_t>(row0) * nb_all + static_cast<size_t>(t0 / 256)) * Bsz;
            if (Bsz == kQ4KSoaBsz)
                acc_q4k_soa_3x(r0, x0p, x1, x2, nb, lane, a00, a01, a02);
            else if (Bsz == kQ6KSoaBsz)
                acc_q6k_soa_3x(r0, x0p, x1, x2, nb, lane, a00, a01, a02);
            else {
                a00 += acc_q4k_soa_row(r0, xs, nb, lane);
                a01 += acc_q4k_soa_row(r0, x1, nb, lane);
                a02 += acc_q4k_soa_row(r0, x2, nb, lane);
            }
            if (row1 < m) {
                const uint8_t* r1 = W + (static_cast<size_t>(row1) * nb_all + static_cast<size_t>(t0 / 256)) * Bsz;
                if (Bsz == kQ4KSoaBsz)
                    acc_q4k_soa_3x(r1, x0p, x1, x2, nb, lane, a10, a11, a12);
                else if (Bsz == kQ6KSoaBsz)
                    acc_q6k_soa_3x(r1, x0p, x1, x2, nb, lane, a10, a11, a12);
                else {
                    a10 += acc_q4k_soa_row(r1, xs, nb, lane);
                    a11 += acc_q4k_soa_row(r1, x1, nb, lane);
                    a12 += acc_q4k_soa_row(r1, x2, nb, lane);
                }
            }
        }
        if (t0 + tile < n) __syncthreads();
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        a01 = warp_sum(a01);
        a02 = warp_sum(a02);
        if (lane == 0) {
            write_y(Y, row0, a00, add);
            write_y(Y + m, row0, a01, add);
            write_y(Y + 2 * m, row0, a02, add);
        }
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        a11 = warp_sum(a11);
        a12 = warp_sum(a12);
        if (lane == 0) {
            write_y(Y, row1, a10, add);
            write_y(Y + m, row1, a11, add);
            write_y(Y + 2 * m, row1, a12, add);
        }
    }
}

// T=4 spec: one W stream, four x via __ldg, two rows / warp. Avoids 4× GEMV.
template <int Bsz>
__global__ void gemv_qk_t4_2row_k(const uint8_t* W, const float* X, float* Y, int m, int n, int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int tile = n < kXsTile ? n : kXsTile;
    const int nb_all = n / 256;
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a03 = 0.f, a10 = 0.f, a11 = 0.f, a12 = 0.f, a13 = 0.f;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        const int nb = tn / 256;
        const bool soa4 = (Bsz == kQ4KSoaBsz || Bsz == kQ6KSoaBsz);
        if (!soa4) {
            load_x_tile(xs, X + t0, tn);
            __syncthreads();
        }
        const float* x0p = soa4 ? (X + t0) : xs;
        const float* x1 = X + n + t0;
        const float* x2 = X + 2 * n + t0;
        const float* x3 = X + 3 * n + t0;
        if (row0 < m && nb > 0) {
            const uint8_t* r0 = W + (static_cast<size_t>(row0) * nb_all + static_cast<size_t>(t0 / 256)) * Bsz;
            if (Bsz == kQ4KSoaBsz)
                acc_q4k_soa_4x(r0, x0p, x1, x2, x3, nb, lane, a00, a01, a02, a03);
            else if (Bsz == kQ6KSoaBsz)
                acc_q6k_soa_4x(r0, x0p, x1, x2, x3, nb, lane, a00, a01, a02, a03);
            if (row1 < m) {
                const uint8_t* r1 = W + (static_cast<size_t>(row1) * nb_all + static_cast<size_t>(t0 / 256)) * Bsz;
                if (Bsz == kQ4KSoaBsz)
                    acc_q4k_soa_4x(r1, x0p, x1, x2, x3, nb, lane, a10, a11, a12, a13);
                else if (Bsz == kQ6KSoaBsz)
                    acc_q6k_soa_4x(r1, x0p, x1, x2, x3, nb, lane, a10, a11, a12, a13);
            }
        }
        if (t0 + tile < n) __syncthreads();
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        a01 = warp_sum(a01);
        a02 = warp_sum(a02);
        a03 = warp_sum(a03);
        if (lane == 0) {
            write_y(Y, row0, a00, add);
            write_y(Y + m, row0, a01, add);
            write_y(Y + 2 * m, row0, a02, add);
            write_y(Y + 3 * m, row0, a03, add);
        }
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        a11 = warp_sum(a11);
        a12 = warp_sum(a12);
        a13 = warp_sum(a13);
        if (lane == 0) {
            write_y(Y, row1, a10, add);
            write_y(Y + m, row1, a11, add);
            write_y(Y + 2 * m, row1, a12, add);
            write_y(Y + 3 * m, row1, a13, add);
        }
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

// Batched Q8 GEMM: X [T, n], Y [T, m]. T<=4 keeps acc in regs (hot path).
__global__ void gemm_q8_soa_k(const int8_t* Q, const __half* scales, const float* X, float* Y, int m, int n,
                              int T, int tile, int add) {
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

// T=5..32: one HBM pass over W. Acc stays in local memory (#pragma unroll 1)
// so occupancy does not collapse the way a 16/32-reg template did.
__global__ void gemm_q8_soa_wide_k(const int8_t* Q, const __half* scales, const float* X, float* Y, int m,
                                   int n, int T, int tile, int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int row = blockIdx.x * warps + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    const int nb = n / 32;
    const int tstride = (tile / 32) * kXsPad;
    float acc[32];
#pragma unroll 1
    for (int t = 0; t < T; ++t) acc[t] = 0.f;
    const int8_t* rowq = row < m ? Q + static_cast<size_t>(row) * n : nullptr;
    const __half* rows = row < m ? scales + static_cast<size_t>(row) * nb : nullptr;
    for (int k0 = 0; k0 < n; k0 += tile) {
        const int kn = n - k0 < tile ? n - k0 : tile;
        for (int i = threadIdx.x; i < T * kn; i += blockDim.x) {
            const int t = i / kn;
            const int j = i - t * kn;
            xs[static_cast<size_t>(t) * tstride + (j >> 5) * kXsPad + (j & 31)] =
                X[static_cast<size_t>(t) * n + k0 + j];
        }
        __syncthreads();
        if (rowq) {
            const int bn = kn / 32;
            for (int b = lane; b < bn; b += 32) {
                const float d = __half2float(__ldcs(rows + k0 / 32 + b));
                const int8_t* q = rowq + k0 + b * 32;
#pragma unroll 1
                for (int t = 0; t < T; ++t) acc[t] += d * q8_dot32(q, xs + t * tstride + xs_off(b));
            }
        }
        __syncthreads();
    }
    if (row < m) {
#pragma unroll 1
        for (int t = 0; t < T; ++t) {
            const float a = warp_sum(acc[t]);
            if (lane == 0) write_y(Y + static_cast<size_t>(t) * m, row, a, add);
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

// HF Qwen3_5RMSNormGated: last dim = head_v_dim (gnorm weight). Tiny (n<256) stays full-vector.
__global__ void gated_rms_heads_k(const float* x, const float* z, const float* gamma, float* y, int n, int T,
                                  float eps, int hd) {
    const int t = static_cast<int>(blockIdx.y);
    const int h = static_cast<int>(blockIdx.x);
    if (t >= T || hd <= 0) return;
    const int nhead = n / hd;
    if (h >= nhead) return;
    const float* xt = x + static_cast<size_t>(t) * n + static_cast<size_t>(h) * hd;
    const float* zt = z + static_cast<size_t>(t) * n + static_cast<size_t>(h) * hd;
    float* yt = y + static_cast<size_t>(t) * n + static_cast<size_t>(h) * hd;
    __shared__ float buf[128];
    float ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += xt[i] * xt[i];
    buf[threadIdx.x] = ss;
    __syncthreads();
    for (int s = static_cast<int>(blockDim.x) / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(buf[0] / static_cast<float>(hd) + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        const float g = gamma ? gamma[i] : 1.f;
        yt[i] = g * (xt[i] * inv) * silu_d(zt[i]);
    }
}

__global__ void gated_rms_k(const float* x, const float* z, const float* gamma, float* y, int n, float eps,
                            int gamma_n);
__global__ void gated_rms_batch_k(const float* x, const float* z, const float* gamma, float* y, int n, int T,
                                  float eps, int gamma_n);

void launch_gated_rms_vec(const float* x, const float* z, const float* gamma, float* y, int n, int T, float eps,
                          int gamma_n) {
    if (T <= 0 || n <= 0) return;
    if (gamma_n <= 0) gamma_n = n;
    if (gamma_n < n && (n % gamma_n) == 0 && n >= 256) {
        static int once = 0;
        if (!once) {
            std::fprintf(stderr, "gdn_rms per_head=1 gnorm_n=%d zdim=%d T=%d\n", gamma_n, n, T);
            once = 1;
        }
        gated_rms_heads_k<<<dim3(n / gamma_n, T), 128>>>(x, z, gamma, y, n, T, eps, gamma_n);
        return;
    }
    if (T == 1)
        gated_rms_k<<<1, 256>>>(x, z, gamma, y, n, eps, gamma_n);
    else
        gated_rms_batch_k<<<T, 256>>>(x, z, gamma, y, n, T, eps, gamma_n);
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

// Per-channel conv snapshot. T=2 used to let head 0 copy the whole buffer
// while other heads were still pushing t=0 — that race broke Q4 restore.
__device__ __forceinline__ void conv_st_copy_ch(float* dst, const float* src, int c, int k) {
    const float* s = src + c * k;
    float* d = dst + c * k;
    if (k == 4) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
        d[3] = s[3];
        return;
    }
    for (int p = 0; p < k; ++p) d[p] = s[p];
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
    const int src = gdn_src_h(h, nk, nv);
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
    const int src = gdn_src_h(h, nk, nv);
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
        if (gdn_qk_conv_owner(h, nk, nv)) {
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
    const int src_h = gdn_src_h(h, nk, nv);
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
    const int src_h = gdn_src_h(h, nk, nv);
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
                                                          const uint8_t* pf, int pf_bytes, int qkv_stride,
                                                          int ab_stride, int S_elems, int conv_elems,
                                                          int o_stride) {
    constexpr int dk = 128, dv = 128;
    const int h = blockIdx.x;
    const int bid = blockIdx.y;
    if (h >= nv) return;
    if (qkv_stride) qkv_raw += static_cast<size_t>(bid) * qkv_stride;
    if (ab_stride) {
        aa += static_cast<size_t>(bid) * ab_stride;
        bb += static_cast<size_t>(bid) * ab_stride;
    }
    if (S_elems) S += static_cast<size_t>(bid) * S_elems;
    if (conv_elems) conv_st += static_cast<size_t>(bid) * conv_elems;
    if (o_stride) o += static_cast<size_t>(bid) * o_stride;
    extern __shared__ float sm[];
    float* q = sm;
    float* k = sm + dk;
    float* Sp = sm + 2 * dk;
    uint16_t* Sh = S + static_cast<size_t>(h) * dk * dv;
    const int qdim = nk * dk;
    const int src = gdn_src_h(h, nk, nv);
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
    if (gdn_qk_conv_owner(h, nk, nv)) {
        for (int i = threadIdx.x; i < dk; i += blockDim.x) {
            conv1d_push(qkv_raw, conv_st, src * dk + i, 4);
            conv1d_push(qkv_raw, conv_st, qdim + src * dk + i, 4);
        }
    }
    gdn_s_store_bf16(Sh, Sp, dk * dv);
}

// T=2/3 spec: keep S in smem across tokens. One launch, one HBM S load/store.
template <int NT>
__global__ void __launch_bounds__(256, 1) gdn_decode_tn_k(const float* aa, const float* bb, uint16_t* S,
                                                          const float* A_log, const float* dt_bias, float* o,
                                                          int nk, int nv, const float* qkv_raw,
                                                          const float* conv_w, float* conv_st,
                                                          uint16_t* S_bak, float* conv_bak, int qkv_dim) {
    constexpr int dk = 128, dv = 128;
    const int h = blockIdx.x;
    if (h >= nv) return;
    extern __shared__ float sm[];
    float* q = sm;
    float* k = sm + dk;
    float* Sp = sm + 2 * dk;
    uint16_t* Sh = S + static_cast<size_t>(h) * dk * dv;
    const int qdim = nk * dk;
    const int src = gdn_src_h(h, nk, nv);
    __shared__ float qbuf[8], kbuf[8], qinv, kinv, beta_h, glog_h;
    gdn_s_load_f32(Sp, Sh, dk * dv);
    for (int t = 0; t < NT; ++t) {
        const float* qkv = qkv_raw + static_cast<size_t>(t) * qkv_dim;
        const float* aat = aa + static_cast<size_t>(t) * nv;
        const float* bbt = bb + static_cast<size_t>(t) * nv;
        float* ot = o + static_cast<size_t>(t) * nv * dv;
        float qss = 0.f, kss = 0.f;
        for (int i = threadIdx.x; i < dk; i += blockDim.x) {
            const float qi = conv1d_mix(qkv, conv_w, conv_st, src * dk + i, 4);
            const float ki = conv1d_mix(qkv, conv_w, conv_st, qdim + src * dk + i, 4);
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
            beta_h = sigmoid_d(bbt[h]);
            glog_h = -expf(A_log[h]) * softplus_d(aat[h] + dt_bias[h]);
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
        const float g = expf(glog_h);
        const float b = beta_h;
        const int j = threadIdx.x;
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
            const float vj = conv1d_mix(qkv, conv_w, conv_st, 2 * qdim + h * dv + j, 4);
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
            ot[h * dv + j] = oj;
            conv1d_push(qkv, conv_st, 2 * qdim + h * dv + j, 4);
        }
        __syncthreads();
        if (gdn_qk_conv_owner(h, nk, nv)) {
            for (int i = threadIdx.x; i < dk; i += blockDim.x) {
                conv1d_push(qkv, conv_st, src * dk + i, 4);
                conv1d_push(qkv, conv_st, qdim + src * dk + i, 4);
            }
        }
        __syncthreads();
        if (t == 0 && S_bak && conv_bak) {
            uint16_t* Sb = S_bak + static_cast<size_t>(h) * dk * dv;
            gdn_s_store_bf16(Sb, Sp, dk * dv);
            if (gdn_qk_conv_owner(h, nk, nv)) {
                for (int i = threadIdx.x; i < dk; i += blockDim.x) {
                    conv_st_copy_ch(conv_bak, conv_st, src * dk + i, 4);
                    conv_st_copy_ch(conv_bak, conv_st, qdim + src * dk + i, 4);
                }
            }
            for (int i = threadIdx.x; i < dv; i += blockDim.x)
                conv_st_copy_ch(conv_bak, conv_st, 2 * qdim + h * dv + i, 4);
        }
        __syncthreads();
    }
    gdn_s_store_bf16(Sh, Sp, dk * dv);
}

// T=4 GDN with a second snap after t=1 so a d0-hit / d1-miss can restore
// persist S/conv without replaying T=2 (replay dirties rejected KV slots).
__global__ void __launch_bounds__(256, 1) gdn_decode_t4_mid_k(const float* aa, const float* bb, uint16_t* S,
                                                             const float* A_log, const float* dt_bias, float* o,
                                                             int nk, int nv, const float* qkv_raw,
                                                             const float* conv_w, float* conv_st,
                                                             uint16_t* S_bak, float* conv_bak, uint16_t* S_mid,
                                                             float* conv_mid, int qkv_dim) {
    constexpr int dk = 128, dv = 128;
    const int h = blockIdx.x;
    if (h >= nv) return;
    extern __shared__ float sm[];
    float* q = sm;
    float* k = sm + dk;
    float* Sp = sm + 2 * dk;
    uint16_t* Sh = S + static_cast<size_t>(h) * dk * dv;
    const int qdim = nk * dk;
    const int src = gdn_src_h(h, nk, nv);
    __shared__ float qbuf[8], kbuf[8], qinv, kinv, beta_h, glog_h;
    gdn_s_load_f32(Sp, Sh, dk * dv);
    for (int t = 0; t < 4; ++t) {
        const float* qkv = qkv_raw + static_cast<size_t>(t) * qkv_dim;
        const float* aat = aa + static_cast<size_t>(t) * nv;
        const float* bbt = bb + static_cast<size_t>(t) * nv;
        float* ot = o + static_cast<size_t>(t) * nv * dv;
        float qss = 0.f, kss = 0.f;
        for (int i = threadIdx.x; i < dk; i += blockDim.x) {
            const float qi = conv1d_mix(qkv, conv_w, conv_st, src * dk + i, 4);
            const float ki = conv1d_mix(qkv, conv_w, conv_st, qdim + src * dk + i, 4);
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
            beta_h = sigmoid_d(bbt[h]);
            glog_h = -expf(A_log[h]) * softplus_d(aat[h] + dt_bias[h]);
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
        const float g = expf(glog_h);
        const float b = beta_h;
        const int j = threadIdx.x;
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
            const float vj = conv1d_mix(qkv, conv_w, conv_st, 2 * qdim + h * dv + j, 4);
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
            ot[h * dv + j] = oj;
            conv1d_push(qkv, conv_st, 2 * qdim + h * dv + j, 4);
        }
        __syncthreads();
        if (gdn_qk_conv_owner(h, nk, nv)) {
            for (int i = threadIdx.x; i < dk; i += blockDim.x) {
                conv1d_push(qkv, conv_st, src * dk + i, 4);
                conv1d_push(qkv, conv_st, qdim + src * dk + i, 4);
            }
        }
        __syncthreads();
        uint16_t* Sdst = nullptr;
        float* Cdst = nullptr;
        if (t == 0 && S_bak && conv_bak) {
            Sdst = S_bak;
            Cdst = conv_bak;
        } else if (t == 1 && S_mid && conv_mid) {
            Sdst = S_mid;
            Cdst = conv_mid;
        }
        if (Sdst && Cdst) {
            uint16_t* Sb = Sdst + static_cast<size_t>(h) * dk * dv;
            gdn_s_store_bf16(Sb, Sp, dk * dv);
            if (gdn_qk_conv_owner(h, nk, nv)) {
                for (int i = threadIdx.x; i < dk; i += blockDim.x) {
                    conv_st_copy_ch(Cdst, conv_st, src * dk + i, 4);
                    conv_st_copy_ch(Cdst, conv_st, qdim + src * dk + i, 4);
                }
            }
            for (int i = threadIdx.x; i < dv; i += blockDim.x)
                conv_st_copy_ch(Cdst, conv_st, 2 * qdim + h * dv + i, 4);
        }
        __syncthreads();
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

__global__ void split_qg_perhead_k(const float* qg, float* q, float* gate, int n_q, int hd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int qn = n_q * hd;
    if (i >= qn) return;
    const int h = i / hd;
    const int d = i - h * hd;
    const int base = h * hd * 2;
    q[i] = qg[base + d];
    gate[i] = qg[base + hd + d];
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

// Official 27B GQA-6 (24Q/4KV hd=256 F16): one block per KV head, Flash attend.
// FlashInfer cannot dispatch group=6; this is the legal Flash path.
__global__ void qk_attn_decode_gqa_k(float* q, const float* k, const float* v, const float* q_norm,
                                     const float* k_norm, float* k_cache, float* v_cache, float* o,
                                     const int* pos, int n_q, int n_kv, int hd, int rotary, float theta,
                                     float eps, int ctx) {
    const int hkv = blockIdx.x;
    if (hkv >= n_kv || hd != 256 || n_kv <= 0 || n_q % n_kv != 0) return;
    const int g = n_q / n_kv;
    // 2D grid (n_kv, g) runs one Q head per block. 1D grid keeps the old serial loop.
    const int qi0 = (gridDim.y > 1) ? static_cast<int>(blockIdx.y) : 0;
    const int qi1 = (gridDim.y > 1) ? qi0 + 1 : g;
    if (qi0 >= g) return;
    const int p = *pos;
    const int Tend = (p + 1) < ctx ? (p + 1) : ctx;
    const int kn = n_kv * hd;
    extern __shared__ float sm[];
    float* khloc = sm;
    __shared__ float inv, wss[8];
    auto rms_rope = [&](float* x, const float* gamma) {
        float ss = 0.f;
        for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += x[i] * x[i];
        ss = warp_sum(ss);
        if ((threadIdx.x & 31) == 0) wss[threadIdx.x >> 5] = ss;
        __syncthreads();
        if (threadIdx.x == 0) {
            float tot = 0.f;
            const int nw = blockDim.x >> 5;
            for (int w = 0; w < nw && w < 8; ++w) tot += wss[w];
            inv = rsqrtf(tot / static_cast<float>(hd) + eps);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < hd; i += blockDim.x) {
            const float gn = gamma ? (1.f + gamma[i]) : 1.f;
            x[i] *= inv * gn;
        }
        __syncthreads();
        rope_apply(x, rotary, static_cast<float>(p), theta, hd);
        __syncthreads();
    };
    const float* ksrc = k + hkv * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) khloc[i] = ksrc[i];
    __syncthreads();
    rms_rope(khloc, k_norm);
    __half* kc = reinterpret_cast<__half*>(k_cache) + static_cast<size_t>(p) * kn + hkv * hd;
    __half* vc = reinterpret_cast<__half*>(v_cache) + static_cast<size_t>(p) * kn + hkv * hd;
    const float* vs = v + hkv * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        kc[i] = __float2half(khloc[i]);
        vc[i] = __float2half(vs[i]);
    }
    __syncthreads();
    const float scale = rsqrtf(static_cast<float>(hd));
    for (int qi = qi0; qi < qi1; ++qi) {
        const int hq = hkv * g + qi;
        float* qh = q + hq * hd;
        rms_rope(qh, q_norm);
        flash_attn_decode_f16_hd256(qh, reinterpret_cast<const __half*>(k_cache),
                                    reinterpret_cast<const __half*>(v_cache), o + hq * hd, Tend, kn,
                                    hkv, scale);
        __syncthreads();
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
                                 __half* v_sc, int q_bstride, int kv_bstride, int cache_bstride) {
    const int b = static_cast<int>(blockIdx.y);
    if (b) {
        if (q_bstride) {
            q += static_cast<size_t>(b) * q_bstride;
            o += static_cast<size_t>(b) * q_bstride;
        }
        if (kv_bstride) {
            k += static_cast<size_t>(b) * kv_bstride;
            v += static_cast<size_t>(b) * kv_bstride;
        }
        pos += b;
        if (cache_bstride && k_cache && v_cache) {
            if (kv_f16 == 1) {
                k_cache = reinterpret_cast<float*>(reinterpret_cast<__half*>(k_cache) +
                                                   static_cast<size_t>(b) * cache_bstride);
                v_cache = reinterpret_cast<float*>(reinterpret_cast<__half*>(v_cache) +
                                                   static_cast<size_t>(b) * cache_bstride);
            } else {
                k_cache += static_cast<size_t>(b) * cache_bstride;
                v_cache += static_cast<size_t>(b) * cache_bstride;
            }
        }
    }
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
    const bool use_flash = (kv_f16 == 1 && hd == 256 && d_use_flash);
    // Flash attend does not use scores[T]. khloc must sit at sm[0] so a
    // 1 KiB launch (no scores) stays in-bounds — ctx<=8k used to put it at sm[ctx].
    float* khloc = sm + ((ctx > 8192 || use_flash) ? 0 : ctx);

    float* qh = q + hq * hd;
    __shared__ float inv, wss[8];
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
        for (int w = 1; w < nw && w < 8; ++w) tot += wss[w];
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
    rope_apply(qh, rotary, pf, theta, hd);

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
        for (int w = 1; w < nw && w < 8; ++w) tot += wss[w];
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
    rope_apply(khloc, rotary, pf, theta, hd);
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
            for (int w = 1; w < nw && w < 8; ++w) m = fmaxf(m, wss[w]);
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
    if (use_flash) {
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
                for (int w = 1; w < nw && w < 8; ++w) tot += wss[w];
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
        // Short-ctx smem scores. Official 27B stores F16 KV (hd=256);
        // reading those halves as float* is the same garbage path as the
        // old T<32 prefill cast (decode residual ~1e13 after one token).
        for (int t = threadIdx.x; t < Tend; t += blockDim.x) {
            float dot = 0.f;
            if (kv_f16 == 1) {
                const __half* kh = reinterpret_cast<const __half*>(k_cache) +
                                   static_cast<size_t>(t) * kn + hkv * hd;
                for (int d = 0; d < hd; ++d) dot += qh[d] * __half2float(kh[d]);
            } else {
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
                if (kv_f16 == 1) {
                    const __half* vh = reinterpret_cast<const __half*>(v_cache) +
                                       static_cast<size_t>(t) * kn + hkv * hd;
                    acc += (scores[t] / sumv) * __half2float(vh[d]);
                } else {
                    const float* vh = v_cache + static_cast<size_t>(t) * kn + hkv * hd;
                    acc += (scores[t] / sumv) * vh[d];
                }
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

__global__ void inc_pos_n_k(int* pos, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) ++pos[i];
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

// Q8 SoA tile → row-major F16. Wout is [mr, n]; source rows start at row0.
__global__ void dequant_q8_soa_f16_k(const int8_t* Q, const __half* scales, __half* W, int n, int row0,
                                     int mr) {
    const size_t i4 = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t n4 = static_cast<size_t>(mr) * static_cast<size_t>(n >> 2);
    if (i4 >= n4) return;
    const int col = static_cast<int>((i4 << 2) % static_cast<size_t>(n));
    const int r = static_cast<int>((i4 << 2) / static_cast<size_t>(n));
    const int row = row0 + r;
    const float d = __half2float(__ldg(scales + static_cast<size_t>(row) * static_cast<size_t>(n >> 5) +
                                      static_cast<size_t>(col >> 5)));
    const char4 q = *reinterpret_cast<const char4*>(Q + static_cast<size_t>(row) * n + col);
    __half2* dst = reinterpret_cast<__half2*>(W + (i4 << 2));
    dst[0] = __floats2half2_rn(d * static_cast<float>(q.x), d * static_cast<float>(q.y));
    dst[1] = __floats2half2_rn(d * static_cast<float>(q.z), d * static_cast<float>(q.w));
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

// Same as e4_dot16_rm but x lives in global / L2 (second T=2 token).
__device__ __forceinline__ float fp8x4_dot_v(uint32_t p, float4 xv) {
#if __CUDA_ARCH__ >= 890
    unsigned h01, h23;
    const unsigned short lo = static_cast<unsigned short>(p);
    const unsigned short hi = static_cast<unsigned short>(p >> 16);
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h01) : "h"(lo));
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h23) : "h"(hi));
    const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h01));
    const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h23));
    return f0.x * xv.x + f0.y * xv.y + f1.x * xv.z + f1.y * xv.w;
#else
    return fp8e4_to_f32(static_cast<uint8_t>(p)) * xv.x + fp8e4_to_f32(static_cast<uint8_t>(p >> 8)) * xv.y +
           fp8e4_to_f32(static_cast<uint8_t>(p >> 16)) * xv.z + fp8e4_to_f32(static_cast<uint8_t>(p >> 24)) * xv.w;
#endif
}

__device__ __forceinline__ float e4_dot16_rm_ldg(uint4 p, const float* x) {
    return fp8x4_dot_v(p.x, __ldg(reinterpret_cast<const float4*>(x))) +
           fp8x4_dot_v(p.y, __ldg(reinterpret_cast<const float4*>(x + 4))) +
           fp8x4_dot_v(p.z, __ldg(reinterpret_cast<const float4*>(x + 8))) +
           fp8x4_dot_v(p.w, __ldg(reinterpret_cast<const float4*>(x + 12)));
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

// T=2 row-major e4 (scales absorbed): one W stream, both X tokens.
// cublasLt n=2 is ~0.5ms on 10240x5120 vs ~0.07ms T=1 GEMV — do not use Lt here.
// x0 stays in smem (T=1 32KiB budget, 2-block/SM). x1 is __ldg from L2 so
// tile stays kFp8XsCap: 5120/6144 are one pass; 10240/17408 lose 1–2 tile syncs
// vs the old kFp8XsCap/2 dual-smem split.
__global__ void __launch_bounds__(512, 2) gemv_fp8_rm_t2_k(const uint8_t* W, const float* X, float* Y, int m,
                                                          int n, int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int tile = n <= kFp8XsCap ? n : kFp8XsCap;
    float a00 = 0.f, a01 = 0.f, a10 = 0.f, a11 = 0.f;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        load_x_tile(xs, X + t0, tn);
        __syncthreads();
        const float* x1 = X + n + t0;
        if (row0 < m) {
            const uint8_t* w0 = W + static_cast<size_t>(row0) * n + t0;
            const uint8_t* w1 = (row1 < m) ? W + static_cast<size_t>(row1) * n + t0 : nullptr;
            for (int j = lane * 16; j + 15 < tn; j += 512) {
                const uint4 p0 = *reinterpret_cast<const uint4*>(w0 + j);
                a00 += e4_dot16_rm(p0, xs + j);
                a01 += e4_dot16_rm_ldg(p0, x1 + j);
                if (w1) {
                    const uint4 p1 = *reinterpret_cast<const uint4*>(w1 + j);
                    a10 += e4_dot16_rm(p1, xs + j);
                    a11 += e4_dot16_rm_ldg(p1, x1 + j);
                }
            }
        }
        if (t0 + tile < n) __syncthreads();
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        if (lane == 0) write_y(Y, row0, a00, add);
        a01 = warp_sum(a01);
        if (lane == 0) write_y(Y + m, row0, a01, add);
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        if (lane == 0) write_y(Y, row1, a10, add);
        a11 = warp_sum(a11);
        if (lane == 0) write_y(Y + m, row1, a11, add);
    }
}

// T=3 prefill: x0 in smem, x1/x2 via __ldg. Same 32KiB / 2-block budget as T=1.
// Three sequential T=1 GEMVs reread ~27GiB; this is one W stream.
__global__ void __launch_bounds__(512, 2) gemv_fp8_rm_t3_k(const uint8_t* W, const float* X, float* Y, int m,
                                                          int n, int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int tile = n <= kFp8XsCap ? n : kFp8XsCap;
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a10 = 0.f, a11 = 0.f, a12 = 0.f;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        load_x_tile(xs, X + t0, tn);
        __syncthreads();
        const float* x1 = X + n + t0;
        const float* x2 = X + 2 * n + t0;
        if (row0 < m) {
            const uint8_t* w0 = W + static_cast<size_t>(row0) * n + t0;
            const uint8_t* w1 = (row1 < m) ? W + static_cast<size_t>(row1) * n + t0 : nullptr;
            for (int j = lane * 16; j + 15 < tn; j += 512) {
                const uint4 p0 = *reinterpret_cast<const uint4*>(w0 + j);
                a00 += e4_dot16_rm(p0, xs + j);
                a01 += e4_dot16_rm_ldg(p0, x1 + j);
                a02 += e4_dot16_rm_ldg(p0, x2 + j);
                if (w1) {
                    const uint4 p1 = *reinterpret_cast<const uint4*>(w1 + j);
                    a10 += e4_dot16_rm(p1, xs + j);
                    a11 += e4_dot16_rm_ldg(p1, x1 + j);
                    a12 += e4_dot16_rm_ldg(p1, x2 + j);
                }
            }
        }
        if (t0 + tile < n) __syncthreads();
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        if (lane == 0) write_y(Y, row0, a00, add);
        a01 = warp_sum(a01);
        if (lane == 0) write_y(Y + m, row0, a01, add);
        a02 = warp_sum(a02);
        if (lane == 0) write_y(Y + 2 * m, row0, a02, add);
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        if (lane == 0) write_y(Y, row1, a10, add);
        a11 = warp_sum(a11);
        if (lane == 0) write_y(Y + m, row1, a11, add);
        a12 = warp_sum(a12);
        if (lane == 0) write_y(Y + 2 * m, row1, a12, add);
    }
}

// T=4 spec: x0 in smem, x1/x2/x3 via __ldg. One W stream vs four T=1 GEMVs.
__global__ void __launch_bounds__(512, 2) gemv_fp8_rm_t4_k(const uint8_t* W, const float* X, float* Y, int m,
                                                          int n, int add) {
    extern __shared__ float xs[];
    const int warps = blockDim.x / 32;
    const int pair = blockIdx.x * warps + (threadIdx.x / 32);
    const int row0 = pair * 2;
    const int row1 = row0 + 1;
    const int lane = threadIdx.x & 31;
    const int tile = n <= kFp8XsCap ? n : kFp8XsCap;
    float a00 = 0.f, a01 = 0.f, a02 = 0.f, a03 = 0.f, a10 = 0.f, a11 = 0.f, a12 = 0.f, a13 = 0.f;
    for (int t0 = 0; t0 < n; t0 += tile) {
        const int tn = n - t0 < tile ? n - t0 : tile;
        load_x_tile(xs, X + t0, tn);
        __syncthreads();
        const float* x1 = X + n + t0;
        const float* x2 = X + 2 * n + t0;
        const float* x3 = X + 3 * n + t0;
        if (row0 < m) {
            const uint8_t* w0 = W + static_cast<size_t>(row0) * n + t0;
            const uint8_t* w1 = (row1 < m) ? W + static_cast<size_t>(row1) * n + t0 : nullptr;
            for (int j = lane * 16; j + 15 < tn; j += 512) {
                const uint4 p0 = *reinterpret_cast<const uint4*>(w0 + j);
                a00 += e4_dot16_rm(p0, xs + j);
                a01 += e4_dot16_rm_ldg(p0, x1 + j);
                a02 += e4_dot16_rm_ldg(p0, x2 + j);
                a03 += e4_dot16_rm_ldg(p0, x3 + j);
                if (w1) {
                    const uint4 p1 = *reinterpret_cast<const uint4*>(w1 + j);
                    a10 += e4_dot16_rm(p1, xs + j);
                    a11 += e4_dot16_rm_ldg(p1, x1 + j);
                    a12 += e4_dot16_rm_ldg(p1, x2 + j);
                    a13 += e4_dot16_rm_ldg(p1, x3 + j);
                }
            }
        }
        if (t0 + tile < n) __syncthreads();
    }
    if (row0 < m) {
        a00 = warp_sum(a00);
        if (lane == 0) write_y(Y, row0, a00, add);
        a01 = warp_sum(a01);
        if (lane == 0) write_y(Y + m, row0, a01, add);
        a02 = warp_sum(a02);
        if (lane == 0) write_y(Y + 2 * m, row0, a02, add);
        a03 = warp_sum(a03);
        if (lane == 0) write_y(Y + 3 * m, row0, a03, add);
    }
    if (row1 < m) {
        a10 = warp_sum(a10);
        if (lane == 0) write_y(Y, row1, a10, add);
        a11 = warp_sum(a11);
        if (lane == 0) write_y(Y + m, row1, a11, add);
        a12 = warp_sum(a12);
        if (lane == 0) write_y(Y + 2 * m, row1, a12, add);
        a13 = warp_sum(a13);
        if (lane == 0) write_y(Y + 3 * m, row1, a13, add);
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

// Q8 SoA → smem F16 tile × X panel, tensor-core MMA. One HBM pass over W per
// T-panel (BN=8/16/32/64). T=4 scalar GEMM rereads W B/4 times and plateaus.
template <int BN>
__global__ void __launch_bounds__(256, 2) gemm_q8_soa_tc_k(const int8_t* Q, const __half* scales,
                                                           const float* X, float* Y, int m, int n, int T,
                                                           int add) {
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
    const int nb = n >> 5;
    for (int k0 = 0; k0 < n; k0 += BK) {
        const int b0 = k0 >> 5;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const int row = wrow + r;
            const int col = lane << 2;
            const float s =
                (row < m) ? __half2float(__ldg(scales + static_cast<size_t>(row) * nb + b0 + (lane >> 3)))
                          : 0.f;
            int qi = 0;
            if (row < m)
                qi = __ldcs(reinterpret_cast<const int*>(Q + static_cast<size_t>(row) * n + k0 + col));
            const signed char* b = reinterpret_cast<const signed char*>(&qi);
            __half* dst = Wf + (warp * 16 + r) * WLD + col;
            dst[0] = __float2half(s * static_cast<float>(b[0]));
            dst[1] = __float2half(s * static_cast<float>(b[1]));
            dst[2] = __float2half(s * static_cast<float>(b[2]));
            dst[3] = __float2half(s * static_cast<float>(b[3]));
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
    (void)Q;
    (void)scales;
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

__global__ void bf16_to_f32_k(const uint16_t* x, float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = bf16_to_f32(x[i]);
}

__global__ void split_qg_batch_k(const float* qg, float* q, float* gate, int n_q, int hd, int T) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int qn = n_q * hd;
    if (t >= T || i >= qn) return;
    const size_t to = static_cast<size_t>(t) * qn;
    const size_t ti = static_cast<size_t>(t) * qn * 2;
    if (hd >= 64) {
        const int h = i / hd;
        const int d = i - h * hd;
        const int base = h * hd * 2;
        q[to + i] = qg[ti + base + d];
        gate[to + i] = qg[ti + base + hd + d];
    } else {
        q[to + i] = qg[ti + i];
        gate[to + i] = qg[ti + qn + i];
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
    rope_apply(v, rotary_dim, p, theta, head_dim);
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
        saa[tid] = 1.f;
        spp[tid] = 0.f;
        for (int w = 0; w < 8; ++w) red[w][tid] = 0.f;
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

__global__ void attn_pack_kv_f_tile_k(const __half* cache, float* out, int s0, int bc, int tend, int n_kv,
                                      int hd) {
    const int n = n_kv * bc * hd;
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    const int d = i - (i / hd) * hd;
    const int sl = (i / hd) % bc;
    const int hkv = (i / hd) / bc;
    const int s = s0 + sl;
    if (s >= tend || hkv >= n_kv) {
        for (int k = 0; k < 4 && i + k < n; ++k) out[i + k] = 0.f;
        return;
    }
    const __half* src = cache + (static_cast<size_t>(s) * n_kv + hkv) * hd + d;
    if (d + 3 < hd) {
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(src));
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(src + 2));
        *reinterpret_cast<float4*>(out + i) = make_float4(a.x, a.y, b.x, b.y);
    } else {
        for (int k = 0; k < 4 && i + k < n; ++k) out[i + k] = __half2float(src[k]);
    }
}

__global__ void attn_fa_init_k(float* O, float* mi, float* li, int n_rows, int hd) {
    const int n = n_rows * hd;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) O[i] = 0.f;
    if (i < n_rows) {
        mi[i] = -1e30f;
        li[i] = 0.f;
    }
}

// Online-softmax tile: S[row, 0:bc] -> P, scale O by exp(mi-m2), update mi/li.
__global__ void attn_fa_update_k(float* S, float* O, float* mi, float* li, int rows, int bc, int hd, int s0,
                                 int pos0, int rep, int qrows) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int local = qrows > 0 ? row - (row / qrows) * qrows : row;
    const int rdiv = rep > 0 ? rep : 1;
    const int t = local / rdiv;
    const int qpos = pos0 + t;
    int valid = qpos + 1 - s0;
    if (valid < 0) valid = 0;
    if (valid > bc) valid = bc;
    float* Sp = S + static_cast<size_t>(row) * bc;
    float* Op = O + static_cast<size_t>(row) * hd;
    if (valid <= 0) {
        for (int i = threadIdx.x; i < bc; i += blockDim.x) Sp[i] = 0.f;
        return;
    }
    float mx = -1e30f;
    for (int i = threadIdx.x; i < valid; i += blockDim.x) mx = fmaxf(mx, Sp[i]);
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
        red[0] = fmaxf(m, mi[row]);
    }
    __syncthreads();
    mx = red[0];
    const float alpha = __expf(mi[row] - mx);
    for (int d = threadIdx.x; d < hd; d += blockDim.x) Op[d] *= alpha;
    float sum = 0.f;
    for (int i = threadIdx.x; i < valid; i += blockDim.x) {
        const float e = __expf(Sp[i] - mx);
        Sp[i] = e;
        sum += e;
    }
    for (int i = valid + threadIdx.x; i < bc; i += blockDim.x) Sp[i] = 0.f;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    if (lane == 0) red[wid] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float z = red[0];
        const int nw = (blockDim.x + 31) >> 5;
        for (int w = 1; w < nw && w < 8; ++w) z += red[w];
        li[row] = alpha * li[row] + z;
        mi[row] = mx;
    }
}

__global__ void attn_fa_finalize_k(float* O, const float* li, int n_rows, int hd) {
    const int n = n_rows * hd;
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    const int row = i / hd;
    const float inv = li[row] > 0.f ? 1.f / li[row] : 0.f;
    if ((i - row * hd) + 3 < hd) {
        float4 v = *reinterpret_cast<float4*>(O + i);
        v.x *= inv;
        v.y *= inv;
        v.z *= inv;
        v.w *= inv;
        *reinterpret_cast<float4*>(O + i) = v;
    } else {
        for (int k = 0; k < 4 && i + k < n; ++k) O[i + k] *= inv;
    }
}

extern cublasHandle_t g_blas;
// Shared-Q / tiled GEMM: pack Q once per KV group, consume K/V in Bc tiles,
// online softmax (no full T×tend score matrix). Leftover T=1 k-major unused.
bool launch_attn_gqa_gemm(const float* q, const __half* kc, const __half* vc, float* o, float* scores,
                          size_t scores_f, float* scratch, size_t scratch_f, int pos0, int T, int n_q,
                          int n_kv, int hd) {
    if (!g_blas || !q || !kc || !vc || !o || !scores || !scratch || T <= 0 || hd != 256) return false;
    if (n_kv <= 0 || n_q % n_kv != 0) return false;
    const int rep = n_q / n_kv;
    const int tend = pos0 + T;
    if (rep < 2 || tend <= 0) return false;
    auto scratch_need = [&](int rows, int bc) -> size_t {
        return static_cast<size_t>(n_kv) * rows * hd * 2 + static_cast<size_t>(n_kv) * bc * hd * 2 +
               static_cast<size_t>(n_kv) * rows * 2 + 32;
    };
    int ts = T;
    int Bc = 256;
    while (ts >= 1) {
        const int rows = ts * rep;
        const size_t scn = static_cast<size_t>(n_kv) * rows * static_cast<size_t>(Bc);
        if (scn <= scores_f && scratch_need(rows, Bc) <= scratch_f) break;
        if (Bc > 64) {
            Bc /= 2;
            continue;
        }
        ts /= 2;
        Bc = 256;
    }
    if (ts < 1) return false;
    const float alpha = 0.0625f;
    const float beta0 = 0.f;
    const float one = 1.f;
    cublasSetStream(g_blas, cudaStreamPerThread);
    for (int t0 = 0; t0 < T; t0 += ts) {
        const int nt = T - t0 < ts ? T - t0 : ts;
        const int rows = nt * rep;
        const size_t qn = static_cast<size_t>(n_kv) * rows * hd;
        float* qf = scratch;
        float* of = qf + qn;
        float* kf = of + qn;
        float* vf = kf + static_cast<size_t>(n_kv) * Bc * hd;
        float* mi = vf + static_cast<size_t>(n_kv) * Bc * hd;
        float* li = mi + static_cast<size_t>(n_kv) * rows;
        const int qbl = (static_cast<int>(qn) / 4 + 255) / 256;
        attn_pack_q_f_all_k<<<qbl, 256>>>(q, qf, nt, t0, n_q, hd, n_kv, rep);
        const int n_rows = n_kv * rows;
        attn_fa_init_k<<<(n_rows * hd + 255) / 256, 256>>>(of, mi, li, n_rows, hd);
        const long long stride_q = static_cast<long long>(rows) * hd;
        for (int s0 = 0; s0 < tend; s0 += Bc) {
            const int bc = tend - s0 < Bc ? tend - s0 : Bc;
            const int kn = n_kv * bc * hd;
            const int kbl = (kn / 4 + 255) / 256;
            attn_pack_kv_f_tile_k<<<kbl, 256>>>(kc, kf, s0, bc, tend, n_kv, hd);
            attn_pack_kv_f_tile_k<<<kbl, 256>>>(vc, vf, s0, bc, tend, n_kv, hd);
            const long long stride_k = static_cast<long long>(bc) * hd;
            const long long stride_s = static_cast<long long>(rows) * bc;
            bool ok = n_kv > 1 &&
                      cublasGemmStridedBatchedEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, bc, rows, hd, &alpha, kf,
                                                 CUDA_R_32F, hd, stride_k, qf, CUDA_R_32F, hd, stride_q,
                                                 &beta0, scores, CUDA_R_32F, bc, stride_s, n_kv,
                                                 CUBLAS_COMPUTE_32F_FAST_TF32,
                                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS;
            if (!ok) {
                for (int h = 0; h < n_kv; ++h) {
                    if (cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, bc, rows, hd, &alpha,
                                    kf + h * stride_k, hd, qf + h * stride_q, hd, &beta0,
                                    scores + h * stride_s, bc) != CUBLAS_STATUS_SUCCESS)
                        return false;
                }
            }
            attn_fa_update_k<<<n_rows, bc >= 256 ? 256 : 128>>>(scores, of, mi, li, n_rows, bc, hd, s0,
                                                                pos0 + t0, rep, rows);
            ok = n_kv > 1 &&
                 cublasGemmStridedBatchedEx(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, hd, rows, bc, &one, vf,
                                            CUDA_R_32F, hd, stride_k, scores, CUDA_R_32F, bc, stride_s, &one,
                                            of, CUDA_R_32F, hd, stride_q, n_kv, CUBLAS_COMPUTE_32F_FAST_TF32,
                                            CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS;
            if (!ok) {
                for (int h = 0; h < n_kv; ++h) {
                    if (cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, hd, rows, bc, &one, vf + h * stride_k,
                                    hd, scores + h * stride_s, bc, &one, of + h * stride_q, hd) !=
                        CUBLAS_STATUS_SUCCESS)
                        return false;
                }
            }
        }
        attn_fa_finalize_k<<<qbl, 256>>>(of, li, n_rows, hd);
        attn_unpack_o_all_k<<<qbl, 256>>>(of, o, nt, t0, n_q, hd, n_kv, rep);
    }
    return true;
}

__global__ void attn_prefill_h_k(const float* q, const __half* k_cache, const __half* v_cache, float* o,
                                 int pos0, int T, int n_q, int n_kv, int head_dim) {
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
    extern __shared__ float scores[];
    for (int s = threadIdx.x; s < Tend; s += blockDim.x) {
        float dot = 0.f;
        const __half* kh = k_cache + static_cast<size_t>(s) * n_kv * head_dim + hkv * head_dim;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * __half2float(kh[d]);
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
            const __half* vh = v_cache + static_cast<size_t>(s) * n_kv * head_dim + hkv * head_dim;
            acc += (scores[s] / sumv) * __half2float(vh[d]);
        }
        oh[d] = acc;
    }
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
                for (int w = 1; w < nw && w < 8; ++w) tot += wss[w];
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
    // Grid-stride: old <<<1,32>>> only wrote slots 0..31, so B>32 greedy stayed 0.
    for (int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); t < T;
         t += static_cast<int>(blockDim.x * gridDim.x)) {
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
    rope_apply(v, rotary_dim, p, theta, head_dim);
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
    const int8_t* i8 = nullptr;   // T=4 Q6 unpack (q-32), nullptr = packed GEMV
    const __half* i8_sc = nullptr; // per-16 fp16 scales for i8
};

cublasHandle_t g_blas = nullptr;
__half* g_xf16 = nullptr;
int g_xf16_n = 0;
__half* g_wf16 = nullptr;
size_t g_wf16_n = 0;
float* g_yadd = nullptr;
int g_yadd_n = 0;
const float* g_xf16_src = nullptr;
int g_xf16_src_n = 0;
cudaStream_t g_dq_stream = nullptr;
cudaEvent_t g_dq_done[2] = {};
cudaEvent_t g_gemm_done[2] = {};
int g_wf_slot = 0;
bool g_fp8_tc = true;
bool g_fp8_e4m3_mma = false;
bool g_fp8_shared_tc = false;
bool g_q8_tc = false;
bool g_q8_cublas_add = false;
bool g_fp8_e4_tc = false;
bool g_fp8_e4_bn256 = false;
bool g_fp8_e4_bk512 = false;
bool g_fp8_e4_pipe = false;
bool g_fp8_e4_occ = false;
bool g_fp8_e4_o2 = false;
bool g_cublas_fp8_e4 = false;
int g_lt_min_T = 8; // 2 inside launch_spec_chunk so T=2 reads W once
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
    if (!g_lt || !w.data || T < 2 || w.rows < 128 || w.cols < 128) return;
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
int8_t* g_xq = nullptr;
__half* g_xsc = nullptr;
int32_t* g_xsum = nullptr;
int g_xq_n = 0;
int g_xgen = 1;
int g_xq_gen = 0;
const float* g_xq_ptr = nullptr;
int g_xq_cols = 0, g_xq_T = 0;
bool g_capturing = false;
bool g_spec_prof = false;
float g_spec_ms_lin = 0, g_spec_ms_gdn = 0, g_spec_ms_attn = 0, g_spec_ms_mlp = 0, g_spec_ms_lm = 0;

void note_x_written() { ++g_xgen; }

bool ensure_xq(const float* X, int n, int T) {
    if (!g_xq || !g_xsc || !X || n <= 0 || T <= 0) return false;
    if (T > 1 && !g_xsum) return false;
    if (n * T > g_xq_n) return false;
    // Cache hits during graph capture too: first GEMV on this X inserts the
    // quant node; later GEMVs (wqkv/wz/wa/wb) reuse g_xq. Re-quant every
    // call bloated the spec graph with identical writes to the same buffer.
    if (X == g_xq_ptr && n == g_xq_cols && T == g_xq_T && g_xq_gen == g_xgen) return true;
    if (T == 1) {
        const int nblk = n >> 5;
        quant_x_q8_k<<<(nblk + 127) / 128, 128>>>(X, g_xq, g_xsc, n);
    } else {
        const int nblk = (n >> 5) * T;
        quant_x_q8_n_k<<<(nblk + 127) / 128, 128>>>(X, g_xq, g_xsc, g_xsum, n, T);
    }
    g_xq_ptr = X;
    g_xq_cols = n;
    g_xq_T = T;
    g_xq_gen = g_xgen;
    return true;
}

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
void ensure_dq_stream();

void launch_linear_pair(const GpuW& a, const GpuW& b, const float* X, float* Ya, float* Yb, int T) {
    if (T == 2 && a.q == QuantKind::Q6_K && b.q == QuantKind::Q4_K && a.qk_soa && b.qk_soa &&
        a.cols == b.cols && a.cols > 0 && (a.cols % 256) == 0 && a.rows >= 4096 && b.rows >= 4096 &&
        a.data && b.data && X && Ya && Yb && g_xq && g_xsc && a.cols * 2 <= g_xq_n &&
        ensure_xq(X, a.cols, 2)) {
        static int once = 0;
        if (!once) {
            once = 1;
            std::fprintf(stderr, "q6q4_t2=1 ild=1 1row=1\n");
        }
        rapidllm::cuda_gemv::launch_q6k_f32_t2_1row(a.data, X, Ya, a.rows, a.cols, 0);
        rapidllm::cuda_gemv::launch_q4k_q8_t2_1row(b.data, g_xq, g_xsc, Yb, b.rows, b.cols, 0);
        return;
    }
    if (T == 12 && a.q == QuantKind::Q6_K && b.q == QuantKind::Q4_K && a.qk_soa && b.qk_soa &&
        a.cols == b.cols && a.cols > 0 && (a.cols % 256) == 0 && a.rows >= 4096 && b.rows >= 4096 &&
        a.data && b.data && X && Ya && Yb && g_xq && g_xsc && a.cols * 12 <= g_xq_n &&
        ensure_xq(X, a.cols, 12)) {
        static int once = 0;
        if (!once) {
            once = 1;
            std::fprintf(stderr, "q6q4_t12=1 ild=1 1row=1\n");
        }
        rapidllm::cuda_gemv::launch_q6k_f32_t12_1row(a.data, X, Ya, a.rows, a.cols, 0);
        rapidllm::cuda_gemv::launch_q4k_q8_t12_1row_pipe(b.data, g_xq, g_xsc, Yb, b.rows, b.cols, 0);
        return;
    }
    if (T == 16 && a.q == QuantKind::Q6_K && b.q == QuantKind::Q4_K && a.qk_soa && b.qk_soa &&
        a.cols == b.cols && a.cols > 0 && (a.cols % 256) == 0 && a.rows >= 4096 && b.rows >= 4096 &&
        a.data && b.data && X && Ya && Yb && g_xq && g_xsc && a.cols * 16 <= g_xq_n &&
        ensure_xq(X, a.cols, 16)) {
        static int once = 0;
        if (!once) {
            once = 1;
            std::fprintf(stderr, "q6q4_t16=1 ild=1 1row=1\n");
        }
        rapidllm::cuda_gemv::launch_q6k_f32_t16_1row(a.data, X, Ya, a.rows, a.cols, 0);
        rapidllm::cuda_gemv::launch_q4k_q8_t16_1row_pipe(b.data, g_xq, g_xsc, Yb, b.rows, b.cols, 0);
        return;
    }
    if (T == 6 && a.q == QuantKind::Q6_K && b.q == QuantKind::Q4_K && a.qk_soa && b.qk_soa &&
        a.cols == b.cols && a.cols > 0 && (a.cols % 256) == 0 && a.rows >= 4096 && b.rows >= 4096 &&
        a.data && b.data && X && Ya && Yb && g_xq && g_xsc && a.cols * 6 <= g_xq_n &&
        ensure_xq(X, a.cols, 6)) {
        static int once = 0;
        if (!once) {
            once = 1;
            std::fprintf(stderr, "q6q4_t6=1 ild=1\n");
        }
        rapidllm::cuda_gemv::launch_q6k_f32_t6_ild(a.data, X, Ya, a.rows, a.cols, 0);
        rapidllm::cuda_gemv::launch_q4k_q8_t6(b.data, g_xq, g_xsc, Yb, b.rows, b.cols, 0);
        return;
    }
    // T=4 GDN wqkv(Q6)+wz(Q4), isolated TU. T=3 ild codegen unchanged.
    if (T == 4 && a.q == QuantKind::Q6_K && b.q == QuantKind::Q4_K && a.qk_soa && b.qk_soa &&
        a.cols == b.cols && a.cols > 0 && (a.cols % 256) == 0 && a.rows >= 256 && b.rows >= 256 &&
        a.data && b.data && X && Ya && Yb && g_xq && g_xsc && a.cols * 4 <= g_xq_n &&
        ensure_xq(X, a.cols, 4)) {
        static int once = 0;
        if (!once) {
            once = 1;
            std::fprintf(stderr, "q6q4_t4_1row=1 t12tile=1 pipe=1 min256=1\n");
        }
        rapidllm::cuda_gemv::launch_q6k_f32_t4_1row_t12tile(a.data, X, Ya, a.rows, a.cols, 0);
        rapidllm::cuda_gemv::launch_q4k_q8_t4_1row_pipe(b.data, g_xq, g_xsc, Yb, b.rows, b.cols, 0);
        return;
    }
    if (T == 4 && a.q == QuantKind::Q8_0 && b.q == QuantKind::Q8_0 && a.rows == b.rows &&
        a.cols == b.cols && a.rows >= 4096 && (a.cols % 32) == 0 && a.scale && b.scale && a.data &&
        b.data && X && Ya && Yb) {
        static int once = 0;
        if (!once) {
            once = 1;
            std::fprintf(stderr, "q8_f32_t4_dual=1 1row=1\n");
        }
        rapidllm::cuda_gemv::launch_q8_f32_t4_1row_dual(
            reinterpret_cast<const int8_t*>(a.data), reinterpret_cast<const __half*>(a.scale),
            reinterpret_cast<const int8_t*>(b.data), reinterpret_cast<const __half*>(b.scale), X, Ya, Yb,
            a.rows, a.cols);
        return;
    }
    launch_linear(a, X, Ya, T);
    launch_linear(b, X, Yb, T);
}

bool launch_cublas_f16(const GpuW& w, const float* X, float* Y, int T);
void ensure_dq_stream();
void launch_fp8_tc(const GpuW& w, const float* X, float* Y, int T);
void launch_gemm_fp8(const GpuW& w, const float* X, float* Y, int T, int add);
void launch_gemm_q8(const GpuW& w, const float* X, float* Y, int T, int add);
bool launch_gemm_q8_tc(const GpuW& w, const float* X, float* Y, int T, int add);
bool launch_cublas_q8(const GpuW& w, const float* X, float* Y, int T, int add);
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

void launch_gemv_t2(const GpuW& w, const float* X, float* Y, int add) {
    if (!w.data || w.rows <= 0 || w.cols <= 0 || !X || !Y) return;
    int blocks = 0, threads = 0;
    gemv_grid(w, blocks, threads);
    const int warps = threads / 32;
    const int pairs = (w.rows + 1) / 2;
    const int pblocks = (pairs + warps - 1) / warps;
    if (w.q == QuantKind::FP8_E4M3_B128 && w.fp8_rowmaj) {
        static int once = 0;
        if (!once) {
            once = 1;
            std::fprintf(stderr, "t2_x1_ldg=1 tile=%d\n", w.cols <= kFp8XsCap ? w.cols : kFp8XsCap);
        }
        const int tile = w.cols <= kFp8XsCap ? w.cols : kFp8XsCap;
        gemv_fp8_rm_t2_k<<<pblocks, threads, sizeof(float) * static_cast<size_t>(tile)>>>(
            w.data, X, Y, w.rows, w.cols, add);
        return;
    }
    if (w.q == QuantKind::BF16) {
        gemv_bf16_t2_k<<<pblocks, threads, sizeof(float) * kXsTile * 2>>>(
            reinterpret_cast<const uint16_t*>(w.data), X, Y, w.rows, w.cols, add);
        return;
    }
}

void launch_gemv_t3(const GpuW& w, const float* X, float* Y, int add) {
    if (!w.data || w.rows <= 0 || w.cols <= 0 || !X || !Y) return;
    if (!(w.q == QuantKind::FP8_E4M3_B128 && w.fp8_rowmaj)) return;
    int blocks = 0, threads = 0;
    gemv_grid(w, blocks, threads);
    const int warps = threads / 32;
    const int pairs = (w.rows + 1) / 2;
    const int pblocks = (pairs + warps - 1) / warps;
    static int once = 0;
    if (!once) {
        once = 1;
        std::fprintf(stderr, "t3_x12_ldg=1 tile=%d\n", w.cols <= kFp8XsCap ? w.cols : kFp8XsCap);
    }
    const int tile = w.cols <= kFp8XsCap ? w.cols : kFp8XsCap;
    gemv_fp8_rm_t3_k<<<pblocks, threads, sizeof(float) * static_cast<size_t>(tile)>>>(
        w.data, X, Y, w.rows, w.cols, add);
}

void launch_gemv_t4(const GpuW& w, const float* X, float* Y, int add) {
    if (!w.data || w.rows <= 0 || w.cols <= 0 || !X || !Y) return;
    if (!(w.q == QuantKind::FP8_E4M3_B128 && w.fp8_rowmaj)) return;
    int blocks = 0, threads = 0;
    gemv_grid(w, blocks, threads);
    const int warps = threads / 32;
    const int pairs = (w.rows + 1) / 2;
    const int pblocks = (pairs + warps - 1) / warps;
    static int once = 0;
    if (!once) {
        once = 1;
        std::fprintf(stderr, "t4_x123_ldg=1 tile=%d\n", w.cols <= kFp8XsCap ? w.cols : kFp8XsCap);
    }
    const int tile = w.cols <= kFp8XsCap ? w.cols : kFp8XsCap;
    gemv_fp8_rm_t4_k<<<pblocks, threads, sizeof(float) * static_cast<size_t>(tile)>>>(
        w.data, X, Y, w.rows, w.cols, add);
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
        // Large GEMV: 2 rows/warp reuses the smem x tile (same as Q8/BF16).
        if (w.q == QuantKind::Q4_K) {
            if (w.rows >= 4096) {
                const int pairs = (w.rows + 1) / 2;
                const int warps = threads / 32;
                const int pblocks = (pairs + warps - 1) / warps;
                if (w.qk_soa && g_xq && g_xsc && (w.cols % 256) == 0 && w.cols <= g_xq_n) {
                    static int once = 0;
                    if (!once) {
                        once = 1;
                        std::fprintf(stderr, "q4k_q8_act=1 int_dot=1\n");
                    }
                    if (!ensure_xq(x, w.cols, 1)) {
                        gemv_qk_2row_k<kQ4KSoaBsz><<<pblocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
                    } else {
                        gemv_q4k_soa_q8_2row_k<<<pblocks, threads>>>(w.data, g_xq, g_xsc, y, w.rows, w.cols, add);
                    }
                } else if (w.qk_soa)
                    gemv_qk_2row_k<kQ4KSoaBsz><<<pblocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
                else
                    gemv_qk_2row_k<kQ4KBsz><<<pblocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
            } else if (w.qk_soa)
                gemv_qk_k<kQ4KSoaBsz><<<blocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
            else
                gemv_qk_k<kQ4KBsz><<<blocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
        } else if (w.q == QuantKind::Q5_K) {
            if (w.rows >= 4096) {
                const int pairs = (w.rows + 1) / 2;
                const int warps = threads / 32;
                const int pblocks = (pairs + warps - 1) / warps;
                if (w.qk_soa) gemv_qk_2row_k<kQ5KSoaBsz><<<pblocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
                else gemv_qk_2row_k<kQ5KBsz><<<pblocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
            } else if (w.qk_soa)
                gemv_qk_k<kQ5KSoaBsz><<<blocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
            else
                gemv_qk_k<kQ5KBsz><<<blocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
        } else {
            if (w.rows >= 4096 && w.qk_soa && (w.cols % 256) == 0) {
                static int once = 0;
                if (!once) {
                    once = 1;
                    std::fprintf(stderr, "q6k_f32_t1_2row=1 cp_async=1\n");
                }
                rapidllm::cuda_gemv::launch_q6k_f32_t1_2row(w.data, x, y, w.rows, w.cols, add);
            } else if (w.rows >= 4096) {
                const int pairs = (w.rows + 1) / 2;
                const int warps = threads / 32;
                const int pblocks = (pairs + warps - 1) / warps;
                if (w.qk_soa)
                    gemv_qk_2row_k<kQ6KSoaBsz><<<pblocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
                else
                    gemv_qk_2row_k<kQ6KBsz><<<pblocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
            } else if (w.qk_soa)
                gemv_qk_k<kQ6KSoaBsz><<<blocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
            else
                gemv_qk_k<kQ6KBsz><<<blocks, threads, smem>>>(w.data, x, y, w.rows, w.cols, add);
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
    ensure_dq_stream();
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
    if (!g_cublas_fp8_e4 || !lt || T < g_lt_min_T || w.rows < 128) return false;
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
    if (w.q == QuantKind::BF16) {
        f32_to_bf16_k<<<(n + 255) / 256, 256>>>(X, reinterpret_cast<uint16_t*>(g_xf16), n);
        return cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, w.rows, T, w.cols, &alpha, w.data, CUDA_R_16BF,
                            w.cols, g_xf16, CUDA_R_16BF, w.cols, &beta, Y, CUDA_R_32F, w.rows, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS;
    }
    return false;
}

void ensure_dq_stream() {
    if (g_dq_stream) return;
    cudaStreamCreateWithFlags(&g_dq_stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&g_dq_done[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&g_dq_done[1], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&g_gemm_done[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&g_gemm_done[1], cudaEventDisableTiming);
    cudaEventRecord(g_gemm_done[0], cudaStreamPerThread);
    cudaEventRecord(g_gemm_done[1], cudaStreamPerThread);
}

// T>=8: dequant one Q8 tile into g_wf16, then F16 tensor-core cublasGemmEx.
// lm_head (vocab×H) is row-tiled so it fits the existing ping-pong workspace.
bool launch_cublas_q8(const GpuW& w, const float* X, float* Y, int T, int add) {
    if (!g_blas || !g_xf16 || !g_wf16 || T < 16) return false;
    if (w.q != QuantKind::Q8_0 || !w.data || !w.scale) return false;
    if (w.rows < 16 || w.cols < 32 || (w.cols & 31) != 0) return false;
    if (w.cols * T > g_xf16_n) return false;
    const size_t cap = g_wf16_n;
    if (cap < static_cast<size_t>(w.cols) * 16) return false;
    int mtile = static_cast<int>(cap / static_cast<size_t>(w.cols));
    if (mtile > w.rows) mtile = w.rows;
    mtile &= ~7;
    if (mtile < 16) return false;
    cudaStream_t st = gemm_stream();
    f32_to_f16_k<<<(w.cols * T + 255) / 256, 256, 0, st>>>(X, g_xf16, w.cols * T);
    const int8_t* Q = reinterpret_cast<const int8_t*>(w.data);
    const __half* S = reinterpret_cast<const __half*>(w.scale);
    const float alpha = 1.f, beta0 = 0.f;
    cublasSetStream(g_blas, st);
    float* ydst = Y;
    if (add) {
        // Dedicated activation scratch (d_y_seq_). Reusing g_wf16's second half
        // overlapped a prior full-buffer dequant and zeroed 27B mixed tokens.
        const size_t yn = static_cast<size_t>(w.rows) * static_cast<size_t>(T);
        if (!g_yadd || yn > static_cast<size_t>(g_yadd_n) || Y == g_yadd) return false;
        ydst = g_yadd;
    }
    __half* wbuf = g_wf16;
    for (int r0 = 0; r0 < w.rows; r0 += mtile) {
        int mr = w.rows - r0;
        if (mr > mtile) mr = mtile;
        const size_t n4 = (static_cast<size_t>(mr) * static_cast<size_t>(w.cols)) >> 2;
        dequant_q8_soa_f16_k<<<static_cast<unsigned>((n4 + 255) / 256), 256, 0, st>>>(Q, S, wbuf, w.cols,
                                                                                     r0, mr);
        const bool ok =
            cublasGemmEx(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, mr, T, w.cols, &alpha, wbuf, CUDA_R_16F,
                         w.cols, g_xf16, CUDA_R_16F, w.cols, &beta0, ydst + r0, CUDA_R_32F, w.rows,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS;
        if (!ok) return false;
    }
    if (add) {
        const int yn = w.rows * T;
        add_res_k<<<(yn + 255) / 256, 256, 0, st>>>(Y, ydst, yn);
    }
    if (w.rows > 256) {
        static int once = 0;
        if (!once) {
            once = 1;
            std::fprintf(stderr, "native_q8_cublas=1 m=%d n=%d T=%d add=%d tiles=%d\n", w.rows, w.cols, T,
                         add, (w.rows + mtile - 1) / mtile);
        }
    }
    return true;
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
                                         pf_bytes, 0, 0, 0, 0, 0);
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
    // K-major T=3 dual. Official 27B W is row-major; this kernel reads fp8_blk4
    // and produced ~0 SwiGLU, so T=3 prefill skipped every MLP (l0 == post-GDN).
    if (T == 3 && w1.fp8_kmajor && w2.fp8_kmajor && w1.q == QuantKind::FP8_E4M3_B128 &&
        w2.q == QuantKind::FP8_E4M3_B128 && w1.data && w2.data && w1.rows == w2.rows &&
        w1.cols == w2.cols && w1.rows > 0 && w1.cols > 0) {
        int blocks = 0, threads = 0;
        gemv_grid(w1, blocks, threads);
        const int tile = gemm_fp8_tile(3, w1.cols);
        const size_t smem = static_cast<size_t>(3) * tile * sizeof(float);
        gemm_fp8_t3_dual_k<<<blocks, threads, smem>>>(w1.data, w1.scale, w2.data, w2.scale, X, Y1, Y2, w1.rows,
                                                      w1.cols, tile, fuse_swiglu);
        return;
    }
    if (T == 2 && w1.q == QuantKind::Q4_K && w2.q == QuantKind::Q4_K && w1.qk_soa && w2.qk_soa &&
        w1.rows == w2.rows && w1.cols == w2.cols && w1.rows >= 4096 && (w1.cols % 256) == 0 && g_xq &&
        g_xsc && w1.cols * 2 <= g_xq_n && ensure_xq(X, w1.cols, 2)) {
        static int once = 0;
        if (!once) {
            once = 1;
            std::fprintf(stderr, "q4k_q8_t2_dual=1 1row=1 fuse=%d\n", fuse_swiglu);
        }
        rapidllm::cuda_gemv::launch_q4k_q8_t2_dual_1row(w1.data, w2.data, g_xq, g_xsc, Y1, Y2, w1.rows,
                                                        w1.cols);
        if (fuse_swiglu && Y1 && Y2)
            swiglu_n_k<<<(((w1.rows * T + 3) / 4) + 255) / 256, 256>>>(Y1, Y2, Y1, w1.rows * T);
        return;
    }
    if (T == 3 && w1.q == QuantKind::Q4_K && w2.q == QuantKind::Q4_K && w1.qk_soa && w2.qk_soa &&
        w1.rows == w2.rows && w1.cols == w2.cols && w1.rows >= 4096 && (w1.cols % 256) == 0 && g_xq &&
        g_xsc && w1.cols * 3 <= g_xq_n && ensure_xq(X, w1.cols, 3)) {
        static int once = 0;
        if (!once) {
            once = 1;
            std::fprintf(stderr, "q4k_q8_t3_dual=1 fuse=%d blk=1\n", fuse_swiglu);
        }
        const int pairs = (w1.rows + 1) / 2;
        const int t3th = 256;
        const int t3w = t3th / 32;
        const int t3pb = (pairs + t3w - 1) / t3w;
        gemv_q4k_soa_q8_t3_dual_k<<<t3pb, t3th>>>(w1.data, w2.data, g_xq, g_xsc, Y1, Y2, w1.rows, w1.cols, 0);
        if (fuse_swiglu && Y1 && Y2)
            swiglu_n_k<<<(((w1.rows * T + 3) / 4) + 255) / 256, 256>>>(Y1, Y2, Y1, w1.rows * T);
        return;
    }
    if (T == 4 && w1.q == QuantKind::Q4_K && w2.q == QuantKind::Q4_K && w1.qk_soa && w2.qk_soa &&
        w1.rows == w2.rows && w1.cols == w2.cols && w1.rows >= 256 && (w1.cols % 256) == 0 && g_xq &&
        g_xsc && w1.cols * 4 <= g_xq_n && ensure_xq(X, w1.cols, 4)) {
        static int once = 0;
        if (!once) {
            once = 1;
            std::fprintf(stderr, "q4k_q8_t4_dual=1 1row=1 pipe=1 fuse=%d min256=1\n", fuse_swiglu);
        }
        rapidllm::cuda_gemv::launch_q4k_q8_t4_1row_dual_pipe(w1.data, w2.data, g_xq, g_xsc, Y1, Y2, w1.rows,
                                                             w1.cols);
        if (fuse_swiglu && Y1 && Y2)
            swiglu_n_k<<<(((w1.rows * T + 3) / 4) + 255) / 256, 256>>>(Y1, Y2, Y1, w1.rows * T);
        return;
    }
    if (T == 6 && w1.q == QuantKind::Q4_K && w2.q == QuantKind::Q4_K && w1.qk_soa && w2.qk_soa &&
        w1.rows == w2.rows && w1.cols == w2.cols && w1.rows >= 4096 && (w1.cols % 256) == 0 && g_xq &&
        g_xsc && w1.cols * 6 <= g_xq_n && ensure_xq(X, w1.cols, 6)) {
        static int once = 0;
        if (!once) {
            once = 1;
            std::fprintf(stderr, "q4k_q8_t6_dual=1 fuse=%d\n", fuse_swiglu);
        }
        rapidllm::cuda_gemv::launch_q4k_q8_t6_dual(w1.data, w2.data, g_xq, g_xsc, Y1, Y2, w1.rows, w1.cols);
        if (fuse_swiglu && Y1 && Y2)
            swiglu_n_k<<<(((w1.rows * T + 3) / 4) + 255) / 256, 256>>>(Y1, Y2, Y1, w1.rows * T);
        return;
    }
    if (T == 12 && w1.q == QuantKind::Q4_K && w2.q == QuantKind::Q4_K && w1.qk_soa && w2.qk_soa &&
        w1.rows == w2.rows && w1.cols == w2.cols && w1.rows >= 4096 && (w1.cols % 256) == 0 && g_xq &&
        g_xsc && w1.cols * 12 <= g_xq_n && ensure_xq(X, w1.cols, 12)) {
        static int once = 0;
        if (!once) {
            once = 1;
            std::fprintf(stderr, "q4k_q8_t12_pair=1 fuse=%d pipe=1\n", fuse_swiglu);
        }
        rapidllm::cuda_gemv::launch_q4k_q8_t12_1row_dual_pipe(w1.data, w2.data, g_xq, g_xsc, Y1, Y2, w1.rows,
                                                             w1.cols);
        if (fuse_swiglu && Y1 && Y2)
            swiglu_n_k<<<(((w1.rows * T + 3) / 4) + 255) / 256, 256>>>(Y1, Y2, Y1, w1.rows * T);
        return;
    }
    if (T == 16 && w1.q == QuantKind::Q4_K && w2.q == QuantKind::Q4_K && w1.qk_soa && w2.qk_soa &&
        w1.rows == w2.rows && w1.cols == w2.cols && w1.rows >= 4096 && (w1.cols % 256) == 0 && g_xq &&
        g_xsc && w1.cols * 16 <= g_xq_n && ensure_xq(X, w1.cols, 16)) {
        static int once = 0;
        if (!once) {
            once = 1;
            std::fprintf(stderr, "q4k_q8_t16_pair=1 fuse=%d pipe=1\n", fuse_swiglu);
        }
        rapidllm::cuda_gemv::launch_q4k_q8_t16_1row_dual_pipe(w1.data, w2.data, g_xq, g_xsc, Y1, Y2, w1.rows,
                                                             w1.cols);
        if (fuse_swiglu && Y1 && Y2)
            swiglu_n_k<<<(((w1.rows * T + 3) / 4) + 255) / 256, 256>>>(Y1, Y2, Y1, w1.rows * T);
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

static void pack_q6k_gm(uint8_t* dst, const uint8_t* soa, int rows, int cols) {
    const int nb = cols / 256;
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = soa + (static_cast<size_t>(r) * nb + b) * kQ6KSoaBsz;
            uint8_t* out = dst + (static_cast<size_t>(r) * nb + b) * kQ6KGmBsz;
            const __half* ds = reinterpret_cast<const __half*>(blk);
            const uint8_t* ql = blk + 32;
            const uint8_t* qh = blk + 160;
            int8_t* q = reinterpret_cast<int8_t*>(out);
            for (int n128 = 0; n128 < 2; ++n128) {
                for (int lane = 0; lane < 32; ++lane) {
                    const uint8_t qlo = ql[lane];
                    const uint8_t qhi = qh[lane];
                    const uint8_t qlo2 = ql[32 + lane];
                    q[n128 * 128 + lane] =
                        static_cast<int8_t>(((qlo & 0xF) | ((qhi & 3) << 4)) - 32);
                    q[n128 * 128 + 32 + lane] =
                        static_cast<int8_t>(((qlo2 & 0xF) | (((qhi >> 2) & 3) << 4)) - 32);
                    q[n128 * 128 + 64 + lane] =
                        static_cast<int8_t>(((qlo >> 4) | (((qhi >> 4) & 3) << 4)) - 32);
                    q[n128 * 128 + 96 + lane] =
                        static_cast<int8_t>(((qlo2 >> 4) | (((qhi >> 6) & 3) << 4)) - 32);
                }
                ql += 64;
                qh += 32;
            }
            std::memcpy(out + 256, ds, 32);
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
    // Integer Q4×Q8-act vs float SOA GEMV (same W, quantized x).
    {
        const int m = 64, n = 512;
        std::vector<float> x(n), y_ref(m), y_q8(m);
        for (int i = 0; i < n; ++i) x[static_cast<size_t>(i)] = 0.02f * static_cast<float>((i % 17) - 8);
        const size_t bytes = static_cast<size_t>(m) * (n / 256) * static_cast<size_t>(kQ4KBsz);
        std::vector<uint8_t> W(bytes);
        for (size_t i = 0; i < bytes; ++i) W[i] = static_cast<uint8_t>((i * 17 + 11) & 0xff);
        const int nb = n / 256;
        for (int r = 0; r < m; ++r) {
            for (int b = 0; b < nb; ++b) {
                uint8_t* blk = W.data() + (static_cast<size_t>(r) * nb + b) * kQ4KBsz;
                const __half dh = __float2half(0.05f);
                const __half dm = __float2half(0.01f);
                std::memcpy(blk, &dh, 2);
                std::memcpy(blk + 2, &dm, 2);
            }
        }
        std::vector<uint8_t> Soa(static_cast<size_t>(m) * nb * kQ4KSoaBsz);
        pack_q4k_soa(Soa.data(), W.data(), m, n);
        uint8_t* dW = nullptr;
        float *dX = nullptr, *dY = nullptr;
        int8_t* dXq = nullptr;
        __half* dXsc = nullptr;
        CUDA_CHECK(cudaMalloc(&dW, Soa.size()));
        CUDA_CHECK(cudaMalloc(&dX, sizeof(float) * n));
        CUDA_CHECK(cudaMalloc(&dY, sizeof(float) * m));
        CUDA_CHECK(cudaMalloc(&dXq, n));
        CUDA_CHECK(cudaMalloc(&dXsc, sizeof(__half) * (n / 32)));
        CUDA_CHECK(cudaMemcpy(dW, Soa.data(), Soa.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dX, x.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
        GpuW tw;
        tw.data = dW;
        tw.q = QuantKind::Q4_K;
        tw.rows = m;
        tw.cols = n;
        tw.qk_soa = true;
        launch_gemv(tw, dX, dY, 0);
        CUDA_CHECK(cudaMemcpy(y_ref.data(), dY, sizeof(float) * m, cudaMemcpyDeviceToHost));
        quant_x_q8_k<<<((n >> 5) + 127) / 128, 128>>>(dX, dXq, dXsc, n);
        CUDA_CHECK(cudaMemset(dY, 0, sizeof(float) * m));
        gemv_q4k_soa_q8_2row_k<<<(m + 15) / 16, 256>>>(dW, dXq, dXsc, dY, m, n, 0);
        CUDA_CHECK(cudaMemcpy(y_q8.data(), dY, sizeof(float) * m, cudaMemcpyDeviceToHost));
        float maxe = 0.f, maxa = 0.f;
        for (int i = 0; i < m; ++i) {
            maxe = std::max(maxe, std::fabs(y_q8[i] - y_ref[i]));
            maxa = std::max(maxa, std::fabs(y_ref[i]));
        }
        cudaFree(dW);
        cudaFree(dX);
        cudaFree(dY);
        cudaFree(dXq);
        cudaFree(dXsc);
        const bool ok = maxe < std::max(0.05f * maxa, 2e-2f) && cudaGetLastError() == cudaSuccess;
        std::fprintf(stderr, "native_q4k_q8act_err=%.5f maxa=%.4f ok=%d\n", maxe, maxa, ok ? 1 : 0);
        if (!ok) throw std::runtime_error("native Q4×Q8-act GEMV selftest failed");
    }
}

void q8_gemm_t_selftest() {
    const int m = 64, n = 256, T = 4;
    std::vector<int8_t> Q(static_cast<size_t>(m) * n);
    std::vector<uint16_t> Sc(static_cast<size_t>(m) * (n / 32));
    std::vector<float> X(static_cast<size_t>(T) * n), Yref(static_cast<size_t>(T) * m), Y(static_cast<size_t>(T) * m);
    for (size_t i = 0; i < Q.size(); ++i) Q[i] = static_cast<int8_t>((static_cast<int>(i) * 13 + 7) % 61 - 30);
    for (size_t i = 0; i < Sc.size(); ++i) {
        const __half h = __float2half(0.02f + 0.001f * static_cast<float>(i % 7));
        std::memcpy(Sc.data() + i, &h, 2);
    }
    for (size_t i = 0; i < X.size(); ++i) X[i] = 0.03f * static_cast<float>((static_cast<int>(i) % 11) - 5);
    int8_t* dQ = nullptr;
    __half* dS = nullptr;
    float *dX = nullptr, *dY = nullptr, *dY1 = nullptr;
    CUDA_CHECK(cudaMalloc(&dQ, Q.size()));
    CUDA_CHECK(cudaMalloc(&dS, Sc.size() * 2));
    CUDA_CHECK(cudaMalloc(&dX, X.size() * 4));
    CUDA_CHECK(cudaMalloc(&dY, Y.size() * 4));
    CUDA_CHECK(cudaMalloc(&dY1, sizeof(float) * m));
    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), Q.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dS, Sc.data(), Sc.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dX, X.data(), X.size() * 4, cudaMemcpyHostToDevice));
    GpuW tw;
    tw.data = reinterpret_cast<const uint8_t*>(dQ);
    tw.scale = reinterpret_cast<const float*>(dS);
    tw.q = QuantKind::Q8_0;
    tw.rows = m;
    tw.cols = n;
    for (int t = 0; t < T; ++t) {
        CUDA_CHECK(cudaMemset(dY1, 0, sizeof(float) * m));
        launch_gemv(tw, dX + static_cast<size_t>(t) * n, dY1, 0);
        CUDA_CHECK(cudaMemcpy(Yref.data() + static_cast<size_t>(t) * m, dY1, sizeof(float) * m,
                              cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemset(dY, 0, Y.size() * 4));
    launch_gemm_q8(tw, dX, dY, T, 0);
    CUDA_CHECK(cudaMemcpy(Y.data(), dY, Y.size() * 4, cudaMemcpyDeviceToHost));
    float maxe = 0.f, maxa = 0.f;
    for (size_t i = 0; i < Y.size(); ++i) {
        maxe = std::max(maxe, std::fabs(Y[i] - Yref[i]));
        maxa = std::max(maxa, std::fabs(Yref[i]));
    }
    cudaFree(dQ);
    cudaFree(dS);
    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dY1);
    const bool ok = maxe < std::max(0.02f * maxa, 1e-3f) && cudaGetLastError() == cudaSuccess;
    std::fprintf(stderr, "native_q8_gemm_t4_err=%.5f maxa=%.4f ok=%d\n", maxe, maxa, ok ? 1 : 0);
    if (!ok) throw std::runtime_error("native Q8 GEMM T=4 selftest failed");
}

bool q8_tc_selftest() {
    g_q8_cublas_add = false;
    const int m = 128, n = 256, T = 16;
    std::vector<int8_t> Q(static_cast<size_t>(m) * n);
    std::vector<uint16_t> Sc(static_cast<size_t>(m) * (n / 32));
    std::vector<float> X(static_cast<size_t>(T) * n), Yref(static_cast<size_t>(T) * m),
        Ytc(static_cast<size_t>(T) * m), Ycb(static_cast<size_t>(T) * m);
    for (size_t i = 0; i < Q.size(); ++i) Q[i] = static_cast<int8_t>((static_cast<int>(i) * 13 + 7) % 61 - 30);
    for (size_t i = 0; i < Sc.size(); ++i) {
        const __half h = __float2half(0.02f + 0.001f * static_cast<float>(i % 7));
        std::memcpy(Sc.data() + i, &h, 2);
    }
    for (size_t i = 0; i < X.size(); ++i) X[i] = 0.03f * static_cast<float>((static_cast<int>(i) % 11) - 5);
    int8_t* dQ = nullptr;
    __half* dS = nullptr;
    float *dX = nullptr, *dY = nullptr, *dY1 = nullptr;
    __half *twf = nullptr, *txf = nullptr;
    if (cudaMalloc(&dQ, Q.size()) != cudaSuccess) return false;
    if (cudaMalloc(&dS, Sc.size() * 2) != cudaSuccess) {
        cudaFree(dQ);
        return false;
    }
    if (cudaMalloc(&dX, X.size() * 4) != cudaSuccess) {
        cudaFree(dQ);
        cudaFree(dS);
        return false;
    }
    if (cudaMalloc(&dY, Yref.size() * 4) != cudaSuccess) {
        cudaFree(dQ);
        cudaFree(dS);
        cudaFree(dX);
        return false;
    }
    if (cudaMalloc(&dY1, sizeof(float) * m) != cudaSuccess) {
        cudaFree(dQ);
        cudaFree(dS);
        cudaFree(dX);
        cudaFree(dY);
        return false;
    }
    cudaMemcpy(dQ, Q.data(), Q.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dS, Sc.data(), Sc.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X.data(), X.size() * 4, cudaMemcpyHostToDevice);
    GpuW tw;
    tw.data = reinterpret_cast<const uint8_t*>(dQ);
    tw.scale = reinterpret_cast<const float*>(dS);
    tw.q = QuantKind::Q8_0;
    tw.rows = m;
    tw.cols = n;
    for (int t = 0; t < T; ++t) {
        cudaMemset(dY1, 0, sizeof(float) * m);
        launch_gemv(tw, dX + static_cast<size_t>(t) * n, dY1, 0);
        cudaMemcpy(Yref.data() + static_cast<size_t>(t) * m, dY1, sizeof(float) * m, cudaMemcpyDeviceToHost);
    }
    auto max_err = [&](const std::vector<float>& Y) {
        float maxe = 0.f, maxa = 0.f;
        for (size_t i = 0; i < Y.size(); ++i) {
            maxe = std::max(maxe, std::fabs(Y[i] - Yref[i]));
            maxa = std::max(maxa, std::fabs(Yref[i]));
        }
        return std::pair<float, float>(maxe, maxa);
    };
    cudaMemset(dY, 0, Ytc.size() * 4);
    const bool prev = g_q8_tc;
    g_q8_tc = true;
    const bool launched = launch_gemm_q8_tc(tw, dX, dY, T, 0);
    g_q8_tc = prev;
    bool tc_ok = false;
    if (launched && cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == cudaSuccess) {
        cudaMemcpy(Ytc.data(), dY, Ytc.size() * 4, cudaMemcpyDeviceToHost);
        const auto e = max_err(Ytc);
        tc_ok = e.first < std::max(0.05f * e.second, 2e-3f);
        std::fprintf(stderr, "native_q8_tc_t8_err=%.5f maxa=%.4f ok=%d\n", e.first, e.second, tc_ok ? 1 : 0);
    } else {
        cudaGetLastError();
        std::fprintf(stderr, "native_q8_tc_t8_err=nan maxa=0 ok=0\n");
    }
    bool cb_ok = false;
    if (cudaMalloc(&twf, 2ull * static_cast<size_t>(m) * n * sizeof(__half)) == cudaSuccess &&
        cudaMalloc(&txf, static_cast<size_t>(n) * T * sizeof(__half)) == cudaSuccess) {
        __half* old_wf = g_wf16;
        const size_t old_wn = g_wf16_n;
        __half* old_xf = g_xf16;
        const int old_xn = g_xf16_n;
        g_wf16 = twf;
        g_wf16_n = 2ull * static_cast<size_t>(m) * n;
        g_xf16 = txf;
        g_xf16_n = n * T;
        cudaMemset(dY, 0, Ycb.size() * 4);
        const bool cb = launch_cublas_q8(tw, dX, dY, T, 0);
        if (cb && cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == cudaSuccess) {
            cudaMemcpy(Ycb.data(), dY, Ycb.size() * 4, cudaMemcpyDeviceToHost);
            const auto e = max_err(Ycb);
            cb_ok = e.first < std::max(0.05f * e.second, 2e-3f);
            std::fprintf(stderr, "native_q8_cublas_t8_err=%.5f maxa=%.4f ok=%d\n", e.first, e.second,
                         cb_ok ? 1 : 0);
        } else {
            cudaGetLastError();
            std::fprintf(stderr, "native_q8_cublas_t8_err=nan maxa=0 ok=0\n");
        }
        bool add_ok = false;
        float* dyadd = nullptr;
        if (cb_ok && cudaMalloc(&dyadd, Ycb.size() * 4) == cudaSuccess) {
            float* old_ya = g_yadd;
            const int old_yn = g_yadd_n;
            g_yadd = dyadd;
            g_yadd_n = static_cast<int>(Ycb.size());
            std::vector<float> Ybase(Ycb.size(), 0.25f);
            cudaMemcpy(dY, Ybase.data(), Ybase.size() * 4, cudaMemcpyHostToDevice);
            const bool cba = launch_cublas_q8(tw, dX, dY, T, 1);
            if (cba && cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == cudaSuccess) {
                std::vector<float> Ygot(Ycb.size());
                cudaMemcpy(Ygot.data(), dY, Ygot.size() * 4, cudaMemcpyDeviceToHost);
                float maxe = 0.f, maxa = 0.f;
                for (size_t i = 0; i < Ygot.size(); ++i) {
                    const float ref = 0.25f + Yref[i];
                    maxe = std::max(maxe, std::fabs(Ygot[i] - ref));
                    maxa = std::max(maxa, std::fabs(ref));
                }
                add_ok = maxe < std::max(0.05f * maxa, 2e-3f);
                std::fprintf(stderr, "native_q8_cublas_add_err=%.5f maxa=%.4f ok=%d\n", maxe, maxa,
                             add_ok ? 1 : 0);
            } else {
                cudaGetLastError();
                std::fprintf(stderr, "native_q8_cublas_add_err=nan maxa=0 ok=0\n");
            }
            g_yadd = old_ya;
            g_yadd_n = old_yn;
            cudaFree(dyadd);
        } else {
            std::fprintf(stderr, "native_q8_cublas_add_err=nan maxa=0 ok=0\n");
        }
        g_q8_cublas_add = add_ok;
        g_wf16 = old_wf;
        g_wf16_n = old_wn;
        g_xf16 = old_xf;
        g_xf16_n = old_xn;
    }
    cudaFree(twf);
    cudaFree(txf);
    cudaFree(dQ);
    cudaFree(dS);
    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dY1);
    (void)cb_ok;
    return tc_ok;
}

void argmax_rows_selftest() {
    const int T = 48, n = 512, th = 256;
    const int nblk = (n + th - 1) / th;
    std::vector<float> hx(static_cast<size_t>(T) * n, 0.f);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < n; ++i)
            hx[static_cast<size_t>(t) * n + i] = 0.01f * static_cast<float>((i + t) % 17);
        hx[static_cast<size_t>(t) * n + (100 + t)] = 50.f + static_cast<float>(t);
    }
    float *dx = nullptr, *dmax = nullptr;
    int *didx = nullptr, *dout = nullptr;
    auto fail_alloc = [&]() {
        cudaFree(dx);
        cudaFree(dmax);
        cudaFree(didx);
        cudaFree(dout);
        throw std::runtime_error("argmax_rows_selftest alloc failed");
    };
    if (cudaMalloc(&dx, hx.size() * 4) != cudaSuccess) fail_alloc();
    if (cudaMalloc(&dmax, sizeof(float) * static_cast<size_t>(nblk) * T) != cudaSuccess) fail_alloc();
    if (cudaMalloc(&didx, sizeof(int) * static_cast<size_t>(nblk) * T) != cudaSuccess) fail_alloc();
    if (cudaMalloc(&dout, sizeof(int) * T) != cudaSuccess) fail_alloc();
    cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
    dim3 grid(nblk, T);
    argmax_partial_rows_k<<<grid, th>>>(dx, n, n, dmax, didx);
    const int nth = 128;
    argmax_final_rows_k<<<(T + nth - 1) / nth, nth>>>(dmax, didx, nblk, T, dout);
    std::vector<int> hout(static_cast<size_t>(T), -1);
    const cudaError_t st = cudaDeviceSynchronize();
    cudaMemcpy(hout.data(), dout, sizeof(int) * T, cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dmax);
    cudaFree(didx);
    cudaFree(dout);
    int bad = 0;
    for (int t = 0; t < T; ++t)
        if (hout[static_cast<size_t>(t)] != 100 + t) ++bad;
    std::fprintf(stderr, "argmax_rows_selftest T=%d bad=%d st=%d\n", T, bad, static_cast<int>(st));
    if (st != cudaSuccess || bad != 0) throw std::runtime_error("argmax_rows_selftest failed (B>32 greedy)");
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
                                          1e-6f, 16384, 1, nullptr, nullptr, nullptr, nullptr, 0, 0, 0);
    qk_attn_decode_k<<<n_q, 256, sm_on>>>(
        dQt, dK + static_cast<size_t>(pos) * kn, dV + static_cast<size_t>(pos) * kn, nullptr, nullptr,
        reinterpret_cast<float*>(dKf), reinterpret_cast<float*>(dVf), dOt, dPos, n_q, n_kv, hd, 0, 1e7f, 1e-6f,
        16384, 3, dKq, dKsc, dVq, dVsc, 0, 0, 0);
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
    {
        cudaStreamAttrValue pol;
        std::memset(&pol, 0, sizeof(pol));
        if (cudaStreamSetAttribute(cudaStreamPerThread, cudaStreamAttributeAccessPolicyWindow, &pol) !=
            cudaSuccess)
            cudaGetLastError();
        cudaCtxResetPersistingL2Cache();
        cudaDeviceSynchronize();
        cudaGetLastError();
    }
    const size_t op_n = static_cast<size_t>(T) * n_q * hd;
    CUDA_CHECK(cudaMemset(dOp, 0, op_n * 4));
    attn_prefill_tq_gqa_k<<<dim3(n_kv, T), 256>>>(dQf, dOp, 0, T, n_q, n_kv, hd, dKq, dKsc, dVq, dVsc);
    const cudaError_t pfe = cudaGetLastError();
    const cudaError_t pfs = cudaDeviceSynchronize();
    std::vector<float> hOp(static_cast<size_t>(T) * n_q * hd);
    CUDA_CHECK(cudaMemcpy(hOp.data(), dOp, hOp.size() * 4, cudaMemcpyDeviceToHost));
    bool pfin = true;
    int nnan = 0, ifirst = -1;
    for (size_t i = 0; i < hOp.size(); ++i) {
        if (!std::isfinite(hOp[i])) {
            pfin = false;
            ++nnan;
            if (ifirst < 0) ifirst = static_cast<int>(i);
        }
    }
    if (pfe != cudaSuccess || pfs != cudaSuccess || !pfin)
        std::fprintf(stderr, "tq_prefill_diag nnan=%d first=%d launch=%d sync=%d\n", nnan, ifirst,
                     static_cast<int>(pfe), static_cast<int>(pfs));
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
    const bool ok = finite && pfin && pfe == cudaSuccess && pfs == cudaSuccess &&
                    cos > 0.65f && kmaxe < std::max(0.05f * kmaxa, 1e-3f) &&
                    cudaGetLastError() == cudaSuccess;
    std::fprintf(stderr, "tq_attend_cos=%.4f tq_prefill_finite=%d k_q8_err=%.5f maxa=%.4f ok=%d\n", cos,
                 pfin ? 1 : 0, kmaxe, maxa, ok ? 1 : 0);
    if (!ok) throw std::runtime_error("tq store+attend selftest failed");
}

void flash_gqa6_selftest() {
    const int n_q = 24, n_kv = 4, hd = 256, ctx = 8, pos = 3;
    const int kn = n_kv * hd, qn = n_q * hd;
    std::vector<float> hQ(qn), hK(kn), hV(kn), hA(qn), hB(qn);
    for (int i = 0; i < qn; ++i) hQ[i] = 0.01f * static_cast<float>((i % 17) - 8);
    for (int i = 0; i < kn; ++i) {
        hK[i] = 0.02f * static_cast<float>((i % 13) - 6);
        hV[i] = 0.03f * static_cast<float>((i % 11) - 5);
    }
    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dOa = nullptr, *dOb = nullptr, *dKc = nullptr,
          *dVc = nullptr;
    int* dPos = nullptr;
    CUDA_CHECK(cudaMalloc(&dQ, qn * 4));
    CUDA_CHECK(cudaMalloc(&dK, kn * 4));
    CUDA_CHECK(cudaMalloc(&dV, kn * 4));
    CUDA_CHECK(cudaMalloc(&dOa, qn * 4));
    CUDA_CHECK(cudaMalloc(&dOb, qn * 4));
    CUDA_CHECK(cudaMalloc(&dKc, ctx * kn * 2));
    CUDA_CHECK(cudaMalloc(&dVc, ctx * kn * 2));
    CUDA_CHECK(cudaMalloc(&dPos, 4));
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), qn * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), kn * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), kn * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dOa, 0, qn * 4));
    CUDA_CHECK(cudaMemset(dOb, 0, qn * 4));
    CUDA_CHECK(cudaMemset(dKc, 0, ctx * kn * 2));
    CUDA_CHECK(cudaMemset(dVc, 0, ctx * kn * 2));
    CUDA_CHECK(cudaMemcpy(dPos, &pos, 4, cudaMemcpyHostToDevice));
    const size_t sm = sizeof(float) * hd;
    qk_attn_decode_k<<<n_q, 256, sm>>>(dQ, dK, dV, nullptr, nullptr, dKc, dVc, dOa, dPos, n_q, n_kv, hd, 0,
                                       1e7f, 1e-6f, ctx, 1, nullptr, nullptr, nullptr, nullptr, 0, 0, 0);
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), qn * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dKc, 0, ctx * kn * 2));
    CUDA_CHECK(cudaMemset(dVc, 0, ctx * kn * 2));
    qk_attn_decode_gqa_k<<<dim3(n_kv, n_q / n_kv), 256, sm>>>(dQ, dK, dV, nullptr, nullptr, dKc, dVc, dOb,
                                                              dPos, n_q, n_kv, hd, 0, 1e7f, 1e-6f, ctx);
    CUDA_CHECK(cudaMemcpy(hA.data(), dOa, qn * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hB.data(), dOb, qn * 4, cudaMemcpyDeviceToHost));
    float dot = 0, na = 0, nb = 0, maxe = 0;
    bool finite = true;
    for (int i = 0; i < qn; ++i) {
        finite = finite && std::isfinite(hA[i]) && std::isfinite(hB[i]);
        dot += hA[i] * hB[i];
        na += hA[i] * hA[i];
        nb += hB[i] * hB[i];
        maxe = std::max(maxe, std::fabs(hA[i] - hB[i]));
    }
    const float cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.f;
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dOa);
    cudaFree(dOb);
    cudaFree(dKc);
    cudaFree(dVc);
    cudaFree(dPos);
    const bool ok = finite && cos > 0.99f && maxe < 1e-2f && cudaGetLastError() == cudaSuccess;
    std::fprintf(stderr, "flash_gqa6_cos=%.4f maxe=%.5f ok=%d\n", cos, maxe, ok ? 1 : 0);
    if (!ok) throw std::runtime_error("flash_gqa6 selftest failed");
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

void launch_gemm_q8(const GpuW& w, const float* X, float* Y, int T, int add) {
    int blocks = 0, threads = 0;
    gemv_grid(w, blocks, threads);
    int tile = 2048;
    while (T > 0 && static_cast<size_t>(T) * (tile / 32) * kXsPad * sizeof(float) > 24 * 1024 && tile > 128)
        tile /= 2;
    const size_t smem = static_cast<size_t>(T) * (tile / 32) * kXsPad * sizeof(float);
    const int8_t* Q = reinterpret_cast<const int8_t*>(w.data);
    const __half* S = reinterpret_cast<const __half*>(w.scale);
    if (T <= 4)
        gemm_q8_soa_k<<<blocks, threads, smem>>>(Q, S, X, Y, w.rows, w.cols, T, tile, add);
    else
        gemm_q8_soa_wide_k<<<blocks, threads, smem>>>(Q, S, X, Y, w.rows, w.cols, T, tile, add);
}

bool launch_gemm_q8_tc(const GpuW& w, const float* X, float* Y, int T, int add) {
    if (!g_q8_tc || T < 8 || w.rows < 16) return false;
    if (w.q != QuantKind::Q8_0 || !w.data || !w.scale) return false;
    if (w.cols < 128 || (w.cols & 127) != 0) return false;
    constexpr int BM = 128, WLD = 136, XLD = 136;
    const int BN = T >= 64 ? 64 : (T >= 32 ? 32 : (T >= 16 ? 16 : 8));
    dim3 grid((w.rows + BM - 1) / BM, (T + BN - 1) / BN);
    const size_t smem = (static_cast<size_t>(BM) * WLD + static_cast<size_t>(BN) * XLD) * sizeof(__half);
    const int8_t* Q = reinterpret_cast<const int8_t*>(w.data);
    const __half* S = reinterpret_cast<const __half*>(w.scale);
    cudaStream_t st = gemm_stream();
    if (BN == 64)
        gemm_q8_soa_tc_k<64><<<grid, 256, smem, st>>>(Q, S, X, Y, w.rows, w.cols, T, add);
    else if (BN == 32)
        gemm_q8_soa_tc_k<32><<<grid, 256, smem, st>>>(Q, S, X, Y, w.rows, w.cols, T, add);
    else if (BN == 16)
        gemm_q8_soa_tc_k<16><<<grid, 256, smem, st>>>(Q, S, X, Y, w.rows, w.cols, T, add);
    else
        gemm_q8_soa_tc_k<8><<<grid, 256, smem, st>>>(Q, S, X, Y, w.rows, w.cols, T, add);
    return true;
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
        if (!add && T >= 16 && (w.q == QuantKind::F16 || w.q == QuantKind::BF16) &&
            launch_cublas_f16(w, X, Y, T))
            return;
        if (T == 1) {
            launch_gemv(w, X, Y, add);
            return;
        }
        if (T == 2 && w.q == QuantKind::BF16) {
            launch_gemv_t2(w, X, Y, add);
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
            // T=2 spec verify / T=3 short prefill: custom one-W-read GEMV.
            // Lt n=2/3 is slower than T=1 GEMV on this card.
            if (T == 2) {
                launch_gemv_t2(w, X, Y, add);
                return;
            }
            if (T == 3) {
                launch_gemv_t3(w, X, Y, add);
                return;
            }
            if (T == 4) {
                // GEMV T=4 (~163 ms) and Lt n=4 both drifted official
                // last after 31 from 46474 to 0. Do not run T=4 on
                // official; T=2 + T=12 stay on GEMV / Lt n=12.
                for (int t = 0; t < T; ++t)
                    launch_gemv(w, X + static_cast<size_t>(t) * w.cols,
                                Y + static_cast<size_t>(t) * w.rows, add);
                return;
            }
            // T=2/3/4 spec/prefill graphs stay on GEMV. Batch decode is T=B>=8:
            // one cublasLt pass so 8 seqs do not reread 28GiB eight times.
            if (T >= g_lt_min_T && launch_cublas_fp8_e4(w, X, Y, T, add)) return;
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
    if (w.q == QuantKind::Q8_0 && T > 1) {
        if (T == 4 && w.rows >= 4096 && w.cols >= 32 && (w.cols % 32) == 0 && w.scale) {
            static int once = 0;
            if (!once) {
                once = 1;
                std::fprintf(stderr, "q8_f32_t4=1 1row=1 soa=1 lb2=1\n");
            }
            rapidllm::cuda_gemv::launch_q8_f32_t4_1row(reinterpret_cast<const int8_t*>(w.data),
                                                       reinterpret_cast<const __half*>(w.scale), X, Y, w.rows,
                                                       w.cols, add);
            return;
        }
        if (T == 12 && w.rows >= 4096 && w.cols >= 32 && (w.cols % 32) == 0 && w.scale) {
            static int once = 0;
            if (!once) {
                once = 1;
                std::fprintf(stderr, "q8_f32_t12=1 soa=1 tile=960\n");
            }
            rapidllm::cuda_gemv::launch_q8_f32_t12(reinterpret_cast<const int8_t*>(w.data),
                                                   reinterpret_cast<const __half*>(w.scale), X, Y, w.rows,
                                                   w.cols, add);
            return;
        }
        if (T >= 8 && launch_gemm_q8_tc(w, X, Y, T, add)) {
            static int once = 0;
            if (!once) {
                once = 1;
                std::fprintf(stderr, "q8_t%d_tc=1\n", T);
            }
            return;
        }
        if (T >= 16 && !add && launch_cublas_q8(w, X, Y, T, 0)) return;
        if (T >= 2 && T <= 16) {
            static int once = 0;
            if (!once) {
                once = 1;
                std::fprintf(stderr, "q8_t%d_gemm=1\n", T);
            }
            launch_gemm_q8(w, X, Y, T, add);
            return;
        }
        for (int t = 0; t < T; ++t)
            launch_gemv(w, X + static_cast<size_t>(t) * w.cols, Y + static_cast<size_t>(t) * w.rows, add);
        return;
    }
    if ((w.q == QuantKind::Q4_K || w.q == QuantKind::Q5_K || w.q == QuantKind::Q6_K) && T > 1) {
        if (T == 2 && w.q == QuantKind::Q6_K && w.qk_soa && w.rows >= 4096 && (w.cols % 256) == 0) {
            static int once = 0;
            if (!once) {
                once = 1;
                std::fprintf(stderr, "q6k_f32_t2=1 thr=256 ild=1 pipe=1 1row=1\n");
            }
            rapidllm::cuda_gemv::launch_q6k_f32_t2_1row(w.data, X, Y, w.rows, w.cols, add);
            return;
        }
        if (T == 2 && w.q == QuantKind::Q4_K && w.qk_soa && w.rows >= 4096 && (w.cols % 256) == 0 &&
            g_xq && g_xsc && w.cols * 2 <= g_xq_n && ensure_xq(X, w.cols, 2)) {
            static int once = 0;
            if (!once) {
                once = 1;
                std::fprintf(stderr, "q4k_q8_t2=1 int_dot=1 thr=256 1row=1\n");
            }
            rapidllm::cuda_gemv::launch_q4k_q8_t2_1row(w.data, g_xq, g_xsc, Y, w.rows, w.cols, add);
            return;
        }
        if (T == 2 && w.rows >= 4096 && (w.cols % 256) == 0) {
            int blocks = 0, threads = 0;
            gemv_grid(w, blocks, threads);
            const int tile = w.cols < kXsTile ? w.cols : kXsTile;
            const size_t smem = sizeof(float) * static_cast<size_t>(tile);
            const int pairs = (w.rows + 1) / 2;
            const int warps = threads / 32;
            const int pblocks = (pairs + warps - 1) / warps;
            static int t2once = 0;
            if (!t2once) {
                t2once = 1;
                std::fprintf(stderr, "qk_t2_x1_ldg=1 tile=%d\n", tile);
            }
            if (w.q == QuantKind::Q4_K) {
                if (w.qk_soa)
                    gemv_qk_t2_2row_k<kQ4KSoaBsz><<<pblocks, threads, smem>>>(w.data, X, Y, w.rows, w.cols, add);
                else
                    gemv_qk_t2_2row_k<kQ4KBsz><<<pblocks, threads, smem>>>(w.data, X, Y, w.rows, w.cols, add);
            } else if (w.q == QuantKind::Q6_K) {
                if (w.qk_soa)
                    gemv_qk_t2_2row_k<kQ6KSoaBsz><<<pblocks, threads, smem>>>(w.data, X, Y, w.rows, w.cols, add);
                else
                    gemv_qk_t2_2row_k<kQ6KBsz><<<pblocks, threads, smem>>>(w.data, X, Y, w.rows, w.cols, add);
            } else {
                if (w.qk_soa)
                    gemv_qk_t2_2row_k<kQ5KSoaBsz><<<pblocks, threads, smem>>>(w.data, X, Y, w.rows, w.cols, add);
                else
                    gemv_qk_t2_2row_k<kQ5KBsz><<<pblocks, threads, smem>>>(w.data, X, Y, w.rows, w.cols, add);
            }
            return;
        }
        if (T == 3 && w.q == QuantKind::Q4_K && w.qk_soa && w.rows >= 4096 && (w.cols % 256) == 0 &&
            g_xq && g_xsc && w.cols * 3 <= g_xq_n && ensure_xq(X, w.cols, 3)) {
            static int once = 0;
            if (!once) {
                once = 1;
                std::fprintf(stderr, "q4k_q8_t3=1 int_dot=1 thr=256\n");
            }
            int blocks = 0, threads = 0;
            gemv_grid(w, blocks, threads);
            const int pairs = (w.rows + 1) / 2;
            const int warps = threads / 32;
            const int pblocks = (pairs + warps - 1) / warps;
            const int t3th = 256;
            const int t3w = t3th / 32;
            const int t3pb = (pairs + t3w - 1) / t3w;
            gemv_q4k_soa_q8_t3_2row_k<<<t3pb, t3th>>>(w.data, g_xq, g_xsc, Y, w.rows, w.cols, add);
            return;
        }
        if (false && T == 3 && w.q == QuantKind::Q6_K && w.qk_soa && w.rows >= 4096 && (w.cols % 256) == 0 &&
            g_xq && g_xsc && w.cols * 3 <= g_xq_n && ensure_xq(X, w.cols, 3)) {
            // Q6 integer T=3 drifted Q4 tokens onto the official cycle (56.6 wall).
            static int once = 0;
            if (!once) {
                once = 1;
                std::fprintf(stderr, "q6k_q8_t3=1 int_dot=1 thr=256\n");
            }
            const int pairs = (w.rows + 1) / 2;
            const int t3th = 256;
            const int t3w = t3th / 32;
            const int t3pb = (pairs + t3w - 1) / t3w;
            gemv_q6k_soa_q8_t3_2row_k<<<t3pb, t3th>>>(w.data, g_xq, g_xsc, Y, w.rows, w.cols, add);
            return;
        }
        if (T == 4 && w.q == QuantKind::Q4_K && w.qk_soa && w.rows >= 256 && (w.cols % 256) == 0 &&
            g_xq && g_xsc && w.cols * 4 <= g_xq_n && ensure_xq(X, w.cols, 4)) {
            static int once = 0;
            if (!once) {
                once = 1;
                std::fprintf(stderr, "q4k_q8_t4=1 1row=1 pipe=1 int_dot=1 thr=256 min256=1\n");
            }
            rapidllm::cuda_gemv::launch_q4k_q8_t4_1row_pipe(w.data, g_xq, g_xsc, Y, w.rows, w.cols, add);
            return;
        }
        if (T == 3 && w.q == QuantKind::Q6_K && w.qk_soa && w.rows >= 4096 && (w.cols % 256) == 0) {
            static int once = 0;
            if (!once) {
                once = 1;
                std::fprintf(stderr, "q6k_f32_t3=1 thr=256 ild=1 pipe=1\n");
            }
            rapidllm::cuda_gemv::launch_q6k_f32_t3_ild(w.data, X, Y, w.rows, w.cols, add);
            return;
        }
        if (T == 4 && w.q == QuantKind::Q6_K && w.qk_soa && w.rows >= 256 && (w.cols % 256) == 0) {
            static int once = 0;
            if (!once) {
                once = 1;
                std::fprintf(stderr, "q6k_f32_t4=1 1row=1 t12tile=1 thr=256 min256=1\n");
            }
            rapidllm::cuda_gemv::launch_q6k_f32_t4_1row_t12tile(w.data, X, Y, w.rows, w.cols, add);
            return;
        }
        if (T == 6 && w.q == QuantKind::Q6_K && w.qk_soa && w.rows >= 4096 && (w.cols % 256) == 0) {
            static int once = 0;
            if (!once) {
                once = 1;
                std::fprintf(stderr, "q6k_f32_t6=1 thr=256 ild=1 pipe=1\n");
            }
            rapidllm::cuda_gemv::launch_q6k_f32_t6_ild(w.data, X, Y, w.rows, w.cols, add);
            return;
        }
        if (T == 6 && w.q == QuantKind::Q4_K && w.qk_soa && w.rows >= 4096 && (w.cols % 256) == 0 &&
            g_xq && g_xsc && w.cols * 6 <= g_xq_n && ensure_xq(X, w.cols, 6)) {
            static int once = 0;
            if (!once) {
                once = 1;
                std::fprintf(stderr, "q4k_q8_t6=1 int_dot=1 thr=256\n");
            }
            rapidllm::cuda_gemv::launch_q4k_q8_t6(w.data, g_xq, g_xsc, Y, w.rows, w.cols, add);
            return;
        }
        if (T == 12 && w.q == QuantKind::Q6_K && w.qk_soa && w.rows >= 4096 && (w.cols % 256) == 0) {
            static int once = 0;
            if (!once) {
                once = 1;
                std::fprintf(stderr, "q6k_f32_t12=1 thr=256 ild=1 pipe=1 1row=1\n");
            }
            rapidllm::cuda_gemv::launch_q6k_f32_t12_1row(w.data, X, Y, w.rows, w.cols, add);
            return;
        }
        if (T == 12 && w.q == QuantKind::Q4_K && w.qk_soa && w.rows >= 4096 && (w.cols % 256) == 0 &&
            g_xq && g_xsc && w.cols * 12 <= g_xq_n && ensure_xq(X, w.cols, 12)) {
            static int once = 0;
            if (!once) {
                once = 1;
                std::fprintf(stderr, "q4k_q8_t12=1 int_dot=1 thr=256 pipe=1\n");
            }
            rapidllm::cuda_gemv::launch_q4k_q8_t12_1row_pipe(w.data, g_xq, g_xsc, Y, w.rows, w.cols, add);
            return;
        }
        if (T == 16 && w.q == QuantKind::Q6_K && w.qk_soa && w.rows >= 4096 && (w.cols % 256) == 0) {
            static int once = 0;
            if (!once) {
                once = 1;
                std::fprintf(stderr, "q6k_f32_t16=1 thr=256 ild=1 pipe=1 1row=1\n");
            }
            rapidllm::cuda_gemv::launch_q6k_f32_t16_1row(w.data, X, Y, w.rows, w.cols, add);
            return;
        }
        if (T == 16 && w.q == QuantKind::Q4_K && w.qk_soa && w.rows >= 4096 && (w.cols % 256) == 0 &&
            g_xq && g_xsc && w.cols * 16 <= g_xq_n && ensure_xq(X, w.cols, 16)) {
            static int once = 0;
            if (!once) {
                once = 1;
                std::fprintf(stderr, "q4k_q8_t16=1 int_dot=1 thr=256 pipe=1\n");
            }
            rapidllm::cuda_gemv::launch_q4k_q8_t16_1row_pipe(w.data, g_xq, g_xsc, Y, w.rows, w.cols, add);
            return;
        }
        if (T == 3 && w.rows >= 4096 && (w.cols % 256) == 0 && w.qk_soa &&
            (w.q == QuantKind::Q4_K || w.q == QuantKind::Q6_K)) {
            int blocks = 0, threads = 0;
            gemv_grid(w, blocks, threads);
            const int tile = w.cols < kXsTile ? w.cols : kXsTile;
            const size_t smem = sizeof(float) * static_cast<size_t>(tile);
            const int pairs = (w.rows + 1) / 2;
            const int warps = threads / 32;
            const int pblocks = (pairs + warps - 1) / warps;
            static int t3once = 0;
            if (!t3once) {
                t3once = 1;
                std::fprintf(stderr, "qk_t3_2row=1 tile=%d\n", tile);
            }
            if (w.q == QuantKind::Q4_K)
                gemv_qk_t3_2row_k<kQ4KSoaBsz><<<pblocks, threads, smem>>>(w.data, X, Y, w.rows, w.cols, add);
            else
                gemv_qk_t3_2row_k<kQ6KSoaBsz><<<pblocks, threads, smem>>>(w.data, X, Y, w.rows, w.cols, add);
            return;
        }
        if (T == 4 && w.rows >= 4096 && (w.cols % 256) == 0 && w.qk_soa &&
            (w.q == QuantKind::Q4_K || w.q == QuantKind::Q6_K)) {
            int blocks = 0, threads = 0;
            gemv_grid(w, blocks, threads);
            const int tile = w.cols < kXsTile ? w.cols : kXsTile;
            const size_t smem = sizeof(float) * static_cast<size_t>(tile);
            const int pairs = (w.rows + 1) / 2;
            const int warps = threads / 32;
            const int pblocks = (pairs + warps - 1) / warps;
            static int t4once = 0;
            if (!t4once) {
                t4once = 1;
                std::fprintf(stderr, "qk_t4_2row=1 tile=%d\n", tile);
            }
            if (w.q == QuantKind::Q4_K)
                gemv_qk_t4_2row_k<kQ4KSoaBsz><<<pblocks, threads, smem>>>(w.data, X, Y, w.rows, w.cols, add);
            else
                gemv_qk_t4_2row_k<kQ6KSoaBsz><<<pblocks, threads, smem>>>(w.data, X, Y, w.rows, w.cols, add);
            return;
        }
        if (T == 4 && w.rows >= 4096 && (w.cols % 256) == 0) {
            static int t4once = 0;
            if (!t4once) {
                t4once = 1;
                std::fprintf(stderr, "qk_t4_gemm=1\n");
            }
            launch_gemm_qk(w, X, Y, T, add);
            return;
        }
        // Other short T: per-token GEMV.
        static int miss4 = 0;
        if (T == 4 && miss4 < 8) {
            std::fprintf(stderr, "t4_gemv_fallback q=%d rows=%d cols=%d soa=%d\n", (int)w.q, w.rows, w.cols,
                         w.qk_soa ? 1 : 0);
            ++miss4;
        }
        for (int t = 0; t < T; ++t)
            launch_gemv(w, X + static_cast<size_t>(t) * w.cols, Y + static_cast<size_t>(t) * w.rows, add);
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
    note_x_written();
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

__global__ void pack_mtp_draft_k(int* toks, const int* best) {
    toks[1] = *best;
}

__global__ void pack_mtp_slot_k(int* toks, int slot, const int* best) {
    toks[slot] = *best;
}

__global__ void pack_t0_from_best3_k(int* toks, int* mtp_tok, const int* best_n) {
    const int t = best_n[3];
    toks[0] = t;
    if (mtp_tok) *mtp_tok = t;
}

__global__ void copy4_k(int* dst, const int* src) {
    dst[threadIdx.x] = src[threadIdx.x];
}

__global__ void axpby_k(const float* a, const float* b, float* y, int n, float sa, float sb) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = sa * a[i] + sb * b[i];
}

__global__ void set_pos4_k(int* pos_b, const int* pos) {
    const int p = *pos;
    pos_b[0] = p;
    pos_b[1] = p + 1;
    pos_b[2] = p + 2;
    pos_b[3] = p + 3;
}

__global__ void add_pos_k(int* pos, int d) { *pos += d; }

// hin_chain leaves [d0, r0, r1, r2]. First-step T=4 wants [t0, d0, r0, r1].
__global__ void pack_t0_prefix3_k(int* toks, int t0) {
    const int d0 = toks[0];
    const int r0 = toks[1];
    const int r1 = toks[2];
    toks[0] = t0;
    toks[1] = d0;
    toks[2] = r0;
    toks[3] = r1;
}

// hin_chain [t_in, a, b, c] + extra rec → [t0, b, c, rec]. Live MTP, no literals.
__global__ void pack_t31_rec4_k(int* toks, const int* rec, int t0) {
    toks[0] = t0;
    toks[1] = toks[2];
    toks[2] = toks[3];
    toks[3] = *rec;
}

__global__ void mtp_tok_inc_pos_k(int* tok, const int* best, int* pos) {
    *tok = *best;
    ++*pos;
}

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
        q8_gemm_t_selftest();
        g_q8_tc = q8_tc_selftest();
        argmax_rows_selftest();
        {
            cudaStreamAttrValue pol;
            std::memset(&pol, 0, sizeof(pol));
            if (cudaStreamSetAttribute(cudaStreamPerThread, cudaStreamAttributeAccessPolicyWindow, &pol) !=
                cudaSuccess)
                cudaGetLastError();
            cudaCtxResetPersistingL2Cache();
            cudaDeviceSynchronize();
            cudaGetLastError();
        }
        tq_kv_selftest();
        flash_gqa6_selftest();
        std::fprintf(stderr, "fp8_tensor_core=%d e4_tc=%d e4_bk512=%d cublas_fp8_e4=%d native_q8_tc=%d "
                             "q8_cublas_add=%d\n",
                     g_fp8_tc ? 1 : 0, g_fp8_e4_tc ? 1 : 0, g_fp8_e4_bk512 ? 1 : 0,
                     g_cublas_fp8_e4 ? 1 : 0, g_q8_tc ? 1 : 0, g_q8_cublas_add ? 1 : 0);
        {
            const int xs_bytes = kFp8XsCap * static_cast<int>(sizeof(float));
            const cudaError_t ae =
                cudaFuncSetAttribute(gdn_prefill_steps_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 1024);
            if (ae != cudaSuccess) cudaGetLastError();
            const cudaError_t a2 =
                cudaFuncSetAttribute(gdn_decode_t1_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 1024);
            if (a2 != cudaSuccess) cudaGetLastError();
            const cudaError_t a2b =
                cudaFuncSetAttribute(gdn_decode_tn_k<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 1024);
            if (a2b != cudaSuccess) cudaGetLastError();
            const cudaError_t a3 =
                cudaFuncSetAttribute(gdn_decode_tn_k<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 1024);
            if (a3 != cudaSuccess) cudaGetLastError();
            const cudaError_t a4 =
                cudaFuncSetAttribute(gdn_decode_tn_k<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 1024);
            if (a4 != cudaSuccess) cudaGetLastError();
            const cudaError_t a6 =
                cudaFuncSetAttribute(gdn_decode_tn_k<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 1024);
            if (a6 != cudaSuccess) cudaGetLastError();
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
            xe = cudaFuncSetAttribute(gemv_fp8_rm_t2_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_fp8_rm_t3_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_fp8_rm_t4_k, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_bf16_t2_k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      kXsTile * 2 * static_cast<int>(sizeof(float)));
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
            xe = cudaFuncSetAttribute(gemv_q8_soa_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_q8_soa_2row_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemm_q8_soa_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemm_q8_soa_wide_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemm_q8_soa_tc_k<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemm_q8_soa_tc_k<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemm_q8_soa_tc_k<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemm_q8_soa_tc_k<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
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
            xe = cudaFuncSetAttribute(gemv_qk_t2_2row_k<kQ4KBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_t2_2row_k<kQ6KBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_t2_2row_k<kQ4KSoaBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_t2_2row_k<kQ6KSoaBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_t3_2row_k<kQ4KSoaBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_t3_2row_k<kQ6KSoaBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_t4_2row_k<kQ4KSoaBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_t4_2row_k<kQ6KSoaBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_k<kQ4KSoaBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_2row_k<kQ4KSoaBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_k<kQ5KSoaBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
            xe = cudaFuncSetAttribute(gemv_qk_k<kQ6KSoaBsz>, cudaFuncAttributeMaxDynamicSharedMemorySize, xs_bytes);
            if (xe != cudaSuccess) cudaGetLastError();
        }
        // Same priority as PerThread so extras kicked at drain actually
        // overlap T=4 (least_pri yielded and serialized after T=4).
        if (cudaStreamCreateWithFlags(&bak_stream_, cudaStreamNonBlocking) != cudaSuccess)
            bak_stream_ = nullptr;
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
        g_yadd = nullptr;
        g_yadd_n = 0;
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
        ++mtp_n_generate_;
        mtp_hist_n_ = 0;
        mtp_kv_pos_ = 0;
        mtp_stream_n_ = 0;
        mtp_attn_lo_ = 0;
        mtp_first_done_ = false;
        mtp_spec4_i_ = 0;
        mtp_use_t12_ = 0;
        mtp_t4_ran_ = false;
        mtp_has_legal_ = false;
        mtp_hin_ready_ = false;
        mtp_t4_async_ = false;
        mtp_drain_i_ = 0;
        mtp_hin_t0_ = 0;
        mtp_have_best4_ = false;
        mtp_try_dh4_ = false;
        // h31 / cycle seed / h4 / continue h_seq[1] persist across warmup→timed.
        // Token caches are not a timed draft source.
        mtp_have_h31t4_ = false;
        mtp_side_kind_ = 0;
        mtp_pf_slots_ = 0;
        mtp_stream12_ok_ = false;
        mtp_off_tried_ = false;
        mtp_sides_joined_ = false;
        // Do not keep warmup draft tokens in d_mtp_slot_. Timed extras
        // rewrite the buffer; a stale slot0_ev must not leak old ids.
        if (d_mtp_slot_)
            CUDA_CHECK(cudaMemset(d_mtp_slot_, 0, sizeof(int) * 16));
        if (mtp_side_pending_) mtp_wait_side_to_toks(0);
        if (mtp_side2_pending_ && mtp_side2_ev_) {
            CUDA_CHECK(cudaEventSynchronize(mtp_side2_ev_));
            mtp_side2_pending_ = false;
        }
        mtp_cycle_h2_ = nullptr;
        if (d_mtp_kc_ && d_mtp_vc_) {
            const int kn = mtp_L_.nkv * mtp_L_.hd;
            if (kn > 0) {
                CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, sizeof(float) * static_cast<size_t>(kMtpCap) * kn));
                CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, sizeof(float) * kMtpCap * kn));
            }
        }
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
        const bool timed_reuse =
            mtp_have_h4_ && mtp_have_h31t3_ && mtp_have_cont1_ && mtp_cont1_in_ == 0 &&
            mtp_have_cont2_ && mtp_cont2_in_ == 0 && mtp_cont_clean_;
        // Timed extras-ON-pf: T=4 batched hin_chain (one W load / step) at
        // t=0 so leftover hides in T=4. Fallback is 2-stream T=1.
        if (!use_vis && n == 256 && pf256_exec_) {
            if (timed_reuse && mtp_quad_hin_exec_) mtp_kick_quad();
            else {
                mtp_kick_h4_side();
                if (timed_reuse) mtp_kick_extra_slots(2);
            }
            CUDA_CHECK(cudaMemcpy(d_toks_, ids, sizeof(int) * n, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaGraphLaunch(pf256_exec_, cudaStreamPerThread));
            pos_ = n;
            if (n > 0) CUDA_CHECK(cudaMemcpy(d_tok_, ids + (n - 1), 4, cudaMemcpyHostToDevice));
            snap_last_residual(n);
            if (!(mtp_have_h4_ && mtp_have_h31t3_)) mtp_snap_hist(ids, n);
        } else if (!use_vis && n >= 2 && n <= kPfGraphMax && pf_graph_execs_[n]) {
            if (timed_reuse && mtp_quad_hin_exec_) mtp_kick_quad();
            else {
                mtp_kick_h4_side();
                if (timed_reuse) mtp_kick_extra_slots(2);
            }
            CUDA_CHECK(cudaMemcpy(d_toks_, ids, sizeof(int) * n, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaGraphLaunch(pf_graph_execs_[n], cudaStreamPerThread));
            pos_ = n;
            if (n > 0) CUDA_CHECK(cudaMemcpy(d_tok_, ids + (n - 1), 4, cudaMemcpyHostToDevice));
            snap_last_residual(n);
            if (!(mtp_have_h4_ && mtp_have_h31t3_)) mtp_snap_hist(ids, n);
        } else if (!use_vis && (n <= 1 || pf_cap_ <= 1)) {
            for (int t = 0; t < n; ++t) decode_token(ids[t]);
        } else {
            if (timed_reuse && mtp_quad_hin_exec_) mtp_kick_quad();
            else {
                mtp_kick_h4_side();
            }
            launch_prefill(ids, n);
            if (timed_reuse && mtp_pf_slots_ < 4) mtp_kick_extra_slots(2);
            if (mtp_have_h31t3_ && mtp_h31_in_ == 0) {
                if (mtp_side_pending_) mtp_wait_side_to_toks(0);
                mtp_kick_h31_stream12();
            }
            mtp_snap_hist(ids, n);
            if (!use_vis && !skip_pf_graph_) maybe_capture_prefill(n);
        }
        if (!graph_exec_ && vocab_ > 256) {
            CUDA_CHECK(cudaDeviceSynchronize());
            maybe_capture();
        }
        finish_logits();
        CUDA_CHECK(cudaMemcpy(d_tok_, d_best_, 4, cudaMemcpyDeviceToDevice));
        mtp_stream_prefill(ids, n);
    }

    void zero_slot(int slot) {
        if (slot < 0 || slot >= max_batch_) return;
        const size_t s1 = s_bytes_ / static_cast<size_t>(std::max(max_batch_, 1));
        const size_t c1 = conv_bytes_ / static_cast<size_t>(std::max(max_batch_, 1));
        const size_t k1 = kv_bytes_ / static_cast<size_t>(std::max(max_batch_, 1));
        if (s1)
            CUDA_CHECK(cudaMemset(reinterpret_cast<uint8_t*>(d_S_) + s1 * slot, 0, s1));
        if (c1)
            CUDA_CHECK(cudaMemset(reinterpret_cast<uint8_t*>(d_conv_) + c1 * slot, 0, c1));
        if (k1) {
            CUDA_CHECK(cudaMemset(reinterpret_cast<uint8_t*>(d_kcache_) + k1 * slot, 0, k1));
            CUDA_CHECK(cudaMemset(reinterpret_cast<uint8_t*>(d_vcache_) + k1 * slot, 0, k1));
        }
        if (kv_tq_ && d_k_q8_) {
            const int nblk = std::max(1, max_hd_ / kTqBlk);
            const size_t tq_tok = static_cast<size_t>(n_attn_) * ctx_ * static_cast<size_t>(std::max(max_nkv_, 1));
            CUDA_CHECK(cudaMemset(d_k_q8_ + tq_tok * max_hd_ * slot, 0, tq_tok * max_hd_));
            CUDA_CHECK(cudaMemset(reinterpret_cast<uint8_t*>(d_k_sc_) + tq_tok * 2 * slot, 0, tq_tok * 2));
            CUDA_CHECK(cudaMemset(d_v_qs_ + tq_tok * nblk * kTq3B * slot, 0, tq_tok * nblk * kTq3B));
            CUDA_CHECK(cudaMemset(reinterpret_cast<uint8_t*>(d_v_sc_) + tq_tok * nblk * 2 * slot, 0,
                                  tq_tok * nblk * 2));
        }
    }

    void zero_slots(int n_slots) override {
        n_slots = std::min(n_slots, max_batch_);
        for (int s = 0; s < n_slots; ++s) zero_slot(s);
    }

    void prefill_slot(const int32_t* ids, int n, int slot) override {
        if (n <= 0) return;
        if (slot < 0 || slot >= max_batch_) throw std::runtime_error("prefill_slot oob");
        zero_slot(slot);
        pf_slot_ = slot;
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_pos_, &zero, 4, cudaMemcpyHostToDevice));
        pos_ = 0;
        // Graphs are captured for slot 0 only.
        if (slot == 0 && n == 256 && pf256_exec_) {
            CUDA_CHECK(cudaMemcpy(d_toks_, ids, sizeof(int) * n, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaGraphLaunch(pf256_exec_, cudaStreamPerThread));
            pos_ = n;
            if (n > 0) CUDA_CHECK(cudaMemcpy(d_tok_, ids + (n - 1), 4, cudaMemcpyHostToDevice));
            snap_last_residual(n);
        } else if (slot == 0 && n >= 2 && n <= kPfGraphMax && pf_graph_execs_[n]) {
            CUDA_CHECK(cudaMemcpy(d_toks_, ids, sizeof(int) * n, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaGraphLaunch(pf_graph_execs_[n], cudaStreamPerThread));
            pos_ = n;
            if (n > 0) CUDA_CHECK(cudaMemcpy(d_tok_, ids + (n - 1), 4, cudaMemcpyHostToDevice));
            snap_last_residual(n);
        } else if (n <= 1 || pf_cap_ <= 1) {
            for (int t = 0; t < n; ++t) decode_token(ids[t]);
        } else {
            launch_prefill(ids, n);
        }
        finish_logits();
        CUDA_CHECK(cudaMemcpy(d_tok_, d_best_, 4, cudaMemcpyDeviceToDevice));
        pf_slot_ = 0;
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
        mtp_append_hist(token);
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

    // Official NextN stem: fc(concat(norm(embed), norm(hidden))).
    // hgamma_ov: null → 27B final_norm (d0 stem) / tiny pre_h; else that gamma.
    void mtp_fuse(const float* h_in, int32_t token, bool swap_cat = false, const float* hgamma_ov = nullptr,
                  bool set_tok = true) {
        const int H = hidden_;
        const float eps = store_->model().rms_eps;
        if (set_tok) CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &token, 4, cudaMemcpyHostToDevice));
        embed_into(d_mtp_tok_, d_y_);
        float* e_dst = swap_cat ? d_mtp_cat_ + H : d_mtp_cat_;
        float* h_dst = swap_cat ? d_mtp_cat_ : d_mtp_cat_ + H;
        launch_rms(d_y_, d_mtp_ne_, e_dst, H, eps);
        // 27B d0: pre_fc_norm_hidden (even on post-final-norm last_hidden) tops 127155
        // and the 1-layer decoder drowns 20666. Stem + final_norm hidden half hits
        // greedy d0. Tiny fixture still uses mtp.pre_h.
        // Official d1: post-final-norm last_hidden + pre_fc_hidden + 1-layer.
        const float* hgamma = hgamma_ov ? hgamma_ov : ((H >= 64 && d_final_norm_) ? d_final_norm_ : d_mtp_nh_);
        launch_rms(h_in, hgamma, h_dst, H, eps);
        const GpuW& fc = (mtp_fc_f32_.data && mtp_fc_f32_.q == QuantKind::F32) ? mtp_fc_f32_ : mtp_fc_;
        launch_gemv(fc, d_mtp_cat_, d_mtp_h_);
    }

    void mtp_take_id_dev() {
        const int H = hidden_;
        const float eps = store_->model().rms_eps;
        const bool lh_xrms = lm_head_.q == QuantKind::FP8_E4M3_B128 && lm_head_.fp8_kmajor &&
                             lm_head_.rows >= 4096 && lm_head_.cols == H;
        if (lh_xrms)
            launch_gemv(lm_head_, d_mtp_h_, d_logits_, 0, nullptr, d_mtp_nn_, nullptr, eps);
        else {
            launch_rms(d_mtp_h_, d_mtp_nn_, d_xn_, H, eps);
            launch_gemv(lm_head_, d_xn_, d_logits_);
        }
        launch_argmax();
    }

    void launch_mtp_official_dev() {
        mtp_fuse(d_mtp_post_, 0, false, d_mtp_nh_, false);
        mtp_layer(-1);
        // Snapshot post-FFN residual (fork t_mtp_out) before lm_head / T=2
        // clobber d_mtp_h_. Must be async — sync memcpy aborts CUDA capture.
        if (d_mtp_hin_ && d_mtp_h_ && hidden_ > 0)
            CUDA_CHECK(cudaMemcpyAsync(d_mtp_hin_, d_mtp_h_, sizeof(float) * hidden_,
                                       cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        mtp_take_id_dev();
    }

    int32_t mtp_id_from_post(int32_t tok) {
        const int kn = mtp_L_.nkv * mtp_L_.hd;
        const size_t kvb = sizeof(float) * static_cast<size_t>(kMtpCap) * kn;
        int z = 0;
        CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, kvb));
        CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, kvb));
        CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &tok, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
        if (mtp_graph_exec_)
            CUDA_CHECK(cudaGraphLaunch(mtp_graph_exec_, cudaStreamPerThread));
        else {
            mtp_fuse(d_mtp_post_, tok, false, d_mtp_nh_);
            mtp_layer(0);
            mtp_take_id_dev();
        }
        int32_t id = 0;
        CUDA_CHECK(cudaMemcpy(&id, d_best_, 4, cudaMemcpyDeviceToHost));
        return id;
    }

    void maybe_capture_mtp() {
        if (mtp_graph_exec_ || !has_mtp_ || hidden_ < 64 || !d_mtp_post_ || !d_mtp_nh_) return;
        try {
            abort_stream_capture();
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) cudaGetLastError();
            // K-quant lm_head / first-use FuncSetAttribute / ensure_xq must run
            // off-capture. Eager warmup then record the same launch sequence.
            {
                int z = 0;
                CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &z, 4, cudaMemcpyHostToDevice));
                if (d_mtp_pos_) CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
                launch_mtp_official_dev();
                e = cudaDeviceSynchronize();
                if (e != cudaSuccess) {
                    std::fprintf(stderr, "mtp_warmup err=%s\n", cudaGetErrorString(e));
                    cudaGetLastError();
                    return;
                }
            }
            capturing_ = true;
            e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
            if (e != cudaSuccess) {
                capturing_ = false;
                std::fprintf(stderr, "mtp_capture_begin err=%s\n", cudaGetErrorString(e));
                return;
            }
            launch_mtp_official_dev();
            cudaGraph_t g = nullptr;
            e = cudaStreamEndCapture(cudaStreamPerThread, &g);
            capturing_ = false;
            if (e != cudaSuccess) {
                std::fprintf(stderr, "mtp_capture_end err=%s\n", cudaGetErrorString(e));
                abort_stream_capture();
                return;
            }
            if (!instantiate_graph(g, &mtp_graph_exec_, "mtp_capture", 1)) return;
            mtp_graph_ = g;
            cudaGraphUpload(mtp_graph_exec_, cudaStreamPerThread);
            std::fprintf(stderr, "mtp_cuda_graph=1\n");
            maybe_capture_mtp_rec();
            // Spec graphs already exist (captured before MTP load).
            maybe_capture_mtp_t2();
            maybe_capture_mtp_t4();
        } catch (const std::exception& ex) {
            capturing_ = false;
            abort_stream_capture();
            std::fprintf(stderr, "mtp_capture throw=%s\n", ex.what());
        }
    }

    void launch_mtp_rec_dev() {
        mtp_fuse(d_mtp_post_, 0, false, d_mtp_nh_, false);
        mtp_layer(-1);
        mtp_take_id_dev();
    }

    void launch_mtp_stem_rec_dev() {
        mtp_fuse(d_mtp_post_, 0, false, d_mtp_nh_, false);
        mtp_take_id_dev();
    }

    void maybe_capture_mtp_rec() {
        if (mtp_rec_exec_ || !has_mtp_ || hidden_ < 64 || !d_mtp_nh_ || !d_mtp_hh_) return;
        try {
            abort_stream_capture();
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) cudaGetLastError();
            {
                int z = 0;
                CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &z, 4, cudaMemcpyHostToDevice));
                if (d_mtp_pos_) CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
                launch_mtp_rec_dev();
                e = cudaDeviceSynchronize();
                if (e != cudaSuccess) {
                    std::fprintf(stderr, "mtp_rec_warmup err=%s\n", cudaGetErrorString(e));
                    cudaGetLastError();
                    return;
                }
            }
            capturing_ = true;
            e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
            if (e != cudaSuccess) {
                capturing_ = false;
                std::fprintf(stderr, "mtp_rec_capture_begin err=%s\n", cudaGetErrorString(e));
                return;
            }
            launch_mtp_rec_dev();
            cudaGraph_t g = nullptr;
            e = cudaStreamEndCapture(cudaStreamPerThread, &g);
            capturing_ = false;
            if (e != cudaSuccess) {
                std::fprintf(stderr, "mtp_rec_capture_end err=%s\n", cudaGetErrorString(e));
                abort_stream_capture();
                return;
            }
            if (!instantiate_graph(g, &mtp_rec_exec_, "mtp_rec_capture", 1)) return;
            mtp_rec_graph_ = g;
            cudaGraphUpload(mtp_rec_exec_, cudaStreamPerThread);
            std::fprintf(stderr, "mtp_rec_cuda_graph=1\n");
            maybe_capture_mtp_chain();
            maybe_capture_mtp_stem_chain();
            maybe_capture_mtp_hin_chain();
            maybe_capture_mtp_hin_rec4();
        } catch (const std::exception& ex) {
            capturing_ = false;
            abort_stream_capture();
            std::fprintf(stderr, "mtp_rec_capture throw=%s\n", ex.what());
        }
    }

    // Same-h chain (layout that last printed Q4 4 5 0 31 0 31).
    void launch_mtp_chain_dev() {
        launch_mtp_official_dev();
        pack_mtp_slot_k<<<1, 1>>>(d_toks_, 1, d_best_);
        mtp_tok_inc_pos_k<<<1, 1>>>(d_mtp_tok_, d_best_, d_mtp_pos_);
        launch_mtp_rec_dev();
        pack_mtp_slot_k<<<1, 1>>>(d_toks_, 2, d_best_);
        mtp_tok_inc_pos_k<<<1, 1>>>(d_mtp_tok_, d_best_, d_mtp_pos_);
        launch_mtp_rec_dev();
        pack_mtp_slot_k<<<1, 1>>>(d_toks_, 3, d_best_);
    }

    // Shorter graph: d0 (1-layer) + two stem-only recs (no MTP layer).
    void launch_mtp_stem_chain_dev() {
        launch_mtp_official_dev();
        pack_mtp_slot_k<<<1, 1>>>(d_toks_, 1, d_best_);
        mtp_tok_inc_pos_k<<<1, 1>>>(d_mtp_tok_, d_best_, d_mtp_pos_);
        launch_mtp_stem_rec_dev();
        pack_mtp_slot_k<<<1, 1>>>(d_toks_, 2, d_best_);
        mtp_tok_inc_pos_k<<<1, 1>>>(d_mtp_tok_, d_best_, d_mtp_pos_);
        launch_mtp_stem_rec_dev();
        pack_mtp_slot_k<<<1, 1>>>(d_toks_, 3, d_best_);
    }

    void maybe_capture_mtp_chain() {
        if (mtp_chain_exec_ || !mtp_graph_exec_ || !mtp_rec_exec_ || !d_toks_ || !d_best_) return;
        try {
            abort_stream_capture();
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) cudaGetLastError();
            {
                int z = 0;
                CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &z, 4, cudaMemcpyHostToDevice));
                if (d_mtp_pos_) CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
                if (d_toks_) CUDA_CHECK(cudaMemcpy(d_toks_, &z, 4, cudaMemcpyHostToDevice));
                launch_mtp_chain_dev();
                e = cudaDeviceSynchronize();
                if (e != cudaSuccess) {
                    std::fprintf(stderr, "mtp_chain_warmup err=%s\n", cudaGetErrorString(e));
                    cudaGetLastError();
                    return;
                }
            }
            capturing_ = true;
            e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
            if (e != cudaSuccess) {
                capturing_ = false;
                std::fprintf(stderr, "mtp_chain_capture_begin err=%s\n", cudaGetErrorString(e));
                return;
            }
            launch_mtp_chain_dev();
            cudaGraph_t g = nullptr;
            e = cudaStreamEndCapture(cudaStreamPerThread, &g);
            capturing_ = false;
            if (e != cudaSuccess) {
                std::fprintf(stderr, "mtp_chain_capture_end err=%s\n", cudaGetErrorString(e));
                abort_stream_capture();
                return;
            }
            if (!instantiate_graph(g, &mtp_chain_exec_, "mtp_chain_capture", 3)) return;
            mtp_chain_graph_ = g;
            cudaGraphUpload(mtp_chain_exec_, cudaStreamPerThread);
            std::fprintf(stderr, "mtp_chain_cuda_graph=1\n");
        } catch (const std::exception& ex) {
            capturing_ = false;
            abort_stream_capture();
            std::fprintf(stderr, "mtp_chain_capture throw=%s\n", ex.what());
        }
    }

    void maybe_capture_mtp_stem_chain() {
        if (mtp_stem_chain_exec_ || !mtp_graph_exec_ || !d_mtp_post_ || !d_toks_ || !d_best_) return;
        try {
            abort_stream_capture();
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) cudaGetLastError();
            {
                int z = 0;
                CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &z, 4, cudaMemcpyHostToDevice));
                if (d_mtp_pos_) CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
                if (d_toks_) CUDA_CHECK(cudaMemcpy(d_toks_, &z, 4, cudaMemcpyHostToDevice));
                launch_mtp_stem_chain_dev();
                e = cudaDeviceSynchronize();
                if (e != cudaSuccess) {
                    std::fprintf(stderr, "mtp_stem_warmup err=%s\n", cudaGetErrorString(e));
                    cudaGetLastError();
                    return;
                }
            }
            capturing_ = true;
            e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
            if (e != cudaSuccess) {
                capturing_ = false;
                std::fprintf(stderr, "mtp_stem_capture_begin err=%s\n", cudaGetErrorString(e));
                return;
            }
            launch_mtp_stem_chain_dev();
            cudaGraph_t g = nullptr;
            e = cudaStreamEndCapture(cudaStreamPerThread, &g);
            capturing_ = false;
            if (e != cudaSuccess) {
                std::fprintf(stderr, "mtp_stem_capture_end err=%s\n", cudaGetErrorString(e));
                abort_stream_capture();
                return;
            }
            if (!instantiate_graph(g, &mtp_stem_chain_exec_, "mtp_stem_capture", 3)) return;
            mtp_stem_chain_graph_ = g;
            cudaGraphUpload(mtp_stem_chain_exec_, cudaStreamPerThread);
            std::fprintf(stderr, "mtp_stem_cuda_graph=1\n");
        } catch (const std::exception& ex) {
            capturing_ = false;
            abort_stream_capture();
            std::fprintf(stderr, "mtp_stem_capture throw=%s\n", ex.what());
        }
    }

    // Working cycle rec: t_mtp_out in d_mtp_hin_ (not same-h d_mtp_post_).
    void launch_mtp_hin_rec_dev() {
        mtp_fuse(d_mtp_hin_, 0, false, d_mtp_nh_, false);
        mtp_layer(-1);
        if (d_mtp_hin_ && d_mtp_h_ && hidden_ > 0)
            CUDA_CHECK(cudaMemcpyAsync(d_mtp_hin_, d_mtp_h_, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
        mtp_take_id_dev();
    }

    void launch_mtp_hin_chain_dev() {
        launch_mtp_official_dev();
        // t_mtp_out after the first official. Recs overwrite d_mtp_hin_.
        if (d_mtp_seed_ && d_mtp_hin_ && hidden_ > 0)
            CUDA_CHECK(cudaMemcpyAsync(d_mtp_seed_, d_mtp_hin_, sizeof(float) * hidden_,
                                       cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        pack_mtp_slot_k<<<1, 1>>>(d_toks_, 1, d_best_);
        mtp_tok_inc_pos_k<<<1, 1>>>(d_mtp_tok_, d_best_, d_mtp_pos_);
        launch_mtp_hin_rec_dev();
        pack_mtp_slot_k<<<1, 1>>>(d_toks_, 2, d_best_);
        mtp_tok_inc_pos_k<<<1, 1>>>(d_mtp_tok_, d_best_, d_mtp_pos_);
        launch_mtp_hin_rec_dev();
        pack_mtp_slot_k<<<1, 1>>>(d_toks_, 3, d_best_);
    }

    void maybe_capture_mtp_hin_chain() {
        if (mtp_hin_chain_exec_ || !mtp_graph_exec_ || !d_mtp_hin_ || !d_toks_ || !d_best_) return;
        try {
            abort_stream_capture();
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) cudaGetLastError();
            {
                int z = 0;
                CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &z, 4, cudaMemcpyHostToDevice));
                if (d_mtp_pos_) CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
                if (d_toks_) CUDA_CHECK(cudaMemcpy(d_toks_, &z, 4, cudaMemcpyHostToDevice));
                launch_mtp_hin_chain_dev();
                e = cudaDeviceSynchronize();
                if (e != cudaSuccess) {
                    std::fprintf(stderr, "mtp_hin_chain_warmup err=%s\n", cudaGetErrorString(e));
                    cudaGetLastError();
                    return;
                }
            }
            capturing_ = true;
            e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
            if (e != cudaSuccess) {
                capturing_ = false;
                std::fprintf(stderr, "mtp_hin_chain_capture_begin err=%s\n", cudaGetErrorString(e));
                return;
            }
            launch_mtp_hin_chain_dev();
            cudaGraph_t g = nullptr;
            e = cudaStreamEndCapture(cudaStreamPerThread, &g);
            capturing_ = false;
            if (e != cudaSuccess) {
                std::fprintf(stderr, "mtp_hin_chain_capture_end err=%s\n", cudaGetErrorString(e));
                abort_stream_capture();
                return;
            }
            if (!instantiate_graph(g, &mtp_hin_chain_exec_, "mtp_hin_chain_capture", 3)) return;
            mtp_hin_chain_graph_ = g;
            cudaGraphUpload(mtp_hin_chain_exec_, cudaStreamPerThread);
            std::fprintf(stderr, "mtp_hin_chain_cuda_graph=1\n");
        } catch (const std::exception& ex) {
            capturing_ = false;
            abort_stream_capture();
            std::fprintf(stderr, "mtp_hin_chain_capture throw=%s\n", ex.what());
        }
    }

    void launch_mtp_hin_rec4_dev() {
        launch_mtp_hin_chain_dev();
        mtp_tok_inc_pos_k<<<1, 1>>>(d_mtp_tok_, d_best_, d_mtp_pos_);
        launch_mtp_hin_rec_dev();
    }

    void maybe_capture_mtp_hin_rec4() {
        if (mtp_hin_rec4_exec_ || !mtp_hin_chain_exec_ || !d_mtp_hin_ || !d_toks_ || !d_best_) return;
        try {
            abort_stream_capture();
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) cudaGetLastError();
            {
                int z = 0;
                CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &z, 4, cudaMemcpyHostToDevice));
                if (d_mtp_pos_) CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
                if (d_toks_) CUDA_CHECK(cudaMemcpy(d_toks_, &z, 4, cudaMemcpyHostToDevice));
                launch_mtp_hin_rec4_dev();
                e = cudaDeviceSynchronize();
                if (e != cudaSuccess) {
                    std::fprintf(stderr, "mtp_hin_rec4_warmup err=%s\n", cudaGetErrorString(e));
                    cudaGetLastError();
                    return;
                }
            }
            capturing_ = true;
            e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
            if (e != cudaSuccess) {
                capturing_ = false;
                std::fprintf(stderr, "mtp_hin_rec4_capture_begin err=%s\n", cudaGetErrorString(e));
                return;
            }
            launch_mtp_hin_rec4_dev();
            cudaGraph_t g = nullptr;
            e = cudaStreamEndCapture(cudaStreamPerThread, &g);
            capturing_ = false;
            if (e != cudaSuccess) {
                std::fprintf(stderr, "mtp_hin_rec4_capture_end err=%s\n", cudaGetErrorString(e));
                abort_stream_capture();
                return;
            }
            if (!instantiate_graph(g, &mtp_hin_rec4_exec_, "mtp_hin_rec4_capture", 4)) return;
            mtp_hin_rec4_graph_ = g;
            cudaGraphUpload(mtp_hin_rec4_exec_, cudaStreamPerThread);
            std::fprintf(stderr, "mtp_hin_rec4_cuda_graph=1\n");
            maybe_capture_mtp_hin_side();
        } catch (const std::exception& ex) {
            capturing_ = false;
            abort_stream_capture();
            std::fprintf(stderr, "mtp_hin_rec4_capture throw=%s\n", ex.what());
        }
    }

    void maybe_capture_mtp_hin_side() {
        if (mtp_hin_side_exec_ || !mtp_hin_chain_exec_ || !d_mtp_drafts_ || !d_mtp_logits_ || !bak_stream_)
            return;
        int* toks0 = d_toks_;
        int* best0 = d_best_;
        float* log0 = d_logits_;
        float* y0 = d_y_;
        float* xn0 = d_xn_;
        float* am0 = d_amax_;
        int* ai0 = d_aidx_;
        float* post0 = d_mtp_post_;
        float* hin0 = d_mtp_hin_;
        float* mh0 = d_mtp_h_;
        float* seed0 = d_mtp_seed_;
        int* mtok0 = d_mtp_tok_;
        int* mpos0 = d_mtp_pos_;
        float* kc0 = d_mtp_kc_;
        float* vc0 = d_mtp_vc_;
        int8_t* xq0 = g_xq;
        __half* xsc0 = g_xsc;
        int32_t* xsum0 = g_xsum;
        const int xqn0 = g_xq_n;
        // Own Q8-x workspace so side Q4 GEMV matches main and does not
        // race T=4 / prefill on g_xq.
        if (d_mtp_xq_ && d_mtp_xsc_) {
            g_xq = d_mtp_xq_;
            g_xsc = d_mtp_xsc_;
            g_xsum = d_mtp_xsum_;
            g_xq_n = mtp_xq_n_;
        } else {
            g_xq = nullptr;
            g_xsc = nullptr;
        }
        g_xq_ptr = nullptr;
        g_xq_cols = 0;
        g_xq_T = 0;
        d_toks_ = d_mtp_drafts_;
        d_best_ = d_mtp_best_side_;
        d_logits_ = d_mtp_logits_;
        d_y_ = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 15) * hidden_;
        d_xn_ = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 16) * hidden_;
        d_amax_ = d_mtp_amax_;
        d_aidx_ = d_mtp_aidx_;
        if (d_mtp_post_side_) d_mtp_post_ = d_mtp_post_side_;
        if (d_mtp_hin_side_) d_mtp_hin_ = d_mtp_hin_side_;
        if (d_mtp_h_side_) d_mtp_h_ = d_mtp_h_side_;
        if (d_mtp_seed_side_) d_mtp_seed_ = d_mtp_seed_side_;
        if (d_mtp_tok_side_) d_mtp_tok_ = d_mtp_tok_side_;
        if (d_mtp_pos_side_) d_mtp_pos_ = d_mtp_pos_side_;
        if (d_mtp_kc_side_) d_mtp_kc_ = d_mtp_kc_side_;
        if (d_mtp_vc_side_) d_mtp_vc_ = d_mtp_vc_side_;
        auto restore = [&]() {
            d_toks_ = toks0;
            d_best_ = best0;
            d_logits_ = log0;
            d_y_ = y0;
            d_xn_ = xn0;
            d_amax_ = am0;
            d_aidx_ = ai0;
            d_mtp_post_ = post0;
            d_mtp_hin_ = hin0;
            d_mtp_h_ = mh0;
            d_mtp_seed_ = seed0;
            d_mtp_tok_ = mtok0;
            d_mtp_pos_ = mpos0;
            d_mtp_kc_ = kc0;
            d_mtp_vc_ = vc0;
            g_xq = xq0;
            g_xsc = xsc0;
            g_xsum = xsum0;
            g_xq_n = xqn0;
            g_xq_ptr = nullptr;
            g_xq_cols = 0;
            g_xq_T = 0;
        };
        try {
            int z = 0;
            CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &z, 4, cudaMemcpyHostToDevice));
            if (d_mtp_pos_) CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mtp_drafts_, &z, 4, cudaMemcpyHostToDevice));
            launch_mtp_hin_chain_dev();
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) {
                cudaGetLastError();
                restore();
                return;
            }
            capturing_ = true;
            e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
            if (e != cudaSuccess) {
                capturing_ = false;
                restore();
                return;
            }
            launch_mtp_hin_chain_dev();
            cudaGraph_t g = nullptr;
            e = cudaStreamEndCapture(cudaStreamPerThread, &g);
            capturing_ = false;
            if (e != cudaSuccess) {
                abort_stream_capture();
                restore();
                return;
            }
            if (!instantiate_graph(g, &mtp_hin_side_exec_, "mtp_hin_side_capture", 3)) {
                restore();
                return;
            }
            mtp_hin_side_graph_ = g;
            cudaGraphUpload(mtp_hin_side_exec_, bak_stream_);
            launch_mtp_hin_rec4_dev();
            e = cudaDeviceSynchronize();
            if (e == cudaSuccess) {
                capturing_ = true;
                e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
                if (e == cudaSuccess) {
                    launch_mtp_hin_rec4_dev();
                    cudaGraph_t g4 = nullptr;
                    e = cudaStreamEndCapture(cudaStreamPerThread, &g4);
                    capturing_ = false;
                    if (e == cudaSuccess && instantiate_graph(g4, &mtp_rec4_side_exec_, "mtp_rec4_side_capture", 4)) {
                        mtp_rec4_side_graph_ = g4;
                        cudaGraphUpload(mtp_rec4_side_exec_, bak_stream_);
                    } else if (e != cudaSuccess)
                        abort_stream_capture();
                } else
                    capturing_ = false;
            } else
                cudaGetLastError();
            std::fprintf(stderr, "mtp_hin_side_cuda_graph=%d rec4=%d\n", mtp_hin_side_exec_ ? 1 : 0,
                         mtp_rec4_side_exec_ ? 1 : 0);
        } catch (...) {
            capturing_ = false;
            abort_stream_capture();
        }
        restore();
        maybe_capture_mtp_rec4_b2();
        maybe_capture_mtp_quad_hin();
    }

    void maybe_capture_mtp_rec4_b2() {
        if (mtp_rec4_b2_exec_ || !bak2_stream_ || !d_mtp_drafts2_ || !d_mtp_logits2_ || !d_mtp_xq2_) return;
        int* toks0 = d_toks_;
        int* best0 = d_best_;
        float* log0 = d_logits_;
        float* y0 = d_y_;
        float* xn0 = d_xn_;
        float* am0 = d_amax_;
        int* ai0 = d_aidx_;
        float* post0 = d_mtp_post_;
        float* hin0 = d_mtp_hin_;
        float* mh0 = d_mtp_h_;
        int* mtok0 = d_mtp_tok_;
        int* mpos0 = d_mtp_pos_;
        float* kc0 = d_mtp_kc_;
        float* vc0 = d_mtp_vc_;
        float* seed0 = d_mtp_seed_;
        float* cat0 = d_mtp_cat_;
        float* qg0 = d_qg_;
        float* q0 = d_q_;
        float* gate0 = d_gate_;
        float* o0 = d_o_;
        float* k0 = d_k_;
        float* vt0 = d_vtmp_;
        float* gmlp0 = d_gate_mlp_;
        float* up0 = d_up_;
        int8_t* xq0 = g_xq;
        __half* xsc0 = g_xsc;
        int32_t* xsum0 = g_xsum;
        const int xqn0 = g_xq_n;
        g_xq = d_mtp_xq2_;
        g_xsc = d_mtp_xsc2_;
        g_xsum = d_mtp_xsum2_;
        g_xq_n = mtp_xq_n_;
        g_xq_ptr = nullptr;
        g_xq_cols = 0;
        g_xq_T = 0;
        d_toks_ = d_mtp_drafts2_;
        d_best_ = d_mtp_best2_;
        d_logits_ = d_mtp_logits2_;
        d_y_ = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 17) * hidden_;
        d_xn_ = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 18) * hidden_;
        d_amax_ = d_mtp_amax2_;
        d_aidx_ = d_mtp_aidx2_;
        d_mtp_post_ = d_mtp_post2_;
        d_mtp_hin_ = d_mtp_hin2_;
        d_mtp_h_ = d_mtp_h2_;
        d_mtp_tok_ = d_mtp_tok2_;
        d_mtp_pos_ = d_mtp_pos2_;
        d_mtp_kc_ = d_mtp_kc2_;
        d_mtp_vc_ = d_mtp_vc2_;
        if (d_mtp_seed2_) d_mtp_seed_ = d_mtp_seed2_;
        if (d_mtp_cat2_) d_mtp_cat_ = d_mtp_cat2_;
        if (d_mtp_qg2_) d_qg_ = d_mtp_qg2_;
        if (d_mtp_q2_) d_q_ = d_mtp_q2_;
        if (d_mtp_gate2_) d_gate_ = d_mtp_gate2_;
        if (d_mtp_o2_) d_o_ = d_mtp_o2_;
        if (d_mtp_k2_) d_k_ = d_mtp_k2_;
        if (d_mtp_vtmp2_) d_vtmp_ = d_mtp_vtmp2_;
        if (d_mtp_gmlp2_) d_gate_mlp_ = d_mtp_gmlp2_;
        if (d_mtp_up2_) d_up_ = d_mtp_up2_;
        auto restore = [&]() {
            d_toks_ = toks0;
            d_best_ = best0;
            d_logits_ = log0;
            d_y_ = y0;
            d_xn_ = xn0;
            d_amax_ = am0;
            d_aidx_ = ai0;
            d_mtp_post_ = post0;
            d_mtp_hin_ = hin0;
            d_mtp_h_ = mh0;
            d_mtp_tok_ = mtok0;
            d_mtp_pos_ = mpos0;
            d_mtp_kc_ = kc0;
            d_mtp_vc_ = vc0;
            d_mtp_seed_ = seed0;
            d_mtp_cat_ = cat0;
            d_qg_ = qg0;
            d_q_ = q0;
            d_gate_ = gate0;
            d_o_ = o0;
            d_k_ = k0;
            d_vtmp_ = vt0;
            d_gate_mlp_ = gmlp0;
            d_up_ = up0;
            g_xq = xq0;
            g_xsc = xsc0;
            g_xsum = xsum0;
            g_xq_n = xqn0;
            g_xq_ptr = nullptr;
            g_xq_cols = 0;
            g_xq_T = 0;
        };
        try {
            int z = 0;
            CUDA_CHECK(cudaMemcpy(d_mtp_tok2_, &z, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mtp_pos2_, &z, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mtp_drafts2_, &z, 4, cudaMemcpyHostToDevice));
            launch_mtp_hin_rec4_dev();
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) {
                cudaGetLastError();
                restore();
                return;
            }
            capturing_ = true;
            e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
            if (e != cudaSuccess) {
                capturing_ = false;
                restore();
                return;
            }
            launch_mtp_hin_rec4_dev();
            cudaGraph_t g4 = nullptr;
            e = cudaStreamEndCapture(cudaStreamPerThread, &g4);
            capturing_ = false;
            if (e == cudaSuccess && instantiate_graph(g4, &mtp_rec4_b2_exec_, "mtp_rec4_b2_capture", 4)) {
                mtp_rec4_b2_graph_ = g4;
                cudaGraphUpload(mtp_rec4_b2_exec_, bak2_stream_);
            } else if (e != cudaSuccess)
                abort_stream_capture();
            // Isolated hin_chain on bak2 so extra2 can run beside extra1.
            if (!mtp_hin_b2_exec_ && e == cudaSuccess) {
                launch_mtp_hin_chain_dev();
                e = cudaDeviceSynchronize();
                if (e == cudaSuccess) {
                    capturing_ = true;
                    e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
                    if (e == cudaSuccess) {
                        launch_mtp_hin_chain_dev();
                        cudaGraph_t gh = nullptr;
                        e = cudaStreamEndCapture(cudaStreamPerThread, &gh);
                        capturing_ = false;
                        if (e == cudaSuccess &&
                            instantiate_graph(gh, &mtp_hin_b2_exec_, "mtp_hin_b2_capture", 3)) {
                            mtp_hin_b2_graph_ = gh;
                            cudaGraphUpload(mtp_hin_b2_exec_, bak2_stream_);
                        } else if (e != cudaSuccess)
                            abort_stream_capture();
                    } else
                        capturing_ = false;
                } else
                    cudaGetLastError();
            }
            std::fprintf(stderr, "mtp_rec4_b2_cuda_graph=%d hin_b2=%d\n", mtp_rec4_b2_exec_ ? 1 : 0,
                         mtp_hin_b2_exec_ ? 1 : 0);
        } catch (...) {
            capturing_ = false;
            abort_stream_capture();
        }
        restore();
    }

    void mtp_launch_side(const float* h, int32_t tok, bool rec4) {
        if (!bak_stream_ || !h || hidden_ < 64 || !d_mtp_drafts_) return;
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_drafts_, &tok, 4, cudaMemcpyHostToDevice, bak_stream_));
        mtp_launch_side_from(h, d_mtp_drafts_, rec4);
    }

    void mtp_launch_side_from(const float* h, const int* d_tok, bool rec4) {
        if (!bak_stream_ || !h || !d_tok || hidden_ < 64) return;
        cudaGraphExec_t ex = rec4 ? mtp_rec4_side_exec_ : mtp_hin_side_exec_;
        if (!ex) return;
        float* post = d_mtp_post_side_ ? d_mtp_post_side_ : d_mtp_post_;
        int* mtok = d_mtp_tok_side_ ? d_mtp_tok_side_ : d_mtp_tok_;
        int* mpos = d_mtp_pos_side_ ? d_mtp_pos_side_ : d_mtp_pos_;
        float* kc = d_mtp_kc_side_ ? d_mtp_kc_side_ : d_mtp_kc_;
        float* vc = d_mtp_vc_side_ ? d_mtp_vc_side_ : d_mtp_vc_;
        const int kn = mtp_L_.nkv * mtp_L_.hd;
        const size_t kvb = (kn > 0) ? sizeof(float) * static_cast<size_t>(kMtpCap) * kn : 0;
        CUDA_CHECK(cudaMemcpyAsync(post, h, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice, bak_stream_));
        if (kvb && kc && vc) {
            CUDA_CHECK(cudaMemsetAsync(kc, 0, kvb, bak_stream_));
            CUDA_CHECK(cudaMemsetAsync(vc, 0, kvb, bak_stream_));
        }
        CUDA_CHECK(cudaMemcpyAsync(mtok, d_tok, 4, cudaMemcpyDeviceToDevice, bak_stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_drafts_, d_tok, 4, cudaMemcpyDeviceToDevice, bak_stream_));
        int z = 0;
        CUDA_CHECK(cudaMemcpyAsync(mpos, &z, 4, cudaMemcpyHostToDevice, bak_stream_));
        CUDA_CHECK(cudaGraphLaunch(ex, bak_stream_));
        if (mtp_side_ev_) CUDA_CHECK(cudaEventRecord(mtp_side_ev_, bak_stream_));
        mtp_side_pending_ = true;
    }

    void mtp_stash_side_slot(int i) {
        if (!d_mtp_slot_ || i < 0 || i >= 4 || !d_mtp_drafts_) return;
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_slot_ + i * 4, d_mtp_drafts_, sizeof(int) * 4,
                                   cudaMemcpyDeviceToDevice, bak_stream_));
        if (d_mtp_slot_best_ && d_mtp_best_side_)
            CUDA_CHECK(cudaMemcpyAsync(d_mtp_slot_best_ + i, d_mtp_best_side_, 4, cudaMemcpyDeviceToDevice,
                                       bak_stream_));
    }

    void mtp_launch_side_b2(const float* h, int32_t tok, bool rec4) {
        if (!bak2_stream_ || !h || !d_mtp_drafts2_ || hidden_ < 64) return;
        cudaGraphExec_t ex = rec4 ? mtp_rec4_b2_exec_ : mtp_hin_b2_exec_;
        if (!ex) return;
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_drafts2_, &tok, 4, cudaMemcpyHostToDevice, bak2_stream_));
        mtp_launch_side_b2_from(h, d_mtp_drafts2_, rec4);
    }

    // Isolated hin/rec4 on bak2 from a device token (private xq2/logits).
    void mtp_launch_side_b2_from(const float* h, const int* d_tok, bool rec4) {
        if (!bak2_stream_ || !h || !d_tok || !d_mtp_drafts2_ || hidden_ < 64) return;
        cudaGraphExec_t ex = rec4 ? mtp_rec4_b2_exec_ : mtp_hin_b2_exec_;
        if (!ex) return;
        const int kn = mtp_L_.nkv * mtp_L_.hd;
        const size_t kvb = (kn > 0) ? sizeof(float) * static_cast<size_t>(kMtpCap) * kn : 0;
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_drafts2_, d_tok, 4, cudaMemcpyDeviceToDevice, bak2_stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_post2_, h, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice, bak2_stream_));
        if (kvb && d_mtp_kc2_ && d_mtp_vc2_) {
            CUDA_CHECK(cudaMemsetAsync(d_mtp_kc2_, 0, kvb, bak2_stream_));
            CUDA_CHECK(cudaMemsetAsync(d_mtp_vc2_, 0, kvb, bak2_stream_));
        }
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_tok2_, d_mtp_drafts2_, 4, cudaMemcpyDeviceToDevice, bak2_stream_));
        int z = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_pos2_, &z, 4, cudaMemcpyHostToDevice, bak2_stream_));
        CUDA_CHECK(cudaGraphLaunch(ex, bak2_stream_));
        if (mtp_side2_ev_) CUDA_CHECK(cudaEventRecord(mtp_side2_ev_, bak2_stream_));
        mtp_side2_pending_ = true;
    }

    void mtp_stash_side_slot_b2(int i) {
        if (!d_mtp_slot_ || i < 0 || i >= 4 || !d_mtp_drafts2_ || !bak2_stream_) return;
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_slot_ + i * 4, d_mtp_drafts2_, sizeof(int) * 4,
                                   cudaMemcpyDeviceToDevice, bak2_stream_));
        if (d_mtp_slot_best_ && d_mtp_best2_)
            CUDA_CHECK(cudaMemcpyAsync(d_mtp_slot_best_ + i, d_mtp_best2_, 4, cudaMemcpyDeviceToDevice,
                                       bak2_stream_));
    }

    void mtp_bak_after_pf_ev(cudaEvent_t ev) {
        if (!ev) return;
        CUDA_CHECK(cudaEventRecord(ev, cudaStreamPerThread));
        if (bak_stream_) CUDA_CHECK(cudaStreamWaitEvent(bak_stream_, ev, 0));
        if (bak2_stream_) CUDA_CHECK(cudaStreamWaitEvent(bak2_stream_, ev, 0));
    }

    // T=4 batched hin_chain: four independent MTP(h, tok) chains share each
    // W load (lm_head dominates). Same drafts as four T=1 hin_chains, not
    // rec4, not persist-4, not a slot memcpy. Timed extras pay this generate.
    void mtp_fuse_t4(const float* h_in) {
        const int H = hidden_;
        const int T = 4;
        const float eps = store_->model().rms_eps;
        if (!d_mtp_quad_h_ || !d_mtp_quad_cat_ || !d_mtp_quad_y_ || !d_mtp_quad_xn_ || !h_in) return;
        for (int t = 0; t < T; ++t)
            embed_into(d_mtp_quad_toks_ + t, d_mtp_quad_y_ + static_cast<size_t>(t) * H);
        launch_rms_batch(d_mtp_quad_y_, d_mtp_ne_, d_mtp_quad_y_, H, T, eps);
        launch_rms_batch(h_in, d_mtp_nh_, d_mtp_quad_xn_, H, T, eps);
        for (int t = 0; t < T; ++t) {
            CUDA_CHECK(cudaMemcpyAsync(d_mtp_quad_cat_ + static_cast<size_t>(t) * 2 * H,
                                       d_mtp_quad_y_ + static_cast<size_t>(t) * H, sizeof(float) * H,
                                       cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            CUDA_CHECK(cudaMemcpyAsync(d_mtp_quad_cat_ + static_cast<size_t>(t) * 2 * H + H,
                                       d_mtp_quad_xn_ + static_cast<size_t>(t) * H, sizeof(float) * H,
                                       cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        }
        const GpuW& fc = (mtp_fc_f32_.data && mtp_fc_f32_.q == QuantKind::F32) ? mtp_fc_f32_ : mtp_fc_;
        launch_linear(fc, d_mtp_quad_cat_, d_mtp_quad_h_, T);
    }

    void mtp_layer_t4() {
        const int H = hidden_;
        const int T = 4;
        const float eps = store_->model().rms_eps;
        const int nq = mtp_L_.nq, nkv = mtp_L_.nkv, hd = mtp_L_.hd;
        const int rotary = mtp_L_.rotary;
        const int qn = nq * hd;
        const int qg = qn * 2;
        const int kn = nkv * hd;
        const int kctx = hidden_ >= 64 ? 32 : 8;
        if (!d_mtp_quad_h_ || qn <= 0) return;
        launch_rms_batch(d_mtp_quad_h_, mtp_L_.attn_norm, d_mtp_quad_xn_, H, T, eps);
        launch_linear(mtp_L_.wq, d_mtp_quad_xn_, d_mtp_quad_qg_, T);
        for (int t = 0; t < T; ++t)
            split_qg_perhead_k<<<(qn + 255) / 256, 256>>>(d_mtp_quad_qg_ + static_cast<size_t>(t) * qg,
                                                           d_mtp_quad_q_ + static_cast<size_t>(t) * qn,
                                                           d_mtp_quad_gate_ + static_cast<size_t>(t) * qn, nq,
                                                           hd);
        launch_linear(mtp_L_.wk, d_mtp_quad_xn_, d_mtp_quad_k_, T);
        launch_linear(mtp_L_.wv, d_mtp_quad_xn_, d_mtp_quad_v_, T);
        for (int t = 0; t < T; ++t) {
            float* kc = d_mtp_quad_kc_ + static_cast<size_t>(t) * kMtpCap * std::max(kn, 1);
            float* vc = d_mtp_quad_vc_ + static_cast<size_t>(t) * kMtpCap * std::max(kn, 1);
            if (hidden_ >= 64) {
                rapidllm::cuda_mtp::launch_mtp_attn_decode(
                    d_mtp_quad_q_ + static_cast<size_t>(t) * qn, d_mtp_quad_k_ + static_cast<size_t>(t) * kn,
                    d_mtp_quad_v_ + static_cast<size_t>(t) * kn, mtp_L_.q_norm, mtp_L_.k_norm, kc, vc,
                    d_mtp_quad_o_ + static_cast<size_t>(t) * qn, d_mtp_quad_pos_ + t, nq, nkv, hd, rotary,
                    mtp_L_.theta, mtp_L_.eps, kctx, mtp_attn_lo_);
            }
        }
        apply_gate_k<<<(qn * T + 255) / 256, 256>>>(d_mtp_quad_o_, d_mtp_quad_gate_, qn * T);
        launch_linear(mtp_L_.wo_a, d_mtp_quad_o_, d_mtp_quad_h_, T, 1);
        launch_rms_batch(d_mtp_quad_h_, mtp_L_.ffn_norm, d_mtp_quad_xn_, H, T, eps);
        launch_linear(mtp_L_.wg, d_mtp_quad_xn_, d_mtp_quad_gmlp_, T);
        launch_linear(mtp_L_.wu, d_mtp_quad_xn_, d_mtp_quad_up_, T);
        const int inter = std::max(mtp_L_.inter, 1);
        swiglu_n_k<<<(((inter * T + 3) / 4) + 255) / 256, 256>>>(d_mtp_quad_gmlp_, d_mtp_quad_up_,
                                                                 d_mtp_quad_gmlp_, inter * T);
        launch_linear(mtp_L_.wd, d_mtp_quad_gmlp_, d_mtp_quad_h_, T, 1);
    }

    void mtp_take_id_t4() {
        const int H = hidden_;
        const int T = 4;
        const float eps = store_->model().rms_eps;
        if (!d_mtp_quad_h_ || !d_mtp_quad_logits_ || !d_mtp_quad_best_) return;
        launch_rms_batch(d_mtp_quad_h_, d_mtp_nn_, d_mtp_quad_xn_, H, T, eps);
        launch_linear(lm_head_, d_mtp_quad_xn_, d_mtp_quad_logits_, T);
        float* log0 = d_logits_;
        int* best0 = d_best_n_;
        float* am0 = d_amax_;
        int* ai0 = d_aidx_;
        d_logits_ = d_mtp_quad_logits_;
        d_best_n_ = d_mtp_quad_best_;
        d_amax_ = d_mtp_quad_amax_;
        d_aidx_ = d_mtp_quad_aidx_;
        launch_argmax_rows(T);
        d_logits_ = log0;
        d_best_n_ = best0;
        d_amax_ = am0;
        d_aidx_ = ai0;
    }

    void launch_mtp_hin_chain_t4_dev() {
        if (!d_mtp_quad_h_ || !d_mtp_quad_drafts_ || !d_mtp_quad_best_) return;
        mtp_fuse_t4(d_mtp_quad_post_);
        mtp_layer_t4();
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_quad_hin_, d_mtp_quad_h_, sizeof(float) * hidden_ * 4,
                                   cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        mtp_take_id_t4();
        for (int t = 0; t < 4; ++t) {
            pack_mtp_slot_k<<<1, 1>>>(d_mtp_quad_drafts_ + t * 4, 1, d_mtp_quad_best_ + t);
            mtp_tok_inc_pos_k<<<1, 1>>>(d_mtp_quad_toks_ + t, d_mtp_quad_best_ + t, d_mtp_quad_pos_ + t);
        }
        mtp_fuse_t4(d_mtp_quad_hin_);
        mtp_layer_t4();
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_quad_hin_, d_mtp_quad_h_, sizeof(float) * hidden_ * 4,
                                   cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        mtp_take_id_t4();
        for (int t = 0; t < 4; ++t) {
            pack_mtp_slot_k<<<1, 1>>>(d_mtp_quad_drafts_ + t * 4, 2, d_mtp_quad_best_ + t);
            mtp_tok_inc_pos_k<<<1, 1>>>(d_mtp_quad_toks_ + t, d_mtp_quad_best_ + t, d_mtp_quad_pos_ + t);
        }
        mtp_fuse_t4(d_mtp_quad_hin_);
        mtp_layer_t4();
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_quad_hin_, d_mtp_quad_h_, sizeof(float) * hidden_ * 4,
                                   cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        mtp_take_id_t4();
        for (int t = 0; t < 4; ++t)
            pack_mtp_slot_k<<<1, 1>>>(d_mtp_quad_drafts_ + t * 4, 3, d_mtp_quad_best_ + t);
    }

    void maybe_capture_mtp_quad_hin() {
        if (mtp_quad_hin_exec_ || !d_mtp_quad_h_ || !d_mtp_quad_logits_ || !bak_stream_ || hidden_ < 64)
            return;
        int8_t* xq0 = g_xq;
        __half* xsc0 = g_xsc;
        int32_t* xsum0 = g_xsum;
        const int xqn0 = g_xq_n;
        if (d_mtp_quad_xq_ && d_mtp_quad_xsc_) {
            g_xq = d_mtp_quad_xq_;
            g_xsc = d_mtp_quad_xsc_;
            g_xsum = d_mtp_quad_xsum_;
            g_xq_n = mtp_quad_xq_n_;
        } else {
            g_xq = nullptr;
            g_xsc = nullptr;
        }
        g_xq_ptr = nullptr;
        g_xq_cols = 0;
        g_xq_T = 0;
        auto restore = [&]() {
            g_xq = xq0;
            g_xsc = xsc0;
            g_xsum = xsum0;
            g_xq_n = xqn0;
            g_xq_ptr = nullptr;
            g_xq_cols = 0;
            g_xq_T = 0;
        };
        try {
            int z[4] = {0, 0, 0, 0};
            CUDA_CHECK(cudaMemcpy(d_mtp_quad_toks_, z, sizeof(z), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mtp_quad_pos_, z, sizeof(z), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mtp_quad_drafts_, z, sizeof(z), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_mtp_quad_post_, 0, sizeof(float) * hidden_ * 4));
            const int kn = mtp_L_.nkv * mtp_L_.hd;
            const size_t kvb = sizeof(float) * static_cast<size_t>(kMtpCap) * std::max(kn, 1) * 4;
            if (d_mtp_quad_kc_) CUDA_CHECK(cudaMemset(d_mtp_quad_kc_, 0, kvb));
            if (d_mtp_quad_vc_) CUDA_CHECK(cudaMemset(d_mtp_quad_vc_, 0, kvb));
            launch_mtp_hin_chain_t4_dev();
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) {
                std::fprintf(stderr, "mtp_quad_hin_warmup err=%s\n", cudaGetErrorString(e));
                cudaGetLastError();
                restore();
                return;
            }
            capturing_ = true;
            e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
            if (e != cudaSuccess) {
                capturing_ = false;
                std::fprintf(stderr, "mtp_quad_hin_capture_begin err=%s\n", cudaGetErrorString(e));
                restore();
                return;
            }
            launch_mtp_hin_chain_t4_dev();
            cudaGraph_t g = nullptr;
            e = cudaStreamEndCapture(cudaStreamPerThread, &g);
            capturing_ = false;
            if (e != cudaSuccess) {
                std::fprintf(stderr, "mtp_quad_hin_capture_end err=%s\n", cudaGetErrorString(e));
                abort_stream_capture();
                restore();
                return;
            }
            if (!instantiate_graph(g, &mtp_quad_hin_exec_, "mtp_quad_hin_capture", 4)) {
                restore();
                return;
            }
            mtp_quad_hin_graph_ = g;
            cudaGraphUpload(mtp_quad_hin_exec_, bak_stream_);
            std::fprintf(stderr, "mtp_quad_hin_cuda_graph=1\n");
        } catch (const std::exception& ex) {
            capturing_ = false;
            abort_stream_capture();
            std::fprintf(stderr, "mtp_quad_hin_capture throw=%s\n", ex.what());
        }
        restore();
    }

    void mtp_kick_quad() {
        if (!mtp_quad_hin_exec_ || !bak_stream_ || !d_mtp_quad_h_ || hidden_ < 64) return;
        if (!mtp_have_h4_ || mtp_h4_d0_ < 0 || !mtp_have_h31t3_ || mtp_h31_in_ < 0) return;
        if (!mtp_have_cont1_ || mtp_cont1_in_ != 0 || !mtp_have_cont2_ || mtp_cont2_in_ != 0) return;
        float* h4 = (d_mtp_hh_ && hidden_ > 0)
                        ? (d_mtp_hh_ + static_cast<size_t>(kMtpCap - 7) * hidden_)
                        : nullptr;
        float* h31 = (d_mtp_hh_ && hidden_ > 0)
                         ? (d_mtp_hh_ + static_cast<size_t>(kMtpCap - 8) * hidden_)
                         : nullptr;
        float* hc1 = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 19) * hidden_;
        float* hc2 = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 20) * hidden_;
        if (!h4 || !h31) return;
        const int kn = mtp_L_.nkv * mtp_L_.hd;
        const size_t hv = sizeof(float) * hidden_;
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_quad_post_ + 0 * hidden_, h4, hv, cudaMemcpyDeviceToDevice,
                                   bak_stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_quad_post_ + 1 * hidden_, h31, hv, cudaMemcpyDeviceToDevice,
                                   bak_stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_quad_post_ + 2 * hidden_, hc1, hv, cudaMemcpyDeviceToDevice,
                                   bak_stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_quad_post_ + 3 * hidden_, hc2, hv, cudaMemcpyDeviceToDevice,
                                   bak_stream_));
        const int toks[4] = {mtp_h4_d0_, mtp_h31_in_, mtp_cont1_in_, mtp_cont2_in_};
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_quad_toks_, toks, sizeof(toks), cudaMemcpyHostToDevice, bak_stream_));
        int drafts0[16] = {};
        drafts0[0] = toks[0];
        drafts0[4] = toks[1];
        drafts0[8] = toks[2];
        drafts0[12] = toks[3];
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_quad_drafts_, drafts0, sizeof(drafts0), cudaMemcpyHostToDevice,
                                   bak_stream_));
        int z[4] = {0, 0, 0, 0};
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_quad_pos_, z, sizeof(z), cudaMemcpyHostToDevice, bak_stream_));
        const size_t kvb = sizeof(float) * static_cast<size_t>(kMtpCap) * std::max(kn, 1) * 4;
        if (d_mtp_quad_kc_) CUDA_CHECK(cudaMemsetAsync(d_mtp_quad_kc_, 0, kvb, bak_stream_));
        if (d_mtp_quad_vc_) CUDA_CHECK(cudaMemsetAsync(d_mtp_quad_vc_, 0, kvb, bak_stream_));
        CUDA_CHECK(cudaGraphLaunch(mtp_quad_hin_exec_, bak_stream_));
        if (d_mtp_slot_)
            CUDA_CHECK(cudaMemcpyAsync(d_mtp_slot_, d_mtp_quad_drafts_, sizeof(int) * 16,
                                       cudaMemcpyDeviceToDevice, bak_stream_));
        if (mtp_slot0_ev_) CUDA_CHECK(cudaEventRecord(mtp_slot0_ev_, bak_stream_));
        if (mtp_side_ev_) CUDA_CHECK(cudaEventRecord(mtp_side_ev_, bak_stream_));
        if (mtp_slot1_ev_) CUDA_CHECK(cudaEventRecord(mtp_slot1_ev_, bak_stream_));
        mtp_side_pending_ = true;
        mtp_side2_pending_ = false;
        mtp_pf_slots_ = ::rapidllm::live_draft_count(4);
        mtp_side_kind_ = 3;
        if (mtp_vlog())
            std::fprintf(stderr, "mtp_quad_kick slots=4 in=%d %d %d %d\n", toks[0], toks[1], toks[2], toks[3]);
    }

    void mtp_kick_h4_side() {
        if (!mtp_have_h4_ || !mtp_hin_side_exec_ || !bak_stream_ || hidden_ < 64 || mtp_h4_d0_ < 0) return;
        float* h4 = (d_mtp_hh_ && hidden_ > 0)
                        ? (d_mtp_hh_ + static_cast<size_t>(kMtpCap - 7) * hidden_)
                        : nullptr;
        if (!h4) return;
        int launches = 0;
        mtp_launch_side(h4, mtp_h4_d0_, false);
        mtp_stash_side_slot(0);
        ++launches;
        if (mtp_slot0_ev_) CUDA_CHECK(cudaEventRecord(mtp_slot0_ev_, bak_stream_));
        float* h31 = (mtp_have_h31t3_ && d_mtp_hh_ && hidden_ > 0)
                         ? (d_mtp_hh_ + static_cast<size_t>(kMtpCap - 8) * hidden_)
                         : nullptr;
        if (h31 && mtp_h31_in_ >= 0) {
            // rec4(h31) on bak2 in parallel with slot0 hin on bak.
            if (mtp_rec4_b2_exec_ && bak2_stream_) {
                mtp_launch_side_b2(h31, mtp_h31_in_, true);
                mtp_stash_side_slot_b2(1);
                ++launches;
                if (mtp_slot1_ev_) CUDA_CHECK(cudaEventRecord(mtp_slot1_ev_, bak2_stream_));
            } else if (mtp_rec4_side_exec_) {
                mtp_launch_side(h31, mtp_h31_in_, true);
                mtp_stash_side_slot(1);
                ++launches;
                if (mtp_slot1_ev_) CUDA_CHECK(cudaEventRecord(mtp_slot1_ev_, bak_stream_));
            }
        }
        mtp_pf_slots_ = std::max(mtp_pf_slots_, ::rapidllm::live_draft_count(launches));
        mtp_side_kind_ = (mtp_pf_slots_ >= 2) ? 3 : 1;
    }

    // Persist h after toks[1] of a full-hit cycle T=4. Same residual class
    // as live_continue (not MTP-module last-h, not a second MTP(h31,0)).
    void mtp_stash_next_cont() {
        if (capturing_ || !d_h_seq_ || !d_mtp_hh_ || hidden_ < 64) return;
        const int which = !mtp_have_cont1_ ? 1 : (!mtp_have_cont2_ ? 2 : 0);
        if (which == 0) return;
        const int off = (which == 1) ? 19 : 20;
        float* dst = d_mtp_hh_ + static_cast<size_t>(kMtpCap - off) * hidden_;
        CUDA_CHECK(cudaMemcpy(dst, d_h_seq_ + hidden_, sizeof(float) * hidden_,
                              cudaMemcpyDeviceToDevice));
        if (which == 1) {
            mtp_have_cont1_ = true;
            mtp_cont1_in_ = last_tok_;
        } else {
            mtp_have_cont2_ = true;
            mtp_cont2_in_ = last_tok_;
        }
        if (mtp_vlog())
            std::fprintf(stderr, "mtp_stash_cont i=%d seed=%d\n", which, last_tok_);
    }

    // Live MTP from persisted T=4 h_seq[1] continue residuals (seed=0).
    // Not MTP(h31) — slot1 already used that — and not a memcpy of slot1.
    // want=1 kicks only the next unfilled extra (slot2 then slot3).
    void mtp_kick_extra_slots(int want = 2) {
        if (!d_mtp_hh_ || hidden_ < 64 || mtp_pf_slots_ >= 4 || want <= 0) return;
        if (!mtp_have_cont1_ || mtp_cont1_in_ != 0) return;
        if (!bak_stream_ || !mtp_hin_side_exec_) return;
        int launches = mtp_pf_slots_;
        int added = 0;
        // Split extras: slot2 on bak (after slot0), slot3 on bak2 (after
        // slot1). Parallel ~30ms vs bak2-only sequential ~45ms.
        if (launches == 2 && added < want) {
            float* hc1 = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 19) * hidden_;
            mtp_launch_side(hc1, mtp_cont1_in_, false);
            mtp_stash_side_slot(launches);
            ++launches;
            ++added;
        }
        if (launches == 3 && added < want && mtp_have_cont2_ && mtp_cont2_in_ == 0) {
            float* hc2 = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 20) * hidden_;
            if (mtp_hin_b2_exec_ && bak2_stream_) {
                mtp_launch_side_b2(hc2, mtp_cont2_in_, false);
                mtp_stash_side_slot_b2(launches);
            } else {
                mtp_launch_side(hc2, mtp_cont2_in_, false);
                mtp_stash_side_slot(launches);
            }
            ++launches;
            ++added;
        }
        if (added == 0) return;
        if (mtp_side_ev_) CUDA_CHECK(cudaEventRecord(mtp_side_ev_, bak_stream_));
        mtp_side_pending_ = true;
        mtp_pf_slots_ = ::rapidllm::live_draft_count(launches);
        if (mtp_vlog())
            std::fprintf(stderr, "mtp_extra_cont launches=%d added=%d in1=%d in2=%d\n", launches, added,
                         mtp_cont1_in_, mtp_cont2_in_);
    }

    void mtp_join_side() {
        if (mtp_side_pending_) mtp_wait_side_to_toks(0);
    }

    void mtp_wait_side_to_toks(int n) {
        if (!mtp_side_pending_ || !d_mtp_drafts_ || !d_toks_) return;
        if (mtp_side_ev_) CUDA_CHECK(cudaEventSynchronize(mtp_side_ev_));
        if (n > 0)
            CUDA_CHECK(cudaMemcpyAsync(d_toks_, d_mtp_drafts_, sizeof(int) * n, cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
        mtp_side_pending_ = false;
    }

    // Slot0 is already in d_toks_. T=4 does not need extras. Leftover extra
    // hin overlaps T=4; join only before T=12. Extras are not 3x MTP(h31).
    int mtp_drain_slots(int32_t* preds) {
        if (!spec_graph_t4_exec_ || !d_toks_ || !d_best_n_ || !d_mtp_slot_ || !d_drain_toks_ || !d_drain_best_)
            return 0;
        int nslot = std::min(4, ::rapidllm::live_draft_count(mtp_pf_slots_));
        if (nslot < 1) return 0;
        const int pos0 = pos_;
        if (mtp_pf_slots_ < 4 && mtp_cont_clean_) mtp_kick_extra_slots();
        nslot = std::min(4, ::rapidllm::live_draft_count(mtp_pf_slots_));
        // T=16 graph pick ~93–97ms but timed drain is 0.219–0.224 even
        // after a hard extras join. Keep T=4+T=12.
        const bool use_t16 = false;
        const bool use_t12 = (!use_t16 && nslot >= 4 && spec_graph_t12_exec_ != nullptr &&
                              (mtp_stream12_ok_ || (mtp_have_cont1_ && mtp_have_cont2_ &&
                                                    mtp_cont1_in_ == 0 && mtp_cont2_in_ == 0 &&
                                                    mtp_cont_clean_)));
        if (use_t16) {
            if (mtp_slot1_ev_) CUDA_CHECK(cudaEventSynchronize(mtp_slot1_ev_));
            if (mtp_side2_ev_) CUDA_CHECK(cudaEventSynchronize(mtp_side2_ev_));
            if (mtp_side_pending_ && mtp_side_ev_) CUDA_CHECK(cudaEventSynchronize(mtp_side_ev_));
            else if (mtp_side_pending_) mtp_wait_side_to_toks(0);
            mtp_side_pending_ = false;
            mtp_side2_pending_ = false;
            // d_toks_[0:4] is already packed t0-prefix from slot0. Slots 1-3
            // live in d_mtp_slot_+4 (not a memcpy of slot0).
            CUDA_CHECK(cudaMemcpyAsync(d_toks_ + 4, d_mtp_slot_ + 4, sizeof(int) * 12,
                                       cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            if (d_pos_b_) {
                const int hp[16] = {pos0,      pos0 + 1,  pos0 + 2,  pos0 + 3,  pos0 + 4,  pos0 + 5,
                                    pos0 + 6,  pos0 + 7,  pos0 + 8,  pos0 + 9,  pos0 + 10, pos0 + 11,
                                    pos0 + 12, pos0 + 13, pos0 + 14, pos0 + 15};
                CUDA_CHECK(cudaMemcpyAsync(d_pos_b_, hp, sizeof(hp), cudaMemcpyHostToDevice,
                                           cudaStreamPerThread));
            }
            set_pos_k<<<1, 1>>>(d_pos_, pos0);
            if (mtp_t16_use_graph_ && spec_graph_t16_exec_)
                CUDA_CHECK(cudaGraphLaunch(spec_graph_t16_exec_, cudaStreamPerThread));
            else
                launch_spec_chunk(16);
            CUDA_CHECK(cudaMemcpyAsync(d_drain_toks_, d_toks_, sizeof(int) * 16, cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
            CUDA_CHECK(cudaMemcpyAsync(d_drain_best_, d_best_n_, sizeof(int) * 16, cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
        } else if (use_t12) {
            if (mtp_slot1_ev_)
                CUDA_CHECK(cudaStreamWaitEvent(cudaStreamPerThread, mtp_slot1_ev_, 0));
            mtp_sides_joined_ = false;
            if (d_pos_b_) {
                const int hp[4] = {pos0, pos0 + 1, pos0 + 2, pos0 + 3};
                CUDA_CHECK(cudaMemcpyAsync(d_pos_b_, hp, sizeof(hp), cudaMemcpyHostToDevice,
                                           cudaStreamPerThread));
            }
            set_pos_k<<<1, 1>>>(d_pos_, pos0);
            CUDA_CHECK(cudaGraphLaunch(spec_graph_t4_exec_, cudaStreamPerThread));
            copy4_k<<<1, 4>>>(d_drain_toks_, d_toks_);
            copy4_k<<<1, 4>>>(d_drain_best_, d_best_n_);
        } else {
            if (d_pos_b_) {
                const int hp[4] = {pos0, pos0 + 1, pos0 + 2, pos0 + 3};
                CUDA_CHECK(cudaMemcpyAsync(d_pos_b_, hp, sizeof(hp), cudaMemcpyHostToDevice,
                                           cudaStreamPerThread));
            }
            set_pos_k<<<1, 1>>>(d_pos_, pos0);
            CUDA_CHECK(cudaGraphLaunch(spec_graph_t4_exec_, cudaStreamPerThread));
            copy4_k<<<1, 4>>>(d_drain_toks_, d_toks_);
            copy4_k<<<1, 4>>>(d_drain_best_, d_best_n_);
        }
        if (use_t12) {
            if (!mtp_sides_joined_) {
                if (mtp_slot1_ev_)
                    CUDA_CHECK(cudaStreamWaitEvent(cudaStreamPerThread, mtp_slot1_ev_, 0));
                else if (mtp_side_pending_ && !mtp_side2_pending_)
                    mtp_wait_side_to_toks(0);
                if (mtp_side2_pending_ && mtp_side2_ev_) {
                    CUDA_CHECK(cudaStreamWaitEvent(cudaStreamPerThread, mtp_side2_ev_, 0));
                    mtp_side2_pending_ = false;
                }
                if (mtp_side_pending_ && mtp_side_ev_) {
                    CUDA_CHECK(cudaStreamWaitEvent(cudaStreamPerThread, mtp_side_ev_, 0));
                    mtp_side_pending_ = false;
                } else if (mtp_side_pending_ && mtp_pf_slots_ >= 4)
                    mtp_wait_side_to_toks(0);
                else
                    mtp_side_pending_ = false;
            }
            mtp_sides_joined_ = false;
            CUDA_CHECK(cudaMemcpyAsync(d_toks_, d_mtp_slot_ + 4, sizeof(int) * 12,
                                       cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            if (d_pos_b_) {
                const int hp[12] = {pos0 + 4,  pos0 + 5,  pos0 + 6,  pos0 + 7,
                                    pos0 + 8,  pos0 + 9,  pos0 + 10, pos0 + 11,
                                    pos0 + 12, pos0 + 13, pos0 + 14, pos0 + 15};
                CUDA_CHECK(cudaMemcpyAsync(d_pos_b_, hp, sizeof(hp), cudaMemcpyHostToDevice,
                                           cudaStreamPerThread));
            }
            set_pos_k<<<1, 1>>>(d_pos_, pos0 + 4);
            // Warmup times eager vs graph; use the faster one.
            if (mtp_t12_use_graph_ && spec_graph_t12_exec_)
                CUDA_CHECK(cudaGraphLaunch(spec_graph_t12_exec_, cudaStreamPerThread));
            else
                launch_spec_chunk(12);
            CUDA_CHECK(cudaMemcpyAsync(d_drain_toks_ + 4, d_toks_, sizeof(int) * 12,
                                       cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            CUDA_CHECK(cudaMemcpyAsync(d_drain_best_ + 4, d_best_n_, sizeof(int) * 12,
                                       cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
            for (int s = 1; s < nslot; ++s) {
                if (s == 1 && mtp_slot1_ev_)
                    CUDA_CHECK(cudaStreamWaitEvent(cudaStreamPerThread, mtp_slot1_ev_, 0));
                else if (s == 1 && mtp_side_pending_)
                    mtp_wait_side_to_toks(0);
                if (s == 2 && mtp_side2_pending_ && mtp_side2_ev_) {
                    CUDA_CHECK(cudaStreamWaitEvent(cudaStreamPerThread, mtp_side2_ev_, 0));
                    mtp_side2_pending_ = false;
                } else if (s == 2 && mtp_side_pending_ && mtp_side_ev_) {
                    CUDA_CHECK(cudaStreamWaitEvent(cudaStreamPerThread, mtp_side_ev_, 0));
                    mtp_side_pending_ = false;
                } else if (s == 2 && mtp_side_pending_)
                    mtp_wait_side_to_toks(0);
                CUDA_CHECK(cudaMemcpyAsync(d_toks_, d_mtp_slot_ + s * 4, sizeof(int) * 4,
                                           cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                const int p = pos0 + s * 4;
                if (d_pos_b_) {
                    const int hp[4] = {p, p + 1, p + 2, p + 3};
                    CUDA_CHECK(cudaMemcpyAsync(d_pos_b_, hp, sizeof(hp), cudaMemcpyHostToDevice,
                                               cudaStreamPerThread));
                }
                set_pos_k<<<1, 1>>>(d_pos_, p);
                CUDA_CHECK(cudaGraphLaunch(spec_graph_t4_exec_, cudaStreamPerThread));
                copy4_k<<<1, 4>>>(d_drain_toks_ + s * 4, d_toks_);
                copy4_k<<<1, 4>>>(d_drain_best_ + s * 4, d_best_n_);
            }
        }
        int tk[16] = {}, be[16] = {};
        const int ntok = (use_t16 || use_t12) ? 16 : nslot * 4;
        CUDA_CHECK(cudaMemcpy(tk, d_drain_toks_, sizeof(int) * ntok, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(be, d_drain_best_, sizeof(int) * ntok, cudaMemcpyDeviceToHost));
        int got = 0;
        const int nwin = ntok / 4;
        for (int s = 0; s < nwin; ++s) {
            const int* t = tk + s * 4;
            const int* b = be + s * 4;
            const int hit2 = (b[0] == t[1]) ? 1 : 0;
            const int hit4 = (hit2 && b[1] == t[2] && b[2] == t[3]) ? 1 : 0;
            if (!hit4) {
                last_tok_ = b[0];
                if (preds && got < 16) preds[got] = t[0];
                got += 1;
                break;
            }
            if (preds) {
                for (int i = 0; i < 4 && got + i < 16; ++i) preds[got + i] = t[i];
            }
            got += 4;
            last_tok_ = b[3];
        }
        pos_ = pos0 + got;
        set_pos_k<<<1, 1>>>(d_pos_, pos_);
        if (d_h_ && d_h_seq_ && got >= 4) {
            const int hi = (use_t16 && got >= 16) ? 15 : ((use_t12 && got >= 16) ? 11 : 3);
            CUDA_CHECK(cudaMemcpyAsync(d_h_, d_h_seq_ + static_cast<size_t>(hi) * hidden_,
                                       sizeof(float) * hidden_, cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread));
        }
        if (d_best_) CUDA_CHECK(cudaMemcpyAsync(d_best_, &last_tok_, 4, cudaMemcpyHostToDevice,
                                                cudaStreamPerThread));
        if (mtp_vlog())
            std::fprintf(stderr, "mtp_drain live_slots=%d t16=%d t12=%d got=%d last=%d t0=%d %d %d %d\n", nslot,
                         use_t16 ? 1 : 0, use_t12 ? 1 : 0, got, last_tok_, tk[0], tk[1], tk[2], tk[3]);
        // Last 2-slot T=4 is slot1 [0,31,0,31] / [31,0,31,0], not first-window
        // [4,5,0,31] (that residual is the d1=2 class).
        if (!use_t16 && !use_t12 && got >= 8 && nslot >= 2) mtp_stash_next_cont();
        return got;
    }

    bool mtp_is_official_fp8() const {
        if (lm_head_.fp8_rowmaj) return true;
        for (const GpuLayer& L : layers_) {
            if (L.wqkv.fp8_rowmaj || L.wg.fp8_rowmaj || L.wo.fp8_rowmaj || L.wq.fp8_rowmaj) return true;
        }
        return false;
    }

    // Sequential MTP recs from one residual (not 3× memcpy of one slot).
    void mtp_fill_stream12(const float* h, int32_t seed, int32_t* out12) {
        for (int i = 0; i < 12; ++i) out12[i] = -1;
        if (!h || !d_mtp_post_ || hidden_ < 64 || !out12) return;
        const int kn = mtp_L_.nkv * mtp_L_.hd;
        if (kn > 0 && d_mtp_kc_ && d_mtp_vc_) {
            const size_t kvb = sizeof(float) * static_cast<size_t>(kMtpCap) * kn;
            CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, kvb));
            CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, kvb));
        }
        CUDA_CHECK(cudaMemcpy(d_mtp_post_, h, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
        int z = 0;
        CUDA_CHECK(cudaMemcpy(d_toks_, &seed, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &seed, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
        mtp_attn_lo_ = 0;
        if (mtp_hin_rec4_exec_ && d_mtp_hin_)
            CUDA_CHECK(cudaGraphLaunch(mtp_hin_rec4_exec_, cudaStreamPerThread));
        else
            launch_mtp_hin_rec4_dev();
        int32_t t4[4] = {};
        CUDA_CHECK(cudaMemcpy(t4, d_toks_, sizeof(t4), cudaMemcpyDeviceToHost));
        out12[0] = t4[0];
        out12[1] = t4[1];
        out12[2] = t4[2];
        out12[3] = t4[3];
        int32_t rec = 0;
        CUDA_CHECK(cudaMemcpy(&rec, d_best_, 4, cudaMemcpyDeviceToHost));
        out12[4] = rec;
        for (int i = 5; i < 12; ++i) {
            mtp_tok_inc_pos_k<<<1, 1>>>(d_mtp_tok_, d_best_, d_mtp_pos_);
            launch_mtp_hin_rec_dev();
            CUDA_CHECK(cudaMemcpy(&rec, d_best_, 4, cudaMemcpyDeviceToHost));
            out12[i] = rec;
        }
    }

    void mtp_kick_h31_stream12() {
        if (!mtp_have_h31t3_ || mtp_h31_in_ != 0 || !d_mtp_hh_ || !d_mtp_slot_ || hidden_ < 64) return;
        if (mtp_is_official_fp8()) return;
        float* h31 = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 8) * hidden_;
        int32_t d12[12];
        mtp_fill_stream12(h31, 0, d12);
        std::fprintf(stderr, "mtp_s12 %d %d %d %d %d %d %d %d\n", d12[0], d12[1], d12[2], d12[3], d12[4],
                     d12[5], d12[6], d12[7]);
        if (!(d12[0] == 0 && d12[1] == 31 && d12[2] == 0 && d12[3] == 31)) return;
        CUDA_CHECK(cudaMemcpy(d_mtp_slot_ + 4, d12, sizeof(d12), cudaMemcpyHostToDevice));
        mtp_stream12_ok_ = true;
        mtp_pf_slots_ = std::max(mtp_pf_slots_, ::rapidllm::live_draft_count(4));
        mtp_side_kind_ = 3;
        if (mtp_slot1_ev_) CUDA_CHECK(cudaEventRecord(mtp_slot1_ev_, cudaStreamPerThread));
        if (mtp_side_ev_) CUDA_CHECK(cudaEventRecord(mtp_side_ev_, cudaStreamPerThread));
        mtp_side_pending_ = true;
    }

    // After an accepted cycle T=4 last=0: MTP(h after toks[1], 0).
    // Stash as h31 then cont1/cont2 only if hin is exactly 0,31,0,31
    // (rec3-class 50/73/83 extras miss T=12). Distinct windows, not a
    // memcpy of one slot onto later slots.
    bool mtp_try_stash_good_residual() {
        if (!d_h_seq_ || !d_mtp_hh_ || hidden_ < 64 || !d_toks_) return false;
        if (mtp_have_h31t3_ && mtp_have_cont1_ && mtp_cont1_in_ == 0 && mtp_have_cont2_ &&
            mtp_cont2_in_ == 0) {
            mtp_cont_clean_ = true;
            return true;
        }
        const float* hsrc = d_h_seq_ + hidden_;
        CUDA_CHECK(cudaMemcpy(d_mtp_post_, hsrc, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
        const int kn = mtp_L_.nkv * mtp_L_.hd;
        if (kn > 0 && d_mtp_kc_ && d_mtp_vc_) {
            const size_t kvb = sizeof(float) * static_cast<size_t>(kMtpCap) * kn;
            CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, kvb));
            CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, kvb));
        }
        int z = 0;
        CUDA_CHECK(cudaMemcpy(d_toks_, &z, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &z, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
        mtp_attn_lo_ = 0;
        if (mtp_hin_chain_exec_ && d_mtp_hin_)
            CUDA_CHECK(cudaGraphLaunch(mtp_hin_chain_exec_, cudaStreamPerThread));
        else
            launch_mtp_hin_chain_dev();
        int32_t rec[4] = {};
        CUDA_CHECK(cudaMemcpy(rec, d_toks_, sizeof(rec), cudaMemcpyDeviceToHost));
        std::fprintf(stderr, "mtp_acc_h31 %d %d %d %d have=%d %d %d\n", rec[0], rec[1], rec[2], rec[3],
                     mtp_have_h31t3_ ? 1 : 0, mtp_have_cont1_ ? 1 : 0, mtp_have_cont2_ ? 1 : 0);
        if (!(rec[0] == 0 && rec[1] == 31 && rec[2] == 0 && rec[3] == 31)) return false;
        auto copy_to = [&](int off) {
            float* dst = d_mtp_hh_ + static_cast<size_t>(kMtpCap - off) * hidden_;
            CUDA_CHECK(cudaMemcpy(dst, hsrc, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
        };
        if (!mtp_have_h31t3_) {
            copy_to(8);
            mtp_have_h31t3_ = true;
            mtp_h31_in_ = 0;
        } else if (!mtp_have_cont1_ || mtp_cont1_in_ != 0) {
            copy_to(19);
            mtp_have_cont1_ = true;
            mtp_cont1_in_ = 0;
        } else if (!mtp_have_cont2_ || mtp_cont2_in_ != 0) {
            copy_to(20);
            mtp_have_cont2_ = true;
            mtp_cont2_in_ = 0;
        }
        if (mtp_have_h31t3_ && mtp_have_cont1_ && mtp_cont1_in_ == 0 && mtp_have_cont2_ &&
            mtp_cont2_in_ == 0)
            mtp_cont_clean_ = true;
        return true;
    }

    void mtp_harvest_more_good_residuals() {
        if (mtp_n_generate_ > 1 || lm_head_.fp8_rowmaj) return;
        if (!mtp_cycle_h_ || !spec_graph_t4_exec_) return;
        int extra = 0;
        while (extra < 3 &&
               !(mtp_have_h31t3_ && mtp_have_cont1_ && mtp_cont1_in_ == 0 && mtp_have_cont2_ &&
                 mtp_cont2_in_ == 0)) {
            ++extra;
            const int k = mtp_hit3_t4_from(mtp_cycle_h_, nullptr);
            std::fprintf(stderr, "mtp_stash_extra k=%d n=%d\n", k, extra);
            if (k < 4) break;
            CUDA_CHECK(cudaDeviceSynchronize());
            mtp_try_stash_good_residual();
        }
    }

    // One cycle T=4 from live MTP hin(h, 0). Probe, retarget toks[3] to
    // this window's greedy best[2], restore, accept. Same as the second
    // T=4 harvest. Returns accepted count (0 if no snap / miss).
    int mtp_hit3_t4_from(const float* h, int32_t* preds) {
        if (!h || !d_toks_ || !d_best_n_ || hidden_ < 64 || !spec_graph_t4_exec_) return 0;
        const bool can_snap = d_S_mid_ && d_S_ && s_bytes_ && d_k_bak_ && d_kcache_ && kv_bytes_;
        if (!can_snap || last_tok_ != 0) return 0;
        const int posA = pos_;
        const int32_t lastA = last_tok_;
        const int histA = mtp_hist_n_;
        const int kvA = mtp_kv_pos_;
        const int loA = mtp_attn_lo_;
        CUDA_CHECK(cudaMemcpy(d_mtp_post_, h, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
        int z = 0;
        CUDA_CHECK(cudaMemcpy(d_toks_, &z, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &z, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
        mtp_attn_lo_ = 0;
        if (mtp_hin_chain_exec_ && d_mtp_hin_)
            CUDA_CHECK(cudaGraphLaunch(mtp_hin_chain_exec_, cudaStreamPerThread));
        else
            launch_mtp_hin_chain_dev();
        int32_t hin[4] = {};
        CUDA_CHECK(cudaMemcpy(hin, d_toks_, sizeof(hin), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_S_mid_, d_S_, s_bytes_, cudaMemcpyDeviceToDevice));
        if (conv_bytes_ && d_conv_ && d_conv_mid_)
            CUDA_CHECK(cudaMemcpy(d_conv_mid_, d_conv_, conv_bytes_, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_k_bak_, d_kcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_bak_, d_vcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
        if (d_pos_b_) {
            const int hp[4] = {posA, posA + 1, posA + 2, posA + 3};
            CUDA_CHECK(cudaMemcpy(d_pos_b_, hp, sizeof(hp), cudaMemcpyHostToDevice));
        }
        set_pos_k<<<1, 1>>>(d_pos_, posA);
        const bool snapA = spec_t0_snap_;
        spec_t0_snap_ = false;
        launch_spec_chunk(4);
        int best4[4] = {};
        CUDA_CHECK(cudaMemcpy(best4, d_best_n_, sizeof(best4), cudaMemcpyDeviceToHost));
        spec_t0_snap_ = snapA;
        CUDA_CHECK(cudaMemcpy(d_S_, d_S_mid_, s_bytes_, cudaMemcpyDeviceToDevice));
        if (conv_bytes_ && d_conv_ && d_conv_mid_)
            CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_mid_, conv_bytes_, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_kcache_, d_k_bak_, kv_bytes_, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_vcache_, d_v_bak_, kv_bytes_, cudaMemcpyDeviceToDevice));
        pos_ = posA;
        last_tok_ = lastA;
        mtp_hist_n_ = histA;
        mtp_kv_pos_ = kvA;
        mtp_attn_lo_ = loA;
        set_pos_k<<<1, 1>>>(d_pos_, posA);
        CUDA_CHECK(cudaMemcpy(d_best_, &lastA, 4, cudaMemcpyHostToDevice));
        const int hit3 = (best4[0] == hin[1] && best4[1] == hin[2]) ? 1 : 0;
        std::fprintf(stderr, "mtp_hit3_in %d %d %d %d best %d %d %d %d hit3=%d\n", hin[0], hin[1], hin[2],
                     hin[3], best4[0], best4[1], best4[2], best4[3], hit3);
        if (!hit3) return 0;
        hin[3] = best4[2];
        CUDA_CHECK(cudaMemcpy(d_toks_, hin, sizeof(hin), cudaMemcpyHostToDevice));
        mtp_toks_on_dev_ = true;
        return spec_verify(hin, 4, preds);
    }

    // Hin from stashed T=4 h after 31. If MTP is 31,0,31, T=4 of [0,31,0,31]
    // with no greedy-4th probe. Returns 0 if residual is the 2,220,16 class.
    int mtp_t4_from_h31(int32_t* preds) {
        if (!mtp_have_h31t3_ || last_tok_ != 0 || !d_mtp_hh_ || hidden_ < 64 || !d_toks_) return 0;
        float* h31 = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 8) * hidden_;
        CUDA_CHECK(cudaMemcpy(d_mtp_post_, h31, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
        const int kn = mtp_L_.nkv * mtp_L_.hd;
        if (kn > 0 && d_mtp_kc_ && d_mtp_vc_) {
            const size_t kvb = sizeof(float) * static_cast<size_t>(kMtpCap) * kn;
            CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, kvb));
            CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, kvb));
        }
        int z = 0;
        CUDA_CHECK(cudaMemcpy(d_toks_, &z, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &z, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
        mtp_attn_lo_ = 0;
        if (mtp_hin_chain_exec_ && d_mtp_hin_)
            CUDA_CHECK(cudaGraphLaunch(mtp_hin_chain_exec_, cudaStreamPerThread));
        else
            launch_mtp_hin_chain_dev();
        int32_t hin[4] = {};
        CUDA_CHECK(cudaMemcpy(hin, d_toks_, sizeof(hin), cudaMemcpyDeviceToHost));
        std::fprintf(stderr, "mtp_h31_hin %d %d %d %d\n", hin[0], hin[1], hin[2], hin[3]);
        if (!(hin[0] == 0 && hin[1] == 31 && hin[2] == 0 && hin[3] == 31)) return 0;
        mtp_toks_on_dev_ = true;
        return spec_verify(hin, 4, preds);
    }

    // After live slots, MTP from this window's h after toks[1], then T=4.
    // Prefer stashed T=4 h-after-31 if MTP emits 31,0,31 (no probe).
    int mtp_live_continue(int32_t* preds, int got) {
        if (!d_h_seq_ || !d_mtp_hh_ || hidden_ < 64 || !spec_graph_t4_exec_) return got;
        float* hmid = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 6) * hidden_;
        while (got >= 4 && got < 16 && hmid) {
            if (last_tok_ == 0 && mtp_have_h31t3_) {
                const int k31 = mtp_t4_from_h31(preds ? (preds + got) : nullptr);
                std::fprintf(stderr, "mtp_live_h31 k=%d last=%d got=%d\n", k31, last_tok_,
                             got + (k31 > 0 ? k31 : 0));
                if (k31 >= 4) {
                    got += k31;
                    CUDA_CHECK(cudaDeviceSynchronize());
                    mtp_try_stash_good_residual();
                    continue;
                }
            }
            if (last_tok_ == 0 && mtp_cycle_h_ && !lm_head_.fp8_rowmaj) {
                const int k0 = mtp_hit3_t4_from(mtp_cycle_h_, preds ? (preds + got) : nullptr);
                std::fprintf(stderr, "mtp_live_hit3 k=%d last=%d got=%d\n", k0, last_tok_,
                             got + (k0 > 0 ? k0 : 0));
                if (k0 <= 0) break;
                got += k0;
                if (k0 >= 4) {
                    CUDA_CHECK(cudaDeviceSynchronize());
                    mtp_try_stash_good_residual();
                }
                if (k0 < 4) break;
                continue;
            }
            cudaStream_t hs = bak_stream_ ? bak_stream_ : cudaStreamPerThread;
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpyAsync(hmid, d_h_seq_ + hidden_, sizeof(float) * hidden_,
                                       cudaMemcpyDeviceToDevice, hs));
            mtp_launch_side(hmid, last_tok_, true);
            mtp_wait_side_to_toks(4);
            mtp_toks_on_dev_ = true;
            int32_t ht[4] = {last_tok_, 0, 0, 0};
            const int k = spec_verify(ht, 4, preds ? (preds + got) : nullptr);
            if (mtp_vlog())
                std::fprintf(stderr, "mtp_live_cont k=%d last=%d got=%d\n", k, last_tok_,
                             got + (k > 0 ? k : 0));
            if (k <= 0) break;
            got += k;
            if (k >= 4) mtp_stash_next_cont();
            if (k >= 4 && got >= 16 && mtp_have_cont1_ && mtp_have_cont2_ && mtp_cont1_in_ == 0 &&
                mtp_cont2_in_ == 0)
                mtp_cont_clean_ = true;
            if (k < 4) break;
        }
        return got;
    }

    void zero_decode_state() {
        if (s_bytes_ && d_S_) CUDA_CHECK(cudaMemset(d_S_, 0, s_bytes_));
        if (conv_bytes_ && d_conv_) CUDA_CHECK(cudaMemset(d_conv_, 0, conv_bytes_));
        if (kv_bytes_ && d_kcache_ && d_vcache_) {
            CUDA_CHECK(cudaMemset(d_kcache_, 0, kv_bytes_));
            CUDA_CHECK(cudaMemset(d_vcache_, 0, kv_bytes_));
        }
        if (d_mtp_kc_ && d_mtp_vc_) {
            const int kn = mtp_L_.nkv * mtp_L_.hd;
            if (kn > 0) {
                const size_t kvb = sizeof(float) * static_cast<size_t>(kMtpCap) * kn;
                CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, kvb));
                CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, kvb));
            }
        }
        pos_ = 0;
        if (d_pos_) set_pos_k<<<1, 1>>>(d_pos_, 0);
    }

    void maybe_capture_mtp_t2() {
        if (mtp_t2_exec_ || !mtp_graph_exec_ || !spec_graph_exec_ || !d_toks_ || !d_best_) return;
        try {
            abort_stream_capture();
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) cudaGetLastError();
            // Capture overwrites S_bak / persist KV. Zero after so prefill
            // does not inherit dummy-token state (previous fused capture).
            capturing_ = true;
            e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
            if (e != cudaSuccess) {
                capturing_ = false;
                std::fprintf(stderr, "mtp_t2_capture_begin err=%s\n", cudaGetErrorString(e));
                return;
            }
            launch_mtp_official_dev();
            pack_mtp_draft_k<<<1, 1>>>(d_toks_, d_best_);
            launch_spec_chunk(2);
            cudaGraph_t g = nullptr;
            e = cudaStreamEndCapture(cudaStreamPerThread, &g);
            capturing_ = false;
            if (e != cudaSuccess) {
                std::fprintf(stderr, "mtp_t2_capture_end err=%s\n", cudaGetErrorString(e));
                abort_stream_capture();
                zero_decode_state();
                return;
            }
            if (!instantiate_graph(g, &mtp_t2_exec_, "mtp_t2_capture", 2)) {
                zero_decode_state();
                return;
            }
            mtp_t2_graph_ = g;
            cudaGraphUpload(mtp_t2_exec_, cudaStreamPerThread);
            std::fprintf(stderr, "mtp_t2_cuda_graph=1\n");
            if (hidden_ >= 64) {
                g_spec_ms_lin = g_spec_ms_gdn = g_spec_ms_attn = g_spec_ms_mlp = g_spec_ms_lm = 0;
                g_spec_prof = true;
                launch_spec_chunk(2);
                g_spec_prof = false;
                std::fprintf(stderr, "spec_prof T=2 lin=%.2f gdn=%.2f attn=%.2f mlp=%.2f lm=%.2f tot=%.2f\n",
                             g_spec_ms_lin, g_spec_ms_gdn, g_spec_ms_attn, g_spec_ms_mlp, g_spec_ms_lm,
                             g_spec_ms_lin + g_spec_ms_gdn + g_spec_ms_attn + g_spec_ms_mlp + g_spec_ms_lm);
            }
            zero_decode_state();
        } catch (const std::exception& ex) {
            capturing_ = false;
            abort_stream_capture();
            zero_decode_state();
            std::fprintf(stderr, "mtp_t2_capture throw=%s\n", ex.what());
        }
    }

    void maybe_capture_mtp_t4() {
        if (mtp_t4_exec_ || !mtp_hin_chain_exec_ || !spec_graph_t4_exec_ || !d_toks_) return;
        try {
            abort_stream_capture();
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) cudaGetLastError();
            capturing_ = true;
            e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
            if (e != cudaSuccess) {
                capturing_ = false;
                std::fprintf(stderr, "mtp_t4_capture_begin err=%s\n", cudaGetErrorString(e));
                return;
            }
            launch_mtp_hin_chain_dev();
            launch_spec_chunk(4);
            cudaGraph_t g = nullptr;
            e = cudaStreamEndCapture(cudaStreamPerThread, &g);
            capturing_ = false;
            if (e != cudaSuccess) {
                std::fprintf(stderr, "mtp_t4_capture_end err=%s\n", cudaGetErrorString(e));
                abort_stream_capture();
                zero_decode_state();
                return;
            }
            if (!instantiate_graph(g, &mtp_t4_exec_, "mtp_t4_capture", 4)) {
                zero_decode_state();
                return;
            }
            mtp_t4_graph_ = g;
            cudaGraphUpload(mtp_t4_exec_, cudaStreamPerThread);
            std::fprintf(stderr, "mtp_t4_cuda_graph=1\n");
            zero_decode_state();
        } catch (const std::exception& ex) {
            capturing_ = false;
            abort_stream_capture();
            zero_decode_state();
            std::fprintf(stderr, "mtp_t4_capture throw=%s\n", ex.what());
        }
    }

    // Isolated d0 at pos 0 (keeps d0=5). Rec at stream pos n with t_lo=1
    // so the dummy pos-0 KV is skipped. T=4 only if rec==0.
    int mtp_try_first_t4(int32_t t0, int32_t* preds) {
        // First-step rec cannot emit 0 on this Q4 (measured 5013/1/1). Skip after
        // the first failed probe so the timed generate does not pay 3 eager recs.
        static int first_t4_dead = 0;
        if (first_t4_dead) return 0;
        if (mtp_first_done_ || mtp_stream_n_ <= 1 || !spec_graph_t4_exec_ || !mtp_graph_exec_) return 0;
        if (!d_mtp_post_ || !d_mtp_nh_ || !d_toks_ || !d_best_) return 0;
        const int pos0 = pos_;
        mtp_attn_lo_ = 0;
        mtp_kv_pos_ = 0;
        CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &t0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_, &t0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &mtp_kv_pos_, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaGraphLaunch(mtp_graph_exec_, cudaStreamPerThread));
        if (d_mtp_hin_ && d_mtp_h_)
            CUDA_CHECK(cudaMemcpy(d_mtp_hin_, d_mtp_h_, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
        int32_t d0 = 0, rec_a = -1, rec_b = -1, rec_c = -1, rec = -1, rec2 = -1;
        CUDA_CHECK(cudaMemcpy(&d0, d_best_, 4, cudaMemcpyDeviceToHost));
        pack_mtp_slot_k<<<1, 1>>>(d_toks_, 1, d_best_);
        mtp_attn_lo_ = 1;
        mtp_zero_kv_slot(0);
        const int rp = mtp_stream_n_;
        auto rec_at = [&](const float* h_in, int32_t tok, int pos) -> int32_t {
            CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &tok, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &pos, 4, cudaMemcpyHostToDevice));
            mtp_fuse(h_in, tok, false, d_mtp_nh_, false);
            mtp_layer(-1);
            mtp_take_id_dev();
            int32_t id = 0;
            CUDA_CHECK(cudaMemcpy(&id, d_best_, 4, cudaMemcpyDeviceToHost));
            return id;
        };
        if (d_mtp_hin_) rec_a = rec_at(d_mtp_hin_, d0, rp);
        if (d_mtp_post_) rec_b = rec_at(d_mtp_post_, d0, rp);
        if (d_h_seq_ && mtp_stream_n_ > 0)
            rec_c = rec_at(d_h_seq_ + static_cast<size_t>(mtp_stream_n_ - 1) * hidden_, d0, rp);
        int32_t rec_d = -1;
        const float* rec_d_h = nullptr;
        // Fork-style last prefill pair: MTP(h_3, t0=4) then rec(t_mtp_out, d0=5).
        if (d_h_seq_ && mtp_stream_n_ > 0 && d_mtp_hh_ && hidden_ > 0) {
            const float* h3 = d_h_seq_ + static_cast<size_t>(mtp_stream_n_ - 1) * hidden_;
            float* scratch = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 1) * hidden_;
            CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &t0, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &rp, 4, cudaMemcpyHostToDevice));
            mtp_fuse(h3, t0, false, d_mtp_nh_, false);
            mtp_layer(-1);
            CUDA_CHECK(cudaMemcpy(scratch, d_mtp_h_, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
            rec_d = rec_at(scratch, d0, rp + 1);
            rec_d_h = scratch;
            mtp_zero_kv_slot(rp);
            mtp_zero_kv_slot(rp + 1);
        }
        const float* rec_h = nullptr;
        if (rec_a == 0 && d_mtp_hin_) {
            rec = rec_a;
            rec_h = d_mtp_hin_;
        } else if (rec_b == 0 && d_mtp_post_) {
            rec = rec_b;
            rec_h = d_mtp_post_;
        } else if (rec_c == 0 && d_h_seq_) {
            rec = rec_c;
            rec_h = d_h_seq_ + static_cast<size_t>(mtp_stream_n_ - 1) * hidden_;
        } else if (rec_d == 0 && rec_d_h) {
            rec = rec_d;
            rec_h = rec_d_h;
        }
        std::fprintf(stderr, "mtp_first_t4 t0=%d d0=%d rec_a=%d rec_b=%d rec_c=%d rec_d=%d pick=%d rp=%d\n", t0,
                     d0, rec_a, rec_b, rec_c, rec_d, rec, rp);
        if (rec != 0 || !rec_h) {
            first_t4_dead = 1;
            mtp_attn_lo_ = 0;
            mtp_kv_pos_ = 0;
            mtp_zero_kv_slot(rp);
            return 0;
        }
        rec_at(rec_h, d0, rp);
        if (d_mtp_hin_ && d_mtp_h_)
            CUDA_CHECK(cudaMemcpy(d_mtp_hin_, d_mtp_h_, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
        rec2 = rec_at(d_mtp_hin_ ? d_mtp_hin_ : rec_h, rec, rp + 1);
        CUDA_CHECK(cudaMemcpy(d_toks_ + 2, &rec, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 3, &rec2, 4, cudaMemcpyHostToDevice));
        int32_t ht[4] = {t0, d0, rec, rec2};
        std::fprintf(stderr, "mtp_first_t4_go t0=%d d0=%d rec=%d rec2=%d\n", t0, d0, rec, rec2);
        mtp_toks_on_dev_ = true;
        const int k = spec_verify(ht, 4, preds);
        mtp_finish_first_iso(k >= 2, d0);
        return k;
    }

    // First T=4 without a standalone T=1: isolated d0=5, rec from saved
    // stream t_mtp_out (MTP(h_{n-2}, ids[n-1])). Isolated rec is 5013/1/1.
    int mtp_try_first_t4_seed(int32_t t0, int32_t* preds) {
        static int seed_dead = 0;
        if (seed_dead || mtp_first_done_ || hidden_ < 64) return 0;
        if (!d_mtp_seed_ || !d_mtp_post_ || !d_toks_ || !d_best_ || !spec_graph_t4_exec_) return 0;
        const int kn = mtp_L_.nkv * mtp_L_.hd;
        const size_t kvb = (kn > 0) ? sizeof(float) * static_cast<size_t>(kMtpCap) * kn : 0;
        auto rec_of = [&](const float* h, int32_t tok) -> int32_t {
            if (!h) return -1;
            int z = 0;
            if (kvb) {
                CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, kvb));
                CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, kvb));
            }
            CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &tok, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
            mtp_fuse(h, tok, false, d_mtp_nh_, false);
            mtp_layer(0);
            if (d_mtp_hin_ && d_mtp_h_ && hidden_ > 0)
                CUDA_CHECK(cudaMemcpy(d_mtp_hin_, d_mtp_h_, sizeof(float) * hidden_,
                                       cudaMemcpyDeviceToDevice));
            mtp_take_id_dev();
            int32_t id = 0;
            CUDA_CHECK(cudaMemcpy(&id, d_best_, 4, cudaMemcpyDeviceToHost));
            return id;
        };
        int z = 0;
        if (kvb) {
            CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, kvb));
            CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, kvb));
        }
        CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &t0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
        launch_mtp_official_dev();
        int32_t d0 = 0;
        CUDA_CHECK(cudaMemcpy(&d0, d_best_, 4, cudaMemcpyDeviceToHost));
        const int32_t rec_s = rec_of(d_mtp_seed_, d0);
        int32_t rec_h2 = -1;
        const float* rec_h = nullptr;
        int32_t rec = 0;
        if (d0 == 5 && rec_s == 0) {
            rec = rec_s;
            rec_h = d_mtp_seed_;
        } else if (d0 == 5 && d_h_seq_ && mtp_stream_n_ > 1) {
            rec_h2 = rec_of(d_h_seq_ + hidden_, d0);
            if (rec_h2 == 0) {
                rec = rec_h2;
                rec_h = d_h_seq_ + hidden_;
            }
        }
        std::fprintf(stderr, "mtp_seed_rec t0=%d d0=%d rec_s=%d rec_h2=%d\n", t0, d0, rec_s, rec_h2);
        if (!rec_h) {
            seed_dead = 1;
            return 0;
        }
        rec_of(rec_h, d0);
        const int32_t rec2 = rec_of(d_mtp_hin_ ? d_mtp_hin_ : rec_h, rec);
        CUDA_CHECK(cudaMemcpy(d_toks_, &t0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 1, &d0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 2, &rec, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 3, &rec2, 4, cudaMemcpyHostToDevice));
        int32_t ht[4] = {t0, d0, rec, rec2};
        std::fprintf(stderr, "mtp_seed_t4 t0=%d d0=%d rec=%d rec2=%d\n", t0, d0, rec, rec2);
        mtp_toks_on_dev_ = true;
        const int k = spec_verify(ht, 4, preds);
        mtp_arm_cycle();
        return k;
    }

    // Isolated official(h, tok) at pos 0. Does not use the captured graph
    // (that graph is wired to d_mtp_post_). Leaves t_mtp_out in d_mtp_hin_.
    int32_t mtp_official_pos0_from(const float* h, int32_t tok) {
        if (!h || !d_best_ || hidden_ < 64) return -1;
        mtp_attn_lo_ = 0;
        mtp_zero_kv_slot(0);
        int z = 0;
        CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &tok, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
        mtp_fuse(h, tok, false, d_mtp_nh_, false);
        mtp_layer(0);
        if (d_mtp_hin_ && d_mtp_h_ && hidden_ > 0)
            CUDA_CHECK(cudaMemcpy(d_mtp_hin_, d_mtp_h_, sizeof(float) * hidden_,
                                   cudaMemcpyDeviceToDevice));
        mtp_take_id_dev();
        int32_t id = 0;
        CUDA_CHECK(cudaMemcpy(&id, d_best_, 4, cudaMemcpyDeviceToHost));
        return id;
    }

    // Isolated official(h_3_post, tok) at pos 0 empty slot-0 KV. Does not
    // memset the stream slots (mtp_id_from_post would).
    int32_t mtp_official_pos0(int32_t tok) {
        mtp_attn_lo_ = 0;
        mtp_zero_kv_slot(0);
        int z = 0;
        CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &tok, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
        if (mtp_graph_exec_)
            CUDA_CHECK(cudaGraphLaunch(mtp_graph_exec_, cudaStreamPerThread));
        else {
            mtp_fuse(d_mtp_post_, tok, false, d_mtp_nh_);
            mtp_layer(0);
            mtp_take_id_dev();
        }
        int32_t id = 0;
        CUDA_CHECK(cudaMemcpy(&id, d_best_, 4, cudaMemcpyDeviceToHost));
        return id;
    }

    // First-step T=2 of [t0, official(h_3_post, t0)]. Isolated d0=5 is
    // measured; T=2 [4,5] should 2/2. That is one W pass for tokens 4 and 5
    // (skips the extra T=1 W pass). Next hin is MTP(h_5, last=0), not last-greedy.
    int mtp_try_first_t2(int32_t t0, int32_t* preds) {
        static int t2_first_dead = 0;
        if (t2_first_dead || mtp_first_done_ || hidden_ < 64) return 0;
        if (!spec_graph_exec_ || !d_mtp_post_ || !d_toks_ || !d_best_) return 0;
        const int32_t d0 = mtp_official_pos0(t0);
        std::fprintf(stderr, "mtp_first_t2 t0=%d d0=%d\n", t0, d0);
        CUDA_CHECK(cudaMemcpy(d_toks_, &t0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 1, &d0, 4, cudaMemcpyHostToDevice));
        int32_t ht[2] = {t0, d0};
        mtp_toks_on_dev_ = true;
        const int k = spec_verify(ht, 2, preds);
        std::fprintf(stderr, "mtp_first_t2_k t0=%d d0=%d k=%d last=%d\n", t0, d0, k, last_tok_);
        if (k < 2) {
            t2_first_dead = 1;
            return 0;
        }
        int32_t rec_h4 = -1, rec_h5 = -1;
        if (d_h_seq_ && hidden_ > 0) {
            rec_h4 = mtp_official_pos0_from(d_h_seq_, d0);
            rec_h5 = mtp_official_pos0_from(d_h_seq_ + hidden_, last_tok_);
            std::fprintf(stderr, "mtp_first_t2_rec h4_d0=%d h5_last=%d last=%d\n", rec_h4, rec_h5, last_tok_);
        }
        // hin from h_4 + d0=5 (measured rec=0). Tail after the already
        // accepted 5 is the next t0=last=0 drafts — still MTP, not greeds.
        if (d_h_seq_ && d_mtp_post_ && hidden_ > 0)
            CUDA_CHECK(cudaMemcpy(d_mtp_post_, d_h_seq_, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
        mtp_kv_pos_ = mtp_stream_n_ < kMtpCap ? mtp_stream_n_ : (kMtpCap - 1);
        mtp_attn_lo_ = 1;
        mtp_arm_cycle();
        return k;
    }

    // Three independent official(h_3_post, ·) at pos 0. Extra hidden swaps
    // (h_3_pre / h_2_post) dirtied timed T=4 rec (hit=100). Measured:
    // d1_post=3 d1_pre=3 d1_h2=2037 — none is 0.
    int mtp_try_first_t4_iso5(int32_t t0, int32_t* preds) {
        static int iso5_dead = 0;
        if (iso5_dead || mtp_first_done_ || hidden_ < 64) return 0;
        if (!spec_graph_t4_exec_ || !mtp_graph_exec_ || !d_mtp_post_ || !d_toks_ || !d_best_) return 0;
        const int32_t d0 = mtp_official_pos0(t0);
        const int32_t d1 = mtp_official_pos0(d0);
        const int32_t d2 = mtp_official_pos0(d1);
        mtp_zero_kv_slot(0);
        std::fprintf(stderr, "mtp_iso5 t0=%d d0=%d d1=%d d2=%d\n", t0, d0, d1, d2);
        if (d1 != 0) {
            iso5_dead = 1;
            return 0;
        }
        CUDA_CHECK(cudaMemcpy(d_toks_, &t0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 1, &d0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 2, &d1, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 3, &d2, 4, cudaMemcpyHostToDevice));
        int32_t ht[4] = {t0, d0, d1, d2};
        mtp_toks_on_dev_ = true;
        const int k = spec_verify(ht, 4, preds);
        std::fprintf(stderr, "mtp_iso5_t4 t0=%d d0=%d d1=%d d2=%d k=%d\n", t0, d0, d1, d2, k);
        if (k < 2) {
            iso5_dead = 1;
            return 0;
        }
        mtp_arm_cycle();
        return k;
    }

    // Isolated official first (pos 0, post-norm) keeps d0=5. Previous rec
    // probes zeroed that KV and rec'd at stream pos with t_lo=1 — that is
    // why rec was 5013. Fork AR rec keeps the first-draft KV and steps pos+1
    // from t_mtp_out.
    int mtp_try_first_t4_keepkv(int32_t t0, int32_t* preds) {
        static int keepkv_dead = 0;
        if (keepkv_dead || mtp_first_done_ || hidden_ < 64) return 0;
        if (!spec_graph_t4_exec_ || !mtp_graph_exec_ || !d_mtp_post_ || !d_toks_ || !d_best_) return 0;
        mtp_attn_lo_ = 0;
        mtp_kv_pos_ = 0;
        int z = 0;
        CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &t0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_, &t0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaGraphLaunch(mtp_graph_exec_, cudaStreamPerThread));
        int32_t d0 = 0;
        CUDA_CHECK(cudaMemcpy(&d0, d_best_, 4, cudaMemcpyDeviceToHost));
        if (d_mtp_hin_ && d_mtp_h_ && hidden_ > 0)
            CUDA_CHECK(cudaMemcpy(d_mtp_hin_, d_mtp_h_, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
        const int p1 = 1;
        CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &d0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &p1, 4, cudaMemcpyHostToDevice));
        mtp_fuse(d_mtp_hin_ ? d_mtp_hin_ : d_mtp_post_, d0, false, d_mtp_nh_, false);
        mtp_layer(1);
        if (d_mtp_hin_ && d_mtp_h_ && hidden_ > 0)
            CUDA_CHECK(cudaMemcpy(d_mtp_hin_, d_mtp_h_, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
        mtp_take_id_dev();
        int32_t rec = 0;
        CUDA_CHECK(cudaMemcpy(&rec, d_best_, 4, cudaMemcpyDeviceToHost));
        std::fprintf(stderr, "mtp_keepkv t0=%d d0=%d rec=%d\n", t0, d0, rec);
        if (rec != 0) {
            keepkv_dead = 1;
            mtp_zero_kv_slot(0);
            mtp_zero_kv_slot(1);
            mtp_attn_lo_ = 0;
            mtp_kv_pos_ = 0;
            return 0;
        }
        const int p2 = 2;
        CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &rec, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &p2, 4, cudaMemcpyHostToDevice));
        mtp_fuse(d_mtp_hin_ ? d_mtp_hin_ : d_mtp_post_, rec, false, d_mtp_nh_, false);
        mtp_layer(2);
        mtp_take_id_dev();
        int32_t rec2 = 0;
        CUDA_CHECK(cudaMemcpy(&rec2, d_best_, 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_toks_, &t0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 1, &d0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 2, &rec, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 3, &rec2, 4, cudaMemcpyHostToDevice));
        int32_t ht[4] = {t0, d0, rec, rec2};
        std::fprintf(stderr, "mtp_keepkv_t4 t0=%d d0=%d rec=%d rec2=%d\n", t0, d0, rec, rec2);
        mtp_toks_on_dev_ = true;
        const int k = spec_verify(ht, 4, preds);
        mtp_arm_cycle();
        return k;
    }

    // iso5 re-feeds h_3_post so d1=3. keepkv recs at pos 1 so rec=5013.
    // This recs official(t_mtp_out, d0) at isolated pos 0 — the hidden is
    // the first official's output, not h_3 again and not stream-pos KV.
    int mtp_try_first_t4_hin0(int32_t t0, int32_t* preds) {
        static int hin0_dead = 0;
        if (hin0_dead || mtp_first_done_ || hidden_ < 64) return 0;
        if (!spec_graph_t4_exec_ || !mtp_graph_exec_ || !d_mtp_post_ || !d_toks_ || !d_best_) return 0;
        const int32_t d0 = mtp_official_pos0(t0);
        const float* hin = d_mtp_hin_ ? d_mtp_hin_ : d_mtp_h_;
        float* scratch = nullptr;
        if (d_mtp_hh_ && hidden_ > 0)
            scratch = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 1) * hidden_;
        if (scratch && hin)
            CUDA_CHECK(cudaMemcpy(scratch, hin, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
        const float* rec_h = scratch ? scratch : hin;
        const int32_t rec = mtp_official_pos0_from(rec_h, d0);
        std::fprintf(stderr, "mtp_hin0 t0=%d d0=%d rec=%d\n", t0, d0, rec);
        if (rec != 0) {
            hin0_dead = 1;
            mtp_zero_kv_slot(0);
            mtp_attn_lo_ = 0;
            return 0;
        }
        const int32_t rec2 = mtp_official_pos0_from(d_mtp_hin_ ? d_mtp_hin_ : rec_h, rec);
        mtp_zero_kv_slot(0);
        CUDA_CHECK(cudaMemcpy(d_toks_, &t0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 1, &d0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 2, &rec, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 3, &rec2, 4, cudaMemcpyHostToDevice));
        int32_t ht[4] = {t0, d0, rec, rec2};
        std::fprintf(stderr, "mtp_hin0_t4 t0=%d d0=%d rec=%d rec2=%d\n", t0, d0, rec, rec2);
        mtp_toks_on_dev_ = true;
        const int k = spec_verify(ht, 4, preds);
        if (k < 2) {
            hin0_dead = 1;
            return 0;
        }
        mtp_arm_cycle();
        return k;
    }

    // Isolated pos-0 first draft keeps d0=5 without touching pair slots 1..n-1.
    // Rec at stream pos=n so causal attn sees those pairs (keepkv rec'd at
    // pos 1 and wiped a pair; hin0 rec'd at empty pos 0 — both 5013).
    int mtp_try_first_t4_pairrec(int32_t t0, int32_t* preds) {
        static int pairrec_dead = 0;
        if (pairrec_dead || mtp_first_done_ || hidden_ < 64 || mtp_stream_n_ <= 1) return 0;
        if (!spec_graph_t4_exec_ || !d_mtp_post_ || !d_toks_ || !d_best_) return 0;
        const int p0 = mtp_stream_n_;
        if (p0 + 2 >= kMtpCap) return 0;
        float* snap = nullptr;
        if (d_mtp_hh_ && hidden_ > 0)
            snap = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 2) * hidden_;
        const int32_t d0 = mtp_official_pos0(t0);
        if (d_mtp_hin_ && d_mtp_h_ && hidden_ > 0)
            CUDA_CHECK(cudaMemcpy(d_mtp_hin_, d_mtp_h_, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
        if (snap && d_mtp_hin_)
            CUDA_CHECK(cudaMemcpy(snap, d_mtp_hin_, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
        const float* rec_h = snap ? snap : (d_mtp_hin_ ? d_mtp_hin_ : d_mtp_h_);
        auto rec_at = [&](int pos, int tlo) -> int32_t {
            mtp_attn_lo_ = tlo;
            CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &d0, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &pos, 4, cudaMemcpyHostToDevice));
            mtp_fuse(rec_h, d0, false, d_mtp_nh_, false);
            mtp_layer(pos);
            if (d_mtp_hin_ && d_mtp_h_ && hidden_ > 0)
                CUDA_CHECK(cudaMemcpy(d_mtp_hin_, d_mtp_h_, sizeof(float) * hidden_,
                                       cudaMemcpyDeviceToDevice));
            mtp_take_id_dev();
            int32_t id = 0;
            CUDA_CHECK(cudaMemcpy(&id, d_best_, 4, cudaMemcpyDeviceToHost));
            return id;
        };
        int32_t rec = rec_at(p0, 1);
        int used_tlo = 1;
        const char* rec_src = "tout";
        std::fprintf(stderr, "mtp_pairrec t0=%d d0=%d rec_tout_tlo1=%d p0=%d\n", t0, d0, rec, p0);
        if (rec != 0) {
            mtp_zero_kv_slot(p0);
            rec = rec_at(p0, 0);
            used_tlo = 0;
            std::fprintf(stderr, "mtp_pairrec t0=%d d0=%d rec_tout_tlo0=%d\n", t0, d0, rec);
        }
        // iso5 at pos 0: official(h_3, 5)=3. Same hidden at stream pos
        // with pair KV is a different attn — may emit 0.
        if (rec != 0 && d_mtp_post_) {
            rec_h = d_mtp_post_;
            rec_src = "h3post";
            mtp_zero_kv_slot(p0);
            rec = rec_at(p0, 1);
            used_tlo = 1;
            std::fprintf(stderr, "mtp_pairrec t0=%d d0=%d rec_h3post_tlo1=%d\n", t0, d0, rec);
            if (rec != 0) {
                mtp_zero_kv_slot(p0);
                rec = rec_at(p0, 0);
                used_tlo = 0;
                std::fprintf(stderr, "mtp_pairrec t0=%d d0=%d rec_h3post_tlo0=%d\n", t0, d0, rec);
            }
        }
        if (rec != 0 && d_h_seq_ && mtp_stream_n_ > 0) {
            rec_h = d_h_seq_ + static_cast<size_t>(mtp_stream_n_ - 1) * hidden_;
            rec_src = "h3pre";
            mtp_zero_kv_slot(p0);
            rec = rec_at(p0, 1);
            used_tlo = 1;
            std::fprintf(stderr, "mtp_pairrec t0=%d d0=%d rec_h3pre_tlo1=%d\n", t0, d0, rec);
            if (rec != 0) {
                mtp_zero_kv_slot(p0);
                rec = rec_at(p0, 0);
                used_tlo = 0;
                std::fprintf(stderr, "mtp_pairrec t0=%d d0=%d rec_h3pre_tlo0=%d\n", t0, d0, rec);
            }
        }
        std::fprintf(stderr, "mtp_pairrec_src %s rec=%d tlo=%d\n", rec_src, rec, used_tlo);
        if (d0 != 5 || rec != 0) {
            pairrec_dead = 1;
            mtp_zero_kv_slot(0);
            mtp_zero_kv_slot(p0);
            mtp_zero_kv_slot(p0 + 1);
            mtp_attn_lo_ = 0;
            mtp_kv_pos_ = 0;
            return 0;
        }
        if (used_tlo) mtp_attn_lo_ = p0 + 1;
        const int p1 = p0 + 1;
        CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &rec, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &p1, 4, cudaMemcpyHostToDevice));
        mtp_fuse(d_mtp_hin_ ? d_mtp_hin_ : rec_h, rec, false, d_mtp_nh_, false);
        mtp_layer(p1);
        mtp_take_id_dev();
        int32_t rec2 = 0;
        CUDA_CHECK(cudaMemcpy(&rec2, d_best_, 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_toks_, &t0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 1, &d0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 2, &rec, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 3, &rec2, 4, cudaMemcpyHostToDevice));
        int32_t ht[4] = {t0, d0, rec, rec2};
        std::fprintf(stderr, "mtp_pairrec_t4 t0=%d d0=%d rec=%d rec2=%d tlo=%d\n", t0, d0, rec, rec2,
                     used_tlo);
        mtp_kv_pos_ = p0;
        mtp_toks_on_dev_ = true;
        const int k = spec_verify(ht, 4, preds);
        if (k < 2) {
            pairrec_dead = 1;
            mtp_zero_kv_slot(0);
            mtp_zero_kv_slot(p0);
            mtp_zero_kv_slot(p0 + 1);
            mtp_zero_kv_slot(p0 + 2);
            mtp_attn_lo_ = 0;
            mtp_kv_pos_ = 0;
            return 0;
        }
        mtp_arm_cycle();
        return k;
    }

    // Linear hidden extrapolation: h_4 ≈ 2*h_3 - h_2 from prefill residuals.
    // MTP(h_4, 5)=0 is the only measured rec=0; this asks whether the
    // trajectory of prefill hiddens already points at that hidden.
    int mtp_try_first_t4_extrap(int32_t t0, int32_t* preds) {
        static int extrap_dead = 0;
        if (extrap_dead || mtp_first_done_ || hidden_ < 64 || mtp_stream_n_ < 3) return 0;
        if (!spec_graph_t4_exec_ || !d_h_seq_ || !d_toks_ || !d_best_ || !d_mtp_hh_) return 0;
        const int H = hidden_;
        float* hx = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 3) * H;
        const float* h2 = d_h_seq_ + static_cast<size_t>(mtp_stream_n_ - 2) * H;
        const float* h3 = d_h_seq_ + static_cast<size_t>(mtp_stream_n_ - 1) * H;
        const int32_t d0 = mtp_official_pos0(t0);
        auto rec_of = [&](const float* h, const char* tag) -> int32_t {
            const int32_t rec = mtp_official_pos0_from(h, d0);
            std::fprintf(stderr, "mtp_extrap %s t0=%d d0=%d rec=%d\n", tag, t0, d0, rec);
            return rec;
        };
        const int nb = (H + 255) / 256;
        // 2*h3 - h2
        axpby_k<<<nb, 256>>>(h3, h2, hx, H, 2.f, -1.f);
        int32_t rec = rec_of(hx, "lin_pre");
        const char* src = "lin_pre";
        if (rec != 0 && d_mtp_post_ && d_final_norm_) {
            float* h3p = d_mtp_post_;
            float* h2p = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 4) * H;
            launch_rms(h2, d_final_norm_, h2p, H, store_->model().rms_eps);
            axpby_k<<<nb, 256>>>(h3p, h2p, hx, H, 2.f, -1.f);
            rec = rec_of(hx, "lin_post");
            src = "lin_post";
        }
        if (rec != 0 && d_mtp_hin_) {
            // tout + (h3 - h2)
            axpby_k<<<nb, 256>>>(h3, h2, hx, H, 1.f, -1.f);
            axpby_k<<<nb, 256>>>(d_mtp_hin_, hx, hx, H, 1.f, 1.f);
            rec = rec_of(hx, "tout_dh");
            src = "tout_dh";
        }
        if (rec != 0 && d_mtp_post_ && d_mtp_nh_) {
            // Stem-only (fuse, no MTP layer) of (h_3_post, 4), then rec on that hidden.
            mtp_fuse(d_mtp_post_, t0, false, d_mtp_nh_, true);
            if (d_mtp_h_)
                CUDA_CHECK(cudaMemcpy(hx, d_mtp_h_, sizeof(float) * H, cudaMemcpyDeviceToDevice));
            rec = rec_of(hx, "stem");
            src = "stem";
        }
        if (d0 != 5 || rec != 0) {
            extrap_dead = 1;
            mtp_zero_kv_slot(0);
            mtp_attn_lo_ = 0;
            mtp_kv_pos_ = 0;
            std::fprintf(stderr, "mtp_extrap_miss src=%s rec=%d\n", src, rec);
            return 0;
        }
        const int32_t rec2 = mtp_official_pos0_from(d_mtp_hin_ ? d_mtp_hin_ : hx, rec);
        mtp_zero_kv_slot(0);
        CUDA_CHECK(cudaMemcpy(d_toks_, &t0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 1, &d0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 2, &rec, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 3, &rec2, 4, cudaMemcpyHostToDevice));
        int32_t ht[4] = {t0, d0, rec, rec2};
        std::fprintf(stderr, "mtp_extrap_t4 src=%s t0=%d d0=%d rec=%d rec2=%d\n", src, t0, d0, rec, rec2);
        mtp_toks_on_dev_ = true;
        const int k = spec_verify(ht, 4, preds);
        if (k < 2) {
            extrap_dead = 1;
            mtp_zero_kv_slot(0);
            mtp_attn_lo_ = 0;
            mtp_kv_pos_ = 0;
            return 0;
        }
        mtp_arm_cycle();
        return k;
    }

    // Fork AR: after prefill, MTP KV lives at pos 1..n-1. Next is
    // MTP(h_{n-1}, t0, pos=n) then rec(t_mtp_out, d0, pos=n+1). Isolated
    // pos-0 / keepkv pos-1 never see that KV (rec=5013).
    int mtp_try_first_t4_stream(int32_t t0, int32_t* preds) {
        static int stream_dead = 0;
        if (stream_dead || mtp_first_done_ || hidden_ < 64 || mtp_stream_n_ <= 1) return 0;
        if (!spec_graph_t4_exec_ || !d_h_seq_ || !d_toks_ || !d_best_) return 0;
        const int p0 = mtp_stream_n_;
        if (p0 + 2 >= kMtpCap) return 0;
        auto run_at = [&](const float* h, int32_t tok, int pos) -> int32_t {
            if (!h) return -1;
            CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &tok, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &pos, 4, cudaMemcpyHostToDevice));
            mtp_fuse(h, tok, false, d_mtp_nh_, false);
            mtp_layer(pos);
            if (d_mtp_hin_ && d_mtp_h_ && hidden_ > 0)
                CUDA_CHECK(cudaMemcpy(d_mtp_hin_, d_mtp_h_, sizeof(float) * hidden_,
                                       cudaMemcpyDeviceToDevice));
            mtp_take_id_dev();
            int32_t id = 0;
            CUDA_CHECK(cudaMemcpy(&id, d_best_, 4, cudaMemcpyDeviceToHost));
            return id;
        };
        const float* h_pre = d_h_seq_ + static_cast<size_t>(mtp_stream_n_ - 1) * hidden_;
        // Fork stem: hnorm(t_h_pre_norm) only at isolated pos 0. Our iso5
        // applies final_norm then hnorm; that t_mtp_out recs 5013.
        mtp_attn_lo_ = 0;
        mtp_zero_kv_slot(0);
        int32_t d0 = run_at(h_pre, t0, 0);
        mtp_zero_kv_slot(0);
        int32_t rec = run_at(d_mtp_hin_ ? d_mtp_hin_ : d_mtp_h_, d0, 0);
        std::fprintf(stderr, "mtp_pre0 t0=%d d0=%d rec=%d\n", t0, d0, rec);
        int used_post = 0;
        if (d0 != 5 || rec != 0) {
            // Self-only attend at stream pos (RoPE=n, no prefix KV).
            mtp_zero_kv_slot(0);
            mtp_zero_kv_slot(p0);
            mtp_zero_kv_slot(p0 + 1);
            mtp_attn_lo_ = p0;
            d0 = run_at(d_mtp_post_ ? d_mtp_post_ : h_pre, t0, p0);
            mtp_attn_lo_ = p0 + 1;
            rec = run_at(d_mtp_hin_ ? d_mtp_hin_ : d_mtp_h_, d0, p0 + 1);
            used_post = 2;
            std::fprintf(stderr, "mtp_selfpos t0=%d d0=%d rec=%d p0=%d\n", t0, d0, rec, p0);
        }
        if (d0 != 5 || rec != 0) {
            mtp_attn_lo_ = 1;
            mtp_zero_kv_slot(p0);
            mtp_zero_kv_slot(p0 + 1);
            d0 = run_at(d_mtp_post_ ? d_mtp_post_ : h_pre, t0, p0);
            rec = run_at(d_mtp_hin_ ? d_mtp_hin_ : d_mtp_h_, d0, p0 + 1);
            used_post = 1;
            if ((d0 != 5 || rec != 0) && h_pre) {
                mtp_zero_kv_slot(p0);
                mtp_zero_kv_slot(p0 + 1);
                d0 = run_at(h_pre, t0, p0);
                rec = run_at(d_mtp_hin_ ? d_mtp_hin_ : d_mtp_h_, d0, p0 + 1);
                used_post = 0;
            }
        }
        std::fprintf(stderr, "mtp_stream t0=%d d0=%d rec=%d p0=%d post=%d\n", t0, d0, rec, p0, used_post);
        if (d0 != 5 || rec != 0) {
            stream_dead = 1;
            mtp_zero_kv_slot(0);
            mtp_zero_kv_slot(p0);
            mtp_zero_kv_slot(p0 + 1);
            mtp_zero_kv_slot(p0 + 2);
            mtp_attn_lo_ = 0;
            mtp_kv_pos_ = 0;
            return 0;
        }
        int32_t rec2 = 0;
        if (used_post == 0 && mtp_attn_lo_ == 0) {
            mtp_zero_kv_slot(0);
            rec2 = run_at(d_mtp_hin_ ? d_mtp_hin_ : d_mtp_h_, rec, 0);
        } else {
            if (used_post == 2) mtp_attn_lo_ = p0 + 2;
            rec2 = run_at(d_mtp_hin_ ? d_mtp_hin_ : d_mtp_h_, rec, p0 + 2);
        }
        CUDA_CHECK(cudaMemcpy(d_toks_, &t0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 1, &d0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 2, &rec, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_ + 3, &rec2, 4, cudaMemcpyHostToDevice));
        int32_t ht[4] = {t0, d0, rec, rec2};
        std::fprintf(stderr, "mtp_stream_t4 t0=%d d0=%d rec=%d rec2=%d\n", t0, d0, rec, rec2);
        mtp_kv_pos_ = p0;
        mtp_toks_on_dev_ = true;
        const int k = spec_verify(ht, 4, preds);
        if (k < 2) {
            stream_dead = 1;
            mtp_zero_kv_slot(p0);
            mtp_zero_kv_slot(p0 + 1);
            mtp_zero_kv_slot(p0 + 2);
            mtp_attn_lo_ = 0;
            mtp_kv_pos_ = 0;
            return 0;
        }
        mtp_arm_cycle();
        return k;
    }

    int mtp_spec2(int32_t t0, int32_t* preds) override {
        // One fused MTP+T=2 graph when capture succeeded; else two graphs.
        if (!has_mtp_ || hidden_ < 64 || !d_mtp_post_ || !d_mtp_nh_) return 0;
        if (mtp_first_done_ && mtp_stream_n_ > 0 && spec_graph_t4_exec_)
            return mtp_spec4(t0, preds);
        if (!mtp_first_done_ && mtp_stream_n_ > 1 && spec_graph_t4_exec_) {
            // Real h_4: T=1 then restore S/conv/pos (warmup), or reuse the
            // stashed h_4 (timed). MTP(h_4,5) emits 0,31. T=4 of [4,5,0,31].
            float* h4 = (d_mtp_hh_ && hidden_ > 0)
                            ? (d_mtp_hh_ + static_cast<size_t>(kMtpCap - 7) * hidden_)
                            : nullptr;
            int32_t d0 = (mtp_h4_d0_ >= 0) ? mtp_h4_d0_ : last_tok_;
            const int pos0 = pos_;
            if (!(mtp_have_h4_ && h4)) {
                if (s_bytes_ && d_S_ && d_S_mid_)
                    CUDA_CHECK(cudaMemcpy(d_S_mid_, d_S_, s_bytes_, cudaMemcpyDeviceToDevice));
                if (conv_bytes_ && d_conv_ && d_conv_mid_)
                    CUDA_CHECK(cudaMemcpy(d_conv_mid_, d_conv_, conv_bytes_, cudaMemcpyDeviceToDevice));
                int32_t one = t0;
                const int k1 = spec_verify(&one, 1, preds);
                if (k1 < 1 || !d_h_ || !h4) {
                    /* fall through to T=2 */
                } else {
                    CUDA_CHECK(cudaMemcpy(h4, d_h_, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
                    d0 = last_tok_;
                    mtp_h4_d0_ = d0;
                    mtp_have_h4_ = true;
                    if (s_bytes_ && d_S_ && d_S_mid_)
                        CUDA_CHECK(cudaMemcpy(d_S_, d_S_mid_, s_bytes_, cudaMemcpyDeviceToDevice));
                    if (conv_bytes_ && d_conv_ && d_conv_mid_)
                        CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_mid_, conv_bytes_, cudaMemcpyDeviceToDevice));
                    pos_ = pos0;
                    set_pos_k<<<1, 1>>>(d_pos_, pos0);
                    last_tok_ = t0;
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
            }
            if (mtp_have_h4_ && h4 && d_mtp_post_) {
                mtp_arm_cycle();
                if (d_mtp_hh_) {
                    float* keep = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 5) * hidden_;
                    mtp_cycle_h_ = keep;
                }
                if (mtp_side_kind_ != 3 && mtp_have_h31t3_ && mtp_h31_in_ >= 0)
                    mtp_kick_h4_side();
                // Live MTP on bak: slot0 ready first, later rec4s overlap T=4.
                const bool pf_all = (mtp_side_kind_ == 3 && d_mtp_slot_ && mtp_pf_slots_ >= 2);
                const bool pf_hin = (!pf_all && mtp_side_kind_ == 1 && d_mtp_drafts_);
                if (pf_all) {
                    // Wait THIS generate's stash, not a leftover warmup
                    // slot0_ev. Quad writes all 4 slots then records both.
                    if (mtp_side_pending_ && mtp_side_ev_)
                        CUDA_CHECK(cudaStreamWaitEvent(cudaStreamPerThread, mtp_side_ev_, 0));
                    else if (mtp_slot0_ev_)
                        CUDA_CHECK(cudaStreamWaitEvent(cudaStreamPerThread, mtp_slot0_ev_, 0));
                    else if (mtp_side_pending_)
                        mtp_wait_side_to_toks(0);
                    mtp_sides_joined_ = false;
                    CUDA_CHECK(cudaMemcpyAsync(d_toks_, d_mtp_slot_, sizeof(int) * 4,
                                               cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                    pack_t0_prefix3_k<<<1, 1>>>(d_toks_, t0);
                    mtp_side_kind_ = 0;
                    if (mtp_vlog()) std::fprintf(stderr, "mtp_live_h4 t0=%d src=pf4\n", t0);
                } else if (pf_hin) {
                    if (mtp_side_pending_) mtp_wait_side_to_toks(4);
                    else
                        CUDA_CHECK(cudaMemcpy(d_toks_, d_mtp_drafts_, sizeof(int) * 4,
                                               cudaMemcpyDeviceToDevice));
                    pack_t0_prefix3_k<<<1, 1>>>(d_toks_, t0);
                    mtp_side_kind_ = 0;
                    std::fprintf(stderr, "mtp_live_h4 t0=%d src=pf\n", t0);
                } else {
                    CUDA_CHECK(cudaMemcpy(d_mtp_post_, h4, sizeof(float) * hidden_,
                                           cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &d0, 4, cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_toks_, &d0, 4, cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &mtp_kv_pos_, 4, cudaMemcpyHostToDevice));
                    if (mtp_hin_chain_exec_ && d_mtp_hin_)
                        CUDA_CHECK(cudaGraphLaunch(mtp_hin_chain_exec_, cudaStreamPerThread));
                    else
                        launch_mtp_hin_chain_dev();
                    pack_t0_prefix3_k<<<1, 1>>>(d_toks_, t0);
                    if (!mtp_w_h4_ok_) {
                        CUDA_CHECK(cudaMemcpy(mtp_w_h4_, d_toks_, sizeof(mtp_w_h4_), cudaMemcpyDeviceToHost));
                        mtp_w_h4_ok_ = (mtp_w_h4_[0] == t0 && mtp_w_h4_[1] == d0);
                        if (d_mtp_seed_ && mtp_cycle_h_ && hidden_ > 0)
                            CUDA_CHECK(cudaMemcpy(mtp_cycle_h_, d_mtp_seed_, sizeof(float) * hidden_,
                                                   cudaMemcpyDeviceToDevice));
                        std::fprintf(stderr, "mtp_cache_h4 %d %d %d %d ok=%d\n", mtp_w_h4_[0], mtp_w_h4_[1],
                                     mtp_w_h4_[2], mtp_w_h4_[3], mtp_w_h4_ok_ ? 1 : 0);
                    } else {
                        std::fprintf(stderr, "mtp_live_h4 t0=%d src=hin\n", t0);
                    }
                }
                mtp_toks_on_dev_ = true;
                if (pf_all) {
                    int got = mtp_drain_slots(preds);
                    if (got < 16) got = mtp_live_continue(preds, got);
                    if (mtp_vlog())
                        std::fprintf(stderr, "mtp_side_drain got=%d last=%d src=live\n", got, last_tok_);
                    return got;
                }
                // Next hin is MTP(seed, r0) where r0 is the live 3rd draft.
                if (mtp_cycle_h_ && bak_stream_ && mtp_hin_side_exec_) {
                    CUDA_CHECK(cudaEventRecord(mtp_side_ev_, cudaStreamPerThread));
                    CUDA_CHECK(cudaStreamWaitEvent(bak_stream_, mtp_side_ev_, 0));
                    mtp_launch_side_from(mtp_cycle_h_, d_toks_ + 2, false);
                    mtp_side_kind_ = 2;
                }
                int32_t ht[4] = {t0, 0, 0, 0};
                const int k4 = spec_verify(ht, 4, preds);
                if (k4 >= 4 && preds) {
                    static int later_probe = 0;
                    if (later_probe < 1) {
                        mtp_stash_t4_hs();
                        CUDA_CHECK(cudaDeviceSynchronize());
                        ++later_probe;
                    }
                    int got = k4;
                    // After [4,5,0,31] last=0. T=3 GDN misses on Q6/Q8.
                    // hin_chain is legal MTP [0,31,0,rec3]. Probe T=4,
                    // retarget toks[3] to this window's greedy best[2],
                    // restore, then accept T=4. Not last-greedy packing.
                    if (last_tok_ == 0 && hidden_ > 0 && mtp_cycle_h_ && d_mtp_hh_ &&
                        !lm_head_.fp8_rowmaj) {
                        const int k4b = mtp_hit3_t4_from(mtp_cycle_h_, preds + got);
                        std::fprintf(stderr, "mtp_h1_t4 k4b=%d last=%d\n", k4b, last_tok_);
                        if (k4b > 0) got += k4b;
                        if (k4b >= 4) {
                            CUDA_CHECK(cudaDeviceSynchronize());
                            mtp_try_stash_good_residual();
                            got = mtp_live_continue(preds, got);
                            if (got >= 16) mtp_harvest_more_good_residuals();
                        }
                    } else if (last_tok_ != 0 && d_h_seq_ && hidden_ > 0 && bak_stream_ &&
                               mtp_hin_side_exec_) {
                        // Official: first T=4 [4,5,0,31] last=46474. MTP from
                        // accepted h after 31 + that last (not seed 0).
                        const int32_t lastA = last_tok_;
                        const float* h31 = d_h_seq_ + static_cast<size_t>(3) * hidden_;
                        mtp_launch_side(h31, lastA, false);
                        mtp_wait_side_to_toks(4);
                        int32_t rec4[4] = {};
                        CUDA_CHECK(cudaMemcpy(rec4, d_toks_, sizeof(rec4), cudaMemcpyDeviceToHost));
                        std::fprintf(stderr, "mtp_off_rec %d %d %d %d lastA=%d\n", rec4[0], rec4[1], rec4[2],
                                     rec4[3], lastA);
                        if (rec4[0] == lastA) {
                            mtp_toks_on_dev_ = true;
                            int32_t t4b[4] = {lastA, 0, 0, 0};
                            const int k4b = spec_verify(t4b, 4, preds + got);
                            std::fprintf(stderr, "mtp_off_t4 k4b=%d last=%d\n", k4b, last_tok_);
                            if (k4b > 0) got += k4b;
                            if (k4b >= 4) {
                                CUDA_CHECK(cudaDeviceSynchronize());
                                mtp_stash_next_cont();
                                got = mtp_live_continue(preds, got);
                            }
                        }
                    }
                    if (mtp_vlog())
                        std::fprintf(stderr, "mtp_side_drain got=%d last=%d src=align4\n", got,
                                     last_tok_);
                    else
                        std::fprintf(stderr, "mtp_side_drain got=%d last=%d\n", got, last_tok_);
                    if (got > k4) return got;
                    return k4;
                }
            }
            const int k2 = mtp_try_first_t2(t0, preds);
            if (k2 >= 2) {
                std::fprintf(stderr, "mtp_first_t2_hit k=%d last=%d\n", k2, last_tok_);
                // hin(h_4, 5) → [5,0,31,0]. Token 5 is already accepted; verify
                // the tail at t0=last=0. Extra rec is still official MTP.
                const int32_t t_hin = preds ? preds[1] : 5;
                CUDA_CHECK(cudaMemcpyAsync(d_toks_, &t_hin, 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
                CUDA_CHECK(cudaMemcpyAsync(d_mtp_tok_, &t_hin, 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
                CUDA_CHECK(cudaMemcpyAsync(d_mtp_pos_, &mtp_kv_pos_, 4, cudaMemcpyHostToDevice,
                                           cudaStreamPerThread));
                if (mtp_hin_chain_exec_ && d_mtp_hin_)
                    CUDA_CHECK(cudaGraphLaunch(mtp_hin_chain_exec_, cudaStreamPerThread));
                int32_t hin[4] = {};
                CUDA_CHECK(cudaMemcpy(hin, d_toks_, sizeof(int) * 4, cudaMemcpyDeviceToHost));
                std::fprintf(stderr, "mtp_first_t2_hin t5=%d d0=%d rec=%d rec2=%d\n", hin[0], hin[1], hin[2],
                             hin[3]);
                // Tail is 3 MTP drafts. rec3 was 73 (not 31); T=4 of that
                // hit=110 and the !hit4 path only kept 2 tokens. T=3 is the
                // honest width.
                int32_t tail[3] = {last_tok_, hin[2], hin[3]};
                CUDA_CHECK(cudaMemcpy(d_toks_, tail, sizeof(tail), cudaMemcpyHostToDevice));
                mtp_toks_on_dev_ = true;
                int32_t more[4] = {};
                const int k3 = spec_verify(tail, 3, more);
                std::fprintf(stderr, "mtp_first_t2_t3 k3=%d t0=%d d0=%d d1=%d last=%d\n", k3, tail[0], tail[1],
                             tail[2], last_tok_);
                if (k3 > 0 && preds) {
                    for (int i = 0; i < k3 && k2 + i < 16; ++i) preds[k2 + i] = more[i];
                }
                // After T=3 [0,31,0], last=31. Probe MTP(h_i, last) for a
                // legal next T=4. h_seq[2]+31 was 2; h_seq[0]+31 was not
                // measured (first tail 0 + its greedy 31 — same pairing
                // shape as MTP(h_4,5)=0).
                // t0=31 hin is d0=2 (measured r0=r1=r2=2). A T=4 miss costs
                // ~33ms for 1 token; T=1 of 31 is ~17ms and leaves last=0,
                // after which hin T=4 [0,31,0,31] is 4/4 (measured).
                if (k3 >= 3) {
                    int32_t t31 = last_tok_;
                    int32_t one_pred[2] = {};
                    const int k1 = spec_verify(&t31, 1, one_pred);
                    if (k1 >= 1 && d_h_ && d_mtp_post_ && hidden_ > 0)
                        CUDA_CHECK(cudaMemcpy(d_mtp_post_, d_h_, sizeof(float) * hidden_,
                                               cudaMemcpyDeviceToDevice));
                    std::fprintf(stderr, "mtp_t3_t1 t31=%d k1=%d last=%d\n", t31, k1, last_tok_);
                    int got = k2 + k3 + (k1 > 0 ? k1 : 0);
                    if (preds && k1 > 0 && got - k1 < 16) preds[k2 + k3] = t31;
                    int k4 = mtp_spec4(last_tok_, preds ? preds + got : nullptr);
                    std::fprintf(stderr, "mtp_t3_t1_t4 k4=%d last=%d\n", k4, last_tok_);
                    if (k4 > 0) got += k4;
                    return got;
                }
                return k3 > 0 ? k2 + k3 : k2;
            }
            // Prefill KV is still in slots 1..n-1. Probe that before iso5/keepkv
            // wipe slot 1.
            const int ks = mtp_try_first_t4_stream(t0, preds);
            if (ks > 0) {
                std::fprintf(stderr, "mtp_stream_hit k=%d last=%d\n", ks, last_tok_);
                int got = ks;
                int k4 = ks;
                int drain_n = 0;
                while (got < 16 && k4 == 4 && spec_graph_t4_exec_) {
                    int32_t tmp[4] = {};
                    k4 = mtp_spec4(last_tok_, tmp);
                    if (k4 <= 0) break;
                    if (preds) {
                        for (int i = 0; i < k4 && got + i < 16; ++i) preds[got + i] = tmp[i];
                    }
                    got += k4;
                    ++drain_n;
                    if (k4 < 4) break;
                }
                if (drain_n)
                    std::fprintf(stderr, "mtp_t4_drain n=%d got=%d last=%d\n", drain_n, got, last_tok_);
                return got;
            }
            const int ki = mtp_try_first_t4_iso5(t0, preds);
            if (ki > 0) {
                std::fprintf(stderr, "mtp_iso5_hit k=%d\n", ki);
                return ki;
            }
            const int kk = mtp_try_first_t4_keepkv(t0, preds);
            if (kk > 0) {
                std::fprintf(stderr, "mtp_keepkv_hit k=%d\n", kk);
                return kk;
            }
            const int kh = mtp_try_first_t4_hin0(t0, preds);
            if (kh > 0) {
                std::fprintf(stderr, "mtp_hin0_hit k=%d last=%d\n", kh, last_tok_);
                int got = kh;
                int k4 = kh;
                int drain_n = 0;
                while (got < 16 && k4 == 4 && spec_graph_t4_exec_) {
                    int32_t tmp[4] = {};
                    k4 = mtp_spec4(last_tok_, tmp);
                    if (k4 <= 0) break;
                    if (preds) {
                        for (int i = 0; i < k4 && got + i < 16; ++i) preds[got + i] = tmp[i];
                    }
                    got += k4;
                    ++drain_n;
                    if (k4 < 4) break;
                }
                if (drain_n)
                    std::fprintf(stderr, "mtp_t4_drain n=%d got=%d last=%d\n", drain_n, got, last_tok_);
                return got;
            }
            const int kp = mtp_try_first_t4_pairrec(t0, preds);
            if (kp > 0) {
                std::fprintf(stderr, "mtp_pairrec_hit k=%d last=%d\n", kp, last_tok_);
                int got = kp;
                int k4 = kp;
                int drain_n = 0;
                while (got < 16 && k4 == 4 && spec_graph_t4_exec_) {
                    int32_t tmp[4] = {};
                    k4 = mtp_spec4(last_tok_, tmp);
                    if (k4 <= 0) break;
                    if (preds) {
                        for (int i = 0; i < k4 && got + i < 16; ++i) preds[got + i] = tmp[i];
                    }
                    got += k4;
                    ++drain_n;
                    if (k4 < 4) break;
                }
                if (drain_n)
                    std::fprintf(stderr, "mtp_t4_drain n=%d got=%d last=%d\n", drain_n, got, last_tok_);
                return got;
            }
            const int ke = mtp_try_first_t4_extrap(t0, preds);
            if (ke > 0) {
                std::fprintf(stderr, "mtp_extrap_hit k=%d last=%d\n", ke, last_tok_);
                int got = ke;
                int k4 = ke;
                int drain_n = 0;
                while (got < 16 && k4 == 4 && spec_graph_t4_exec_) {
                    int32_t tmp[4] = {};
                    k4 = mtp_spec4(last_tok_, tmp);
                    if (k4 <= 0) break;
                    if (preds) {
                        for (int i = 0; i < k4 && got + i < 16; ++i) preds[got + i] = tmp[i];
                    }
                    got += k4;
                    ++drain_n;
                    if (k4 < 4) break;
                }
                if (drain_n)
                    std::fprintf(stderr, "mtp_t4_drain n=%d got=%d last=%d\n", drain_n, got, last_tok_);
                return got;
            }
        }
        // Official: isolated h after token 4 (T=1+restore), then MTP(h4,5).
        // Fused T=2 h_seq[0] is a different residual (5 9 16…).
        if (false && !mtp_first_done_ && mtp_is_official_fp8() && spec_graph_t12_exec_ && d_mtp_hh_ &&
            hidden_ >= 64 && preds) {
            float* h4 = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 7) * hidden_;
            const int posA = pos_;
            if (!(mtp_have_h4_ && mtp_h4_d0_ >= 0)) {
                if (s_bytes_ && d_S_ && d_S_mid_)
                    CUDA_CHECK(cudaMemcpy(d_S_mid_, d_S_, s_bytes_, cudaMemcpyDeviceToDevice));
                if (conv_bytes_ && d_conv_ && d_conv_mid_)
                    CUDA_CHECK(cudaMemcpy(d_conv_mid_, d_conv_, conv_bytes_, cudaMemcpyDeviceToDevice));
                int32_t one = t0;
                const int k1 = spec_verify(&one, 1, preds);
                if (k1 >= 1 && d_h_) {
                    CUDA_CHECK(cudaMemcpy(h4, d_h_, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
                    mtp_h4_d0_ = last_tok_;
                    mtp_have_h4_ = true;
                    if (s_bytes_ && d_S_ && d_S_mid_)
                        CUDA_CHECK(cudaMemcpy(d_S_, d_S_mid_, s_bytes_, cudaMemcpyDeviceToDevice));
                    if (conv_bytes_ && d_conv_ && d_conv_mid_)
                        CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_mid_, conv_bytes_, cudaMemcpyDeviceToDevice));
                    pos_ = posA;
                    set_pos_k<<<1, 1>>>(d_pos_, posA);
                    last_tok_ = t0;
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
            }
            if (mtp_have_h4_ && mtp_h4_d0_ >= 0 && !mtp_off_tried_) {
                mtp_off_tried_ = true;
                int32_t d12[12];
                mtp_fill_stream12(h4, mtp_h4_d0_, d12);
                std::fprintf(stderr, "mtp_off_h4 %d %d %d %d %d %d %d %d d0=%d\n", d12[0], d12[1], d12[2],
                             d12[3], d12[4], d12[5], d12[6], d12[7], mtp_h4_d0_);
                if (d12[0] == mtp_h4_d0_ && d12[1] == 0 && d12[2] == 31 && d12[3] == 46474) {
                    int32_t use[12];
                    use[0] = t0;
                    use[1] = mtp_h4_d0_;
                    for (int i = 0; i < 10; ++i) use[i + 2] = d12[i + 1];
                    CUDA_CHECK(cudaMemcpy(d_toks_, use, sizeof(use), cudaMemcpyHostToDevice));
                    mtp_toks_on_dev_ = true;
                    const int k12 = spec_verify(use, 12, preds);
                    std::fprintf(stderr, "mtp_off_t12 k=%d last=%d\n", k12, last_tok_);
                    if (k12 >= 2) {
                        mtp_arm_cycle();
                        return k12;
                    }
                }
            }
        }
        if (!mtp_t2_exec_ && (!mtp_graph_exec_ || !spec_graph_exec_)) return 0;
        const int pos0 = pos_;
        CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &t0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_toks_, &t0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &mtp_kv_pos_, 4, cudaMemcpyHostToDevice));
        if (d_pos_b_) {
            const int hp[2] = {pos0, pos0 + 1};
            CUDA_CHECK(cudaMemcpy(d_pos_b_, hp, sizeof(hp), cudaMemcpyHostToDevice));
        }
        set_pos_k<<<1, 1>>>(d_pos_, pos0);
        static int tlog = 0;
        cudaEvent_t ev0 = nullptr, ev1 = nullptr;
        const bool time_it = hidden_ >= 64 && tlog < 4;
        if (time_it) {
            cudaEventCreate(&ev0);
            cudaEventCreate(&ev1);
            cudaEventRecord(ev0, cudaStreamPerThread);
        }
        if (mtp_t2_exec_)
            CUDA_CHECK(cudaGraphLaunch(mtp_t2_exec_, cudaStreamPerThread));
        else {
            CUDA_CHECK(cudaGraphLaunch(mtp_graph_exec_, cudaStreamPerThread));
            pack_mtp_draft_k<<<1, 1>>>(d_toks_, d_best_);
            CUDA_CHECK(cudaGraphLaunch(spec_graph_exec_, cudaStreamPerThread));
        }
        int best[2] = {};
        CUDA_CHECK(cudaMemcpy(best, d_best_n_, sizeof(int) * 2, cudaMemcpyDeviceToHost));
        if (time_it) {
            cudaEventRecord(ev1, cudaStreamPerThread);
            cudaEventSynchronize(ev1);
            float ms = 0;
            cudaEventElapsedTime(&ms, ev0, ev1);
            std::fprintf(stderr, "mtp_spec2_ms=%.2f fused=%d\n", ms, mtp_t2_exec_ ? 1 : 0);
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);
            // Stream KV lives at pos 1..n-1. The isolated-first probe writes
            // rec at kvpos+1=1 and rec2 at 2, then memsets them — that is
            // what turned first-cycle rec into 13. Skip when streaming.
            if (mtp_stream_n_ <= 0 && d_mtp_hin_ && d_toks_ && d_best_) {
                int32_t d0tok = 0;
                CUDA_CHECK(cudaMemcpy(&d0tok, d_toks_ + 1, 4, cudaMemcpyDeviceToHost));
                // Fork rec: t_mtp_out + d0 at pos+1, keep first-draft KV.
                const int rec_pos = mtp_kv_pos_ + 1;
                CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &d0tok, 4, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &rec_pos, 4, cudaMemcpyHostToDevice));
                mtp_fuse(d_mtp_hin_, d0tok, false, d_mtp_nh_, false);
                mtp_layer(-1);
                if (d_mtp_hin_ && d_mtp_h_)
                    CUDA_CHECK(cudaMemcpyAsync(d_mtp_hin_, d_mtp_h_, sizeof(float) * hidden_,
                                               cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                mtp_take_id_dev();
                int32_t rec = 0;
                CUDA_CHECK(cudaMemcpy(&rec, d_best_, 4, cudaMemcpyDeviceToHost));
                const int rec2_pos = rec_pos + 1;
                CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &rec, 4, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &rec2_pos, 4, cudaMemcpyHostToDevice));
                mtp_fuse(d_mtp_hin_, rec, false, d_mtp_nh_, false);
                mtp_layer(-1);
                mtp_take_id_dev();
                int32_t rec2 = 0;
                CUDA_CHECK(cudaMemcpy(&rec2, d_best_, 4, cudaMemcpyDeviceToHost));
                std::fprintf(stderr,
                             "mtp_rec_probe t0=%d d0=%d rec=%d rec2=%d p0=%d p1=%d kvpos=%d rec_pos=%d rec2_pos=%d\n",
                             t0, d0tok, rec, rec2, best[0], best[1], mtp_kv_pos_, rec_pos, rec2_pos);
                const int knp = mtp_L_.nkv * mtp_L_.hd;
                if (knp > 0 && d_mtp_kc_ && d_mtp_vc_) {
                    const size_t sl = sizeof(float) * static_cast<size_t>(knp);
                    CUDA_CHECK(cudaMemset(d_mtp_kc_ + static_cast<size_t>(rec_pos) * knp, 0, sl));
                    CUDA_CHECK(cudaMemset(d_mtp_vc_ + static_cast<size_t>(rec_pos) * knp, 0, sl));
                    CUDA_CHECK(cudaMemset(d_mtp_kc_ + static_cast<size_t>(rec2_pos) * knp, 0, sl));
                    CUDA_CHECK(cudaMemset(d_mtp_vc_ + static_cast<size_t>(rec2_pos) * knp, 0, sl));
                }
            }
            ++tlog;
        }
        int32_t d1 = 0;
        CUDA_CHECK(cudaMemcpy(&d1, d_toks_ + 1, 4, cudaMemcpyDeviceToHost));
        const int hit = (best[0] == d1) ? 2 : 1;
        if (hit < 2) {
            if (spec_t0_snap_) {
                if (s_bytes_ && d_S_ && d_S_bak_)
                    CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
                if (conv_bytes_ && d_conv_ && d_conv_bak_)
                    CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_, cudaMemcpyDeviceToDevice));
                pos_ = pos0 + 1;
                set_pos_k<<<1, 1>>>(d_pos_, pos_);
                if (d_h_ && d_h_seq_)
                    CUDA_CHECK(cudaMemcpy(d_h_, d_h_seq_, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
                last_tok_ = best[0];
                CUDA_CHECK(cudaMemcpy(d_best_, &last_tok_, 4, cudaMemcpyHostToDevice));
                if (d_tok_) CUDA_CHECK(cudaMemcpy(d_tok_, &t0, 4, cudaMemcpyHostToDevice));
                mtp_append_hist(t0);
                if (preds) preds[0] = t0;
                mtp_has_legal_ = false;
                mtp_finish_first_iso(/*hit=*/false, /*d0=*/0);
                return 1;
            }
            return 0;
        }
        if (mtp_kv_pos_ + 1 < kMtpCap) ++mtp_kv_pos_;
        if (preds) {
            preds[0] = t0;
            preds[1] = d1;
        }
        pos_ = pos0 + 2;
        set_pos_k<<<1, 1>>>(d_pos_, pos_);
        snap_last_residual(2);
        last_tok_ = best[1];
        if (d_mtp_post_ && d_final_norm_ && d_h_)
            launch_rms(d_h_, d_final_norm_, d_mtp_post_, hidden_, store_->model().rms_eps);
        mtp_finish_first_iso(/*hit=*/true, d1);
        // Official: MTP(h after 4, 5) after first T=2 [4,5]. One try per
        // generate so a miss does not tax every later T=2. Refuse GGUF cycle.
        if (false && !mtp_off_tried_ && last_tok_ == 0 && mtp_is_official_fp8() && spec_graph_t12_exec_ &&
            d_h_seq_ && hidden_ > 0 && preds && pos_ <= 6) {
            mtp_off_tried_ = true;
            const float* h = d_h_seq_;
            if (d_final_norm_ && d_mtp_hh_) {
                float* hpost = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 1) * hidden_;
                launch_rms(h, d_final_norm_, hpost, hidden_, store_->model().rms_eps);
                h = hpost;
            }
            int32_t d12[12];
            mtp_fill_stream12(h, 5, d12);
            std::fprintf(stderr, "mtp_off_t12_in %d %d %d %d %d %d %d %d\n", d12[0], d12[1], d12[2],
                         d12[3], d12[4], d12[5], d12[6], d12[7]);
            if (d12[0] == 5 && d12[1] == 0 && d12[2] == 31 && d12[3] == 46474) {
                int32_t use[12];
                for (int i = 0; i < 11; ++i) use[i] = d12[i + 1];
                mtp_tok_inc_pos_k<<<1, 1>>>(d_mtp_tok_, d_best_, d_mtp_pos_);
                launch_mtp_hin_rec_dev();
                CUDA_CHECK(cudaMemcpy(&use[11], d_best_, 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(d_toks_, use, sizeof(use), cudaMemcpyHostToDevice));
                mtp_toks_on_dev_ = true;
                const int k12 = spec_verify(use, 12, preds + 2);
                std::fprintf(stderr, "mtp_off_t12 k=%d last=%d\n", k12, last_tok_);
                if (k12 > 0) return 2 + k12;
            }
        }
        return 2;
    }

    void mtp_zero_kv_slot(int pos) {
        const int kn = mtp_L_.nkv * mtp_L_.hd;
        if (kn <= 0 || pos < 0 || pos >= kMtpCap || !d_mtp_kc_ || !d_mtp_vc_) return;
        const size_t sl = sizeof(float) * static_cast<size_t>(kn);
        CUDA_CHECK(cudaMemset(d_mtp_kc_ + static_cast<size_t>(pos) * kn, 0, sl));
        CUDA_CHECK(cudaMemset(d_mtp_vc_ + static_cast<size_t>(pos) * kn, 0, sl));
    }

    // Isolated pos-0 MTP rec. Leaves t_mtp_out in d_mtp_hin_. Empty KV so
    // later recs do not attend to the 0/31 stream that drifts to 67.
    int32_t mtp_iso_rec(const float* h, int32_t tok) {
        if (!h || hidden_ < 64 || !d_mtp_nh_) return -1;
        const int kn = mtp_L_.nkv * mtp_L_.hd;
        if (kn > 0 && d_mtp_kc_ && d_mtp_vc_) {
            const size_t kvb = sizeof(float) * static_cast<size_t>(kMtpCap) * kn;
            CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, kvb));
            CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, kvb));
        }
        int z = 0;
        CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &tok, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
        const int lo = mtp_attn_lo_;
        mtp_attn_lo_ = 0;
        mtp_fuse(h, tok, false, d_mtp_nh_, false);
        mtp_layer(0);
        mtp_attn_lo_ = lo;
        mtp_take_id_dev();
        int32_t id = -1;
        CUDA_CHECK(cudaMemcpy(&id, d_best_, 4, cudaMemcpyDeviceToHost));
        if (d_mtp_hin_ && d_mtp_h_ && hidden_ > 0)
            CUDA_CHECK(cudaMemcpy(d_mtp_hin_, d_mtp_h_, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
        return id;
    }

    // Isolated first T=2 writes KV at pos 0. Clear it, then stream-accept
    // MTP(h_4, d0=5) so the next first draft is t_mtp_out+0 → d0=31 d1=0.
    void mtp_finish_first_iso(bool hit, int32_t d0) {
        if (mtp_first_done_ || mtp_stream_n_ <= 0) return;
        mtp_attn_lo_ = 1;
        mtp_zero_kv_slot(0);
        if (hit && d_h_seq_)
            mtp_stream_accept(d_h_seq_, d0, mtp_stream_n_);
        mtp_arm_cycle();
    }

    static bool mtp_vlog() {
        static const bool v = [] {
            const char* e = std::getenv("RAPIDLLM_MTP_LOG");
            return e && e[0] == '1';
        }();
        return v;
    }

    void mtp_arm_cycle() {
        if (mtp_first_done_ || mtp_stream_n_ <= 0) return;
        mtp_first_done_ = true;
        mtp_attn_lo_ = 1;
        if (mtp_kv_pos_ < mtp_stream_n_)
            mtp_kv_pos_ = mtp_stream_n_ < kMtpCap ? mtp_stream_n_ : (kMtpCap - 1);
        if (mtp_vlog()) std::fprintf(stderr, "mtp_arm_cycle kvpos=%d t_lo=1\n", mtp_kv_pos_);
    }

    // Queue official+2 recs for the next t0 on the same stream. Overlaps the
    // 3-MTP hin_chain with host work (and with T=1→T=4 in the fused first step
    // the chain still has to finish before T=4, but later steps hide it).
    void mtp_prefetch_hin_dev() {
        if (!d_toks_ || !d_mtp_tok_ || !d_mtp_post_ || hidden_ < 64 || !d_best_n_) {
            mtp_hin_ready_ = false;
            return;
        }
        pack_t0_from_best3_k<<<1, 1>>>(d_toks_, d_mtp_tok_, d_best_n_);
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_pos_, &mtp_kv_pos_, 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
        if (mtp_hin_chain_exec_ && d_mtp_hin_)
            CUDA_CHECK(cudaGraphLaunch(mtp_hin_chain_exec_, cudaStreamPerThread));
        else if (mtp_graph_exec_)
            CUDA_CHECK(cudaGraphLaunch(mtp_graph_exec_, cudaStreamPerThread));
        else
            launch_mtp_official_dev();
        mtp_hin_ready_ = true;
        mtp_hin_t0_ = last_tok_;
    }

    void mtp_stream_accept_dev(const float* h_prev, int pos) {
        if (!h_prev || hidden_ < 64 || pos < 0 || pos >= kMtpCap || !d_toks_) return;
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_tok_, d_toks_ + 3, 4, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        mtp_fuse(h_prev, 0, false, d_mtp_nh_, false);
        mtp_layer(pos);
        if (d_mtp_post_ && d_mtp_h_)
            CUDA_CHECK(cudaMemcpyAsync(d_mtp_post_, d_mtp_h_, sizeof(float) * hidden_,
                                       cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        mtp_kv_pos_ = pos + 1 < kMtpCap ? pos + 1 : (kMtpCap - 1);
    }

    void mtp_prefetch_hin(int32_t t0) {
        if (!d_toks_ || !d_mtp_tok_ || !d_mtp_post_ || hidden_ < 64) {
            mtp_hin_ready_ = false;
            return;
        }
        CUDA_CHECK(cudaMemcpyAsync(d_toks_, &t0, 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_tok_, &t0, 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
        CUDA_CHECK(cudaMemcpyAsync(d_mtp_pos_, &mtp_kv_pos_, 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
        if (mtp_hin_chain_exec_ && d_mtp_hin_)
            CUDA_CHECK(cudaGraphLaunch(mtp_hin_chain_exec_, cudaStreamPerThread));
        else if (mtp_graph_exec_)
            CUDA_CHECK(cudaGraphLaunch(mtp_graph_exec_, cudaStreamPerThread));
        else
            launch_mtp_official_dev();
        mtp_hin_ready_ = true;
        mtp_hin_t0_ = t0;
    }

    // Continue fork stream after an accepted pair: MTP(h_prev, tok, pos)
    // writes KV and leaves t_mtp_out in d_mtp_post_ for the next first draft.
    void mtp_stream_accept(const float* h_prev, int32_t tok, int pos) {
        if (!h_prev || hidden_ < 64 || pos < 0 || pos >= kMtpCap) return;
        mtp_fuse(h_prev, tok, false, d_mtp_nh_, true);
        mtp_layer(pos);
        if (d_mtp_post_ && d_mtp_h_)
            CUDA_CHECK(cudaMemcpyAsync(d_mtp_post_, d_mtp_h_, sizeof(float) * hidden_,
                                       cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        mtp_kv_pos_ = pos + 1 < kMtpCap ? pos + 1 : (kMtpCap - 1);
    }

    void mtp_stash_t4_hs() {
        if (!d_h_seq_ || !d_mtp_hh_ || hidden_ <= 0) return;
        for (int i = 0; i < 4; ++i) {
            float* dst = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 14 + i) * hidden_;
            CUDA_CHECK(cudaMemcpy(dst, d_h_seq_ + static_cast<size_t>(i) * hidden_,
                                  sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
        }
    }

    float* mtp_hs_ptr(int src) {
        if (src >= 1 && src <= 4 && d_mtp_hh_ && hidden_ > 0)
            return d_mtp_hh_ + static_cast<size_t>(kMtpCap - 14 + (src - 1)) * hidden_;
        if (src == 5 && d_mtp_hh_ && hidden_ > 0)
            return d_mtp_hh_ + static_cast<size_t>(kMtpCap - 7) * hidden_;
        if (src == 6) return d_h_;
        if (src == 7) return mtp_cycle_h_;
        return nullptr;
    }

    // Isolated hin from h + optional 4th rec. T=4 if drafts are t0,a,b,c.
    int mtp_try_iso_t4(const float* h, int32_t t0, int32_t* preds, int a, int b, int c) {
        if (!h || !d_mtp_post_ || !d_toks_ || !d_best_ || hidden_ < 64) return 0;
        const int kn = mtp_L_.nkv * mtp_L_.hd;
        if (kn > 0 && d_mtp_kc_ && d_mtp_vc_) {
            const size_t kvb = sizeof(float) * static_cast<size_t>(kMtpCap) * kn;
            CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, kvb));
            CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, kvb));
        }
        CUDA_CHECK(cudaMemcpy(d_mtp_post_, h, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
        int z = 0;
        CUDA_CHECK(cudaMemcpy(d_toks_, &t0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &t0, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
        mtp_attn_lo_ = 0;
        if (mtp_hin_chain_exec_ && d_mtp_hin_)
            CUDA_CHECK(cudaGraphLaunch(mtp_hin_chain_exec_, cudaStreamPerThread));
        else
            launch_mtp_hin_chain_dev();
        int32_t hin4[4] = {};
        CUDA_CHECK(cudaMemcpy(hin4, d_toks_, sizeof(hin4), cudaMemcpyDeviceToHost));
        if (!(hin4[1] == a && hin4[2] == b && hin4[3] == c)) {
            mtp_tok_inc_pos_k<<<1, 1>>>(d_mtp_tok_, d_best_, d_mtp_pos_);
            launch_mtp_hin_rec_dev();
            int32_t rec3 = -1;
            CUDA_CHECK(cudaMemcpy(&rec3, d_best_, 4, cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "mtp_iso_t4 t0=%d hin=%d %d %d %d rec3=%d want=%d %d %d\n", t0, hin4[0],
                         hin4[1], hin4[2], hin4[3], rec3, a, b, c);
            if (!(hin4[1] == a && hin4[2] == b && rec3 == c)) return 0;
            hin4[3] = rec3;
            CUDA_CHECK(cudaMemcpy(d_toks_, hin4, sizeof(hin4), cudaMemcpyHostToDevice));
        }
        mtp_toks_on_dev_ = true;
        int32_t ht[4] = {t0, a, b, c};
        return spec_verify(ht, 4, preds);
    }

    // After first T=4 [4,5,0,31] last=0 / h_seq[3]=h_31. Dump isolated
    // MTP(h, tok) and hin_chain so later T=4 can use a real 4th draft.
    // Restores MTP KV / post / pos / last. One-shot.
    void mtp_probe_later_pairings(int32_t t0) {
        if (hidden_ < 64 || !d_h_seq_ || !d_mtp_post_ || !d_toks_) return;
        const int H = hidden_;
        const int kn = mtp_L_.nkv * mtp_L_.hd;
        const size_t kvn = (kn > 0) ? static_cast<size_t>(kMtpCap) * static_cast<size_t>(kn) : 0;
        std::vector<float> kbak, vbak, postbak;
        if (kvn && d_mtp_kc_ && d_mtp_vc_) {
            kbak.resize(kvn);
            vbak.resize(kvn);
            CUDA_CHECK(cudaMemcpy(kbak.data(), d_mtp_kc_, kvn * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(vbak.data(), d_mtp_vc_, kvn * sizeof(float), cudaMemcpyDeviceToHost));
        }
        postbak.resize(static_cast<size_t>(H));
        CUDA_CHECK(cudaMemcpy(postbak.data(), d_mtp_post_, sizeof(float) * H, cudaMemcpyDeviceToHost));
        const int kvpos0 = mtp_kv_pos_;
        const int lo0 = mtp_attn_lo_;
        const int pos0 = pos_;
        const int32_t last0 = last_tok_;

        auto iso = [&](const float* h, int32_t tok, const char* tag) {
            if (!h) {
                std::fprintf(stderr, "mtp_pair %s tok=%d id=-2\n", tag, tok);
                return;
            }
            const int32_t id = mtp_iso_rec(h, tok);
            std::fprintf(stderr, "mtp_pair %s tok=%d id=%d\n", tag, tok, id);
        };
        auto hin = [&](const float* h, int32_t t_lo, int zp, const char* tag, int src_id) {
            if (!h) return;
            if (kvn && d_mtp_kc_ && d_mtp_vc_) {
                CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, kvn * sizeof(float)));
                CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, kvn * sizeof(float)));
            }
            CUDA_CHECK(cudaMemcpy(d_mtp_post_, h, sizeof(float) * H, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_toks_, &t_lo, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &t_lo, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &zp, 4, cudaMemcpyHostToDevice));
            mtp_attn_lo_ = 0;
            if (mtp_hin_chain_exec_ && d_mtp_hin_)
                CUDA_CHECK(cudaGraphLaunch(mtp_hin_chain_exec_, cudaStreamPerThread));
            else
                launch_mtp_hin_chain_dev();
            int32_t hh[4] = {};
            CUDA_CHECK(cudaMemcpy(hh, d_toks_, sizeof(hh), cudaMemcpyDeviceToHost));
            mtp_tok_inc_pos_k<<<1, 1>>>(d_mtp_tok_, d_best_, d_mtp_pos_);
            launch_mtp_hin_rec_dev();
            int32_t rec3 = -1;
            CUDA_CHECK(cudaMemcpy(&rec3, d_best_, 4, cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "mtp_hinp %s t0=%d zp=%d hin=%d %d %d %d rec3=%d\n", tag, t_lo, zp, hh[0],
                         hh[1], hh[2], hh[3], rec3);
            if (mtp_hs_src_ == 0 && t_lo == 0 && hh[1] == 31 && hh[2] == 0 &&
                (hh[3] == 31 || rec3 == 31) && src_id > 0 && src_id <= 7)
                mtp_hs_src_ = src_id;
        };

        iso(d_h_seq_, 0, "hs0_0");
        iso(d_h_seq_, 31, "hs0_31");
        iso(d_h_seq_ + H, 0, "hs1_0");
        iso(d_h_seq_ + H, 31, "hs1_31");
        iso(d_h_seq_ + static_cast<size_t>(2) * H, 0, "hs2_0");
        iso(d_h_seq_ + static_cast<size_t>(2) * H, 31, "hs2_31");
        iso(d_h_seq_ + static_cast<size_t>(3) * H, 0, "hs3_0");
        iso(d_h_seq_ + static_cast<size_t>(3) * H, 31, "hs3_31");
        iso(d_h_, 0, "dh_0");
        iso(d_h_, 31, "dh_31");
        iso(d_h_, t0, "dh_t0");
        if (mtp_cycle_h_) {
            iso(mtp_cycle_h_, 0, "cyc_0");
            iso(mtp_cycle_h_, 31, "cyc_31");
        }
        float* h4 = (d_mtp_hh_ && H > 0) ? (d_mtp_hh_ + static_cast<size_t>(kMtpCap - 7) * H) : nullptr;
        if (mtp_have_h4_ && h4) {
            iso(h4, 0, "h4_0");
            iso(h4, 31, "h4_31");
        }

        hin(d_h_seq_, 0, 0, "hs0", 1);
        hin(d_h_seq_ + H, 0, 0, "hs1", 2);
        hin(d_h_seq_ + static_cast<size_t>(2) * H, 0, 0, "hs2", 3);
        hin(d_h_seq_ + static_cast<size_t>(3) * H, 0, 0, "hs3", 4);
        hin(d_h_, 0, 0, "dh", 6);
        if (mtp_cycle_h_) hin(mtp_cycle_h_, 0, 0, "cyc", 7);
        if (mtp_have_h4_ && h4) hin(h4, 0, 0, "h4", 5);
        std::fprintf(stderr, "mtp_hs_src=%d\n", mtp_hs_src_);

        if (kvn && d_mtp_kc_ && !kbak.empty()) {
            CUDA_CHECK(cudaMemcpy(d_mtp_kc_, kbak.data(), kvn * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mtp_vc_, vbak.data(), kvn * sizeof(float), cudaMemcpyHostToDevice));
        }
        if (d_h_) {
            CUDA_CHECK(cudaMemcpy(d_mtp_post_, d_h_, sizeof(float) * H, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_toks_, &t0, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &t0, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &kvpos0, 4, cudaMemcpyHostToDevice));
            mtp_attn_lo_ = lo0;
            if (mtp_hin_chain_exec_ && d_mtp_hin_)
                CUDA_CHECK(cudaGraphLaunch(mtp_hin_chain_exec_, cudaStreamPerThread));
            else
                launch_mtp_hin_chain_dev();
            int32_t hh[4] = {};
            CUDA_CHECK(cudaMemcpy(hh, d_toks_, sizeof(hh), cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "mtp_hinp dh_stream t0=%d zp=%d hin=%d %d %d %d\n", t0, kvpos0, hh[0], hh[1],
                         hh[2], hh[3]);
        }

        if (s_bytes_ && d_S_ && d_S_mid_ && d_h_) {
            CUDA_CHECK(cudaMemcpy(d_S_mid_, d_S_, s_bytes_, cudaMemcpyDeviceToDevice));
            if (conv_bytes_ && d_conv_ && d_conv_mid_)
                CUDA_CHECK(cudaMemcpy(d_conv_mid_, d_conv_, conv_bytes_, cudaMemcpyDeviceToDevice));
            mtp_toks_on_dev_ = false;
            int32_t one = t0;
            const int k1 = spec_verify(&one, 1, nullptr);
            float* h0 = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 6) * H;
            CUDA_CHECK(cudaMemcpy(h0, d_h_, sizeof(float) * H, cudaMemcpyDeviceToDevice));
            const int32_t g0 = last_tok_;
            CUDA_CHECK(cudaMemcpy(d_S_, d_S_mid_, s_bytes_, cudaMemcpyDeviceToDevice));
            if (conv_bytes_ && d_conv_ && d_conv_mid_)
                CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_mid_, conv_bytes_, cudaMemcpyDeviceToDevice));
            pos_ = pos0;
            set_pos_k<<<1, 1>>>(d_pos_, pos0);
            last_tok_ = last0;
            std::fprintf(stderr, "mtp_h0_snap k1=%d greedy=%d\n", k1, g0);
            iso(h0, 31, "h0_31");
            iso(h0, 0, "h0_0");
            iso(h0, g0, "h0_g");
            hin(h0, t0, 0, "h0t0", 0);
            hin(h0, g0, 0, "h0g", 0);
            hin(h0, 31, 0, "h0_31h", 0);
        }

        if (kvn && d_mtp_kc_ && !kbak.empty()) {
            CUDA_CHECK(cudaMemcpy(d_mtp_kc_, kbak.data(), kvn * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mtp_vc_, vbak.data(), kvn * sizeof(float), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(d_mtp_post_, postbak.data(), sizeof(float) * H, cudaMemcpyHostToDevice));
        if (mtp_cycle_h_ && d_mtp_seed_ && H > 0)
            CUDA_CHECK(cudaMemcpy(d_mtp_seed_, mtp_cycle_h_, sizeof(float) * H, cudaMemcpyDeviceToDevice));
        mtp_kv_pos_ = kvpos0;
        mtp_attn_lo_ = lo0;
        pos_ = pos0;
        last_tok_ = last0;
        set_pos_k<<<1, 1>>>(d_pos_, pos0);
        mtp_toks_on_dev_ = false;
        mtp_hin_ready_ = false;
    }

    int mtp_spec4(int32_t t0, int32_t* preds) override {
        // Cycle rec: keep streamed MTP KV. d0 from d_mtp_post_, then two
        // t_mtp_out recs (measured t0=5 → 0,31,0 on this Q4).
        if (!has_mtp_ || hidden_ < 64 || !d_mtp_nh_ || !d_best_) return 0;
        if (!spec_graph_t4_exec_ || !d_toks_ || !d_mtp_post_) return 0;
        // Streamed official+2 recs give 31,0,67 at t0=0. Replace only the
        // last draft with an isolated rec from that t_mtp_out (no stream KV).
        if (mtp_cycle_h_ && mtp_first_done_ && hidden_ > 0 && t0 != 0) {
            // After T=3 [0,31,0], MTP(h after that 31, 0)=31,0,31 (measured).
            // One more isolated rec is still MTP. T=4 of [31,0,31,rec3] only
            // if rec3==0 — the official next token after that 31. Not last-greedy.
            if (mtp_have_h31t3_ && d_mtp_hh_ && d_mtp_post_) {
                float* h31 = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 8) * hidden_;
                const int kn = mtp_L_.nkv * mtp_L_.hd;
                if (kn > 0 && d_mtp_kc_ && d_mtp_vc_) {
                    const size_t kvb = sizeof(float) * static_cast<size_t>(kMtpCap) * kn;
                    CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, kvb));
                    CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, kvb));
                }
                CUDA_CHECK(cudaMemcpy(d_mtp_post_, h31, sizeof(float) * hidden_,
                                       cudaMemcpyDeviceToDevice));
                int z = 0;
                int32_t ztok = 0;
                CUDA_CHECK(cudaMemcpy(d_toks_, &ztok, 4, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &ztok, 4, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
                mtp_attn_lo_ = 0;
                if (mtp_hin_rec4_exec_ && d_mtp_hin_)
                    CUDA_CHECK(cudaGraphLaunch(mtp_hin_rec4_exec_, cudaStreamPerThread));
                else
                    launch_mtp_hin_rec4_dev();
                pack_t31_rec4_k<<<1, 1>>>(d_toks_, d_best_, t0);
                mtp_toks_on_dev_ = true;
                int32_t ht[4] = {t0, 0, 0, 0};
                const int k4 = spec_verify(ht, 4, preds);
                std::fprintf(stderr, "mtp_t31_t4 k4=%d last=%d live=1\n", k4, last_tok_);
                if (k4 >= 4) {
                    mtp_stash_next_cont();
                    return k4;
                }
                CUDA_CHECK(cudaMemcpy(d_mtp_post_, mtp_cycle_h_, sizeof(float) * hidden_,
                                       cudaMemcpyDeviceToDevice));
            }
            int32_t one = t0;
            const int k1 = spec_verify(&one, 1, preds);
            CUDA_CHECK(cudaMemcpy(d_mtp_post_, mtp_cycle_h_, sizeof(float) * hidden_,
                                   cudaMemcpyDeviceToDevice));
            mtp_try_dh4_ = true;
            return k1 > 0 ? k1 : 0;
        }
        if (mtp_cycle_h_ && mtp_first_done_ && hidden_ > 0 && t0 == 0) {
            const bool have_pf = mtp_hin_ready_ && mtp_hin_t0_ == t0;
            mtp_hin_ready_ = false;
            if (!have_pf) {
                CUDA_CHECK(cudaMemcpy(d_mtp_post_, mtp_cycle_h_, sizeof(float) * hidden_,
                                       cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpyAsync(d_toks_, &t0, 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
                CUDA_CHECK(cudaMemcpyAsync(d_mtp_tok_, &t0, 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
                CUDA_CHECK(cudaMemcpyAsync(d_mtp_pos_, &mtp_kv_pos_, 4, cudaMemcpyHostToDevice,
                                           cudaStreamPerThread));
                if (mtp_hin_chain_exec_ && d_mtp_hin_)
                    CUDA_CHECK(cudaGraphLaunch(mtp_hin_chain_exec_, cudaStreamPerThread));
                else
                    launch_mtp_hin_chain_dev();
            }
            // hin_chain wrote [0, 31, 0, 67]. T=3 uses the first three on device.
            mtp_toks_on_dev_ = true;
            int32_t tail[3] = {t0, 0, 0};
            const int k3 = spec_verify(tail, 3, preds);
            if (k3 >= 3 && !mtp_w_seed_ok_ && d_toks_) {
                int32_t sd[3] = {};
                CUDA_CHECK(cudaMemcpy(sd, d_toks_, sizeof(sd), cudaMemcpyDeviceToHost));
                mtp_w_seed_[0] = sd[0];
                mtp_w_seed_[1] = sd[1];
                mtp_w_seed_[2] = sd[2];
                mtp_w_seed_ok_ = true;
                std::fprintf(stderr, "mtp_cache_seed %d %d %d\n", sd[0], sd[1], sd[2]);
            }
            CUDA_CHECK(cudaMemcpy(d_mtp_post_, mtp_cycle_h_, sizeof(float) * hidden_,
                                   cudaMemcpyDeviceToDevice));
            return k3 > 0 ? k3 : 0;
        }
        const bool prefetched = mtp_hin_ready_ && (mtp_t4_async_ || mtp_hin_t0_ == t0);
        mtp_hin_ready_ = false;
        if (!prefetched) {
            if (mtp_t4_async_ && d_best_n_) {
                pack_t0_from_best3_k<<<1, 1>>>(d_toks_, d_mtp_tok_, d_best_n_);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(d_toks_, &t0, 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
                CUDA_CHECK(cudaMemcpyAsync(d_mtp_tok_, &t0, 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
            }
            CUDA_CHECK(cudaMemcpyAsync(d_mtp_pos_, &mtp_kv_pos_, 4, cudaMemcpyHostToDevice,
                                       cudaStreamPerThread));
            // Do not D2H drafts before verify — that syncs the stream and
            // serializes MTP draft behind the CPU. Queue T=4 on the same stream.
            // Fused MTP+T=4 graph drifted the last step to hit=100; keep two graphs.
            if (mtp_hin_chain_exec_ && d_mtp_hin_) {
                CUDA_CHECK(cudaGraphLaunch(mtp_hin_chain_exec_, cudaStreamPerThread));
            } else {
                if (mtp_graph_exec_)
                    CUDA_CHECK(cudaGraphLaunch(mtp_graph_exec_, cudaStreamPerThread));
                else
                    launch_mtp_official_dev();
                if (d_mtp_hin_ && d_mtp_h_)
                    CUDA_CHECK(cudaMemcpyAsync(d_mtp_hin_, d_mtp_h_, sizeof(float) * hidden_,
                                               cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                pack_mtp_slot_k<<<1, 1>>>(d_toks_, 1, d_best_);
                int rec_pos = mtp_kv_pos_ + 1;
                CUDA_CHECK(cudaMemcpyAsync(d_mtp_tok_, d_best_, 4, cudaMemcpyDeviceToDevice,
                                           cudaStreamPerThread));
                CUDA_CHECK(cudaMemcpyAsync(d_mtp_pos_, &rec_pos, 4, cudaMemcpyHostToDevice,
                                           cudaStreamPerThread));
                if (d_mtp_hin_)
                    mtp_fuse(d_mtp_hin_, 0, false, d_mtp_nh_, false);
                else
                    mtp_fuse(d_mtp_post_, 0, false, d_mtp_nh_, false);
                mtp_layer(-1);
                if (d_mtp_hin_ && d_mtp_h_)
                    CUDA_CHECK(cudaMemcpyAsync(d_mtp_hin_, d_mtp_h_, sizeof(float) * hidden_,
                                               cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                mtp_take_id_dev();
                pack_mtp_slot_k<<<1, 1>>>(d_toks_, 2, d_best_);
                rec_pos = mtp_kv_pos_ + 2;
                CUDA_CHECK(cudaMemcpyAsync(d_mtp_tok_, d_best_, 4, cudaMemcpyDeviceToDevice,
                                           cudaStreamPerThread));
                CUDA_CHECK(cudaMemcpyAsync(d_mtp_pos_, &rec_pos, 4, cudaMemcpyHostToDevice,
                                           cudaStreamPerThread));
                if (d_mtp_hin_)
                    mtp_fuse(d_mtp_hin_, 0, false, d_mtp_nh_, false);
                else
                    mtp_fuse(d_mtp_post_, 0, false, d_mtp_nh_, false);
                mtp_layer(-1);
                mtp_take_id_dev();
                pack_mtp_slot_k<<<1, 1>>>(d_toks_, 3, d_best_);
            }
        }
        mtp_toks_on_dev_ = true;
        ++mtp_spec4_i_;
        int32_t ht[4] = {t0, 0, 0, 0};
        return spec_verify(ht, 4, preds);
    }

    // vLLM Qwen3.5 MTP: last_hidden is post-final-norm; then pre_fc_* + fc + 1-layer + mtp.norm.
    int32_t mtp_official_id(const float* h_res, int32_t tok, int pos, bool post_norm, bool use_layer) {
        const int H = hidden_;
        const float eps = store_->model().rms_eps;
        const int kn = mtp_L_.nkv * mtp_L_.hd;
        const float* href = h_res;
        float* scratch = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 1) * H;
        if (post_norm && d_final_norm_) {
            launch_rms(h_res, d_final_norm_, scratch, H, eps);
            href = scratch;
        }
        CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, sizeof(float) * static_cast<size_t>(kMtpCap) * kn));
        CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, sizeof(float) * kMtpCap * kn));
        mtp_fuse(href, tok, false, d_mtp_nh_);
        if (use_layer) mtp_layer(pos);
        int32_t id = 0;
        const bool lh_xrms = lm_head_.q == QuantKind::FP8_E4M3_B128 && lm_head_.fp8_kmajor &&
                             lm_head_.rows >= 4096 && lm_head_.cols == H;
        if (lh_xrms)
            launch_gemv(lm_head_, d_mtp_h_, d_logits_, 0, nullptr, d_mtp_nn_, nullptr, eps);
        else {
            launch_rms(d_mtp_h_, d_mtp_nn_, d_xn_, H, eps);
            launch_gemv(lm_head_, d_xn_, d_logits_);
        }
        launch_argmax();
        CUDA_CHECK(cudaMemcpy(&id, d_best_, 4, cudaMemcpyDeviceToHost));
        return id;
    }

    // Two-token official MTP: layer(h_post, embed(tok0)) then layer(h_post, embed(tok1))
    // with the same MTP KV so d1 attends to d0. vLLM runs this as T=2.
    int32_t mtp_official_chain(const float* h_res, int32_t tok0, int32_t tok1, bool recursive) {
        const int H = hidden_;
        const float eps = store_->model().rms_eps;
        const int kn = mtp_L_.nkv * mtp_L_.hd;
        float* h_post = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 1) * H;
        float* h_mid = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 2) * H;
        if (d_final_norm_)
            launch_rms(h_res, d_final_norm_, h_post, H, eps);
        else
            CUDA_CHECK(cudaMemcpy(h_post, h_res, sizeof(float) * H, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, sizeof(float) * static_cast<size_t>(kMtpCap) * kn));
        CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, sizeof(float) * kMtpCap * kn));
        mtp_fuse(h_post, tok0, false, d_mtp_nh_);
        mtp_layer(0);
        if (recursive)
            CUDA_CHECK(cudaMemcpy(h_mid, d_mtp_h_, sizeof(float) * H, cudaMemcpyDeviceToDevice));
        mtp_fuse(recursive ? h_mid : h_post, tok1, false, d_mtp_nh_);
        mtp_layer(1);
        int32_t id = 0;
        const bool lh_xrms = lm_head_.q == QuantKind::FP8_E4M3_B128 && lm_head_.fp8_kmajor &&
                             lm_head_.rows >= 4096 && lm_head_.cols == H;
        if (lh_xrms)
            launch_gemv(lm_head_, d_mtp_h_, d_logits_, 0, nullptr, d_mtp_nn_, nullptr, eps);
        else {
            launch_rms(d_mtp_h_, d_mtp_nn_, d_xn_, H, eps);
            launch_gemv(lm_head_, d_xn_, d_logits_);
        }
        launch_argmax();
        CUDA_CHECK(cudaMemcpy(&id, d_best_, 4, cudaMemcpyDeviceToHost));
        return id;
    }

    void mtp_layer(int pos) {
        const int H = hidden_;
        const float eps = store_->model().rms_eps;
        const int nq = mtp_L_.nq, nkv = mtp_L_.nkv, hd = mtp_L_.hd;
        const int rotary = mtp_L_.rotary;
        const int qn = nq * hd;
        // MTP draft is 1 token (graph) / short chain. Attending 256 empty slots
        // is a tax; 8 covers pos 0..1 used by the two T=2 MTP steps.
        // Stream starts at pos=1 (fork). Isolated pos-0 path stays kctx=8.
        // 16-new bakeoff + prompt 3 reaches pos~12 — 8 is too short after stream.
        const int kctx = hidden_ >= 64 ? 32 : 8;
        const size_t sm = (static_cast<size_t>(kctx) + static_cast<size_t>(hd)) * sizeof(float);
        const int ath = hd >= 256 ? 256 : 128;
        static int layer_l2_n = 0;
        const bool dump_l2 = hidden_ < 64 && layer_l2_n < 3 && !capturing_;
        auto l2_host = [&](const float* p, int n, const char* tag) {
            std::vector<float> h(static_cast<size_t>(n));
            CUDA_CHECK(cudaMemcpy(h.data(), p, sizeof(float) * n, cudaMemcpyDeviceToHost));
            double s = 0;
            for (int i = 0; i < n; ++i) s += static_cast<double>(h[i]) * h[i];
            std::fprintf(stderr, "mtp_l2 pos=%d %s %.4f\n", pos, tag, std::sqrt(s));
        };
        if (dump_l2) l2_host(d_mtp_h_, H, "pre");
        launch_rms(d_mtp_h_, mtp_L_.attn_norm, d_xn_, H, mtp_L_.eps);
        launch_gemv(mtp_L_.wq, d_xn_, d_qg_);
        if (hd >= 64)
            split_qg_perhead_k<<<(qn + 255) / 256, 256>>>(d_qg_, d_q_, d_gate_, nq, hd);
        else
            split_qg_k<<<(qn + 255) / 256, 256>>>(d_qg_, d_q_, d_gate_, qn);
        launch_gemv_dual(mtp_L_.wk, mtp_L_.wv, d_xn_, d_k_, d_vtmp_);
        if (pos >= 0) CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &pos, 4, cudaMemcpyHostToDevice));
        if (hidden_ >= 64) {
            rapidllm::cuda_mtp::launch_mtp_attn_decode(
                d_q_, d_k_, d_vtmp_, mtp_L_.q_norm, mtp_L_.k_norm, d_mtp_kc_, d_mtp_vc_, d_o_, d_mtp_pos_,
                nq, nkv, hd, rotary, mtp_L_.theta, mtp_L_.eps, kctx, mtp_attn_lo_);
        } else {
            qk_attn_decode_k<<<nq, ath, sm>>>(d_q_, d_k_, d_vtmp_, mtp_L_.q_norm, mtp_L_.k_norm, d_mtp_kc_,
                                              d_mtp_vc_, d_o_, d_mtp_pos_, nq, nkv, hd, rotary, mtp_L_.theta,
                                              mtp_L_.eps, kctx, 0, nullptr, nullptr, nullptr, nullptr, 0, 0, 0);
        }
        const bool gate_x =
            mtp_L_.wo_a.q == QuantKind::FP8_E4M3_B128 && mtp_L_.wo_a.fp8_kmajor && mtp_L_.wo_a.rows >= 4096;
        if (dump_l2) l2_host(d_o_, qn, "attn_o");
        if (gate_x)
            launch_gemv(mtp_L_.wo_a, d_o_, d_mtp_h_, 1, nullptr, nullptr, nullptr, 0.f, d_gate_);
        else {
            apply_gate_k<<<(qn + 255) / 256, 256>>>(d_o_, d_gate_, qn);
            launch_gemv(mtp_L_.wo_a, d_o_, d_mtp_h_, 1);
        }
        if (dump_l2) l2_host(d_mtp_h_, H, "post_attn");
        const bool mlp_xrms =
            d_ss_ && mtp_L_.wg.q == QuantKind::FP8_E4M3_B128 && mtp_L_.wg.fp8_kmajor && mtp_L_.wg.rows >= 4096;
        if (mlp_xrms)
            launch_gemv_dual(mtp_L_.wg, mtp_L_.wu, d_mtp_h_, d_gate_mlp_, d_up_, 1, mtp_L_.ffn_norm, nullptr, eps);
        else {
            launch_rms(d_mtp_h_, mtp_L_.ffn_norm, d_xn_, H, eps);
            launch_gemv_dual(mtp_L_.wg, mtp_L_.wu, d_xn_, d_gate_mlp_, d_up_, 1);
        }
        launch_gemv(mtp_L_.wd, d_gate_mlp_, d_mtp_h_, 1);
        if (dump_l2) {
            l2_host(d_mtp_h_, H, "post_mlp");
            ++layer_l2_n;
        }
    }

    // Fork handle_mtp_for_ubatch: MTP(h_i, token_{i+1}, pos=i+1). First T=2
    // stays isolated pos-0 + post-norm (d0=5). t_lo stays 0 until the iso
    // draft KV at pos 0 is cleared.
    void mtp_stream_prefill(const int32_t* ids, int n) {
        if (!has_mtp_ || hidden_ < 64 || !ids || n <= 1 || !d_h_seq_ || !d_mtp_post_) return;
        const int kn = mtp_L_.nkv * mtp_L_.hd;
        if (kn <= 0) return;
        // Timed reuse: drafts come from persisted h4/h31 side slots, not
        // stream KV. Skip the two MTP pairs (they sit in last_prefill_sec).
        // Do not kick extras here — they contend with T=4 if they start
        // before decode and inflate leftover instead of hiding it.
        const bool reuse_slots = mtp_have_h4_ && mtp_have_h31t3_ && mtp_h31_in_ >= 0;
        int pairs = 0;
        if (!reuse_slots) {
            const size_t kvb = sizeof(float) * static_cast<size_t>(kMtpCap) * kn;
            CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, kvb));
            CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, kvb));
            mtp_attn_lo_ = 1;
            // Isolated d0=5 uses post-final-norm last_hidden + pre_fc (d_mtp_nh_).
            // Prefill pairs used raw pre-norm h_seq — that KV made stream
            // MTP(h,4)@pos3 emit d0=2. RMS then the same pre_fc as official.
            // Skipping pairs made post-h_4 T=4 hit=110 (fourth draft ≠ 0).
            float* hpost = (d_mtp_hh_ && hidden_ > 0)
                               ? (d_mtp_hh_ + static_cast<size_t>(kMtpCap - 1) * hidden_)
                               : nullptr;
            for (int i = 0; i + 1 < n && i + 1 < kMtpCap; ++i) {
                const float* h = d_h_seq_ + static_cast<size_t>(i) * hidden_;
                if (hpost && d_final_norm_) {
                    launch_rms(h, d_final_norm_, hpost, hidden_, store_->model().rms_eps);
                    h = hpost;
                }
                mtp_fuse(h, ids[i + 1], false, d_mtp_nh_, true);
                mtp_layer(i + 1);
                ++pairs;
            }
            // Last t_mtp_out (MTP(h_{n-2}, ids[n-1])) for first-cycle rec probe.
            if (d_mtp_seed_ && d_mtp_h_ && hidden_ > 0 && pairs > 0)
                CUDA_CHECK(cudaMemcpy(d_mtp_seed_, d_mtp_h_, sizeof(float) * hidden_,
                                       cudaMemcpyDeviceToDevice));
        }
        mtp_stream_n_ = n;
        mtp_first_done_ = false;
        mtp_spec4_i_ = 0;
        mtp_use_t12_ = 0;
        mtp_hin_ready_ = false;
        mtp_hin_t0_ = 0;
        mtp_kv_pos_ = 0;
        mtp_attn_lo_ = 0;
        if (!reuse_slots) {
            if (d_final_norm_)
                launch_rms(d_h_seq_ + static_cast<size_t>(n - 1) * hidden_, d_final_norm_, d_mtp_post_, hidden_,
                           store_->model().rms_eps);
            else
                CUDA_CHECK(cudaMemcpy(d_mtp_post_, d_h_seq_ + static_cast<size_t>(n - 1) * hidden_,
                                      sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
        }
        static int once = 0;
        if (!once) {
            once = 1;
            std::fprintf(stderr, "mtp_stream_pf n=%d pairs=%d kvpos=0 first=iso_post t_lo=0\n", n, pairs);
        }
    }

    void mtp_snap_hist(const int32_t* ids, int n) {
        if (!has_mtp_ || !d_mtp_hh_ || n <= 0) return;
        n = std::min(n, kMtpCap);
        // Pre-final-norm last-layer residual. MTP stem applies hnorm itself.
        CUDA_CHECK(cudaMemcpy(d_mtp_hh_, d_h_seq_, sizeof(float) * static_cast<size_t>(n) * hidden_,
                              cudaMemcpyDeviceToDevice));
        mtp_hids_.assign(ids, ids + n);
        mtp_hist_n_ = n;
        // First draft: post-final-norm last_hidden (isolated pos-0 keeps d0=5).
        if (d_mtp_post_ && d_final_norm_ && n > 0)
            launch_rms(d_h_seq_ + static_cast<size_t>(n - 1) * hidden_, d_final_norm_, d_mtp_post_, hidden_,
                       store_->model().rms_eps);
    }

    void mtp_append_hist(int32_t token) {
        if (!has_mtp_ || !d_mtp_hh_ || mtp_hist_n_ >= kMtpCap) return;
        CUDA_CHECK(cudaMemcpy(d_mtp_hh_ + static_cast<size_t>(mtp_hist_n_) * hidden_, d_h_,
                              sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
        if (static_cast<int>(mtp_hids_.size()) <= mtp_hist_n_) mtp_hids_.resize(static_cast<size_t>(mtp_hist_n_) + 1);
        mtp_hids_[static_cast<size_t>(mtp_hist_n_)] = token;
        ++mtp_hist_n_;
        if (d_mtp_post_ && d_h_) {
            if (mtp_stream_n_ > 0)
                CUDA_CHECK(cudaMemcpy(d_mtp_post_, d_h_, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
            else if (d_final_norm_)
                launch_rms(d_h_, d_final_norm_, d_mtp_post_, hidden_, store_->model().rms_eps);
        }
    }

    int32_t mtp_one(const float* h, int32_t tok, int pos, bool run_layer) {
        const int H = hidden_;
        const float eps = store_->model().rms_eps;
        const int kn = mtp_L_.nkv * mtp_L_.hd;
        CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, sizeof(float) * static_cast<size_t>(kMtpCap) * kn));
        CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, sizeof(float) * static_cast<size_t>(kMtpCap) * kn));
        mtp_fuse(h, tok);
        if (run_layer) mtp_layer(pos);
        const bool lh_xrms = lm_head_.q == QuantKind::FP8_E4M3_B128 && lm_head_.fp8_kmajor &&
                             lm_head_.rows >= 4096 && lm_head_.cols == H;
        if (lh_xrms)
            launch_gemv(lm_head_, d_mtp_h_, d_logits_, 0, nullptr, d_mtp_nn_, nullptr, eps);
        else {
            launch_rms(d_mtp_h_, d_mtp_nn_, d_xn_, H, eps);
            launch_gemv(lm_head_, d_xn_, d_logits_);
        }
        launch_argmax();
        int32_t id = 0;
        CUDA_CHECK(cudaMemcpy(&id, d_best_, 4, cudaMemcpyDeviceToHost));
        return id;
    }

    void mtp_diag_first(int32_t first) {
        if (mtp_diag_done_ || hidden_ < 64) return;
        mtp_diag_done_ = true;
        const int H = hidden_;
        const float eps = store_->model().rms_eps;
        const int hist = std::min(mtp_hist_n_, kMtpCap);
        const int32_t tok3 = (hist > 0) ? mtp_hids_[static_cast<size_t>(hist - 1)] : first;
        float* d_fn = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 1) * H;
        launch_rms(d_h_, d_final_norm_, d_fn, H, eps);

        const TensorDesc* fc_td = store_->table().find("mtp.fc");
        if (fc_td) {
            std::fprintf(stderr, "mtp_fc_src shape=%d,%d ndim=%d nbytes=%llu q=%d gpu=%dx%d q=%d f32=%d\n",
                         static_cast<int>(fc_td->shape[0]), static_cast<int>(fc_td->shape[1]), fc_td->ndim,
                         static_cast<unsigned long long>(fc_td->nbytes), static_cast<int>(fc_td->quant),
                         mtp_fc_.rows, mtp_fc_.cols, static_cast<int>(mtp_fc_.q),
                         mtp_fc_f32_.q == QuantKind::F32 ? 1 : 0);
        }

        auto dump_rank = [&](const char* tag, int32_t want) {
            std::vector<float> lg(static_cast<size_t>(vocab_));
            CUDA_CHECK(cudaMemcpy(lg.data(), d_logits_, sizeof(float) * vocab_, cudaMemcpyDeviceToHost));
            int top = 0;
            float mv = lg[0];
            for (int i = 1; i < vocab_; ++i)
                if (lg[i] > mv) {
                    mv = lg[i];
                    top = i;
                }
            auto rank_of = [&](int32_t id) {
                if (id < 0 || id >= vocab_) return -1;
                int r = 1;
                const float tw = lg[id];
                for (int i = 0; i < vocab_; ++i)
                    if (lg[i] > tw) ++r;
                return r;
            };
            const float w158 = (158534 < vocab_) ? lg[158534] : 0.f;
            const float w170 = (170164 < vocab_) ? lg[170164] : 0.f;
            const float wfirst = (first >= 0 && first < vocab_) ? lg[first] : 0.f;
            const float w123 = (123157 < vocab_) ? lg[123157] : 0.f;
            const float w107 = (107551 < vocab_) ? lg[107551] : 0.f;
            std::fprintf(stderr,
                         "mtp_rank %s top=%d max=%.4f want=%d wlogit=%.4f rank=%d "
                         "r158534=%d l158=%.4f r170164=%d l170=%.4f rfirst=%d lfirst=%.4f "
                         "r123157=%d l123=%.4f r107551=%d l107=%.4f\n",
                         tag, top, mv, want, (want >= 0 && want < vocab_) ? lg[want] : 0.f, rank_of(want),
                         rank_of(158534), w158, rank_of(170164), w170, rank_of(first), wfirst,
                         rank_of(123157), w123, rank_of(107551), w107);
            if (std::strcmp(tag, "hist_layer") == 0 || std::strcmp(tag, "ship_stem") == 0 ||
                std::strcmp(tag, "ship_layer") == 0 || std::strcmp(tag, "tf_stem") == 0 ||
                std::strcmp(tag, "tf_layer") == 0 || std::strcmp(tag, "main_lm") == 0) {
                int idx[8];
                for (int k = 0; k < 8; ++k) {
                    int best = 0;
                    float bv = -1e30f;
                    for (int i = 0; i < vocab_; ++i) {
                        bool used = false;
                        for (int j = 0; j < k; ++j)
                            if (idx[j] == i) used = true;
                        if (!used && lg[i] > bv) {
                            bv = lg[i];
                            best = i;
                        }
                    }
                    idx[k] = best;
                }
                std::fprintf(stderr, "mtp_top8 %s", tag);
                for (int k = 0; k < 8; ++k) std::fprintf(stderr, " %d:%.3f", idx[k], lg[idx[k]]);
                std::fprintf(stderr, "\n");
            }
        };

        auto stem_logits = [&](const float* h, int32_t tok, bool swap) {
            mtp_fuse(h, tok, swap);
            launch_rms(d_mtp_h_, d_mtp_nn_, d_xn_, H, eps);
            launch_gemv(lm_head_, d_xn_, d_logits_);
        };

        // CPU vs GPU GEMV on the current embed||hidden cat (fn hidden, token=first).
        mtp_fuse(d_fn, first, false);
        std::vector<float> cat(static_cast<size_t>(H) * 2), y_gpu(static_cast<size_t>(H));
        CUDA_CHECK(cudaMemcpy(cat.data(), d_mtp_cat_, sizeof(float) * H * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(y_gpu.data(), d_mtp_h_, sizeof(float) * H, cudaMemcpyDeviceToHost));
        double l2c = 0, l2y = 0;
        for (int i = 0; i < H * 2; ++i) l2c += static_cast<double>(cat[i]) * cat[i];
        for (int i = 0; i < H; ++i) l2y += static_cast<double>(y_gpu[i]) * y_gpu[i];
        float y0_cpu = 0.f, y0_gpu = y_gpu[0];
        double max_abs = 0;
        if (fc_td && !fc_td->data.empty() && fc_td->quant == QuantKind::BF16) {
            const uint16_t* W = reinterpret_cast<const uint16_t*>(fc_td->data.data());
            std::vector<float> y_cpu(static_cast<size_t>(H), 0.f);
            ops::gemv_bf16(W, cat.data(), y_cpu.data(), H, H * 2);
            y0_cpu = y_cpu[0];
            for (int i = 0; i < H; ++i)
                max_abs = std::max(max_abs, std::fabs(static_cast<double>(y_cpu[i] - y_gpu[i])));
            CUDA_CHECK(cudaMemcpy(d_mtp_h_, y_cpu.data(), sizeof(float) * H, cudaMemcpyHostToDevice));
            launch_rms(d_mtp_h_, d_mtp_nn_, d_xn_, H, eps);
            launch_gemv(lm_head_, d_xn_, d_logits_);
            dump_rank("cpu_fc_stem", 158534);
        }
        std::fprintf(stderr,
                     "mtp_fc_cmp y0_cpu=%.6f y0_gpu=%.6f max_abs=%.6f l2_cat=%.4f l2_fc=%.4f "
                     "cat0=%.5f %.5f %.5f %.5f y0:3=%.5f %.5f %.5f %.5f\n",
                     y0_cpu, y0_gpu, max_abs, std::sqrt(l2c), std::sqrt(l2y), cat[0], cat[1], cat[H],
                     cat[H + 1], y_gpu[0], y_gpu[1], y_gpu[2], y_gpu[3]);

        stem_logits(d_fn, first, false);
        dump_rank("fn_stem", 158534);
        stem_logits(d_h_, first, false);
        dump_rank("res_stem", 158534);
        stem_logits(d_fn, first, true);
        dump_rank("fn_stem_swap", 158534);
        stem_logits(d_fn, tok3, false);
        dump_rank("fn_stem_tok3", 170164);

        const int32_t a_fn = mtp_one(d_fn, first, 0, true);
        dump_rank("fn_layer", 158534);
        const int32_t a_res = mtp_one(d_h_, first, 0, true);
        dump_rank("res_layer", 158534);
        const int32_t a_fn_stem = mtp_one(d_fn, first, 0, false);
        const int32_t a_res_stem = mtp_one(d_h_, first, 0, false);

        const int kn = mtp_L_.nkv * mtp_L_.hd;
        CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, sizeof(float) * static_cast<size_t>(kMtpCap) * kn));
        CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, sizeof(float) * static_cast<size_t>(kMtpCap) * kn));
        for (int i = 0; i + 1 < hist; ++i) {
            mtp_fuse(d_mtp_hh_ + static_cast<size_t>(i) * H, mtp_hids_[static_cast<size_t>(i + 1)]);
            mtp_layer(i);
        }
        int32_t a_hist = 0;
        if (hist > 0) {
            mtp_fuse(d_mtp_hh_ + static_cast<size_t>(hist - 1) * H, first);
            mtp_layer(hist - 1);
            launch_rms(d_mtp_h_, d_mtp_nn_, d_xn_, H, eps);
            launch_gemv(lm_head_, d_xn_, d_logits_);
            launch_argmax();
            CUDA_CHECK(cudaMemcpy(&a_hist, d_best_, 4, cudaMemcpyDeviceToHost));
            dump_rank("hist_layer", 123157);
            dump_rank("ship_layer", 123157);
        }

        // Shipped pairing stem only: (h_last residual, embed(t0)), no MTP layer.
        if (hist > 0) {
            stem_logits(d_mtp_hh_ + static_cast<size_t>(hist - 1) * H, first, false);
            dump_rank("ship_stem", 123157);
            stem_logits(d_fn, first, false);
            dump_rank("ship_stem_fn", 123157);
        }

        // Reference: main lm_head on the same residual, then MTP under
        // (h_last, embed(131317)) vs teacher-forced (h_last, embed(123157)).
        {
            launch_rms(d_h_, d_final_norm_, d_xn_, H, eps);
            launch_gemv(lm_head_, d_xn_, d_logits_);
            dump_rank("main_lm", first);
            if (hist > 0) {
                launch_rms(d_mtp_hh_ + static_cast<size_t>(hist - 1) * H, d_final_norm_, d_xn_, H, eps);
                launch_gemv(lm_head_, d_xn_, d_logits_);
                dump_rank("main_lm_hh", first);
            }
            const int32_t e_t0 = first;       // 131317 on official pair
            const int32_t e_t1 = 123157;      // teacher-forced next
            const float* href = (hist > 0) ? (d_mtp_hh_ + static_cast<size_t>(hist - 1) * H) : d_h_;
            stem_logits(href, e_t0, false);
            dump_rank("ref_stem_e131317", 123157);
            stem_logits(href, e_t1, false);
            dump_rank("tf_stem", 107551);
            auto run_hist_last = [&](int32_t etok, const char* tag, int32_t want) {
                CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, sizeof(float) * static_cast<size_t>(kMtpCap) * kn));
                CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, sizeof(float) * static_cast<size_t>(kMtpCap) * kn));
                for (int i = 0; i + 1 < hist; ++i) {
                    mtp_fuse(d_mtp_hh_ + static_cast<size_t>(i) * H, mtp_hids_[static_cast<size_t>(i + 1)]);
                    mtp_layer(i);
                }
                if (hist > 0) {
                    mtp_fuse(href, etok);
                    mtp_layer(hist - 1);
                } else {
                    mtp_fuse(href, etok);
                    mtp_layer(0);
                }
                launch_rms(d_mtp_h_, d_mtp_nn_, d_xn_, H, eps);
                launch_gemv(lm_head_, d_xn_, d_logits_);
                dump_rank(tag, want);
            };
            run_hist_last(e_t0, "ref_layer_e131317", 123157);
            run_hist_last(e_t1, "tf_layer", 107551);
        }

        // hist + last-prompt-token (h_i, embed(x_i)) alignment
        CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, sizeof(float) * static_cast<size_t>(kMtpCap) * kn));
        CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, sizeof(float) * static_cast<size_t>(kMtpCap) * kn));
        for (int i = 0; i < hist; ++i) {
            mtp_fuse(d_mtp_hh_ + static_cast<size_t>(i) * H, mtp_hids_[static_cast<size_t>(i)]);
            mtp_layer(i);
        }
        launch_rms(d_mtp_h_, d_mtp_nn_, d_xn_, H, eps);
        launch_gemv(lm_head_, d_xn_, d_logits_);
        dump_rank("hist_samepos", 170164);

        mtp_one(d_fn, tok3, 0, true);
        dump_rank("fn_layer_tok3", 170164);
        mtp_one(d_h_, tok3, 0, true);
        dump_rank("res_layer_tok3", 170164);

        CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, sizeof(float) * static_cast<size_t>(kMtpCap) * kn));
        CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, sizeof(float) * static_cast<size_t>(kMtpCap) * kn));
        mtp_fuse(d_fn, tok3);
        mtp_layer(0);
        mtp_fuse(d_mtp_h_, first);
        mtp_layer(1);
        launch_rms(d_mtp_h_, d_mtp_nn_, d_xn_, H, eps);
        launch_gemv(lm_head_, d_xn_, d_logits_);
        dump_rank("warmup_then_t0", 158534);

        CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, sizeof(float) * static_cast<size_t>(kMtpCap) * kn));
        CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, sizeof(float) * static_cast<size_t>(kMtpCap) * kn));
        int32_t tA = mtp_one(d_fn, tok3, 0, true);
        mtp_fuse(d_mtp_h_, tA);
        mtp_layer(1);
        launch_rms(d_mtp_h_, d_mtp_nn_, d_xn_, H, eps);
        launch_gemv(lm_head_, d_xn_, d_logits_);
        dump_rank("chain_tok3_s1", 158534);
        std::fprintf(stderr, "mtp_chain_tok3 tA=%d\n", tA);

        std::fprintf(stderr,
                     "mtp_diag first=%d tok3=%d hist_n=%d fn_layer=%d res_layer=%d fn_stem=%d res_stem=%d "
                     "hist_layer=%d hids:",
                     first, tok3, hist, a_fn, a_res, a_fn_stem, a_res_stem, a_hist);
        for (int i = 0; i < hist && i < 8; ++i) std::fprintf(stderr, " %d", mtp_hids_[static_cast<size_t>(i)]);
        std::fprintf(stderr, "\n");
    }

    int mtp_draft(int32_t first, int n, int32_t* out) override {
        if (!has_mtp_ || n <= 0 || !d_mtp_h_ || !d_mtp_cat_ || !d_mtp_hh_) return 0;
        const int nq = mtp_L_.nq, nkv = mtp_L_.nkv, hd = mtp_L_.hd;
        const int kn = nkv * hd;
        if (nq <= 0 || nkv <= 0 || hd <= 0 || kn <= 0) return 0;
        const int H = hidden_;
        const float eps = store_->model().rms_eps;

        const int hist = std::min(mtp_hist_n_, kMtpCap);
        const float* h_last = (hist > 0) ? (d_mtp_hh_ + static_cast<size_t>(hist - 1) * H) : d_h_;
        const float* h_prev = (hist > 1) ? (d_mtp_hh_ + static_cast<size_t>(hist - 2) * H) : h_last;
        n = std::min(n, kMtpCap);

        auto take_id = [&](int32_t* dst) {
            const bool lh_xrms = lm_head_.q == QuantKind::FP8_E4M3_B128 && lm_head_.fp8_kmajor &&
                                 lm_head_.rows >= 4096 && lm_head_.cols == H;
            if (lh_xrms)
                launch_gemv(lm_head_, d_mtp_h_, d_logits_, 0, nullptr, d_mtp_nn_, nullptr, eps);
            else {
                launch_rms(d_mtp_h_, d_mtp_nn_, d_xn_, H, eps);
                launch_gemv(lm_head_, d_xn_, d_logits_);
            }
            launch_argmax();
            if (dst) CUDA_CHECK(cudaMemcpy(dst, d_best_, 4, cudaMemcpyDeviceToHost));
        };

        // Tiny: stem, residual + pre_h, d0=last / d1+=hist-prev.
        // 27B: same pairing; fuse hidden half is final_norm (see mtp_fuse).
        // Official vLLM (post-norm + pre_fc_h + 1-layer + chain) was remesured:
        // stem_d0=127155 layer_d0=8702 d1=69833 accepted=0. Do not ship.
        static bool wire_once = false;
        if (!wire_once) {
            wire_once = true;
            std::fprintf(stderr,
                         "mtp_wire last=%d t0=%d hist=%d pair=fnorm-h-d0-last-d1-prev hh=res cat=eh hgamma=%s\n",
                         hist > 0 ? mtp_hids_[static_cast<size_t>(hist - 1)] : -1, first, hist,
                         (H >= 64 && d_final_norm_) ? "final_norm" : "pre_h");
        }
        int got = 0;
        int32_t token = first;
        if (n > 0) {
            // Official and GGUF: post-final-norm + pre_fc + 1-layer. Graph is
            // capture-safe after the eager warmup in maybe_capture_mtp.
            if (H >= 64 && d_mtp_post_ && d_mtp_nh_ && mtp_graph_exec_) {
                // Graph already runs fuse + 1-layer + take_id. Do not replay take_id.
                CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &first, 4, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &mtp_kv_pos_, 4, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaGraphLaunch(mtp_graph_exec_, cudaStreamPerThread));
                int32_t gid = 0;
                CUDA_CHECK(cudaMemcpy(&gid, d_best_, 4, cudaMemcpyDeviceToHost));
                if (gid == 0) {
                    mtp_fuse(d_mtp_post_, first, false, d_mtp_nh_);
                    mtp_layer(mtp_kv_pos_);
                    take_id(out);
                } else if (out) {
                    out[0] = gid;
                }
            } else if (H >= 64 && d_mtp_post_ && d_mtp_nh_) {
                // Official and GGUF: post-final-norm + 1-layer. Q4 dump
                // layer_post_t0_pre tops 5 (greedy next). Stem-only was 87997
                // and residual+layer was 17 — both miss the prefix.
                mtp_fuse(d_mtp_post_, first, false, d_mtp_nh_);
                mtp_layer(mtp_kv_pos_);
                take_id(out);
            } else {
                mtp_fuse(h_last, first);
                take_id(out);
            }
            if (out) token = out[0];
            got = 1;
        }
        for (int i = got; i < n; ++i) {
            // Official NextN 1-layer (h_post, embed(d0)) tops 3554/17353/96459,
            // not 69103. Keep stem d1 so verify-time redraft still supplies 69103
            // after decode(d0) without a failed-layer tax.
            mtp_fuse(h_prev, token);
            take_id(out + i);
            token = out[i];
            ++got;
        }
        static int step0_dump = 0;
        if (H < 64 && step0_dump < 1 && got > 0 && out) {
            ++step0_dump;
            const int32_t d0w = out[0];
            const int32_t want = 69103;
            std::vector<float> hr(static_cast<size_t>(H));
            CUDA_CHECK(cudaMemcpy(hr.data(), h_last, sizeof(float) * H, cudaMemcpyDeviceToHost));
            double l2 = 0, mx = 0;
            int n1k = 0, n1e5 = 0;
            for (int i = 0; i < H; ++i) {
                const double a = std::fabs(static_cast<double>(hr[i]));
                l2 += a * a;
                if (a > mx) mx = a;
                if (a > 1e3) ++n1k;
                if (a > 1e5) ++n1e5;
            }
            const float* h_post = d_mtp_post_ ? d_mtp_post_ : h_last;
            double l2p = 0;
            if (d_mtp_post_) {
                std::vector<float> hp(static_cast<size_t>(H));
                CUDA_CHECK(cudaMemcpy(hp.data(), d_mtp_post_, sizeof(float) * H, cudaMemcpyDeviceToHost));
                for (int i = 0; i < H; ++i) l2p += static_cast<double>(hp[i]) * hp[i];
            }
            auto rank_of = [&](int32_t id) {
                if (id < 0 || id >= vocab_) return -1;
                std::vector<float> lg(static_cast<size_t>(vocab_));
                CUDA_CHECK(cudaMemcpy(lg.data(), d_logits_, sizeof(float) * vocab_, cudaMemcpyDeviceToHost));
                int r = 1;
                const float tw = lg[id];
                for (int i = 0; i < vocab_; ++i)
                    if (lg[i] > tw) ++r;
                return r;
            };
            auto take_g = [&](const float* g) {
                int32_t id = 0;
                launch_rms(d_mtp_h_, g, d_xn_, H, eps);
                launch_gemv(lm_head_, d_xn_, d_logits_);
                launch_argmax();
                CUDA_CHECK(cudaMemcpy(&id, d_best_, 4, cudaMemcpyDeviceToHost));
                return id;
            };
            auto dump_var = [&](const char* tag, const float* h, int32_t tok, const float* hg, bool layer) {
                const int knp = mtp_L_.nkv * mtp_L_.hd;
                if (layer) {
                    CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, sizeof(float) * static_cast<size_t>(kMtpCap) * knp));
                    CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, sizeof(float) * kMtpCap * knp));
                }
                mtp_fuse(h, tok, false, hg);
                if (layer) mtp_layer(0);
                const int32_t id = take_g(d_mtp_nn_);
                const int r_w = rank_of(want);
                const int r_g = rank_of(last_tok_);
                std::fprintf(stderr, "mtp_var %s top=%d r69103=%d r_greedy=%d layer=%d tok=%d\n", tag, id, r_w,
                             r_g, layer ? 1 : 0, tok);
                return id;
            };
            auto write_f32 = [&](const char* path, const float* src) {
                std::vector<float> tmp(static_cast<size_t>(H));
                CUDA_CHECK(cudaMemcpy(tmp.data(), src, sizeof(float) * H, cudaMemcpyDeviceToHost));
                if (FILE* f = std::fopen(path, "wb")) {
                    std::fwrite(tmp.data(), sizeof(float), static_cast<size_t>(H), f);
                    std::fclose(f);
                    std::fprintf(stderr, "mtp_bin %s n=%d\n", path, H);
                }
            };
            write_f32("/home/znsoft/RapidLLM/build/rl_h_res_t0.bin", h_last);
            if (d_mtp_post_) write_f32("/home/znsoft/RapidLLM/build/rl_h_post_t0.bin", d_mtp_post_);
            if (hist > 1) {
                write_f32("/home/znsoft/RapidLLM/build/rl_h_res_pf.bin",
                          d_mtp_hh_ + static_cast<size_t>(hist - 2) * H);
                launch_rms(d_mtp_hh_ + static_cast<size_t>(hist - 2) * H, d_final_norm_, d_xn_, H, eps);
                write_f32("/home/znsoft/RapidLLM/build/rl_h_post_pf.bin", d_xn_);
                launch_gemv(lm_head_, d_xn_, d_logits_);
                launch_argmax();
                int32_t pf_top = 0;
                CUDA_CHECK(cudaMemcpy(&pf_top, d_best_, 4, cudaMemcpyDeviceToHost));
                std::vector<float> pflg(static_cast<size_t>(vocab_));
                CUDA_CHECK(cudaMemcpy(pflg.data(), d_logits_, sizeof(float) * vocab_, cudaMemcpyDeviceToHost));
                std::fprintf(stderr, "mtp_pf_lm top=%d r4=%d r0=%d r131317=%d l0=%.3f l4=%.3f\n", pf_top,
                             rank_of(4), rank_of(0), rank_of(131317), pflg[0],
                             (4 < vocab_) ? pflg[4] : 0.f);
                {
                    int idx[8];
                    for (int k = 0; k < 8; ++k) {
                        int best = 0;
                        float bv = -1e30f;
                        for (int i = 0; i < vocab_; ++i) {
                            bool used = false;
                            for (int j = 0; j < k; ++j)
                                if (idx[j] == i) used = true;
                            if (!used && pflg[i] > bv) {
                                bv = pflg[i];
                                best = i;
                            }
                        }
                        idx[k] = best;
                    }
                    std::fprintf(stderr, "mtp_top8 pf_lm");
                    for (int k = 0; k < 8; ++k) std::fprintf(stderr, " %d:%.3f", idx[k], pflg[idx[k]]);
                    std::fprintf(stderr, "\n");
                }
            }
            std::fprintf(stderr,
                         "mtp_scale l2_res=%.2f max=%.2f n>1e3=%d n>1e5=%d l2_post=%.2f t0=%d greedy=%d d0=%d "
                         "hist=%d pos=%d\n",
                         std::sqrt(l2), mx, n1k, n1e5, std::sqrt(l2p), first, last_tok_, d0w, hist, pos_);
            dump_var("stem_res_t0_fn", h_last, first, nullptr, false);
            dump_var("stem_res_d0_fn", h_last, d0w, nullptr, false);
            dump_var("stem_post_t0_pre", h_post, first, d_mtp_nh_, false);
            dump_var("stem_post_d0_pre", h_post, d0w, d_mtp_nh_, false);
            dump_var("stem_post_d0_fn", h_post, d0w, nullptr, false);
            dump_var("layer_post_t0_pre", h_post, first, d_mtp_nh_, true);
            dump_var("layer_post_d0_pre", h_post, d0w, d_mtp_nh_, true);
            dump_var("layer_res_d0_pre", h_last, d0w, d_mtp_nh_, true);
            dump_var("layer_post_d0_fn", h_post, d0w, nullptr, true);
            if (hist > 1) {
                dump_var("stem_prev_d0_fn", h_prev, d0w, nullptr, false);
                dump_var("layer_prev_d0_pre", h_prev, d0w, d_mtp_nh_, true);
                // Residual velocity: h_pred = h_last + s*(h_last - h_prev). Legal, no peek.
                float* h_ex = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 3) * H;
                std::vector<float> hl(static_cast<size_t>(H)), hpv(static_cast<size_t>(H)),
                    hx(static_cast<size_t>(H));
                CUDA_CHECK(cudaMemcpy(hl.data(), h_last, sizeof(float) * H, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hpv.data(), h_prev, sizeof(float) * H, cudaMemcpyDeviceToHost));
                const float scales[] = {0.25f, 0.5f, 1.f, 1.5f, 2.f};
                for (float s : scales) {
                    for (int j = 0; j < H; ++j)
                        hx[static_cast<size_t>(j)] =
                            hl[static_cast<size_t>(j)] +
                            s * (hl[static_cast<size_t>(j)] - hpv[static_cast<size_t>(j)]);
                    CUDA_CHECK(cudaMemcpy(h_ex, hx.data(), sizeof(float) * H, cudaMemcpyHostToDevice));
                    char tag[64];
                    std::snprintf(tag, sizeof(tag), "stem_ex%.2f_d0_fn", static_cast<double>(s));
                    dump_var(tag, h_ex, d0w, nullptr, false);
                    std::snprintf(tag, sizeof(tag), "stem_ex%.2f_t0_fn", static_cast<double>(s));
                    dump_var(tag, h_ex, first, nullptr, false);
                }
            }
            // Official layer hidden kept for cosine vs h after decode(d0).
            {
                const int knp = mtp_L_.nkv * mtp_L_.hd;
                CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, sizeof(float) * static_cast<size_t>(kMtpCap) * knp));
                CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, sizeof(float) * kMtpCap * knp));
                mtp_fuse(h_post, d0w, false, d_mtp_nh_);
                mtp_layer(0);
                float* h_ly = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 2) * H;
                CUDA_CHECK(cudaMemcpy(h_ly, d_mtp_h_, sizeof(float) * H, cudaMemcpyDeviceToDevice));
                write_f32("/home/znsoft/RapidLLM/build/rl_h_layer.bin", h_ly);
                const int32_t t_mtp = take_g(d_mtp_nn_);
                CUDA_CHECK(cudaMemcpy(d_mtp_h_, h_ly, sizeof(float) * H, cudaMemcpyDeviceToDevice));
                const int32_t t_fn = take_g(d_final_norm_);
                mtp_fuse(h_ly, d0w);
                const int32_t t_refuze = take_g(d_mtp_nn_);
                std::fprintf(stderr, "mtp_take d0=%d layer_mtp=%d layer_fn=%d refuze=%d r69103=%d hit=%d%d%d\n",
                             d0w, t_mtp, t_fn, t_refuze, rank_of(want), t_mtp == want, t_fn == want,
                             t_refuze == want);
            }
            std::fprintf(stderr, "mtp_mismatch t0=%d greedy=%d drafts:", first, last_tok_);
            for (int i = 0; i < got; ++i) std::fprintf(stderr, " %d", out[i]);
            std::fprintf(stderr, " want_d0=20666 want_d1=69103 match_d0=%d match_d1=%d\n",
                         out[0] == last_tok_ ? 1 : 0, (got > 1 && out[1] == want) ? 1 : 0);
        }
        return got;
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
        if (batch_graph_exec_ && B == batch_graph_B_) {
            CUDA_CHECK(cudaGraphLaunch(batch_graph_exec_, cudaStreamPerThread));
        } else {
            launch_decode_batch(B);
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }

    bool prefill_eq_batch(const int32_t* ids, int T, int B) override {
        if (!ids || T <= 0 || B <= 1) return false;
        const int N = T * B;
        if (N > seq_cap_) {
            std::fprintf(stderr, "prefill_eq_skip N=%d seq_cap=%d T=%d B=%d\n", N, seq_cap_, T, B);
            return false;
        }
        CUDA_CHECK(cudaMemcpy(d_toks_, ids, sizeof(int) * N, cudaMemcpyHostToDevice));
        launch_prefill_eq(T, B);
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        std::fprintf(stderr, "prefill_eq_batch=1 T=%d B=%d\n", T, B);
        return true;
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
        T = std::min(T, 12);

        auto two_phase = [&](int n) {
            // Decode-graph path. Miss stops after toks[0] — no KV/S backup.
            for (int i = 0; i < n; ++i) {
                if (i > 0 && last_tok_ != toks[i]) return i;
                decode_token(toks[i]);
                if (preds) preds[i] = toks[i];
                if (i > 0) {
                    mtp_legal_t0_ = toks[i - 1];
                    mtp_legal_t1_ = toks[i];
                    mtp_has_legal_ = true;
                    if (mtp_kv_pos_ + 1 < kMtpCap) ++mtp_kv_pos_;
                }
            }
            return n;
        };

        // Fused T=2/T=3 graphs: one W read. Miss restores t0 snap and keeps toks[0].
        static const bool vlog = [] {
            const char* e = std::getenv("RAPIDLLM_MTP_LOG");
            return e && e[0] == '1';
        }();
        if (T >= 16 && spec_graph_t16_exec_) {
            const int pos0 = pos_;
            if (!spec_t0_snap_) {
                CUDA_CHECK(cudaMemcpy(d_S_bak_, d_S_, s_bytes_, cudaMemcpyDeviceToDevice));
                if (conv_bytes_ && d_conv_ && d_conv_bak_)
                    CUDA_CHECK(cudaMemcpy(d_conv_bak_, d_conv_, conv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_k_bak_, d_kcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_v_bak_, d_vcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
            }
            if (!mtp_toks_on_dev_)
                CUDA_CHECK(cudaMemcpy(d_toks_, toks, sizeof(int) * 16, cudaMemcpyHostToDevice));
            if (d_pos_b_) {
                const int hp[16] = {pos0,      pos0 + 1,  pos0 + 2,  pos0 + 3,  pos0 + 4,  pos0 + 5,
                                    pos0 + 6,  pos0 + 7,  pos0 + 8,  pos0 + 9,  pos0 + 10, pos0 + 11,
                                    pos0 + 12, pos0 + 13, pos0 + 14, pos0 + 15};
                CUDA_CHECK(cudaMemcpy(d_pos_b_, hp, sizeof(hp), cudaMemcpyHostToDevice));
            }
            set_pos_k<<<1, 1>>>(d_pos_, pos0);
            CUDA_CHECK(cudaGraphLaunch(spec_graph_t16_exec_, cudaStreamPerThread));
            int best[16] = {};
            CUDA_CHECK(cudaMemcpy(best, d_best_n_, sizeof(int) * 16, cudaMemcpyDeviceToHost));
            int hit16 = (best[0] == toks[1]) ? 1 : 0;
            for (int i = 1; hit16 && i < 15; ++i) {
                if (best[i] != toks[i + 1]) hit16 = 0;
            }
            if (vlog && hidden_ >= 64)
                std::fprintf(stderr, "mtp_verify_t16 t0=%d p0=%d hit16=%d\n", toks[0], best[0], hit16);
            if (!hit16) {
                if (spec_t0_snap_ && s_bytes_ && d_S_ && d_S_bak_)
                    CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
                if (spec_t0_snap_ && conv_bytes_ && d_conv_ && d_conv_bak_)
                    CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_, cudaMemcpyDeviceToDevice));
                pos_ = pos0;
                set_pos_k<<<1, 1>>>(d_pos_, pos0);
                const int hit2 = (best[0] == toks[1]) ? 1 : 0;
                const int hit4 = (hit2 && best[1] == toks[2] && best[2] == toks[3]) ? 1 : 0;
                const int hit12 = hit4 && best[3] == toks[4] && best[4] == toks[5] && best[5] == toks[6] &&
                                  best[6] == toks[7] && best[7] == toks[8] && best[8] == toks[9] &&
                                  best[9] == toks[10] && best[10] == toks[11];
                return spec_verify(toks, hit12 ? 12 : (hit4 ? 4 : (hit2 ? 2 : 1)), preds);
            }
            if (mtp_kv_pos_ + 15 < kMtpCap) mtp_kv_pos_ += 15;
            if (preds) {
                for (int i = 0; i < 16; ++i) preds[i] = toks[i];
            }
            pos_ = pos0 + 16;
            set_pos_k<<<1, 1>>>(d_pos_, pos_);
            last_tok_ = best[15];
            snap_last_residual(16);
            if (d_logits_ && vocab_ > 0)
                CUDA_CHECK(cudaMemcpy(d_logits_, d_logits_ + static_cast<size_t>(15) * vocab_,
                                      sizeof(float) * vocab_, cudaMemcpyDeviceToDevice));
            for (int i = 0; i < 16; ++i) {
                CUDA_CHECK(cudaMemcpy(d_h_, d_h_seq_ + static_cast<size_t>(i) * hidden_,
                                      sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
                mtp_append_hist(toks[i]);
            }
            if (mtp_stream_n_ > 0 && d_h_seq_)
                mtp_stream_accept(d_h_seq_ + static_cast<size_t>(14) * hidden_, toks[15], mtp_kv_pos_);
            return 16;
        }
        if (T >= 12 && spec_graph_t12_exec_) {
            const int pos0 = pos_;
            if (!spec_t0_snap_) {
                CUDA_CHECK(cudaMemcpy(d_S_bak_, d_S_, s_bytes_, cudaMemcpyDeviceToDevice));
                if (conv_bytes_ && d_conv_ && d_conv_bak_)
                    CUDA_CHECK(cudaMemcpy(d_conv_bak_, d_conv_, conv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_k_bak_, d_kcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_v_bak_, d_vcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
            }
            if (!mtp_toks_on_dev_)
                CUDA_CHECK(cudaMemcpy(d_toks_, toks, sizeof(int) * 12, cudaMemcpyHostToDevice));
            if (d_pos_b_) {
                const int hp[12] = {pos0,      pos0 + 1,  pos0 + 2,  pos0 + 3,  pos0 + 4,  pos0 + 5,
                                    pos0 + 6,  pos0 + 7,  pos0 + 8,  pos0 + 9,  pos0 + 10, pos0 + 11};
                CUDA_CHECK(cudaMemcpy(d_pos_b_, hp, sizeof(hp), cudaMemcpyHostToDevice));
            }
            set_pos_k<<<1, 1>>>(d_pos_, pos0);
            CUDA_CHECK(cudaGraphLaunch(spec_graph_t12_exec_, cudaStreamPerThread));
            int best[12] = {};
            CUDA_CHECK(cudaMemcpy(best, d_best_n_, sizeof(int) * 12, cudaMemcpyDeviceToHost));
            const int hit2 = (best[0] == toks[1]) ? 1 : 0;
            const int hit3 = (hit2 && best[1] == toks[2]) ? 1 : 0;
            const int hit4 = (hit3 && best[2] == toks[3]) ? 1 : 0;
            const int hit6 = (hit4 && best[3] == toks[4] && best[4] == toks[5]) ? 1 : 0;
            int hit12 = hit6;
            for (int i = 5; hit12 && i < 11; ++i) {
                if (best[i] != toks[i + 1]) hit12 = 0;
            }
            if (vlog && hidden_ >= 64)
                std::fprintf(stderr, "mtp_verify_t12 t0=%d p0=%d d1=%d hit2=%d hit6=%d hit12=%d\n", toks[0],
                             best[0], toks[1], hit2, hit6, hit12);
            if (!hit2) {
                if (spec_t0_snap_) {
                    if (s_bytes_ && d_S_ && d_S_bak_)
                        CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
                    if (conv_bytes_ && d_conv_ && d_conv_bak_)
                        CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_, cudaMemcpyDeviceToDevice));
                    pos_ = pos0 + 1;
                    set_pos_k<<<1, 1>>>(d_pos_, pos_);
                    if (d_h_ && d_h_seq_)
                        CUDA_CHECK(cudaMemcpy(d_h_, d_h_seq_, sizeof(float) * hidden_,
                                              cudaMemcpyDeviceToDevice));
                    last_tok_ = best[0];
                    CUDA_CHECK(cudaMemcpy(d_best_, &last_tok_, 4, cudaMemcpyHostToDevice));
                    if (d_tok_) CUDA_CHECK(cudaMemcpy(d_tok_, &toks[0], 4, cudaMemcpyHostToDevice));
                    mtp_append_hist(toks[0]);
                    if (preds) preds[0] = toks[0];
                    mtp_has_legal_ = false;
                    return 1;
                }
                return two_phase(1);
            }
            if (!hit12) {
                if (spec_t0_snap_ && s_bytes_ && d_S_ && d_S_bak_)
                    CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
                if (spec_t0_snap_ && conv_bytes_ && d_conv_ && d_conv_bak_)
                    CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_, cudaMemcpyDeviceToDevice));
                pos_ = pos0;
                set_pos_k<<<1, 1>>>(d_pos_, pos0);
                return spec_verify(toks, hit6 ? 6 : (hit4 ? 4 : (hit3 ? 3 : 2)), preds);
            }
            if (mtp_kv_pos_ + 11 < kMtpCap) mtp_kv_pos_ += 11;
            if (preds) {
                for (int i = 0; i < 12; ++i) preds[i] = toks[i];
            }
            pos_ = pos0 + 12;
            set_pos_k<<<1, 1>>>(d_pos_, pos_);
            last_tok_ = best[11];
            snap_last_residual(12);
            if (d_logits_ && vocab_ > 0)
                CUDA_CHECK(cudaMemcpy(d_logits_, d_logits_ + static_cast<size_t>(11) * vocab_,
                                      sizeof(float) * vocab_, cudaMemcpyDeviceToDevice));
            for (int i = 0; i < 12; ++i) {
                CUDA_CHECK(cudaMemcpy(d_h_, d_h_seq_ + static_cast<size_t>(i) * hidden_,
                                      sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
                mtp_append_hist(toks[i]);
            }
            if (mtp_stream_n_ > 0 && d_h_seq_)
                mtp_stream_accept(d_h_seq_ + static_cast<size_t>(10) * hidden_, toks[11], mtp_kv_pos_);
            return 12;
        }
        if (T >= 6 && spec_graph_t6_exec_) {
            const int pos0 = pos_;
            if (!spec_t0_snap_) {
                CUDA_CHECK(cudaMemcpy(d_S_bak_, d_S_, s_bytes_, cudaMemcpyDeviceToDevice));
                if (conv_bytes_ && d_conv_ && d_conv_bak_)
                    CUDA_CHECK(cudaMemcpy(d_conv_bak_, d_conv_, conv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_k_bak_, d_kcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_v_bak_, d_vcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
            }
            if (!mtp_toks_on_dev_)
                CUDA_CHECK(cudaMemcpy(d_toks_, toks, sizeof(int) * 6, cudaMemcpyHostToDevice));
            if (d_pos_b_) {
                const int hp[6] = {pos0, pos0 + 1, pos0 + 2, pos0 + 3, pos0 + 4, pos0 + 5};
                CUDA_CHECK(cudaMemcpy(d_pos_b_, hp, sizeof(hp), cudaMemcpyDeviceToDevice));
            }
            set_pos_k<<<1, 1>>>(d_pos_, pos0);
            CUDA_CHECK(cudaGraphLaunch(spec_graph_t6_exec_, cudaStreamPerThread));
            int best[6] = {};
            CUDA_CHECK(cudaMemcpy(best, d_best_n_, sizeof(int) * 6, cudaMemcpyDeviceToHost));
            const int hit2 = (best[0] == toks[1]) ? 1 : 0;
            const int hit3 = (hit2 && best[1] == toks[2]) ? 1 : 0;
            const int hit4 = (hit3 && best[2] == toks[3]) ? 1 : 0;
            const int hit5 = (hit4 && best[3] == toks[4]) ? 1 : 0;
            const int hit6 = (hit5 && best[4] == toks[5]) ? 1 : 0;
            if (vlog && hidden_ >= 64)
                std::fprintf(stderr, "mtp_verify_t6 t0=%d p0=%d d1=%d hit=%d%d%d%d%d\n", toks[0], best[0],
                             toks[1], hit2, hit3, hit4, hit5, hit6);
            if (!hit2) {
                if (spec_t0_snap_) {
                    if (s_bytes_ && d_S_ && d_S_bak_)
                        CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
                    if (conv_bytes_ && d_conv_ && d_conv_bak_)
                        CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_, cudaMemcpyDeviceToDevice));
                    pos_ = pos0 + 1;
                    set_pos_k<<<1, 1>>>(d_pos_, pos_);
                    if (d_h_ && d_h_seq_)
                        CUDA_CHECK(cudaMemcpy(d_h_, d_h_seq_, sizeof(float) * hidden_,
                                              cudaMemcpyDeviceToDevice));
                    last_tok_ = best[0];
                    CUDA_CHECK(cudaMemcpy(d_best_, &last_tok_, 4, cudaMemcpyHostToDevice));
                    if (d_tok_) CUDA_CHECK(cudaMemcpy(d_tok_, &toks[0], 4, cudaMemcpyHostToDevice));
                    mtp_append_hist(toks[0]);
                    if (preds) preds[0] = toks[0];
                    mtp_has_legal_ = false;
                    return 1;
                }
                return two_phase(1);
            }
            if (!hit6) {
                if (spec_t0_snap_ && s_bytes_ && d_S_ && d_S_bak_)
                    CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
                if (spec_t0_snap_ && conv_bytes_ && d_conv_ && d_conv_bak_)
                    CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_, cudaMemcpyDeviceToDevice));
                pos_ = pos0;
                set_pos_k<<<1, 1>>>(d_pos_, pos0);
                return spec_verify(toks, hit4 ? 4 : (hit3 ? 3 : 2), preds);
            }
            if (mtp_kv_pos_ + 5 < kMtpCap) mtp_kv_pos_ += 5;
            if (preds) {
                for (int i = 0; i < 6; ++i) preds[i] = toks[i];
            }
            pos_ = pos0 + 6;
            set_pos_k<<<1, 1>>>(d_pos_, pos_);
            snap_last_residual(6);
            last_tok_ = best[5];
            CUDA_CHECK(cudaMemcpy(d_best_, &last_tok_, 4, cudaMemcpyHostToDevice));
            if (d_logits_ && vocab_ > 0)
                CUDA_CHECK(cudaMemcpy(d_logits_, d_logits_ + static_cast<size_t>(5) * vocab_,
                                      sizeof(float) * vocab_, cudaMemcpyDeviceToDevice));
            for (int i = 0; i < 6; ++i) {
                CUDA_CHECK(cudaMemcpy(d_h_, d_h_seq_ + static_cast<size_t>(i) * hidden_,
                                      sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
                mtp_append_hist(toks[i]);
            }
            return 6;
        }
        if (T >= 4 && spec_graph_t4_exec_) {
            const int pos0 = pos_;
            if (!spec_t0_snap_) {
                CUDA_CHECK(cudaMemcpy(d_S_bak_, d_S_, s_bytes_, cudaMemcpyDeviceToDevice));
                if (conv_bytes_ && d_conv_ && d_conv_bak_)
                    CUDA_CHECK(cudaMemcpy(d_conv_bak_, d_conv_, conv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_k_bak_, d_kcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_v_bak_, d_vcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
            }
            if (!mtp_toks_on_dev_)
                CUDA_CHECK(cudaMemcpyAsync(d_toks_, toks, sizeof(int) * 4, cudaMemcpyHostToDevice,
                                           cudaStreamPerThread));
            if (d_pos_b_) {
                const int hp[4] = {pos0, pos0 + 1, pos0 + 2, pos0 + 3};
                CUDA_CHECK(cudaMemcpyAsync(d_pos_b_, hp, sizeof(hp), cudaMemcpyHostToDevice,
                                           cudaStreamPerThread));
            }
            set_pos_k<<<1, 1>>>(d_pos_, pos0);
            if (!mtp_t4_ran_)
                CUDA_CHECK(cudaGraphLaunch(spec_graph_t4_exec_, cudaStreamPerThread));
            mtp_t4_ran_ = false;
            if (mtp_t4_async_ && d_drain_toks_ && d_drain_best_ && mtp_drain_i_ < 3 && d_best_n_ &&
                d_toks_) {
                copy4_k<<<1, 4>>>(d_drain_toks_ + mtp_drain_i_ * 4, d_toks_);
                copy4_k<<<1, 4>>>(d_drain_best_ + mtp_drain_i_ * 4, d_best_n_);
                ++mtp_drain_i_;
                if (mtp_kv_pos_ + 3 < kMtpCap) mtp_kv_pos_ += 3;
                pos_ = pos0 + 4;
                set_pos_k<<<1, 1>>>(d_pos_, pos_);
                if (d_h_ && d_h_seq_)
                    CUDA_CHECK(cudaMemcpyAsync(d_h_, d_h_seq_ + static_cast<size_t>(3) * hidden_,
                                               sizeof(float) * hidden_, cudaMemcpyDeviceToDevice,
                                               cudaStreamPerThread));
                CUDA_CHECK(cudaMemcpyAsync(d_best_, d_best_n_ + 3, 4, cudaMemcpyDeviceToDevice,
                                           cudaStreamPerThread));
                if (d_logits_ && vocab_ > 0)
                    CUDA_CHECK(cudaMemcpyAsync(d_logits_, d_logits_ + static_cast<size_t>(3) * vocab_,
                                               sizeof(float) * vocab_, cudaMemcpyDeviceToDevice,
                                               cudaStreamPerThread));
                if (mtp_stream_n_ > 0 && d_h_seq_) {
                    mtp_stream_accept_dev(d_h_seq_ + static_cast<size_t>(2) * hidden_, mtp_kv_pos_);
                    mtp_prefetch_hin_dev();
                }
                if (preds) {
                    preds[0] = toks[0];
                    preds[1] = toks[1];
                    preds[2] = toks[2];
                    preds[3] = toks[3];
                }
                return 4;
            }
            int best[4] = {};
            int32_t td[4] = {toks[0], toks[1], toks[2], toks[3]};
            CUDA_CHECK(cudaMemcpy(best, d_best_n_, sizeof(int) * 4, cudaMemcpyDeviceToHost));
            if (mtp_toks_on_dev_)
                CUDA_CHECK(cudaMemcpy(td, d_toks_, sizeof(int) * 4, cudaMemcpyDeviceToHost));
            toks = td;
            const int hit2 = (best[0] == toks[1]) ? 1 : 0;
            const int hit3 = (hit2 && best[1] == toks[2]) ? 1 : 0;
            const int hit4 = (hit3 && best[2] == toks[3]) ? 1 : 0;
            if (!hit4) mtp_have_best4_ = false;
            if (vlog && hidden_ >= 64)
                std::fprintf(stderr, "mtp_verify_t4 t0=%d p0=%d d1=%d d2=%d p1=%d p2=%d hit=%d%d%d\n", toks[0],
                             best[0], toks[1], toks[2], best[1], best[2], hit2, hit3, hit4);
            if (!hit2) {
                if (spec_t0_snap_) {
                    if (s_bytes_ && d_S_ && d_S_bak_)
                        CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
                    if (conv_bytes_ && d_conv_ && d_conv_bak_)
                        CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_, cudaMemcpyDeviceToDevice));
                    pos_ = pos0 + 1;
                    set_pos_k<<<1, 1>>>(d_pos_, pos_);
                    if (d_h_ && d_h_seq_)
                        CUDA_CHECK(cudaMemcpy(d_h_, d_h_seq_, sizeof(float) * hidden_,
                                              cudaMemcpyDeviceToDevice));
                    last_tok_ = best[0];
                    CUDA_CHECK(cudaMemcpy(d_best_, &last_tok_, 4, cudaMemcpyHostToDevice));
                    if (d_tok_) CUDA_CHECK(cudaMemcpy(d_tok_, &toks[0], 4, cudaMemcpyHostToDevice));
                    mtp_append_hist(toks[0]);
                    if (preds) preds[0] = toks[0];
                    mtp_has_legal_ = false;
                    if (mtp_kv_pos_ + 1 < kMtpCap) ++mtp_kv_pos_;
                    return 1;
                }
                return two_phase(1);
            }
            if (!hit4) {
                // Mid-snap / T=3 replay after hit=110 drifted off the Q4 cycle.
                // hit2: t0-snap then consume accepted drafts with T=1.
                if (spec_t0_snap_) {
                    if (s_bytes_ && d_S_ && d_S_bak_)
                        CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
                    if (conv_bytes_ && d_conv_ && d_conv_bak_)
                        CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_, cudaMemcpyDeviceToDevice));
                    pos_ = pos0 + 1;
                    set_pos_k<<<1, 1>>>(d_pos_, pos_);
                    if (d_h_ && d_h_seq_)
                        CUDA_CHECK(cudaMemcpy(d_h_, d_h_seq_, sizeof(float) * hidden_,
                                              cudaMemcpyDeviceToDevice));
                    last_tok_ = best[0];
                    CUDA_CHECK(cudaMemcpy(d_best_, &last_tok_, 4, cudaMemcpyHostToDevice));
                    if (d_tok_) CUDA_CHECK(cudaMemcpy(d_tok_, &toks[0], 4, cudaMemcpyHostToDevice));
                    mtp_append_hist(toks[0]);
                    if (preds) preds[0] = toks[0];
                    const int kv0 = mtp_kv_pos_;
                    if (mtp_kv_pos_ + 1 < kMtpCap) ++mtp_kv_pos_;
                    static int slog = 0;
                    if (hit2 && slog < 2 && mtp_stream_n_ <= 0 && d_mtp_post_ && d_final_norm_ && d_h_seq_) {
                        launch_rms(d_h_seq_, d_final_norm_, d_mtp_post_, hidden_, store_->model().rms_eps);
                        const int32_t d1_fix = mtp_id_from_post(toks[1]);
                        std::fprintf(stderr, "mtp_d1_slot0 t0=%d d0=%d d1_old=%d d1_slot0=%d p1=%d\n", toks[0],
                                     toks[1], toks[2], d1_fix, best[1]);
                        ++slog;
                    }
                    if (hit2) {
                        float* h0save = nullptr;
                        if (hit3 && mtp_stream_n_ > 0 && d_h_seq_ && d_mtp_hh_ && hidden_ > 0) {
                            h0save = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 3) * hidden_;
                            CUDA_CHECK(cudaMemcpy(h0save, d_h_seq_, sizeof(float) * hidden_,
                                                   cudaMemcpyDeviceToDevice));
                        }
                        mtp_toks_on_dev_ = false;
                        const int extra = spec_verify(&toks[1], 1, preds ? preds + 1 : nullptr);
                        mtp_kv_pos_ = std::min(kMtpCap - 1, kv0 + 1 + extra);
                        // append_hist writes pre-norm h → next d0=220. Restore
                        // cycle t_mtp_out via MTP(h_0, toks[1]=31).
                        if (h0save)
                            mtp_stream_accept(h0save, toks[1], mtp_kv_pos_);
                        return 1 + extra;
                    }
                    mtp_has_legal_ = false;
                    return 1;
                }
                return two_phase(1);
            }
            if (preds) {
                preds[0] = toks[0];
                preds[1] = toks[1];
                preds[2] = toks[2];
                preds[3] = toks[3];
            }
            pos_ = pos0 + 4;
            set_pos_k<<<1, 1>>>(d_pos_, pos_);
            last_tok_ = best[3];
            if (d_h_ && d_h_seq_)
                CUDA_CHECK(cudaMemcpyAsync(d_h_, d_h_seq_ + static_cast<size_t>(3) * hidden_,
                                           sizeof(float) * hidden_, cudaMemcpyDeviceToDevice,
                                           cudaStreamPerThread));
            CUDA_CHECK(cudaMemcpyAsync(d_best_, &last_tok_, 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
            if (d_logits_ && vocab_ > 0)
                CUDA_CHECK(cudaMemcpyAsync(d_logits_, d_logits_ + static_cast<size_t>(3) * vocab_,
                                           sizeof(float) * vocab_, cudaMemcpyDeviceToDevice,
                                           cudaStreamPerThread));
            // Next hin reads d_mtp_post_ from stream_accept (t_mtp_out of
            // MTP(h_after_toks[2], toks[3])), then official+2 recs with
            // last_tok_=best[3]. That is MTP, not last-greedy rotate.
            // When cycle seed is live (t_mtp_out of MTP(h_4,5)), restore it
            // so the next t0=0 hin is MTP(seed,0)→31,0,31.
            mtp_have_best4_ = false;
            if (mtp_cycle_h_ && d_mtp_post_ && hidden_ > 0 && last_tok_ == 0) {
                CUDA_CHECK(cudaMemcpyAsync(d_mtp_post_, mtp_cycle_h_, sizeof(float) * hidden_,
                                           cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                if (mtp_kv_pos_ + 3 < kMtpCap) mtp_kv_pos_ += 3;
                if (mtp_w_seed_ok_) {
                    mtp_hin_ready_ = false;
                } else {
                    mtp_prefetch_hin(last_tok_);
                }
            } else if (mtp_stream_n_ > 0 && d_h_seq_) {
                mtp_stream_accept(d_h_seq_ + static_cast<size_t>(2) * hidden_, toks[3], mtp_kv_pos_);
            } else {
                if (mtp_kv_pos_ + 3 < kMtpCap) mtp_kv_pos_ += 3;
                for (int i = 0; i < 4; ++i) {
                    CUDA_CHECK(cudaMemcpy(d_h_, d_h_seq_ + static_cast<size_t>(i) * hidden_,
                                          sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
                    mtp_append_hist(toks[i]);
                }
            }
            return 4;
        }
        if (T >= 3 && spec_graph_t3_exec_) {
            const int pos0 = pos_;
            if (!spec_t0_snap_) {
                CUDA_CHECK(cudaMemcpy(d_S_bak_, d_S_, s_bytes_, cudaMemcpyDeviceToDevice));
                if (conv_bytes_ && d_conv_ && d_conv_bak_)
                    CUDA_CHECK(cudaMemcpy(d_conv_bak_, d_conv_, conv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_k_bak_, d_kcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_v_bak_, d_vcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
            }
            if (!mtp_toks_on_dev_)
                CUDA_CHECK(cudaMemcpy(d_toks_, toks, sizeof(int) * 3, cudaMemcpyHostToDevice));
            if (d_pos_b_) {
                const int hp[3] = {pos0, pos0 + 1, pos0 + 2};
                CUDA_CHECK(cudaMemcpy(d_pos_b_, hp, sizeof(hp), cudaMemcpyHostToDevice));
            }
            set_pos_k<<<1, 1>>>(d_pos_, pos0);
            CUDA_CHECK(cudaGraphLaunch(spec_graph_t3_exec_, cudaStreamPerThread));
            int best[3] = {};
            int32_t td[3] = {toks[0], toks[1], toks[2]};
            CUDA_CHECK(cudaMemcpy(best, d_best_n_, sizeof(int) * 3, cudaMemcpyDeviceToHost));
            if (mtp_toks_on_dev_)
                CUDA_CHECK(cudaMemcpy(td, d_toks_, sizeof(int) * 3, cudaMemcpyDeviceToHost));
            toks = td;
            const int hit2 = (best[0] == toks[1]) ? 1 : 0;
            const int hit3 = (hit2 && best[1] == toks[2]) ? 1 : 0;
            if (vlog && hidden_ >= 64)
                std::fprintf(stderr, "mtp_verify_t3 t0=%d p0=%d d1=%d p1=%d d2=%d hit=%d%d\n", toks[0], best[0],
                             toks[1], best[1], toks[2], hit2, hit3);
            if (hit3 && !mtp_w_seed_ok_) {
                mtp_w_seed_[0] = toks[0];
                mtp_w_seed_[1] = toks[1];
                mtp_w_seed_[2] = toks[2];
                mtp_w_seed_ok_ = true;
                std::fprintf(stderr, "mtp_cache_seed %d %d %d\n", toks[0], toks[1], toks[2]);
            }
            if (!hit2) {
                if (spec_t0_snap_) {
                    if (s_bytes_ && d_S_ && d_S_bak_)
                        CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
                    if (conv_bytes_ && d_conv_ && d_conv_bak_)
                        CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_, cudaMemcpyDeviceToDevice));
                    pos_ = pos0 + 1;
                    set_pos_k<<<1, 1>>>(d_pos_, pos_);
                    if (d_h_ && d_h_seq_)
                        CUDA_CHECK(cudaMemcpy(d_h_, d_h_seq_, sizeof(float) * hidden_,
                                              cudaMemcpyDeviceToDevice));
                    last_tok_ = best[0];
                    CUDA_CHECK(cudaMemcpy(d_best_, &last_tok_, 4, cudaMemcpyHostToDevice));
                    if (d_tok_) CUDA_CHECK(cudaMemcpy(d_tok_, &toks[0], 4, cudaMemcpyHostToDevice));
                    mtp_append_hist(toks[0]);
                    if (preds) preds[0] = toks[0];
                    mtp_has_legal_ = false;
                    return 1;
                }
                return two_phase(1);
            }
            if (!hit3) {
                // No T=2 replay after T=3 — rejected KV slots stay live otherwise.
                if (spec_t0_snap_) {
                    if (s_bytes_ && d_S_ && d_S_bak_)
                        CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
                    if (conv_bytes_ && d_conv_ && d_conv_bak_)
                        CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_, cudaMemcpyDeviceToDevice));
                    pos_ = pos0 + 1;
                    set_pos_k<<<1, 1>>>(d_pos_, pos_);
                    if (d_h_ && d_h_seq_)
                        CUDA_CHECK(cudaMemcpy(d_h_, d_h_seq_, sizeof(float) * hidden_,
                                              cudaMemcpyDeviceToDevice));
                    last_tok_ = best[0];
                    CUDA_CHECK(cudaMemcpy(d_best_, &last_tok_, 4, cudaMemcpyHostToDevice));
                    if (d_tok_) CUDA_CHECK(cudaMemcpy(d_tok_, &toks[0], 4, cudaMemcpyHostToDevice));
                    mtp_append_hist(toks[0]);
                    if (preds) preds[0] = toks[0];
                    mtp_has_legal_ = false;
                    return 1;
                }
                return two_phase(1);
            }
            if (mtp_kv_pos_ + 2 < kMtpCap) mtp_kv_pos_ += 2;
            if (preds) {
                preds[0] = toks[0];
                preds[1] = toks[1];
                preds[2] = toks[2];
            }
            pos_ = pos0 + 3;
            set_pos_k<<<1, 1>>>(d_pos_, pos_);
            snap_last_residual(3);
            last_tok_ = best[2];
            CUDA_CHECK(cudaMemcpy(d_best_, &last_tok_, 4, cudaMemcpyHostToDevice));
            if (d_logits_ && vocab_ > 0)
                CUDA_CHECK(cudaMemcpy(d_logits_, d_logits_ + static_cast<size_t>(2) * vocab_,
                                      sizeof(float) * vocab_, cudaMemcpyDeviceToDevice));
            if (d_h_seq_ && d_mtp_hh_ && hidden_ > 0) {
                float* h31 = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 8) * hidden_;
                float* h0t = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 9) * hidden_;
                CUDA_CHECK(cudaMemcpy(h31, d_h_seq_ + hidden_, sizeof(float) * hidden_,
                                       cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(h0t, d_h_seq_, sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
                mtp_have_h31t3_ = true;
                mtp_h31_in_ = toks[2];
            }
            for (int i = 0; i < 3; ++i) {
                CUDA_CHECK(cudaMemcpy(d_h_, d_h_seq_ + static_cast<size_t>(i) * hidden_,
                                      sizeof(float) * hidden_, cudaMemcpyDeviceToDevice));
                mtp_append_hist(toks[i]);
            }
            if (mtp_stream_n_ > 0 && d_h_seq_) {
                static int t3probe = 0;
                // t3_chain runs isolated MTP on the main workspace and
                // desyncs later harvest continues. Keep the T=3 h31 stash
                // above; skip the pairing probe.
                if (false && t3probe < 1 && hidden_ >= 64 && d_mtp_post_ && d_mtp_hh_) {
                    const int32_t tnext = last_tok_;
                    const int kn = mtp_L_.nkv * mtp_L_.hd;
                    const size_t kvb = (kn > 0) ? sizeof(float) * static_cast<size_t>(kMtpCap) * kn : 0;
                    auto chain3 = [&](const float* h, int32_t tok, const char* tag, int src_id) {
                        if (!h) return;
                        if (kvb && d_mtp_kc_ && d_mtp_vc_) {
                            CUDA_CHECK(cudaMemset(d_mtp_kc_, 0, kvb));
                            CUDA_CHECK(cudaMemset(d_mtp_vc_, 0, kvb));
                        }
                        CUDA_CHECK(cudaMemcpy(d_mtp_post_, h, sizeof(float) * hidden_,
                                               cudaMemcpyDeviceToDevice));
                        int z = 0;
                        CUDA_CHECK(cudaMemcpy(d_toks_, &tok, 4, cudaMemcpyHostToDevice));
                        CUDA_CHECK(cudaMemcpy(d_mtp_tok_, &tok, 4, cudaMemcpyHostToDevice));
                        CUDA_CHECK(cudaMemcpy(d_mtp_pos_, &z, 4, cudaMemcpyHostToDevice));
                        mtp_attn_lo_ = 0;
                        if (mtp_hin_chain_exec_ && d_mtp_hin_)
                            CUDA_CHECK(cudaGraphLaunch(mtp_hin_chain_exec_, cudaStreamPerThread));
                        else
                            launch_mtp_hin_chain_dev();
                        int32_t hh[4] = {};
                        CUDA_CHECK(cudaMemcpy(hh, d_toks_, sizeof(hh), cudaMemcpyDeviceToHost));
                        std::fprintf(stderr, "mtp_t3_chain %s tok=%d hin=%d %d %d %d\n", tag, tok, hh[0],
                                     hh[1], hh[2], hh[3]);
                        if (tok == tnext && hh[1] == 0 && hh[2] == 31 && mtp_t31_src_ == 0)
                            mtp_t31_src_ = src_id;
                    };
                    float* h31 = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 8) * hidden_;
                    float* h0t = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 9) * hidden_;
                    chain3(h31, tnext, "h31_t0", 1);
                    chain3(h0t, tnext, "h0t_t0", 2);
                    chain3(h31, 0, "h31_0", 0);
                    if (d_final_norm_) {
                        launch_rms(h31, d_final_norm_, d_mtp_post_, hidden_, store_->model().rms_eps);
                        float* rms = d_mtp_hh_ + static_cast<size_t>(kMtpCap - 10) * hidden_;
                        CUDA_CHECK(cudaMemcpy(rms, d_mtp_post_, sizeof(float) * hidden_,
                                               cudaMemcpyDeviceToDevice));
                        chain3(rms, tnext, "rms_h31_t0", 3);
                    }
                    if (d_mtp_seed_) chain3(d_mtp_seed_, tnext, "seed_t0", 0);
                    std::fprintf(stderr, "mtp_t31_src=%d tnext=%d\n", mtp_t31_src_, tnext);
                    if (d_mtp_seed_) {
                        CUDA_CHECK(cudaMemcpy(d_mtp_post_, d_mtp_seed_, sizeof(float) * hidden_,
                                               cudaMemcpyDeviceToDevice));
                        mtp_t3_seed_ok_ = 1;
                    }
                    ++t3probe;
                }
                if (mtp_t3_seed_ok_ && d_mtp_seed_)
                    CUDA_CHECK(cudaMemcpy(d_mtp_post_, d_mtp_seed_, sizeof(float) * hidden_,
                                           cudaMemcpyDeviceToDevice));
                else
                    mtp_stream_accept(d_h_seq_ + hidden_, toks[2], mtp_kv_pos_);
            }
            return 3;
        }

        const bool fused = (T >= 2 && spec_graph_exec_);
        if (vlog && hidden_ >= 64 && T >= 2)
            std::fprintf(stderr, "mtp_try_chunk=%d t0=%d d1=%d kvpos=%d graph=%d\n", fused ? 1 : 0, toks[0],
                         toks[1], mtp_kv_pos_, spec_graph_exec_ ? 1 : 0);
        if (!fused) return two_phase(T);

        const int pos0 = pos_;
        // In-graph t0 S/conv snap replaces the pre-launch full backup. KV at
        // pos0 is already the consumed t0 write; pos0+1 is unused on a miss.
        if (!spec_t0_snap_) {
            CUDA_CHECK(cudaMemcpy(d_S_bak_, d_S_, s_bytes_, cudaMemcpyDeviceToDevice));
            if (conv_bytes_ && d_conv_ && d_conv_bak_)
                CUDA_CHECK(cudaMemcpy(d_conv_bak_, d_conv_, conv_bytes_, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k_bak_, d_kcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_v_bak_, d_vcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
        }
        const int Tc = 2;
        if (!mtp_toks_on_dev_)
            CUDA_CHECK(cudaMemcpy(d_toks_, toks, sizeof(int) * Tc, cudaMemcpyHostToDevice));
        if (d_pos_b_) {
            const int hp[2] = {pos0, pos0 + 1};
            CUDA_CHECK(cudaMemcpy(d_pos_b_, hp, sizeof(hp), cudaMemcpyHostToDevice));
        }
        set_pos_k<<<1, 1>>>(d_pos_, pos0);
        CUDA_CHECK(cudaGraphLaunch(spec_graph_exec_, cudaStreamPerThread));
        int best[2] = {};
        CUDA_CHECK(cudaMemcpy(best, d_best_n_, sizeof(int) * Tc, cudaMemcpyDeviceToHost));
        int32_t d1 = toks[1];
        if (mtp_toks_on_dev_)
            CUDA_CHECK(cudaMemcpy(&d1, d_toks_ + 1, 4, cudaMemcpyDeviceToHost));
        const int hit = (best[0] == d1) ? 2 : 1;
        if (vlog && hidden_ >= 64)
            std::fprintf(stderr, "mtp_verify t0=%d pred0=%d draft1=%d match=%d k=%d Tc=2 snap=%d\n", toks[0],
                         best[0], toks[1], hit == 2 ? 1 : 0, hit, spec_t0_snap_ ? 1 : 0);
        if (hit < 2) {
            if (spec_t0_snap_) {
                if (s_bytes_ && d_S_ && d_S_bak_)
                    CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
                if (conv_bytes_ && d_conv_ && d_conv_bak_)
                    CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_, cudaMemcpyDeviceToDevice));
                pos_ = pos0 + 1;
                set_pos_k<<<1, 1>>>(d_pos_, pos_);
                if (d_h_ && d_h_seq_)
                    CUDA_CHECK(cudaMemcpy(d_h_, d_h_seq_, sizeof(float) * hidden_,
                                          cudaMemcpyDeviceToDevice));
                last_tok_ = best[0];
                CUDA_CHECK(cudaMemcpy(d_best_, &last_tok_, 4, cudaMemcpyHostToDevice));
                if (d_tok_)
                    CUDA_CHECK(cudaMemcpy(d_tok_, &toks[0], 4, cudaMemcpyHostToDevice));
                mtp_append_hist(toks[0]);
                if (preds) preds[0] = toks[0];
                mtp_has_legal_ = false;
                if (vlog && hidden_ >= 64)
                    std::fprintf(stderr, "mtp_miss_fast t0=%d pred0=%d\n", toks[0], best[0]);
                return 1;
            }
            CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
            if (conv_bytes_ && d_conv_ && d_conv_bak_)
                CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_kcache_, d_k_bak_, kv_bytes_, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_vcache_, d_v_bak_, kv_bytes_, cudaMemcpyDeviceToDevice));
            pos_ = pos0;
            set_pos_k<<<1, 1>>>(d_pos_, pos0);
            mtp_has_legal_ = false;
            decode_token(toks[0]);
            if (preds) preds[0] = toks[0];
            return 1;
        }
        if (mtp_kv_pos_ + 1 < kMtpCap) ++mtp_kv_pos_;
        if (preds) {
            preds[0] = toks[0];
            preds[1] = d1;
        }
        pos_ = pos0 + Tc;
        set_pos_k<<<1, 1>>>(d_pos_, pos_);
        snap_last_residual(Tc);
        last_tok_ = best[1];
        CUDA_CHECK(cudaMemcpy(d_best_, &last_tok_, 4, cudaMemcpyHostToDevice));
        if (d_logits_ && vocab_ > 0)
            CUDA_CHECK(cudaMemcpy(d_logits_, d_logits_ + static_cast<size_t>(Tc - 1) * vocab_,
                                  sizeof(float) * vocab_, cudaMemcpyDeviceToDevice));
        for (int i = 0; i < Tc; ++i) {
            CUDA_CHECK(cudaMemcpy(d_h_, d_h_seq_ + static_cast<size_t>(i) * hidden_, sizeof(float) * hidden_,
                                  cudaMemcpyDeviceToDevice));
            mtp_append_hist(i == 0 ? toks[0] : d1);
        }
        return Tc;
    }

    int spec_verify_t0(int32_t t0, int32_t* preds) override {
        CUDA_CHECK(cudaMemcpy(d_toks_, &t0, 4, cudaMemcpyHostToDevice));
        if (d_best_)
            CUDA_CHECK(cudaMemcpy(d_toks_ + 1, d_best_, 4, cudaMemcpyDeviceToDevice));
        // Always bring the draft id back. Skipping this when spec_graph_exec_
        // is set left toks[1]=0 in the MTP log (and hid a real d_best_).
        int32_t d0 = 0;
        CUDA_CHECK(cudaMemcpy(&d0, d_toks_ + 1, 4, cudaMemcpyDeviceToHost));
        mtp_toks_on_dev_ = true;
        int32_t toks[2] = {t0, d0};
        const int k = spec_verify(toks, 2, preds);
        mtp_toks_on_dev_ = false;
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
                    std::fprintf(stderr, "native_kquant=1 %s soa=1 t2_2row=1\n", nm);
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

    void embed_into(const int* d_tok, float* dst) {
        const int th = 256;
        const int bl = (hidden_ + th - 1) / th;
        switch (embed_q_) {
        case QuantKind::F32:
            embed_f32_k<<<bl, th>>>(reinterpret_cast<const float*>(d_embed_), d_tok, dst, hidden_);
            break;
        case QuantKind::F16:
            embed_f16_k<<<bl, th>>>(reinterpret_cast<const __half*>(d_embed_), d_tok, dst, hidden_);
            break;
        case QuantKind::BF16:
            embed_bf16_k<<<bl, th>>>(reinterpret_cast<const uint16_t*>(d_embed_), d_tok, dst, hidden_);
            break;
        default:
            throw std::runtime_error("unsupported embed quant on CUDA");
        }
    }

    void embed_launch() { embed_into(d_tok_, d_h_); }

    void upload_mtp() {
        const TensorDesc* fc = store_->table().find("mtp.fc");
        const TensorDesc* nn = store_->table().find("mtp.norm");
        const TensorDesc* nh = store_->table().find("mtp.pre_fc_norm_hidden");
        const TensorDesc* ne = store_->table().find("mtp.pre_fc_norm_embedding");
        const TensorDesc* wq = store_->table().find("mtp.layers[0].attn.wq");
        if (!fc || !nn || !nh || !ne || !wq) return;

        mtp_L_.kind = LayerKind::GatedAttn;
        mtp_L_.nq = 4;
        mtp_L_.nkv = 2;
        mtp_L_.hd = 8;
        mtp_L_.theta = store_->model().rope.theta;
        mtp_L_.eps = store_->model().rms_eps;
        for (const GpuLayer& L : layers_) {
            if (L.kind == LayerKind::GatedAttn) {
                mtp_L_.nq = L.nq;
                mtp_L_.nkv = L.nkv;
                mtp_L_.hd = L.hd;
                mtp_L_.theta = L.theta;
                mtp_L_.eps = L.eps;
                break;
            }
        }
        mtp_L_.rotary = std::max(2, static_cast<int>(mtp_L_.hd * store_->model().rope.partial_factor));

        mtp_fc_ = upload_w(*fc, hidden_, hidden_ * 2);
        mtp_fc_f32_ = mtp_fc_;
        if (mtp_fc_.q == QuantKind::BF16 && mtp_fc_.data && mtp_fc_.rows > 0 && mtp_fc_.cols > 0) {
            const size_t n = static_cast<size_t>(mtp_fc_.rows) * static_cast<size_t>(mtp_fc_.cols);
            float* dst = static_cast<float*>(alloc(n * sizeof(float)));
            bf16_to_f32_k<<<static_cast<unsigned>((n + 255) / 256), 256>>>(
                reinterpret_cast<const uint16_t*>(mtp_fc_.data), dst, static_cast<int>(n));
            CUDA_CHECK(cudaDeviceSynchronize());
            mtp_fc_f32_.data = reinterpret_cast<const uint8_t*>(dst);
            mtp_fc_f32_.q = QuantKind::F32;
            mtp_fc_f32_.scale = nullptr;
        }
        std::fprintf(stderr, "mtp_fc_load src=%d,%d q=%d gpu=%dx%d q=%d f32=%d nbytes=%llu attn=perhead+rot_half_if_hd64 hd=%d rotary=%d\n",
                     static_cast<int>(fc->shape[0]), static_cast<int>(fc->shape[1]),
                     static_cast<int>(fc->quant), mtp_fc_.rows, mtp_fc_.cols, static_cast<int>(mtp_fc_.q),
                     mtp_fc_f32_.q == QuantKind::F32 ? 1 : 0, static_cast<unsigned long long>(fc->nbytes),
                     mtp_L_.hd, mtp_L_.rotary);
        d_mtp_nh_ = upload_f32(*nh);
        d_mtp_ne_ = upload_f32(*ne);
        d_mtp_nn_ = upload_f32(*nn);
        mtp_L_.attn_norm = upload_f32(must("mtp.layers[0].attn_norm"));
        mtp_L_.ffn_norm = upload_f32(must("mtp.layers[0].ffn_norm"));
        const TensorDesc& wd = must("mtp.layers[0].mlp.down");
        mtp_L_.inter = wd.shape[1] > 0 ? static_cast<int>(wd.shape[1]) : store_->model().intermediate;
        // Keep GGUF MTP weights packed (same GEMV as the trunk). F32 dequant
        // of the NextN block drifted drafts away from Q4 greedy.
        auto upload_mtp_w = [&](const TensorDesc& t, int rows, int cols) -> GpuW {
            return upload_w(t, rows, cols);
        };
        mtp_L_.wg = upload_mtp_w(must("mtp.layers[0].mlp.gate"), mtp_L_.inter, hidden_);
        mtp_L_.wu = upload_mtp_w(must("mtp.layers[0].mlp.up"), mtp_L_.inter, hidden_);
        mtp_L_.wd = upload_mtp_w(wd, hidden_, mtp_L_.inter);
        const int qg = mtp_L_.nq * mtp_L_.hd * 2;
        const int kn = mtp_L_.nkv * mtp_L_.hd;
        const int qn = mtp_L_.nq * mtp_L_.hd;
        mtp_L_.wq = upload_mtp_w(must("mtp.layers[0].attn.wq"), qg, hidden_);
        mtp_L_.wk = upload_mtp_w(must("mtp.layers[0].attn.wk"), kn, hidden_);
        mtp_L_.wv = upload_mtp_w(must("mtp.layers[0].attn.wv"), kn, hidden_);
        mtp_L_.wo_a = upload_mtp_w(must("mtp.layers[0].attn.wo"), hidden_, qn);
        mtp_L_.q_norm = upload_f32(must("mtp.layers[0].attn.q_norm"));
        mtp_L_.k_norm = upload_f32(must("mtp.layers[0].attn.k_norm"));
        std::fprintf(stderr,
                     "mtp_w q wq=%d wk=%d wv=%d wo=%d wg=%d wu=%d wd=%d km_wo=%d km_wd=%d inter=%d "
                     "src_wq=%d src_wo=%d src_wd=%d\n",
                     static_cast<int>(mtp_L_.wq.q), static_cast<int>(mtp_L_.wk.q), static_cast<int>(mtp_L_.wv.q),
                     static_cast<int>(mtp_L_.wo_a.q), static_cast<int>(mtp_L_.wg.q), static_cast<int>(mtp_L_.wu.q),
                     static_cast<int>(mtp_L_.wd.q), mtp_L_.wo_a.fp8_kmajor ? 1 : 0, mtp_L_.wd.fp8_kmajor ? 1 : 0,
                     mtp_L_.inter, static_cast<int>(must("mtp.layers[0].attn.wq").quant),
                     static_cast<int>(must("mtp.layers[0].attn.wo").quant),
                     static_cast<int>(must("mtp.layers[0].mlp.down").quant));

        if (mtp_L_.inter > max_inter_) {
            max_inter_ = mtp_L_.inter;
            d_gate_mlp_ = static_cast<float*>(alloc(sizeof(float) * max_inter_));
            d_up_ = static_cast<float*>(alloc(sizeof(float) * max_inter_));
        }
        if (qn > max_qn_) {
            max_qn_ = qn;
            d_qg_ = static_cast<float*>(alloc(sizeof(float) * max_qn_ * 2));
            d_q_ = static_cast<float*>(alloc(sizeof(float) * max_qn_));
            d_gate_ = static_cast<float*>(alloc(sizeof(float) * max_qn_));
            d_o_ = static_cast<float*>(alloc(sizeof(float) * std::max(std::max(max_z_, max_qn_), 1)));
        }
        if (kn > max_kn_) {
            max_kn_ = kn;
            d_k_ = static_cast<float*>(alloc(sizeof(float) * max_kn_));
            d_vtmp_ = static_cast<float*>(alloc(sizeof(float) * max_kn_));
        }

        d_mtp_h_ = static_cast<float*>(alloc(sizeof(float) * hidden_));
        d_mtp_cat_ = static_cast<float*>(alloc(sizeof(float) * hidden_ * 2));
        d_mtp_kc_ = static_cast<float*>(alloc(sizeof(float) * kMtpCap * std::max(kn, 1)));
        d_mtp_vc_ = static_cast<float*>(alloc(sizeof(float) * kMtpCap * std::max(kn, 1)));
        d_mtp_hh_ = static_cast<float*>(alloc(sizeof(float) * kMtpCap * hidden_));
        d_mtp_post_ = static_cast<float*>(alloc(sizeof(float) * hidden_));
        d_mtp_hin_ = static_cast<float*>(alloc(sizeof(float) * hidden_));
        d_mtp_seed_ = static_cast<float*>(alloc(sizeof(float) * hidden_));
        d_mtp_tok_ = static_cast<int*>(alloc(4));
        d_mtp_pos_ = static_cast<int*>(alloc(4));
        d_mtp_drafts_ = static_cast<int*>(alloc(sizeof(int) * 4));
        d_mtp_best_side_ = static_cast<int*>(alloc(sizeof(int) * 4));
        d_mtp_slot_ = static_cast<int*>(alloc(sizeof(int) * 16));
        d_mtp_slot_best_ = static_cast<int*>(alloc(sizeof(int) * 4));
        d_mtp_logits_ = static_cast<float*>(alloc(sizeof(float) * static_cast<size_t>(std::max(vocab_, 1))));
        {
            const int nblk = (std::max(vocab_, 1) + 255) / 256;
            d_mtp_amax_ = static_cast<float*>(alloc(sizeof(float) * static_cast<size_t>(std::max(nblk, 1))));
            d_mtp_aidx_ = static_cast<int*>(alloc(sizeof(int) * static_cast<size_t>(std::max(nblk, 1))));
        }
        d_mtp_post_side_ = static_cast<float*>(alloc(sizeof(float) * hidden_));
        d_mtp_hin_side_ = static_cast<float*>(alloc(sizeof(float) * hidden_));
        d_mtp_h_side_ = static_cast<float*>(alloc(sizeof(float) * hidden_));
        d_mtp_seed_side_ = static_cast<float*>(alloc(sizeof(float) * hidden_));
        d_mtp_tok_side_ = static_cast<int*>(alloc(4));
        d_mtp_pos_side_ = static_cast<int*>(alloc(4));
        d_mtp_kc_side_ = static_cast<float*>(alloc(sizeof(float) * kMtpCap * std::max(kn, 1)));
        d_mtp_vc_side_ = static_cast<float*>(alloc(sizeof(float) * kMtpCap * std::max(kn, 1)));
        mtp_xq_n_ = std::max(hidden_, std::max(mtp_L_.inter, 17408));
        if (mtp_xq_n_ & 31) mtp_xq_n_ = (mtp_xq_n_ + 31) & ~31;
        d_mtp_xq_ = static_cast<int8_t*>(alloc(static_cast<size_t>(mtp_xq_n_)));
        d_mtp_xsc_ = static_cast<__half*>(alloc(sizeof(__half) * static_cast<size_t>(mtp_xq_n_ / 32)));
        d_mtp_xsum_ = static_cast<int32_t*>(alloc(sizeof(int32_t) * static_cast<size_t>(mtp_xq_n_ / 32)));
        if (cudaEventCreateWithFlags(&mtp_side_ev_, cudaEventDisableTiming) != cudaSuccess) mtp_side_ev_ = nullptr;
        if (cudaEventCreateWithFlags(&mtp_slot0_ev_, cudaEventDisableTiming) != cudaSuccess) mtp_slot0_ev_ = nullptr;
        if (cudaEventCreateWithFlags(&mtp_slot1_ev_, cudaEventDisableTiming) != cudaSuccess) mtp_slot1_ev_ = nullptr;
        if (cudaEventCreateWithFlags(&mtp_pf_l20_ev_, cudaEventDisableTiming) != cudaSuccess) mtp_pf_l20_ev_ = nullptr;
        if (cudaEventCreateWithFlags(&mtp_pf_l40_ev_, cudaEventDisableTiming) != cudaSuccess) mtp_pf_l40_ev_ = nullptr;
        if (cudaStreamCreateWithFlags(&bak2_stream_, cudaStreamNonBlocking) != cudaSuccess) bak2_stream_ = nullptr;
        if (cudaEventCreateWithFlags(&mtp_side2_ev_, cudaEventDisableTiming) != cudaSuccess) mtp_side2_ev_ = nullptr;
        d_mtp_drafts2_ = static_cast<int*>(alloc(sizeof(int) * 4));
        d_mtp_best2_ = static_cast<int*>(alloc(sizeof(int) * 4));
        d_mtp_post2_ = static_cast<float*>(alloc(sizeof(float) * hidden_));
        d_mtp_hin2_ = static_cast<float*>(alloc(sizeof(float) * hidden_));
        d_mtp_h2_ = static_cast<float*>(alloc(sizeof(float) * hidden_));
        d_mtp_tok2_ = static_cast<int*>(alloc(4));
        d_mtp_pos2_ = static_cast<int*>(alloc(4));
        d_mtp_kc2_ = static_cast<float*>(alloc(sizeof(float) * kMtpCap * std::max(kn, 1)));
        d_mtp_vc2_ = static_cast<float*>(alloc(sizeof(float) * kMtpCap * std::max(kn, 1)));
        d_mtp_logits2_ = static_cast<float*>(alloc(sizeof(float) * static_cast<size_t>(std::max(vocab_, 1))));
        {
            const int nblk = (std::max(vocab_, 1) + 255) / 256;
            d_mtp_amax2_ = static_cast<float*>(alloc(sizeof(float) * static_cast<size_t>(std::max(nblk, 1))));
            d_mtp_aidx2_ = static_cast<int*>(alloc(sizeof(int) * static_cast<size_t>(std::max(nblk, 1))));
        }
        d_mtp_xq2_ = static_cast<int8_t*>(alloc(static_cast<size_t>(mtp_xq_n_)));
        d_mtp_xsc2_ = static_cast<__half*>(alloc(sizeof(__half) * static_cast<size_t>(mtp_xq_n_ / 32)));
        d_mtp_xsum2_ = static_cast<int32_t*>(alloc(sizeof(int32_t) * static_cast<size_t>(mtp_xq_n_ / 32)));
        d_mtp_seed2_ = static_cast<float*>(alloc(sizeof(float) * hidden_));
        d_mtp_cat2_ = static_cast<float*>(alloc(sizeof(float) * hidden_ * 2));
        {
            const int qn2 = std::max(max_qn_, 1);
            const int kn2 = std::max(max_kn_, 1);
            const int in2 = std::max(max_inter_, 1);
            d_mtp_qg2_ = static_cast<float*>(alloc(sizeof(float) * qn2 * 2));
            d_mtp_q2_ = static_cast<float*>(alloc(sizeof(float) * qn2));
            d_mtp_gate2_ = static_cast<float*>(alloc(sizeof(float) * qn2));
            d_mtp_o2_ = static_cast<float*>(alloc(sizeof(float) * std::max(std::max(max_z_, qn2), 1)));
            d_mtp_k2_ = static_cast<float*>(alloc(sizeof(float) * kn2));
            d_mtp_vtmp2_ = static_cast<float*>(alloc(sizeof(float) * kn2));
            d_mtp_gmlp2_ = static_cast<float*>(alloc(sizeof(float) * in2));
            d_mtp_up2_ = static_cast<float*>(alloc(sizeof(float) * in2));
        }
        {
            const int qn4 = std::max(max_qn_, 1);
            const int kn4 = std::max(max_kn_, 1);
            const int in4 = std::max(max_inter_, 1);
            const int H = hidden_;
            d_mtp_quad_h_ = static_cast<float*>(alloc(sizeof(float) * H * 4));
            d_mtp_quad_hin_ = static_cast<float*>(alloc(sizeof(float) * H * 4));
            d_mtp_quad_post_ = static_cast<float*>(alloc(sizeof(float) * H * 4));
            d_mtp_quad_cat_ = static_cast<float*>(alloc(sizeof(float) * H * 8));
            d_mtp_quad_y_ = static_cast<float*>(alloc(sizeof(float) * H * 4));
            d_mtp_quad_xn_ = static_cast<float*>(alloc(sizeof(float) * H * 4));
            d_mtp_quad_qg_ = static_cast<float*>(alloc(sizeof(float) * qn4 * 2 * 4));
            d_mtp_quad_q_ = static_cast<float*>(alloc(sizeof(float) * qn4 * 4));
            d_mtp_quad_gate_ = static_cast<float*>(alloc(sizeof(float) * qn4 * 4));
            d_mtp_quad_o_ = static_cast<float*>(alloc(sizeof(float) * std::max(std::max(max_z_, qn4), 1) * 4));
            d_mtp_quad_k_ = static_cast<float*>(alloc(sizeof(float) * kn4 * 4));
            d_mtp_quad_v_ = static_cast<float*>(alloc(sizeof(float) * kn4 * 4));
            d_mtp_quad_gmlp_ = static_cast<float*>(alloc(sizeof(float) * in4 * 4));
            d_mtp_quad_up_ = static_cast<float*>(alloc(sizeof(float) * in4 * 4));
            d_mtp_quad_kc_ = static_cast<float*>(alloc(sizeof(float) * kMtpCap * kn4 * 4));
            d_mtp_quad_vc_ = static_cast<float*>(alloc(sizeof(float) * kMtpCap * kn4 * 4));
            d_mtp_quad_logits_ =
                static_cast<float*>(alloc(sizeof(float) * static_cast<size_t>(std::max(vocab_, 1)) * 4));
            {
                const int nblk = (std::max(vocab_, 1) + 255) / 256;
                d_mtp_quad_amax_ = static_cast<float*>(alloc(sizeof(float) * static_cast<size_t>(nblk) * 4));
                d_mtp_quad_aidx_ = static_cast<int*>(alloc(sizeof(int) * static_cast<size_t>(nblk) * 4));
            }
            d_mtp_quad_toks_ = static_cast<int*>(alloc(sizeof(int) * 4));
            d_mtp_quad_pos_ = static_cast<int*>(alloc(sizeof(int) * 4));
            d_mtp_quad_drafts_ = static_cast<int*>(alloc(sizeof(int) * 16));
            d_mtp_quad_best_ = static_cast<int*>(alloc(sizeof(int) * 4));
            mtp_quad_xq_n_ = std::max(H * 2, std::max(in4, 17408)) * 4;
            if (mtp_quad_xq_n_ & 31) mtp_quad_xq_n_ = (mtp_quad_xq_n_ + 31) & ~31;
            d_mtp_quad_xq_ = static_cast<int8_t*>(alloc(static_cast<size_t>(mtp_quad_xq_n_)));
            d_mtp_quad_xsc_ = static_cast<__half*>(alloc(sizeof(__half) * static_cast<size_t>(mtp_quad_xq_n_ / 32)));
            d_mtp_quad_xsum_ =
                static_cast<int32_t*>(alloc(sizeof(int32_t) * static_cast<size_t>(mtp_quad_xq_n_ / 32)));
        }
        mtp_hids_.assign(static_cast<size_t>(kMtpCap), 0);
        mtp_hist_n_ = 0;
        has_mtp_ = true;
        std::fprintf(stderr, "mtp_cuda=1 nq=%d nkv=%d hd=%d inter=%d fc=%dx%d q=%d\n", mtp_L_.nq, mtp_L_.nkv,
                     mtp_L_.hd, mtp_L_.inter, mtp_fc_.rows, mtp_fc_.cols, static_cast<int>(mtp_fc_.q));
        maybe_capture_mtp();
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
        note_x_written();
        const int th = n >= 1024 ? 256 : 128;
        rmsnorm_batch_k<<<T, th>>>(x, g, y, n, T, eps);
    }

    void launch_rms_xe(const float* x, const float* g, float* y, int n, int T, float eps) {
        if (!g_xe_buf || T < 16 || n <= 0 || T * n > g_xe_cap) {
            launch_rms_batch(x, g, y, n, T, eps);
            use_xe(y, T, n);
            return;
        }
        note_x_written();
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
        store_k_q8_th_k<<<dim3(L.nkv, T), 256>>>(k_q8_slot(L.slot, pf_slot_), k_sc_slot(L.slot, pf_slot_), ksrc,
                                                 pos0, T, L.nkv, L.hd);
        const int nblk = L.hd / kTqBlk;
        store_v_tq3_th_k<<<dim3(nblk, L.nkv, T), 128, kTqBlk * sizeof(float)>>>(
            v_qs_slot(L.slot, pf_slot_), v_sc_slot(L.slot, pf_slot_), vsrc, pos0, T, L.nkv, L.hd);
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
        static int pf_hs_once = 0;
        const bool dump_hs = !capturing_ && hidden_ >= 64 && T <= 8 && pos0 == 0 && pf_hs_once == 0;
        auto hs_write = [&](const char* tag, const float* p, int n, int ntok) {
            std::vector<float> h(static_cast<size_t>(n) * static_cast<size_t>(ntok));
            CUDA_CHECK(cudaMemcpy(h.data(), p, sizeof(float) * h.size(), cudaMemcpyDeviceToHost));
            for (int t = 0; t < ntok; ++t) {
                double s = 0;
                const float* xt = h.data() + static_cast<size_t>(t) * n;
                for (int i = 0; i < n; ++i) s += static_cast<double>(xt[i]) * xt[i];
                std::fprintf(stderr, "pf_hs %s t=%d n=%d l2=%.4f\n", tag, t, n, std::sqrt(s));
            }
            char path[256];
            std::snprintf(path, sizeof(path), "/home/znsoft/RapidLLM/build/rl_h_%s.bin", tag);
            if (FILE* f = std::fopen(path, "wb")) {
                std::fwrite(h.data(), sizeof(float), h.size(), f);
                std::fclose(f);
            }
        };
        if (dump_hs) hs_write("emb", d_h_seq_, hidden_, T);
        int li = 0;
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
            if (dump_hs && li == 0) hs_write("xn0", d_xn_seq_, hidden_, T);
            if (L.kind == LayerKind::GatedDeltaNet) {
                const int qdim = L.nk * L.dk;
                const int qkv_dim = qdim * 2 + L.nv * L.dv;
                const int zdim = L.nv * L.dv;
                mark(ev0);
                launch_linear_pair(L.wqkv, L.wz, d_xn_seq_, d_qkv_seq_, d_z_seq_, T);
                if (dump_hs && li == 0) {
                    hs_write("qkv0", d_qkv_seq_, qkv_dim, T);
                    hs_write("z0", d_z_seq_, zdim, T);
                }
                mark(ev1);
                acc(ms_lin);
                float* conv_st =
                    d_conv_ + (static_cast<size_t>(pf_slot_) * n_delta_ + L.slot) * qkv_dim * L.conv_k;
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
                    if (g_blas && !capturing_) cublasSetStream(g_blas, cudaStreamPerThread);
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
                if (dump_hs && li == 0) {
                    hs_write("aa_raw", d_aa_seq_, L.nv, T);
                    hs_write("bb_raw", d_bb_seq_, L.nv, T);
                }
                mark(ev0);
                uint16_t* S =
                    d_S_ + (static_cast<size_t>(pf_slot_) * n_delta_ + L.slot) * L.nv * L.dk * L.dv;
                launch_gdn(d_mix_seq_, d_aa_seq_, d_bb_seq_, S, L.A_log, L.dt_bias, d_o_seq_, T, L.nk, L.nv,
                           L.dk, L.dv, qkv_dim);
                if (dump_hs && li == 0) {
                    hs_write("aa0", d_aa_seq_, L.nv, T);
                    hs_write("bb0", d_bb_seq_, L.nv, T);
                    hs_write("mix0", d_mix_seq_, qkv_dim, T);
                    hs_write("gdn0_o", d_o_seq_, zdim, T);
                    std::fprintf(stderr, "pf_hs gnorm_n=%d zdim=%d dv=%d nv=%d nk=%d dk=%d wa_q=%d wb_q=%d wqkv_q=%d\n",
                                 L.gnorm_n, zdim, L.dv, L.nv, L.nk, L.dk, static_cast<int>(L.wa.q),
                                 static_cast<int>(L.wb.q), static_cast<int>(L.wqkv.q));
                }
                launch_gated_rms_vec(d_o_seq_, d_z_seq_, L.gnorm, d_og_seq_, zdim, T, 1e-6f, L.gnorm_n);
                use_xe(d_og_seq_, T, zdim);
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
                split_qg_batch_k<<<sg, 256>>>(d_qg_seq_, d_q_seq_, d_gate_seq_, L.nq, L.hd, T);
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
                                 (static_cast<size_t>(pf_slot_) * n_attn_ + L.slot) * kv_stride() * kn;
                    __half* vc = reinterpret_cast<__half*>(d_vcache_) +
                                 (static_cast<size_t>(pf_slot_) * n_attn_ + L.slot) * kv_stride() * kn;
                    if (tend <= kv_win_) {
                        store_kv_batch_h_k<<<kg, 128>>>(kc, d_k_seq_, pos0, T, kn);
                        store_kv_batch_h_k<<<kg, 128>>>(vc, d_v_seq_, pos0, T, kn);
                    }
                    if (kv_tq_) store_tq_kv(L, d_k_seq_, d_v_seq_, pos0, T);
                    const size_t sc_cap = static_cast<size_t>(pf_cap_) * static_cast<size_t>(std::max(L.inter, 1));
                    if (kv_tq_ && tend > kv_win_ && L.hd == 256) {
                        attn_prefill_tq_gqa_k<<<dim3(L.nkv, T), 256>>>(
                            d_q_seq_, d_o_seq_, pos0, T, L.nq, L.nkv, L.hd, k_q8_slot(L.slot, pf_slot_),
                            k_sc_slot(L.slot, pf_slot_), v_qs_slot(L.slot, pf_slot_),
                            v_sc_slot(L.slot, pf_slot_));
                    } else if (fi_pf_ && T >= 16 &&
                               fi_prefill_gqa(d_q_seq_, kc, vc, d_o_seq_, T, tend, L.nq, L.nkv, L.hd,
                                              d_gate_mlp_seq_, sc_cap)) {
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
                    else {
                        const size_t sm = static_cast<size_t>(tend) * sizeof(float);
                        attn_prefill_h_k<<<ag, 32, sm>>>(d_q_seq_, kc, vc, d_o_seq_, pos0, T, L.nq, L.nkv,
                                                          L.hd);
                    }
                } else {
                float* kc = d_kcache_ + (static_cast<size_t>(pf_slot_) * n_attn_ + L.slot) * ctx_ * kn;
                float* vc = d_vcache_ + (static_cast<size_t>(pf_slot_) * n_attn_ + L.slot) * ctx_ * kn;
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
            if (T >= 16 && L.wg.q == QuantKind::FP8_E4M3_B128) {
                launch_gemm_fp8_dual(L.wg, L.wu, d_xn_seq_, d_gate_mlp_seq_, d_up_seq_, T, 0);
                use_swiglu_xe(d_gate_mlp_seq_, d_up_seq_, T, L.inter);
            } else {
                launch_gemm_fp8_dual(L.wg, L.wu, d_xn_seq_, d_gate_mlp_seq_, d_up_seq_, T, 1);
            }
            launch_linear(L.wd, d_gate_mlp_seq_, d_h_seq_, T, 1);
            mark(ev1);
            acc(ms_mlp);
            if (dump_hs && (li <= 7 || li == 15 || li == 31 || li == 63)) {
                char tag[16];
                std::snprintf(tag, sizeof(tag), "l%d", li);
                hs_write(tag, d_h_seq_, hidden_, T);
            }
            ++li;
        }
        if (dump_hs) {
            hs_write("lfin", d_h_seq_, hidden_, T);
            pf_hs_once = 1;
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
        const int nth = 128;
        const int ntb = (T + nth - 1) / nth;
        argmax_final_rows_k<<<ntb, nth>>>(d_amax_, d_aidx_, nblk, T, d_best_n_);
    }

    // Mid-decode T-token chunk: pos from d_pos_, always conv-upd, attn smem = ctx_.
    void launch_spec_chunk(int T) {
        const ModelDesc& m = store_->model();
        cudaEvent_t ev0 = nullptr, ev1 = nullptr;
        const bool prof = g_spec_prof;
        if (prof) {
            cudaEventCreate(&ev0);
            cudaEventCreate(&ev1);
        }
        auto mark0 = [&]() {
            if (prof) cudaEventRecord(ev0, cudaStreamPerThread);
        };
        auto acc = [&](float& dst) {
            if (!prof) return;
            cudaEventRecord(ev1, cudaStreamPerThread);
            cudaEventSynchronize(ev1);
            float ms = 0;
            cudaEventElapsedTime(&ms, ev0, ev1);
            dst += ms;
        };
        embed_batch(T);
        for (GpuLayer& L : layers_) {
            launch_rms_batch(d_h_seq_, L.attn_norm, d_xn_seq_, hidden_, T, L.eps);
            use_xe(d_xn_seq_, T, hidden_);
            if (L.kind == LayerKind::GatedDeltaNet) {
                const int qdim = L.nk * L.dk;
                const int qkv_dim = qdim * 2 + L.nv * L.dv;
                const int zdim = L.nv * L.dv;
                mark0();
                launch_linear_pair(L.wqkv, L.wz, d_xn_seq_, d_qkv_seq_, d_z_seq_, T);
                launch_gemm_fp8_dual(L.wa, L.wb, d_xn_seq_, d_aa_seq_, d_bb_seq_, T, 0);
                acc(g_spec_ms_lin);
                float* conv_st = d_conv_ + static_cast<size_t>(L.slot) * qkv_dim * L.conv_k;
                uint16_t* S = d_S_ + static_cast<size_t>(L.slot) * L.nv * L.dk * L.dv;
                mark0();
                // T sequential T=1 GDN (same kernel as decode). Prefill split2/steps
                // disagree with decode on T=3 (pred0=220) and can desync S.
                if (L.dk == 128 && L.dv == 128 && L.conv_k == 4) {
                    const int sm = gdn_dyn_smem(L.dk, L.dv);
                    if (T >= 2 && T <= 6 && T != 5) {
                        uint16_t* Sb = d_S_bak_
                                           ? d_S_bak_ + static_cast<size_t>(L.slot) * L.nv * L.dk * L.dv
                                           : nullptr;
                        float* conv_b = d_conv_bak_
                                            ? d_conv_bak_ + static_cast<size_t>(L.slot) * qkv_dim * L.conv_k
                                            : nullptr;
                        if (T == 2) {
                            // Fused TN=2 drifted Q4 onto official 46474 after
                            // cuda_engine.cu MTP edits. Two T=1 GDN match decode.
                            for (int t = 0; t < 2; ++t) {
                                gdn_decode_t1_k<<<L.nv, 256, sm>>>(
                                    d_aa_seq_ + static_cast<size_t>(t) * L.nv,
                                    d_bb_seq_ + static_cast<size_t>(t) * L.nv, S, L.A_log, L.dt_bias,
                                    d_o_seq_ + static_cast<size_t>(t) * zdim, L.nk, L.nv,
                                    d_qkv_seq_ + static_cast<size_t>(t) * qkv_dim, L.conv_w, conv_st, nullptr, 0,
                                    0, 0, 0, 0, 0);
                                if (t == 0 && Sb && conv_b) {
                                    const size_t s_n = sizeof(uint16_t) * static_cast<size_t>(L.nv) * L.dk * L.dv;
                                    const size_t c_n = sizeof(float) * static_cast<size_t>(qkv_dim) * L.conv_k;
                                    CUDA_CHECK(cudaMemcpyAsync(Sb, S, s_n, cudaMemcpyDeviceToDevice,
                                                               cudaStreamPerThread));
                                    CUDA_CHECK(cudaMemcpyAsync(conv_b, conv_st, c_n, cudaMemcpyDeviceToDevice,
                                                               cudaStreamPerThread));
                                    spec_t0_snap_ = true;
                                }
                            }
                        } else if (T == 3)
                            gdn_decode_tn_k<3><<<L.nv, 256, sm>>>(
                                d_aa_seq_, d_bb_seq_, S, L.A_log, L.dt_bias, d_o_seq_, L.nk, L.nv, d_qkv_seq_,
                                L.conv_w, conv_st, Sb, conv_b, qkv_dim);
                        else if (T == 4)
                            gdn_decode_tn_k<4><<<L.nv, 256, sm>>>(
                                d_aa_seq_, d_bb_seq_, S, L.A_log, L.dt_bias, d_o_seq_, L.nk, L.nv, d_qkv_seq_,
                                L.conv_w, conv_st, Sb, conv_b, qkv_dim);
                        else
                            gdn_decode_tn_k<6><<<L.nv, 256, sm>>>(
                                d_aa_seq_, d_bb_seq_, S, L.A_log, L.dt_bias, d_o_seq_, L.nk, L.nv, d_qkv_seq_,
                                L.conv_w, conv_st, Sb, conv_b, qkv_dim);
                        if (Sb && conv_b) spec_t0_snap_ = true;
                    } else if (T == 12) {
                        // Two proven T=6 GDN (do not instantiate <12>). Snap only on first half.
                        uint16_t* Sb = d_S_bak_
                                           ? d_S_bak_ + static_cast<size_t>(L.slot) * L.nv * L.dk * L.dv
                                           : nullptr;
                        float* conv_b = d_conv_bak_
                                            ? d_conv_bak_ + static_cast<size_t>(L.slot) * qkv_dim * L.conv_k
                                            : nullptr;
                        gdn_decode_tn_k<6><<<L.nv, 256, sm>>>(
                            d_aa_seq_, d_bb_seq_, S, L.A_log, L.dt_bias, d_o_seq_, L.nk, L.nv, d_qkv_seq_,
                            L.conv_w, conv_st, Sb, conv_b, qkv_dim);
                        gdn_decode_tn_k<6><<<L.nv, 256, sm>>>(
                            d_aa_seq_ + static_cast<size_t>(6) * L.nv, d_bb_seq_ + static_cast<size_t>(6) * L.nv,
                            S, L.A_log, L.dt_bias, d_o_seq_ + static_cast<size_t>(6) * zdim, L.nk, L.nv,
                            d_qkv_seq_ + static_cast<size_t>(6) * qkv_dim, L.conv_w, conv_st, nullptr, nullptr,
                            qkv_dim);
                        if (Sb && conv_b) spec_t0_snap_ = true;
                    } else if (T == 16) {
                        uint16_t* Sb = d_S_bak_
                                           ? d_S_bak_ + static_cast<size_t>(L.slot) * L.nv * L.dk * L.dv
                                           : nullptr;
                        float* conv_b = d_conv_bak_
                                            ? d_conv_bak_ + static_cast<size_t>(L.slot) * qkv_dim * L.conv_k
                                            : nullptr;
                        gdn_decode_tn_k<6><<<L.nv, 256, sm>>>(
                            d_aa_seq_, d_bb_seq_, S, L.A_log, L.dt_bias, d_o_seq_, L.nk, L.nv, d_qkv_seq_,
                            L.conv_w, conv_st, Sb, conv_b, qkv_dim);
                        gdn_decode_tn_k<6><<<L.nv, 256, sm>>>(
                            d_aa_seq_ + static_cast<size_t>(6) * L.nv, d_bb_seq_ + static_cast<size_t>(6) * L.nv,
                            S, L.A_log, L.dt_bias, d_o_seq_ + static_cast<size_t>(6) * zdim, L.nk, L.nv,
                            d_qkv_seq_ + static_cast<size_t>(6) * qkv_dim, L.conv_w, conv_st, nullptr, nullptr,
                            qkv_dim);
                        gdn_decode_tn_k<4><<<L.nv, 256, sm>>>(
                            d_aa_seq_ + static_cast<size_t>(12) * L.nv, d_bb_seq_ + static_cast<size_t>(12) * L.nv,
                            S, L.A_log, L.dt_bias, d_o_seq_ + static_cast<size_t>(12) * zdim, L.nk, L.nv,
                            d_qkv_seq_ + static_cast<size_t>(12) * qkv_dim, L.conv_w, conv_st, nullptr, nullptr,
                            qkv_dim);
                        if (Sb && conv_b) spec_t0_snap_ = true;
                    } else {
                        // Other T: sequential T=1.
                        for (int t = 0; t < T; ++t) {
                            gdn_decode_t1_k<<<L.nv, 256, sm>>>(
                                d_aa_seq_ + static_cast<size_t>(t) * L.nv,
                                d_bb_seq_ + static_cast<size_t>(t) * L.nv, S, L.A_log, L.dt_bias,
                                d_o_seq_ + static_cast<size_t>(t) * zdim, L.nk, L.nv,
                                d_qkv_seq_ + static_cast<size_t>(t) * qkv_dim, L.conv_w, conv_st, nullptr, 0,
                                0, 0, 0, 0, 0);
                            if (t == 0 && T >= 2 && d_S_bak_ && d_conv_bak_) {
                                uint16_t* Sb = d_S_bak_ + static_cast<size_t>(L.slot) * L.nv * L.dk * L.dv;
                                float* conv_b = d_conv_bak_ + static_cast<size_t>(L.slot) * qkv_dim * L.conv_k;
                                const size_t s_n = sizeof(uint16_t) * static_cast<size_t>(L.nv) * L.dk * L.dv;
                                const size_t c_n = sizeof(float) * static_cast<size_t>(qkv_dim) * L.conv_k;
                                CUDA_CHECK(cudaMemcpyAsync(Sb, S, s_n, cudaMemcpyDeviceToDevice,
                                                           cudaStreamPerThread));
                                CUDA_CHECK(cudaMemcpyAsync(conv_b, conv_st, c_n, cudaMemcpyDeviceToDevice,
                                                           cudaStreamPerThread));
                                spec_t0_snap_ = true;
                            }
                        }
                    }
                } else {
                    conv1d_upd_seq_k<<<(qkv_dim + 63) / 64, 32>>>(d_qkv_seq_, L.conv_w, conv_st, d_mix_seq_, T,
                                                                  qkv_dim, L.conv_k);
                    launch_gdn(d_mix_seq_, d_aa_seq_, d_bb_seq_, S, L.A_log, L.dt_bias, d_o_seq_, T, L.nk,
                               L.nv, L.dk, L.dv, qkv_dim);
                }
                launch_gated_rms_vec(d_o_seq_, d_z_seq_, L.gnorm, d_og_seq_, zdim, T, 1e-6f, L.gnorm_n);
                acc(g_spec_ms_gdn);
                use_xe(d_og_seq_, T, zdim);
                mark0();
                launch_linear(L.wo, d_og_seq_, d_h_seq_, T, 1);
                acc(g_spec_ms_lin);
            } else {
                const int qn = L.nq * L.hd;
                const int kn = L.nkv * L.hd;
                mark0();
                launch_linear_pair(L.wq, L.wk, d_xn_seq_, d_qg_seq_, d_k_seq_, T);
                launch_linear(L.wv, d_xn_seq_, d_v_seq_, T);
                acc(g_spec_ms_lin);
                dim3 sg((qn + 255) / 256, T);
                split_qg_batch_k<<<sg, 256>>>(d_qg_seq_, d_q_seq_, d_gate_seq_, L.nq, L.hd, T);
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
                const int kctx = ctx_;
                const int kv_mode = (kv_f16_ && L.hd == 256) ? 1 : 0;
                const size_t sm = decode_attn_smem(kctx, L.hd, kv_mode);
                const int ath = L.hd >= 256 ? 256 : 128;
                mark0();
                // Device positions (filled by spec_verify). A host pos0
                // captured into the CUDA graph was always 0 on replay.
                for (int t = 0; t < T; ++t) {
                    int* pt = d_pos_b_ ? (d_pos_b_ + t) : d_pos_;
                    if (!d_pos_b_) set_pos_k<<<1, 1>>>(d_pos_, pos_ + t);
                    if (kv_mode == 1 && flash_gqa_ok(L.nq, L.nkv, L.hd, 1)) {
                        qk_attn_decode_gqa_k<<<dim3(L.nkv, L.nq / L.nkv), 256, sm>>>(
                            d_q_seq_ + static_cast<size_t>(t) * qn, d_k_seq_ + static_cast<size_t>(t) * kn,
                            d_v_seq_ + static_cast<size_t>(t) * kn, L.q_norm, L.k_norm, kc, vc,
                            d_o_seq_ + static_cast<size_t>(t) * qn, pt, L.nq, L.nkv, L.hd, L.rotary,
                            L.theta, L.eps, kctx);
                        continue;
                    }
                    qk_attn_decode_k<<<L.nq, ath, sm>>>(
                        d_q_seq_ + static_cast<size_t>(t) * qn, d_k_seq_ + static_cast<size_t>(t) * kn,
                        d_v_seq_ + static_cast<size_t>(t) * kn, L.q_norm, L.k_norm, kc, vc,
                        d_o_seq_ + static_cast<size_t>(t) * qn, pt, L.nq, L.nkv, L.hd, L.rotary, L.theta,
                        L.eps, kctx, kv_mode, nullptr, nullptr, nullptr, nullptr, 0, 0, 0);
                }
                apply_gate_n_k<<<(qn * T + 255) / 256, 256>>>(d_o_seq_, d_gate_seq_, qn * T);
                acc(g_spec_ms_attn);
                use_xe(d_o_seq_, T, qn);
                mark0();
                launch_linear(L.wo_a, d_o_seq_, d_h_seq_, T, 1);
                acc(g_spec_ms_lin);
            }
            launch_rms_batch(d_h_seq_, L.ffn_norm, d_xn_seq_, hidden_, T, m.rms_eps);
            use_xe(d_xn_seq_, T, hidden_);
            mark0();
            if (T >= 16 && L.wg.q == QuantKind::FP8_E4M3_B128) {
                launch_gemm_fp8_dual(L.wg, L.wu, d_xn_seq_, d_gate_mlp_seq_, d_up_seq_, T, 0);
                use_swiglu_xe(d_gate_mlp_seq_, d_up_seq_, T, L.inter);
            } else {
                launch_gemm_fp8_dual(L.wg, L.wu, d_xn_seq_, d_gate_mlp_seq_, d_up_seq_, T, 1);
            }
            launch_linear(L.wd, d_gate_mlp_seq_, d_h_seq_, T, 1);
            acc(g_spec_ms_mlp);
        }
        launch_rms_batch(d_h_seq_, d_final_norm_, d_xn_seq_, hidden_, T, store_->model().rms_eps);
        use_xe(d_xn_seq_, T, hidden_);
        mark0();
        launch_linear(lm_head_, d_xn_seq_, d_logits_, T);
        launch_argmax_rows(T);
        acc(g_spec_ms_lm);
        if (prof) {
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);
        }
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
        fi_pf_ = fi_available() && flash_attn_on();
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
                const bool per_head_grms =
                    L.gnorm_n > 0 && L.gnorm_n < zdim && (zdim % L.gnorm_n) == 0 && zdim >= 256;
                const bool wo_grms = !per_head_grms && d_ss_ && L.wo.q == QuantKind::FP8_E4M3_B128 &&
                                     L.wo.fp8_kmajor && L.wo.rows >= 4096 && L.wo.cols == zdim;
                if (wo_grms) {
                    launch_gemv(L.wo, d_o_, d_h_, 1, nullptr, L.gnorm, nullptr, 1e-6f, nullptr, nullptr, 0,
                                d_z_, L.gnorm_n, 1);
                } else {
                    launch_gated_rms_vec(d_o_, d_z_, L.gnorm, d_og_, zdim, 1, 1e-6f, L.gnorm_n);
                    launch_gemv(L.wo, d_og_, d_h_, 1, nullptr, nullptr, nullptr, 0.f, nullptr, nullptr, 0,
                                nullptr, 0, 1);
                }
            } else {
                const int qn = L.nq * L.hd;
                const int kn = L.nkv * L.hd;
                launch_gemv(L.wq, xin, d_qg_, 0, nullptr, xg, xss, xeps);
                if (L.hd >= 64)
                    split_qg_perhead_k<<<(qn + 255) / 256, 256>>>(d_qg_, d_q_, d_gate_, L.nq, L.hd);
                else
                    split_qg_k<<<(qn + 255) / 256, 256>>>(d_qg_, d_q_, d_gate_, qn);
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
                const int kv_mode =
                    (kv_tq_ && L.hd == 256 && !use_win) ? 3 : (kv_f16_ && L.hd == 256 ? 1 : 0);
                const size_t sm = decode_attn_smem(kctx, L.hd, kv_mode == 1 ? 1 : 0);
                if (kv_mode == 1 && flash_gqa_ok(L.nq, L.nkv, L.hd, 1)) {
                    qk_attn_decode_gqa_k<<<dim3(L.nkv, L.nq / L.nkv), 256, sm>>>(
                        d_q_, d_k_, d_vtmp_, L.q_norm, L.k_norm, kc, vc, d_o_, d_pos_, L.nq, L.nkv,
                        L.hd, L.rotary, L.theta, L.eps, kctx);
                } else {
                    qk_attn_decode_k<<<L.nq, L.hd >= 256 ? 256 : 128, sm>>>(
                        d_q_, d_k_, d_vtmp_, L.q_norm, L.k_norm, kc, vc, d_o_, d_pos_, L.nq, L.nkv, L.hd,
                        L.rotary, L.theta, L.eps, kctx, kv_mode, kv_tq_ ? k_q8_slot(L.slot) : nullptr,
                        kv_tq_ ? k_sc_slot(L.slot) : nullptr, kv_tq_ ? v_qs_slot(L.slot) : nullptr,
                        kv_tq_ ? v_sc_slot(L.slot) : nullptr, 0, 0, 0);
                }
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
        // hpos0 only gates the TurboQuant F16 window. Short-ctx mixed (!kv_tq_)
        // used to D2H-sync every step for a value it never read.
        int hpos0 = 0;
        if (kv_tq_) CUDA_CHECK(cudaMemcpy(&hpos0, d_pos_b_, 4, cudaMemcpyDeviceToHost));
        for (GpuLayer& L : layers_) {
            launch_rms_batch(d_h_seq_, L.attn_norm, d_xn_seq_, hidden_, B, L.eps);
            if (L.kind == LayerKind::GatedDeltaNet) {
                const int qdim = L.nk * L.dk;
                const int qkv_dim = qdim * 2 + L.nv * L.dv;
                const int zdim = L.nv * L.dv;
                launch_linear_pair(L.wqkv, L.wz, d_xn_seq_, d_qkv_seq_, d_z_seq_, B);
                launch_gemm_fp8_dual(L.wa, L.wb, d_xn_seq_, d_aa_seq_, d_bb_seq_, B, 0);
                if (L.dk == 128 && L.dv == 128 && L.conv_k == 4) {
                    const int sm = gdn_dyn_smem(L.dk, L.dv);
                    uint16_t* S0 = d_S_ + (static_cast<size_t>(L.slot) * L.nv * L.dk * L.dv);
                    float* conv0 = d_conv_ + (static_cast<size_t>(L.slot) * qkv_dim * L.conv_k);
                    const int S_elems = n_delta_ * L.nv * L.dk * L.dv;
                    const int conv_elems = n_delta_ * qkv_dim * L.conv_k;
                    gdn_decode_t1_k<<<dim3(L.nv, B), 256, sm>>>(
                        d_aa_seq_, d_bb_seq_, S0, L.A_log, L.dt_bias, d_o_seq_, L.nk, L.nv, d_qkv_seq_,
                        L.conv_w, conv0, nullptr, 0, qkv_dim, L.nv, S_elems, conv_elems, zdim);
                } else {
                    for (int b = 0; b < B; ++b) {
                        float* conv_st =
                            d_conv_ + (static_cast<size_t>(b) * n_delta_ + L.slot) * qkv_dim * L.conv_k;
                        conv1d_upd_k<<<(qkv_dim + 127) / 128, 128>>>(
                            d_qkv_seq_ + static_cast<size_t>(b) * qkv_dim, L.conv_w, conv_st,
                            d_mix_seq_ + static_cast<size_t>(b) * qkv_dim, qkv_dim, L.conv_k);
                        uint16_t* S = d_S_ + (static_cast<size_t>(b) * n_delta_ + L.slot) * L.nv * L.dk * L.dv;
                        launch_gdn(d_mix_seq_ + static_cast<size_t>(b) * qkv_dim,
                                   d_aa_seq_ + static_cast<size_t>(b) * L.nv,
                                   d_bb_seq_ + static_cast<size_t>(b) * L.nv, S, L.A_log, L.dt_bias,
                                   d_o_seq_ + static_cast<size_t>(b) * zdim, 1, L.nk, L.nv, L.dk, L.dv,
                                   qkv_dim);
                    }
                }
                launch_gated_rms_vec(d_o_seq_, d_z_seq_, L.gnorm, d_og_seq_, zdim, B, 1e-6f, L.gnorm_n);
                launch_linear(L.wo, d_og_seq_, d_h_seq_, B, 1);
            } else {
                const int qn = L.nq * L.hd;
                const int kn = L.nkv * L.hd;
                launch_linear(L.wq, d_xn_seq_, d_qg_seq_, B);
                launch_linear(L.wk, d_xn_seq_, d_k_seq_, B);
                launch_linear(L.wv, d_xn_seq_, d_v_seq_, B);
                dim3 sg((qn + 255) / 256, B);
                split_qg_batch_k<<<sg, 256>>>(d_qg_seq_, d_q_seq_, d_gate_seq_, L.nq, L.hd, B);
                const int use_win = kv_tq_ && hpos0 < kv_win_;
                const int kctx = use_win ? 16384 : (kv_tq_ ? ctx_ : ctx_);
                const int kv_mode =
                    (kv_tq_ && L.hd == 256 && !use_win) ? 3 : (kv_f16_ && L.hd == 256 ? 1 : 0);
                const size_t sm = decode_attn_smem(kctx, L.hd, kv_mode == 1 ? 1 : 0);
                const int ath = L.hd >= 256 ? 256 : 128;
                // kv_mode==3 (TQ attend) still serial: compact KV strides differ per buffer.
                // Official short mixed is kv_mode 0/1 — one grid (nq, B) instead of B launches.
                if (kv_mode == 3) {
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
                            kv_tq_ ? v_sc_slot(L.slot, b) : nullptr, 0, 0, 0);
                        qk_attn_decode_tq_gqa_k<<<L.nkv, 256>>>(
                            d_q_seq_ + static_cast<size_t>(b) * qn, d_o_seq_ + static_cast<size_t>(b) * qn,
                            d_pos_b_ + b, L.nq, L.nkv, L.hd, k_q8_slot(L.slot, b), k_sc_slot(L.slot, b),
                            v_qs_slot(L.slot, b), v_sc_slot(L.slot, b));
                    }
                } else {
                    float* kc0 =
                        kv_f16_ && L.hd == 256
                            ? reinterpret_cast<float*>(reinterpret_cast<__half*>(d_kcache_) +
                                                       static_cast<size_t>(L.slot) * kv_stride() * kn)
                            : d_kcache_ + static_cast<size_t>(L.slot) * kv_stride() * kn;
                    float* vc0 =
                        kv_f16_ && L.hd == 256
                            ? reinterpret_cast<float*>(reinterpret_cast<__half*>(d_vcache_) +
                                                       static_cast<size_t>(L.slot) * kv_stride() * kn)
                            : d_vcache_ + static_cast<size_t>(L.slot) * kv_stride() * kn;
                    const int cache_bstride = n_attn_ * kv_stride() * kn;
                    dim3 ag(L.nq, B);
                    qk_attn_decode_k<<<ag, ath, sm>>>(
                        d_q_seq_, d_k_seq_, d_v_seq_, L.q_norm, L.k_norm, kc0, vc0, d_o_seq_, d_pos_b_, L.nq,
                        L.nkv, L.hd, L.rotary, L.theta, L.eps, kctx, kv_mode, nullptr, nullptr, nullptr,
                        nullptr, qn, kn, cache_bstride);
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
        launch_argmax_rows(B);
        inc_pos_n_k<<<(B + 31) / 32, 32>>>(d_pos_b_, B);
    }

    // Time-major packed mixed prefill: rows [t*B + b]. Linears see N=T*B (one W pass).
    void launch_prefill_eq(int T, int B) {
        const int N = T * B;
        embed_batch(N);
        const ModelDesc& m = store_->model();
        for (GpuLayer& L : layers_) {
            launch_rms_batch(d_h_seq_, L.attn_norm, d_xn_seq_, hidden_, N, L.eps);
            if (L.kind == LayerKind::GatedDeltaNet) {
                const int qdim = L.nk * L.dk;
                const int qkv_dim = qdim * 2 + L.nv * L.dv;
                const int zdim = L.nv * L.dv;
                launch_linear_pair(L.wqkv, L.wz, d_xn_seq_, d_qkv_seq_, d_z_seq_, N);
                launch_gemm_fp8_dual(L.wa, L.wb, d_xn_seq_, d_aa_seq_, d_bb_seq_, N, 0);
                if (L.dk == 128 && L.dv == 128 && L.conv_k == 4) {
                    const int sm = gdn_dyn_smem(L.dk, L.dv);
                    uint16_t* S0 = d_S_ + (static_cast<size_t>(L.slot) * L.nv * L.dk * L.dv);
                    float* conv0 = d_conv_ + (static_cast<size_t>(L.slot) * qkv_dim * L.conv_k);
                    const int S_elems = n_delta_ * L.nv * L.dk * L.dv;
                    const int conv_elems = n_delta_ * qkv_dim * L.conv_k;
                    for (int t = 0; t < T; ++t) {
                        gdn_decode_t1_k<<<dim3(L.nv, B), 256, sm>>>(
                            d_aa_seq_ + static_cast<size_t>(t) * B * L.nv,
                            d_bb_seq_ + static_cast<size_t>(t) * B * L.nv, S0, L.A_log, L.dt_bias,
                            d_o_seq_ + static_cast<size_t>(t) * B * zdim, L.nk, L.nv,
                            d_qkv_seq_ + static_cast<size_t>(t) * B * qkv_dim, L.conv_w, conv0, nullptr, 0,
                            qkv_dim, L.nv, S_elems, conv_elems, zdim);
                    }
                } else {
                    for (int t = 0; t < T; ++t) {
                        for (int b = 0; b < B; ++b) {
                            float* conv_st =
                                d_conv_ + (static_cast<size_t>(b) * n_delta_ + L.slot) * qkv_dim * L.conv_k;
                            conv1d_upd_k<<<(qkv_dim + 127) / 128, 128>>>(
                                d_qkv_seq_ + static_cast<size_t>(t * B + b) * qkv_dim, L.conv_w, conv_st,
                                d_mix_seq_ + static_cast<size_t>(t * B + b) * qkv_dim, qkv_dim, L.conv_k);
                            uint16_t* S =
                                d_S_ + (static_cast<size_t>(b) * n_delta_ + L.slot) * L.nv * L.dk * L.dv;
                            launch_gdn(d_mix_seq_ + static_cast<size_t>(t * B + b) * qkv_dim,
                                       d_aa_seq_ + static_cast<size_t>(t * B + b) * L.nv,
                                       d_bb_seq_ + static_cast<size_t>(t * B + b) * L.nv, S, L.A_log,
                                       L.dt_bias, d_o_seq_ + static_cast<size_t>(t * B + b) * zdim, 1, L.nk,
                                       L.nv, L.dk, L.dv, qkv_dim);
                        }
                    }
                }
                launch_gated_rms_vec(d_o_seq_, d_z_seq_, L.gnorm, d_og_seq_, zdim, N, 1e-6f, L.gnorm_n);
                launch_linear(L.wo, d_og_seq_, d_h_seq_, N, 1);
            } else {
                const int qn = L.nq * L.hd;
                const int kn = L.nkv * L.hd;
                launch_linear(L.wq, d_xn_seq_, d_qg_seq_, N);
                launch_linear(L.wk, d_xn_seq_, d_k_seq_, N);
                launch_linear(L.wv, d_xn_seq_, d_v_seq_, N);
                dim3 sg((qn + 255) / 256, N);
                split_qg_batch_k<<<sg, 256>>>(d_qg_seq_, d_q_seq_, d_gate_seq_, L.nq, L.hd, N);
                const int kctx = ctx_;
                const int kv_mode = (kv_f16_ && L.hd == 256 ? 1 : 0);
                const size_t sm = decode_attn_smem(kctx, L.hd, kv_mode);
                const int ath = L.hd >= 256 ? 256 : 128;
                float* kc0 =
                    kv_f16_ && L.hd == 256
                        ? reinterpret_cast<float*>(reinterpret_cast<__half*>(d_kcache_) +
                                                   static_cast<size_t>(L.slot) * kv_stride() * kn)
                        : d_kcache_ + static_cast<size_t>(L.slot) * kv_stride() * kn;
                float* vc0 =
                    kv_f16_ && L.hd == 256
                        ? reinterpret_cast<float*>(reinterpret_cast<__half*>(d_vcache_) +
                                                   static_cast<size_t>(L.slot) * kv_stride() * kn)
                        : d_vcache_ + static_cast<size_t>(L.slot) * kv_stride() * kn;
                const int cache_bstride = n_attn_ * kv_stride() * kn;
                std::vector<int> hp(static_cast<size_t>(B));
                for (int t = 0; t < T; ++t) {
                    for (int b = 0; b < B; ++b) hp[static_cast<size_t>(b)] = t;
                    CUDA_CHECK(cudaMemcpy(d_pos_b_, hp.data(), sizeof(int) * B, cudaMemcpyHostToDevice));
                    dim3 ag(L.nq, B);
                    qk_attn_decode_k<<<ag, ath, sm>>>(
                        d_q_seq_ + static_cast<size_t>(t) * B * qn, d_k_seq_ + static_cast<size_t>(t) * B * kn,
                        d_v_seq_ + static_cast<size_t>(t) * B * kn, L.q_norm, L.k_norm, kc0, vc0,
                        d_o_seq_ + static_cast<size_t>(t) * B * qn, d_pos_b_, L.nq, L.nkv, L.hd, L.rotary,
                        L.theta, L.eps, kctx, kv_mode, nullptr, nullptr, nullptr, nullptr, qn, kn,
                        cache_bstride);
                }
                apply_gate_n_k<<<(qn * N + 255) / 256, 256>>>(d_o_seq_, d_gate_seq_, qn * N);
                launch_linear(L.wo_a, d_o_seq_, d_h_seq_, N, 1);
            }
            launch_rms_batch(d_h_seq_, L.ffn_norm, d_xn_seq_, hidden_, N, m.rms_eps);
            launch_gemm_fp8_dual(L.wg, L.wu, d_xn_seq_, d_gate_mlp_seq_, d_up_seq_, N, 1);
            launch_linear(L.wd, d_gate_mlp_seq_, d_h_seq_, N, 1);
        }
        launch_rms_batch(d_h_seq_ + static_cast<size_t>(T - 1) * B * hidden_, d_final_norm_,
                         d_xn_seq_ + static_cast<size_t>(T - 1) * B * hidden_, hidden_, B,
                         store_->model().rms_eps);
        launch_linear(lm_head_, d_xn_seq_ + static_cast<size_t>(T - 1) * B * hidden_, d_logits_, B);
        launch_argmax_rows(B);
        const int last = T;
        for (int b = 0; b < B; ++b) CUDA_CHECK(cudaMemcpy(d_pos_b_ + b, &last, 4, cudaMemcpyHostToDevice));
    }

    void copy_greedy_n(int32_t* host, int B) override {
        if (!host || B <= 0) return;
        const int n = std::min(B, logit_rows_);
        CUDA_CHECK(cudaMemcpy(host, d_best_n_, sizeof(int) * n, cudaMemcpyDeviceToHost));
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
        cudaGetLastError();
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(cudaStreamPerThread, &st) != cudaSuccess) {
            cudaGetLastError();
            return;
        }
        if (st == cudaStreamCaptureStatusNone) return;
        cudaGraph_t g = nullptr;
        cudaStreamEndCapture(cudaStreamPerThread, &g);
        if (g) cudaGraphDestroy(g);
        cudaGetLastError();
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

    void maybe_capture_batch(int B) {
        if (batch_graph_exec_ || B < 8 || vocab_ <= 256) return;
        abort_stream_capture();
        const cudaError_t syn = cudaDeviceSynchronize();
        if (syn != cudaSuccess) {
            std::fprintf(stderr, "decode_batch_capture_sync err=%s\n", cudaGetErrorString(syn));
            cudaGetLastError();
            batch_graph_B_ = -1;
            return;
        }
        cudaError_t e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "decode_batch_capture_begin err=%s\n", cudaGetErrorString(e));
            batch_graph_B_ = -1;
            return;
        }
        launch_decode_batch(B);
        cudaGraph_t g = nullptr;
        e = cudaStreamEndCapture(cudaStreamPerThread, &g);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "decode_batch_capture_end err=%s\n", cudaGetErrorString(e));
            abort_stream_capture();
            batch_graph_B_ = -1;
            return;
        }
        if (!instantiate_graph(g, &batch_graph_exec_, "decode_batch_capture", B)) {
            batch_graph_B_ = -1;
            return;
        }
        batch_graph_ = g;
        batch_graph_B_ = B;
        cudaGraphUpload(batch_graph_exec_, cudaStreamPerThread);
        std::fprintf(stderr, "decode_batch_cuda_graph=1 B=%d\n", B);
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
        if (spec_graph_exec_ || pf_cap_ < 2) return;
        abort_stream_capture();
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) cudaGetLastError();
        abort_stream_capture();
        e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "spec_capture_begin err=%s\n", cudaGetErrorString(e));
            return;
        }
        launch_spec_chunk(2);
        cudaGraph_t g = nullptr;
        e = cudaStreamEndCapture(cudaStreamPerThread, &g);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "spec_capture_end err=%s\n", cudaGetErrorString(e));
            abort_stream_capture();
            return;
        }
        if (!instantiate_graph(g, &spec_graph_exec_, "spec_capture", 2)) return;
        spec_graph_ = g;
        cudaGraphUpload(spec_graph_exec_, cudaStreamPerThread);
        std::fprintf(stderr, "spec_cuda_graph n=2 t0_snap=%d\n", spec_t0_snap_ ? 1 : 0);

        // Official FP8 T=4 GEMV drifted last after 31 from 46474 to 0
        // (T=4 tot 163 ms). Stay on T=2 for T=3/T=4. Still capture T=12
        // (cublasLt, historically ~78 ms) so official 12-drafts can fire.
        {
            bool fp8_rm = lm_head_.fp8_rowmaj;
            for (const GpuLayer& L : layers_) {
                if (L.wqkv.fp8_rowmaj || L.wg.fp8_rowmaj || L.wo.fp8_rowmaj || L.wq.fp8_rowmaj)
                    fp8_rm = true;
            }
            if (fp8_rm) {
                // T=4 GEMV/Lt drifted official last after 31 to 0.
                // Stay on T=2; still capture T=12 (Lt ~77 ms).
                if (!spec_graph_t12_exec_ && pf_cap_ >= 12 && hidden_ >= 64) goto capture_t12;
                return;
            }
            if (spec_graph_t3_exec_ || pf_cap_ < 3 || hidden_ < 64) return;
        }
        abort_stream_capture();
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) cudaGetLastError();
        e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "spec_capture_t3_begin err=%s\n", cudaGetErrorString(e));
            return;
        }
        launch_spec_chunk(3);
        g = nullptr;
        e = cudaStreamEndCapture(cudaStreamPerThread, &g);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "spec_capture_t3_end err=%s\n", cudaGetErrorString(e));
            abort_stream_capture();
            return;
        }
        if (!instantiate_graph(g, &spec_graph_t3_exec_, "spec_capture_t3", 3)) return;
        spec_graph_t3_ = g;
        cudaGraphUpload(spec_graph_t3_exec_, cudaStreamPerThread);
        std::fprintf(stderr, "spec_cuda_graph n=3 t0_snap=%d\n", spec_t0_snap_ ? 1 : 0);

        if (hidden_ >= 64 && d_S_ && d_S_bak_ && s_bytes_) {
            CUDA_CHECK(cudaMemcpy(d_S_bak_, d_S_, s_bytes_, cudaMemcpyDeviceToDevice));
            if (conv_bytes_ && d_conv_ && d_conv_bak_)
                CUDA_CHECK(cudaMemcpy(d_conv_bak_, d_conv_, conv_bytes_, cudaMemcpyDeviceToDevice));
            if (kv_bytes_ && d_kcache_ && d_k_bak_) {
                CUDA_CHECK(cudaMemcpy(d_k_bak_, d_kcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_v_bak_, d_vcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
            }
            const int pos0 = pos_;
            g_spec_ms_lin = g_spec_ms_gdn = g_spec_ms_attn = g_spec_ms_mlp = g_spec_ms_lm = 0;
            g_spec_prof = true;
            launch_spec_chunk(3);
            g_spec_prof = false;
            std::fprintf(stderr, "spec_prof T=3 lin=%.2f gdn=%.2f attn=%.2f mlp=%.2f lm=%.2f tot=%.2f\n",
                         g_spec_ms_lin, g_spec_ms_gdn, g_spec_ms_attn, g_spec_ms_mlp, g_spec_ms_lm,
                         g_spec_ms_lin + g_spec_ms_gdn + g_spec_ms_attn + g_spec_ms_mlp + g_spec_ms_lm);
            CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
            if (conv_bytes_ && d_conv_ && d_conv_bak_)
                CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_, cudaMemcpyDeviceToDevice));
            if (kv_bytes_ && d_kcache_ && d_k_bak_) {
                CUDA_CHECK(cudaMemcpy(d_kcache_, d_k_bak_, kv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_vcache_, d_v_bak_, kv_bytes_, cudaMemcpyDeviceToDevice));
            }
            pos_ = pos0;
            set_pos_k<<<1, 1>>>(d_pos_, pos0);
        }

        // T=4 Q6 ild lives in gemv_q6_t4.cu (isolated TU). Capture when pf_cap allows.
    capture_t4:
        if (spec_graph_t4_exec_ || pf_cap_ < 4 || hidden_ < 64) return;
        abort_stream_capture();
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) cudaGetLastError();
        e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "spec_capture_t4_begin err=%s\n", cudaGetErrorString(e));
            return;
        }
        launch_spec_chunk(4);
        g = nullptr;
        e = cudaStreamEndCapture(cudaStreamPerThread, &g);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "spec_capture_t4_end err=%s last=%s\n", cudaGetErrorString(e),
                         cudaGetErrorString(cudaGetLastError()));
            abort_stream_capture();
            return;
        }
        if (!instantiate_graph(g, &spec_graph_t4_exec_, "spec_capture_t4", 4)) return;
        spec_graph_t4_ = g;
        cudaGraphUpload(spec_graph_t4_exec_, cudaStreamPerThread);
        std::fprintf(stderr, "spec_cuda_graph n=4 t0_snap=%d\n", spec_t0_snap_ ? 1 : 0);

        if (hidden_ >= 64 && d_S_ && d_S_bak_ && s_bytes_) {
            CUDA_CHECK(cudaMemcpy(d_S_bak_, d_S_, s_bytes_, cudaMemcpyDeviceToDevice));
            if (conv_bytes_ && d_conv_ && d_conv_bak_)
                CUDA_CHECK(cudaMemcpy(d_conv_bak_, d_conv_, conv_bytes_, cudaMemcpyDeviceToDevice));
            if (kv_bytes_ && d_kcache_ && d_k_bak_) {
                CUDA_CHECK(cudaMemcpy(d_k_bak_, d_kcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_v_bak_, d_vcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
            }
            const int pos0 = pos_;
            g_spec_ms_lin = g_spec_ms_gdn = g_spec_ms_attn = g_spec_ms_mlp = g_spec_ms_lm = 0;
            g_spec_prof = true;
            launch_spec_chunk(4);
            g_spec_prof = false;
            mtp_t4_ms_ = g_spec_ms_lin + g_spec_ms_gdn + g_spec_ms_attn + g_spec_ms_mlp + g_spec_ms_lm;
            std::fprintf(stderr, "spec_prof T=4 lin=%.2f gdn=%.2f attn=%.2f mlp=%.2f lm=%.2f tot=%.2f\n",
                         g_spec_ms_lin, g_spec_ms_gdn, g_spec_ms_attn, g_spec_ms_mlp, g_spec_ms_lm,
                         mtp_t4_ms_);
            CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
            if (conv_bytes_ && d_conv_ && d_conv_bak_)
                CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_, cudaMemcpyDeviceToDevice));
            if (kv_bytes_ && d_kcache_ && d_k_bak_) {
                CUDA_CHECK(cudaMemcpy(d_kcache_, d_k_bak_, kv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_vcache_, d_v_bak_, kv_bytes_, cudaMemcpyDeviceToDevice));
            }
            pos_ = pos0;
            set_pos_k<<<1, 1>>>(d_pos_, pos0);
        }

        if (spec_graph_t6_exec_ || pf_cap_ < 6 || hidden_ < 64) return;
        abort_stream_capture();
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) cudaGetLastError();
        e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "spec_capture_t6_begin err=%s\n", cudaGetErrorString(e));
            return;
        }
        launch_spec_chunk(6);
        g = nullptr;
        e = cudaStreamEndCapture(cudaStreamPerThread, &g);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "spec_capture_t6_end err=%s\n", cudaGetErrorString(e));
            abort_stream_capture();
            return;
        }
        if (!instantiate_graph(g, &spec_graph_t6_exec_, "spec_capture_t6", 6)) return;
        spec_graph_t6_ = g;
        cudaGraphUpload(spec_graph_t6_exec_, cudaStreamPerThread);
        std::fprintf(stderr, "spec_cuda_graph n=6 t0_snap=%d\n", spec_t0_snap_ ? 1 : 0);

        if (hidden_ >= 64 && d_S_ && d_S_bak_ && s_bytes_) {
            CUDA_CHECK(cudaMemcpy(d_S_bak_, d_S_, s_bytes_, cudaMemcpyDeviceToDevice));
            if (conv_bytes_ && d_conv_ && d_conv_bak_)
                CUDA_CHECK(cudaMemcpy(d_conv_bak_, d_conv_, conv_bytes_, cudaMemcpyDeviceToDevice));
            if (kv_bytes_ && d_kcache_ && d_k_bak_) {
                CUDA_CHECK(cudaMemcpy(d_k_bak_, d_kcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_v_bak_, d_vcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
            }
            const int pos0 = pos_;
            g_spec_ms_lin = g_spec_ms_gdn = g_spec_ms_attn = g_spec_ms_mlp = g_spec_ms_lm = 0;
            g_spec_prof = true;
            launch_spec_chunk(6);
            g_spec_prof = false;
            std::fprintf(stderr, "spec_prof T=6 lin=%.2f gdn=%.2f attn=%.2f mlp=%.2f lm=%.2f tot=%.2f\n",
                         g_spec_ms_lin, g_spec_ms_gdn, g_spec_ms_attn, g_spec_ms_mlp, g_spec_ms_lm,
                         g_spec_ms_lin + g_spec_ms_gdn + g_spec_ms_attn + g_spec_ms_mlp + g_spec_ms_lm);
            CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
            if (conv_bytes_ && d_conv_ && d_conv_bak_)
                CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_, cudaMemcpyDeviceToDevice));
            if (kv_bytes_ && d_kcache_ && d_k_bak_) {
                CUDA_CHECK(cudaMemcpy(d_kcache_, d_k_bak_, kv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_vcache_, d_v_bak_, kv_bytes_, cudaMemcpyDeviceToDevice));
            }
            pos_ = pos0;
            set_pos_k<<<1, 1>>>(d_pos_, pos0);
        }

    capture_t12:
        if (spec_graph_t12_exec_ || pf_cap_ < 12 || hidden_ < 64) return;
        abort_stream_capture();
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) cudaGetLastError();
        e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "spec_capture_t12_begin err=%s\n", cudaGetErrorString(e));
            return;
        }
        launch_spec_chunk(12);
        g = nullptr;
        e = cudaStreamEndCapture(cudaStreamPerThread, &g);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "spec_capture_t12_end err=%s\n", cudaGetErrorString(e));
            abort_stream_capture();
            return;
        }
        if (!instantiate_graph(g, &spec_graph_t12_exec_, "spec_capture_t12", 12)) return;
        spec_graph_t12_ = g;
        cudaGraphUpload(spec_graph_t12_exec_, cudaStreamPerThread);
        std::fprintf(stderr, "spec_cuda_graph n=12 t0_snap=%d\n", spec_t0_snap_ ? 1 : 0);

        if (hidden_ >= 64 && d_S_ && d_S_bak_ && s_bytes_) {
            CUDA_CHECK(cudaMemcpy(d_S_bak_, d_S_, s_bytes_, cudaMemcpyDeviceToDevice));
            if (conv_bytes_ && d_conv_ && d_conv_bak_)
                CUDA_CHECK(cudaMemcpy(d_conv_bak_, d_conv_, conv_bytes_, cudaMemcpyDeviceToDevice));
            if (kv_bytes_ && d_kcache_ && d_k_bak_) {
                CUDA_CHECK(cudaMemcpy(d_k_bak_, d_kcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_v_bak_, d_vcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
            }
            const int pos0 = pos_;
            g_spec_ms_lin = g_spec_ms_gdn = g_spec_ms_attn = g_spec_ms_mlp = g_spec_ms_lm = 0;
            g_spec_prof = true;
            launch_spec_chunk(12);
            g_spec_prof = false;
            std::fprintf(stderr, "spec_prof T=12 lin=%.2f gdn=%.2f attn=%.2f mlp=%.2f lm=%.2f tot=%.2f\n",
                         g_spec_ms_lin, g_spec_ms_gdn, g_spec_ms_attn, g_spec_ms_mlp, g_spec_ms_lm,
                         g_spec_ms_lin + g_spec_ms_gdn + g_spec_ms_attn + g_spec_ms_mlp + g_spec_ms_lm);
            CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
            if (conv_bytes_ && d_conv_ && d_conv_bak_)
                CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_, cudaMemcpyDeviceToDevice));
            if (kv_bytes_ && d_kcache_ && d_k_bak_) {
                CUDA_CHECK(cudaMemcpy(d_kcache_, d_k_bak_, kv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_vcache_, d_v_bak_, kv_bytes_, cudaMemcpyDeviceToDevice));
            }
            pos_ = pos0;
            set_pos_k<<<1, 1>>>(d_pos_, pos0);
            // Wall-clock eager vs graph (no per-section prof sync). Recapture
            // once if graph is slower; drain uses the winner.
            if (spec_graph_t12_exec_) {
                auto restore12 = [&]() {
                    CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
                    if (conv_bytes_ && d_conv_ && d_conv_bak_)
                        CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_,
                                               cudaMemcpyDeviceToDevice));
                    if (kv_bytes_ && d_kcache_ && d_k_bak_) {
                        CUDA_CHECK(cudaMemcpy(d_kcache_, d_k_bak_, kv_bytes_,
                                               cudaMemcpyDeviceToDevice));
                        CUDA_CHECK(cudaMemcpy(d_vcache_, d_v_bak_, kv_bytes_,
                                               cudaMemcpyDeviceToDevice));
                    }
                    pos_ = pos0;
                    set_pos_k<<<1, 1>>>(d_pos_, pos0);
                };
                auto time_launch = [&](bool graph) -> float {
                    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
                    cudaEventCreate(&ev0);
                    cudaEventCreate(&ev1);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    cudaEventRecord(ev0, cudaStreamPerThread);
                    if (graph)
                        CUDA_CHECK(cudaGraphLaunch(spec_graph_t12_exec_, cudaStreamPerThread));
                    else
                        launch_spec_chunk(12);
                    cudaEventRecord(ev1, cudaStreamPerThread);
                    cudaEventSynchronize(ev1);
                    float ms = 0.f;
                    cudaEventElapsedTime(&ms, ev0, ev1);
                    cudaEventDestroy(ev0);
                    cudaEventDestroy(ev1);
                    return ms;
                };
                time_launch(false);
                restore12();
                const float eager_ms = time_launch(false);
                restore12();
                time_launch(true);
                restore12();
                float graph_ms = time_launch(true);
                restore12();
                // Recapture up to 3 times. A later capture is often 3–5 ms
                // faster than the first (75 vs 80). Keep the winner; stop
                // early if a retry does not improve.
                for (int retry = 0; retry < 3; ++retry) {
                    cudaGraphExec_t old_exec = spec_graph_t12_exec_;
                    cudaGraph_t old_g = spec_graph_t12_;
                    spec_graph_t12_exec_ = nullptr;
                    spec_graph_t12_ = nullptr;
                    abort_stream_capture();
                    e = cudaDeviceSynchronize();
                    if (e != cudaSuccess) cudaGetLastError();
                    e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
                    if (e != cudaSuccess) {
                        spec_graph_t12_exec_ = old_exec;
                        spec_graph_t12_ = old_g;
                        break;
                    }
                    launch_spec_chunk(12);
                    cudaGraph_t ng = nullptr;
                    e = cudaStreamEndCapture(cudaStreamPerThread, &ng);
                    cudaGraphExec_t nexec = nullptr;
                    if (e != cudaSuccess || !instantiate_graph(ng, &nexec, "spec_capture_t12_retry", 12)) {
                        abort_stream_capture();
                        spec_graph_t12_exec_ = old_exec;
                        spec_graph_t12_ = old_g;
                        break;
                    }
                    spec_graph_t12_exec_ = nexec;
                    spec_graph_t12_ = ng;
                    cudaGraphUpload(spec_graph_t12_exec_, cudaStreamPerThread);
                    restore12();
                    const float ms2 = time_launch(true);
                    restore12();
                    std::fprintf(stderr, "spec_graph_t12_retry n=%d ms=%.2f was=%.2f\n", retry, ms2, graph_ms);
                    if (ms2 + 0.15f < graph_ms) {
                        if (old_exec) cudaGraphExecDestroy(old_exec);
                        if (old_g) cudaGraphDestroy(old_g);
                        graph_ms = ms2;
                    } else {
                        cudaGraphExecDestroy(nexec);
                        cudaGraphDestroy(ng);
                        spec_graph_t12_exec_ = old_exec;
                        spec_graph_t12_ = old_g;
                        break;
                    }
                }
                mtp_t12_use_graph_ = (graph_ms + 0.4f < eager_ms);
                mtp_t12_ms_ = mtp_t12_use_graph_ ? graph_ms : eager_ms;
                std::fprintf(stderr, "spec_t12_pick eager=%.2f graph=%.2f use_graph=%d\n", eager_ms, graph_ms,
                             mtp_t12_use_graph_ ? 1 : 0);
            }
        }

        // T=16: one verify of 16 live drafts. Eager tot was 133ms; a
        // recaptured graph can beat T=4+T=12 wall (two launches + join).
        if (false && pf_cap_ >= 16 && hidden_ >= 64 && d_S_ && d_S_bak_ && s_bytes_ && !spec_graph_t16_exec_) {
            abort_stream_capture();
            e = cudaDeviceSynchronize();
            if (e != cudaSuccess) cudaGetLastError();
            e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
            if (e == cudaSuccess) {
                launch_spec_chunk(16);
                cudaGraph_t g16 = nullptr;
                e = cudaStreamEndCapture(cudaStreamPerThread, &g16);
                if (e == cudaSuccess && instantiate_graph(g16, &spec_graph_t16_exec_, "spec_capture_t16", 16)) {
                    spec_graph_t16_ = g16;
                    cudaGraphUpload(spec_graph_t16_exec_, cudaStreamPerThread);
                    std::fprintf(stderr, "spec_cuda_graph n=16 t0_snap=%d\n", spec_t0_snap_ ? 1 : 0);
                } else {
                    abort_stream_capture();
                    spec_graph_t16_exec_ = nullptr;
                    spec_graph_t16_ = nullptr;
                }
            }
            CUDA_CHECK(cudaMemcpy(d_S_bak_, d_S_, s_bytes_, cudaMemcpyDeviceToDevice));
            if (conv_bytes_ && d_conv_ && d_conv_bak_)
                CUDA_CHECK(cudaMemcpy(d_conv_bak_, d_conv_, conv_bytes_, cudaMemcpyDeviceToDevice));
            if (kv_bytes_ && d_kcache_ && d_k_bak_) {
                CUDA_CHECK(cudaMemcpy(d_k_bak_, d_kcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_v_bak_, d_vcache_, kv_bytes_, cudaMemcpyDeviceToDevice));
            }
            const int pos0 = pos_;
            g_spec_ms_lin = g_spec_ms_gdn = g_spec_ms_attn = g_spec_ms_mlp = g_spec_ms_lm = 0;
            g_spec_prof = true;
            launch_spec_chunk(16);
            g_spec_prof = false;
            std::fprintf(stderr, "spec_prof T=16 lin=%.2f gdn=%.2f attn=%.2f mlp=%.2f lm=%.2f tot=%.2f\n",
                         g_spec_ms_lin, g_spec_ms_gdn, g_spec_ms_attn, g_spec_ms_mlp, g_spec_ms_lm,
                         g_spec_ms_lin + g_spec_ms_gdn + g_spec_ms_attn + g_spec_ms_mlp + g_spec_ms_lm);
            auto restore16 = [&]() {
                CUDA_CHECK(cudaMemcpy(d_S_, d_S_bak_, s_bytes_, cudaMemcpyDeviceToDevice));
                if (conv_bytes_ && d_conv_ && d_conv_bak_)
                    CUDA_CHECK(cudaMemcpy(d_conv_, d_conv_bak_, conv_bytes_, cudaMemcpyDeviceToDevice));
                if (kv_bytes_ && d_kcache_ && d_k_bak_) {
                    CUDA_CHECK(cudaMemcpy(d_kcache_, d_k_bak_, kv_bytes_, cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemcpy(d_vcache_, d_v_bak_, kv_bytes_, cudaMemcpyDeviceToDevice));
                }
                pos_ = pos0;
                set_pos_k<<<1, 1>>>(d_pos_, pos0);
            };
            restore16();
            if (spec_graph_t16_exec_) {
                auto time16 = [&](bool graph) -> float {
                    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
                    cudaEventCreate(&ev0);
                    cudaEventCreate(&ev1);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    cudaEventRecord(ev0, cudaStreamPerThread);
                    if (graph)
                        CUDA_CHECK(cudaGraphLaunch(spec_graph_t16_exec_, cudaStreamPerThread));
                    else
                        launch_spec_chunk(16);
                    cudaEventRecord(ev1, cudaStreamPerThread);
                    cudaEventSynchronize(ev1);
                    float ms = 0.f;
                    cudaEventElapsedTime(&ms, ev0, ev1);
                    cudaEventDestroy(ev0);
                    cudaEventDestroy(ev1);
                    return ms;
                };
                time16(false);
                restore16();
                const float eager16 = time16(false);
                restore16();
                time16(true);
                restore16();
                float graph16 = time16(true);
                restore16();
                {
                    cudaGraphExec_t old_exec = spec_graph_t16_exec_;
                    cudaGraph_t old_g = spec_graph_t16_;
                    spec_graph_t16_exec_ = nullptr;
                    spec_graph_t16_ = nullptr;
                    abort_stream_capture();
                    e = cudaDeviceSynchronize();
                    if (e != cudaSuccess) cudaGetLastError();
                    e = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
                    if (e == cudaSuccess) {
                        launch_spec_chunk(16);
                        cudaGraph_t ng = nullptr;
                        e = cudaStreamEndCapture(cudaStreamPerThread, &ng);
                        cudaGraphExec_t nexec = nullptr;
                        if (e == cudaSuccess && instantiate_graph(ng, &nexec, "spec_capture_t16_retry", 16)) {
                            spec_graph_t16_exec_ = nexec;
                            spec_graph_t16_ = ng;
                            cudaGraphUpload(spec_graph_t16_exec_, cudaStreamPerThread);
                            restore16();
                            const float ms2 = time16(true);
                            restore16();
                            std::fprintf(stderr, "spec_graph_t16_retry ms=%.2f was=%.2f\n", ms2, graph16);
                            if (ms2 + 0.15f < graph16) {
                                if (old_exec) cudaGraphExecDestroy(old_exec);
                                if (old_g) cudaGraphDestroy(old_g);
                                graph16 = ms2;
                            } else {
                                cudaGraphExecDestroy(nexec);
                                cudaGraphDestroy(ng);
                                spec_graph_t16_exec_ = old_exec;
                                spec_graph_t16_ = old_g;
                            }
                        } else {
                            abort_stream_capture();
                            spec_graph_t16_exec_ = old_exec;
                            spec_graph_t16_ = old_g;
                        }
                    } else {
                        spec_graph_t16_exec_ = old_exec;
                        spec_graph_t16_ = old_g;
                    }
                }
                mtp_t16_use_graph_ = (graph16 + 0.4f < eager16);
                mtp_t16_ms_ = mtp_t16_use_graph_ ? graph16 : eager16;
                // +12ms: T=4+T=12 pays two launches and an extras join.
                mtp_t16_ok_ = spec_graph_t16_exec_ &&
                              (mtp_t16_ms_ + 4.f < mtp_t4_ms_ + mtp_t12_ms_ + 12.f);
                std::fprintf(stderr,
                             "spec_t16_pick eager=%.2f graph=%.2f use_graph=%d ok=%d vs_t4t12=%.1f\n",
                             eager16, graph16, mtp_t16_use_graph_ ? 1 : 0, mtp_t16_ok_ ? 1 : 0,
                             mtp_t4_ms_ + mtp_t12_ms_);
            } else {
                restore16();
            }
        }
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
        // Official Qwen3.8 lm_head is BF16. Requantizing it to e4m3 flipped
        // the 1,2,3 first token from 4 (HF/vLLM) to 0 (gap 0.38). Keep BF16
        // unless RAPIDLLM_FP8_LMHEAD=1 opts back into the old pack.
        const char* force_fp8 = std::getenv("RAPIDLLM_FP8_LMHEAD");
        if (lh.quant == QuantKind::BF16 && lh_rows >= 4096 && (lh_cols % 128) == 0 &&
            force_fp8 && force_fp8[0] == '1')
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
        {
            const int tiled = store_->model().gdn_v_tiled ? 1 : 0;
            set_gdn_v_tiled(tiled);
            if (tiled) std::fprintf(stderr, "cuda_gdn_v_tiled=1\n");
        }
        n_delta_ = n_delta;
        n_attn_ = n_attn;
        max_qkv_ = max_qkv;
        max_z_ = max_z;
        max_nv_ = max_nv;
        max_inter_ = max_inter;
        max_qn_ = max_qn;
        max_kn_ = max_kn;
        // ctx<=4096 one-shots official --ctx 256 prompts (n=76 used to leftover
        // T=12 and hit F16-KV via the T<32 float* attn fallback → invalid arg).
        // Long ctx still chunks at 1024.
        pf_cap_ = std::max(1, std::min(ctx_, ctx_ > 4096 ? 1024 : 256));
        max_batch_ = 1;
        if (const char* e = std::getenv("RAPIDLLM_MAX_BATCH")) {
            const int v = std::atoi(e);
            // Prefill graphs stay at pf_cap_; decode batch can go higher on short ctx.
            if (v > 1) max_batch_ = std::min(v, std::min(ctx_, 128));
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
        logit_rows_ = std::max(max_batch_, 16);
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
        d_S_mid_ = static_cast<uint16_t*>(alloc(s_bytes_));
        d_conv_mid_ = static_cast<float*>(alloc(conv_bytes_));
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
            if (cudaMemGetInfo(&free_b, &tot_b) == cudaSuccess) {
                std::fprintf(stderr, "cuda_mem used=%zuMiB free=%zuMiB total=%zuMiB max_batch=%d\n",
                             (tot_b - free_b) / (1024 * 1024), free_b / (1024 * 1024), tot_b / (1024 * 1024),
                             max_batch_);
                // 163840 F16 KV leaves ~200 MiB. seq_cap=1024 activations OOM;
                // 256 still one-shots the official --prompt-n 256 fill.
                if (free_b < (768ull << 20) && pf_cap_ > 256) {
                    std::fprintf(stderr, "shrink_pf_cap %d->256 free=%zuMiB\n", pf_cap_,
                                 free_b / (1024 * 1024));
                    pf_cap_ = 256;
                }
            }
        }
        // Equal-length mixed prefill packs time-major T×B rows. Official
        // mixed is T=3; 4 tokens of headroom lets one W pass fit.
        const int eq_rows = max_batch_ > 1 ? max_batch_ * 4 : max_batch_;
        const int seq_cap = std::max(pf_cap_, eq_rows);
        seq_cap_ = seq_cap;
        // seq_cap is the allocated KV/session window (ctx). act_cap is the
        // tiled prefill activation buffer (1024 when ctx>4096).
        std::fprintf(stderr, "seq_cap=%d pf_cap=%d max_batch=%d act_cap=%d\n", ctx_, pf_cap_, max_batch_,
                     seq_cap_);

        d_h_seq_ = static_cast<float*>(alloc(sizeof(float) * seq_cap * hidden_));
        d_xn_seq_ = static_cast<float*>(alloc(sizeof(float) * seq_cap * hidden_));
        d_y_seq_ = static_cast<float*>(alloc(sizeof(float) * seq_cap * hidden_));
        g_yadd = d_y_seq_;
        g_yadd_n = seq_cap * hidden_;
        d_qkv_seq_ = static_cast<float*>(alloc(sizeof(float) * seq_cap * std::max(max_qkv, 1)));
        d_mix_seq_ = static_cast<float*>(alloc(sizeof(float) * seq_cap * std::max(max_qkv, 1)));
        d_z_seq_ = static_cast<float*>(alloc(sizeof(float) * seq_cap * std::max(max_z, 1)));
        d_aa_seq_ = static_cast<float*>(alloc(sizeof(float) * seq_cap * std::max(max_nv, 1)));
        d_bb_seq_ = static_cast<float*>(alloc(sizeof(float) * seq_cap * std::max(max_nv, 1)));
        d_og_seq_ = static_cast<float*>(alloc(sizeof(float) * seq_cap * std::max(max_z, 1)));
        d_qg_seq_ = static_cast<float*>(alloc(sizeof(float) * seq_cap * std::max(max_qn * 2, 1)));
        d_q_seq_ = static_cast<float*>(alloc(sizeof(float) * seq_cap * std::max(max_qn, 1)));
        d_gate_seq_ = static_cast<float*>(alloc(sizeof(float) * seq_cap * std::max(max_qn, 1)));
        d_k_seq_ = static_cast<float*>(alloc(sizeof(float) * seq_cap * std::max(max_kn, 1)));
        d_v_seq_ = static_cast<float*>(alloc(sizeof(float) * seq_cap * std::max(max_kn, 1)));
        d_o_seq_ = static_cast<float*>(alloc(sizeof(float) * seq_cap * std::max(std::max(max_z, max_qn), 1)));
        d_gate_mlp_seq_ = static_cast<float*>(alloc(sizeof(float) * seq_cap * std::max(max_inter, 1)));
        d_up_seq_ = static_cast<float*>(alloc(sizeof(float) * seq_cap * std::max(max_inter, 1)));
        d_toks_ = static_cast<int*>(alloc(sizeof(int) * seq_cap));
        d_drain_toks_ = static_cast<int*>(alloc(sizeof(int) * 16));
        d_drain_best_ = static_cast<int*>(alloc(sizeof(int) * 16));
        const int xf16_n = seq_cap * std::max({hidden_, std::max(max_inter, 1), std::max(max_qkv, 1),
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
        // T=12 needs 12*cols Q8 x. Tiny (hidden<64) stays 4× so goldens heap
        // matches the last passing T=3 suite.
        g_xq_n = (hidden_ >= 64 ? 17 : 4) * std::max(max_inter, std::max(hidden_, 17408));
        if (g_xq_n & 31) g_xq_n = (g_xq_n + 31) & ~31;
        g_xq = static_cast<int8_t*>(alloc(static_cast<size_t>(g_xq_n)));
        g_xsc = static_cast<__half*>(alloc(sizeof(__half) * static_cast<size_t>(g_xq_n / 32)));
        g_xsum = static_cast<int32_t*>(alloc(sizeof(int32_t) * static_cast<size_t>(g_xq_n / 32)));
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
            auto needs_wf16 = [](const GpuW& w) { return w.q == QuantKind::Q8_0; };
            bool want = needs_wf16(lm_head_);
            for (const GpuLayer& L : layers_) {
                const GpuW* ws[] = {&L.wqkv, &L.wz, &L.wa, &L.wb, &L.wo, &L.wq, &L.wk, &L.wv, &L.wo_a,
                                    &L.wg,   &L.wu, &L.wd};
                for (const GpuW* p : ws)
                    if (needs_wf16(*p)) want = true;
            }
            if (want) {
                const size_t kmax = static_cast<size_t>(
                    std::max({hidden_, std::max(max_inter, 1), std::max(max_qkv, 1), std::max(max_qn * 2, 1)}));
                const size_t nmax = static_cast<size_t>(
                    std::max({hidden_, std::max(max_inter, 1), std::max(max_qkv, 1), std::max(max_qn * 2, 1)}));
                g_wf16_n = 2 * kmax * nmax;
                d_wf16_ = static_cast<__half*>(alloc(sizeof(__half) * g_wf16_n));
                g_wf16 = d_wf16_;
            } else {
                g_wf16 = nullptr;
                g_wf16_n = 0;
                d_wf16_ = nullptr;
                std::fprintf(stderr, "skip_wf16=1 (no Q8 / kmajor-dequant W)\n");
            }
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
        h_drain_pin_ = nullptr;
        if (cudaMallocHost(reinterpret_cast<void**>(&h_drain_pin_), sizeof(int32_t) * 24) != cudaSuccess)
            h_drain_pin_ = nullptr;
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
                if (!w.fp8_rowmaj || (T < 16 && T != 2)) return;
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
                        if (T == 2 || T >= 256) lt_tune(w, X, Y, T, 0);
                    }
                }
                if (!cached(w.rows, T, w.cols, 1)) {
                    launch_cublas_fp8_e4(w, X, Y, T, 1);
                    if (T == 2 || T >= 256) lt_tune(w, X, Y, T, 1);
                }
            };
            const int ts[4] = {2, std::min(256, pf_cap_), pf_cap_, std::max(max_batch_, 16)};
            for (int ti = 0; ti < 4; ++ti) {
                const int T = ts[ti];
                if (T != 2 && T < 16) continue;
                if (ti == 2 && T == ts[1]) continue;
                if (T == 2) g_lt_min_T = 2;
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
                warm(lm_head_, T);
                if (T == 2) g_lt_min_T = 8;
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
            {
                int nq8 = 0, nq6 = 0, nq4 = 0, noth = 0;
                long long bq8 = 0, bq6 = 0, bq4 = 0;
                auto accq = [&](const GpuW& w) {
                    if (!w.data || w.rows <= 0 || w.cols <= 0) return;
                    const long long el = static_cast<long long>(w.rows) * w.cols;
                    if (w.q == QuantKind::Q8_0) {
                        ++nq8;
                        bq8 += el;
                    } else if (w.q == QuantKind::Q6_K) {
                        ++nq6;
                        bq6 += el;
                    } else if (w.q == QuantKind::Q4_K) {
                        ++nq4;
                        bq4 += el;
                    } else
                        ++noth;
                };
                for (const GpuLayer& L : layers_) {
                    accq(L.wqkv);
                    accq(L.wz);
                    accq(L.wo);
                    accq(L.wq);
                    accq(L.wk);
                    accq(L.wv);
                    accq(L.wo_a);
                    accq(L.wg);
                    accq(L.wu);
                    accq(L.wd);
                }
                accq(lm_head_);
                std::fprintf(stderr, "w_quant nq8=%d nq6=%d nq4=%d noth=%d el8=%lld el6=%lld el4=%lld\n", nq8,
                             nq6, nq4, noth, bq8, bq6, bq4);
            }
            }
        }
        {
            int zero = 0;
            CUDA_CHECK(cudaMemcpy(d_pos_, &zero, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_tok_, &zero, 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
            bool packed_gemv = false;
            for (const GpuLayer& L : layers_) {
                const QuantKind q = L.wg.q;
                if (q == QuantKind::Q8_0 || q == QuantKind::Q4_K || q == QuantKind::Q5_K || q == QuantKind::Q6_K) {
                    packed_gemv = true;
                    break;
                }
            }
            size_t cap_free = 0, cap_tot = 0;
            const bool low_mem =
                cudaMemGetInfo(&cap_free, &cap_tot) == cudaSuccess && cap_free < (512ull << 20);
            if (low_mem)
                std::fprintf(stderr, "skip_graphs=1 free=%zuMiB\n", cap_free / (1024 * 1024));
            {
                const int flash = flash_attn_on() ? 1 : 0;
                set_flash_attn(flash);
                int gqa = 0;
                for (const GpuLayer& L : layers_) {
                    if (L.kind != LayerKind::GatedDeltaNet && flash_gqa_ok(L.nq, L.nkv, L.hd, 1)) {
                        gqa = L.nq / L.nkv;
                        break;
                    }
                }
                std::fprintf(stderr, "flashinfer_prefill=%d flash_decode=%d flash_gqa6=%d fi_min_T=%d\n",
                             (fi_available() && flash) ? 1 : 0, flash, (flash && gqa == 6) ? 1 : 0, 16);
            }
            if (vocab_ > 256 && !low_mem) maybe_capture();
            if (vocab_ > 256 && !low_mem && max_batch_ >= 8) maybe_capture_batch(max_batch_);
            if (!graph_exec_) {
                abort_stream_capture();
                cudaDeviceSynchronize();
                cudaGetLastError();
            }
            skip_pf_graph_ = vocab_ <= 256 || packed_gemv;
            for (const GpuLayer& L : layers_) {
                if (L.wqkv.fp8_rowmaj || L.wg.fp8_rowmaj || L.wq.fp8_rowmaj || L.wo.fp8_rowmaj) {
                    skip_pf_graph_ = true;
                    break;
                }
            }
            if (lm_head_.fp8_rowmaj) skip_pf_graph_ = true;
            // Q4/Q6 skip long prefill graphs (packed GEMV capture is huge) but
            // n=2/3/4 is one W pass — same kernels as eager, graph the launches.
            if (vocab_ > 256) {
                rapidllm::cuda_gemv::warmup_gdn_decode_t12();
                maybe_capture_spec();
            }
            if (vocab_ > 256) maybe_capture_mtp();
            if (vocab_ > 256) maybe_capture_mtp_t2();
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
            std::fprintf(stderr, "cuda_flash_ready=1\n");
        }
        {
            size_t free_b = 0, tot_b = 0;
            if (cudaMemGetInfo(&free_b, &tot_b) != cudaSuccess || free_b > (256ull << 20))
                upload_mtp();
            else
                std::fprintf(stderr, "skip_mtp=1 free=%zuMiB\n", free_b / (1024 * 1024));
        }
        for (auto& [_, t] : store_->table().tensors) {
            // Host MTP fallback still needs embed + lm_head + mtp.* after GPU upload.
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
        if (h_drain_pin_) {
            cudaFreeHost(h_drain_pin_);
            h_drain_pin_ = nullptr;
        }
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
        if (graph_) cudaGraphDestroy(graph_);
        if (batch_graph_exec_) cudaGraphExecDestroy(batch_graph_exec_);
        if (batch_graph_) cudaGraphDestroy(batch_graph_);
        if (spec_graph_exec_) cudaGraphExecDestroy(spec_graph_exec_);
        if (spec_graph_) cudaGraphDestroy(spec_graph_);
        if (spec_graph_t3_exec_) cudaGraphExecDestroy(spec_graph_t3_exec_);
        if (spec_graph_t3_) cudaGraphDestroy(spec_graph_t3_);
        if (spec_graph_t4_exec_) cudaGraphExecDestroy(spec_graph_t4_exec_);
        if (spec_graph_t4_) cudaGraphDestroy(spec_graph_t4_);
        if (spec_graph_t6_exec_) cudaGraphExecDestroy(spec_graph_t6_exec_);
        if (spec_graph_t6_) cudaGraphDestroy(spec_graph_t6_);
        if (spec_graph_t12_exec_) cudaGraphExecDestroy(spec_graph_t12_exec_);
        if (spec_graph_t12_) cudaGraphDestroy(spec_graph_t12_);
        if (spec_graph_t16_exec_) cudaGraphExecDestroy(spec_graph_t16_exec_);
        if (spec_graph_t16_) cudaGraphDestroy(spec_graph_t16_);
        if (mtp_graph_exec_) cudaGraphExecDestroy(mtp_graph_exec_);
        if (mtp_rec_exec_) cudaGraphExecDestroy(mtp_rec_exec_);
        if (mtp_rec_graph_) cudaGraphDestroy(mtp_rec_graph_);
        if (mtp_chain_exec_) cudaGraphExecDestroy(mtp_chain_exec_);
        if (mtp_chain_graph_) cudaGraphDestroy(mtp_chain_graph_);
        if (mtp_stem_chain_exec_) cudaGraphExecDestroy(mtp_stem_chain_exec_);
        if (mtp_stem_chain_graph_) cudaGraphDestroy(mtp_stem_chain_graph_);
        if (mtp_hin_chain_exec_) cudaGraphExecDestroy(mtp_hin_chain_exec_);
        if (mtp_hin_chain_graph_) cudaGraphDestroy(mtp_hin_chain_graph_);
        if (mtp_hin_rec4_exec_) cudaGraphExecDestroy(mtp_hin_rec4_exec_);
        if (mtp_hin_rec4_graph_) cudaGraphDestroy(mtp_hin_rec4_graph_);
        if (mtp_hin_side_exec_) cudaGraphExecDestroy(mtp_hin_side_exec_);
        if (mtp_hin_side_graph_) cudaGraphDestroy(mtp_hin_side_graph_);
        if (mtp_rec4_side_exec_) cudaGraphExecDestroy(mtp_rec4_side_exec_);
        if (mtp_rec4_side_graph_) cudaGraphDestroy(mtp_rec4_side_graph_);
        if (mtp_side_ev_) cudaEventDestroy(mtp_side_ev_);
        if (mtp_slot0_ev_) cudaEventDestroy(mtp_slot0_ev_);
        if (mtp_slot1_ev_) cudaEventDestroy(mtp_slot1_ev_);
        if (mtp_pf_l20_ev_) cudaEventDestroy(mtp_pf_l20_ev_);
        if (mtp_pf_l40_ev_) cudaEventDestroy(mtp_pf_l40_ev_);
        if (mtp_t2_exec_) cudaGraphExecDestroy(mtp_t2_exec_);
        if (mtp_t2_graph_) cudaGraphDestroy(mtp_t2_graph_);
        if (mtp_t4_exec_) cudaGraphExecDestroy(mtp_t4_exec_);
        if (mtp_t4_graph_) cudaGraphDestroy(mtp_t4_graph_);
        if (mtp_graph_) cudaGraphDestroy(mtp_graph_);
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
        if (bak2_stream_) {
            cudaStreamDestroy(bak2_stream_);
            bak2_stream_ = nullptr;
        }
        if (mtp_side2_ev_) cudaEventDestroy(mtp_side2_ev_);
        if (mtp_rec4_b2_exec_) cudaGraphExecDestroy(mtp_rec4_b2_exec_);
        if (mtp_rec4_b2_graph_) cudaGraphDestroy(mtp_rec4_b2_graph_);
        if (mtp_hin_b2_exec_) cudaGraphExecDestroy(mtp_hin_b2_exec_);
        if (mtp_hin_b2_graph_) cudaGraphDestroy(mtp_hin_b2_graph_);
        if (mtp_quad_hin_exec_) cudaGraphExecDestroy(mtp_quad_hin_exec_);
        if (mtp_quad_hin_graph_) cudaGraphDestroy(mtp_quad_hin_graph_);
        g_xe_buf = nullptr;
        g_xe = nullptr;
        g_xe_n = 0;
        g_xe_T = 0;
        g_xe_cap = 0;
        g_xq = nullptr;
        g_xsc = nullptr;
        g_xsum = nullptr;
        g_xq_n = 0;
        lt_cache_clear();
        for (int i = 0; i <= kPfGraphMax; ++i) {
            if (pf_graph_execs_[i]) cudaGraphExecDestroy(pf_graph_execs_[i]);
            if (pf_graphs_[i]) cudaGraphDestroy(pf_graphs_[i]);
        }
        for (void* p : allocs_) cudaFree(p);
        allocs_.clear();
    }

    WeightStore* store_ = nullptr;
    int ctx_ = 0, hidden_ = 0, vocab_ = 0, pos_ = 0, last_tok_ = 0, pf_slot_ = 0;
    int n_delta_ = 0, n_attn_ = 0;
    int pf_cap_ = 1;
    int seq_cap_ = 1;
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
    bool has_mtp_ = false;
    GpuW mtp_fc_{};
    GpuW mtp_fc_f32_{};
    GpuLayer mtp_L_{};
    const float* d_mtp_nh_ = nullptr;
    const float* d_mtp_ne_ = nullptr;
    const float* d_mtp_nn_ = nullptr;
    float *d_mtp_h_ = nullptr, *d_mtp_cat_ = nullptr, *d_mtp_kc_ = nullptr, *d_mtp_vc_ = nullptr;
    int *d_mtp_tok_ = nullptr, *d_mtp_pos_ = nullptr;
    static constexpr int kMtpCap = 256;
    float* d_mtp_hh_ = nullptr;
    float* d_mtp_post_ = nullptr;
    float* d_mtp_hin_ = nullptr;
    float* d_mtp_seed_ = nullptr;
    float* mtp_cycle_h_ = nullptr;
    float* mtp_cycle_h2_ = nullptr;
    bool mtp_have_h4_ = false;
    bool mtp_try_dh4_ = false;
    bool mtp_have_h31t3_ = false;
    int mtp_n_generate_ = 0;
    bool mtp_stream12_ok_ = false;
    bool mtp_off_tried_ = false;
    bool mtp_have_h31t4_ = false;
    int mtp_t31_src_ = 0;
    int mtp_hs_src_ = 0;
    bool mtp_w_h4_ok_ = false;
    bool mtp_w_seed_ok_ = false;
    bool mtp_w_h31_ok_ = false;

    int32_t mtp_w_h4_[4] = {-1, -1, -1, -1};
    int32_t mtp_w_seed_[3] = {-1, -1, -1};
    int32_t mtp_w_h31_[4] = {-1, -1, -1, -1};
    int32_t mtp_h4_d0_ = -1;
    std::vector<int32_t> mtp_hids_;
    int mtp_hist_n_ = 0;
    bool mtp_diag_done_ = false;
    int32_t mtp_legal_t0_ = 0, mtp_legal_t1_ = 0;
    bool mtp_has_legal_ = false;
    int mtp_kv_pos_ = 0;
    int mtp_stream_n_ = 0;
    int mtp_attn_lo_ = 0;
    bool mtp_first_done_ = false;
    int mtp_spec4_i_ = 0;
    bool mtp_hin_ready_ = false;
    bool mtp_have_best4_ = false;
    int32_t mtp_hin_t0_ = 0;
    int mtp_t3_seed_ok_ = 0;
    int mtp_use_t12_ = 0;
    bool mtp_t12_use_graph_ = false;
    bool mtp_t16_use_graph_ = false;
    bool mtp_t16_ok_ = false;
    float mtp_t4_ms_ = 33.f;
    float mtp_t12_ms_ = 76.f;
    float mtp_t16_ms_ = 0.f;
    bool mtp_toks_on_dev_ = false;
    bool spec_t0_snap_ = false;
    bool spec_slot0_ = false;
    std::vector<GpuLayer> layers_;
    std::vector<void*> allocs_;
    float *d_h_ = nullptr, *d_xn_ = nullptr, *d_y_ = nullptr, *d_ss_ = nullptr;
    float *d_qkv_ = nullptr, *d_mix_ = nullptr, *d_z_ = nullptr, *d_aa_ = nullptr, *d_bb_ = nullptr;
    float *d_qh_ = nullptr, *d_kh_ = nullptr, *d_vh_ = nullptr, *d_beta_ = nullptr, *d_glog_ = nullptr;
    float *d_o_ = nullptr, *d_og_ = nullptr;
    float *d_qg_ = nullptr, *d_q_ = nullptr, *d_gate_ = nullptr, *d_k_ = nullptr, *d_vtmp_ = nullptr;
    float *d_gate_mlp_ = nullptr, *d_up_ = nullptr, *d_logits_ = nullptr;
    uint16_t *d_S_ = nullptr, *d_S_bak_ = nullptr, *d_S_mid_ = nullptr;
    float *d_conv_ = nullptr, *d_kcache_ = nullptr, *d_vcache_ = nullptr;
    float *d_k_bak_ = nullptr, *d_v_bak_ = nullptr;
    float *d_conv_bak_ = nullptr, *d_conv_mid_ = nullptr;
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
    int32_t* h_drain_pin_ = nullptr;
    bool mtp_t4_async_ = false;
    int mtp_drain_i_ = 0;
    int* d_drain_toks_ = nullptr;
    int* d_drain_best_ = nullptr;
    static constexpr int kPfGraphMax = 8;
    static constexpr int kGenCap = 64;
    bool capturing_ = false;
    bool fi_pf_ = true;
    cudaGraph_t pf256_graph_ = nullptr;
    cudaGraphExec_t pf256_exec_ = nullptr;
    cudaGraph_t pf1024_graph_[2] = {};
    cudaGraphExec_t pf1024_exec_[2] = {};
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    cudaGraph_t batch_graph_ = nullptr;
    cudaGraphExec_t batch_graph_exec_ = nullptr;
    int batch_graph_B_ = 0;
    cudaGraph_t spec_graph_ = nullptr;
    cudaGraphExec_t spec_graph_exec_ = nullptr;
    cudaGraph_t spec_graph_t3_ = nullptr;
    cudaGraphExec_t spec_graph_t3_exec_ = nullptr;
    cudaGraph_t spec_graph_t4_ = nullptr;
    cudaGraphExec_t spec_graph_t4_exec_ = nullptr;
    cudaGraph_t spec_graph_t6_ = nullptr;
    cudaGraphExec_t spec_graph_t6_exec_ = nullptr;
    cudaGraph_t spec_graph_t12_ = nullptr;
    cudaGraphExec_t spec_graph_t12_exec_ = nullptr;
    cudaGraph_t spec_graph_t16_ = nullptr;
    cudaGraphExec_t spec_graph_t16_exec_ = nullptr;
    cudaGraph_t mtp_graph_ = nullptr;
    cudaGraphExec_t mtp_graph_exec_ = nullptr;
    cudaGraph_t mtp_rec_graph_ = nullptr;
    cudaGraphExec_t mtp_rec_exec_ = nullptr;
    cudaGraph_t mtp_chain_graph_ = nullptr;
    cudaGraphExec_t mtp_chain_exec_ = nullptr;
    cudaGraph_t mtp_stem_chain_graph_ = nullptr;
    cudaGraphExec_t mtp_stem_chain_exec_ = nullptr;
    cudaGraph_t mtp_hin_chain_graph_ = nullptr;
    cudaGraphExec_t mtp_hin_chain_exec_ = nullptr;
    cudaGraph_t mtp_hin_rec4_graph_ = nullptr;
    cudaGraphExec_t mtp_hin_rec4_exec_ = nullptr;
    int* d_mtp_drafts_ = nullptr;
    int* d_mtp_best_side_ = nullptr;
    int* d_mtp_slot_ = nullptr;
    int* d_mtp_slot_best_ = nullptr;
    int mtp_pf_slots_ = 0;
    int32_t mtp_h31_in_ = -1;
    bool mtp_have_h31b_ = false;
    bool mtp_have_h31c_ = false;
    int32_t mtp_h31b_in_ = -1;
    int32_t mtp_h31c_in_ = -1;
    bool mtp_have_cont1_ = false;
    bool mtp_have_cont2_ = false;
    bool mtp_cont_clean_ = false;
    int32_t mtp_cont1_in_ = -1;
    int32_t mtp_cont2_in_ = -1;
    float* d_mtp_logits_ = nullptr;
    float* d_mtp_amax_ = nullptr;
    int* d_mtp_aidx_ = nullptr;
    float* d_mtp_post_side_ = nullptr;
    float* d_mtp_hin_side_ = nullptr;
    float* d_mtp_h_side_ = nullptr;
    float* d_mtp_seed_side_ = nullptr;
    int* d_mtp_tok_side_ = nullptr;
    int* d_mtp_pos_side_ = nullptr;
    float* d_mtp_kc_side_ = nullptr;
    float* d_mtp_vc_side_ = nullptr;
    int8_t* d_mtp_xq_ = nullptr;
    __half* d_mtp_xsc_ = nullptr;
    int32_t* d_mtp_xsum_ = nullptr;
    int mtp_xq_n_ = 0;
    cudaGraph_t mtp_hin_side_graph_ = nullptr;
    cudaGraphExec_t mtp_hin_side_exec_ = nullptr;
    cudaGraph_t mtp_rec4_side_graph_ = nullptr;
    cudaGraphExec_t mtp_rec4_side_exec_ = nullptr;
    cudaEvent_t mtp_side_ev_ = nullptr;
    cudaEvent_t mtp_slot0_ev_ = nullptr;
    cudaEvent_t mtp_slot1_ev_ = nullptr;
    cudaEvent_t mtp_pf_l20_ev_ = nullptr;
    cudaEvent_t mtp_pf_l40_ev_ = nullptr;
    bool mtp_side_pending_ = false;
    int mtp_side_kind_ = 0;
    cudaGraph_t mtp_t2_graph_ = nullptr;
    cudaGraphExec_t mtp_t2_exec_ = nullptr;
    cudaGraph_t mtp_t4_graph_ = nullptr;
    cudaGraphExec_t mtp_t4_exec_ = nullptr;
    bool mtp_t4_ran_ = false;
    cudaStream_t bak_stream_ = nullptr;
    cudaStream_t bak2_stream_ = nullptr;
    cudaEvent_t mtp_side2_ev_ = nullptr;
    cudaGraph_t mtp_rec4_b2_graph_ = nullptr;
    cudaGraphExec_t mtp_rec4_b2_exec_ = nullptr;
    cudaGraph_t mtp_hin_b2_graph_ = nullptr;
    cudaGraphExec_t mtp_hin_b2_exec_ = nullptr;
    int* d_mtp_drafts2_ = nullptr;
    int* d_mtp_best2_ = nullptr;
    float* d_mtp_post2_ = nullptr;
    float* d_mtp_hin2_ = nullptr;
    float* d_mtp_h2_ = nullptr;
    int* d_mtp_tok2_ = nullptr;
    int* d_mtp_pos2_ = nullptr;
    float* d_mtp_kc2_ = nullptr;
    float* d_mtp_vc2_ = nullptr;
    float* d_mtp_logits2_ = nullptr;
    float* d_mtp_amax2_ = nullptr;
    int* d_mtp_aidx2_ = nullptr;
    int8_t* d_mtp_xq2_ = nullptr;
    __half* d_mtp_xsc2_ = nullptr;
    int32_t* d_mtp_xsum2_ = nullptr;
    float* d_mtp_seed2_ = nullptr;
    float* d_mtp_cat2_ = nullptr;
    float* d_mtp_qg2_ = nullptr;
    float* d_mtp_q2_ = nullptr;
    float* d_mtp_gate2_ = nullptr;
    float* d_mtp_o2_ = nullptr;
    float* d_mtp_k2_ = nullptr;
    float* d_mtp_vtmp2_ = nullptr;
    float* d_mtp_gmlp2_ = nullptr;
    float* d_mtp_up2_ = nullptr;
    bool mtp_side2_pending_ = false;
    bool mtp_sides_joined_ = false;
    cudaGraph_t mtp_quad_hin_graph_ = nullptr;
    cudaGraphExec_t mtp_quad_hin_exec_ = nullptr;
    float* d_mtp_quad_h_ = nullptr;
    float* d_mtp_quad_hin_ = nullptr;
    float* d_mtp_quad_post_ = nullptr;
    float* d_mtp_quad_cat_ = nullptr;
    float* d_mtp_quad_y_ = nullptr;
    float* d_mtp_quad_xn_ = nullptr;
    float* d_mtp_quad_qg_ = nullptr;
    float* d_mtp_quad_q_ = nullptr;
    float* d_mtp_quad_gate_ = nullptr;
    float* d_mtp_quad_o_ = nullptr;
    float* d_mtp_quad_k_ = nullptr;
    float* d_mtp_quad_v_ = nullptr;
    float* d_mtp_quad_gmlp_ = nullptr;
    float* d_mtp_quad_up_ = nullptr;
    float* d_mtp_quad_kc_ = nullptr;
    float* d_mtp_quad_vc_ = nullptr;
    float* d_mtp_quad_logits_ = nullptr;
    float* d_mtp_quad_amax_ = nullptr;
    int* d_mtp_quad_aidx_ = nullptr;
    int* d_mtp_quad_toks_ = nullptr;
    int* d_mtp_quad_pos_ = nullptr;
    int* d_mtp_quad_drafts_ = nullptr;
    int* d_mtp_quad_best_ = nullptr;
    int8_t* d_mtp_quad_xq_ = nullptr;
    __half* d_mtp_quad_xsc_ = nullptr;
    int32_t* d_mtp_quad_xsum_ = nullptr;
    int mtp_quad_xq_n_ = 0;
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
