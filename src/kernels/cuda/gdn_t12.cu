// T=12 fused GDN. Isolated TU so cuda_engine.cu T=2/T=3/T=4/T=6 occupancy is unchanged.
#include "rapidllm/kernels/gemv_q6_t4.h"

#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float silu_d(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float sigmoid_d(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float softplus_d(float x) { return log1pf(expf(-fabsf(x))) + fmaxf(x, 0.f); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float bf16_to_f32(uint16_t h) {
    return __uint_as_float(static_cast<uint32_t>(h) << 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

__device__ __forceinline__ int gdn_src_h(int h, int nk, int nv, int v_tiled) {
    return v_tiled ? (h % nk) : (h / (nv / nk));
}

__device__ __forceinline__ bool gdn_qk_conv_owner(int h, int nk, int nv, int v_tiled) {
    return v_tiled ? (h < nk) : ((h % (nv / nk)) == 0);
}

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

__global__ void __launch_bounds__(256, 1) gdn_decode_t12_k(const float* aa, const float* bb, uint16_t* S,
                                                           const float* A_log, const float* dt_bias, float* o,
                                                           int nk, int nv, const float* qkv_raw,
                                                           const float* conv_w, float* conv_st, uint16_t* S_bak,
                                                           float* conv_bak, int qkv_dim, int v_tiled) {
    constexpr int NT = 12;
    constexpr int dk = 128, dv = 128;
    const int h = blockIdx.x;
    if (h >= nv) return;
    extern __shared__ float sm[];
    float* q = sm;
    float* k = sm + dk;
    float* Sp = sm + 2 * dk;
    uint16_t* Sh = S + static_cast<size_t>(h) * dk * dv;
    const int qdim = nk * dk;
    const int src = gdn_src_h(h, nk, nv, v_tiled);
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
        if (gdn_qk_conv_owner(h, nk, nv, v_tiled)) {
            for (int i = threadIdx.x; i < dk; i += blockDim.x) {
                conv1d_push(qkv, conv_st, src * dk + i, 4);
                conv1d_push(qkv, conv_st, qdim + src * dk + i, 4);
            }
        }
        __syncthreads();
        if (t == 0 && S_bak && conv_bak) {
            uint16_t* Sb = S_bak + static_cast<size_t>(h) * dk * dv;
            gdn_s_store_bf16(Sb, Sp, dk * dv);
            if (gdn_qk_conv_owner(h, nk, nv, v_tiled)) {
                for (int i = threadIdx.x; i < dk; i += blockDim.x) {
                    conv_st_copy_ch(conv_bak, conv_st, src * dk + i, 4);
                    conv_st_copy_ch(conv_bak, conv_st, qdim + src * dk + i, 4);
                }
            }
            for (int i = threadIdx.x; i < dv; i += blockDim.x)
                conv_st_copy_ch(conv_bak, conv_st, 2 * qdim + h * dv + i, 4);
        }
        // Match 12× T=1: round S through bf16 after every token.
        gdn_s_store_bf16(Sh, Sp, dk * dv);
        __syncthreads();
        gdn_s_load_f32(Sp, Sh, dk * dv);
        __syncthreads();
    }
}

} // namespace

namespace rapidllm::cuda_gemv {

void warmup_gdn_decode_t12() {
    cudaFuncSetAttribute(gdn_decode_t12_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 1024);
}

void launch_gdn_decode_t12(const float* aa, const float* bb, uint16_t* S, const float* A_log,
                           const float* dt_bias, float* o, int nk, int nv, const float* qkv_raw,
                           const float* conv_w, float* conv_st, uint16_t* S_bak, float* conv_bak,
                           int qkv_dim, int v_tiled) {
    if (!aa || !bb || !S || !A_log || !dt_bias || !o || !qkv_raw || !conv_w || !conv_st || nv <= 0)
        return;
    const int sm = (2 * 128 + 128 * 128) * static_cast<int>(sizeof(float));
    gdn_decode_t12_k<<<nv, 256, sm>>>(aa, bb, S, A_log, dt_bias, o, nk, nv, qkv_raw, conv_w, conv_st, S_bak,
                                      conv_bak, qkv_dim, v_tiled);
}

} // namespace rapidllm::cuda_gemv
