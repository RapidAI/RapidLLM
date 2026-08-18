#include "rapidllm/kernels/mtp_attn_decode.h"

#include <cuda_runtime.h>

namespace rapidllm::cuda_mtp {
namespace {

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

__device__ __forceinline__ float warp_sum(float x) {
    for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xffffffff, x, off);
    return x;
}

// q/k RMS + (1+g) + RoPE, store float KV at cache[p], attend t_lo..min(p,ctx-1).
__global__ void mtp_attn_decode_k(float* q, const float* k, const float* v, const float* q_norm,
                                  const float* k_norm, float* k_cache, float* v_cache, float* o,
                                  const int* pos, int n_q, int n_kv, int hd, int rotary, float theta,
                                  float eps, int ctx, int t_lo) {
    const int hq = blockIdx.x;
    if (hq >= n_q || n_kv <= 0 || hd <= 0 || ctx <= 0 || !pos) return;
    const int p = *pos;
    if (p < 0 || p >= ctx) return;
    const int kn = n_kv * hd;
    const int rep = n_q / n_kv;
    const int hkv = hq / rep;
    const int t0 = t_lo < 0 ? 0 : t_lo;
    const int Tend = p + 1;
    extern __shared__ float sm[];
    float* scores = sm;
    float* khloc = sm + ctx;
    __shared__ float inv, wss[8];

    float* qh = q + hq * hd;
    float ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += qh[i] * qh[i];
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
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        const float g = q_norm ? (1.f + q_norm[i]) : 1.f;
        qh[i] *= inv * g;
    }
    __syncthreads();
    rope_apply(qh, rotary, static_cast<float>(p), theta, hd);
    __syncthreads();

    const float* ksrc = k + hkv * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) khloc[i] = ksrc[i];
    __syncthreads();
    ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += khloc[i] * khloc[i];
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
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        const float g = k_norm ? (1.f + k_norm[i]) : 1.f;
        khloc[i] *= inv * g;
    }
    __syncthreads();
    rope_apply(khloc, rotary, static_cast<float>(p), theta, hd);
    __syncthreads();

    float* kc = k_cache + static_cast<size_t>(p) * kn + hkv * hd;
    float* vc = v_cache + static_cast<size_t>(p) * kn + hkv * hd;
    const float* vs = v + hkv * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        kc[i] = khloc[i];
        vc[i] = vs[i];
    }
    __syncthreads();

    const float scale = rsqrtf(static_cast<float>(hd));
    const int t_hi = Tend < ctx ? Tend : ctx;
    for (int t = threadIdx.x; t < t_hi; t += blockDim.x) {
        if (t < t0) {
            scores[t] = -1e30f;
            continue;
        }
        const float* kh = k_cache + static_cast<size_t>(t) * kn + hkv * hd;
        float dot = 0.f;
        for (int d = 0; d < hd; ++d) dot += qh[d] * kh[d];
        scores[t] = dot * scale;
    }
    __syncthreads();
    __shared__ float mx, sumv;
    if (threadIdx.x == 0) {
        float mm = -1e30f;
        for (int t = t0; t < t_hi; ++t) mm = fmaxf(mm, scores[t]);
        mx = mm;
        float s = 0.f;
        for (int t = t0; t < t_hi; ++t) {
            scores[t] = expf(scores[t] - mx);
            s += scores[t];
        }
        sumv = s > 0.f ? s : 1.f;
    }
    __syncthreads();
    float* oh = o + hq * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.f;
        for (int t = t0; t < t_hi; ++t) {
            const float* vh = v_cache + static_cast<size_t>(t) * kn + hkv * hd;
            acc += (scores[t] / sumv) * vh[d];
        }
        oh[d] = acc;
    }
}

} // namespace

void launch_mtp_attn_decode(float* q, const float* k, const float* v, const float* q_norm,
                            const float* k_norm, float* k_cache, float* v_cache, float* o,
                            const int* pos, int n_q, int n_kv, int hd, int rotary, float theta,
                            float eps, int ctx, int t_lo) {
    if (n_q <= 0 || hd <= 0 || ctx <= 0) return;
    const int ath = hd >= 256 ? 256 : 128;
    const size_t sm = (static_cast<size_t>(ctx) + static_cast<size_t>(hd)) * sizeof(float);
    mtp_attn_decode_k<<<n_q, ath, sm>>>(q, k, v, q_norm, k_norm, k_cache, v_cache, o, pos, n_q, n_kv, hd,
                                        rotary, theta, eps, ctx, t_lo);
}

} // namespace rapidllm::cuda_mtp
