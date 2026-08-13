#include "rapidllm/kernels/ops.h"
#include "rapidllm/kernels/cpu_caps.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rapidllm::ops {

float silu(float x) { return x / (1.f + std::exp(-x)); }

float fp16_to_f32(uint16_t h) {
    const uint32_t s = (h >> 15) & 1u;
    const uint32_t e = (h >> 10) & 0x1fu;
    const uint32_t m = h & 0x3ffu;
    uint32_t out;
    if (e == 0) {
        if (m == 0) out = s << 31;
        else {
            uint32_t exp = 127 - 15 + 1;
            uint32_t mant = m;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ffu;
            out = (s << 31) | (exp << 23) | (mant << 13);
        }
    } else if (e == 31) {
        out = (s << 31) | 0x7f800000u | (m << 13);
    } else {
        out = (s << 31) | ((e + (127 - 15)) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

uint16_t f32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t s = (x >> 31) & 1u;
    int32_t e = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) {
        if (e < -10) return static_cast<uint16_t>(s << 15);
        m = (m | 0x800000u) >> (1 - e);
        return static_cast<uint16_t>((s << 15) | (m >> 13));
    }
    if (e >= 31) return static_cast<uint16_t>((s << 15) | 0x7c00u);
    return static_cast<uint16_t>((s << 15) | (static_cast<uint32_t>(e) << 10) | (m >> 13));
}

float e4m3_to_f32(uint8_t b) {
    const int s = (b >> 7) & 1;
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    if (e == 0) {
        if (m == 0) return s ? -0.f : 0.f;
        const float frac = static_cast<float>(m) / 8.f;
        const float v = std::ldexp(frac, 1 - 7);
        return s ? -v : v;
    }
    if (e == 15 && m == 7) return std::numeric_limits<float>::quiet_NaN();
    const float frac = 1.f + static_cast<float>(m) / 8.f;
    const float v = std::ldexp(frac, e - 7);
    return s ? -v : v;
}

uint8_t f32_to_e4m3(float f) {
    if (!std::isfinite(f)) return 0x7F;
    const int s = f < 0 ? 1 : 0;
    f = std::fabs(f);
    if (f == 0.f) return static_cast<uint8_t>(s << 7);
    int exp;
    float frac = std::frexp(f, &exp); // frac in [0.5, 1)
    frac *= 2.f;
    --exp;
    int e = exp + 7;
    if (e <= 0) {
        const float sub = f / std::ldexp(1.f, 1 - 7);
        int mm = static_cast<int>(std::lround(sub * 8.f));
        if (mm <= 0) return static_cast<uint8_t>(s << 7);
        if (mm > 7) mm = 7;
        return static_cast<uint8_t>((s << 7) | mm);
    }
    if (e >= 15) return static_cast<uint8_t>((s << 7) | 0x78 | 0x6);
    int mm = static_cast<int>(std::lround((frac - 1.f) * 8.f));
    if (mm == 8) {
        mm = 0;
        ++e;
        if (e >= 15) return static_cast<uint8_t>((s << 7) | 0x78 | 0x6);
    }
    return static_cast<uint8_t>((s << 7) | (e << 3) | (mm & 7));
}

void rmsnorm(const float* x, const float* w, float* y, int n, float eps) {
    double acc = 0;
    for (int i = 0; i < n; ++i) acc += static_cast<double>(x[i]) * x[i];
    const float inv = 1.f / std::sqrt(static_cast<float>(acc / n) + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * inv * (w ? w[i] : 1.f);
}

void qwen3_rmsnorm(const float* x, const float* gamma, float* y, int n, float eps) {
    if (n >= 64) {
        switch (active_simd()) {
        case SimdKind::Avx512:
            qwen3_rmsnorm_avx512(x, gamma, y, n, eps);
            return;
        case SimdKind::Avx2:
            qwen3_rmsnorm_avx2(x, gamma, y, n, eps);
            return;
        default:
            break;
        }
    }
    double acc = 0;
    for (int i = 0; i < n; ++i) acc += static_cast<double>(x[i]) * x[i];
    const float inv = 1.f / std::sqrt(static_cast<float>(acc / n) + eps);
    for (int i = 0; i < n; ++i) {
        const float g = gamma ? (1.f + gamma[i]) : 1.f;
        y[i] = x[i] * inv * g;
    }
}

void gated_rmsnorm(const float* x, const float* z, const float* gamma, float* y, int n, float eps,
                   int gamma_n) {
    if (gamma_n <= 0) gamma_n = n;
    double acc = 0;
    for (int i = 0; i < n; ++i) acc += static_cast<double>(x[i]) * x[i];
    const float inv = 1.f / std::sqrt(static_cast<float>(acc / n) + eps);
    for (int i = 0; i < n; ++i) {
        const float g = gamma ? gamma[i % gamma_n] : 1.f;
        y[i] = g * (x[i] * inv) * silu(z[i]);
    }
}

void l2norm(const float* x, float* y, int n, float eps) {
    double acc = 0;
    for (int i = 0; i < n; ++i) acc += static_cast<double>(x[i]) * x[i];
    const float inv = 1.f / std::sqrt(static_cast<float>(acc) + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * inv;
}

void gemv_f32_scalar(const float* W, const float* x, float* y, int m, int n) {
    for (int i = 0; i < m; ++i) {
        double acc = 0;
        const float* row = W + static_cast<size_t>(i) * n;
        for (int j = 0; j < n; ++j) acc += static_cast<double>(row[j]) * x[j];
        y[i] = static_cast<float>(acc);
    }
}

void gemv_f32(const float* W, const float* x, float* y, int m, int n) {
    gemv_f32_simd(W, x, y, m, n);
}

void gemm_f32(const float* W, const float* X, float* Y, int m, int n, int batch) {
    for (int b = 0; b < batch; ++b) gemv_f32(W, X + static_cast<size_t>(b) * n, Y + static_cast<size_t>(b) * m, m, n);
}

void gemv_fp8(const uint8_t* W, const float* scale, const float* x, float* y, int m, int n, int block) {
    const int nb_n = (n + block - 1) / block;
    for (int i = 0; i < m; ++i) {
        double acc = 0;
        const int bi = i / block;
        for (int j = 0; j < n; ++j) {
            const int bj = j / block;
            const float s = scale[bi * nb_n + bj];
            acc += static_cast<double>(e4m3_to_f32(W[static_cast<size_t>(i) * n + j])) * s * x[j];
        }
        y[i] = static_cast<float>(acc);
    }
}

static constexpr int QK8_0 = 32;
static constexpr int QK_K = 256;

void dequant_q8_0(const uint8_t* packed, float* out, int n) {
    const int nb = n / QK8_0;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = packed + b * (2 + QK8_0);
        uint16_t dh;
        std::memcpy(&dh, blk, 2);
        const float d = fp16_to_f32(dh);
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
        for (int i = 0; i < QK8_0; ++i) out[b * QK8_0 + i] = d * static_cast<float>(qs[i]);
    }
    for (int i = nb * QK8_0; i < n; ++i) out[i] = 0;
}

void gemv_q8_0_scalar(const uint8_t* packed, const float* x, float* y, int m, int n) {
    // Dispatched / threaded in cpu_caps.cpp (same symbol via wrapper below if needed).
    // Scalar fused dequant-dot kept as fallback name gemv_q8_0_scalar — see cpu_caps.
    const int nb = n / QK8_0;
    const size_t bsz = 2 + QK8_0;
    for (int i = 0; i < m; ++i) {
        const uint8_t* row = packed + static_cast<size_t>(i) * nb * bsz;
        float acc = 0.f;
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = row + b * bsz;
            uint16_t dh;
            std::memcpy(&dh, blk, 2);
            const float d = fp16_to_f32(dh);
            const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
            const float* xb = x + b * QK8_0;
            float s = 0.f;
            for (int k = 0; k < QK8_0; ++k) s += static_cast<float>(qs[k]) * xb[k];
            acc += d * s;
        }
        y[i] = acc;
    }
}

void gemv_f16_scalar(const uint16_t* W, const float* x, float* y, int m, int n) {
    for (int i = 0; i < m; ++i) {
        const uint16_t* row = W + static_cast<size_t>(i) * n;
        float acc = 0.f;
        for (int j = 0; j < n; ++j) acc += fp16_to_f32(row[j]) * x[j];
        y[i] = acc;
    }
}

// Q4_K public superblock: 256 weights, FP16 d/dmin, 12-byte 6-bit scales/mins, 128 bytes of nibbles.
static void q4k_scale_min(const uint8_t* sc, int j, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = sc[j] & 63;
        *m = sc[j + 4] & 63;
    } else {
        *d = static_cast<uint8_t>((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4));
        *m = static_cast<uint8_t>((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4));
    }
}

void dequant_q4_k(const uint8_t* packed, float* out, int n) {
    constexpr int bsz = 2 + 2 + 12 + QK_K / 2;
    const int nb = n / QK_K;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = packed + b * bsz;
        uint16_t dh, dm;
        std::memcpy(&dh, blk, 2);
        std::memcpy(&dm, blk + 2, 2);
        const float d = fp16_to_f32(dh);
        const float minv = fp16_to_f32(dm);
        const uint8_t* sc = blk + 4;
        const uint8_t* q = blk + 16;
        float* y = out + b * QK_K;
        int is = 0;
        for (int grp = 0; grp < 4; ++grp) {
            uint8_t s0, m0, s1, m1;
            q4k_scale_min(sc, is + 0, &s0, &m0);
            q4k_scale_min(sc, is + 1, &s1, &m1);
            const float d1 = d * s0;
            const float m1f = minv * m0;
            const float d2 = d * s1;
            const float m2f = minv * m1;
            for (int l = 0; l < 32; ++l) y[l] = d1 * (q[l] & 0xF) - m1f;
            for (int l = 0; l < 32; ++l) y[32 + l] = d2 * (q[l] >> 4) - m2f;
            q += 32;
            y += 64;
            is += 2;
        }
    }
}

static float dot256(const float* a, const float* b) {
    float s = 0.f;
    for (int i = 0; i < QK_K; ++i) s += a[i] * b[i];
    return s;
}

void gemv_q4_k_scalar(const uint8_t* packed, const float* x, float* y, int m, int n) {
    constexpr int bsz = 2 + 2 + 12 + QK_K / 2;
    const int nb = n / QK_K;
    float tmp[QK_K];
    for (int i = 0; i < m; ++i) {
        const uint8_t* row = packed + static_cast<size_t>(i) * nb * bsz;
        float acc = 0.f;
        for (int b = 0; b < nb; ++b) {
            dequant_q4_k(row + b * bsz, tmp, QK_K);
            acc += dot256(tmp, x + b * QK_K);
        }
        y[i] = acc;
    }
}

void dequant_q5_k(const uint8_t* packed, float* out, int n) {
    constexpr int bsz = 2 + 2 + 12 + QK_K / 8 + QK_K / 2;
    const int nb = n / QK_K;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = packed + b * bsz;
        uint16_t dh, dm;
        std::memcpy(&dh, blk, 2);
        std::memcpy(&dm, blk + 2, 2);
        const float d = fp16_to_f32(dh);
        const float minv = fp16_to_f32(dm);
        const uint8_t* sc = blk + 4;
        const uint8_t* qh = blk + 16;
        const uint8_t* ql = blk + 48;
        float* y = out + b * QK_K;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int grp = 0; grp < 4; ++grp) {
            uint8_t s0, m0, s1, m1;
            q4k_scale_min(sc, is + 0, &s0, &m0);
            q4k_scale_min(sc, is + 1, &s1, &m1);
            const float d1 = d * s0, m1f = minv * m0;
            const float d2 = d * s1, m2f = minv * m1;
            for (int l = 0; l < 32; ++l)
                y[l] = d1 * static_cast<float>((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1f;
            for (int l = 0; l < 32; ++l)
                y[32 + l] = d2 * static_cast<float>((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2f;
            y += 64;
            ql += 32;
            is += 2;
            u1 = static_cast<uint8_t>(u1 << 2);
            u2 = static_cast<uint8_t>(u2 << 2);
        }
    }
}

void gemv_q5_k_scalar(const uint8_t* packed, const float* x, float* y, int m, int n) {
    constexpr int bsz = 2 + 2 + 12 + QK_K / 8 + QK_K / 2;
    const int nb = n / QK_K;
    float tmp[QK_K];
    for (int i = 0; i < m; ++i) {
        const uint8_t* row = packed + static_cast<size_t>(i) * nb * bsz;
        float acc = 0.f;
        for (int b = 0; b < nb; ++b) {
            dequant_q5_k(row + b * bsz, tmp, QK_K);
            acc += dot256(tmp, x + b * QK_K);
        }
        y[i] = acc;
    }
}

void dequant_q6_k(const uint8_t* packed, float* out, int n) {
    constexpr int bsz = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
    const int nb = n / QK_K;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = packed + b * bsz;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + QK_K / 2;
        const int8_t* sc = reinterpret_cast<const int8_t*>(blk + QK_K / 2 + QK_K / 4);
        uint16_t dh;
        std::memcpy(&dh, blk + QK_K / 2 + QK_K / 4 + QK_K / 16, 2);
        const float d = fp16_to_f32(dh);
        float* y = out + b * QK_K;
        for (int n128 = 0; n128 < QK_K; n128 += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = static_cast<int>((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = static_cast<int>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = static_cast<int>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l] = d * static_cast<float>(sc[is]) * static_cast<float>(q1);
                y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
                y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
                y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void gemv_q6_k_scalar(const uint8_t* packed, const float* x, float* y, int m, int n) {
    constexpr int bsz = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
    const int nb = n / QK_K;
    float tmp[QK_K];
    for (int i = 0; i < m; ++i) {
        const uint8_t* row = packed + static_cast<size_t>(i) * nb * bsz;
        float acc = 0.f;
        for (int b = 0; b < nb; ++b) {
            dequant_q6_k(row + b * bsz, tmp, QK_K);
            acc += dot256(tmp, x + b * QK_K);
        }
        y[i] = acc;
    }
}

void linear(QuantKind q, const uint8_t* w, const float* scale, const float* x, float* y, int m, int n) {
    switch (q) {
    case QuantKind::F32:
        gemv_f32(reinterpret_cast<const float*>(w), x, y, m, n);
        break;
    case QuantKind::F16:
        gemv_f16(reinterpret_cast<const uint16_t*>(w), x, y, m, n);
        break;
    case QuantKind::BF16:
        gemv_bf16(reinterpret_cast<const uint16_t*>(w), x, y, m, n);
        break;
    case QuantKind::FP8_E4M3_B128:
        if (!scale) throw std::runtime_error("FP8 GEMV missing scale");
        gemv_fp8(w, scale, x, y, m, n, 128);
        break;
    case QuantKind::Q8_0:
        gemv_q8_0(w, x, y, m, n);
        break;
    case QuantKind::Q4_K:
        gemv_q4_k(w, x, y, m, n);
        break;
    case QuantKind::Q5_K:
        gemv_q5_k(w, x, y, m, n);
        break;
    case QuantKind::Q6_K:
        gemv_q6_k(w, x, y, m, n);
        break;
    case QuantKind::IQUnknown: {
        throw std::runtime_error("quant kind requires one-shot dequant not wired in linear()");
    }
    default:
        throw std::runtime_error("unsupported QuantKind in linear()");
    }
}

void linear(const QuantW& w, const float* x, float* y, int m, int n) {
    linear(w.quant, w.data, w.scale, x, y, m, n);
}

void conv1d_causal_prefill(const float* x, const float* w, float* y, int seq, int dim, int k) {
    // x: [seq, dim], w: [dim, k] (tap 0 = oldest)
    for (int t = 0; t < seq; ++t) {
        for (int c = 0; c < dim; ++c) {
            double acc = 0;
            for (int p = 0; p < k; ++p) {
                const int src = t - (k - 1 - p);
                const float xv = src >= 0 ? x[src * dim + c] : 0.f;
                acc += static_cast<double>(w[c * k + p]) * xv;
            }
            y[t * dim + c] = silu(static_cast<float>(acc));
        }
    }
}

void conv1d_update(const float* x_t, const float* w, float* state, float* y, int dim, int k) {
    // state [dim, k], shift left, append x_t
    for (int c = 0; c < dim; ++c) {
        float* st = state + c * k;
        for (int p = 0; p < k - 1; ++p) st[p] = st[p + 1];
        st[k - 1] = x_t[c];
        double acc = 0;
        for (int p = 0; p < k; ++p) acc += static_cast<double>(w[c * k + p]) * st[p];
        y[c] = silu(static_cast<float>(acc));
    }
}

void rope_partial(float* q, float* k, int n_q, int n_kv, int head_dim, int rotary_dim, int pos, float theta) {
    auto apply = [&](float* t, int n_heads) {
        for (int h = 0; h < n_heads; ++h) {
            float* v = t + h * head_dim;
            for (int i = 0; i < rotary_dim / 2; ++i) {
                const float freq = 1.f / std::pow(theta, static_cast<float>(i) / (rotary_dim / 2));
                const float ang = static_cast<float>(pos) * freq;
                const float c = std::cos(ang), s = std::sin(ang);
                const float x0 = v[2 * i], x1 = v[2 * i + 1];
                v[2 * i] = x0 * c - x1 * s;
                v[2 * i + 1] = x0 * s + x1 * c;
            }
        }
    };
    apply(q, n_q);
    apply(k, n_kv);
}

void delta_recurrent_step_scalar(float* S, const float* q_in, const float* k_in, const float* v,
                                 const float* beta, const float* g_log, float* o,
                                 int n_v, int dk, int dv, float eps_l2) {
    std::vector<float> q(static_cast<size_t>(n_v) * dk);
    std::vector<float> k(static_cast<size_t>(n_v) * dk);
    for (int h = 0; h < n_v; ++h) {
        l2norm(q_in + h * dk, q.data() + h * dk, dk, eps_l2);
        l2norm(k_in + h * dk, k.data() + h * dk, dk, eps_l2);
        const float scale = 1.f / std::sqrt(static_cast<float>(dk));
        for (int i = 0; i < dk; ++i) q[h * dk + i] *= scale;
    }
    for (int h = 0; h < n_v; ++h) {
        const float g = std::exp(g_log[h]);
        float* Sh = S + static_cast<size_t>(h) * dk * dv;
        for (int i = 0; i < dk * dv; ++i) Sh[i] *= g;
        std::vector<float> kv(static_cast<size_t>(dv), 0.f);
        const float* kh = k.data() + h * dk;
        const float* qh = q.data() + h * dk;
        const float* vh = v + h * dv;
        for (int i = 0; i < dk; ++i) {
            const float ki = kh[i];
            const float* srow = Sh + i * dv;
            for (int j = 0; j < dv; ++j) kv[j] += srow[j] * ki;
        }
        std::vector<float> delta(static_cast<size_t>(dv));
        for (int j = 0; j < dv; ++j) delta[j] = (vh[j] - kv[j]) * beta[h];
        for (int i = 0; i < dk; ++i) {
            float* srow = Sh + i * dv;
            const float ki = kh[i];
            for (int j = 0; j < dv; ++j) srow[j] += ki * delta[j];
        }
        float* oh = o + h * dv;
        std::fill(oh, oh + dv, 0.f);
        for (int i = 0; i < dk; ++i) {
            const float qi = qh[i];
            const float* srow = Sh + i * dv;
            for (int j = 0; j < dv; ++j) oh[j] += srow[j] * qi;
        }
    }
}

void delta_recurrent_step(float* S, const float* q, const float* k, const float* v,
                          const float* beta, const float* g_log, float* o,
                          int n_v, int dk, int dv, float eps_l2) {
    delta_recurrent_step_scalar(S, q, k, v, beta, g_log, o, n_v, dk, dv, eps_l2);
}

void attn_prefill(const float* q, const float* k, const float* v, float* o,
                  int seq, int n_q, int n_kv, int head_dim) {
    const int rep = n_q / n_kv;
    const float scale = 1.f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(static_cast<size_t>(seq));
    for (int t = 0; t < seq; ++t) {
        for (int hq = 0; hq < n_q; ++hq) {
            const int hkv = hq / rep;
            const float* qt = q + (static_cast<size_t>(t) * n_q + hq) * head_dim;
            float m = -1e30f;
            for (int s = 0; s <= t; ++s) {
                const float* ks = k + (static_cast<size_t>(s) * n_kv + hkv) * head_dim;
                double acc = 0;
                for (int d = 0; d < head_dim; ++d) acc += static_cast<double>(qt[d]) * ks[d];
                scores[s] = static_cast<float>(acc) * scale;
                m = std::max(m, scores[s]);
            }
            double z = 0;
            for (int s = 0; s <= t; ++s) {
                scores[s] = std::exp(scores[s] - m);
                z += scores[s];
            }
            float* ot = o + (static_cast<size_t>(t) * n_q + hq) * head_dim;
            std::fill(ot, ot + head_dim, 0.f);
            for (int s = 0; s <= t; ++s) {
                const float a = static_cast<float>(scores[s] / z);
                const float* vs = v + (static_cast<size_t>(s) * n_kv + hkv) * head_dim;
                for (int d = 0; d < head_dim; ++d) ot[d] += a * vs[d];
            }
        }
    }
}

void attn_decode(const float* q, const float* k_cache, const float* v_cache, float* o,
                 int pos, int n_q, int n_kv, int head_dim) {
    const int seq = pos + 1;
    const int rep = n_q / n_kv;
    const float scale = 1.f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(static_cast<size_t>(seq));
    for (int hq = 0; hq < n_q; ++hq) {
        const int hkv = hq / rep;
        const float* qt = q + hq * head_dim;
        float m = -1e30f;
        for (int s = 0; s < seq; ++s) {
            const float* ks = k_cache + (static_cast<size_t>(s) * n_kv + hkv) * head_dim;
            double acc = 0;
            for (int d = 0; d < head_dim; ++d) acc += static_cast<double>(qt[d]) * ks[d];
            scores[s] = static_cast<float>(acc) * scale;
            m = std::max(m, scores[s]);
        }
        double z = 0;
        for (int s = 0; s < seq; ++s) {
            scores[s] = std::exp(scores[s] - m);
            z += scores[s];
        }
        float* ot = o + hq * head_dim;
        std::fill(ot, ot + head_dim, 0.f);
        for (int s = 0; s < seq; ++s) {
            const float a = static_cast<float>(scores[s] / z);
            const float* vs = v_cache + (static_cast<size_t>(s) * n_kv + hkv) * head_dim;
            for (int d = 0; d < head_dim; ++d) ot[d] += a * vs[d];
        }
    }
}

void swiglu(const float* gate, const float* up, float* y, int n) {
    for (int i = 0; i < n; ++i) y[i] = silu(gate[i]) * up[i];
}

} // namespace rapidllm::ops
