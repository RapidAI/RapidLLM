#include "rapidllm/kernels/ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace rapidllm::ops {
namespace {

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
float softplus(float x) { return std::log1p(std::exp(-std::fabs(x))) + std::max(x, 0.f); }

struct Scratch {
    std::vector<float> buf;
    size_t off = 0;
    void reset() { off = 0; }
    void reserve(size_t n) {
        if (n > buf.size()) buf.resize(n);
    }
    float* alloc(size_t n) {
        if (off + n > buf.size())
            throw std::runtime_error("fused scratch overflow");
        float* p = buf.data() + off;
        off += n;
        return p;
    }
};

Scratch& tls_scratch() {
    thread_local Scratch s;
    return s;
}

void gemv_w(bool simd, const QuantW& w, const float* x, float* y, int m, int n) {
    if (!simd) {
        switch (w.quant) {
        case QuantKind::F32:
            gemv_f32_scalar(reinterpret_cast<const float*>(w.data), x, y, m, n);
            return;
        case QuantKind::F16:
            gemv_f16_scalar(reinterpret_cast<const uint16_t*>(w.data), x, y, m, n);
            return;
        case QuantKind::Q8_0:
            gemv_q8_0_scalar(w.data, x, y, m, n);
            return;
        default:
            break;
        }
    }
    linear(w, x, y, m, n);
}

} // namespace

void fused_swiglu(const float* gate, const float* up, float* y, int n) {
    swiglu(gate, up, y, n);
}

void fused_rmsnorm_residual(const float* x, const float* y, const float* w, float* out, int n, float eps) {
    qwen3_rmsnorm(x, w, out, n, eps);
    (void)y;
}

void fused_mlp_decode(const FusedMlpArgs& a) {
    Scratch& sc = tls_scratch();
    const size_t H = static_cast<size_t>(a.hidden);
    const size_t I = static_cast<size_t>(a.intermediate);
    sc.reserve(H + I + I + H);
    sc.reset();
    float* xn = sc.alloc(H);
    float* g = sc.alloc(static_cast<size_t>(a.intermediate));
    float* u = sc.alloc(static_cast<size_t>(a.intermediate));
    float* y = sc.alloc(static_cast<size_t>(a.hidden));
    qwen3_rmsnorm(a.x, a.rms_w, xn, a.hidden, a.eps);
    gemv_w(a.use_simd, a.Wg, xn, g, a.intermediate, a.hidden);
    gemv_w(a.use_simd, a.Wu, xn, u, a.intermediate, a.hidden);
    fused_swiglu(g, u, g, a.intermediate);
    gemv_w(a.use_simd, a.Wd, g, y, a.hidden, a.intermediate);
    for (int d = 0; d < a.hidden; ++d) a.x_inout[d] += y[d];
}

void fused_delta_decode(const FusedDeltaArgs& a) {
    const int qdim = a.nk * a.dk;
    const int qkv_dim = qdim * 2 + a.nv * a.dv;
    const int zdim = a.nv * a.dv;
    Scratch& sc = tls_scratch();
    const size_t need = static_cast<size_t>(a.hidden) + qkv_dim + zdim + a.nv * 2 + qkv_dim +
                        static_cast<size_t>(a.nv) * (a.dk + a.dk + a.dv + 2) + zdim + zdim +
                        static_cast<size_t>(a.hidden);
    sc.reserve(need);
    sc.reset();
    float* xn = sc.alloc(static_cast<size_t>(a.hidden));
    float* qkv = sc.alloc(static_cast<size_t>(qkv_dim));
    float* z = sc.alloc(static_cast<size_t>(zdim));
    float* aa = sc.alloc(static_cast<size_t>(a.nv));
    float* bb = sc.alloc(static_cast<size_t>(a.nv));
    qwen3_rmsnorm(a.x, a.rms_w, xn, a.hidden, a.eps);
    gemv_w(a.use_simd, a.Wqkv, xn, qkv, qkv_dim, a.hidden);
    gemv_w(a.use_simd, a.Wz, xn, z, zdim, a.hidden);
    gemv_w(a.use_simd, a.Wa, xn, aa, a.nv, a.hidden);
    gemv_w(a.use_simd, a.Wb, xn, bb, a.nv, a.hidden);

    float* mixed = sc.alloc(static_cast<size_t>(qkv_dim));
    conv1d_update(qkv, a.conv_w, a.conv_state, mixed, qkv_dim, a.conv_k);

    const float* Q = mixed;
    const float* K = mixed + qdim;
    const float* V = mixed + 2 * qdim;
    const int rep = a.nv / a.nk;
    float* qh = sc.alloc(static_cast<size_t>(a.nv) * a.dk);
    float* kh = sc.alloc(static_cast<size_t>(a.nv) * a.dk);
    float* vh = sc.alloc(static_cast<size_t>(a.nv) * a.dv);
    float* beta = sc.alloc(static_cast<size_t>(a.nv));
    float* glog = sc.alloc(static_cast<size_t>(a.nv));
    for (int h = 0; h < a.nv; ++h) {
        const int src = h / rep;
        std::memcpy(qh + h * a.dk, Q + src * a.dk, sizeof(float) * a.dk);
        std::memcpy(kh + h * a.dk, K + src * a.dk, sizeof(float) * a.dk);
        std::memcpy(vh + h * a.dv, V + h * a.dv, sizeof(float) * a.dv);
        beta[h] = sigmoid(bb[h]);
        glog[h] = -std::exp(a.A_log[h]) * softplus(aa[h] + a.dt_bias[h]);
    }
    float* o = sc.alloc(static_cast<size_t>(zdim));
    float* og = sc.alloc(static_cast<size_t>(zdim));
    if (a.use_simd)
        delta_recurrent_step_simd(a.S, qh, kh, vh, beta, glog, o, a.nv, a.dk, a.dv, 1e-6f);
    else
        delta_recurrent_step_scalar(a.S, qh, kh, vh, beta, glog, o, a.nv, a.dk, a.dv, 1e-6f);
    gated_rmsnorm(o, z, a.gnorm, og, zdim, 1e-6f, a.gnorm_n);
    float* y = sc.alloc(static_cast<size_t>(a.hidden));
    gemv_w(a.use_simd, a.Wo, og, y, a.hidden, zdim);
    for (int d = 0; d < a.hidden; ++d) a.x_inout[d] += y[d];
}

void fused_attn_decode(const FusedAttnArgs& a) {
    const int qg_n = a.n_q * a.head_dim * 2;
    const int qn = a.n_q * a.head_dim;
    const int kn = a.n_kv * a.head_dim;
    Scratch& sc = tls_scratch();
    sc.reserve(static_cast<size_t>(a.hidden) + qg_n + qn + qn + kn + kn + qn + a.hidden);
    sc.reset();
    float* xn = sc.alloc(static_cast<size_t>(a.hidden));
    float* qg = sc.alloc(static_cast<size_t>(qg_n));
    float* q = sc.alloc(static_cast<size_t>(qn));
    float* gate = sc.alloc(static_cast<size_t>(qn));
    float* k = sc.alloc(static_cast<size_t>(kn));
    float* v = sc.alloc(static_cast<size_t>(kn));
    qwen3_rmsnorm(a.x, a.rms_w, xn, a.hidden, a.eps);
    gemv_w(a.use_simd, a.Wq, xn, qg, qg_n, a.hidden);
    gemv_w(a.use_simd, a.Wk, xn, k, kn, a.hidden);
    gemv_w(a.use_simd, a.Wv, xn, v, kn, a.hidden);
    for (int j = 0; j < qn; ++j) {
        q[j] = qg[j];
        gate[j] = qg[qn + j];
    }
    for (int h = 0; h < a.n_q; ++h)
        qwen3_rmsnorm(q + h * a.head_dim, a.q_norm, q + h * a.head_dim, a.head_dim, a.eps);
    for (int h = 0; h < a.n_kv; ++h)
        qwen3_rmsnorm(k + h * a.head_dim, a.k_norm, k + h * a.head_dim, a.head_dim, a.eps);
    rope_partial(q, k, a.n_q, a.n_kv, a.head_dim, a.rotary_dim, a.pos, a.theta);
    std::memcpy(a.k_cache + static_cast<size_t>(a.pos) * kn, k, sizeof(float) * kn);
    std::memcpy(a.v_cache + static_cast<size_t>(a.pos) * kn, v, sizeof(float) * kn);
    float* o = sc.alloc(static_cast<size_t>(qn));
    attn_decode(q, a.k_cache, a.v_cache, o, a.pos, a.n_q, a.n_kv, a.head_dim);
    for (int j = 0; j < qn; ++j) o[j] *= sigmoid(gate[j]);
    float* y = sc.alloc(static_cast<size_t>(a.hidden));
    gemv_w(a.use_simd, a.Wo, o, y, a.hidden, qn);
    for (int d = 0; d < a.hidden; ++d) a.x_inout[d] += y[d];
}

} // namespace rapidllm::ops
