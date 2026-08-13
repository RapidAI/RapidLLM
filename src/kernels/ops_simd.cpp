#include "rapidllm/kernels/ops.h"

#include <cmath>
#include <cstring>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#define RAPIDLLM_AVX2 1
#else
#define RAPIDLLM_AVX2 0
#endif

namespace rapidllm::ops {

const char* simd_isa_name() {
#if RAPIDLLM_AVX2
    return "avx2+fma";
#else
    return "scalar-fallback";
#endif
}

#if RAPIDLLM_AVX2
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
#endif

void gemv_f32_simd(const float* W, const float* x, float* y, int m, int n) {
#if RAPIDLLM_AVX2
    const int n8 = n & ~7;
    for (int i = 0; i < m; ++i) {
        const float* row = W + static_cast<size_t>(i) * n;
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        for (; j < n8; j += 8) {
            __m256 w = _mm256_loadu_ps(row + j);
            __m256 xv = _mm256_loadu_ps(x + j);
            acc = _mm256_fmadd_ps(w, xv, acc);
        }
        float s = hsum256(acc);
        for (; j < n; ++j) s += row[j] * x[j];
        y[i] = s;
    }
#else
    gemv_f32_scalar(W, x, y, m, n);
#endif
}

void delta_recurrent_step_simd(float* S, const float* q_in, const float* k_in, const float* v,
                               const float* beta, const float* g_log, float* o,
                               int n_v, int dk, int dv, float eps_l2) {
#if RAPIDLLM_AVX2
    std::vector<float> q(static_cast<size_t>(n_v) * dk);
    std::vector<float> k(static_cast<size_t>(n_v) * dk);
    for (int h = 0; h < n_v; ++h) {
        l2norm(q_in + h * dk, q.data() + h * dk, dk, eps_l2);
        l2norm(k_in + h * dk, k.data() + h * dk, dk, eps_l2);
        const float scale = 1.f / std::sqrt(static_cast<float>(dk));
        const int dk8 = dk & ~7;
        __m256 vs = _mm256_set1_ps(scale);
        float* qh = q.data() + h * dk;
        int i = 0;
        for (; i < dk8; i += 8) _mm256_storeu_ps(qh + i, _mm256_mul_ps(_mm256_loadu_ps(qh + i), vs));
        for (; i < dk; ++i) qh[i] *= scale;
    }
    const int dv8 = dv & ~7;
    for (int h = 0; h < n_v; ++h) {
        const float g = std::exp(g_log[h]);
        float* Sh = S + static_cast<size_t>(h) * dk * dv;
        const __m256 vg = _mm256_set1_ps(g);
        for (int i = 0; i < dk * dv; i += 8) {
            if (i + 8 <= dk * dv) {
                _mm256_storeu_ps(Sh + i, _mm256_mul_ps(_mm256_loadu_ps(Sh + i), vg));
            } else {
                for (int t = i; t < dk * dv; ++t) Sh[t] *= g;
            }
        }
        std::vector<float> kv(static_cast<size_t>(dv), 0.f);
        const float* kh = k.data() + h * dk;
        const float* qh = q.data() + h * dk;
        const float* vh = v + h * dv;
        for (int i = 0; i < dk; ++i) {
            const __m256 ki = _mm256_set1_ps(kh[i]);
            const float* srow = Sh + i * dv;
            int j = 0;
            for (; j < dv8; j += 8) {
                __m256 acc = _mm256_loadu_ps(kv.data() + j);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(srow + j), ki, acc);
                _mm256_storeu_ps(kv.data() + j, acc);
            }
            for (; j < dv; ++j) kv[j] += srow[j] * kh[i];
        }
        const __m256 vb = _mm256_set1_ps(beta[h]);
        std::vector<float> delta(static_cast<size_t>(dv));
        int j = 0;
        for (; j < dv8; j += 8) {
            __m256 d = _mm256_sub_ps(_mm256_loadu_ps(vh + j), _mm256_loadu_ps(kv.data() + j));
            _mm256_storeu_ps(delta.data() + j, _mm256_mul_ps(d, vb));
        }
        for (; j < dv; ++j) delta[j] = (vh[j] - kv[j]) * beta[h];
        for (int i = 0; i < dk; ++i) {
            float* srow = Sh + i * dv;
            const __m256 ki = _mm256_set1_ps(kh[i]);
            j = 0;
            for (; j < dv8; j += 8) {
                __m256 s = _mm256_loadu_ps(srow + j);
                s = _mm256_fmadd_ps(ki, _mm256_loadu_ps(delta.data() + j), s);
                _mm256_storeu_ps(srow + j, s);
            }
            for (; j < dv; ++j) srow[j] += kh[i] * delta[j];
        }
        float* oh = o + h * dv;
        std::memset(oh, 0, sizeof(float) * dv);
        for (int i = 0; i < dk; ++i) {
            const __m256 qi = _mm256_set1_ps(qh[i]);
            const float* srow = Sh + i * dv;
            j = 0;
            for (; j < dv8; j += 8) {
                __m256 acc = _mm256_loadu_ps(oh + j);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(srow + j), qi, acc);
                _mm256_storeu_ps(oh + j, acc);
            }
            for (; j < dv; ++j) oh[j] += srow[j] * qh[i];
        }
    }
#else
    delta_recurrent_step_scalar(S, q_in, k_in, v, beta, g_log, o, n_v, dk, dv, eps_l2);
#endif
}

} // namespace rapidllm::ops
