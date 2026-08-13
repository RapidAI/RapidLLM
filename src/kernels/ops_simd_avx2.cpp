#include "rapidllm/kernels/cpu_caps.h"
#include "rapidllm/kernels/ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <vector>

namespace rapidllm::ops {

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

void gemv_f32_avx2(const float* W, const float* x, float* y, int m, int n) {
    const int n8 = n & ~7;
    const int n32 = n & ~31;
    for (int i = 0; i < m; ++i) {
        const float* row = W + static_cast<size_t>(i) * n;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        int j = 0;
        for (; j < n32; j += 32) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + j), _mm256_loadu_ps(x + j), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + j + 8), _mm256_loadu_ps(x + j + 8), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(row + j + 16), _mm256_loadu_ps(x + j + 16), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(row + j + 24), _mm256_loadu_ps(x + j + 24), acc3);
        }
        acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        for (; j < n8; j += 8)
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + j), _mm256_loadu_ps(x + j), acc0);
        float s = hsum256(acc0);
        for (; j < n; ++j) s += row[j] * x[j];
        y[i] = s;
    }
}

void delta_recurrent_step_avx2(float* S, const float* q_in, const float* k_in, const float* v,
                               const float* beta, const float* g_log, float* o, int n_v, int dk,
                               int dv, float eps_l2) {
    std::vector<float> q(static_cast<size_t>(n_v) * dk);
    std::vector<float> k(static_cast<size_t>(n_v) * dk);
    std::vector<float> kv(static_cast<size_t>(dv));
    std::vector<float> delta(static_cast<size_t>(dv));
    for (int h = 0; h < n_v; ++h) {
        l2norm(q_in + h * dk, q.data() + h * dk, dk, eps_l2);
        l2norm(k_in + h * dk, k.data() + h * dk, dk, eps_l2);
        const float scale = 1.f / std::sqrt(static_cast<float>(dk));
        const int dk8 = dk & ~7;
        const __m256 vs = _mm256_set1_ps(scale);
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
        int t = 0;
        for (; t + 8 <= dk * dv; t += 8)
            _mm256_storeu_ps(Sh + t, _mm256_mul_ps(_mm256_loadu_ps(Sh + t), vg));
        for (; t < dk * dv; ++t) Sh[t] *= g;
        std::fill(kv.begin(), kv.end(), 0.f);
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
                __m256 s = _mm256_fmadd_ps(ki, _mm256_loadu_ps(delta.data() + j), _mm256_loadu_ps(srow + j));
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
                __m256 acc = _mm256_fmadd_ps(_mm256_loadu_ps(srow + j), qi, _mm256_loadu_ps(oh + j));
                _mm256_storeu_ps(oh + j, acc);
            }
            for (; j < dv; ++j) oh[j] += srow[j] * qh[i];
        }
    }
}

static inline float q8_block_dot_avx2(const uint8_t* blk, const __m256 x0, const __m256 x1, const __m256 x2,
                                      const __m256 x3) {
    const __m128i h = _mm_cvtsi32_si128(*reinterpret_cast<const uint16_t*>(blk));
    const float d = _mm_cvtss_f32(_mm_cvtph_ps(h));
    const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
    __m256 acc = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                                   reinterpret_cast<const __m128i*>(qs)))),
                               x0);
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                              reinterpret_cast<const __m128i*>(qs + 8)))),
                          x1, acc);
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                              reinterpret_cast<const __m128i*>(qs + 16)))),
                          x2, acc);
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                              reinterpret_cast<const __m128i*>(qs + 24)))),
                          x3, acc);
    return d * hsum256(acc);
}

void gemv_q8_0_avx2(const uint8_t* packed, const float* x, float* y, int m, int n) {
    const int nb = n / 32;
    constexpr size_t bsz = 34;
    const size_t row_bytes = static_cast<size_t>(nb) * bsz;
    int i = 0;
    for (; i + 1 < m; i += 2) {
        const uint8_t* row0 = packed + static_cast<size_t>(i) * row_bytes;
        const uint8_t* row1 = row0 + row_bytes;
        if (i + 3 < m) _mm_prefetch(reinterpret_cast<const char*>(row1 + row_bytes), _MM_HINT_T0);
        float sum0 = 0.f, sum1 = 0.f;
        for (int b = 0; b < nb; ++b) {
            const float* xb = x + b * 32;
            const __m256 x0 = _mm256_loadu_ps(xb);
            const __m256 x1 = _mm256_loadu_ps(xb + 8);
            const __m256 x2 = _mm256_loadu_ps(xb + 16);
            const __m256 x3 = _mm256_loadu_ps(xb + 24);
            const uint8_t* blk0 = row0 + static_cast<size_t>(b) * bsz;
            const uint8_t* blk1 = row1 + static_cast<size_t>(b) * bsz;
            sum0 += q8_block_dot_avx2(blk0, x0, x1, x2, x3);
            sum1 += q8_block_dot_avx2(blk1, x0, x1, x2, x3);
        }
        y[i] = sum0;
        y[i + 1] = sum1;
    }
    for (; i < m; ++i) {
        const uint8_t* row = packed + static_cast<size_t>(i) * row_bytes;
        float sum = 0.f;
        for (int b = 0; b < nb; ++b) {
            const float* xb = x + b * 32;
            sum += q8_block_dot_avx2(row + static_cast<size_t>(b) * bsz, _mm256_loadu_ps(xb),
                                     _mm256_loadu_ps(xb + 8), _mm256_loadu_ps(xb + 16),
                                     _mm256_loadu_ps(xb + 24));
        }
        y[i] = sum;
    }
}

void gemv_f16_avx2(const uint16_t* W, const float* x, float* y, int m, int n) {
    const int n32 = n & ~31;
    const int n8 = n & ~7;
    for (int i = 0; i < m; ++i) {
        const uint16_t* row = W + static_cast<size_t>(i) * n;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        int j = 0;
        for (; j < n32; j += 32) {
            acc0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row + j))),
                                   _mm256_loadu_ps(x + j), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row + j + 8))),
                                   _mm256_loadu_ps(x + j + 8), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row + j + 16))),
                                   _mm256_loadu_ps(x + j + 16), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row + j + 24))),
                                   _mm256_loadu_ps(x + j + 24), acc3);
        }
        acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        for (; j < n8; j += 8) {
            acc0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row + j))),
                                   _mm256_loadu_ps(x + j), acc0);
        }
        float s = hsum256(acc0);
        for (; j < n; ++j) {
            const __m128i h = _mm_cvtsi32_si128(row[j]);
            s += _mm_cvtss_f32(_mm_cvtph_ps(h)) * x[j];
        }
        y[i] = s;
    }
}

void gemv_bf16_avx2(const uint16_t* W, const float* x, float* y, int m, int n) {
    const int n8 = n & ~7;
    for (int i = 0; i < m; ++i) {
        const uint16_t* row = W + static_cast<size_t>(i) * n;
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        for (; j < n8; j += 8) {
            const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + j));
            const __m256i w32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
            acc = _mm256_fmadd_ps(_mm256_castsi256_ps(w32), _mm256_loadu_ps(x + j), acc);
        }
        float s = hsum256(acc);
        for (; j < n; ++j) {
            uint32_t u = static_cast<uint32_t>(row[j]) << 16;
            float w;
            std::memcpy(&w, &u, 4);
            s += w * x[j];
        }
        y[i] = s;
    }
}

void qwen3_rmsnorm_avx2(const float* x, const float* gamma, float* y, int n, float eps) {
    const int n8 = n & ~7;
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i < n8; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    float ss = hsum256(acc);
    for (; i < n; ++i) ss += x[i] * x[i];
    const float inv = 1.f / std::sqrt(ss / static_cast<float>(n) + eps);
    const __m256 vinv = _mm256_set1_ps(inv);
    const __m256 one = _mm256_set1_ps(1.f);
    i = 0;
    if (gamma) {
        for (; i < n8; i += 8) {
            const __m256 g = _mm256_add_ps(one, _mm256_loadu_ps(gamma + i));
            _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i), vinv), g));
        }
        for (; i < n; ++i) y[i] = x[i] * inv * (1.f + gamma[i]);
    } else {
        for (; i < n8; i += 8) _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
        for (; i < n; ++i) y[i] = x[i] * inv;
    }
}

} // namespace rapidllm::ops
