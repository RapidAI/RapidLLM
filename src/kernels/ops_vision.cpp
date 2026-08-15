#include "rapidllm/kernels/ops.h"
#include "rapidllm/frontend/weight_store.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace rapidllm::ops {
namespace {

const TensorDesc& must_w(const TensorTable& t, const std::string& ir) {
    const TensorDesc* p = t.find(ir);
    if (!p) throw std::runtime_error("vision missing weight " + ir);
    return *p;
}

void dequant_row(const TensorDesc& td, std::vector<float>& out) {
    const size_t n = td.nbytes;
    if (td.quant == QuantKind::F32) {
        out.resize(n / 4);
        std::memcpy(out.data(), td.data.data(), n);
        return;
    }
    if (td.quant == QuantKind::F16) {
        out.resize(n / 2);
        const uint16_t* h = reinterpret_cast<const uint16_t*>(td.data.data());
        for (size_t i = 0; i < out.size(); ++i) out[i] = fp16_to_f32(h[i]);
        return;
    }
    if (td.quant == QuantKind::BF16) {
        out.resize(n / 2);
        const uint16_t* h = reinterpret_cast<const uint16_t*>(td.data.data());
        for (size_t i = 0; i < out.size(); ++i) {
            const uint32_t u = static_cast<uint32_t>(h[i]) << 16;
            float f;
            std::memcpy(&f, &u, 4);
            out[i] = f;
        }
        return;
    }
    throw std::runtime_error("vision unsupported quant " + td.ir_name);
}

void matmul_add(const float* W, const float* b, const float* X, float* Y, int m, int n, int T) {
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < m; ++i) {
            float acc = b ? b[i] : 0.f;
            const float* row = W + static_cast<size_t>(i) * n;
            const float* x = X + static_cast<size_t>(t) * n;
            for (int k = 0; k < n; ++k) acc += row[k] * x[k];
            Y[static_cast<size_t>(t) * m + i] = acc;
        }
    }
}

} // namespace

float gelu_tanh(float x) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    const float u = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.f + std::tanh(u));
}

void layernorm(const float* x, const float* w, const float* b, float* y, int n, float eps) {
    float mean = 0.f;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= static_cast<float>(n);
    float var = 0.f;
    for (int i = 0; i < n; ++i) {
        const float d = x[i] - mean;
        var += d * d;
    }
    const float inv = 1.f / std::sqrt(var / static_cast<float>(n) + eps);
    for (int i = 0; i < n; ++i) {
        const float g = w ? w[i] : 1.f;
        const float bias = b ? b[i] : 0.f;
        y[i] = (x[i] - mean) * inv * g + bias;
    }
}

int vision_grid(int img_h, int img_w, int patch, int merge, int* gh, int* gw) {
    const int ph = patch * merge;
    if (img_h <= 0 || img_w <= 0 || img_h % ph != 0 || img_w % ph != 0) return -1;
    *gh = img_h / ph;
    *gw = img_w / ph;
    return (*gh) * (*gw);
}

int vision_smart_resize(int h, int w, int factor, int min_pixels, int max_pixels, int* oh, int* ow) {
    if (h <= 0 || w <= 0 || factor <= 0) return -1;
    auto rnd = [&](int x) {
        const int q = (x + factor / 2) / factor * factor;
        return std::max(factor, q);
    };
    int hb = rnd(h);
    int wb = rnd(w);
    const double area = static_cast<double>(hb) * wb;
    if (max_pixels > 0 && area > static_cast<double>(max_pixels)) {
        const double beta = std::sqrt(area / static_cast<double>(max_pixels));
        hb = std::max(factor, static_cast<int>(std::floor(h / beta / factor)) * factor);
        wb = std::max(factor, static_cast<int>(std::floor(w / beta / factor)) * factor);
    } else if (min_pixels > 0 && area < static_cast<double>(min_pixels)) {
        const double beta = std::sqrt(static_cast<double>(min_pixels) / area);
        hb = static_cast<int>(std::ceil(h * beta / factor)) * factor;
        wb = static_cast<int>(std::ceil(w * beta / factor)) * factor;
        if (hb < factor) hb = factor;
        if (wb < factor) wb = factor;
    }
    while (min_pixels > 0 && static_cast<long long>(hb) * wb < min_pixels) {
        if (hb <= wb) hb += factor;
        else wb += factor;
    }
    *oh = hb;
    *ow = wb;
    return 0;
}

void vision_resize_bilinear(const float* src, int sh, int sw, float* dst, int dh, int dw) {
    if (sh == dh && sw == dw) {
        std::memcpy(dst, src, sizeof(float) * static_cast<size_t>(dh) * dw * 3);
        return;
    }
    const float yscale = sh > 1 && dh > 1 ? static_cast<float>(sh - 1) / static_cast<float>(dh - 1) : 0.f;
    const float xscale = sw > 1 && dw > 1 ? static_cast<float>(sw - 1) / static_cast<float>(dw - 1) : 0.f;
    for (int y = 0; y < dh; ++y) {
        const float fy = y * yscale;
        int y0 = static_cast<int>(fy);
        int y1 = std::min(y0 + 1, sh - 1);
        const float wy = fy - static_cast<float>(y0);
        for (int x = 0; x < dw; ++x) {
            const float fx = x * xscale;
            int x0 = static_cast<int>(fx);
            int x1 = std::min(x0 + 1, sw - 1);
            const float wx = fx - static_cast<float>(x0);
            float* o = dst + (static_cast<size_t>(y) * dw + x) * 3;
            for (int c = 0; c < 3; ++c) {
                const float v00 = src[(static_cast<size_t>(y0) * sw + x0) * 3 + c];
                const float v01 = src[(static_cast<size_t>(y0) * sw + x1) * 3 + c];
                const float v10 = src[(static_cast<size_t>(y1) * sw + x0) * 3 + c];
                const float v11 = src[(static_cast<size_t>(y1) * sw + x1) * 3 + c];
                const float a = v00 * (1.f - wx) + v01 * wx;
                const float b = v10 * (1.f - wx) + v11 * wx;
                o[c] = a * (1.f - wy) + b * wy;
            }
        }
    }
}

void vision_encode(const float* rgb, int img_h, int img_w, const VisionDesc& V,
                   const TensorTable& weights, float* out) {
    int gh = 0, gw = 0;
    const int n_out = vision_grid(img_h, img_w, V.patch, V.spatial_merge, &gh, &gw);
    if (n_out <= 0) throw std::runtime_error("vision image size not divisible by patch*merge");
    const int gt = img_h / V.patch;
    const int gs = img_w / V.patch;
    const int n_tok = gt * gs;
    const int C = V.in_channels;
    const int P = V.patch;
    const int Tp = V.temporal_patch;
    const int D = V.hidden;

    // Official Qwen2VL processor: image_mean = image_std = 0.5
    std::vector<float> pix(static_cast<size_t>(img_h) * img_w * 3);
    for (size_t i = 0; i < pix.size(); ++i) pix[i] = (rgb[i] - 0.5f) / 0.5f;
    rgb = pix.data();

    std::vector<float> proj, proj_b, pos, x(static_cast<size_t>(n_tok) * D), xn(x.size());
    dequant_row(must_w(weights, "visual.patch_embed"), proj);
    dequant_row(must_w(weights, "visual.patch_embed_bias"), proj_b);
    dequant_row(must_w(weights, "visual.pos_embed"), pos);
    // conv3d: [D, C, Tp, P, P] on two identical frames
    const int kvol = C * Tp * P * P;
    if (static_cast<int>(proj.size()) < D * kvol)
        throw std::runtime_error("visual.patch_embed unexpected size");
    for (int yh = 0; yh < gt; ++yh) {
        for (int xw = 0; xw < gs; ++xw) {
            float* dst = x.data() + (static_cast<size_t>(yh * gs + xw) * D);
            for (int oc = 0; oc < D; ++oc) {
                float acc = proj_b[static_cast<size_t>(oc)];
                const float* wk = proj.data() + static_cast<size_t>(oc) * kvol;
                int ki = 0;
                for (int c = 0; c < C; ++c) {
                    for (int t = 0; t < Tp; ++t) {
                        for (int py = 0; py < P; ++py) {
                            for (int px = 0; px < P; ++px, ++ki) {
                                const int iy = yh * P + py;
                                const int ix = xw * P + px;
                                acc += wk[ki] * rgb[(static_cast<size_t>(iy) * img_w + ix) * 3 + c];
                            }
                        }
                    }
                }
                dst[oc] = acc;
            }
        }
    }
    // bilinear-ish: nearest sample of learned 48x48 pos grid (sqrt(n_pos) if square)
    int pg = 1;
    while (pg * pg < V.n_pos) ++pg;
    if (pg * pg != V.n_pos) pg = 48;
    for (int yh = 0; yh < gt; ++yh) {
        for (int xw = 0; xw < gs; ++xw) {
            const int sy = (gt == 1) ? 0 : yh * (pg - 1) / (gt - 1);
            const int sx = (gs == 1) ? 0 : xw * (pg - 1) / (gs - 1);
            const int pi = std::min(sy, pg - 1) * pg + std::min(sx, pg - 1);
            float* dst = x.data() + (static_cast<size_t>(yh * gs + xw) * D);
            const float* pe = pos.data() + static_cast<size_t>(pi) * D;
            for (int d = 0; d < D; ++d) dst[d] += pe[d];
        }
    }

    std::vector<float> qkv, qkv_b, pr, pr_b, fc1, fc1_b, fc2, fc2_b, n1, n1b, n2, n2b;
    std::vector<float> tmp(static_cast<size_t>(n_tok) * std::max(D, V.intermediate));
    const int hd = D / V.n_heads;
    for (int li = 0; li < V.depth; ++li) {
        const std::string p = "visual.blocks[" + std::to_string(li) + "].";
        dequant_row(must_w(weights, p + "norm1"), n1);
        dequant_row(must_w(weights, p + "norm1_bias"), n1b);
        dequant_row(must_w(weights, p + "attn.qkv"), qkv);
        dequant_row(must_w(weights, p + "attn.qkv_bias"), qkv_b);
        dequant_row(must_w(weights, p + "attn.proj"), pr);
        dequant_row(must_w(weights, p + "attn.proj_bias"), pr_b);
        dequant_row(must_w(weights, p + "norm2"), n2);
        dequant_row(must_w(weights, p + "norm2_bias"), n2b);
        dequant_row(must_w(weights, p + "mlp.fc1"), fc1);
        dequant_row(must_w(weights, p + "mlp.fc1_bias"), fc1_b);
        dequant_row(must_w(weights, p + "mlp.fc2"), fc2);
        dequant_row(must_w(weights, p + "mlp.fc2_bias"), fc2_b);

        for (int t = 0; t < n_tok; ++t)
            layernorm(x.data() + t * D, n1.data(), n1b.data(), xn.data() + t * D, D, 1e-6f);
        std::vector<float> qkv_o(static_cast<size_t>(n_tok) * 3 * D);
        matmul_add(qkv.data(), qkv_b.data(), xn.data(), qkv_o.data(), 3 * D, D, n_tok);
        std::vector<float> attn_o(static_cast<size_t>(n_tok) * D, 0.f);
        const float scale = 1.f / std::sqrt(static_cast<float>(hd));
        for (int h = 0; h < V.n_heads; ++h) {
            for (int qi = 0; qi < n_tok; ++qi) {
                float mx = -1e30f;
                std::vector<float> sc(static_cast<size_t>(n_tok));
                const float* q = qkv_o.data() + static_cast<size_t>(qi) * 3 * D + h * hd;
                for (int kj = 0; kj < n_tok; ++kj) {
                    const float* k = qkv_o.data() + static_cast<size_t>(kj) * 3 * D + D + h * hd;
                    float dot = 0.f;
                    for (int d = 0; d < hd; ++d) dot += q[d] * k[d];
                    sc[static_cast<size_t>(kj)] = dot * scale;
                    mx = std::max(mx, sc[static_cast<size_t>(kj)]);
                }
                float sum = 0.f;
                for (int kj = 0; kj < n_tok; ++kj) {
                    sc[static_cast<size_t>(kj)] = std::exp(sc[static_cast<size_t>(kj)] - mx);
                    sum += sc[static_cast<size_t>(kj)];
                }
                float* oh = attn_o.data() + static_cast<size_t>(qi) * D + h * hd;
                for (int kj = 0; kj < n_tok; ++kj) {
                    const float a = sc[static_cast<size_t>(kj)] / sum;
                    const float* v = qkv_o.data() + static_cast<size_t>(kj) * 3 * D + 2 * D + h * hd;
                    for (int d = 0; d < hd; ++d) oh[d] += a * v[d];
                }
            }
        }
        std::vector<float> proj_o(static_cast<size_t>(n_tok) * D);
        matmul_add(pr.data(), pr_b.data(), attn_o.data(), proj_o.data(), D, D, n_tok);
        for (size_t i = 0; i < x.size(); ++i) x[i] += proj_o[i];
        for (int t = 0; t < n_tok; ++t)
            layernorm(x.data() + t * D, n2.data(), n2b.data(), xn.data() + t * D, D, 1e-6f);
        std::vector<float> h1(static_cast<size_t>(n_tok) * V.intermediate);
        matmul_add(fc1.data(), fc1_b.data(), xn.data(), h1.data(), V.intermediate, D, n_tok);
        for (float& v : h1) v = gelu_tanh(v);
        std::vector<float> h2(static_cast<size_t>(n_tok) * D);
        matmul_add(fc2.data(), fc2_b.data(), h1.data(), h2.data(), D, V.intermediate, n_tok);
        for (size_t i = 0; i < x.size(); ++i) x[i] += h2[i];
    }

    // spatial merge 2x2 → concat last dim, LN, fc1, gelu, fc2
    const int md = D * V.spatial_merge * V.spatial_merge;
    std::vector<float> merged(static_cast<size_t>(n_out) * md);
    for (int oy = 0; oy < gh; ++oy) {
        for (int ox = 0; ox < gw; ++ox) {
            float* dst = merged.data() + (static_cast<size_t>(oy * gw + ox) * md);
            int off = 0;
            for (int my = 0; my < V.spatial_merge; ++my) {
                for (int mx = 0; mx < V.spatial_merge; ++mx) {
                    const int iy = oy * V.spatial_merge + my;
                    const int ix = ox * V.spatial_merge + mx;
                    std::memcpy(dst + off, x.data() + static_cast<size_t>(iy * gs + ix) * D,
                                sizeof(float) * D);
                    off += D;
                }
            }
        }
    }
    std::vector<float> mn, mnb, m1, m1b, m2, m2b, ln(merged.size());
    dequant_row(must_w(weights, "visual.merger.norm"), mn);
    dequant_row(must_w(weights, "visual.merger.norm_bias"), mnb);
    dequant_row(must_w(weights, "visual.merger.fc1"), m1);
    dequant_row(must_w(weights, "visual.merger.fc1_bias"), m1b);
    dequant_row(must_w(weights, "visual.merger.fc2"), m2);
    dequant_row(must_w(weights, "visual.merger.fc2_bias"), m2b);
    for (int t = 0; t < n_out; ++t)
        layernorm(merged.data() + t * md, mn.data(), mnb.data(), ln.data() + t * md, md, 1e-6f);
    const int hid = static_cast<int>(m1b.size());
    std::vector<float> mid(static_cast<size_t>(n_out) * hid);
    matmul_add(m1.data(), m1b.data(), ln.data(), mid.data(), hid, md, n_out);
    for (float& v : mid) v = gelu_tanh(v);
    matmul_add(m2.data(), m2b.data(), mid.data(), out, V.out_hidden, hid, n_out);
}

} // namespace rapidllm::ops
