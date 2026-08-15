#include "rapidllm/runtime/session.h"
#include "rapidllm/runtime/ngram_draft.h"
#include "rapidllm/runtime/image_io.h"

#include "rapidllm/kernels/ops.h"
#include "rapidllm/runtime/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace rapidllm {

static float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
static float softplus(float x) { return std::log1p(std::exp(-std::fabs(x))) + std::max(x, 0.f); }

Session::Session(Device& dev, WeightStore& store, int ctx, bool, bool fuse, bool use_cuda)
    : dev_(&dev), store_(&store), ctx_(ctx), fuse_(fuse) {
    const ModelDesc& m = store.model();
    if (use_cuda && cuda_gen::available()) gpu_ = cuda_gen::Engine::create(store, ctx);
    if (!gpu_) cache_ = std::make_unique<DualCache>(dev, m, ctx, DType::F32);
    logits_.assign(static_cast<size_t>(m.vocab), 0.f);
    last_hidden_.assign(static_cast<size_t>(m.hidden), 0.f);
    pos_ = 0;
}

const TensorDesc& Session::must(std::string_view ir) const {
    const TensorDesc* t = store_->table().find(ir);
    if (!t) throw std::runtime_error(std::string("missing weight ") + std::string(ir));
    return *t;
}

const TensorDesc* Session::find_w(std::string_view ir) const { return store_->table().find(ir); }

bool Session::has_mtp() const {
    return store_->model().has_mtp || (find_w("mtp.fc") && find_w("mtp.norm"));
}

SpecKind Session::resolve_spec(SpecKind s) const {
    if (s == SpecKind::Auto) {
        if (draft_) return SpecKind::Draft;
        return has_mtp() ? SpecKind::Mtp : SpecKind::Ngram;
    }
    return s;
}

void Session::set_draft(Session* draft) {
    if (draft == this) throw std::runtime_error("draft session cannot be the target");
    if (draft && draft->store_->model().vocab != store_->model().vocab)
        throw std::runtime_error("draft vocab != target vocab");
    draft_ = draft;
}

int Session::draft_tokens(int take, int32_t* out) {
    if (!draft_ || take <= 0 || !out || ctx_tokens_.empty()) return 0;
    GenerateConfig dc;
    dc.max_new_tokens = take;
    dc.greedy = true;
    dc.enable_thinking = false;
    dc.spec = SpecKind::Off;
    dc.spec_n = 0;
    dc.fuse = draft_->fuse_;
    return draft_->generate(ctx_tokens_.data(), static_cast<int>(ctx_tokens_.size()), out, take, dc);
}

static const float* f32_ptr(const TensorDesc& t) {
    if (t.quant != QuantKind::F32)
        throw std::runtime_error("expected F32 leftover/norm: " + t.ir_name);
    return reinterpret_cast<const float*>(t.data.data());
}

static ops::QuantW qw(const TensorDesc& t) {
    ops::QuantW w;
    w.data = t.data.data();
    w.scale = t.scale.empty() ? nullptr : t.scale.data();
    w.quant = t.quant;
    return w;
}

static void embed_row(const TensorDesc& emb, int id, float* out, int H) {
    if (emb.quant == QuantKind::F32) {
        std::memcpy(out, f32_ptr(emb) + static_cast<size_t>(id) * H, sizeof(float) * H);
        return;
    }
    if (emb.quant == QuantKind::F16) {
        const uint16_t* h = reinterpret_cast<const uint16_t*>(emb.data.data()) + static_cast<size_t>(id) * H;
        for (int i = 0; i < H; ++i) out[i] = ops::fp16_to_f32(h[i]);
        return;
    }
    if (emb.quant == QuantKind::BF16) {
        const uint16_t* h = reinterpret_cast<const uint16_t*>(emb.data.data()) + static_cast<size_t>(id) * H;
        for (int i = 0; i < H; ++i) {
            uint32_t u = static_cast<uint32_t>(h[i]) << 16;
            std::memcpy(&out[i], &u, 4);
        }
        return;
    }
    throw std::runtime_error("unsupported embed quant");
}

void Session::layer_mlp(int i, float* x, int seq) {
    const ModelDesc& m = store_->model();
    const std::string p = "layers[" + std::to_string(i) + "].";
    const TensorDesc& nrm = must(p + "ffn_norm");
    const TensorDesc& wg = must(p + "mlp.gate");
    const TensorDesc& wu = must(p + "mlp.up");
    const TensorDesc& wd = must(p + "mlp.down");
    const int H = m.hidden;
    const int I = wd.shape[1] > 0 ? static_cast<int>(wd.shape[1]) : m.intermediate;
    std::vector<float> xn(static_cast<size_t>(H));
    std::vector<float> g(static_cast<size_t>(I));
    std::vector<float> u(static_cast<size_t>(I));
    std::vector<float> h(static_cast<size_t>(I));
    std::vector<float> y(static_cast<size_t>(H));
    for (int t = 0; t < seq; ++t) {
        float* xt = x + static_cast<size_t>(t) * H;
        ops::qwen3_rmsnorm(xt, f32_ptr(nrm), xn.data(), H, m.rms_eps);
        ops::linear(qw(wg), xn.data(), g.data(), I, H);
        ops::linear(qw(wu), xn.data(), u.data(), I, H);
        ops::swiglu(g.data(), u.data(), h.data(), I);
        ops::linear(qw(wd), h.data(), y.data(), H, I);
        for (int d = 0; d < H; ++d) xt[d] += y[d];
    }
}

void Session::layer_delta(int i, float* x, int seq, bool prefill, int token_pos) {
    const ModelDesc& m = store_->model();
    const LayerDesc& L = m.layers[static_cast<size_t>(i)];
    const DeltaNetDesc& D = L.delta;
    const std::string p = "layers[" + std::to_string(i) + "].";
    const TensorDesc& nrm = must(p + "attn_norm");
    const TensorDesc& wqkv = must(p + "delta.gemm.in_proj_qkv");
    const TensorDesc& wz = must(p + "delta.gemm.in_proj_z");
    const TensorDesc& wo = must(p + "delta.gemm.out_proj");
    const TensorDesc& wa = must(p + "delta.leftover.in_proj_a");
    const TensorDesc& wb = must(p + "delta.leftover.in_proj_b");
    const TensorDesc& alog = must(p + "delta.leftover.a_log");
    const TensorDesc& dtb = must(p + "delta.leftover.dt_bias");
    const TensorDesc& convw = must(p + "delta.leftover.conv1d");
    const TensorDesc& gn = must(p + "delta.leftover.norm");

    const int H = m.hidden;
    const int nk = D.n_k_heads, nv = D.n_v_heads, dk = D.k_dim, dv = D.v_dim;
    const int qdim = nk * dk, kdim = nk * dk, vdim = nv * dv;
    const int qkv_dim = qdim + kdim + vdim;
    const int zdim = nv * dv;
    const int slot = m.mixer_slot(i);

    std::vector<float> xn(static_cast<size_t>(seq) * H);
    std::vector<float> qkv(static_cast<size_t>(seq) * qkv_dim);
    std::vector<float> qkvc(static_cast<size_t>(seq) * qkv_dim);
    std::vector<float> z(static_cast<size_t>(seq) * zdim);
    std::vector<float> a(static_cast<size_t>(seq) * nv);
    std::vector<float> b(static_cast<size_t>(seq) * nv);

    for (int t = 0; t < seq; ++t) {
        ops::qwen3_rmsnorm(x + static_cast<size_t>(t) * H, f32_ptr(nrm), xn.data() + static_cast<size_t>(t) * H, H,
                           L.rms_eps);
        const float* xt = xn.data() + static_cast<size_t>(t) * H;
        ops::linear(qw(wqkv), xt, qkv.data() + static_cast<size_t>(t) * qkv_dim, qkv_dim, H);
        ops::linear(qw(wz), xt, z.data() + static_cast<size_t>(t) * zdim, zdim, H);
        ops::linear(qw(wa), xt, a.data() + static_cast<size_t>(t) * nv, nv, H);
        ops::linear(qw(wb), xt, b.data() + static_cast<size_t>(t) * nv, nv, H);
    }

    if (prefill) {
        ops::conv1d_causal_prefill(qkv.data(), f32_ptr(convw), qkvc.data(), seq, qkv_dim, D.conv_k);
        // seed conv state with last k samples
        float* st = cache_->conv_ptr(slot);
        std::fill(st, st + qkv_dim * D.conv_k, 0.f);
        for (int t = 0; t < seq; ++t) {
            for (int c = 0; c < qkv_dim; ++c) {
                float* row = st + c * D.conv_k;
                for (int p = 0; p < D.conv_k - 1; ++p) row[p] = row[p + 1];
                row[D.conv_k - 1] = qkv[static_cast<size_t>(t) * qkv_dim + c];
            }
        }
    } else {
        ops::conv1d_update(qkv.data(), f32_ptr(convw), cache_->conv_ptr(slot), qkvc.data(), qkv_dim, D.conv_k);
    }

    float* S = cache_->state_ptr(slot);
    const int rep = nv / nk;
    std::vector<float> qh(static_cast<size_t>(nv) * dk);
    std::vector<float> kh(static_cast<size_t>(nv) * dk);
    std::vector<float> vh(static_cast<size_t>(nv) * dv);
    std::vector<float> beta(static_cast<size_t>(nv));
    std::vector<float> glog(static_cast<size_t>(nv));
    std::vector<float> o(static_cast<size_t>(nv) * dv);
    std::vector<float> og(static_cast<size_t>(nv) * dv);
    std::vector<float> y(static_cast<size_t>(H));

    for (int t = 0; t < seq; ++t) {
        const float* mix = qkvc.data() + static_cast<size_t>(t) * qkv_dim;
        const float* Q = mix;
        const float* K = mix + qdim;
        const float* V = mix + qdim + kdim;
        for (int h = 0; h < nv; ++h) {
            const int src = h / rep;
            std::memcpy(qh.data() + h * dk, Q + src * dk, sizeof(float) * dk);
            std::memcpy(kh.data() + h * dk, K + src * dk, sizeof(float) * dk);
            std::memcpy(vh.data() + h * dv, V + h * dv, sizeof(float) * dv);
            beta[h] = sigmoid(b[static_cast<size_t>(t) * nv + h]);
            const float A = f32_ptr(alog)[h];
            const float dt = f32_ptr(dtb)[h];
            glog[h] = -std::exp(A) * softplus(a[static_cast<size_t>(t) * nv + h] + dt);
        }
        ops::delta_recurrent_step(S, qh.data(), kh.data(), vh.data(), beta.data(), glog.data(), o.data(), nv, dk, dv,
                                  1e-6f);
        ops::gated_rmsnorm(o.data(), z.data() + static_cast<size_t>(t) * zdim, f32_ptr(gn), og.data(), zdim, 1e-6f,
                           gn.shape[0] > 0 ? static_cast<int>(gn.shape[0]) : zdim);
        ops::linear(qw(wo), og.data(), y.data(), H, zdim);
        float* xt = x + static_cast<size_t>(t) * H;
        for (int d = 0; d < H; ++d) xt[d] += y[d];
        (void)token_pos;
    }
}

void Session::layer_attn(int i, float* x, int seq, bool prefill, int token_pos) {
    const ModelDesc& m = store_->model();
    const LayerDesc& L = m.layers[static_cast<size_t>(i)];
    const GatedAttnDesc& A = L.attn;
    const std::string p = "layers[" + std::to_string(i) + "].";
    const TensorDesc& nrm = must(p + "attn_norm");
    const TensorDesc& wq = must(p + "attn.wq");
    const TensorDesc& wk = must(p + "attn.wk");
    const TensorDesc& wv = must(p + "attn.wv");
    const TensorDesc& wo = must(p + "attn.wo");
    const TensorDesc& qn = must(p + "attn.q_norm");
    const TensorDesc& kn = must(p + "attn.k_norm");
    const int H = m.hidden;
    const int nq = A.n_q, nkv = A.n_kv, hd = A.head_dim;
    const int rotary = std::max(2, static_cast<int>(hd * m.rope.partial_factor));
    const int slot = m.mixer_slot(i);

    std::vector<float> xn(static_cast<size_t>(seq) * H);
    std::vector<float> qg(static_cast<size_t>(seq) * nq * hd * 2);
    std::vector<float> q(static_cast<size_t>(seq) * nq * hd);
    std::vector<float> gate(static_cast<size_t>(seq) * nq * hd);
    std::vector<float> k(static_cast<size_t>(seq) * nkv * hd);
    std::vector<float> v(static_cast<size_t>(seq) * nkv * hd);
    std::vector<float> o(static_cast<size_t>(seq) * nq * hd);
    std::vector<float> y(static_cast<size_t>(H));

    for (int t = 0; t < seq; ++t) {
        ops::qwen3_rmsnorm(x + static_cast<size_t>(t) * H, f32_ptr(nrm), xn.data() + t * H, H, L.rms_eps);
        const float* xt = xn.data() + static_cast<size_t>(t) * H;
        ops::linear(qw(wq), xt, qg.data() + t * nq * hd * 2, nq * hd * 2, H);
        ops::linear(qw(wk), xt, k.data() + t * nkv * hd, nkv * hd, H);
        ops::linear(qw(wv), xt, v.data() + t * nkv * hd, nkv * hd, H);
        const float* qgt = qg.data() + static_cast<size_t>(t) * nq * hd * 2;
        for (int j = 0; j < nq * hd; ++j) {
            q[t * nq * hd + j] = qgt[j];
            gate[t * nq * hd + j] = qgt[nq * hd + j];
        }
        for (int h = 0; h < nq; ++h)
            ops::qwen3_rmsnorm(q.data() + (t * nq + h) * hd, f32_ptr(qn), q.data() + (t * nq + h) * hd, hd, L.rms_eps);
        for (int h = 0; h < nkv; ++h)
            ops::qwen3_rmsnorm(k.data() + (t * nkv + h) * hd, f32_ptr(kn), k.data() + (t * nkv + h) * hd, hd, L.rms_eps);
        ops::rope_partial(q.data() + t * nq * hd, k.data() + t * nkv * hd, nq, nkv, hd, rotary,
                          prefill ? t : token_pos, m.rope.theta);
        cache_->append_kv(slot, k.data() + t * nkv * hd, v.data() + t * nkv * hd, prefill ? t : token_pos);
    }

    if (prefill) {
        ops::attn_prefill(q.data(), cache_->k_ptr(slot), cache_->v_ptr(slot), o.data(), seq, nq, nkv, hd);
    } else {
        ops::attn_decode(q.data(), cache_->k_ptr(slot), cache_->v_ptr(slot), o.data(), token_pos, nq, nkv, hd);
    }
    for (int t = 0; t < seq; ++t) {
        for (int j = 0; j < nq * hd; ++j) o[t * nq * hd + j] *= sigmoid(gate[t * nq * hd + j]);
        ops::linear(qw(wo), o.data() + t * nq * hd, y.data(), H, nq * hd);
        float* xt = x + static_cast<size_t>(t) * H;
        for (int d = 0; d < H; ++d) xt[d] += y[d];
    }
}

void Session::forward_hidden(const float* x_in, float* x_out, bool is_prefill, int seq, int token_pos) {
    const ModelDesc& m = store_->model();
    const int H = m.hidden;
    std::memcpy(x_out, x_in, static_cast<size_t>(seq) * H * sizeof(float));
    const bool use_fuse = fuse_ && seq == 1;
    for (int i = 0; i < m.n_layers; ++i) {
        if (use_fuse) {
            const LayerDesc& L = m.layers[static_cast<size_t>(i)];
            const std::string p = "layers[" + std::to_string(i) + "].";
            if (L.kind == LayerKind::GatedDeltaNet) {
                const int slot = m.mixer_slot(i);
                ops::FusedDeltaArgs a{};
                a.x = x_out;
                a.x_inout = x_out;
                a.rms_w = f32_ptr(must(p + "attn_norm"));
                a.Wqkv = qw(must(p + "delta.gemm.in_proj_qkv"));
                a.Wz = qw(must(p + "delta.gemm.in_proj_z"));
                a.Wa = qw(must(p + "delta.leftover.in_proj_a"));
                a.Wb = qw(must(p + "delta.leftover.in_proj_b"));
                a.Wo = qw(must(p + "delta.gemm.out_proj"));
                a.A_log = f32_ptr(must(p + "delta.leftover.a_log"));
                a.dt_bias = f32_ptr(must(p + "delta.leftover.dt_bias"));
                a.conv_w = f32_ptr(must(p + "delta.leftover.conv1d"));
                a.gnorm = f32_ptr(must(p + "delta.leftover.norm"));
                a.S = cache_->state_ptr(slot);
                a.conv_state = cache_->conv_ptr(slot);
                a.hidden = H;
                a.nk = L.delta.n_k_heads;
                a.nv = L.delta.n_v_heads;
                a.dk = L.delta.k_dim;
                a.dv = L.delta.v_dim;
                a.conv_k = L.delta.conv_k;
                a.eps = L.rms_eps;
                a.use_simd = true;
                a.gnorm_n = must(p + "delta.leftover.norm").shape[0] > 0
                                ? static_cast<int>(must(p + "delta.leftover.norm").shape[0])
                                : 0;
                ops::fused_delta_decode(a);
            } else {
                const int slot = m.mixer_slot(i);
                const int rotary = std::max(2, static_cast<int>(L.attn.head_dim * m.rope.partial_factor));
                ops::FusedAttnArgs a{};
                a.x = x_out;
                a.x_inout = x_out;
                a.rms_w = f32_ptr(must(p + "attn_norm"));
                a.Wq = qw(must(p + "attn.wq"));
                a.Wk = qw(must(p + "attn.wk"));
                a.Wv = qw(must(p + "attn.wv"));
                a.Wo = qw(must(p + "attn.wo"));
                a.q_norm = f32_ptr(must(p + "attn.q_norm"));
                a.k_norm = f32_ptr(must(p + "attn.k_norm"));
                a.k_cache = cache_->k_ptr(slot);
                a.v_cache = cache_->v_ptr(slot);
                a.hidden = H;
                a.n_q = L.attn.n_q;
                a.n_kv = L.attn.n_kv;
                a.head_dim = L.attn.head_dim;
                a.rotary_dim = rotary;
                a.pos = token_pos;
                a.theta = m.rope.theta;
                a.eps = L.rms_eps;
                a.use_simd = true;
                ops::fused_attn_decode(a);
            }
            ops::FusedMlpArgs mlp{};
            mlp.x = x_out;
            mlp.x_inout = x_out;
            mlp.rms_w = f32_ptr(must(p + "ffn_norm"));
            mlp.Wg = qw(must(p + "mlp.gate"));
            mlp.Wu = qw(must(p + "mlp.up"));
            mlp.Wd = qw(must(p + "mlp.down"));
            mlp.hidden = H;
            const TensorDesc& wd = must(p + "mlp.down");
            mlp.intermediate = wd.shape[1] > 0 ? static_cast<int>(wd.shape[1]) : m.intermediate;
            mlp.eps = m.rms_eps;
            mlp.use_simd = true;
            ops::fused_mlp_decode(mlp);
        } else {
            if (m.layers[static_cast<size_t>(i)].kind == LayerKind::GatedDeltaNet)
                layer_delta(i, x_out, seq, is_prefill, token_pos);
            else
                layer_attn(i, x_out, seq, is_prefill, token_pos);
            layer_mlp(i, x_out, seq);
        }
    }
    const TensorDesc& fn = must("final_norm");
    std::vector<float> tmp(static_cast<size_t>(H));
    for (int t = 0; t < seq; ++t) {
        ops::qwen3_rmsnorm(x_out + t * H, f32_ptr(fn), tmp.data(), H, m.rms_eps);
        std::memcpy(x_out + t * H, tmp.data(), sizeof(float) * H);
    }
}

void Session::set_vision_embeds(const float* embeds, int n_vis, int placeholder_id) {
    const int H = store_->model().hidden;
    if (n_vis < 0 || (n_vis > 0 && !embeds)) throw std::runtime_error("vision embeds");
    vis_n_ = n_vis;
    vis_ph_ = placeholder_id;
    vis_h_ = 0;
    vis_w_ = 0;
    if (n_vis == 0) {
        vis_embeds_.clear();
        return;
    }
    vis_embeds_.assign(embeds, embeds + static_cast<size_t>(n_vis) * H);
}

void Session::clear_vision() { set_vision_embeds(nullptr, 0, -1); }

int Session::load_image(const std::string& path) {
    const ModelDesc& m = store_->model();
    const VisionDesc& V = m.vision;
    if (!store_->table().find("visual.patch_embed"))
        throw std::runtime_error("image given but visual.* weights were not loaded (pass --image / --vision)");
    ImageRgb im = load_image_rgb(path);
    const int factor = std::max(1, V.patch * V.spatial_merge);
    const int min_px = 65536;
    const int max_vis = 1024;
    const int max_px = max_vis * factor * factor;
    int oh = 0, ow = 0;
    if (ops::vision_smart_resize(im.h, im.w, factor, min_px, max_px, &oh, &ow) != 0)
        throw std::runtime_error("vision smart_resize failed");
    std::vector<float> resized(static_cast<size_t>(oh) * ow * 3);
    ops::vision_resize_bilinear(im.rgb.data(), im.h, im.w, resized.data(), oh, ow);
    int gh = 0, gw = 0;
    const int n_vis = ops::vision_grid(oh, ow, V.patch, V.spatial_merge, &gh, &gw);
    if (n_vis <= 0) throw std::runtime_error("vision grid empty after resize");
    vis_embeds_.assign(static_cast<size_t>(n_vis) * m.hidden, 0.f);
    ops::vision_encode(resized.data(), oh, ow, V, store_->table(), vis_embeds_.data());
    vis_n_ = n_vis;
    vis_ph_ = V.image_token_id;
    vis_h_ = oh;
    vis_w_ = ow;
    return n_vis;
}

static int first_placeholder_span(const int32_t* ids, int n, int ph, int* start) {
    *start = -1;
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (ids[i] == ph) {
            if (*start < 0) *start = i;
            ++count;
        } else if (*start >= 0) {
            break;
        }
    }
    return count;
}

void Session::prefill(const int32_t* ids, int n) {
    const ModelDesc& m = store_->model();
    if (n <= 0 || n > ctx_) throw std::runtime_error("prefill length");
    int vis_off = -1;
    if (vis_n_ > 0) {
        const int ph = vis_ph_ >= 0 ? vis_ph_ : m.vision.image_token_id;
        const int got = first_placeholder_span(ids, n, ph, &vis_off);
        if (got != vis_n_ || vis_off < 0) {
            throw std::runtime_error("vision token count mismatch (prompt has " + std::to_string(got) +
                                     " image pads, encoder produced " + std::to_string(vis_n_) + ")");
        }
    }
    if (gpu_) {
        if (vis_n_ > 0)
            gpu_->set_vision_override(vis_embeds_.data(), vis_n_, vis_off);
        else
            gpu_->set_vision_override(nullptr, 0, -1);
        gpu_->prefill(ids, n);
        gpu_->copy_logits(logits_.data());
        ctx_tokens_.assign(ids, ids + n);
        pos_ = gpu_->pos();
        return;
    }
    cache_->zero();
    const TensorDesc& emb = must("embed");
    const int H = m.hidden;
    std::vector<float> x(static_cast<size_t>(n) * H);
    for (int t = 0; t < n; ++t) {
        const int id = ids[t];
        if (id < 0 || id >= m.vocab) throw std::runtime_error("token oob");
        embed_row(emb, id, x.data() + t * H, H);
    }
    if (vis_n_ > 0 && vis_off >= 0) {
        std::memcpy(x.data() + static_cast<size_t>(vis_off) * H, vis_embeds_.data(),
                    sizeof(float) * static_cast<size_t>(vis_n_) * H);
    }
    hidden_.assign(static_cast<size_t>(n) * H, 0.f);
    forward_hidden(x.data(), hidden_.data(), true, n, 0);
    const TensorDesc& lh = must("lm_head");
    ops::linear(qw(lh), hidden_.data() + static_cast<size_t>(n - 1) * H, logits_.data(), m.vocab, H);
    last_hidden_.assign(hidden_.data() + static_cast<size_t>(n - 1) * H,
                        hidden_.data() + static_cast<size_t>(n) * H);
    ctx_tokens_.assign(ids, ids + n);
    pos_ = n;
}

void Session::decode_token(int32_t token, float* logits) {
    if (gpu_) {
        gpu_->decode_token(token);
        gpu_->copy_logits(logits_.data());
        if (logits) std::memcpy(logits, logits_.data(), sizeof(float) * store_->model().vocab);
        ctx_tokens_.push_back(token);
        pos_ = gpu_->pos();
        return;
    }
    const ModelDesc& m = store_->model();
    const TensorDesc& emb = must("embed");
    const int H = m.hidden;
    std::vector<float> x(static_cast<size_t>(H));
    embed_row(emb, token, x.data(), H);
    std::vector<float> y(static_cast<size_t>(H));
    forward_hidden(x.data(), y.data(), false, 1, pos_);
    const TensorDesc& lh = must("lm_head");
    ops::linear(qw(lh), y.data(), logits_.data(), m.vocab, H);
    last_hidden_ = y;
    if (logits) std::memcpy(logits, logits_.data(), sizeof(float) * m.vocab);
    ctx_tokens_.push_back(token);
    ++pos_;
}

void Session::mtp_layer_step(float* h, int draft_pos) {
    const ModelDesc& m = store_->model();
    const int H = m.hidden;
    int nq = 4, nkv = 2, hd = 8;
    for (const auto& L : m.layers) {
        if (L.kind == LayerKind::GatedAttn) {
            nq = L.attn.n_q;
            nkv = L.attn.n_kv;
            hd = L.attn.head_dim;
            break;
        }
    }
    const int rotary = std::max(2, static_cast<int>(hd * m.rope.partial_factor));
    const int kn = nkv * hd;
    if (static_cast<int>(mtp_k_.size()) < (draft_pos + 1) * kn) {
        mtp_k_.resize(static_cast<size_t>(draft_pos + 8) * kn, 0.f);
        mtp_v_.resize(mtp_k_.size(), 0.f);
    }
    std::vector<float> x(h, h + H);
    // Reuse fused kernels with the MTP weights (includes residual).
    ops::FusedAttnArgs a{};
    a.x = x.data();
    a.x_inout = x.data();
    a.rms_w = f32_ptr(must("mtp.layers[0].attn_norm"));
    a.Wq = qw(must("mtp.layers[0].attn.wq"));
    a.Wk = qw(must("mtp.layers[0].attn.wk"));
    a.Wv = qw(must("mtp.layers[0].attn.wv"));
    a.Wo = qw(must("mtp.layers[0].attn.wo"));
    a.q_norm = f32_ptr(must("mtp.layers[0].attn.q_norm"));
    a.k_norm = f32_ptr(must("mtp.layers[0].attn.k_norm"));
    a.k_cache = mtp_k_.data();
    a.v_cache = mtp_v_.data();
    a.hidden = H;
    a.n_q = nq;
    a.n_kv = nkv;
    a.head_dim = hd;
    a.rotary_dim = rotary;
    a.pos = draft_pos;
    a.theta = m.rope.theta;
    a.eps = m.rms_eps;
    a.use_simd = fuse_;
    ops::fused_attn_decode(a);
    ops::FusedMlpArgs mlp{};
    mlp.x = x.data();
    mlp.x_inout = x.data();
    mlp.rms_w = f32_ptr(must("mtp.layers[0].ffn_norm"));
    mlp.Wg = qw(must("mtp.layers[0].mlp.gate"));
    mlp.Wu = qw(must("mtp.layers[0].mlp.up"));
    mlp.Wd = qw(must("mtp.layers[0].mlp.down"));
    mlp.hidden = H;
    const TensorDesc& wd = must("mtp.layers[0].mlp.down");
    mlp.intermediate = wd.shape[1] > 0 ? static_cast<int>(wd.shape[1]) : m.intermediate;
    mlp.eps = m.rms_eps;
    mlp.use_simd = fuse_;
    ops::fused_mlp_decode(mlp);
    std::memcpy(h, x.data(), sizeof(float) * H);
}

int Session::mtp_draft(int32_t first, int n, int32_t* out) {
    if (!has_mtp() || n <= 0) return 0;
    const ModelDesc& m = store_->model();
    const int H = m.hidden;
    std::vector<float> h = last_hidden_;
    int32_t token = first;
    mtp_k_.assign(static_cast<size_t>(n + 2) * 64, 0.f);
    mtp_v_ = mtp_k_;
    const TensorDesc& emb = must("embed");
    const TensorDesc& fc = must("mtp.fc");
    const TensorDesc& nh = must("mtp.pre_fc_norm_hidden");
    const TensorDesc& ne = must("mtp.pre_fc_norm_embedding");
    const TensorDesc& nn = must("mtp.norm");
    const TensorDesc& lh = must("lm_head");
    std::vector<float> eh(static_cast<size_t>(H)), ee(static_cast<size_t>(H)), cat(static_cast<size_t>(H) * 2);
    std::vector<float> hp(static_cast<size_t>(H)), hn(static_cast<size_t>(H)), lg(static_cast<size_t>(m.vocab));
    int got = 0;
    for (int i = 0; i < n; ++i) {
        ops::qwen3_rmsnorm(h.data(), f32_ptr(nh), eh.data(), H, m.rms_eps);
        std::vector<float> erow(static_cast<size_t>(H));
        embed_row(emb, token, erow.data(), H);
        ops::qwen3_rmsnorm(erow.data(), f32_ptr(ne), ee.data(), H, m.rms_eps);
        std::memcpy(cat.data(), eh.data(), sizeof(float) * H);
        std::memcpy(cat.data() + H, ee.data(), sizeof(float) * H);
        ops::linear(qw(fc), cat.data(), hp.data(), H, 2 * H);
        h = hp;
        mtp_layer_step(h.data(), i);
        ops::qwen3_rmsnorm(h.data(), f32_ptr(nn), hn.data(), H, m.rms_eps);
        ops::linear(qw(lh), hn.data(), lg.data(), m.vocab, H);
        token = greedy_sample(lg.data(), m.vocab);
        out[got++] = token;
    }
    return got;
}

int Session::ngram_draft(const int32_t* ctx, int ctx_n, int32_t first, int n, int32_t* out) const {
    return ngram_draft_tokens(ctx, ctx_n, first, n, out);
}

int Session::generate(const int32_t* ids, int n, int32_t* out, int cap, const GenerateConfig& cfg) {
    const auto t_pf0 = std::chrono::steady_clock::now();
    prefill(ids, n);
    const auto t_pf1 = std::chrono::steady_clock::now();
    last_prefill_sec_ = std::chrono::duration<double>(t_pf1 - t_pf0).count();
    spec_stats_ = {};
    int produced = 0;
    const int want = std::min(cfg.max_new_tokens, cap);
    if (gpu_) {
        SpecKind sk = cfg.spec == SpecKind::Off ? SpecKind::Off : SpecKind::Ngram;
        if (cfg.spec != SpecKind::Off && draft_) sk = SpecKind::Draft;
        const int spec_n = std::max(0, cfg.spec_n);
        const auto t_d0 = std::chrono::steady_clock::now();
        int n_decode_fwd = 0;
        if (sk == SpecKind::Off && want > 0) {
            // Prefill logits already produced token 0. Only decode the remaining
            // want-1 tokens — the old decode_steps(want) ran one unused forward.
            out[0] = gpu_->greedy();
            ctx_tokens_.push_back(out[0]);
            produced = 1;
            const int n_dec = want - 1;
            if (n_dec > 0) {
                gpu_->decode_steps(n_dec);
                int32_t rest[64];
                gpu_->copy_gen_tokens(rest, n_dec);
                const int take = n_dec < 64 ? n_dec : 64;
                for (int i = 0; i < take; ++i) {
                    out[i + 1] = rest[i];
                    ctx_tokens_.push_back(out[i + 1]);
                }
                produced = 1 + take;
            }
            pos_ = gpu_->pos();
            n_decode_fwd = n_dec;
        }
        while (produced < want) {
            const int32_t t0 = gpu_->greedy();
            out[produced++] = t0;
            ctx_tokens_.push_back(t0);
            int32_t drafts[8];
            int nd = 0;
            if (sk != SpecKind::Off && spec_n > 0 && produced < want) {
                const int take = std::min(std::min(spec_n, want - produced), 7);
                if (sk == SpecKind::Draft)
                    nd = draft_tokens(take, drafts);
                else
                    nd = ngram_draft(ctx_tokens_.data(), static_cast<int>(ctx_tokens_.size()), t0, take, drafts);
                spec_stats_.proposed += nd;
                ++spec_stats_.steps;
            }
            if (nd <= 0) {
                gpu_->decode_token(t0);
                pos_ = gpu_->pos();
                continue;
            }
            int32_t toks[8];
            toks[0] = t0;
            for (int i = 0; i < nd; ++i) toks[i + 1] = drafts[i];
            int32_t preds[8];
            const int k = gpu_->spec_verify(toks, 1 + nd, preds);
            for (int i = 0; i + 1 < k && produced < want; ++i) {
                out[produced++] = drafts[i];
                ctx_tokens_.push_back(drafts[i]);
                ++spec_stats_.accepted;
            }
            pos_ = gpu_->pos();
        }
        const auto t_d1 = std::chrono::steady_clock::now();
        last_decode_sec_ = std::chrono::duration<double>(t_d1 - t_d0).count();
        last_decode_tokens_ = n_decode_fwd > 0 ? n_decode_fwd : produced;
        gpu_->copy_logits(logits_.data());
        return produced;
    }
    const SpecKind sk = resolve_spec(cfg.spec);
    const int spec_n = std::max(0, cfg.spec_n);
    const auto t_d0 = std::chrono::steady_clock::now();
    while (produced < want) {
        const int32_t t0 = greedy_sample(logits_.data(), store_->model().vocab);
        out[produced++] = t0;
        int32_t drafts[16];
        int nd = 0;
        if (sk != SpecKind::Off && spec_n > 0 && produced < want) {
            const int take = std::min(spec_n, 16);
            if (sk == SpecKind::Draft)
                nd = draft_tokens(take, drafts);
            else if (sk == SpecKind::Mtp)
                nd = mtp_draft(t0, take, drafts);
            else
                nd = ngram_draft(ctx_tokens_.data(), static_cast<int>(ctx_tokens_.size()), t0, take, drafts);
            spec_stats_.proposed += nd;
            ++spec_stats_.steps;
        }
        decode_token(t0, nullptr);
        if (gpu_) gpu_->copy_logits(logits_.data());
        for (int i = 0; i < nd && produced < want; ++i) {
            const int32_t target = greedy_sample(logits_.data(), store_->model().vocab);
            if (target != drafts[i]) break;
            out[produced++] = drafts[i];
            ++spec_stats_.accepted;
            decode_token(drafts[i], nullptr);
        }
    }
    const auto t_d1 = std::chrono::steady_clock::now();
    last_decode_sec_ = std::chrono::duration<double>(t_d1 - t_d0).count();
    last_decode_tokens_ = produced;
    return produced;
}

int Session::generate_batch(const int32_t* const* prompts, const int* lens, int n_seq, int32_t* out, int* out_n,
                            int cap, const GenerateConfig& cfg) {
    if (n_seq <= 0) return 0;
    if (n_seq == 1) {
        const int g = generate(prompts[0], lens[0], out, cap, cfg);
        if (out_n) out_n[0] = g;
        return g;
    }
    const int want = std::min(cfg.max_new_tokens, cap);
    if (!gpu_ || gpu_->max_batch() < 2) {
        int total = 0;
        for (int i = 0; i < n_seq; ++i) {
            const int g = generate(prompts[i], lens[i], out + static_cast<size_t>(i) * cap, cap, cfg);
            if (out_n) out_n[i] = g;
            total += g;
        }
        return total;
    }
    const int B = std::min(n_seq, gpu_->max_batch());
    // Same-prompt fast path: one prefill, replicate caches, shared-weight decode.
    const auto t_pf0 = std::chrono::steady_clock::now();
    prefill(prompts[0], lens[0]);
    gpu_->replicate_slot0(B);
    const auto t_pf1 = std::chrono::steady_clock::now();
    last_prefill_sec_ = std::chrono::duration<double>(t_pf1 - t_pf0).count();

    std::vector<int> poss(static_cast<size_t>(B), gpu_->pos());
    std::vector<int32_t> toks(static_cast<size_t>(B));
    std::vector<int> done(static_cast<size_t>(B), 0);
    std::vector<float> blogits(static_cast<size_t>(B) * store_->model().vocab);
    if (out_n) {
        for (int i = 0; i < B; ++i) out_n[i] = 0;
    }
    int total = 0;
    const auto t_d0 = std::chrono::steady_clock::now();
    // First token from the shared prefill logits.
    {
        const int32_t t0 = greedy_sample(logits_.data(), store_->model().vocab);
        for (int b = 0; b < B; ++b) {
            out[static_cast<size_t>(b) * cap] = t0;
            if (out_n) out_n[b] = 1;
            toks[static_cast<size_t>(b)] = t0;
            ++total;
        }
    }
    while (total < B * want) {
        int n_act = 0;
        std::vector<int32_t> atok;
        std::vector<int> apos, amap;
        for (int b = 0; b < B; ++b) {
            if (done[static_cast<size_t>(b)] || (out_n && out_n[b] >= want)) {
                done[static_cast<size_t>(b)] = 1;
                continue;
            }
            atok.push_back(toks[static_cast<size_t>(b)]);
            apos.push_back(poss[static_cast<size_t>(b)]);
            amap.push_back(b);
            ++n_act;
        }
        if (n_act == 0) break;
        gpu_->decode_tokens(atok.data(), apos.data(), n_act);
        gpu_->copy_logits_n(blogits.data(), n_act);
        for (int i = 0; i < n_act; ++i) {
            const int b = amap[static_cast<size_t>(i)];
            const int32_t nxt =
                greedy_sample(blogits.data() + static_cast<size_t>(i) * store_->model().vocab, store_->model().vocab);
            poss[static_cast<size_t>(b)] += 1;
            toks[static_cast<size_t>(b)] = nxt;
            const int k = out_n ? out_n[b] : 0;
            if (k < want) {
                out[static_cast<size_t>(b) * cap + k] = nxt;
                if (out_n) ++out_n[b];
                ++total;
            } else {
                done[static_cast<size_t>(b)] = 1;
            }
        }
    }
    const auto t_d1 = std::chrono::steady_clock::now();
    last_decode_sec_ = std::chrono::duration<double>(t_d1 - t_d0).count();
    last_decode_tokens_ = total;
    return total;
}

} // namespace rapidllm
