#include "rapidllm/api.h"

#include "rapidllm/backend/device.h"
#include "rapidllm/frontend/weight_store.h"
#include "rapidllm/runtime/cuda_engine.h"
#include "rapidllm/runtime/planner.h"
#include "rapidllm/runtime/thread_pool.h"
#include "rapidllm/runtime/sampler.h"
#include "rapidllm/runtime/session.h"
#include "rapidllm/runtime/tokenizer.h"
#include "rapidllm/version.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

using namespace rapidllm;

struct RapidLLM {
    std::unique_ptr<Device> dev;
    WeightStore store;
    Tokenizer tok{48};
    int ctx = 32768;
    bool kv_i8 = false;
    bool fuse = true;
    bool use_cuda = false;
};

struct RapidSession {
    RapidLLM* eng = nullptr;
    std::unique_ptr<Session> sess;
    RapidSessionConfig sc{};
};

static void set_err(RapidError* e, RapidStatus c, const std::string& m) {
    if (!e) return;
    e->code = c;
    std::snprintf(e->message, sizeof(e->message), "%s", m.c_str());
}

extern "C" {

int rapidllm_api_version(void) { return RAPIDLLM_API_VERSION; }
const char* rapidllm_version_string(void) { return rapidllm::version(); }

void rapidllm_set_fuse(RapidLLM* e, int fuse) {
    if (!e) return;
    e->fuse = fuse != 0;
}

RapidLLM* rapidllm_load(const RapidConfig* cfg, RapidError* err) {
    try {
        if (!cfg || !cfg->model_path) {
            set_err(err, RAPID_ERR_RANGE, "null config");
            return nullptr;
        }
        auto eng = std::make_unique<RapidLLM>();
        const char* dev = cfg->device ? cfg->device : "cpu";
        if (std::strcmp(dev, "cuda") == 0 || std::strcmp(dev, "gpu") == 0) {
            if (!cuda_gen::available()) {
                set_err(err, RAPID_ERR_DEVICE, "CUDA device requested but no GPU");
                return nullptr;
            }
            eng->use_cuda = true;
        } else if (std::strcmp(dev, "cpu") != 0) {
            set_err(err, RAPID_ERR_DEVICE, "device must be cpu or cuda");
            return nullptr;
        }
        eng->dev = create_device(DeviceKind::CPU);
        eng->ctx = cfg->ctx > 0 ? cfg->ctx : 32768;
        eng->kv_i8 = cfg->kv_i8 != 0;
        eng->fuse = cfg->fuse != 0;
        LoadOptions opt;
        opt.language_only = cfg->language_only != 0;
        opt.mmap = cfg->mmap != 0;
        opt.hugepage = cfg->hugepage != 0;
        opt.repack_int4 = cfg->repack_int4 != 0;
        opt.max_layers = cfg->max_layers;
        if (cfg->threads > 0) set_num_threads(cfg->threads);
        eng->store = WeightStore::open(cfg->model_path, *eng->dev, opt);

        PlannerInput pin;
        pin.weight_bytes = eng->store.table().total_nbytes();
        pin.ctx = eng->ctx;
        pin.available_ram = cfg->max_ram_bytes ? cfg->max_ram_bytes : detect_available_ram();
        pin.pad_bytes = 8ull << 20;
        const ModelDesc& m = eng->store.model();
        for (const LayerDesc& L : m.layers) {
            if (L.kind == LayerKind::GatedAttn) {
                ++pin.n_attn_layers;
                pin.n_kv_heads = L.attn.n_kv;
                pin.head_dim = L.attn.head_dim;
            } else {
                ++pin.n_delta;
                pin.nv = L.delta.n_v_heads;
                pin.dk = L.delta.k_dim;
                pin.dv = L.delta.v_dim;
                pin.conv_k = L.delta.conv_k;
                pin.conv_dim = L.delta.n_k_heads * L.delta.k_dim * 2 + L.delta.n_v_heads * L.delta.v_dim;
            }
        }
        pin.kv_i8 = eng->kv_i8;
        PlannerResult pr = plan_memory(pin);
        if (!pr.ok) {
            set_err(err, RAPID_ERR_NOMEM, pr.message);
            return nullptr;
        }
        eng->tok = Tokenizer(m.vocab);
        if (err) err->code = RAPID_OK;
        return eng.release();
    } catch (const LoadError& e) {
        set_err(err, RAPID_ERR_FORMAT, e.what());
        return nullptr;
    } catch (const std::exception& e) {
        set_err(err, RAPID_ERR_INTERNAL, e.what());
        return nullptr;
    }
}

void rapidllm_free(RapidLLM* e) { delete e; }

int rapidllm_vocab(const RapidLLM* e) { return e ? e->store.model().vocab : 0; }

int rapidllm_encode(RapidLLM* e, const char* utf8, int32_t* ids, int cap, RapidError* err) {
    if (!e || !utf8 || !ids) {
        set_err(err, RAPID_ERR_RANGE, "null");
        return -1;
    }
    return e->tok.encode(utf8, ids, cap);
}

int rapidllm_decode_ids(RapidLLM* e, const int32_t* ids, int n, char* out, int cap, RapidError* err) {
    if (!e || !ids || !out) {
        set_err(err, RAPID_ERR_RANGE, "null");
        return -1;
    }
    const std::string s = e->tok.decode(ids, n);
    if (cap <= 0) return static_cast<int>(s.size());
    const int cpy = std::min(cap - 1, static_cast<int>(s.size()));
    std::memcpy(out, s.data(), static_cast<size_t>(cpy));
    out[cpy] = 0;
    return cpy;
}

RapidSession* rapidllm_session_new(RapidLLM* e, const RapidSessionConfig* sc, RapidError* err) {
    try {
        if (!e) {
            set_err(err, RAPID_ERR_RANGE, "null engine");
            return nullptr;
        }
        auto s = std::make_unique<RapidSession>();
        s->eng = e;
        if (sc) s->sc = *sc;
        else {
            s->sc.enable_thinking = 1;
            s->sc.max_new_tokens = 16;
            s->sc.spec = 3;
            s->sc.spec_n = 3;
        }
        s->sess = std::make_unique<Session>(*e->dev, e->store, e->ctx, e->kv_i8, e->fuse, e->use_cuda);
        return s.release();
    } catch (const std::exception& ex) {
        set_err(err, RAPID_ERR_INTERNAL, ex.what());
        return nullptr;
    }
}

int rapidllm_session_set_draft(RapidSession* target, RapidSession* draft, RapidError* err) {
    try {
        if (!target || !target->sess) {
            set_err(err, RAPID_ERR_RANGE, "null target session");
            return RAPID_ERR_RANGE;
        }
        target->sess->set_draft(draft && draft->sess ? draft->sess.get() : nullptr);
        return RAPID_OK;
    } catch (const std::exception& e) {
        set_err(err, RAPID_ERR_INTERNAL, e.what());
        return RAPID_ERR_INTERNAL;
    }
}

void rapidllm_session_set_max_new(RapidSession* s, int max_new_tokens) {
    if (!s) return;
    s->sc.max_new_tokens = max_new_tokens > 0 ? max_new_tokens : 16;
}

void rapidllm_session_free(RapidSession* s) { delete s; }

int rapidllm_prefill(RapidSession* s, const int32_t* ids, int n, RapidError* err) {
    try {
        s->sess->prefill(ids, n);
        return RAPID_OK;
    } catch (const std::exception& e) {
        set_err(err, RAPID_ERR_INTERNAL, e.what());
        return RAPID_ERR_INTERNAL;
    }
}

int rapidllm_decode(RapidSession* s, int32_t token, RapidLogitsView* view, RapidError* err) {
    try {
        s->sess->decode_token(token, nullptr);
        if (view) {
            view->data = s->sess->last_logits();
            view->vocab = s->eng->store.model().vocab;
        }
        return RAPID_OK;
    } catch (const std::exception& e) {
        set_err(err, RAPID_ERR_INTERNAL, e.what());
        return RAPID_ERR_INTERNAL;
    }
}

int rapidllm_sample(RapidSession* s, const RapidSampleParams*, int32_t* token, RapidError* err) {
    if (!token) {
        set_err(err, RAPID_ERR_RANGE, "null token");
        return RAPID_ERR_RANGE;
    }
    *token = greedy_sample(s->sess->last_logits(), s->eng->store.model().vocab);
    return RAPID_OK;
}

int rapidllm_generate(RapidSession* s, const int32_t* ids, int n, const RapidSampleParams* sp,
                      int32_t* out, int cap, RapidError* err) {
    try {
        GenerateConfig cfg;
        cfg.max_new_tokens = s->sc.max_new_tokens > 0 ? s->sc.max_new_tokens : 16;
        cfg.greedy = !sp || sp->greedy || sp->temperature <= 0.f;
        cfg.enable_thinking = s->sc.enable_thinking != 0;
        cfg.ctx = s->eng->ctx;
        cfg.spec = static_cast<SpecKind>(s->sc.spec);
        cfg.spec_n = s->sc.spec_n > 0 ? s->sc.spec_n : 3;
        const int got = s->sess->generate(ids, n, out, cap, cfg);
        if (err) err->code = RAPID_OK;
        return got;
    } catch (const std::exception& e) {
        set_err(err, RAPID_ERR_INTERNAL, e.what());
        return -1;
    }
}

int rapidllm_generate_batch(RapidSession* s, const int32_t* ids, int n, int n_seq, const RapidSampleParams* sp,
                            int32_t* out, int cap_each, int* out_n, RapidError* err) {
    try {
        if (!s || !s->sess || !ids || n <= 0 || n_seq <= 0) {
            set_err(err, RAPID_ERR_RANGE, "batch generate args");
            return -1;
        }
        GenerateConfig cfg;
        cfg.max_new_tokens = s->sc.max_new_tokens > 0 ? s->sc.max_new_tokens : 16;
        cfg.greedy = !sp || sp->greedy || sp->temperature <= 0.f;
        cfg.enable_thinking = s->sc.enable_thinking != 0;
        cfg.ctx = s->eng->ctx;
        cfg.spec = SpecKind::Off;
        cfg.spec_n = 0;
        std::vector<const int32_t*> ptrs(static_cast<size_t>(n_seq), ids);
        std::vector<int> lens(static_cast<size_t>(n_seq), n);
        const int got = s->sess->generate_batch(ptrs.data(), lens.data(), n_seq, out, out_n, cap_each, cfg);
        if (err) err->code = RAPID_OK;
        return got;
    } catch (const std::exception& e) {
        set_err(err, RAPID_ERR_INTERNAL, e.what());
        return -1;
    }
}

void rapidllm_spec_stats(RapidSession* s, RapidSpecStats* st) {
    if (!s || !st) return;
    const auto ss = s->sess->spec_stats();
    st->proposed = ss.proposed;
    st->accepted = ss.accepted;
    st->steps = ss.steps;
}

void rapidllm_bench_stats(RapidSession* s, double* prefill_s, double* decode_s, int* n_decode) {
    if (!s || !s->sess) return;
    if (prefill_s) *prefill_s = s->sess->last_prefill_sec();
    if (decode_s) *decode_s = s->sess->last_decode_sec();
    if (n_decode) *n_decode = s->sess->last_decode_tokens();
}

int rapidllm_uses_cuda(const RapidSession* s) { return s && s->sess && s->sess->uses_cuda() ? 1 : 0; }

} // extern "C"
