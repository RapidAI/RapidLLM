#pragma once

#include "rapidllm/backend/device.h"
#include "rapidllm/frontend/weight_store.h"
#include "rapidllm/runtime/cuda_engine.h"
#include "rapidllm/runtime/dual_cache.h"
#include "rapidllm/runtime/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rapidllm {

enum class SpecKind { Off = 0, Ngram = 1, Mtp = 2, Auto = 3, Draft = 4 };

struct SpecStats {
    int proposed = 0;
    int accepted = 0;
    int steps = 0;
};

struct GenerateConfig {
    int max_new_tokens = 16;
    bool greedy = true;
    bool enable_thinking = true;
    int ctx = 32768;
    bool fuse = true;
    SpecKind spec = SpecKind::Auto;
    int spec_n = 3;
    int ngram = 3;
};

class Session {
public:
    Session(Device& dev, WeightStore& store, int ctx, bool kv_i8, bool fuse = true, bool use_cuda = false);
    void set_fuse(bool f) { fuse_ = f; }
    bool fuse() const { return fuse_; }
    bool uses_cuda() const { return static_cast<bool>(gpu_); }
    int max_batch() const { return gpu_ ? gpu_->max_batch() : 1; }
    double last_prefill_sec() const { return last_prefill_sec_; }
    double last_decode_sec() const { return last_decode_sec_; }
    int last_decode_tokens() const { return last_decode_tokens_; }

    void prefill(const int32_t* ids, int n);
    void decode_token(int32_t token, float* logits);
    int generate(const int32_t* ids, int n, int32_t* out, int cap, const GenerateConfig& cfg);
    // Continuous batch: n_seq independent prompts, one weight-shared decode step per token.
    // out is n_seq * cap; out_n[i] is tokens written for sequence i. Returns total tokens.
    int generate_batch(const int32_t* const* prompts, const int* lens, int n_seq, int32_t* out, int* out_n,
                       int cap, const GenerateConfig& cfg);

    const ModelDesc& model() const { return store_->model(); }
    const float* last_logits() const { return logits_.data(); }
    const float* last_hidden() const { return last_hidden_.data(); }
    int pos() const { return pos_; }
    bool has_mtp() const;
    SpecStats spec_stats() const { return spec_stats_; }
    int mtp_draft(int32_t first, int n, int32_t* out);
    int ngram_draft(const int32_t* ctx, int ctx_n, int32_t first, int n, int32_t* out) const;
    // Smaller same-vocab model used as speculative draft. Not owned.
    void set_draft(Session* draft);
    Session* draft() const { return draft_; }

private:
    void forward_hidden(const float* x_in, float* x_out, bool is_prefill, int seq, int token_pos);
    void layer_delta(int i, float* x, int seq, bool prefill, int token_pos);
    void layer_attn(int i, float* x, int seq, bool prefill, int token_pos);
    void layer_mlp(int i, float* x, int seq);
    void mtp_layer_step(float* h, int draft_pos);
    const TensorDesc& must(std::string_view ir) const;
    const TensorDesc* find_w(std::string_view ir) const;
    SpecKind resolve_spec(SpecKind s) const;

    int draft_tokens(int take, int32_t* out);

    Device* dev_ = nullptr;
    WeightStore* store_ = nullptr;
    Session* draft_ = nullptr;
    std::unique_ptr<cuda_gen::Engine> gpu_;
    std::unique_ptr<DualCache> cache_;
    std::vector<float> hidden_;
    std::vector<float> scratch_;
    std::vector<float> logits_;
    std::vector<float> last_hidden_;
    std::vector<float> mtp_k_;
    std::vector<float> mtp_v_;
    std::vector<int32_t> ctx_tokens_;
    SpecStats spec_stats_{};
    int pos_ = 0;
    int ctx_ = 0;
    bool fuse_ = true;
    double last_prefill_sec_ = 0;
    double last_decode_sec_ = 0;
    int last_decode_tokens_ = 0;
};

} // namespace rapidllm
