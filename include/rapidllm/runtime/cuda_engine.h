#pragma once

#include "rapidllm/frontend/weight_store.h"

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace rapidllm::cuda_gen {

bool available();

class Engine {
public:
    static std::unique_ptr<Engine> create(WeightStore& store, int ctx);
    virtual ~Engine() = default;

    virtual void prefill(const int32_t* ids, int n) = 0;
    virtual void decode_token(int32_t token) = 0;
    virtual void copy_logits(float* host) const = 0;
    virtual int32_t greedy() const = 0;
    virtual int pos() const = 0;
    virtual int vocab() const = 0;
    virtual int hidden() const = 0;
    virtual int max_batch() const { return 1; }
    // Continuous batch: B independent sequences, one weight pass. poss[b] is each seq pos.
    virtual void decode_tokens(const int32_t* tokens, const int* poss, int B) {
        if (B == 1) decode_token(tokens[0]);
        else
            throw std::runtime_error("CUDA engine missing batch decode");
    }
    virtual void copy_logits_n(float* host, int B) const {
        copy_logits(host);
        (void)B;
    }
    // Copy slot-0 recurrent/KV into slots 1..n-1 (same-prompt continuous batch).
    virtual void replicate_slot0(int /*n_slots*/) {}
    // Speculative verify: consume toks[0..T) with one weight pass from current pos.
    // preds[i] = greedy after toks[i]. Returns n_consumed (>=1). Rolls state back to
    // after toks[0..n_consumed). greedy() is the next token (preds[n_consumed-1]).
    virtual int spec_verify(const int32_t* toks, int T, int32_t* preds) {
        if (T <= 0 || !toks) return 0;
        decode_token(toks[0]);
        if (preds) preds[0] = greedy();
        return 1;
    }
};

} // namespace rapidllm::cuda_gen
