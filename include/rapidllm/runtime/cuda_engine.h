#pragma once

#include "rapidllm/frontend/weight_store.h"

#include <cstdint>
#include <memory>

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
};

} // namespace rapidllm::cuda_gen
