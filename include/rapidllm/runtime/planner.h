#pragma once

#include "rapidllm/frontend/weight_store.h"

#include <cstdint>
#include <string>

namespace rapidllm {

struct PlannerInput {
    uint64_t weight_bytes = 0;
    int ctx = 32768;
    int n_attn_layers = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int n_delta = 0;
    int nv = 0, dk = 0, dv = 0;
    int conv_dim = 0, conv_k = 4;
    bool kv_i8 = false;
    uint64_t available_ram = 0;
    uint64_t pad_bytes = 64ull * 1024 * 1024;
};

struct PlannerResult {
    bool ok = true;
    std::string message;
    uint64_t need_bytes = 0;
    uint64_t kv_bytes = 0;
    uint64_t state_bytes = 0;
};

PlannerResult plan_memory(const PlannerInput& in);
uint64_t detect_available_ram();

} // namespace rapidllm
