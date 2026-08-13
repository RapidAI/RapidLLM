#pragma once

#include <cstdint>

namespace rapidllm {
int32_t greedy_sample(const float* logits, int vocab);
}
