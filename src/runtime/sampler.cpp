#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace rapidllm {

int32_t greedy_sample(const float* logits, int vocab) {
    int32_t best = 0;
    float m = logits[0];
    for (int i = 1; i < vocab; ++i) {
        if (logits[i] > m) {
            m = logits[i];
            best = i;
        }
    }
    return best;
}

} // namespace rapidllm
