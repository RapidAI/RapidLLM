#include "rapidllm/runtime/ngram_draft.h"

#include <algorithm>
#include <vector>

namespace rapidllm {

int ngram_draft_tokens(const int32_t* ctx, int ctx_n, int32_t first, int n, int32_t* out) {
    if (n <= 0 || ctx_n <= 0 || !ctx || !out) return 0;
    std::vector<int32_t> hay;
    hay.reserve(static_cast<size_t>(ctx_n + 1));
    hay.insert(hay.end(), ctx, ctx + ctx_n);
    if (hay.empty() || hay.back() != first) hay.push_back(first);
    const int hn = static_cast<int>(hay.size());
    auto try_ng = [&](int ng) -> int {
        if (ng < 1 || hn <= ng) return 0;
        const int start = hn - ng;
        int found = -1;
        for (int i = 0; i + ng <= start; ++i) {
            bool ok = true;
            for (int j = 0; j < ng; ++j) {
                if (hay[i + j] != hay[start + j]) {
                    ok = false;
                    break;
                }
            }
            if (ok) found = i + ng;
        }
        if (found < 0 || found >= hn) return 0;
        int got = 0;
        for (int k = 0; k < n && found + k < hn; ++k) out[got++] = hay[found + k];
        return got;
    };
    for (int ng = std::min(3, hn - 1); ng >= 2; --ng) {
        const int got = try_ng(ng);
        if (got > 0) return got;
    }
    if (hn >= 2 && hay[hn - 1] == hay[hn - 2]) {
        const int32_t t = hay[hn - 1];
        for (int i = 0; i < n; ++i) out[i] = t;
        return n;
    }
    return try_ng(1);
}

} // namespace rapidllm
