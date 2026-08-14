#pragma once

#include <cstdint>

namespace rapidllm {

// General n-gram draft. No model- or prompt-specific token ids.
// 1) Longest suffix of length 2–3 that also occurs earlier; copy tokens after that hit.
// 2) Else if the last two tokens are equal, copy the last token n times.
// 3) Else unigram suffix match.
int ngram_draft_tokens(const int32_t* ctx, int ctx_n, int32_t first, int n, int32_t* out);

} // namespace rapidllm
