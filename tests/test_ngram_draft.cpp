#include "rapidllm/runtime/ngram_draft.h"

#include <algorithm>
#include <cstdio>
#include <vector>

static int fails = 0;

static void expect_eq(const std::vector<int32_t>& got, const std::vector<int32_t>& want, const char* name) {
    if (got != want) {
        std::fprintf(stderr, "fail %s got", name);
        for (int t : got) std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, " want");
        for (int t : want) std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "\n");
        ++fails;
    }
}

static std::vector<int32_t> draft(const std::vector<int32_t>& ctx, int32_t first, int n) {
    std::vector<int32_t> out(static_cast<size_t>(n), -1);
    const int got = rapidllm::ngram_draft_tokens(ctx.data(), static_cast<int>(ctx.size()), first, n, out.data());
    out.resize(static_cast<size_t>(got));
    return out;
}

int main() {
    // History suffix: [1,2,3,1,2] matches [1,2] earlier, continue with 3.
    expect_eq(draft({1, 2, 3, 1, 2}, 2, 1), {3}, "suffix continue");

    // Last two equal, no longer suffix hit → copy last token.
    expect_eq(draft({7, 8, 8}, 8, 3), {8, 8, 8}, "last two equal");

    // Unigram hit on an earlier 0 copies only the real tail (31 0), not an
    // invented 11-token period-2 fill.
    expect_eq(draft({4, 5, 0, 31, 0}, 0, 11), {31, 0}, "no period2 fill on 4 5 0 31 0");

    // No suffix hit and last two differ → empty.
    expect_eq(draft({10, 11, 12}, 13, 3), {}, "no match");

    // Negative: ctx that looks like the bakeoff greedy prefix. Drafts must follow
    // suffix-match / last-two-equal only — not a named fill of the last token.
    {
        const std::vector<int32_t> ctx = {17, 3, 3, 59, 3};
        auto got = draft(ctx, 3, 3);
        const bool last_two_equal = ctx[ctx.size() - 1] == ctx[ctx.size() - 2];
        if (last_two_equal) {
            std::fprintf(stderr, "fail negative setup last two should differ\n");
            ++fails;
        } else if (!got.empty() && got[0] == ctx.back() &&
                   std::all_of(got.begin(), got.end(), [&](int32_t t) { return t == ctx.back(); })) {
            std::fprintf(stderr, "fail bakeoff-style fill of last token on 17,3,3,59,3\n");
            ++fails;
        }
    }

    if (fails) {
        std::fprintf(stderr, "test_ngram_draft %d failure(s)\n", fails);
        return 1;
    }
    std::printf("test_ngram_draft ok\n");
    return 0;
}
