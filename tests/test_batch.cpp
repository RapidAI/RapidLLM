#include "rapidllm/backend/device.h"
#include "rapidllm/frontend/weight_store.h"
#include "rapidllm/runtime/session.h"

#include "frontend/json_mini.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef RAPIDLLM_FIXTURE_DIR
#define RAPIDLLM_FIXTURE_DIR "."
#endif

static std::string slurp(const std::filesystem::path& p) {
    std::ifstream in(p);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int main() {
    using namespace rapidllm;
    const std::filesystem::path root = RAPIDLLM_FIXTURE_DIR;
    const Json g = parse_json(slurp(root / "tiny_hybrid" / "golden.json"));
    std::vector<int32_t> prompt, want;
    for (const Json& x : g.at("prompt_ids").as_arr()) prompt.push_back(x.as_int());
    for (const Json& x : g.at("greedy_ids").as_arr()) want.push_back(x.as_int());
    const int n_new = g.at("n_new").as_int();

#if defined(_WIN32)
    _putenv_s("RAPIDLLM_MAX_BATCH", "4");
#else
    setenv("RAPIDLLM_MAX_BATCH", "4", 1);
#endif
    auto dev = create_device(DeviceKind::CPU);
    WeightStore store = WeightStore::open(root / "tiny_hybrid", *dev, LoadOptions{});
    Session sess(*dev, store, 64, false, true, true);
    if (sess.uses_cuda() && sess.max_batch() < 2) {
        std::fprintf(stderr, "expected CUDA max_batch>=2 after RAPIDLLM_MAX_BATCH=4\n");
        return 1;
    }
    GenerateConfig cfg;
    cfg.max_new_tokens = n_new;
    cfg.greedy = true;
    cfg.spec = SpecKind::Off;

    const int32_t* ptrs[2] = {prompt.data(), prompt.data()};
    const int lens[2] = {static_cast<int>(prompt.size()), static_cast<int>(prompt.size())};
    std::vector<int32_t> out(static_cast<size_t>(2 * n_new));
    int out_n[2] = {0, 0};
    const int total = sess.generate_batch(ptrs, lens, 2, out.data(), out_n, n_new, cfg);
    std::printf("batch total=%d n0=%d n1=%d uses_cuda=%d max_batch=%d\n", total, out_n[0], out_n[1],
                sess.uses_cuda() ? 1 : 0, sess.max_batch());
    if (out_n[0] != n_new || out_n[1] != n_new) {
        std::fprintf(stderr, "batch lengths mismatch\n");
        return 1;
    }
    for (int s = 0; s < 2; ++s) {
        for (int i = 0; i < n_new; ++i) {
            if (out[static_cast<size_t>(s) * n_new + i] != want[static_cast<size_t>(i)]) {
                std::fprintf(stderr, "batch slot %d token %d got %d want %d\n", s, i,
                             out[static_cast<size_t>(s) * n_new + i], want[static_cast<size_t>(i)]);
                return 1;
            }
        }
    }
    std::printf("test_batch ok both slots match golden");
    for (int t : want) std::printf(" %d", t);
    std::printf("\n");

    // Independent prompts: shared-weight decode, distinct greedy outputs.
    std::vector<int32_t> p1 = prompt;
    std::vector<int32_t> p2 = prompt;
    if (!p2.empty()) p2[0] = p2[0] == 1 ? 2 : 1;
    const int32_t* mptrs[2] = {p1.data(), p2.data()};
    const int mlens[2] = {static_cast<int>(p1.size()), static_cast<int>(p2.size())};
    std::vector<int32_t> mout(static_cast<size_t>(2 * n_new), 0);
    int mn[2] = {0, 0};
    const int mtot = sess.generate_batch(mptrs, mlens, 2, mout.data(), mn, n_new, cfg);
    std::printf("mixed total=%d n0=%d n1=%d\n", mtot, mn[0], mn[1]);
    if (mn[0] != n_new || mn[1] != n_new) {
        std::fprintf(stderr, "mixed batch lengths mismatch\n");
        return 1;
    }
    bool same_out = true;
    for (int i = 0; i < n_new; ++i) {
        if (mout[static_cast<size_t>(i)] != mout[static_cast<size_t>(n_new) + i]) same_out = false;
    }
    bool slot0_golden = true;
    for (int i = 0; i < n_new; ++i) {
        if (mout[static_cast<size_t>(i)] != want[static_cast<size_t>(i)]) slot0_golden = false;
    }
    std::printf("mixed tokens0:");
    for (int i = 0; i < n_new; ++i) std::printf(" %d", mout[static_cast<size_t>(i)]);
    std::printf("\nmixed tokens1:");
    for (int i = 0; i < n_new; ++i) std::printf(" %d", mout[static_cast<size_t>(n_new) + i]);
    std::printf("\n");
    if (same_out) {
        std::fprintf(stderr, "mixed prompts produced identical greedy outputs\n");
        return 1;
    }
    if (sess.uses_cuda() && !slot0_golden) {
        std::fprintf(stderr, "mixed slot0 no longer matches golden\n");
        return 1;
    }
    std::printf("test_batch mixed ok distinct greedy\n");

    // B>32: argmax_final_rows used to launch <<<1,32>>> and leave slots 32+ at 0.
    if (sess.uses_cuda()) {
#if defined(_WIN32)
        _putenv_s("RAPIDLLM_MAX_BATCH", "48");
#else
        setenv("RAPIDLLM_MAX_BATCH", "48", 1);
#endif
        WeightStore store_w = WeightStore::open(root / "tiny_hybrid", *dev, LoadOptions{});
        Session sw(*dev, store_w, 64, false, true, true);
        if (!sw.uses_cuda() || sw.max_batch() < 40) {
            std::fprintf(stderr, "wide batch session max_batch=%d (want >=40)\n", sw.max_batch());
            return 1;
        }
        constexpr int WB = 40;
        std::vector<const int32_t*> wptrs(static_cast<size_t>(WB));
        std::vector<int> wlens(static_cast<size_t>(WB));
        for (int b = 0; b < WB; ++b) {
            wptrs[static_cast<size_t>(b)] = prompt.data();
            wlens[static_cast<size_t>(b)] = static_cast<int>(prompt.size());
        }
        std::vector<int32_t> wout(static_cast<size_t>(WB) * n_new, 0);
        std::vector<int> wn(static_cast<size_t>(WB), 0);
        const int wtot = sw.generate_batch(wptrs.data(), wlens.data(), WB, wout.data(), wn.data(), n_new, cfg);
        int n_zero = 0, n_mismatch = 0;
        for (int b = 0; b < WB; ++b) {
            if (wn[static_cast<size_t>(b)] != n_new) {
                std::fprintf(stderr, "wide slot %d n=%d want %d\n", b, wn[static_cast<size_t>(b)], n_new);
                return 1;
            }
            bool allz = true, match = true;
            for (int i = 0; i < n_new; ++i) {
                const int32_t got = wout[static_cast<size_t>(b) * n_new + i];
                if (got != 0) allz = false;
                if (got != want[static_cast<size_t>(i)]) match = false;
            }
            if (allz) ++n_zero;
            if (!match) ++n_mismatch;
        }
        std::printf("wide B=%d total=%d max_batch=%d zero_seqs=%d mismatch=%d tokens32:", WB, wtot,
                    sw.max_batch(), n_zero, n_mismatch);
        for (int i = 0; i < n_new; ++i) std::printf(" %d", wout[static_cast<size_t>(32) * n_new + i]);
        std::printf("\n");
        if (n_zero || n_mismatch) {
            std::fprintf(stderr, "wide B=%d failed zero=%d mismatch=%d\n", WB, n_zero, n_mismatch);
            return 1;
        }
        std::printf("test_batch wide B=%d ok every slot matches golden\n", WB);
    }
    return 0;
}
