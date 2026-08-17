#include "rapidllm/backend/device.h"
#include "rapidllm/frontend/weight_store.h"
#include "rapidllm/runtime/cuda_engine.h"
#include "rapidllm/runtime/session.h"

#include "frontend/json_mini.h"

#include <algorithm>
#include <cstdio>
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
    const bool have = cuda_gen::available();
    std::printf("cuda_available=%d\n", have ? 1 : 0);

    const std::filesystem::path root = RAPIDLLM_FIXTURE_DIR;
    const Json g = parse_json(slurp(root / "tiny_hybrid" / "golden.json"));
    std::vector<int32_t> prompt, want, mtp_gold;
    for (const Json& x : g.at("prompt_ids").as_arr()) prompt.push_back(x.as_int());
    for (const Json& x : g.at("greedy_ids").as_arr()) want.push_back(x.as_int());
    for (const Json& x : g.at("mtp_draft_ids").as_arr()) mtp_gold.push_back(x.as_int());
    const int n_new = g.at("n_new").as_int();

    auto dev = create_device(DeviceKind::CPU);
    WeightStore store = WeightStore::open(root / "tiny_hybrid", *dev, LoadOptions{});

    if (!have) {
        auto eng = cuda_gen::Engine::create(store, 64);
        if (eng) {
            std::fprintf(stderr, "available=0 but engine created\n");
            return 1;
        }
        std::printf("test_cuda_decode ok (no GPU, stub path)\n");
        return 0;
    }

    Session sess(*dev, store, 64, false, true, true);
    if (!sess.uses_cuda()) {
        std::fprintf(stderr, "Session did not attach CUDA engine\n");
        return 1;
    }
    GenerateConfig cfg;
    cfg.max_new_tokens = n_new;
    cfg.greedy = true;
    cfg.spec = SpecKind::Off;
    std::vector<int32_t> out(static_cast<size_t>(n_new));
    const int got = sess.generate(prompt.data(), static_cast<int>(prompt.size()), out.data(), n_new, cfg);
    out.resize(static_cast<size_t>(got));
    std::printf("cuda tokens:");
    for (int t : out) std::printf(" %d", t);
    std::printf("\n");
    if (out != want) {
        std::fprintf(stderr, "CUDA generate != golden\n");
        return 1;
    }
    if (sess.last_decode_tokens() <= 0 || sess.last_decode_sec() < 0) {
        std::fprintf(stderr, "missing decode timing\n");
        return 1;
    }
    std::printf("decode_tok/s=%.4f\n",
                sess.last_decode_tokens() / std::max(sess.last_decode_sec(), 1e-9));

    // CUDA generate with the loaded MTP head — must not fall back to n-gram.
    WeightStore store_m = WeightStore::open(root / "tiny_hybrid", *dev, LoadOptions{});
    Session sm(*dev, store_m, 64, false, true, true);
    if (!sm.uses_cuda()) {
        std::fprintf(stderr, "MTP session lost CUDA engine\n");
        return 1;
    }
    if (!sm.has_mtp()) {
        std::fprintf(stderr, "fixture missing MTP weights\n");
        return 1;
    }
    GenerateConfig cm = cfg;
    cm.spec = SpecKind::Mtp;
    cm.spec_n = 3;
    std::vector<int32_t> mout(static_cast<size_t>(n_new));
    const int mgot = sm.generate(prompt.data(), static_cast<int>(prompt.size()), mout.data(), n_new, cm);
    mout.resize(static_cast<size_t>(mgot));
    std::printf("cuda_mtp tokens:");
    for (int t : mout) std::printf(" %d", t);
    std::printf("\n");
    const SpecStats st = sm.spec_stats();
    std::printf("cuda_mtp proposed=%d accepted=%d steps=%d\n", st.proposed, st.accepted, st.steps);
    if (mout != want) {
        std::fprintf(stderr, "CUDA MTP generate != golden / spec-off\n");
        return 1;
    }
    if (st.proposed <= 0 || st.steps <= 0) {
        std::fprintf(stderr, "CUDA MTP did not draft (silent n-gram/off?)\n");
        return 1;
    }

    // Standalone draft after CUDA prefill must see last hidden (not zeros).
    WeightStore store_d = WeightStore::open(root / "tiny_hybrid", *dev, LoadOptions{});
    Session sd(*dev, store_d, 64, false, true, true);
    sd.prefill(prompt.data(), static_cast<int>(prompt.size()));
    std::vector<int32_t> drafts(8);
    const int nd = sd.mtp_draft(want[0], 3, drafts.data());
    drafts.resize(static_cast<size_t>(nd));
    std::printf("cuda_mtp_draft:");
    for (int t : drafts) std::printf(" %d", t);
    std::printf("\n");
    if (drafts != mtp_gold) {
        std::fprintf(stderr, "CUDA last-hidden MTP draft != fixture golden\n");
        return 1;
    }

    // flash_gqa6_selftest runs in EngineImpl ctor (24Q/4KV hd=256 vs per-head flash).
    std::printf("test_cuda_decode ok GPU matches golden\n");
    return 0;
}
