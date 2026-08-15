#include "rapidllm/backend/device.h"
#include "rapidllm/frontend/weight_store.h"
#include "rapidllm/runtime/cuda_engine.h"
#include "rapidllm/runtime/session.h"

#include "frontend/json_mini.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
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
    std::vector<int32_t> prompt, want, mtp_gold;
    for (const Json& x : g.at("prompt_ids").as_arr()) prompt.push_back(x.as_int());
    for (const Json& x : g.at("greedy_ids").as_arr()) want.push_back(x.as_int());
    for (const Json& x : g.at("mtp_draft_ids").as_arr()) mtp_gold.push_back(x.as_int());
    const int n_new = g.at("n_new").as_int();

    auto run = [&](SpecKind spec) {
        auto dev = create_device(DeviceKind::CPU);
        WeightStore store = WeightStore::open(root / "tiny_hybrid", *dev, LoadOptions{});
        Session sess(*dev, store, 64, false, true);
        GenerateConfig cfg;
        cfg.max_new_tokens = n_new;
        cfg.greedy = true;
        cfg.spec = spec;
        cfg.spec_n = 3;
        std::vector<int32_t> out(static_cast<size_t>(n_new));
        const int got = sess.generate(prompt.data(), static_cast<int>(prompt.size()), out.data(), n_new, cfg);
        out.resize(static_cast<size_t>(got));
        return std::make_pair(out, sess.spec_stats());
    };

    const auto [off, st0] = run(SpecKind::Off);
    const auto [mtp, stm] = run(SpecKind::Mtp);
    const auto [ng, stn] = run(SpecKind::Ngram);

    auto dump = [](const char* tag, const std::vector<int32_t>& v) {
        std::printf("%s:", tag);
        for (int t : v) std::printf(" %d", t);
        std::printf("\n");
    };
    dump("golden", want);
    dump("off   ", off);
    dump("mtp   ", mtp);
    dump("ngram ", ng);
    dump("mtp_g ", mtp_gold);

    if (off != want || mtp != want || ng != want) {
        std::fprintf(stderr, "spec generate changed greedy tokens\n");
        return 1;
    }

    auto dev = create_device(DeviceKind::CPU);
    WeightStore store = WeightStore::open(root / "tiny_hybrid", *dev, LoadOptions{});
    Session sess(*dev, store, 64, false, true);
    if (!sess.has_mtp()) {
        std::fprintf(stderr, "fixture missing MTP weights\n");
        return 1;
    }
    sess.prefill(prompt.data(), static_cast<int>(prompt.size()));
    std::vector<int32_t> drafts(8);
    const int nd = sess.mtp_draft(want[0], 3, drafts.data());
    drafts.resize(static_cast<size_t>(nd));
    dump("mtp_c ", drafts);
    if (drafts != mtp_gold) {
        std::fprintf(stderr, "MTP draft != golden\n");
        return 1;
    }

    // n-gram: repeated prompt should propose something
    const int32_t ctx[] = {1, 2, 3, 1, 2, 3};
    int32_t ndr[4]{};
    const int nn = sess.ngram_draft(ctx, 6, 1, 3, ndr);
    std::printf("ngram_draft n=%d first=%d mtp_accept=%d/%d ngram_prop=%d\n", nn, ndr[0], stm.accepted,
                stm.proposed, stn.proposed);

    {
        auto dev2 = create_device(DeviceKind::CPU);
        WeightStore store2 = WeightStore::open(root / "tiny_hybrid", *dev2, LoadOptions{});
        Session target(*dev2, store2, 64, false, true);
        Session draft(*dev2, store2, 64, false, true);
        target.set_draft(&draft);
        GenerateConfig cfg;
        cfg.max_new_tokens = n_new;
        cfg.greedy = true;
        cfg.spec = SpecKind::Draft;
        cfg.spec_n = 3;
        std::vector<int32_t> out(static_cast<size_t>(n_new));
        const int got = target.generate(prompt.data(), static_cast<int>(prompt.size()), out.data(), n_new, cfg);
        out.resize(static_cast<size_t>(got));
        dump("draft ", out);
        if (out != want) {
            std::fprintf(stderr, "draft-model generate changed greedy tokens\n");
            return 1;
        }
        std::printf("draft_model accept=%d/%d\n", target.spec_stats().accepted, target.spec_stats().proposed);
    }

    if (cuda_gen::available()) {
        auto devc = create_device(DeviceKind::CPU);
        WeightStore storec = WeightStore::open(root / "tiny_hybrid", *devc, LoadOptions{});
        Session sc(*devc, storec, 64, false, true, true);
        if (!sc.uses_cuda()) {
            std::fprintf(stderr, "test_spec CUDA session missing engine\n");
            return 1;
        }
        GenerateConfig cc;
        cc.max_new_tokens = n_new;
        cc.greedy = true;
        cc.spec = SpecKind::Mtp;
        cc.spec_n = 3;
        std::vector<int32_t> outc(static_cast<size_t>(n_new));
        const int gc = sc.generate(prompt.data(), static_cast<int>(prompt.size()), outc.data(), n_new, cc);
        outc.resize(static_cast<size_t>(gc));
        dump("cuda_mtp", outc);
        std::printf("cuda_mtp_proposed=%d\n", sc.spec_stats().proposed);
        if (outc != want || sc.spec_stats().proposed <= 0) {
            std::fprintf(stderr, "CUDA MTP generate failed greedy match or did not draft\n");
            return 1;
        }
    }

    std::printf("test_spec ok (greedy invariant + MTP draft match)\n");
    return 0;
}
