#include "rapidllm/backend/device.h"
#include "rapidllm/frontend/weight_store.h"
#include "rapidllm/runtime/session.h"

#include "frontend/json_mini.h"

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
    const std::filesystem::path root = RAPIDLLM_FIXTURE_DIR;
    const Json g = parse_json(slurp(root / "tiny_hybrid" / "golden.json"));
    std::vector<int32_t> prompt, want;
    for (const Json& x : g.at("prompt_ids").as_arr()) prompt.push_back(x.as_int());
    for (const Json& x : g.at("greedy_ids").as_arr()) want.push_back(x.as_int());
    const int n_new = g.at("n_new").as_int();

    auto run = [&](const std::filesystem::path& model, bool fuse) -> std::vector<int32_t> {
        auto dev = create_device(DeviceKind::CPU);
        WeightStore store = WeightStore::open(model, *dev, LoadOptions{});
        Session sess(*dev, store, 64, false, fuse);
        GenerateConfig cfg;
        cfg.max_new_tokens = n_new;
        cfg.greedy = true;
        cfg.fuse = fuse;
        std::vector<int32_t> out(static_cast<size_t>(n_new));
        const int got = sess.generate(prompt.data(), static_cast<int>(prompt.size()), out.data(), n_new, cfg);
        out.resize(static_cast<size_t>(got));
        return out;
    };

    const auto hf_off = run(root / "tiny_hybrid", false);
    const auto hf_on = run(root / "tiny_hybrid", true);
    const auto gg_off = run(root / "tiny_hybrid.gguf", false);
    const auto gg_on = run(root / "tiny_hybrid.gguf", true);

    auto dump = [](const char* tag, const std::vector<int32_t>& v) {
        std::printf("%s:", tag);
        for (int t : v) std::printf(" %d", t);
        std::printf("\n");
    };
    dump("golden     ", want);
    dump("hf fuse=off", hf_off);
    dump("hf fuse=on ", hf_on);
    dump("gg fuse=off", gg_off);
    dump("gg fuse=on ", gg_on);

    if (hf_off != want || hf_on != want || gg_off != want || gg_on != want) {
        std::fprintf(stderr, "greedy != golden (fuse on/off)\n");
        return 1;
    }

    auto run_cuda = [&]() -> std::vector<int32_t> {
        auto dev = create_device(DeviceKind::CPU);
        WeightStore store = WeightStore::open(root / "tiny_hybrid", *dev, LoadOptions{});
        Session sess(*dev, store, 64, false, true, true);
        GenerateConfig cfg;
        cfg.max_new_tokens = n_new;
        cfg.greedy = true;
        cfg.fuse = true;
        cfg.spec = SpecKind::Off;
        std::vector<int32_t> out(static_cast<size_t>(n_new));
        const int got = sess.generate(prompt.data(), static_cast<int>(prompt.size()), out.data(), n_new, cfg);
        out.resize(static_cast<size_t>(got));
        std::printf("cuda_session uses_cuda=%d\n", sess.uses_cuda() ? 1 : 0);
        return out;
    };
    const auto cu = run_cuda();
    dump("cuda/auto  ", cu);
    if (cu != want) {
        std::fprintf(stderr, "cuda/auto greedy != golden\n");
        return 1;
    }
    std::printf("test_hybrid ok fuse=on and fuse=off match golden\n");
    return 0;
}
