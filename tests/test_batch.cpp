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
    return 0;
}
