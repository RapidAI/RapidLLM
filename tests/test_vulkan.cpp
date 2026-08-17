#include "rapidllm/api.h"
#include "rapidllm/backend/device.h"
#include "rapidllm/backend/vulkan_device.h"
#include "rapidllm/frontend/weight_store.h"
#include "rapidllm/runtime/session.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifndef RAPIDLLM_FIXTURE_DIR
#define RAPIDLLM_FIXTURE_DIR "fixtures"
#endif

int main() {
    const std::string path = std::string(RAPIDLLM_FIXTURE_DIR) + "/tiny_hybrid";
    if (!rapidllm::vulkan_available()) {
        std::printf("vulkan_available=0 err=%s\n", rapidllm::vulkan_last_error());
        // Still exercise the shipped CLI contract: load must name the backend.
        RapidConfig cfg{};
        cfg.model_path = path.c_str();
        cfg.device = "vulkan";
        cfg.ctx = 64;
        cfg.fuse = 1;
        RapidError err{};
        RapidLLM* eng = rapidllm_load(&cfg, &err);
        if (eng) {
            std::printf("unexpected vulkan load without ICD\n");
            rapidllm_free(eng);
            return 1;
        }
        std::printf("rapidllm_load vulkan: %s\n", err.message);
        return 0;
    }
    std::printf("vulkan_available=1\n");
    RapidConfig cfg{};
    cfg.model_path = path.c_str();
    cfg.device = "vulkan";
    cfg.ctx = 64;
    cfg.fuse = 1;
    RapidError err{};
    RapidLLM* eng = rapidllm_load(&cfg, &err);
    if (!eng) {
        std::printf("load failed: %s\n", err.message);
        return 1;
    }
    RapidSessionConfig sc{};
    sc.max_new_tokens = 6;
    sc.spec = 0;
    RapidSession* sess = rapidllm_session_new(eng, &sc, &err);
    if (!sess) {
        std::printf("session: %s\n", err.message);
        rapidllm_free(eng);
        return 1;
    }
    const int32_t prompt[3] = {1, 2, 3};
    int32_t out[8] = {};
    RapidSampleParams sp{};
    sp.greedy = 1;
    const int n = rapidllm_generate(sess, prompt, 3, &sp, out, 6, &err);
    if (n <= 0) {
        std::printf("generate failed: %s\n", err.message);
        rapidllm_session_free(sess);
        rapidllm_free(eng);
        return 1;
    }
    int nz = 0;
    std::printf("vulkan tokens:");
    for (int i = 0; i < n; ++i) {
        std::printf(" %d", out[i]);
        if (out[i] != 0) ++nz;
    }
    std::printf("\n");
    std::printf("device=%s n=%d nonzero=%d\n", rapidllm_device_name(sess), n, nz);
    rapidllm_session_free(sess);
    rapidllm_free(eng);
    if (nz == 0) {
        std::printf("FAIL all-zero vulkan tokens\n");
        return 1;
    }
    std::printf("test_vulkan ok\n");
    return 0;
}
