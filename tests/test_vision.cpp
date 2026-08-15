#include "rapidllm/backend/device.h"
#include "rapidllm/frontend/weight_store.h"
#include "rapidllm/kernels/ops.h"
#include "rapidllm/runtime/image_io.h"
#include "rapidllm/runtime/session.h"
#include "rapidllm/runtime/tokenizer.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef RAPIDLLM_FIXTURE_DIR
#define RAPIDLLM_FIXTURE_DIR "."
#endif

static int fails = 0;
#define CHECK(cond)                                                                          \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);             \
            ++fails;                                                                         \
        }                                                                                    \
    } while (0)

static std::filesystem::path write_ppm(const std::filesystem::path& dir) {
    const int w = 64, h = 64;
    const auto p = dir / "vl_test.ppm";
    std::ofstream out(p, std::ios::binary);
    out << "P6\n" << w << " " << h << "\n255\n";
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const unsigned char r = static_cast<unsigned char>(x * 4);
            const unsigned char g = static_cast<unsigned char>(y * 4);
            const unsigned char b = 128;
            out.put(static_cast<char>(r));
            out.put(static_cast<char>(g));
            out.put(static_cast<char>(b));
        }
    }
    return p;
}

int main() {
    using namespace rapidllm;
    int gh = 0, gw = 0;
    CHECK(ops::vision_grid(64, 64, 16, 2, &gh, &gw) == 4);
    CHECK(gh == 2 && gw == 2);
    CHECK(ops::vision_grid(65, 64, 16, 2, &gh, &gw) < 0);

    int oh = 0, ow = 0;
    CHECK(ops::vision_smart_resize(100, 80, 32, 65536, 1024 * 32 * 32, &oh, &ow) == 0);
    CHECK(oh % 32 == 0 && ow % 32 == 0);
    CHECK(oh * ow >= 65536);

    std::vector<float> src(2 * 2 * 3, 0.f);
    src[0] = 1.f;
    src[5] = 1.f; // (0,1) G
    std::vector<float> dst(4 * 4 * 3, 0.f);
    ops::vision_resize_bilinear(src.data(), 2, 2, dst.data(), 4, 4);
    CHECK(dst[0] > 0.9f);

    int32_t text[] = {1, 2, 3};
    int32_t ids[16];
    const int n = pack_vl_prompt(4, text, 3, ids, 16, 10, 11, 12);
    CHECK(n == 9);
    CHECK(ids[0] == 10 && ids[1] == 11 && ids[5] == 12 && ids[6] == 1 && ids[8] == 3);

    const auto tmp = std::filesystem::temp_directory_path();
    const auto ppm = write_ppm(tmp);
    ImageRgb im = load_image_rgb(ppm.string());
    CHECK(im.h == 64 && im.w == 64);
    CHECK(im.rgb.size() == 64ull * 64 * 3);
    CHECK(im.rgb[0] < 0.02f);

    const std::filesystem::path root = RAPIDLLM_FIXTURE_DIR;
    auto dev = create_device(DeviceKind::CPU);
    WeightStore store = WeightStore::open(root / "tiny_hybrid", *dev, LoadOptions{});
    Session a(*dev, store, 64, false, true);
    Session b(*dev, store, 64, false, true);
    const int32_t prompt[] = {1, 2, 2, 3};
    GenerateConfig cfg;
    cfg.max_new_tokens = 4;
    cfg.greedy = true;
    cfg.spec = SpecKind::Off;
    int32_t out_a[4] = {}, out_b[4] = {};
    CHECK(a.generate(prompt, 4, out_a, 4, cfg) == 4);
    const int H = store.model().hidden;
    std::vector<float> vis(static_cast<size_t>(2) * H, 0.35f);
    for (int i = 0; i < H; ++i) vis[static_cast<size_t>(H) + i] = -0.35f;
    b.set_vision_embeds(vis.data(), 2, /*placeholder*/ 2);
    CHECK(b.generate(prompt, 4, out_b, 4, cfg) == 4);
    bool differ = false;
    for (int i = 0; i < 4; ++i)
        if (out_a[i] != out_b[i]) differ = true;
    CHECK(differ);
    std::printf("plain:");
    for (int t : out_a) std::printf(" %d", t);
    std::printf("\nvision:");
    for (int t : out_b) std::printf(" %d", t);
    std::printf("\n");

    if (fails) {
        std::fprintf(stderr, "%d checks failed\n", fails);
        return 1;
    }
    std::printf("test_vision ok\n");
    return 0;
}
