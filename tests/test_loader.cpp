#include "rapidllm/frontend/weight_store.h"
#include "rapidllm/backend/device.h"

#include <cstdio>
#include <filesystem>
#include <string>

static int fails = 0;
#define CHECK(cond)                                                                          \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);             \
            ++fails;                                                                         \
        }                                                                                    \
    } while (0)

#ifndef RAPIDLLM_FIXTURE_DIR
#define RAPIDLLM_FIXTURE_DIR "."
#endif

int main() {
    using namespace rapidllm;
    const std::filesystem::path root = RAPIDLLM_FIXTURE_DIR;
    LoadOptions opt;

    auto hf = make_loader(root / "tiny_hybrid");
    TensorTable th = hf->load(root / "tiny_hybrid", opt);
    CHECK(th.source == SourceKind::HfFp8Dir);
    CHECK(th.find("embed") != nullptr);
    CHECK(th.find("lm_head") != nullptr);
    CHECK(th.find("final_norm") != nullptr);
    CHECK(th.find("layers[0].delta.leftover.a_log") != nullptr);
    CHECK(th.find("layers[3].attn.wq") != nullptr);
    CHECK(th.find("layers[0].delta.leftover.conv1d") != nullptr);
    bool saw_visual = false;
    for (const auto& [ir, t] : th.tensors) {
        if (t.src_name.find("visual") != std::string::npos) saw_visual = true;
    }
    CHECK(!saw_visual);
    CHECK(map_gguf_name("blk.64.nextn.eh_proj.weight") == "mtp.fc");
    CHECK(map_gguf_name("blk.64.nextn.enorm.weight") == "mtp.pre_fc_norm_embedding");
    CHECK(map_gguf_name("blk.64.nextn.hnorm.weight") == "mtp.pre_fc_norm_hidden");
    CHECK(map_gguf_name("blk.64.nextn.shared_head_norm.weight") == "mtp.norm");
    CHECK(map_gguf_name("blk.64.nextn.attn_q.weight") == "mtp.layers[0].attn.wq");
    CHECK(map_gguf_name("nextn.fc.weight") == "mtp.fc");
    CHECK(map_gguf_name("blk.0.attn_q.weight") == "layers[0].attn.wq");
    CHECK(map_gguf_name("blk.0.ssm_a") == "layers[0].delta.leftover.a_log");
    CHECK(map_gguf_name("blk.0.ssm_a_log") == "layers[0].delta.leftover.a_log");
    CHECK(map_gguf_name("blk.0.ssm_a.weight") == "layers[0].delta.leftover.in_proj_a");
    CHECK(map_gguf_name("blk.0.ssm_alpha.weight") == "layers[0].delta.leftover.in_proj_a");
    CHECK(map_hf_name("model.visual.patch_embed.proj.weight") == "visual.patch_embed");
    CHECK(map_hf_name("model.visual.blocks.0.attn.qkv.weight") == "visual.blocks[0].attn.qkv");
    CHECK(map_hf_name("model.visual.merger.linear_fc2.bias") == "visual.merger.fc2_bias");
    CHECK(th.find("mtp.fc") != nullptr);
    CHECK(th.model.has_mtp);
    std::printf("HF loaded tensors=%zu layers=%d\n", th.tensors.size(), th.model.n_layers);

    auto gg = make_loader(root / "tiny_hybrid.gguf");
    TensorTable tg = gg->load(root / "tiny_hybrid.gguf", opt);
    CHECK(tg.source == SourceKind::GgufFile);
    CHECK(tg.find("embed") != nullptr);
    CHECK(tg.find("layers[0].delta.leftover.a_log") != nullptr);
    CHECK(tg.find("layers[3].attn.wq") != nullptr);
    saw_visual = false;
    for (const auto& [ir, t] : tg.tensors) {
        if (t.src_name.find("visual") != std::string::npos) saw_visual = true;
    }
    CHECK(!saw_visual);
    CHECK(tg.find("mtp.fc") != nullptr);
    std::printf("GGUF loaded tensors=%zu layers=%d\n", tg.tensors.size(), tg.model.n_layers);

    bool rej = false;
    try {
        make_loader(root / "bad_fp8")->load(root / "bad_fp8", opt);
    } catch (const LoadError& e) {
        rej = std::string(e.what()).find("weight_scale_inv") != std::string::npos ||
              std::string(e.what()).find("FP8") != std::string::npos;
        std::printf("bad_fp8 rejected: %s\n", e.what());
    }
    CHECK(rej);

    rej = false;
    try {
        make_loader(root / "bad_gguf.gguf")->load(root / "bad_gguf.gguf", opt);
    } catch (const LoadError& e) {
        rej = true;
        std::printf("bad_gguf rejected: %s\n", e.what());
    }
    CHECK(rej);

    rej = false;
    try {
        make_loader(root / "bad_leftover.gguf")->load(root / "bad_leftover.gguf", opt);
    } catch (const LoadError& e) {
        rej = std::string(e.what()).find("quant") != std::string::npos ||
              std::string(e.what()).find("leftover") != std::string::npos ||
              std::string(e.what()).find("DeltaNet") != std::string::npos;
        std::printf("bad_leftover rejected: %s\n", e.what());
    }
    CHECK(rej);

    if (fails) {
        std::fprintf(stderr, "%d loader checks failed\n", fails);
        return 1;
    }
    std::printf("test_loader ok\n");
    return 0;
}
