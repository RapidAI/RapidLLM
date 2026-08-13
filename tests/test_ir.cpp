#include "rapidllm/ir/model_desc.h"

#include <cstdio>
#include <stdexcept>
#include <string>

static int fails = 0;
#define CHECK(cond)                                                                          \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);             \
            ++fails;                                                                         \
        }                                                                                    \
    } while (0)

int main() {
    using namespace rapidllm;
    ModelDesc m = make_qwen36_27b_desc();
    CHECK(m.vocab == 248320);
    CHECK(m.hidden == 5120);
    CHECK(m.n_layers == 64);
    CHECK(m.layers.size() == 64);
    int nd = 0, na = 0;
    for (int i = 0; i < 64; ++i) {
        if (m.layers[static_cast<size_t>(i)].kind == LayerKind::GatedDeltaNet) ++nd;
        else ++na;
    }
    CHECK(nd == 48);
    CHECK(na == 16);
    CHECK(m.kind_of(0) == LayerKind::GatedDeltaNet);
    CHECK(m.kind_of(3) == LayerKind::GatedAttn);
    CHECK(m.mixer_slot(0) == 0);
    CHECK(m.mixer_slot(1) == 1);
    CHECK(m.mixer_slot(3) == 0);
    CHECK(m.mixer_slot(7) == 1);
    CHECK(m.vision.depth == 27);
    CHECK(m.vision.hidden == 1152);
    CHECK(m.vision.out_hidden == 5120);
    CHECK(m.vision.image_token_id == 248056);

    bool threw = false;
    try {
        (void)layer_kind_from_type("not_a_real_kind");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);

    ModelDesc t = make_tiny_hybrid_desc();
    CHECK(t.n_layers == 8);
    CHECK(t.hidden == 32);
    CHECK(t.layers[3].kind == LayerKind::GatedAttn);

    if (fails) {
        std::fprintf(stderr, "%d checks failed\n", fails);
        return 1;
    }
    std::printf("test_ir ok\n");
    return 0;
}
