#include "rapidllm/api.h"
#include "rapidllm/kernels/nv_gemv.h"
#include "rapidllm/kernels/ops.h"
#include "rapidllm/runtime/thread_pool.h"
#include "rapidllm/version.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage() {
    std::fprintf(stderr,
                 "rapidllm %s\n"
                 "  rapidllm -m <hf-dir|file.gguf> [--device cpu|cuda] [--ctx 32768] [--prompt TEXT]\n"
                 "           [--max-new N] [--fuse=on|off] [--spec off|ngram|mtp|auto] [--spec-n N]\n"
                 "  rapidllm bench -m <path> [--device cpu|cuda] [--fuse=on|off] [--micro]\n",
                 rapidllm_version_string());
}

int main(int argc, char** argv) {
    std::string model, prompt = "1,2,3", device = "cpu";
    int ctx = 32768;
    int max_new = 8;
    int max_ram_mb = 0;
    bool bench = false;
    bool thinking = true;
    int fuse = 1;
    bool micro = false;
    int spec = 3;
    int spec_n = 3;
    int threads = 0;
    int max_layers = -1;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* n) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", n);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "bench") {
            bench = true;
            thinking = false;
        } else if (a == "-m" || a == "--model")
            model = need("-m");
        else if (a == "--device")
            device = need("--device");
        else if (a == "--ctx")
            ctx = std::atoi(need("--ctx"));
        else if (a == "--prompt")
            prompt = need("--prompt");
        else if (a == "--max-new")
            max_new = std::atoi(need("--max-new"));
        else if (a == "--max-ram-mb")
            max_ram_mb = std::atoi(need("--max-ram-mb"));
        else if (a == "--thinking")
            thinking = true;
        else if (a == "--no-thinking")
            thinking = false;
        else if (a == "--fuse=off" || (a == "--fuse" && i + 1 < argc && std::string(argv[i + 1]) == "off")) {
            if (a == "--fuse") ++i;
            fuse = 0;
        } else if (a == "--fuse=on" || (a == "--fuse" && i + 1 < argc && std::string(argv[i + 1]) == "on")) {
            if (a == "--fuse") ++i;
            fuse = 1;
        } else if (a == "--micro")
            micro = true;
        else if (a == "--spec") {
            const std::string v = need("--spec");
            if (v == "off") spec = 0;
            else if (v == "ngram") spec = 1;
            else if (v == "mtp") spec = 2;
            else spec = 3;
        } else if (a == "--spec-n")
            spec_n = std::atoi(need("--spec-n"));
        else if (a == "--threads")
            threads = std::atoi(need("--threads"));
        else if (a == "--max-layers")
            max_layers = std::atoi(need("--max-layers"));
        else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (a.rfind("-", 0) == 0) {
            std::fprintf(stderr, "unknown flag %s\n", a.c_str());
            return 2;
        } else if (model.empty())
            model = a;
        else
            prompt = a;
    }

    if (bench) thinking = false;

    if (micro) {
        using clock = std::chrono::steady_clock;
        constexpr int M = 2048, N = 2048, REPS = 8;
        std::vector<float> W(static_cast<size_t>(M) * N), x(N), ys(M), yv(M);
        for (int i = 0; i < M * N; ++i) W[i] = 0.001f * static_cast<float>((i * 17) % 100);
        for (int i = 0; i < N; ++i) x[i] = 0.01f * static_cast<float>((i % 13) - 6);
        auto t_sc0 = clock::now();
        for (int r = 0; r < REPS; ++r) rapidllm::ops::gemv_f32_scalar(W.data(), x.data(), ys.data(), M, N);
        auto t_sc1 = clock::now();
        auto t_si0 = clock::now();
        for (int r = 0; r < REPS; ++r) rapidllm::ops::gemv_f32_simd(W.data(), x.data(), yv.data(), M, N);
        auto t_si1 = clock::now();
        const double sc_s = std::chrono::duration<double>(t_sc1 - t_sc0).count() / REPS;
        const double si_s = std::chrono::duration<double>(t_si1 - t_si0).count() / REPS;
        const double bytes = 2.0 * M * N * 4.0;
        if (threads > 0) rapidllm::set_num_threads(threads);
        std::printf("isa=%s fuse_default=%d threads=%d\n", rapidllm::ops::simd_isa_name(), fuse,
                    rapidllm::num_threads());
        std::printf("gemv_scalar  M=%d N=%d  %.3f ms  %.2f GB/s\n", M, N, sc_s * 1e3, bytes / sc_s / 1e9);
        std::printf("gemv_simd    M=%d N=%d  %.3f ms  %.2f GB/s\n", M, N, si_s * 1e3, bytes / si_s / 1e9);

        constexpr int NV = 48, DK = 128, DV = 128;
        std::vector<float> S1(NV * DK * DV), S2(NV * DK * DV), q(NV * DK), k(NV * DK), v(NV * DV);
        std::vector<float> beta(NV, 0.5f), glog(NV, -0.1f), o1(NV * DV), o2(NV * DV);
        for (size_t i = 0; i < S1.size(); ++i) S1[i] = S2[i] = 0.001f * static_cast<float>(i % 9);
        for (size_t i = 0; i < q.size(); ++i) q[i] = k[i] = 0.01f * static_cast<float>((i % 7) - 3);
        for (size_t i = 0; i < v.size(); ++i) v[i] = 0.01f * static_cast<float>((i % 5) - 2);
        auto d0 = clock::now();
        for (int r = 0; r < 16; ++r)
            rapidllm::ops::delta_recurrent_step_scalar(S1.data(), q.data(), k.data(), v.data(), beta.data(),
                                                       glog.data(), o1.data(), NV, DK, DV, 1e-6f);
        auto d1 = clock::now();
        auto d2 = clock::now();
        for (int r = 0; r < 16; ++r)
            rapidllm::ops::delta_recurrent_step_simd(S2.data(), q.data(), k.data(), v.data(), beta.data(),
                                                     glog.data(), o2.data(), NV, DK, DV, 1e-6f);
        auto d3 = clock::now();
        const double ds = std::chrono::duration<double, std::micro>(d1 - d0).count() / 16;
        const double dv = std::chrono::duration<double, std::micro>(d3 - d2).count() / 16;
        std::printf("delta_scalar 48x128x128  %.1f us\n", ds);
        std::printf("delta_simd   48x128x128  %.1f us\n", dv);

        std::vector<signed char> Wi(static_cast<size_t>(M) * N);
        std::vector<float> rscale(M), ynv(M);
        rapidllm::nv::pack_weight_int8(W.data(), Wi.data(), rscale.data(), M, N);
        auto n0 = clock::now();
        for (int r = 0; r < REPS; ++r) rapidllm::nv::gemv_dp4a_ref(Wi.data(), rscale.data(), x.data(), ynv.data(), M, N);
        auto n1 = clock::now();
        const double nv_s = std::chrono::duration<double>(n1 - n0).count() / REPS;
        std::printf("nv_dp4a_ref  %s  %.3f ms  %.2f GB/s\n", rapidllm::nv::nv_kernel_name(), nv_s * 1e3,
                    bytes / nv_s / 1e9);
        const bool cuda_ran = rapidllm::nv::gemv_dp4a_cuda(Wi.data(), rscale.data(), x.data(), ynv.data(), M, N);
        std::printf("nv_dp4a_cuda ran=%d\n", cuda_ran ? 1 : 0);
        if (model.empty()) return 0;
    }

    if (model.empty()) {
        usage();
        return 2;
    }

    RapidConfig cfg{};
    cfg.model_path = model.c_str();
    cfg.device = device.c_str();
    cfg.threads = threads;
    cfg.max_layers = max_layers;
    cfg.ctx = ctx;
    cfg.kv_i8 = 0;
    cfg.fuse = fuse;
    cfg.mmap = 1;
    cfg.hugepage = 0;
    cfg.language_only = 1;
    cfg.repack_int4 = 0;
    cfg.max_ram_bytes = max_ram_mb > 0 ? static_cast<unsigned long long>(max_ram_mb) * 1024ull * 1024ull : 0;

    RapidError err{};
    std::fprintf(stderr, "loading %s ...\n", model.c_str());
    std::fflush(stderr);
    RapidLLM* eng = rapidllm_load(&cfg, &err);
    if (!eng) {
        std::fprintf(stderr, "load failed: %s\n", err.message);
        return err.code == RAPID_ERR_NOMEM ? 3 : 1;
    }
    std::fprintf(stderr, "loaded vocab=%d hidden layers ready\n", rapidllm_vocab(eng));
    std::fflush(stderr);

    RapidSessionConfig sc{};
    sc.enable_thinking = thinking ? 1 : 0;
    sc.preserve_thinking = 0;
    sc.max_new_tokens = max_new;
    sc.spec = spec;
    sc.spec_n = spec_n;
    std::fprintf(stderr, "session ctx=%d fuse=%d\n", ctx, fuse);
    std::fflush(stderr);
    RapidSession* sess = rapidllm_session_new(eng, &sc, &err);
    if (!sess) {
        std::fprintf(stderr, "session: %s\n", err.message);
        rapidllm_free(eng);
        return 1;
    }

    std::vector<int32_t> ids(4096);
    const int n = rapidllm_encode(eng, prompt.c_str(), ids.data(), static_cast<int>(ids.size()), &err);
    if (n <= 0) {
        std::fprintf(stderr, "encode failed\n");
        rapidllm_session_free(sess);
        rapidllm_free(eng);
        return 1;
    }
    std::fprintf(stderr, "generate n_prompt=%d max_new=%d\n", n, max_new);
    std::fflush(stderr);

    RapidSampleParams sp{};
    sp.greedy = 1;
    sp.temperature = 0.f;
    std::vector<int32_t> out(static_cast<size_t>(max_new));

    if (bench) {
        // Same as vLLM bakeoff: generate after load is warmed, then the timed run.
        const int w = rapidllm_generate(sess, ids.data(), n, &sp, out.data(), max_new, &err);
        if (w < 0) {
            std::fprintf(stderr, "warmup generate failed: %s\n", err.message);
            rapidllm_session_free(sess);
            rapidllm_free(eng);
            return 1;
        }
        std::fprintf(stderr, "warmup n_new=%d\n", w);
        std::fflush(stderr);
    }

    const auto t0 = std::chrono::steady_clock::now();
    const int got = rapidllm_generate(sess, ids.data(), n, &sp, out.data(), max_new, &err);
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();

    if (got < 0) {
        std::fprintf(stderr, "generate failed: %s\n", err.message);
        rapidllm_session_free(sess);
        rapidllm_free(eng);
        return 1;
    }

    char decoded[4096];
    rapidllm_decode_ids(eng, out.data(), got, decoded, 4096, &err);
    std::printf("tokens:");
    for (int i = 0; i < got; ++i) std::printf(" %d", out[i]);
    std::printf("\ntext: %s\n", decoded);

    if (bench) {
        const double tps = sec > 0 ? got / sec : 0;
        double prefill_s = 0, decode_s = 0;
        int n_dec = 0;
        rapidllm_bench_stats(sess, &prefill_s, &decode_s, &n_dec);
        const double decode_tps = (decode_s > 0 && n_dec > 0) ? n_dec / decode_s : 0;
        std::printf("thinking=off\n");
        std::printf("fuse=%s isa=%s device=%s\n", fuse ? "on" : "off", rapidllm::ops::simd_isa_name(),
                    rapidllm_uses_cuda(sess) ? "cuda" : "cpu");
        std::printf("tok/s=%.4f decode_tok/s=%.4f prefill_s=%.4f decode_s=%.4f n_new=%d sec=%.4f\n", tps,
                    decode_tps, prefill_s, decode_s, got, sec);
    }

    RapidSpecStats ss{};
    rapidllm_spec_stats(sess, &ss);
    std::printf("spec=%d spec_n=%d proposed=%d accepted=%d steps=%d\n", spec, spec_n, ss.proposed,
                ss.accepted, ss.steps);

    rapidllm_session_free(sess);
    rapidllm_free(eng);
    return 0;
}
