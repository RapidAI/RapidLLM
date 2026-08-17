#include "rapidllm/api.h"
#include "rapidllm/kernels/nv_gemv.h"
#include "rapidllm/kernels/ops.h"
#include "rapidllm/runtime/thread_pool.h"
#include "rapidllm/runtime/tokenizer.h"
#include "rapidllm/server/http_serve.h"
#include "rapidllm/version.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef _WIN32
#include <stdlib.h>
#endif
#include <string>
#include <vector>

static void usage() {
    std::fprintf(stderr,
                 "rapidllm %s\n"
                 "  rapidllm -m <hf-dir|file.gguf> [--device cpu|cuda|vulkan] [--ctx 32768] [--prompt TEXT]\n"
                 "           [--max-new N] [--fuse=on|off] [--spec off|ngram|mtp|auto] [--spec-n N]\n"
                 "           [--dump-mtp-draft [N]]\n"
                 "           [--image PATH] [--vision] [--batch N] [--prompt-n N]\n"
                 "           [--kv-type f16|q8k_tq3v]\n"
                 "           --image runs the ViT and splices visual tokens into generate\n"
                 "  rapidllm bench -m <path> [--device cpu|cuda|vulkan] [--fuse=on|off] [--micro]\n"
                 "  rapidllm serve -m <path> [--host 127.0.0.1] [--port 8080] [--device cpu|cuda|vulkan]\n"
                 "           OpenAI: POST /v1/chat/completions  POST /v1/responses\n"
                 "           Anthropic: POST /v1/messages\n",
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
    int language_only = 1;
    std::string image_path;
    int batch_n = 1;
    int prompt_n = 0;
    bool mixed = false;
    int dump_mtp = 0;
    bool serve = false;
    std::string host = "127.0.0.1";
    int port = 8080;

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
        } else if (a == "serve") {
            serve = true;
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
            else if (v == "auto") spec = 3;
            else if (v == "draft") {
                std::fprintf(stderr, "--spec draft removed: Qwen3.6/3.8 MTP replaces the external draft model. "
                                     "Use --spec mtp or --spec auto.\n");
                return 2;
            } else {
                std::fprintf(stderr, "unknown --spec %s (off|ngram|mtp|auto)\n", v.c_str());
                return 2;
            }
        } else if (a == "--draft") {
            std::fprintf(stderr, "--draft removed: Qwen3.6/3.8 MTP replaces the external draft model. "
                                 "Use --spec mtp or --spec auto.\n");
            return 2;
        } else if (a == "--spec-n")
            spec_n = std::atoi(need("--spec-n"));
        else if (a == "--dump-mtp-draft") {
            dump_mtp = 3;
            if (i + 1 < argc && argv[i + 1][0] != '-') dump_mtp = std::atoi(need("--dump-mtp-draft"));
            if (dump_mtp <= 0) dump_mtp = 3;
        }
        else if (a == "--threads")
            threads = std::atoi(need("--threads"));
        else if (a == "--max-layers")
            max_layers = std::atoi(need("--max-layers"));
        else if (a == "--image") {
            image_path = need("--image");
            language_only = 0;
        } else if (a == "--vision")
            language_only = 0;
        else if (a == "--batch")
            batch_n = std::atoi(need("--batch"));
        else if (a == "--mixed")
            mixed = true;
        else if (a == "--prompt-n")
            prompt_n = std::atoi(need("--prompt-n"));
        else if (a == "--kv-type") {
            const std::string v = need("--kv-type");
            if (v == "f16" || v == "fp16") {
#ifdef _WIN32
                _putenv_s("RAPIDLLM_KV_TQ", "0");
#else
                setenv("RAPIDLLM_KV_TQ", "0", 1);
#endif
            } else if (v == "q8k_tq3v" || v == "tq" || v == "turboquant") {
#ifdef _WIN32
                _putenv_s("RAPIDLLM_KV_TQ", "1");
#else
                setenv("RAPIDLLM_KV_TQ", "1", 1);
#endif
            } else {
                std::fprintf(stderr, "unknown --kv-type %s (f16|q8k_tq3v)\n", v.c_str());
                return 2;
            }
        }
        else if (a == "--host")
            host = need("--host");
        else if (a == "--port")
            port = std::atoi(need("--port"));
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
    cfg.language_only = language_only;
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

    if (batch_n > 1) {
        const std::string bv = std::to_string(batch_n);
#if defined(_WIN32)
        _putenv_s("RAPIDLLM_MAX_BATCH", bv.c_str());
#else
        setenv("RAPIDLLM_MAX_BATCH", bv.c_str(), 1);
#endif
    }

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

    if (serve) {
        const int rc = rapidllm::serve::serve_listen(host.c_str(), port, eng, sess, model);
        rapidllm_session_free(sess);
        rapidllm_free(eng);
        return rc;
    }

    std::vector<int32_t> ids;
    int n = 0;
    int n_vis = 0;
    if (!image_path.empty()) {
        if (rapidllm_session_load_image(sess, image_path.c_str(), &n_vis, &err) != RAPID_OK) {
            std::fprintf(stderr, "image: %s\n", err.message);
            rapidllm_session_free(sess);
            rapidllm_free(eng);
            return 1;
        }
        std::fprintf(stderr, "image %s vis_tokens=%d\n", image_path.c_str(), n_vis);
        std::fflush(stderr);
    }

    if (prompt_n > 0) {
        // Non-repeating ids so n-gram spec cannot fake long-ctx throughput.
        std::vector<int32_t> text(static_cast<size_t>(prompt_n));
        const int vocab = rapidllm_vocab(eng);
        const int span = vocab > 20000 ? 10000 : (vocab > 256 ? vocab - 256 : 1);
        for (int i = 0; i < prompt_n; ++i) text[static_cast<size_t>(i)] = 256 + (i % span);
        if (n_vis > 0) {
            ids.assign(static_cast<size_t>(prompt_n + n_vis + 8), 0);
            n = rapidllm::pack_vl_prompt(n_vis, text.data(), prompt_n, ids.data(),
                                         static_cast<int>(ids.size()));
        } else {
            ids = std::move(text);
            n = prompt_n;
        }
    } else {
        ids.assign(static_cast<size_t>(ctx > 4096 ? ctx : 4096), 0);
        if (n_vis > 0)
            n = rapidllm_encode_vl(eng, sess, prompt.c_str(), ids.data(), static_cast<int>(ids.size()), &err);
        else
            n = rapidllm_encode(eng, prompt.c_str(), ids.data(), static_cast<int>(ids.size()), &err);
    }
    if (n <= 0) {
        std::fprintf(stderr, "encode failed\n");
        rapidllm_session_free(sess);
        rapidllm_free(eng);
        return 1;
    }
    std::fprintf(stderr, "generate n_prompt=%d max_new=%d\n", n, max_new);
    std::fflush(stderr);

    if (dump_mtp > 0) {
        if (rapidllm_prefill(sess, ids.data(), n, &err) != RAPID_OK) {
            std::fprintf(stderr, "prefill: %s\n", err.message);
            rapidllm_session_free(sess);
            rapidllm_free(eng);
            return 1;
        }
        RapidSampleParams sp{};
        sp.greedy = 1;
        int32_t first = 0;
        if (rapidllm_sample(sess, &sp, &first, &err) != RAPID_OK) {
            std::fprintf(stderr, "sample: %s\n", err.message);
            rapidllm_session_free(sess);
            rapidllm_free(eng);
            return 1;
        }
        std::vector<int32_t> a(static_cast<size_t>(dump_mtp), 0), b(static_cast<size_t>(dump_mtp), 0);
        const int na = rapidllm_mtp_draft(sess, first, dump_mtp, a.data(), &err);
        const int nb = rapidllm_mtp_draft(sess, first, dump_mtp, b.data(), &err);
        std::printf("mtp_first=%d has_mtp=%d cuda=%d\n", first, rapidllm_has_mtp(sess),
                    rapidllm_uses_cuda(sess));
        std::printf("mtp_draft1:");
        for (int i = 0; i < na; ++i) std::printf(" %d", a[static_cast<size_t>(i)]);
        std::printf("\nmtp_draft2:");
        for (int i = 0; i < nb; ++i) std::printf(" %d", b[static_cast<size_t>(i)]);
        std::printf("\n");
        int same = (na == nb && na > 0);
        int nonzero = 0;
        for (int i = 0; i < na && same; ++i) {
            if (a[static_cast<size_t>(i)] != b[static_cast<size_t>(i)]) same = 0;
            if (a[static_cast<size_t>(i)] != 0) nonzero = 1;
        }
        std::printf("mtp_draft_match=%d mtp_draft_nonzero=%d n=%d\n", same, nonzero, na);
        rapidllm_session_free(sess);
        rapidllm_free(eng);
        return (same && nonzero) ? 0 : 1;
    }

    RapidSampleParams sp{};
    sp.greedy = 1;
    sp.temperature = 0.f;
    std::vector<int32_t> out(static_cast<size_t>(max_new));

    std::vector<int32_t> packed;
    std::vector<int> packed_lens;
    if (mixed && batch_n > 1) {
        packed_lens.assign(static_cast<size_t>(batch_n), n);
        packed.resize(static_cast<size_t>(batch_n) * static_cast<size_t>(n));
        const int vocab = rapidllm_vocab(eng);
        const int span = vocab > 20000 ? 10000 : (vocab > 256 ? vocab - 256 : 1);
        for (int b = 0; b < batch_n; ++b) {
            for (int i = 0; i < n; ++i) {
                int32_t tok = ids[static_cast<size_t>(i)];
                if (prompt_n > 0)
                    tok = 256 + ((i + b * 9973) % span);
                else
                    tok = 256 + ((tok + b * 17 + i) % span);
                packed[static_cast<size_t>(b) * n + i] = tok;
            }
        }
        std::fprintf(stderr, "mixed_batch n_seq=%d n_prompt=%d\n", batch_n, n);
    }

    if (bench) {
        // Same as vLLM bakeoff: generate after load is warmed, then the timed run.
        int w = 0;
        if (batch_n > 1) {
            std::vector<int32_t> wout(static_cast<size_t>(batch_n) * max_new, 0);
            std::vector<int> wn(static_cast<size_t>(batch_n), 0);
            if (mixed)
                w = rapidllm_generate_batch_var(sess, packed.data(), packed_lens.data(), batch_n, &sp, wout.data(),
                                                max_new, wn.data(), &err);
            else
                w = rapidllm_generate_batch(sess, ids.data(), n, batch_n, &sp, wout.data(), max_new, wn.data(),
                                            &err);
        } else {
            w = rapidllm_generate(sess, ids.data(), n, &sp, out.data(), max_new, &err);
        }
        if (w < 0) {
            std::fprintf(stderr, "warmup generate failed: %s\n", err.message);
            rapidllm_session_free(sess);
            rapidllm_free(eng);
            return 1;
        }
        std::fprintf(stderr, "warmup n_new=%d batch=%d\n", w, batch_n);
        std::fflush(stderr);
    }

    const auto t0 = std::chrono::steady_clock::now();
    int got = 0;
    std::vector<int> out_n;
    if (batch_n > 1) {
        out.assign(static_cast<size_t>(batch_n) * max_new, 0);
        out_n.assign(static_cast<size_t>(batch_n), 0);
        if (mixed)
            got = rapidllm_generate_batch_var(sess, packed.data(), packed_lens.data(), batch_n, &sp, out.data(),
                                              max_new, out_n.data(), &err);
        else
            got = rapidllm_generate_batch(sess, ids.data(), n, batch_n, &sp, out.data(), max_new, out_n.data(),
                                          &err);
    } else {
        got = rapidllm_generate(sess, ids.data(), n, &sp, out.data(), max_new, &err);
    }
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
    if (batch_n > 1) {
        int zseq = 0;
        for (int b = 0; b < batch_n; ++b) {
            const int nn = out_n.empty() ? 0 : out_n[static_cast<size_t>(b)];
            bool allz = nn > 0;
            for (int i = 0; i < nn; ++i) {
                if (out[static_cast<size_t>(b) * max_new + i] != 0) {
                    allz = false;
                    break;
                }
            }
            if (allz) ++zseq;
        }
        std::fprintf(stderr, "batch_zero_seqs=%d/%d\n", zseq, batch_n);
    }

    if (bench) {
        const double tps = sec > 0 ? got / sec : 0;
        double prefill_s = 0, decode_s = 0;
        int n_dec = 0;
        rapidllm_bench_stats(sess, &prefill_s, &decode_s, &n_dec);
        const double decode_tps = (decode_s > 0 && n_dec > 0) ? n_dec / decode_s : 0;
        std::printf("thinking=off\n");
        std::printf("fuse=%s isa=%s device=%s\n", fuse ? "on" : "off", rapidllm::ops::simd_isa_name(),
                    rapidllm_device_name(sess));
        std::printf("tok/s=%.4f decode_tok/s=%.4f prefill_s=%.4f decode_s=%.4f n_new=%d batch=%d sec=%.4f\n", tps,
                    decode_tps, prefill_s, decode_s, got, batch_n, sec);
        // Same meters as llama-cli `[ Prompt: | Generation: ]`, plus Wall.
        // tok/s= above stays n_new/(prefill+decode). Prompt uses n_prompt/prefill_s.
        const double prompt_tps = (prefill_s > 0 && n > 0) ? static_cast<double>(n) / prefill_s : 0;
        std::printf("[ Prompt: %.1f t/s | Generation: %.1f t/s | Wall: %.4f t/s ]\n", prompt_tps, decode_tps,
                    tps);
    }

    RapidSpecStats ss{};
    rapidllm_spec_stats(sess, &ss);
    std::printf("spec=%d spec_n=%d proposed=%d accepted=%d steps=%d\n", spec, spec_n, ss.proposed,
                ss.accepted, ss.steps);

    rapidllm_session_free(sess);
    rapidllm_free(eng);
    return 0;
}
