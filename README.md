# RapidLLM

[English](README.md) · [中文](README.zh-CN.md)

C++20 local inference engine for **Qwen3.6-27B** hybrid text models.

This is not a wrapper around llama.cpp, ggml, vLLM, or MLC.

| | |
| --- | --- |
| Engine license | **MIT** (`LICENSE`) |
| Model weights | Apache-2.0 (Qwen) |
| Language | C++20 (Python only for goldens / fixtures) |
| Platforms | Windows x86-64 · Linux x86-64 |
| Version | 0.1.0 |

Design: [docs/architecture.md](docs/architecture.md) · [docs/设计方案.md](docs/设计方案.md)

## What it is

RapidLLM loads official HuggingFace **block-FP8** directories and community **GGUF** files into one `ModelDesc` + `TensorTable`. The scheduler walks `layer_types[]` and dispatches:

- **48 × Gated DeltaNet** (`linear_attention`) — O(1) FP32 recurrent state
- **16 × Gated Attention** (`full_attention`) — GQA KV cache

The default path is **text-only**. Vision tensors are skipped unless `--vision` or `--image` is set. MTP is parsed and can draft tokens.

## Why a hybrid engine

Short decode is still a weight-bandwidth problem (22–30 GB of linears). Long context is where the architecture wins:

| State | Size | Grows with sequence? |
| --- | --- | --- |
| Gated Attention KV (16 layers, FP16) | 64 KiB / token | yes |
| DeltaNet recurrent `S` (48 layers, FP32) | ~144 MiB | no |
| conv1d window | ~7.5 MiB | no |

At 32K context, KV is about 2 GiB FP16 (1 GiB INT8). The 48 linear-attention layers do not grow a KV cache.

## Features

- First-class **HF FP8** loader (`config.json` + safetensors / index)
- First-class **GGUF** loader (Q4_K / Q5_K / Q6_K / Q8_0; unknown IQ dequantized once)
- CPU decode by default: **AVX2** required, **AVX-512** dispatched at runtime
- Optional **CUDA** (`-DRAPIDLLM_WITH_CUDA=ON`) with host-ref kernels when no device is present
- Dual cache: attention KV + DeltaNet recurrent + causal conv state
- Memory planner: refuse or shrink context instead of letting the OS OOM-kill the process
- Fused decode path (`--fuse=on|off`) for A/B against unfused ops
- Speculative decode: `off` / `ngram` / `mtp` / `auto` / `draft` (smaller same-vocab model via `--draft`)
- Continuous batch: `--batch N` (same prompt, one shared weight pass per step; CUDA via `RAPIDLLM_MAX_BATCH`)
- HTTP serve: OpenAI `/v1/chat/completions` + `/v1/responses`, Anthropic `/v1/messages`
- Qwen3.6 ViT encoder (`vision_encode`); `--vision` / `--image` keep `visual.*` weights
- Thinking on by default; `bench` and `serve` force `enable_thinking=false`
- Versioned **C API** (`include/rapidllm/api.h`) + CLI

## Requirements

- CMake ≥ 3.20
- C++20 compiler (MSVC, Clang-cl, or GCC/Clang)
- Python 3 (fixture / golden generation only)
- x86-64 with AVX2
- Optional: CUDA Toolkit (sm_75 / sm_86 / sm_89)

**RAM guide**

| Machine RAM | Recommended weights | Default ctx |
| --- | --- | --- |
| ≤ 16 GB | tiny fixture only | — |
| 32 GB | GGUF Q4_K (~16.8 GB) | 32K, INT8 KV |
| 64 GB | official FP8 or Q4_K / Q5_K | 32K |
| ≥ 96 GB | FP8; 262K is the capability cap | 32K–128K |

Factory default `ctx = 32768`. Official FP8 text weights are ~27–28 GiB resident — a 32 GB box must use GGUF Q4_K or INT4 repack. Loading the vision tower adds another ~1–2 GiB.

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

With NVIDIA kernels:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAPIDLLM_WITH_CUDA=ON
cmake --build build
```

On Windows without Ninja, omit `-G Ninja` and use the Visual Studio generator. `rapidllm serve` links `ws2_32` on Windows.

## CLI

```text
rapidllm -m <hf-dir|file.gguf> [--device cpu|cuda] [--ctx 32768] [--prompt TEXT]
         [--max-new N] [--threads N] [--max-layers N] [--max-ram-mb N]
         [--thinking | --no-thinking] [--fuse=on|off]
         [--spec off|ngram|mtp|auto|draft] [--spec-n N] [--draft <hf-dir|file.gguf>]
         [--image PATH] [--vision] [--batch N]

rapidllm bench -m <path> [--device cpu|cuda] [--fuse=on|off] [--micro] [--batch N]

rapidllm serve -m <path> [--host 127.0.0.1] [--port 8080] [--device cpu|cuda]
```

Examples:

```bash
rapidllm -m /path/to/Qwen3.6-27B-FP8 --device cpu --ctx 32768 --prompt "Hello"
rapidllm -m model.gguf --device cpu --prompt "Hello"
rapidllm -m target.gguf --draft draft.gguf --spec draft --spec-n 3 --prompt "Hello"
rapidllm bench -m model.gguf
rapidllm bench -m model.gguf --device cuda --batch 4
rapidllm bench --micro
rapidllm serve -m /path/to/Qwen3.6-27B-FP8 --host 127.0.0.1 --port 8080 --device cuda
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--device` | `cpu` | `cuda` requires a CUDA build |
| `--ctx` | `32768` | planner may shrink or refuse |
| `--max-new` | `8` | generated tokens |
| `--fuse` | `on` | fused DeltaNet / Attn / MLP decode |
| `--spec` | `auto` | `draft` needs `--draft` |
| `--draft` | — | smaller same-vocab draft model; implies `--spec draft` |
| `--batch` | `1` | copies of the same prompt; sets `RAPIDLLM_MAX_BATCH` |
| `--vision` / `--image` | off | load `visual.*` (encoder is CPU; generate does not yet consume the image) |
| `--thinking` | on | `bench` and `serve` always turn it off |

Weight sources:

- Official: [`Qwen/Qwen3.6-27B-FP8`](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)
- Community GGUF (32 GB path): [`unsloth/Qwen3.6-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) Q4_K_M

## HTTP serve

`rapidllm serve` is a single-connection JSON server (no SSE yet). Thinking is off.

| Method | Path | Protocol |
| --- | --- | --- |
| `GET` | `/health`, `/v1/health` | liveness |
| `GET` | `/v1/models` | model list |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions |
| `POST` | `/v1/responses` | OpenAI Responses |
| `POST` | `/v1/messages` | Anthropic Messages |

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

`max_tokens` is clamped to 4096. `temperature <= 0` is greedy.

## C API

```c
#include "rapidllm/api.h"

RapidConfig cfg = {0};
cfg.model_path = "model.gguf";
cfg.device = "cpu";
cfg.ctx = 32768;
cfg.fuse = 1;
cfg.language_only = 1;

RapidError err = {0};
RapidLLM* eng = rapidllm_load(&cfg, &err);
RapidSessionConfig sc = {0};
sc.enable_thinking = 1;
sc.max_new_tokens = 64;
sc.spec = 3; /* auto; 4 = draft-model */
RapidSession* sess = rapidllm_session_new(eng, &sc, &err);

int32_t ids[256], out[64];
int n = rapidllm_encode(eng, "Hello", ids, 256, &err);
RapidSampleParams sp = {0};
sp.greedy = 1;
int got = rapidllm_generate(sess, ids, n, &sp, out, 64, &err);

/* Same prompt, N sequences, one shared weight pass per step (CUDA). */
int out_n[4];
int32_t bout[4 * 64];
rapidllm_generate_batch(sess, ids, n, 4, &sp, bout, 64, out_n, &err);

rapidllm_session_free(sess);
rapidllm_free(eng);
```

Attach a draft model with `rapidllm_session_set_draft`. `RAPIDLLM_API_VERSION` is `1`. See `include/rapidllm/api.h`.

## Tests

CMake generates a synthetic hybrid fixture (`python/goldens/tiny_hybrid.py`) and wires CTest:

```bash
ctest --test-dir build --output-on-failure
```

| Test | What it checks |
| --- | --- |
| `test_ir` | `ModelDesc`, `layer_types[]`, `VisionDesc` |
| `test_loader` | HF FP8 + GGUF + reject-bad fixtures + visual name map |
| `test_hybrid` | greedy tokens vs golden, fuse on/off, both formats |
| `test_simd_bench` | AVX2 / AVX-512 vs scalar |
| `test_nv` | DP4A GEMV reference |
| `test_spec` | n-gram / MTP / draft-model speculative decode |
| `test_batch` | continuous batch generate |
| `test_protocol` | OpenAI / Anthropic parse + render + `/health` |
| `test_cuda_decode` | CUDA path or host-ref fallback |

## Layout

```text
include/rapidllm/   public headers (C API, IR, runtime, kernels, server)
src/api/            extern "C" implementation
src/cli/            rapidllm, rapidllm_version
src/server/         HTTP serve (OpenAI + Anthropic)
src/frontend/       HF safetensors, GGUF, name map, WeightStore
src/ir/             ModelDesc + VisionDesc
src/kernels/        scalar / AVX2 / AVX-512 / fused / vision / CUDA
src/runtime/        session, DualCache, planner, tokenizer, sampler
src/backend/cpu/    CpuDevice
shaders/            GLSL compute (not wired into the default build)
python/goldens/     tiny hybrid fixture + golden tokens
tests/              unit / golden / loader / spec / batch / protocol
docs/               architecture (EN/ZH)
```

## Non-goals (v1)

- No SSE / token streaming on `serve` (JSON request → JSON response)
- No image-conditioned generate in the CLI (encoder exists; tokens are not inserted yet)
- No video encoder
- No MoE expert path
- No ARM / Apple Silicon
- No linking or vendoring llama.cpp / ggml / FLA

## License

Engine: [MIT](LICENSE)  
Model weights: Apache-2.0 (Qwen)
