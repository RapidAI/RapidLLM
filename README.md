# RapidLLM

[English](README.md) · [中文](README.zh-CN.md)

C++20 local inference engine for **Qwen3.5 / 3.6 / 3.8-27B** hybrid text models.

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
- Speculative decode: `off` / `ngram` / `mtp` / `auto` / `draft`. Recommended draft: [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B)
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
rapidllm -m /path/to/Qwen3.6-27B-FP8 --draft /path/to/Qwen3.5-0.8B --spec draft --spec-n 3 --prompt "Hello"
rapidllm bench -m model.gguf
rapidllm bench -m model.gguf --device cuda --batch 4
rapidllm bench -m /path/to/Qwen3.6-27B-FP8 --device cuda --ctx 131072 --prompt-n 8192 --max-new 8 --spec off
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
| `--draft` | — | draft weights; implies `--spec draft`. Use [`Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| `--batch` | `1` | copies of the same prompt; sets `RAPIDLLM_MAX_BATCH` |
| `--prompt-n` | — | synthesize `N` non-repeating token ids (skips the tokenizer). For long-ctx benches. |
| `--vision` / `--image` | off | load `visual.*` (encoder is CPU; generate does not yet consume the image) |
| `--thinking` | on | `bench` and `serve` always turn it off |

Weight sources:

- Target (latest): [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) — same `qwen3_5` hybrid IR as 3.6-27B
- Target (previous): [`Qwen/Qwen3.6-27B-FP8`](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)
- Community GGUF: [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) Q8_0 / Q6_K (also [`unsloth/Qwen3.6-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) Q4_K_M)
- Draft (recommended for 3.5 / 3.6 / **3.8**): [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) — same family and vocab **248320**, 24-layer hybrid (18 DeltaNet + 6 Gated Attn), hidden 1024

## Long-context limit numbers (RTX 6000 Ada 48 GB)

Same box as the short-prompt bakeoff: **NVIDIA RTX 6000 Ada Generation, 49140 MiB**. Official FP8 text weights ~28 GiB. RapidLLM CUDA KV is FP32 (about **128 KiB / token** across the 16 gated-attention layers). vLLM uses paged FP16 KV (about **64 KiB / token**).

All RapidLLM rows below are `--device cuda --fuse=on --spec off --no-thinking` (decode CUDA graph + fused GDN/RMS/MLP). `--spec auto` / draft is not used here: a repeating prompt would let n-gram fake throughput.

`--prompt-n N` fills `N` non-repeating ids. `tok/s` is wall (prefill + 8–16 new tokens). `decode_tok/s` is the decode-only rate after that fill. Prefill tok/s = `N / prefill_s`.

### RapidLLM official FP8

| Window `--ctx` | Filled tokens | Prefill s | Prefill tok/s | Decode tok/s | Wall tok/s (8–16 new) | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 256 (short bakeoff) | 3 | 0.037 | — | **32.02** | **31.65** | keep; vs vLLM 19.58 = **1.62×** |
| 4096 | 64 | 0.60 | 107 | 31.84 | 9.74 | still the short-ctx attn kernel |
| 16384 | 2048 | 23.68 | 86.5 | 15.64 | 0.33 | online softmax (`ctx>8192`) |
| 131072 | 256 | 2.61 | 98 | **27.97** | 2.80 | 128k window allocated |
| 131072 | 8192 | 121.17 | 67.6 | **6.31** | 0.065 | attn already dominates decode |
| 131072 | 131056 | — | — | — | — | full fill not finished (O(N²) attn; ~hours) |
| 200000 | 256 | — | — | — | — | **CUDA OOM** (~26 GiB FP32 KV + 28 GiB weights > 48 GiB) |

vLLM on this card reported that 200k needs **12.39 GiB** KV vs **12.05 GiB** free at `gpu_memory_utilization=0.90` (estimated max len **194432**).

### vLLM FP8 (graphs on, `enforce_eager=False`, `max_model_len=131072`)

Same token-id pattern, 8 new tokens, wall `tok/s` includes prefill:

| Window | Filled tokens | Wall s | Wall tok/s | vLLM logged in/out tok/s |
| --- | ---: | ---: | ---: | --- |
| 131072 | 256 | 0.456 | **17.54** | in 563 / out 17.6 |
| 131072 | 2048 | 1.013 | **7.90** | in 1738 / out 6.8 |
| 131072 | 8192 | 3.274 | **2.44** | in 2504 / out 2.45 |
| 200000 | — | — | — | load failed: KV 12.39 GiB > 12.05 GiB |

Short official pair (prompt `1,2,3`, 16 new): vLLM **19.5801** tok/s.

### GGUF Q6_K / Q8_0 on CUDA

Files: `bartowski/Qwen_Qwen3.6-27B-GGUF` → `Qwen_Qwen3.6-27B-Q6_K.gguf` (23 GiB), `Qwen_Qwen3.6-27B-Q8_0.gguf` (28 GiB).

CUDA session **OOM even at `--ctx 256`** on this 48 GB card. Q8 stays packed (~28 GiB) but embed is dequantized to FP32 (~5 GiB) and the engine also keeps GDN S / conv / workspace; the extra copies do not leave enough headroom. Q6_K is requantized to the FP8 GEMV path at load (same ~28 GiB device footprint) and hits the same wall.

So there is **no CUDA limit number** for Q6/Q8 on 48 GB. CPU GGUF still loads; that is not the GPU peak.

### How to reproduce

```bash
# RapidLLM 128k window, 8k fill (fits 48 GB)
rapidllm bench -m /path/to/Qwen3.6-27B-FP8 \
  --device cuda --ctx 131072 --prompt-n 8192 --max-new 8 \
  --spec off --fuse=on --no-thinking

# vLLM 128k window (graphs on)
# see scratch vllm_long.py: max_model_len=131072, TokensPrompt, 8 new tokens
```

Full 128k/200k **fills** are not a useful default: RapidLLM prefill attn is still O(N²) on the 16 gated-attention layers (8k fill already 121 s). Decode at a 128k **allocation** with a short fill is the number that isolates the window cost.

## Speculative decode

`--spec auto` (the default) picks a draft source in this order:

1. Attached `--draft` session
2. The target's own MTP head (`mtp.fc` / `mtp.norm`)
3. N-gram continuation from already generated tokens

CUDA without `--draft` uses n-gram only. `set_draft` requires matching vocab; architecture may differ.

Recommended draft for Qwen3.6-27B **and Qwen3.8-27B**: [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B). `set_draft` only requires matching vocab; architecture may differ. 3.8 is still `qwen3_5` with vocab **248320**, so the 0.8B draft stays valid.

| | Qwen3.8 / 3.6-27B (target) | Qwen3.5-0.8B (draft) |
| --- | --- | --- |
| Family | `qwen3_5` hybrid | same |
| Vocab | 248320 | **248320** |
| Layers | 64 (48 DeltaNet + 16 Attn) | 24 (18 DeltaNet + 6 Attn) |
| Hidden | 5120 | 1024 |
| DeltaNet V heads | 48 | 16 |
| Attn | 24 Q / 4 KV, hd 256 | 8 Q / 2 KV, hd 256 |

```bash
rapidllm -m /path/to/Qwen3.8-27B-FP8 \
  --draft /path/to/Qwen3.5-0.8B \
  --spec draft --spec-n 3 \
  --prompt "Hello"
```

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
