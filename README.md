# RapidLLM

[English](README.md) · [中文](README.zh-CN.md)

C++20 local inference engine for the **Qwen3.6-27B** and **Qwen3.8-27B** hybrid architecture.

This is not a wrapper around llama.cpp, ggml, vLLM, or MLC.

| | |
| --- | --- |
| Engine license | **MIT** (`LICENSE`) |
| Model weights | Apache-2.0 (Qwen) |
| Language | C++20 (Python only for goldens / fixtures) |
| Platforms | Windows x86-64 · Linux x86-64 |
| Version | 0.1.0 |

**Supported now**

| | |
| --- | --- |
| Architecture | Qwen3.6-27B and Qwen3.8-27B (`qwen3_5` hybrid: 48 Gated DeltaNet + 16 Gated Attention) |
| Speculative decode | `--spec off\|ngram\|mtp\|auto\|draft` |
| MTP | Target model's own multi-token prediction head (`mtp.fc` / `mtp.norm`) as draft |
| Continuous batch | `--batch N` / `rapidllm_generate_batch` — one shared weight pass per step |
| vs vLLM (same box) | Short-decode **1.50–1.80×**; hybrid GGUF; 163k–262k alloc. See [Conclusion vs vLLM](#conclusion-vs-vllm) |

Design: [docs/architecture.md](docs/architecture.md) · [docs/设计方案.md](docs/设计方案.md)

## What it is

RapidLLM loads official HuggingFace **block-FP8** directories and community **GGUF** files into one `ModelDesc` + `TensorTable`. Qwen3.6-27B and Qwen3.8-27B share this IR. The scheduler walks `layer_types[]` and dispatches:

- **48 × Gated DeltaNet** (`linear_attention`) — O(1) FP32 recurrent state
- **16 × Gated Attention** (`full_attention`) — GQA KV cache

The default path is **text-only**. Vision tensors are skipped unless `--vision` or `--image` is set. MTP is loaded and used for speculative decode (`--spec mtp` / `auto`).

## Why a hybrid engine

Short decode is still a weight-bandwidth problem (22–30 GB of linears). Long context is where the architecture wins:

| State | Size | Grows with sequence? |
| --- | --- | --- |
| Gated Attention KV (16 layers, FP16) | 64 KiB / token | yes |
| Gated Attention KV (`--kv-type q8k_tq3v`) | ~22 KiB / token | yes (q8 K + TurboQuant-3 V) |
| DeltaNet recurrent `S` (48 layers, FP32) | ~144 MiB | no |
| conv1d window | ~7.5 MiB | no |

At 32K context, KV is about 2 GiB FP16. `--ctx>163840` (or `--kv-type q8k_tq3v` / `RAPIDLLM_KV_TQ=1`) keeps K as q8 and compresses V with a Walsh–Hadamard + 3-bit Lloyd-Max codebook so a 48 GB card can allocate **262144**. Compute for prefixes that fit the F16 window (8k) stays on the existing F16 attn path; T=1 FP8 GEMV is unchanged. `RAPIDLLM_KV_TQ=0` forces a full F16 cache (will OOM at 262k on 48 GB). The 48 linear-attention layers do not grow a KV cache.

## Features

- First-class **HF FP8** loader (`config.json` + safetensors / index)
- First-class **GGUF** loader (Q4_K / Q5_K / Q6_K / Q8_0; unknown IQ dequantized once)
- CPU decode by default: **AVX2** required, **AVX-512** dispatched at runtime
- Optional **CUDA** (`-DRAPIDLLM_WITH_CUDA=ON`) with host-ref kernels when no device is present
- Dual cache: attention KV + DeltaNet recurrent + causal conv state
- Optional TurboQuant-style KV (`--kv-type q8k_tq3v`): q8 keys + 3-bit WHT values, ~3× smaller than FP16
- Memory planner: refuse or shrink context instead of letting the OS OOM-kill the process
- Fused decode path (`--fuse=on|off`) for A/B against unfused ops
- **Speculative decode**: `off` / `ngram` / `mtp` / `auto` / `draft`
- **MTP**: 27B's embedded 1-layer MTP head drafts tokens; `--spec mtp` or default `auto` when no `--draft` is attached
- **Continuous batch**: `--batch N` (same prompt, one shared weight pass per step; CUDA via `RAPIDLLM_MAX_BATCH`)
- External draft model: [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- HTTP serve: OpenAI `/v1/chat/completions` + `/v1/responses`, Anthropic `/v1/messages`
- Qwen3.5 / 3.6 / 3.8 ViT + CLI image+text generate (`--image PATH`)
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
rapidllm -m /path/to/Qwen3.8-27B-FP8 --image photo.png --prompt "What is in this image?" --max-new 64 --device cuda
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--device` | `cpu` | `cuda` requires a CUDA build |
| `--ctx` | `32768` | planner may shrink or refuse |
| `--max-new` | `8` | generated tokens |
| `--fuse` | `on` | fused DeltaNet / Attn / MLP decode |
| `--spec` | `auto` | `draft` needs `--draft` |
| `--kv-type` | `f16`; auto `q8k_tq3v` if `--ctx>163840` | `q8k_tq3v` = K q8 + V TurboQuant-3. Env `RAPIDLLM_KV_TQ=0\|1` |
| `--draft` | — | draft weights; implies `--spec draft`. Use [`Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| `--batch` | `1` | copies of the same prompt; sets `RAPIDLLM_MAX_BATCH` |
| `--prompt-n` | — | synthesize `N` non-repeating token ids (skips the tokenizer). For long-ctx benches. |
| `--vision` / `--image` | off | load `visual.*`. `--image PATH` runs the ViT and splices visual tokens into generate |
| `--thinking` | on | `bench` and `serve` always turn it off |

Weight sources:

- Target (latest): [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) — same `qwen3_5` hybrid IR as 3.6-27B
- Target (previous): [`Qwen/Qwen3.6-27B-FP8`](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)
- Community GGUF: [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) Q4_K_M / Q5_K_M / Q6_K / Q8_0 (also [`unsloth/Qwen3.6-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF))
- Draft (recommended for 3.5 / 3.6 / **3.8**): [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) — same family and vocab **248320**, 24-layer hybrid (18 DeltaNet + 6 Gated Attn), hidden 1024

## Qwen3.8-27B quant bakeoff (RTX 6000 Ada 48 GB)

Same official pair as 3.6: prompt `1,2,3`, 16 new, `--spec off --fuse=on --device cuda --ctx 256`, thinking off. Wall tok/s after warmup generate. Q4_K / Q5_K / Q6_K stay packed and use native CUDA GEMV (set `RAPIDLLM_REQUANT_KQUANT=1` to requant to FP8). Q8_0 stays packed and uses the native Q8 GEMV.

| Weights | Size | Wall tok/s | Decode tok/s | Prefill s | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Official FP8 | 30.9 GB | **28.99** | **30.72** | 0.064 | T=1 FP8 GEMV (repeat 28.99) |
| Unsloth Q4_K_M | 17.1 GB | **25.00** | 26.93 | 0.083 | native packed GEMV (`native_kquant=1`; repeat 24.67) |
| Unsloth Q5_K_M | 19.8 GB | — | — | — | not re-run this pass |
| Unsloth Q6_K | 22.9 GB | **22.98** | 23.65 | 0.062 | native packed GEMV (`native_kquant=1`; repeat 22.93) |
| Unsloth Q8_0 | 29.0 GB | **34.77** | 35.58 | 0.038 | native Q8 SoA GEMV (`native_q8=1`; repeat 34.70) |

```bash
rapidllm bench -m /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  --device cuda --ctx 256 --max-new 16 --spec off --fuse=on --no-thinking --prompt 1,2,3
```

Q8_0 community files may quantize DeltaNet `in_proj_a` / `in_proj_b`; the loader dequants those leftovers to F32. `A_log` / conv / norms still reject if quantized.

## Qwen3.6 / 3.8-27B vs vLLM (RTX 6000 Ada 48 GB)

Same-box official pair: prompt `1,2,3`, 16 new tokens, thinking off, prefix-cache off. RapidLLM is `--spec off --fuse=on --device cuda --ctx 256`. Wall tok/s is after load + warmup generate.

vLLM is **0.21.1rc1.dev260+g10d264a2b**, graphs on (`enforce_eager=False`, FULL + PIECEWISE). vLLM **cannot load hybrid GGUF** (`qwen35` is not supported). GGUF rows therefore compare RapidLLM GGUF to the **same family's official FP8** on vLLM. Qwen3.6 Q4_K / Q5_K were not on the box.

Q4_K / Q5_K / Q6_K stay packed and use native CUDA GEMV. Q8_0 uses the native Q8 GEMV. `RAPIDLLM_REQUANT_KQUANT=1` restores the old requant-to-FP8 load path.

| Model | Weights | RapidLLM wall | RapidLLM decode | vLLM wall | vs vLLM |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.6-27B | Official FP8 | **31.67** | 32.04 | **19.36** | **1.635×** |
| Qwen3.6-27B | Q6_K GGUF | **29.69** | 30.24 | 19.36 (FP8) | **1.533×** |
| Qwen3.6-27B | Q8_0 GGUF | **26.23** | 27.40 | 19.36 (FP8) | **1.355×** |
| Qwen3.8-27B | Official FP8 | **28.99** | 30.72 | **19.34** | **1.499×** |
| Qwen3.8-27B | Q4_K_M GGUF | **25.00** | 26.93 | 19.34 (FP8) | **1.293×** |
| Qwen3.8-27B | Q5_K_M GGUF | — | — | 19.34 (FP8) | not re-run |
| Qwen3.8-27B | Q6_K GGUF | **22.98** | 23.65 | 19.34 (FP8) | **1.188×** |
| Qwen3.8-27B | Q8_0 GGUF | **34.77** | 35.58 | 19.34 (FP8) | **1.798×** |

```bash
rapidllm bench -m /path/to/Qwen3.8-27B-FP8 \
  --device cuda --ctx 256 --max-new 16 --spec off --fuse=on --no-thinking --prompt 1,2,3
```

Did not run: Qwen3.6 Q4_K / Q5_K (weights not present). vLLM GGUF load of Qwen3.8-27B-Q4_K_M failed with `GGUF model with architecture qwen35 is not supported yet.`

## Conclusion vs vLLM

Same box (RTX 6000 Ada 48 GB), same official pair (short prompt, 16 new, graphs on). Numbers above.

**Where RapidLLM is ahead**

1. **Single-request decode.** Official FP8 wall tok/s is **1.50–1.64×** vLLM (3.8: 28.99 vs 19.34; 3.6: 31.67 vs 19.36). Native Q8_0 is **1.80×** the same vLLM FP8 run (34.77). Decode is fused C++/CUDA with no Python in the hot path.
2. **Hybrid GGUF.** vLLM 0.21.1rc1 cannot load `qwen35` GGUF. RapidLLM runs Q4_K / Q5_K / Q6_K / Q8_0 on the same IR as official FP8, so a 32 GB / 48 GB box can stay on packed weights.
3. **This hybrid architecture.** The engine owns `layer_types[]`: 48 O(1) DeltaNet states + 16-layer KV. vLLM treats the model as a generic serving graph; RapidLLM's dual cache and fused GDN/RMS/MLP are written for this 3:1 mix.
4. **Long-window allocation.** Without TurboQuant, RapidLLM allocates **163840**. With `--kv-type q8k_tq3v` it allocates **262144** on 48 GB. vLLM on this card failed a 200k load (KV 12.39 GiB > 12.05 GiB free at `gpu_memory_utilization=0.90`).
5. **Short-fill decode at a large window.** 128k allocated, 256-token fill: RapidLLM decode **27.43** tok/s vs vLLM wall **17.54**.
6. **Local / embeddable.** C API + CLI + one-process `serve`. No PyTorch runtime. CPU AVX2/AVX-512 path exists; vLLM does not.

**Where vLLM is ahead (do not oversell)**

- **Long prefill.** At 128k / 2048–8192 fill, vLLM wall tok/s is higher (7.90 vs 7.17; 2.44 vs 1.93). RapidLLM prefill attention on the 16 gated layers is still O(N²).
- **Multi-user serving.** vLLM pages independent sequences, continuous-batches different prompts, and speaks a production OpenAI HTTP stack. RapidLLM `--batch N` is **same-prompt** weight sharing; `serve` is single-connection and not SSE.
- **Ecosystem.** Sampling, tools, LoRA, and multi-GPU TP/PP are vLLM's job. RapidLLM is a single-GPU / CPU engine for this one architecture family.

**Pick RapidLLM** for local Qwen3.6 / 3.8-27B decode (especially GGUF, 128k–262k allocation, or no Python). **Pick vLLM** for a shared endpoint with mixed prompts and long prefills.

## Qwen3.8-27B max context and concurrent TPS (RTX 6000 Ada 48 GB)

Official FP8, `--device cuda --fuse=on --spec off --no-thinking`. Wall tok/s after warmup generate. RapidLLM CUDA KV is FP32 (~**128 KiB / token**). GPU is 49140 MiB.

### Single-request max window

Allocate `--ctx`, then generate 8 new tokens from prompt `1,2,3` (3-token fill). This measures the largest window that fits, not a full-window prefill.

| `--ctx` | Result | Peak MiB | Decode tok/s | Wall tok/s |
| ---: | --- | ---: | ---: | ---: |
| 131072 | ok | 44292 | 32.07 | 29.11 |
| 147456 | ok | 46340 | 32.03 | 28.85 |
| **163840** | **max without TurboQuant** | **48388** | 31.97 | 28.56 |
| 167936 | CUDA OOM at session | — | — | — |
| 172032 | CUDA OOM at session | — | — | — |
| 200000 | not tried (above OOM) | — | — | — |

`--ctx 163840` + `--prompt-n 256` + 8 new: prefill **2.69 s**, decode **27.83** tok/s, wall **2.72** tok/s (prefill dominates). Peak still 48388 MiB.

`--ctx 256` is ~27.8 GiB; each extra 16384 tokens of KV is ~2.00 GiB. 167936 needs ~0.5 GiB more than 163840 and does not fit.

`serve` is single-connection; this engine's concurrent path is `--batch N` (same prompt, one shared weight pass). `ctx<=4096` caps `RAPIDLLM_MAX_BATCH` at 32 (`pf_cap`).

### Concurrent batch TPS

Official pair: `--ctx 256 --max-new 16 --prompt 1,2,3`. `tok/s` is **aggregate** (all sequences). Per-seq decode = decode tok/s / batch.

| `--batch` | Wall tok/s | Decode tok/s | Per-seq decode | Peak MiB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 31.63 | 32.01 | 32.01 | 27832 |
| 2 | 50.00 | 53.12 | 26.56 | 28024 |
| 4 | 87.90 | 92.77 | 23.19 | 28404 |
| 8 | 92.11 | 94.77 | 11.85 | 29168 |
| 16 | 94.13 | 95.57 | 5.97 | 30704 |
| **24** | **94.49** | 95.48 | 3.98 | 32240 |
| 32 | 94.42 | 95.38 | 2.98 | 33776 |

Peak aggregate is **~94.5 tok/s** at batch 24–32. After batch 8 the weight scan is already saturated; more sequences only add a little.

### 262k TurboQuant + continuous batch

`--ctx 262144` with TurboQuant (`RAPIDLLM_KV_TQ=1` / `--kv-type q8k_tq3v`, auto when `--ctx>163840`) allocates on 48 GB: K q8 + V tq3 persist ~5.7–6.1 GiB/seq plus an 8k F16 scratch window. Full F16 at 262k is ~16–17 GiB/seq and does not fit. Re-measured 2026-08-15 on the same RTX 6000 Ada 48 GB box, concurrency swept until aggregate TPS stopped rising (batch 3 OOMs on every format). Flags: `--device cuda --fuse=on --spec off --no-thinking --max-new 16 --prompt 1,2,3 --kv-type q8k_tq3v`, continuous batch (`--batch N`, one shared weight pass). CUDA graphs are on (Q8 capture fails and falls back to eager). Q4/Q6 use the default requant-to-FP8 GEMV; Q8 stays packed. `--spec` is off because a repeating `1,2,3` prompt would let n-gram fake throughput. `tok/s` is **aggregate**. Peak MiB is `cuda_mem_ready` after graphs.

| Weights | `--batch` | Wall tok/s | Decode tok/s | Per-seq decode | Peak MiB | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Official FP8 | 1 | 29.06 | 30.81 | 30.81 | 35925 | ok |
| Official FP8 | **2** | **40.76** | **43.91** | 21.96 | 42325 | **max that fits** |
| Official FP8 | 3 | — | — | — | 46857 | CUDA OOM at session |
| Q4_K_M | 1 | 31.15 | 31.56 | 31.56 | 38991 | ok |
| Q4_K_M | **2** | **49.03** | **52.15** | 26.07 | 45781 | ok |
| Q4_K_M | 3 | — | — | — | — | CUDA OOM at session |
| Q6_K | 1 | 30.44 | 31.04 | 31.04 | 39761 | ok |
| Q6_K | **2** | **46.68** | **49.80** | 24.90 | 46551 | ok |
| Q6_K | 3 | — | — | — | — | CUDA OOM at session |
| Q8_0 | 1 | 34.88 | 35.71 | 35.71 | 40631 | ok (eager; graph capture failed) |
| Q8_0 | **2** | **55.50** | **59.54** | 29.77 | 47421 | **highest aggregate TPS** |
| Q8_0 | 3 | — | — | — | — | CUDA OOM at session |

Per-format peak at 262k (short prompt, continuous batch):

| Weights | Max `--batch` | Peak wall tok/s | Peak decode tok/s |
| --- | ---: | ---: | ---: |
| Official FP8 | 2 | **40.76** | 43.91 |
| Q4_K_M | 2 | **49.03** | 52.15 |
| Q6_K | 2 | **46.68** | 49.80 |
| Q8_0 | 2 | **55.50** | 59.54 |

Fill-256 (still inside the 8k F16 window) at `--ctx 262144`, same box and flags, earlier run. Q6/Q8 prefill stays on packed GEMM, so wall tok/s drops even when decode is fast:

| Weights | `--batch` | Wall tok/s | Decode tok/s | Prefill s | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official FP8 | 1 | 26.99 | 30.36 | 0.066 | 35925 |
| Official FP8 | 2 | 37.95 | 44.87 | 0.065 | 42325 |
| Q4_K_M | 1 | 23.97 | 31.17 | 0.109 | 38991 |
| Q4_K_M | 2 | 39.08 | 53.11 | 0.108 | 45781 |
| Q6_K | 1 | 9.16 | 30.74 | 0.646 | 39761 |
| Q6_K | 2 | 16.65 | 51.15 | 0.648 | 46551 |
| Q8_0 | 1 | 2.65 | 35.31 | 2.817 | 40631 |
| Q8_0 | 2 | 5.13 | 60.46 | 2.856 | 47421 |

On this 48 GB card the 262k concurrency cap is **`--batch 2`** for FP8 / Q4 / Q6 / Q8. Peak total TPS is **55.50 tok/s** (Q8_0, batch 2, 3-token prompt). Official FP8 peaks at **40.76 tok/s** (batch 2). Batch 3+ OOMs because persist KV scales linearly (~5.7–6.1 GiB/seq).

```bash
# max window that allocates on 48 GB without TurboQuant
rapidllm bench -m /path/to/Qwen3.8-27B-FP8 \
  --device cuda --ctx 163840 --max-new 8 --spec off --fuse=on --no-thinking --prompt 1,2,3

# 262k + TurboQuant + continuous batch (highest official-FP8 total TPS)
RAPIDLLM_KV_TQ=1 rapidllm bench -m /path/to/Qwen3.8-27B-FP8 \
  --device cuda --ctx 262144 --batch 2 --max-new 16 --spec off --fuse=on --no-thinking --prompt 1,2,3

# 262k peak aggregate TPS on this box
RAPIDLLM_KV_TQ=1 rapidllm bench -m /path/to/Qwen3.8-27B-Q8_0.gguf \
  --device cuda --ctx 262144 --batch 2 --max-new 16 --spec off --fuse=on --no-thinking --prompt 1,2,3

# short-ctx concurrent aggregate TPS
rapidllm bench -m /path/to/Qwen3.8-27B-FP8 \
  --device cuda --ctx 256 --batch 24 --max-new 16 --spec off --fuse=on --no-thinking --prompt 1,2,3
```

## Image + text

`--image PATH` loads PNG / JPEG / PPM / BMP, runs the in-tree ViT (CPU), and splices `vision_start + image_pad × N + vision_end` in front of the prompt. Generate replaces those `image_pad` embeddings with the encoder output (CPU and CUDA). Needs an HF checkpoint that still has `visual.*` (pass `--image` so they are not skipped). Draft spec is turned off because the draft model does not see the picture.

```bash
rapidllm -m /path/to/Qwen3.8-27B-FP8 --image photo.png \
  --prompt "What is in this image?" --max-new 64 --device cuda --spec off
```

Video is not supported.

## Long-context limit numbers (RTX 6000 Ada 48 GB)

Same box as the short-prompt bakeoff: **NVIDIA RTX 6000 Ada Generation, 49140 MiB**. Official FP8 text weights ~28 GiB. RapidLLM CUDA KV is FP32 (about **128 KiB / token** across the 16 gated-attention layers). vLLM uses paged FP16 KV (about **64 KiB / token**).

All RapidLLM rows below are `--device cuda --fuse=on --spec off --no-thinking` (decode CUDA graph + fused GDN/RMS/MLP). `--spec auto` / draft is not used here: a repeating prompt would let n-gram fake throughput.

`--prompt-n N` fills `N` non-repeating ids. `tok/s` is wall (prefill + 8–16 new tokens). `decode_tok/s` is the decode-only rate after that fill. Prefill tok/s = `N / prefill_s`.

### RapidLLM official FP8

| Window `--ctx` | Filled tokens | Prefill s | Prefill tok/s | Decode tok/s | Wall tok/s (8–16 new) | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 256 (short bakeoff) | 3 | 0.038 | — | **32.03** | **31.61** | keep; vs vLLM 19.36 = **1.63×** |
| 4096 | 64 | 0.60 | 107 | 31.84 | 9.74 | still the short-ctx attn kernel |
| 16384 | 2048 | 23.68 | 86.5 | 15.64 | 0.33 | old T=4 GEMM (superseded below) |
| 131072 | 256 | **0.073** | **3500** | **27.43** | **24.35** | cublasLt FP8 + F32 GQA-GEMM attn; vs vLLM 17.54 |
| 131072 | 2048 | **0.70** | **2920** | 16.89 | **7.17** | two 1024-token tiles; vs vLLM 7.90 (still short) |
| 131072 | 4096 | **1.49** | **2746** | 11.83 | **3.84** | 4×1024 tiles; no paired vLLM |
| 131072 | 8192 | **3.21** | **2550** | 7.44 | **1.93** | vs vLLM 2.44 |
| 131072 | 16384 | **7.22** | **2270** | 4.27 | **0.90** | 16×1024 tiles; tokens non-zero |
| 131072 | 131056 | — | — | — | — | full fill not useful (O(N²) on 16 attn layers) |
| 200000 | 256 | — | — | — | — | **CUDA OOM**. Measured max alloc for 3.8-FP8 is **163840** |

vLLM on this card reported that 200k needs **12.39 GiB** KV vs **12.05 GiB** free at `gpu_memory_utilization=0.90` (estimated max len **194432**).

### vLLM FP8 (graphs on, `enforce_eager=False`, `max_model_len=131072`)

Same token-id pattern, 8 new tokens, wall `tok/s` includes prefill:

| Window | Filled tokens | Wall s | Wall tok/s | vLLM logged in/out tok/s |
| --- | ---: | ---: | ---: | --- |
| 131072 | 256 | 0.456 | **17.54** | in 563 / out 17.6 |
| 131072 | 2048 | 1.013 | **7.90** | in 1738 / out 6.8 |
| 131072 | 8192 | 3.274 | **2.44** | in 2504 / out 2.45 |
| 200000 | — | — | — | load failed: KV 12.39 GiB > 12.05 GiB |

Short official pair (prompt `1,2,3`, 16 new): vLLM **19.36** tok/s (3.6-27B FP8, graphs on). See the vs-vLLM table above.

### GGUF Q6_K / Q8_0 on CUDA

Qwen3.6 Q6_K / Q8_0 and Qwen3.8 Q4_K / Q5_K / Q6_K / Q8_0 all load and decode at `--ctx 256` on this 48 GB card (see the vs-vLLM table above). Embeddings are dequantized to FP32 (~5 GiB). Q4–Q6 requant to the FP8 GEMV path; Q8 stays packed.

Long-ctx GGUF on 48 GB is still tight: Q8 packed weights (~28–29 GiB) + FP32 embed + FP32 KV leave little room past a short window. Use official FP8 (or Q4/Q5) for 128k allocation benches.

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

## Speculative decode and MTP

Qwen3.6 / 3.8-27B ship an embedded **MTP** head (`mtp.fc`, `mtp.norm`, `mtp_num_hidden_layers=1`). RapidLLM parses it and runs it as a draft: `--spec mtp` always uses that head; `--spec auto` uses it when no `--draft` session is attached.

`--spec auto` (the default) picks a draft source in this order:

1. Attached `--draft` session (recommended: Qwen3.5-0.8B)
2. The target's own **MTP** head
3. N-gram continuation from already generated tokens

```bash
rapidllm -m /path/to/Qwen3.8-27B-FP8 --spec mtp --spec-n 3 --prompt "Hello"
```

CUDA without `--draft` uses n-gram only (MTP draft on CUDA is CPU-session). `set_draft` requires matching vocab; architecture may differ.

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

## Continuous batch

`--batch N` runs **N copies of the same prompt** with one shared weight pass per decode step. On CUDA, set `RAPIDLLM_MAX_BATCH` (the CLI does this from `--batch`). CPU falls back to N sequential generates.

```bash
rapidllm bench -m /path/to/Qwen3.8-27B-FP8 --device cuda --batch 4 --spec off
```

C API: `rapidllm_generate_batch`. Session API: `Session::generate_batch`.

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
- No video encoder
- No MoE expert path
- No ARM / Apple Silicon
- No linking or vendoring llama.cpp / ggml / FLA

## License

Engine: [MIT](LICENSE)  
Model weights: Apache-2.0 (Qwen)
