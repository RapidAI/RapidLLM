# RapidLLM

[English](#english) · [中文](#中文)

C++20 local inference engine for **Qwen3.6-27B** hybrid text models.  
面向 **Qwen3.6-27B** 混合文本模型的 C++20 本地推理引擎。

This is not a wrapper around llama.cpp, ggml, vLLM, or MLC.  
本项目不是 llama.cpp / ggml / vLLM / MLC 的包装层。

| | |
| --- | --- |
| Engine / 引擎许可 | **MIT** (`LICENSE`) |
| Model weights / 模型权重 | Apache-2.0 (Qwen) |
| Language / 实现语言 | C++20 (Python only for goldens / fixtures) |
| Platforms / 目标平台 | Windows x86-64 · Linux x86-64 |
| Version / 版本 | 0.1.0 |

Design docs: [docs/architecture.md](docs/architecture.md) · [docs/设计方案.md](docs/设计方案.md)

---

## English

### What it is

RapidLLM loads official HuggingFace **block-FP8** directories and community **GGUF** files into one `ModelDesc` + `TensorTable`. The scheduler walks `layer_types[]` and dispatches:

- **48 × Gated DeltaNet** (`linear_attention`) — O(1) FP32 recurrent state
- **16 × Gated Attention** (`full_attention`) — GQA KV cache

v1 runs the **language model only**. Vision tensors are skipped at load time. MTP is parsed and can draft tokens; it is not a full multimodal stack.

### Why a hybrid engine

Short decode is still a weight-bandwidth problem (22–30 GB of linears). Long context is where the architecture wins:

| State | Size | Grows with sequence? |
| --- | --- | --- |
| Gated Attention KV (16 layers, FP16) | 64 KiB / token | yes |
| DeltaNet recurrent `S` (48 layers, FP32) | ~144 MiB | no |
| conv1d window | ~7.5 MiB | no |

At 32K context, KV is about 2 GiB FP16 (1 GiB INT8). The 48 linear-attention layers do not grow a KV cache.

### Features

- First-class **HF FP8** loader (`config.json` + safetensors / index)
- First-class **GGUF** loader (Q4_K / Q5_K / Q6_K / Q8_0; unknown IQ dequantized once)
- CPU decode by default: **AVX2** required, **AVX-512** dispatched at runtime
- Optional **CUDA** (`-DRAPIDLLM_WITH_CUDA=ON`) with host-ref kernels when no device is present
- Dual cache: attention KV + DeltaNet recurrent + causal conv state
- Memory planner: refuse or shrink context instead of letting the OS OOM-kill the process
- Fused decode path (`--fuse=on|off`) for A/B against unfused ops
- Speculative decode: `off` / `ngram` / `mtp` / `auto`
- Thinking on by default; `rapidllm bench` forces `enable_thinking=false`
- Versioned **C API** (`include/rapidllm/api.h`) + CLI

### Requirements

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

Factory default `ctx = 32768`. Official FP8 text weights are ~27–28 GiB resident — a 32 GB box must use GGUF Q4_K or INT4 repack.

### Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

With NVIDIA kernels:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAPIDLLM_WITH_CUDA=ON
cmake --build build
```

On Windows without Ninja, omit `-G Ninja` and use the Visual Studio generator.

### CLI

```text
rapidllm -m <hf-dir|file.gguf> [--device cpu|cuda] [--ctx 32768] [--prompt TEXT]
         [--max-new N] [--threads N] [--max-layers N] [--max-ram-mb N]
         [--thinking | --no-thinking] [--fuse=on|off]
         [--spec off|ngram|mtp|auto] [--spec-n N]

rapidllm bench -m <path> [--device cpu|cuda] [--fuse=on|off] [--micro]
```

Examples:

```bash
rapidllm -m /path/to/Qwen3.6-27B-FP8 --device cpu --ctx 32768 --prompt "Hello"
rapidllm -m model.gguf --device cpu --prompt "Hello"
rapidllm bench -m model.gguf
rapidllm bench --micro
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--device` | `cpu` | `cuda` requires a CUDA build |
| `--ctx` | `32768` | planner may shrink or refuse |
| `--max-new` | `8` | generated tokens |
| `--fuse` | `on` | fused DeltaNet / Attn / MLP decode |
| `--spec` | `auto` | n-gram or MTP draft |
| `--thinking` | on | bench always turns it off |

Weight sources:

- Official: [`Qwen/Qwen3.6-27B-FP8`](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)
- Community GGUF (32 GB path): [`unsloth/Qwen3.6-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) Q4_K_M

### C API

```c
#include "rapidllm/api.h"

RapidConfig cfg = {0};
cfg.model_path = "model.gguf";
cfg.device = "cpu";
cfg.ctx = 32768;
cfg.fuse = 1;

RapidError err = {0};
RapidLLM* eng = rapidllm_load(&cfg, &err);
RapidSessionConfig sc = {0};
sc.enable_thinking = 1;
sc.max_new_tokens = 64;
sc.spec = 3; /* auto */
RapidSession* sess = rapidllm_session_new(eng, &sc, &err);

int32_t ids[256], out[64];
int n = rapidllm_encode(eng, "Hello", ids, 256, &err);
RapidSampleParams sp = {0};
sp.greedy = 1;
int got = rapidllm_generate(sess, ids, n, &sp, out, 64, &err);

rapidllm_session_free(sess);
rapidllm_free(eng);
```

`RAPIDLLM_API_VERSION` is currently `1`. See `include/rapidllm/api.h` for the full surface (`prefill` / `decode` / `sample`, spec stats, bench stats).

### Tests

CMake generates a synthetic hybrid fixture (`python/goldens/tiny_hybrid.py`) and wires CTest:

```bash
ctest --test-dir build --output-on-failure
```

| Test | What it checks |
| --- | --- |
| `test_ir` | `ModelDesc`, `layer_types[]` |
| `test_loader` | HF FP8 + GGUF + reject-bad fixtures |
| `test_hybrid` | greedy tokens vs golden, fuse on/off, both formats |
| `test_simd_bench` | AVX2 / AVX-512 vs scalar |
| `test_nv` | DP4A GEMV reference |
| `test_spec` | n-gram / MTP speculative decode |
| `test_cuda_decode` | CUDA path or host-ref fallback |

### Layout

```text
include/rapidllm/   public headers (C API, IR, runtime, kernels)
src/api/            extern "C" implementation
src/cli/            rapidllm, rapidllm_version
src/frontend/       HF safetensors, GGUF, name map, WeightStore
src/ir/             ModelDesc
src/kernels/        scalar / AVX2 / AVX-512 / fused / CUDA
src/runtime/        session, DualCache, planner, tokenizer, sampler
src/backend/cpu/    CpuDevice
python/goldens/     tiny hybrid fixture + golden tokens
tests/              unit / golden / loader / spec
docs/               architecture (EN/ZH)
```

### Non-goals (v1)

- No OpenAI-compatible HTTP server
- No vision / video encoder
- No MoE expert path
- No ARM / Apple Silicon
- No linking or vendoring llama.cpp / ggml / FLA

---

## 中文

### 这是什么

RapidLLM 把官方 HuggingFace **block-FP8** 目录和社区 **GGUF** 文件加载成同一套 `ModelDesc` + `TensorTable`。调度器按 `layer_types[]` 分发：

- **48 层 Gated DeltaNet**（`linear_attention`）——与序列长度无关的 FP32 循环状态
- **16 层 Gated Attention**（`full_attention`）——GQA KV 缓存

v1 **只跑语言模型**。视觉张量在加载时跳过。MTP 会解析并可用于草稿 token，但不是完整多模态栈。

### 为什么要为混合架构自研

短 decode 仍是权重带宽问题（线性层 22–30 GB）。长上下文才是这颗模型的胜场：

| 状态 | 体积 | 是否随序列增长 |
| --- | --- | --- |
| Gated Attention KV（16 层，FP16） | 64 KiB / token | 是 |
| DeltaNet 循环态 `S`（48 层，FP32） | 约 144 MiB | 否 |
| conv1d 滑窗 | 约 7.5 MiB | 否 |

32K 上下文下，KV 约 2 GiB FP16（INT8 为 1 GiB）。48 层线性注意力不长 KV。

### 功能

- 一等 **HF FP8** 加载器（`config.json` + safetensors / index）
- 一等 **GGUF** 加载器（Q4_K / Q5_K / Q6_K / Q8_0；未知 IQ 一次反量化）
- 默认 CPU decode：**AVX2** 必选，运行时派遣 **AVX-512**
- 可选 **CUDA**（`-DRAPIDLLM_WITH_CUDA=ON`）；无设备时可编译，host-ref 可测
- 双缓存：注意力 KV + DeltaNet 循环态 + 因果 conv 状态
- MemoryPlanner：超预算时缩短 ctx 或拒载，避免被操作系统杀掉
- 融合 decode（`--fuse=on|off`），便于与未融合算子对拍
- 投机解码：`off` / `ngram` / `mtp` / `auto`
- 默认开启 thinking；`rapidllm bench` 强制 `enable_thinking=false`
- 带版本号的 **C API**（`include/rapidllm/api.h`）+ CLI

### 依赖

- CMake ≥ 3.20
- C++20 编译器（MSVC、Clang-cl 或 GCC/Clang）
- Python 3（仅用于生成 fixture / golden）
- 带 AVX2 的 x86-64
- 可选：CUDA Toolkit（sm_75 / sm_86 / sm_89）

**内存分档**

| 机器内存 | 推荐权重 | 默认 ctx |
| --- | --- | --- |
| ≤ 16 GB | 仅 tiny fixture | — |
| 32 GB | GGUF Q4_K（约 16.8 GB） | 32K，INT8 KV |
| 64 GB | 官方 FP8 或 Q4_K / Q5_K | 32K |
| ≥ 96 GB | FP8；262K 是能力上限 | 32K–128K |

出厂默认 `ctx = 32768`。官方 FP8 文本权重常驻约 27–28 GiB，32 GB 机器必须走 GGUF Q4_K 或 INT4 repack。

### 编译

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

打开 NVIDIA kernel：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAPIDLLM_WITH_CUDA=ON
cmake --build build
```

Windows 上若没有 Ninja，去掉 `-G Ninja`，使用 Visual Studio 生成器即可。

### 命令行

```text
rapidllm -m <hf-dir|file.gguf> [--device cpu|cuda] [--ctx 32768] [--prompt TEXT]
         [--max-new N] [--threads N] [--max-layers N] [--max-ram-mb N]
         [--thinking | --no-thinking] [--fuse=on|off]
         [--spec off|ngram|mtp|auto] [--spec-n N]

rapidllm bench -m <path> [--device cpu|cuda] [--fuse=on|off] [--micro]
```

示例：

```bash
rapidllm -m /path/to/Qwen3.6-27B-FP8 --device cpu --ctx 32768 --prompt "你好"
rapidllm -m model.gguf --device cpu --prompt "你好"
rapidllm bench -m model.gguf
rapidllm bench --micro
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--device` | `cpu` | `cuda` 需要 CUDA 构建 |
| `--ctx` | `32768` | planner 可能缩短或拒载 |
| `--max-new` | `8` | 生成 token 数 |
| `--fuse` | `on` | 融合 DeltaNet / Attn / MLP decode |
| `--spec` | `auto` | n-gram 或 MTP 草稿 |
| `--thinking` | 开 | bench 始终关闭 |

权重来源：

- 官方：[`Qwen/Qwen3.6-27B-FP8`](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)
- 社区 GGUF（32 GB 主路径）：[`unsloth/Qwen3.6-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) Q4_K_M

### C API

```c
#include "rapidllm/api.h"

RapidConfig cfg = {0};
cfg.model_path = "model.gguf";
cfg.device = "cpu";
cfg.ctx = 32768;
cfg.fuse = 1;

RapidError err = {0};
RapidLLM* eng = rapidllm_load(&cfg, &err);
RapidSessionConfig sc = {0};
sc.enable_thinking = 1;
sc.max_new_tokens = 64;
sc.spec = 3; /* auto */
RapidSession* sess = rapidllm_session_new(eng, &sc, &err);

int32_t ids[256], out[64];
int n = rapidllm_encode(eng, "你好", ids, 256, &err);
RapidSampleParams sp = {0};
sp.greedy = 1;
int got = rapidllm_generate(sess, ids, n, &sp, out, 64, &err);

rapidllm_session_free(sess);
rapidllm_free(eng);
```

`RAPIDLLM_API_VERSION` 当前为 `1`。完整接口见 `include/rapidllm/api.h`（`prefill` / `decode` / `sample`、投机统计、bench 统计）。

### 测试

CMake 会调用 `python/goldens/tiny_hybrid.py` 生成合成混合架构 fixture，并注册 CTest：

```bash
ctest --test-dir build --output-on-failure
```

| 测试 | 内容 |
| --- | --- |
| `test_ir` | `ModelDesc`、`layer_types[]` |
| `test_loader` | HF FP8 + GGUF，以及非法 fixture 拒载 |
| `test_hybrid` | greedy token 对拍 golden；fuse 开关；两种格式 |
| `test_simd_bench` | AVX2 / AVX-512 相对标量 |
| `test_nv` | DP4A GEMV 参考实现 |
| `test_spec` | n-gram / MTP 投机解码 |
| `test_cuda_decode` | CUDA 路径或 host-ref 回退 |

### 仓库布局

```text
include/rapidllm/   对外头文件（C API、IR、运行时、kernel）
src/api/            extern "C" 实现
src/cli/            rapidllm、rapidllm_version
src/frontend/       HF safetensors、GGUF、名字映射、WeightStore
src/ir/             ModelDesc
src/kernels/        标量 / AVX2 / AVX-512 / 融合 / CUDA
src/runtime/        session、DualCache、planner、tokenizer、sampler
src/backend/cpu/    CpuDevice
python/goldens/     tiny hybrid fixture 与 golden token
tests/              单元 / golden / 加载器 / 投机
docs/               架构设计（中英）
```

### v1 明确不做

- 不做 OpenAI 兼容 HTTP
- 不做视觉 / 视频编码器
- 不做 MoE 专家路径
- 不做 ARM / Apple Silicon
- 不链接、不内嵌 llama.cpp / ggml / FLA 源码

---

## License

Engine: [MIT](LICENSE)  
Model weights: Apache-2.0 (Qwen)
