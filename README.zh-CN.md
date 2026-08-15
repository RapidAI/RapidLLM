# RapidLLM

[English](README.md) · [中文](README.zh-CN.md)

面向 **Qwen3.6-27B** 与 **Qwen3.8-27B** 混合架构的 C++20 本地推理引擎。

本项目不是 llama.cpp / ggml / vLLM / MLC 的包装层。

| | |
| --- | --- |
| 引擎许可 | **MIT**（`LICENSE`） |
| 模型权重 | Apache-2.0（Qwen） |
| 实现语言 | C++20（Python 仅用于 golden / fixture） |
| 目标平台 | Windows x86-64 · Linux x86-64 |
| 版本 | 0.1.0 |

**当前支持**

| | |
| --- | --- |
| 架构 | Qwen3.6-27B、Qwen3.8-27B（`qwen3_5` hybrid：48 层 Gated DeltaNet + 16 层 Gated Attention） |
| 投机解码 | `--spec off\|ngram\|mtp\|auto\|draft` |
| MTP | 目标模型自带多 token 预测头（`mtp.fc` / `mtp.norm`）作草稿 |
| 连续批 | `--batch N` / `rapidllm_generate_batch` — 每步共享一次权重扫描 |
| 相对 vLLM（同机） | 短 decode **1.50–1.80×**；能跑 hybrid GGUF；163k–262k 能分配。见 [相对 vLLM 的结论](#相对-vllm-的结论) |

设计文档：[docs/设计方案.md](docs/设计方案.md) · [docs/architecture.md](docs/architecture.md)

## 这是什么

RapidLLM 把官方 HuggingFace **block-FP8** 目录和社区 **GGUF** 文件加载成同一套 `ModelDesc` + `TensorTable`。Qwen3.6-27B 与 Qwen3.8-27B 共用这套 IR。调度器按 `layer_types[]` 分发：

- **48 层 Gated DeltaNet**（`linear_attention`）——与序列长度无关的 FP32 循环状态
- **16 层 Gated Attention**（`full_attention`）——GQA KV 缓存

默认走 **纯文本**。未指定 `--vision` / `--image` 时跳过视觉张量。MTP 会加载并用于投机解码（`--spec mtp` / `auto`）。

## 为什么要为混合架构自研

短 decode 仍是权重带宽问题（线性层 22–30 GB）。长上下文才是这颗模型的胜场：

| 状态 | 体积 | 是否随序列增长 |
| --- | --- | --- |
| Gated Attention KV（16 层，FP16） | 64 KiB / token | 是 |
| Gated Attention KV（`--kv-type q8k_tq3v`） | 约 22 KiB / token | 是（K=q8，V=TurboQuant-3） |
| DeltaNet 循环态 `S`（48 层，FP32） | 约 144 MiB | 否 |
| conv1d 滑窗 | 约 7.5 MiB | 否 |

32K 上下文下，KV 约 2 GiB FP16。`--ctx>163840`（或 `--kv-type q8k_tq3v` / `RAPIDLLM_KV_TQ=1`）把 K 压成 q8、V 压成 WHT+3bit，48 GB 才能分配 **262144**。落在 F16 窗口（8k）内的前缀仍走原来的 F16 attn，T=1 FP8 GEMV 不动。`RAPIDLLM_KV_TQ=0` 强制全量 F16（262k 会 OOM）。48 层线性注意力不长 KV。

## 功能

- 一等 **HF FP8** 加载器（`config.json` + safetensors / index）
- 一等 **GGUF** 加载器（Q4_K / Q5_K / Q6_K / Q8_0；未知 IQ 一次反量化）
- 默认 CPU decode：**AVX2** 必选，运行时派遣 **AVX-512**
- 可选 **CUDA**（`-DRAPIDLLM_WITH_CUDA=ON`）；无设备时可编译，host-ref 可测
- 双缓存：注意力 KV + DeltaNet 循环态 + 因果 conv 状态
- 可选 TurboQuant 风格 KV（`--kv-type q8k_tq3v`）：K 用 q8、V 用 3-bit WHT，大约是 FP16 的 1/3
- MemoryPlanner：超预算时缩短 ctx 或拒载，避免被操作系统杀掉
- 融合 decode（`--fuse=on|off`），便于与未融合算子对拍
- **投机解码**：`off` / `ngram` / `mtp` / `auto` / `draft`
- **MTP**：27B 内嵌 1 层 MTP 头出草稿 token；`--spec mtp`，或未挂 `--draft` 时默认 `auto` 会用它
- **连续批**：`--batch N`（同一 prompt 多副本，每步共享一次权重扫描；CUDA 用 `RAPIDLLM_MAX_BATCH`）
- 外挂草稿模型：[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- HTTP 服务：OpenAI `/v1/chat/completions`、`/v1/responses`，以及 Anthropic `/v1/messages`
- Qwen3.5 / 3.6 / 3.8 ViT + CLI 图文联合生成（`--image PATH`）
- 默认开启 thinking；`bench` 与 `serve` 强制 `enable_thinking=false`
- 带版本号的 **C API**（`include/rapidllm/api.h`）+ CLI

## 依赖

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

出厂默认 `ctx = 32768`。官方 FP8 文本权重常驻约 27–28 GiB，32 GB 机器必须走 GGUF Q4_K 或 INT4 repack。加载视觉塔大约再加 1–2 GiB。

## 编译

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

打开 NVIDIA kernel：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAPIDLLM_WITH_CUDA=ON
cmake --build build
```

Windows 上若没有 Ninja，去掉 `-G Ninja`，使用 Visual Studio 生成器即可。`rapidllm serve` 在 Windows 上会链接 `ws2_32`。

## 命令行

```text
rapidllm -m <hf-dir|file.gguf> [--device cpu|cuda] [--ctx 32768] [--prompt TEXT]
         [--max-new N] [--threads N] [--max-layers N] [--max-ram-mb N]
         [--thinking | --no-thinking] [--fuse=on|off]
         [--spec off|ngram|mtp|auto|draft] [--spec-n N] [--draft <hf-dir|file.gguf>]
         [--image PATH] [--vision] [--batch N]

rapidllm bench -m <path> [--device cpu|cuda] [--fuse=on|off] [--micro] [--batch N]

rapidllm serve -m <path> [--host 127.0.0.1] [--port 8080] [--device cpu|cuda]
```

示例：

```bash
rapidllm -m /path/to/Qwen3.6-27B-FP8 --device cpu --ctx 32768 --prompt "你好"
rapidllm -m model.gguf --device cpu --prompt "你好"
rapidllm -m /path/to/Qwen3.6-27B-FP8 --draft /path/to/Qwen3.5-0.8B --spec draft --spec-n 3 --prompt "你好"
rapidllm bench -m model.gguf
rapidllm bench -m model.gguf --device cuda --batch 4
rapidllm bench -m /path/to/Qwen3.6-27B-FP8 --device cuda --ctx 131072 --prompt-n 8192 --max-new 8 --spec off
rapidllm -m /path/to/Qwen3.8-27B-FP8 --image photo.png --prompt "图里有什么？" --max-new 64 --device cuda
rapidllm bench --micro
rapidllm serve -m /path/to/Qwen3.6-27B-FP8 --host 127.0.0.1 --port 8080 --device cuda
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--device` | `cpu` | `cuda` 需要 CUDA 构建 |
| `--ctx` | `32768` | planner 可能缩短或拒载 |
| `--max-new` | `8` | 生成 token 数 |
| `--fuse` | `on` | 融合 DeltaNet / Attn / MLP decode |
| `--spec` | `auto` | `draft` 需要 `--draft` |
| `--kv-type` | `f16`；`--ctx>163840` 自动 `q8k_tq3v` | `q8k_tq3v` = K q8 + V TurboQuant-3。环境变量 `RAPIDLLM_KV_TQ=0\|1` |
| `--draft` | — | 草稿权重；隐含 `--spec draft`。推荐 [`Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| `--batch` | `1` | 同一 prompt 的副本数；会设置 `RAPIDLLM_MAX_BATCH` |
| `--prompt-n` | — | 直接合成 `N` 个不重复 token id（不走分词）。长上下文 bench 用 |
| `--vision` / `--image` | 关 | 加载 `visual.*`。`--image PATH` 会跑 ViT 并把视觉 token 拼进 generate |
| `--thinking` | 开 | `bench` 与 `serve` 始终关闭 |

权重来源：

- 最新目标：[`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) — 与 3.6-27B 同一套 `qwen3_5` hybrid IR
- 上一版目标：[`Qwen/Qwen3.6-27B-FP8`](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)
- 社区 GGUF：[`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) Q4_K_M / Q5_K_M / Q6_K / Q8_0（亦可 [`unsloth/Qwen3.6-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF)）
- 推荐草稿（3.5 / 3.6 / **3.8 通用**）：[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) — 同族，词表 **248320**，24 层 hybrid（18 DeltaNet + 6 Gated Attn），hidden 1024

## Qwen3.8-27B 量化 bakeoff（RTX 6000 Ada 48 GB）

与 3.6 同一套官方 pair：prompt `1,2,3`，16 个新 token，`--spec off --fuse=on --device cuda --ctx 256`，thinking 关。墙钟 tok/s 是 warmup generate 之后。Q4_K / Q5_K / Q6_K 保持 packed，走原生 CUDA GEMV（`RAPIDLLM_REQUANT_KQUANT=1` 才 requant 成 FP8）。Q8_0 保持 packed，走原生 Q8 GEMV。

| 权重 | 体积 | 墙钟 tok/s | Decode tok/s | Prefill 秒 | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| 官方 FP8 | 30.9 GB | **28.99** | **30.72** | 0.064 | T=1 FP8 GEMV（复测 28.99） |
| Unsloth Q4_K_M | 17.1 GB | **25.00** | 26.93 | 0.083 | 原生 packed GEMV（`native_kquant=1`；复测 24.67） |
| Unsloth Q5_K_M | 19.8 GB | — | — | — | 本轮未重跑 |
| Unsloth Q6_K | 22.9 GB | **22.98** | 23.65 | 0.062 | 原生 packed GEMV（`native_kquant=1`；复测 22.93） |
| Unsloth Q8_0 | 29.0 GB | **34.77** | 35.58 | 0.038 | 原生 Q8 SoA GEMV（`native_q8=1`；复测 34.70） |

```bash
rapidllm bench -m /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  --device cuda --ctx 256 --max-new 16 --spec off --fuse=on --no-thinking --prompt 1,2,3
```

社区 Q8_0 有时会量化 DeltaNet 的 `in_proj_a` / `in_proj_b`；loader 会把这些 leftover 反量化成 F32。`A_log` / conv / norm 若被量化仍会拒载。

## Qwen3.6 / 3.8-27B 对比 vLLM（RTX 6000 Ada 48 GB）

同机官方 pair：prompt `1,2,3`，16 个新 token，thinking 关，prefix-cache 关。RapidLLM 为 `--spec off --fuse=on --device cuda --ctx 256`。墙钟 tok/s 是加载 + warmup generate 之后。

vLLM 为 **0.21.1rc1.dev260+g10d264a2b**，graphs on（`enforce_eager=False`，FULL + PIECEWISE）。vLLM **不能加载 hybrid GGUF**（`qwen35` 尚未支持）。GGUF 行因此是 RapidLLM GGUF 对 **同族官方 FP8** 的 vLLM。本机没有 Qwen3.6 的 Q4_K / Q5_K。

Q4_K / Q5_K / Q6_K 保持 packed，走原生 CUDA GEMV。Q8_0 走原生 Q8 GEMV。`RAPIDLLM_REQUANT_KQUANT=1` 可恢复旧的 requant-to-FP8 加载路径。

| 模型 | 权重 | RapidLLM 墙钟 | RapidLLM decode | vLLM 墙钟 | 相对 vLLM |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.6-27B | 官方 FP8 | **31.67** | 32.04 | **19.36** | **1.635×** |
| Qwen3.6-27B | Q6_K GGUF | **29.69** | 30.24 | 19.36（FP8） | **1.533×** |
| Qwen3.6-27B | Q8_0 GGUF | **26.23** | 27.40 | 19.36（FP8） | **1.355×** |
| Qwen3.8-27B | 官方 FP8 | **28.99** | 30.72 | **19.34** | **1.499×** |
| Qwen3.8-27B | Q4_K_M GGUF | **25.00** | 26.93 | 19.34（FP8） | **1.293×** |
| Qwen3.8-27B | Q5_K_M GGUF | — | — | 19.34（FP8） | 本轮未重跑 |
| Qwen3.8-27B | Q6_K GGUF | **22.98** | 23.65 | 19.34（FP8） | **1.188×** |
| Qwen3.8-27B | Q8_0 GGUF | **34.77** | 35.58 | 19.34（FP8） | **1.798×** |

```bash
rapidllm bench -m /path/to/Qwen3.8-27B-FP8 \
  --device cuda --ctx 256 --max-new 16 --spec off --fuse=on --no-thinking --prompt 1,2,3
```

未跑：Qwen3.6 Q4_K / Q5_K（本机没有权重）。vLLM 加载 Qwen3.8-27B-Q4_K_M 失败：`GGUF model with architecture qwen35 is not supported yet.`

## 相对 vLLM 的结论

同机（RTX 6000 Ada 48 GB），同一套官方 pair（短 prompt、16 个新 token、graphs on）。数字见上表。

**RapidLLM 领先的地方**

1. **单请求 decode。** 官方 FP8 墙钟是 vLLM 的 **1.50–1.64×**（3.8：28.99 vs 19.34；3.6：31.67 vs 19.36）。原生 Q8_0 相对同一趟 vLLM FP8 是 **1.80×**（34.77）。Decode 是融合 C++/CUDA，热路径没有 Python。
2. **hybrid GGUF。** vLLM 0.21.1rc1 加载不了 `qwen35` GGUF。RapidLLM 的 Q4_K / Q5_K / Q6_K / Q8_0 与官方 FP8 走同一套 IR，32 GB / 48 GB 机器可以继续用 packed 权重。
3. **这套混合架构。** 引擎按 `layer_types[]` 调度：48 层 O(1) DeltaNet 状态 + 16 层 KV。vLLM 把它当通用 serving 图；RapidLLM 的双缓存和融合 GDN/RMS/MLP 是为 3:1 交错写的。
4. **长窗口分配。** 不开 TurboQuant 能分配 **163840**；`--kv-type q8k_tq3v` 在 48 GB 上能分配 **262144**。同机 vLLM 加载 200k 失败（KV 12.39 GiB > `gpu_memory_utilization=0.90` 时剩余 12.05 GiB）。
5. **大窗口、短填的 decode。** 分配 128k、填 256 token：RapidLLM decode **27.43** tok/s，vLLM 墙钟 **17.54**。
6. **本地 / 可嵌入。** C API + CLI + 单进程 `serve`。没有 PyTorch 运行时。有 CPU AVX2/AVX-512 路径；vLLM 没有。

**vLLM 仍然更强的地方（不要夸大）**

- **长 prefill。** 128k 窗口填 2048–8192 时，vLLM 墙钟更高（7.90 vs 7.17；2.44 vs 1.93）。RapidLLM 16 层 gated attn 的 prefill 仍是 O(N²)。
- **多用户 serving。** vLLM 分页、把不同 prompt 连续批在一起，并且是完整的 OpenAI HTTP 栈。RapidLLM 的 `--batch N` 是**同一 prompt** 共享权重；`serve` 是单连接，没有 SSE。
- **生态。** 采样、工具、LoRA、多卡 TP/PP 是 vLLM 的地盘。RapidLLM 是面向这一族架构的单卡 / CPU 引擎。

**选 RapidLLM**：本地跑 Qwen3.6 / 3.8-27B 的 decode（尤其是 GGUF、128k–262k 分配、或不想上 Python）。**选 vLLM**：共享端点、混合请求、长 prefill。

## Qwen3.8-27B 单请求最大窗口与并发 TPS（RTX 6000 Ada 48 GB）

官方 FP8，`--device cuda --fuse=on --spec off --no-thinking`。墙钟 tok/s 是 warmup generate 之后。RapidLLM CUDA KV 是 FP32（约 **128 KiB / token**）。显卡 49140 MiB。

### 单请求最大窗口

分配 `--ctx`，再用 prompt `1,2,3`（3 token）生成 8 个新 token。测的是**能装下的最大窗口**，不是把窗口填满。

| `--ctx` | 结果 | 峰值 MiB | Decode tok/s | 墙钟 tok/s |
| ---: | --- | ---: | ---: | ---: |
| 131072 | 通过 | 44292 | 32.07 | 29.11 |
| 147456 | 通过 | 46340 | 32.03 | 28.85 |
| **163840** | **不开 TurboQuant 时的上限** | **48388** | 31.97 | 28.56 |
| 167936 | session 时 CUDA OOM | — | — | — |
| 172032 | session 时 CUDA OOM | — | — | — |
| 200000 | 未试（已在 OOM 之上） | — | — | — |

`--ctx 163840` + `--prompt-n 256` + 8 新：prefill **2.69 s**，decode **27.83** tok/s，墙钟 **2.72** tok/s（prefill 占主导）。峰值仍是 48388 MiB。

`--ctx 256` 约 27.8 GiB；每多 16384 token 的 KV 约 2.00 GiB。167936 比 163840 再多约 0.5 GiB，装不下。

`serve` 是单连接；并发路径是 `--batch N`（同一 prompt，每步共享一次权重扫描）。`ctx<=4096` 时 `RAPIDLLM_MAX_BATCH` 上限 32（`pf_cap`）。

### 并发 batch TPS

官方 pair：`--ctx 256 --max-new 16 --prompt 1,2,3`。`tok/s` 是**合计**（所有序列）。每条序列的 decode = decode tok/s / batch。

| `--batch` | 墙钟 tok/s | Decode tok/s | 每条 decode | 峰值 MiB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 31.63 | 32.01 | 32.01 | 27832 |
| 2 | 50.00 | 53.12 | 26.56 | 28024 |
| 4 | 87.90 | 92.77 | 23.19 | 28404 |
| 8 | 92.11 | 94.77 | 11.85 | 29168 |
| 16 | 94.13 | 95.57 | 5.97 | 30704 |
| **24** | **94.49** | 95.48 | 3.98 | 32240 |
| 32 | 94.42 | 95.38 | 2.98 | 33776 |

合计峰值约 **94.5 tok/s**（batch 24–32）。batch 8 之后权重扫描已经饱和，再加序列几乎不再涨。

### 262k TurboQuant + 连续批

`--ctx 262144` 开 TurboQuant（`RAPIDLLM_KV_TQ=1` / `--kv-type q8k_tq3v`，`--ctx>163840` 时自动打开）才能在 48 GB 上分配：K=q8、V=tq3 持久化约 5.7–6.1 GiB/路，外加 8k F16 工作窗口。全量 F16 的 262k 大约 16–17 GiB/路，装不下。2026-08-15 同机（RTX 6000 Ada 48 GB）重测，并发加到合计 TPS 不再涨为止（四种格式 batch 3 都会 OOM）。开关：`--device cuda --fuse=on --spec off --no-thinking --max-new 16 --prompt 1,2,3 --kv-type q8k_tq3v`，连续批（`--batch N`，每步共享一次权重扫描）。CUDA graph 开启（Q8 捕获失败，走 eager）。Q4/Q6 默认 requant 成 FP8 GEMV；Q8 保持 packed。`--spec` 关掉：重复的 `1,2,3` 会让 n-gram 虚增吞吐。`tok/s` 是**合计**。峰值 MiB 取 graph 打完后的 `cuda_mem_ready`。

| 权重 | `--batch` | 墙钟 tok/s | Decode tok/s | 每条 decode | 峰值 MiB | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 官方 FP8 | 1 | 29.06 | 30.81 | 30.81 | 35925 | 通过 |
| 官方 FP8 | **2** | **40.76** | **43.91** | 21.96 | 42325 | **能跑的最大并发** |
| 官方 FP8 | 3 | — | — | — | 46857 | session 时 CUDA OOM |
| Q4_K_M | 1 | 31.15 | 31.56 | 31.56 | 38991 | 通过 |
| Q4_K_M | **2** | **49.03** | **52.15** | 26.07 | 45781 | 通过 |
| Q4_K_M | 3 | — | — | — | — | session 时 CUDA OOM |
| Q6_K | 1 | 30.44 | 31.04 | 31.04 | 39761 | 通过 |
| Q6_K | **2** | **46.68** | **49.80** | 24.90 | 46551 | 通过 |
| Q6_K | 3 | — | — | — | — | session 时 CUDA OOM |
| Q8_0 | 1 | 34.88 | 35.71 | 35.71 | 40631 | 通过（eager；graph 捕获失败） |
| Q8_0 | **2** | **55.50** | **59.54** | 29.77 | 47421 | **合计 TPS 最高** |
| Q8_0 | 3 | — | — | — | — | session 时 CUDA OOM |

262k 各格式峰值（短 prompt，连续批）：

| 权重 | 最大 `--batch` | 峰值墙钟 tok/s | 峰值 decode tok/s |
| --- | ---: | ---: | ---: |
| 官方 FP8 | 2 | **40.76** | 43.91 |
| Q4_K_M | 2 | **49.03** | 52.15 |
| Q6_K | 2 | **46.68** | 49.80 |
| Q8_0 | 2 | **55.50** | 59.54 |

fill-256（仍在 8k F16 窗口内）`--ctx 262144`，同机同开关的更早一轮。Q6/Q8 的 prefill 仍走 packed GEMM，所以墙钟会掉，decode 仍然快：

| 权重 | `--batch` | 墙钟 tok/s | Decode tok/s | Prefill s | 峰值 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 官方 FP8 | 1 | 26.99 | 30.36 | 0.066 | 35925 |
| 官方 FP8 | 2 | 37.95 | 44.87 | 0.065 | 42325 |
| Q4_K_M | 1 | 23.97 | 31.17 | 0.109 | 38991 |
| Q4_K_M | 2 | 39.08 | 53.11 | 0.108 | 45781 |
| Q6_K | 1 | 9.16 | 30.74 | 0.646 | 39761 |
| Q6_K | 2 | 16.65 | 51.15 | 0.648 | 46551 |
| Q8_0 | 1 | 2.65 | 35.31 | 2.817 | 40631 |
| Q8_0 | 2 | 5.13 | 60.46 | 2.856 | 47421 |

这张 48 GB 卡在 262k 下，FP8 / Q4 / Q6 / Q8 的并发上限都是 **`--batch 2`**。合计 TPS 峰值是 **55.50 tok/s**（Q8_0，batch 2，3-token prompt）。官方 FP8 峰值是 **40.76 tok/s**（batch 2）。batch 3+ 会 OOM：持久化 KV 按路线性涨（约 5.7–6.1 GiB/路）。

```bash
# 不开 TurboQuant 时 48 GB 能分配的最大窗口
rapidllm bench -m /path/to/Qwen3.8-27B-FP8 \
  --device cuda --ctx 163840 --max-new 8 --spec off --fuse=on --no-thinking --prompt 1,2,3

# 262k + TurboQuant + 连续批（官方 FP8 合计 TPS 最高点）
RAPIDLLM_KV_TQ=1 rapidllm bench -m /path/to/Qwen3.8-27B-FP8 \
  --device cuda --ctx 262144 --batch 2 --max-new 16 --spec off --fuse=on --no-thinking --prompt 1,2,3

# 这台机器上 262k 合计 TPS 最高
RAPIDLLM_KV_TQ=1 rapidllm bench -m /path/to/Qwen3.8-27B-Q8_0.gguf \
  --device cuda --ctx 262144 --batch 2 --max-new 16 --spec off --fuse=on --no-thinking --prompt 1,2,3

# 短 ctx 并发合计 TPS
rapidllm bench -m /path/to/Qwen3.8-27B-FP8 \
  --device cuda --ctx 256 --batch 24 --max-new 16 --spec off --fuse=on --no-thinking --prompt 1,2,3
```

## 图文联合生成

`--image PATH` 读 PNG / JPEG / PPM / BMP，跑自研 ViT（CPU），并在 prompt 前插入 `vision_start + image_pad × N + vision_end`。generate 会把这些 `image_pad` 的 embedding 换成编码器输出（CPU 与 CUDA）。需要 HF 权重里仍有 `visual.*`（加 `--image` 才不会跳过）。草稿投机会关掉：草稿模型看不到图。

```bash
rapidllm -m /path/to/Qwen3.8-27B-FP8 --image photo.png \
  --prompt "图里有什么？" --max-new 64 --device cuda --spec off
```

视频还不支持。

## 长上下文极限数字（RTX 6000 Ada 48 GB）

机器与短 prompt bakeoff 相同：**NVIDIA RTX 6000 Ada Generation，49140 MiB**。官方 FP8 文本权重约 28 GiB。RapidLLM CUDA 的 KV 是 FP32（16 层 gated attn 合计约 **128 KiB / token**）。vLLM 是分页 FP16 KV（约 **64 KiB / token**）。

下面 RapidLLM 行都是 `--device cuda --fuse=on --spec off --no-thinking`（decode CUDA graph + 融合 GDN/RMS/MLP）。这里不用 `--spec auto` / draft：重复 prompt 会让 n-gram 把吞吐刷假。

`--prompt-n N` 填 `N` 个不重复 id。`tok/s` 是墙钟（prefill + 8–16 个新 token）。`decode_tok/s` 是填满之后的逐步 decode。Prefill tok/s = `N / prefill_s`。

### RapidLLM 官方 FP8

| 窗口 `--ctx` | 已填 token | Prefill 秒 | Prefill tok/s | Decode tok/s | 墙钟 tok/s（8–16 新） | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 256（短 bakeoff） | 3 | 0.038 | — | **32.03** | **31.61** | keep；对 vLLM 19.36 = **1.63×** |
| 4096 | 64 | 0.60 | 107 | 31.84 | 9.74 | 仍走短 ctx attn 核 |
| 16384 | 2048 | 23.68 | 86.5 | 15.64 | 0.33 | 旧 T=4 GEMM（已被下面取代） |
| 131072 | 256 | **0.43** | **595** | **28.51** | **11.93** | cuBLAS prefill + 切分 GDN + warp attn；vLLM 17.54 |
| 131072 | 2048 | **3.25** | **630** | 15.86 | **2.17** | 两段 1024 token；vLLM 7.90 |
| 131072 | 8192 | 121.17 | 67.6 | **6.31** | 0.065 | cuBLAS prefill 后尚未重跑 |
| 131072 | 131056 | — | — | — | — | 整窗填满未跑完（attn O(N²)，约数小时） |
| 200000 | 256 | — | — | — | — | **CUDA OOM**。3.8-FP8 实测最大可分配 **163840** |

vLLM 在这张卡上报告 200k 需要 **12.39 GiB** KV，`gpu_memory_utilization=0.90` 只剩 **12.05 GiB**（估计上限 **194432**）。

### vLLM FP8（graphs on，`enforce_eager=False`，`max_model_len=131072`）

同一套 token id，8 个新 token，墙钟 `tok/s` 含 prefill：

| 窗口 | 已填 token | 墙钟秒 | 墙钟 tok/s | vLLM 日志 in/out tok/s |
| --- | ---: | ---: | ---: | --- |
| 131072 | 256 | 0.456 | **17.54** | in 563 / out 17.6 |
| 131072 | 2048 | 1.013 | **7.90** | in 1738 / out 6.8 |
| 131072 | 8192 | 3.274 | **2.44** | in 2504 / out 2.45 |
| 200000 | — | — | — | 加载失败：KV 12.39 GiB > 12.05 GiB |

短官方 pair（prompt `1,2,3`，16 新）：vLLM **19.36** tok/s（3.6-27B FP8，graphs on）。见上面的 vs-vLLM 表。

### GGUF Q6_K / Q8_0 的 CUDA

Qwen3.6 的 Q6_K / Q8_0 以及 Qwen3.8 的 Q4_K / Q5_K / Q6_K / Q8_0 在这张 48 GB 卡上都能以 `--ctx 256` 加载并 decode（见上面的 vs-vLLM 表）。Embed 会反量化成 FP32（约 5 GiB）。Q4–Q6 加载时 requant 成 FP8 GEMV 路径；Q8 保持 packed。

48 GB 上长上下文的 GGUF 仍然紧：Q8 packed 权重（约 28–29 GiB）+ FP32 embed + FP32 KV，短窗口以外余量很小。128k 分配 bench 请用官方 FP8（或 Q4/Q5）。

### 怎么复现

```bash
# RapidLLM 128k 窗口，填 8k（48 GB 装得下）
rapidllm bench -m /path/to/Qwen3.6-27B-FP8 \
  --device cuda --ctx 131072 --prompt-n 8192 --max-new 8 \
  --spec off --fuse=on --no-thinking
```

整窗填满 128k/200k 不是默认该跑的：RapidLLM prefill attn 在 16 层 gated attn 上仍是 O(N²)（填 8k 已经 121 s）。看窗口代价，用 128k **分配** + 短填的 decode 数字即可。

## 投机解码与 MTP

Qwen3.6 / 3.8-27B 自带 **MTP** 头（`mtp.fc`、`mtp.norm`，`mtp_num_hidden_layers=1`）。RapidLLM 会解析并当作草稿跑：`--spec mtp` 固定用这颗头；未挂 `--draft` 时 `--spec auto` 也会用它。

`--spec auto`（默认）按此顺序选草稿：

1. 已挂上的 `--draft` session（推荐 Qwen3.5-0.8B）
2. 目标模型自带 **MTP** 头
3. 从已生成上下文做 n-gram 续写

```bash
rapidllm -m /path/to/Qwen3.8-27B-FP8 --spec mtp --spec-n 3 --prompt "你好"
```

CUDA 且未传 `--draft` 时只走 n-gram（MTP 草稿在 CPU session）。`set_draft` 要求词表一致，架构可以不同。

Qwen3.6-27B **和 Qwen3.8-27B** 的推荐草稿都是 [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B)。`set_draft` 只要求词表一致，架构可以不同。3.8 仍是 `qwen3_5`、词表 **248320**，所以 0.8B 草稿继续可用。

| | Qwen3.8 / 3.6-27B（目标） | Qwen3.5-0.8B（草稿） |
| --- | --- | --- |
| 架构族 | `qwen3_5` hybrid | 同族 |
| 词表 | 248320 | **248320** |
| 层数 | 64（48 DeltaNet + 16 Attn） | 24（18 DeltaNet + 6 Attn） |
| hidden | 5120 | 1024 |
| DeltaNet V 头 | 48 | 16 |
| Attn | 24 Q / 4 KV，hd 256 | 8 Q / 2 KV，hd 256 |

```bash
rapidllm -m /path/to/Qwen3.8-27B-FP8 \
  --draft /path/to/Qwen3.5-0.8B \
  --spec draft --spec-n 3 \
  --prompt "你好"
```

## 连续批

`--batch N` 对 **同一 prompt 跑 N 份副本**，每步 decode 只扫一遍权重。CUDA 上通过 `RAPIDLLM_MAX_BATCH` 生效（CLI 会按 `--batch` 设置）。CPU 会退回成 N 次串行 generate。

```bash
rapidllm bench -m /path/to/Qwen3.8-27B-FP8 --device cuda --batch 4 --spec off
```

C API：`rapidllm_generate_batch`。Session API：`Session::generate_batch`。

## HTTP 服务

`rapidllm serve` 是单连接 JSON 服务（尚无 SSE）。thinking 关闭。

| 方法 | 路径 | 协议 |
| --- | --- | --- |
| `GET` | `/health`、`/v1/health` | 存活检查 |
| `GET` | `/v1/models` | 模型列表 |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions |
| `POST` | `/v1/responses` | OpenAI Responses |
| `POST` | `/v1/messages` | Anthropic Messages |

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

`max_tokens` 上限 4096。`temperature <= 0` 走 greedy。

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
sc.spec = 3; /* auto；4 = draft-model */
RapidSession* sess = rapidllm_session_new(eng, &sc, &err);

int32_t ids[256], out[64];
int n = rapidllm_encode(eng, "你好", ids, 256, &err);
RapidSampleParams sp = {0};
sp.greedy = 1;
int got = rapidllm_generate(sess, ids, n, &sp, out, 64, &err);

/* 同一 prompt、N 条序列，每步共享一次权重扫描（CUDA）。 */
int out_n[4];
int32_t bout[4 * 64];
rapidllm_generate_batch(sess, ids, n, 4, &sp, bout, 64, out_n, &err);

rapidllm_session_free(sess);
rapidllm_free(eng);
```

用 `rapidllm_session_set_draft` 挂接草稿模型。`RAPIDLLM_API_VERSION` 为 `1`。完整接口见 `include/rapidllm/api.h`。

## 测试

CMake 会调用 `python/goldens/tiny_hybrid.py` 生成合成混合架构 fixture，并注册 CTest：

```bash
ctest --test-dir build --output-on-failure
```

| 测试 | 内容 |
| --- | --- |
| `test_ir` | `ModelDesc`、`layer_types[]`、`VisionDesc` |
| `test_loader` | HF FP8 + GGUF、非法 fixture 拒载、视觉名字映射 |
| `test_hybrid` | greedy token 对拍 golden；fuse 开关；两种格式 |
| `test_simd_bench` | AVX2 / AVX-512 相对标量 |
| `test_nv` | DP4A GEMV 参考实现 |
| `test_spec` | n-gram / MTP / draft-model 投机解码 |
| `test_batch` | 连续 batch 生成 |
| `test_protocol` | OpenAI / Anthropic 解析与渲染，以及 `/health` |
| `test_cuda_decode` | CUDA 路径或 host-ref 回退 |

## 仓库布局

```text
include/rapidllm/   对外头文件（C API、IR、运行时、kernel、server）
src/api/            extern "C" 实现
src/cli/            rapidllm、rapidllm_version
src/server/         HTTP 服务（OpenAI + Anthropic）
src/frontend/       HF safetensors、GGUF、名字映射、WeightStore
src/ir/             ModelDesc + VisionDesc
src/kernels/        标量 / AVX2 / AVX-512 / 融合 / vision / CUDA
src/runtime/        session、DualCache、planner、tokenizer、sampler
src/backend/cpu/    CpuDevice
shaders/            GLSL compute（未接入默认构建）
python/goldens/     tiny hybrid fixture 与 golden token
tests/              单元 / golden / 加载器 / 投机 / batch / protocol
docs/               架构设计（中英）
```

## v1 明确不做

- `serve` 不做 SSE / token 流式（一次 JSON 请求对应一次 JSON 响应）
- 不做视频编码器
- 不做 MoE 专家路径
- 不做 ARM / Apple Silicon
- 不链接、不内嵌 llama.cpp / ggml / FLA 源码

## 许可

引擎：[MIT](LICENSE)  
模型权重：Apache-2.0（Qwen）
