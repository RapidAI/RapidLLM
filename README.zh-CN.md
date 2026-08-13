# RapidLLM

[English](README.md) · [中文](README.zh-CN.md)

面向 **Qwen3.6-27B** 混合文本模型的 C++20 本地推理引擎。

本项目不是 llama.cpp / ggml / vLLM / MLC 的包装层。

| | |
| --- | --- |
| 引擎许可 | **MIT**（`LICENSE`） |
| 模型权重 | Apache-2.0（Qwen） |
| 实现语言 | C++20（Python 仅用于 golden / fixture） |
| 目标平台 | Windows x86-64 · Linux x86-64 |
| 版本 | 0.1.0 |

设计文档：[docs/设计方案.md](docs/设计方案.md) · [docs/architecture.md](docs/architecture.md)

## 这是什么

RapidLLM 把官方 HuggingFace **block-FP8** 目录和社区 **GGUF** 文件加载成同一套 `ModelDesc` + `TensorTable`。调度器按 `layer_types[]` 分发：

- **48 层 Gated DeltaNet**（`linear_attention`）——与序列长度无关的 FP32 循环状态
- **16 层 Gated Attention**（`full_attention`）——GQA KV 缓存

默认走 **纯文本**。未指定 `--vision` / `--image` 时跳过视觉张量。MTP 会解析并可用于草稿 token。

## 为什么要为混合架构自研

短 decode 仍是权重带宽问题（线性层 22–30 GB）。长上下文才是这颗模型的胜场：

| 状态 | 体积 | 是否随序列增长 |
| --- | --- | --- |
| Gated Attention KV（16 层，FP16） | 64 KiB / token | 是 |
| DeltaNet 循环态 `S`（48 层，FP32） | 约 144 MiB | 否 |
| conv1d 滑窗 | 约 7.5 MiB | 否 |

32K 上下文下，KV 约 2 GiB FP16（INT8 为 1 GiB）。48 层线性注意力不长 KV。

## 功能

- 一等 **HF FP8** 加载器（`config.json` + safetensors / index）
- 一等 **GGUF** 加载器（Q4_K / Q5_K / Q6_K / Q8_0；未知 IQ 一次反量化）
- 默认 CPU decode：**AVX2** 必选，运行时派遣 **AVX-512**
- 可选 **CUDA**（`-DRAPIDLLM_WITH_CUDA=ON`）；无设备时可编译，host-ref 可测
- 双缓存：注意力 KV + DeltaNet 循环态 + 因果 conv 状态
- MemoryPlanner：超预算时缩短 ctx 或拒载，避免被操作系统杀掉
- 融合 decode（`--fuse=on|off`），便于与未融合算子对拍
- 投机解码：`off` / `ngram` / `mtp` / `auto` / `draft`。推荐草稿：[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- 连续 batch：`--batch N`（同一 prompt 多副本，每步共享一次权重扫描；CUDA 用 `RAPIDLLM_MAX_BATCH`）
- HTTP 服务：OpenAI `/v1/chat/completions`、`/v1/responses`，以及 Anthropic `/v1/messages`
- Qwen3.6 ViT 编码器（`vision_encode`）；`--vision` / `--image` 会保留 `visual.*` 权重
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
| `--draft` | — | 草稿权重；隐含 `--spec draft`。推荐 [`Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| `--batch` | `1` | 同一 prompt 的副本数；会设置 `RAPIDLLM_MAX_BATCH` |
| `--vision` / `--image` | 关 | 加载 `visual.*`（编码器在 CPU；generate 尚未吃进图像） |
| `--thinking` | 开 | `bench` 与 `serve` 始终关闭 |

权重来源：

- 目标模型：[`Qwen/Qwen3.6-27B-FP8`](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)
- 社区 GGUF（32 GB 主路径）：[`unsloth/Qwen3.6-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) Q4_K_M
- 推荐草稿：[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) — 同属 `qwen3_5`，词表同为 **248320**，24 层 hybrid（18 DeltaNet + 6 Gated Attn），hidden 1024

`--spec auto` 按此顺序选草稿：已挂上的 `--draft` session → 目标模型自带 MTP 头 → n-gram 续写。CUDA 且未传 `--draft` 时只走 n-gram。`set_draft` 要求词表一致，架构可以不同。

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
- CLI 不做图文联合生成（编码器已实现，尚未把图像 token 插入 generate）
- 不做视频编码器
- 不做 MoE 专家路径
- 不做 ARM / Apple Silicon
- 不链接、不内嵌 llama.cpp / ggml / FLA 源码

## 许可

引擎：[MIT](LICENSE)  
模型权重：Apache-2.0（Qwen）
