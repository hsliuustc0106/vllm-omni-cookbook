---
layout: post
title: "理解 PR #5720 — MiniMax-H3 模块化流水线：两套 DiT，一组共享组件"
date: 2026-08-24 18:00:00 +0800
author: hsliuustc0106
summary: >-
  MiniMax-H3 如何让两套任务专用 DiT 共享一组组件，以及一份 4×H200 E2E 剖析：
  50 个 sigma 点的 warm latency 为 85.85 s，其中 80.62 s 用于去噪。
tags: [MiniMax-H3, H200]
category: PR Analysis
feature: pipeline
lang: zh
pair: /2026-08-24-understanding-pr-5720-minimax-h3-modular-pipeline/
permalink: /zh/2026-08-24-understanding-pr-5720-minimax-h3-modular-pipeline/
image: /assets/figures/minimax-h3-modular-pipeline/fig1-architecture.svg
usage:
  - label: "合并服务"
    blurb: "覆盖三种请求任务"
    title: "vllm serve · 同时加载 MiniMax-H3 两个分区"
    code: |
      export MODEL=MiniMaxAI/MiniMax-H3
      export PORT=8091

      CUDA_VISIBLE_DEVICES=0,1,2,3 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
      vllm serve "${MODEL}" \
        --omni \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --trust-remote-code \
        --num-gpus 4 \
        --usp 4 \
        --ring 1 \
        --vae-patch-parallel-size 4 \
        --vae-parallel-mode tile \
        --vae-use-tiling
    note: >-
      这是 pinned recipe 面向四张大显存 GPU 的合并配置，不是“任意四卡都能跑”的承诺。
      两套 DiT 都常驻；启动前必须按所选硬件核对容量。
  - label: "仅 FL2VA"
    blurb: "T2VA + 首尾帧条件 FL2VA"
    title: "vllm serve · 启动时只选 FL2VA 分区"
    code: |
      export MODEL=MiniMaxAI/MiniMax-H3
      export PORT=8091

      CUDA_VISIBLE_DEVICES=0,1,2,3 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
      vllm serve "${MODEL}" \
        --omni \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --trust-remote-code \
        --task-type fl2va \
        --num-gpus 4 \
        --usp 4 \
        --ring 1 \
        --text-encoder-tp-size 4 \
        --vae-patch-parallel-size 4 \
        --vae-parallel-mode tile \
        --vae-use-tiling \
        --diffusion-attention-backend CUDNN_ATTN
    note: >-
      这是实测的 4×H200 low-latency topology。它只下载并加载 FL2VA；请求可以使用
      task=t2va 或 task=fl2va，而 task=ref2va 会因缺少对应 DiT 被拒绝。
  - label: "仅 Ref2VA"
    blurb: "参考素材驱动生成"
    title: "vllm serve · 启动时只选 Ref2VA 分区"
    code: |
      export MODEL=MiniMaxAI/MiniMax-H3
      export PORT=8091

      CUDA_VISIBLE_DEVICES=0,1,2,3 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
      vllm serve "${MODEL}" \
        --omni \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --trust-remote-code \
        --task-type ref2va \
        --num-gpus 4 \
        --usp 4 \
        --ring 1 \
        --vae-patch-parallel-size 4 \
        --vae-parallel-mode tile \
        --vae-use-tiling
    note: >-
      只下载并加载 Ref2VA。请求要显式使用 task=ref2va 和受支持的视觉参考组合；
      这个服务不提供 T2VA 或 FL2VA。
decisions:
  - when: "一个端点必须接住所有 H3 任务"
    pick: "合并启动"
    why: "两套任务专用 DiT 同时可用，而 tokenizer、encoder 和 VAEs 只实例化一次。"
  - when: "输入只有文本或关键帧"
    pick: "仅 FL2VA 启动"
    why: "T2VA 与 FL2VA 本来就共用一个分区；不加载 Ref2VA，避免下载和加载永远用不到的 DiT。"
  - when: "所有请求都带参考素材"
    pick: "仅 Ref2VA 启动"
    why: "服务契约收窄到 Ref2VA，误发的 T2VA/FL2VA 路由会直接被拒绝。"
  - when: "要分析时间花在哪里"
    pick: "固定一个分区和一个任务"
    why: "固定工作负载后，startup、prompt、denoise、decode、CPU encoding 和 transport 的边界才能复核。"
  - when: "不确定机器能否装下"
    pick: "分别预算 storage、host RAM 与 HBM"
    why: "少下载一份 checkpoint，并不等于 GPU 与 host memory 会按同样数值下降。"
---

## 摘要 {#tldr}

**[PR #5720](https://github.com/vllm-project/vllm-omni/pull/5720)
把 MiniMax-H3 整理成“一间车间、两台专用引擎”：FL2VA DiT 负责文生视频+音频
(T2VA)和首/尾帧条件 FL2VA，Ref2VA DiT 负责参考素材驱动请求；两者共用一套
tokenizer、processor、Qwen3-VL encoder、video VAE 和 audio VAE。** 合并服务同时
加载两套 DiT，再由 `extra_params.task` 为每个请求选路；单分区服务则只加载一套。
这个类比的边界在容量上：共享组件只装一次，不代表两套 DiT 很小，checkpoint
storage、host RAM 与 GPU HBM 仍要分开预算。

本文以
[`072bfc02`](https://github.com/vllm-project/vllm-omni/commit/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65)
为 shipped baseline。4×H200 上，一条 profiled FL2VA-only 1344×768 T2VA 工作负载
（124 帧、50 个 sigma 点 / 49 次 denoiser evaluation）的 warm client receipt time 为
**85.845 ± 0.161 s**（mean ± sample SD，n=3）：其中 **80.616 s denoise**、
1.859 s VAE decode，剩余 3.370 s 分布在 prompt encoding、engine/IPC residual、
CPU MP4 encoding 和 HTTP residual。它只描述这条 fully pinned profiled condition，
不能推广到所有 H3 请求；unprofiled control 不稳定，因此 profiler overhead 仍未定量。

| Baseline 事实 | Shipped 行为 |
|---|---|
| Checkpoint 分区 | FL2VA 与 Ref2VA |
| 请求任务 | `t2va`、`fl2va`、`ref2va` |
| 共享组件 | tokenizer、processor、保留层 Qwen3-VL encoder、video VAE、audio VAE |
| 普通默认调度 | 50 个 sigma 点 → 49 次 denoiser evaluation |
| Checkpoint storage | pinned recipe 给出每个分区约 135 GiB；两者合计约 270 GiB |
| Profiled 4×H200 warm E2E | 85.845 ± 0.161 s；80.616 s denoise（1344×768、124 帧、50 点 / 49 次 evaluation） |

## 背景 {#background}

用户看到的问题不是 MiniMax-H3 缺少两类任务，而是这两类任务看起来像两台完整机器，
尽管绝大多数部件其实相同。可以把它想成两条配送路线：引擎不同，但装货台、地图和
包装线完全一样；为了两条路线复制整座仓库，运营会变得没必要地笨重。

[PR #5691](https://github.com/vllm-project/vllm-omni/pull/5691) 首先引入原始
MiniMax-H3 diffusion pipeline。Checkpoint 发布了 FL2VA 与 Ref2VA 两个分区，任务
差别主要落在 diffusion transformer (DiT，也就是反复执行去噪的大网络)上；
tokenization、多模态 prompt encoding，以及视频/音频压缩组件是共用的。PR #5720
把这种结构落实成代码：从 repository root 启动时，两套 DiT 可以围绕一组共享组件
构建；需要单分区部署时，`--task-type` 仍然保留这条路径。

后续两个 merged PR 又把本文 baseline 的边界钉牢：
[PR #5752](https://github.com/vllm-project/vllm-omni/pull/5752) 对齐官方
FL2VA/Ref2VA 输入矩阵与校验规则；
[PR #5824](https://github.com/vllm-project/vllm-omni/pull/5824) 让
Qwen3-VL encoder loader 在参数或 fused source shard 缺失时 fail closed。也就是说，
这不只是一张架构图：不完整 checkpoint 会在 startup 失败，超出已加载分区能力的
请求也会被明确拒绝。

## 心智模型：一间车间，两台引擎 {#mental-model}

最实用的心智模型，是共享一张前台和一条成品线，中间放两台专用引擎。前台读懂请求，
其中一台引擎反复去噪，成品线再把 latent 还原成同步视频和音频。

![MiniMax-H3 模块化架构：FL2VA 与 Ref2VA DiT 共用 encoding 和 VAE 组件]({{ site.baseurl }}/assets/figures/minimax-h3-modular-pipeline/fig1-architecture.svg)

各部件可以精确对应：

- **前台：** tokenizer、Qwen3-VL processor 和保留层 Qwen3-VL text/vision encoder，
  负责构造 DiT 读取的上下文。
- **两台引擎：** `transformer` 是 FL2VA DiT；合并模式下的
  `transformers_ref` 是 Ref2VA DiT。
- **成品线：** 共享 video VAE 与 audio VAE 解码最终 latent；serving layer 将结果
  打包成 H.264 视频和同步立体声音频。

类比失效的地方也必须说清：参考图像、视频与音频可能在去噪前先经过 VAE encode，
分布式部署也可能切分或搬移组件。这些细节会改变内存和延迟，但属于后续 parallelism/
offload 文章；本文只先把逻辑 ownership 讲准确。

## 架构与数据流 {#architecture-dataflow}

Startup 决定服务器里有哪些引擎；随后，请求经过共享准备、其中一台引擎与共享解码。
它更像在流水线中间放一个道岔，而不是从原始输入到 MP4 复制两条完整生产线。

[`pipeline_minimax_h3.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py#L162-L177)
里的 startup resolver 将 model-level 选择映射到分区：

```python
if task in {"auto", "combined"}:
    return "combined"
if task in {"t2va", "fl2va"}:
    return "fl2va"
if task == "ref2va":
    return "ref2va"
```

这个选择真正改变下载与构造内容，不只是换一个标签：

| Startup 模式 | 构造的 DiT | 共享组件来源 | 接受的请求任务 |
|---|---|---|---|
| 合并（repository model，不传 `--task-type`） | FL2VA + Ref2VA | FL2VA；Ref2VA 只贡献 model index 与 transformer | `t2va`、`fl2va`、`ref2va` |
| `--task-type fl2va`（或直接传 `FL2VA` 路径） | 仅 FL2VA | FL2VA | `t2va`、`fl2va` |
| `--task-type ref2va`（或直接传 `Ref2VA` 路径） | 仅 Ref2VA | Ref2VA | `ref2va` |

[Pinned recipe](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/recipes/MiniMaxAI/MiniMax-H3.md#start-a-server)
记录每个分区约 134 GiB BF16 safetensors（磁盘约 135 GiB），所以合并服务需要约
270 GiB checkpoint storage；task-selected 服务只下载所选分区。这只是 **storage**
事实。Host loading 峰值与 GPU residency 取决于 loader、topology 与 offload 设置，
必须另行实测。

## 请求如何选择 DiT {#task-routing}

这里有两个用途不同的开关：`--task-type` 是服务器启动时装进配电箱的断路器，
`extra_params.task` 是某一个请求按下的按钮。请求无法选择 startup 根本没有加载的
DiT。

| 请求任务 | 选择的 DiT | 输入要求 |
|---|---|---|
| `t2va` | FL2VA | 只有文本 |
| `fl2va` | FL2VA | 一或两张关键帧，遵守 shipped 首/尾帧契约 |
| `ref2va` | Ref2VA | 受支持的视觉参考组合；拒绝纯音频输入 |

当前 resolver 也提供 implicit fallback：Ref2VA-only 服务默认 `ref2va`；其它情况下，
video/audio 输入推断为 `ref2va`，image 输入推断为 `fl2va`，纯文本推断为 `t2va`。
生产环境更推荐显式填写 `task`，因为日志、拒绝原因与 profiling manifest 都会更明确。

例如，同一个合并 endpoint 可以把纯文本请求路由到 FL2VA 分区：

```bash
curl -sS -X POST "http://127.0.0.1:${PORT}/v1/videos/sync" \
  -F 'prompt=A quiet moonlit harbor with synchronized waves and wind.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":8.7,"audio_flow_shift":3.0}' \
  -o t2va.mp4
```

FL2VA 与 Ref2VA 所需的 multipart reference 字段和限制，请直接使用
[current recipe](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/recipes/MiniMaxAI/MiniMax-H3.md#http-api-examples)，
不要复制历史 PR test command。

## 流水线阶段 {#pipeline-stages}

请求会经过 encode、diffuse、decode 三个 model stage，但“encode”并不是一个万能
计时桶。它像厨房小票上的项目名：只有说明备料和最终打包是否也记在这一行，数字才有
意义。

| 阶段 | 发生什么 | 频率 | Source boundary |
|---|---|---|---|
| 校验 + 准备 references | task/input 校验、视频裁切、shape/duration 解析 | 每请求一次 | serving + pipeline preparation |
| Prompt/context encode | tokenizer/processor + 保留层 Qwen3-VL encoder | 每请求一次 | `encode_prompt` |
| Reference encode | 需要时，video/audio VAE 将条件转成 latent rows | 每组 reference 一次 | 独立 profiler target，不自动属于 `encode_prompt` |
| Diffuse | 所选 DiT 沿相邻 sigma point 更新打包的 video/audio rows | 每次 denoiser evaluation 一次 | `diffuse` |
| Decode | 共享 video VAE 与 audio VAE 还原最终媒体 | 每个 output 一次 | `decode` |
| 打包响应 | host CPU 编码/封装 MP4，HTTP 传输 body | 每响应一次 | 不属于三个 model-stage total |

核心实现分布在四处：

- [`pipeline_minimax_h3.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py#L660-L825)
  构造所选 DiT 与共享组件；它的
  [`forward`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py#L1731-L1998)
  依次校验、encode、diffuse、decode。
- [`packed_sequence.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/packed_sequence.py)
  将 text、待生成 video/audio 与任务专用条件 rows 排成 DiT 读取的联合 sequence。
- [`time_request.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/time_request.py#L33-L70)
  构造普通 uniform sigma points。
- [`denoise_loop.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/denoise_loop.py#L139-L203)
  在每两个相邻点之间执行一次 DiT evaluation。

最后一条解决了最容易混淆的术语：普通路径的 `num_inference_steps=N` 会构造 **N 个
sigma 点、执行 N−1 次 denoiser evaluation**。因此默认 `50` 是 49 次 evaluation；
请求值 `4` 是四点、三次 evaluation，并不是后续 Turbo 的 four-NFE 契约——Turbo
请求五个点。Turbo 本身不在本文 baseline 范围内。

## 时间花在哪里，以及怎样正确测量 {#measured-profile}

一张 timing chart 是某一单订单的收据，不是所有菜品的统一价目表。Resolution、frame
count、references、sigma count、topology、backend、compile 状态与 warmup 都会移动
边界；缺少这些上下文的百分比无法复用。

[`cookbook#39`](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/39)
规定的实验已经通过 `gpu run` 在四张 H200 上执行（cluster 的 raw `nvidia-smi` label
为 `NVIDIA L20X`）。配置是 FL2VA-only T2VA、Ulysses 4 / Ring 1 / DiT TP1、
text-encoder TP4、VAE patch-parallel 4/tile、BF16、CUDNN attention 与 regional
`torch.compile`。请求为 1344×768、requested 5.0 s、24 FPS、124 aligned frames、
seed 1101，以及普通路径 50 个 sigma 点 / 49 次 denoiser evaluation。第一条 cold
request 被排除，表格统计三条 warm profiled request。

![MiniMax-H3 四张 H200 实测 startup 与完整响应时间分解]({{ site.baseurl }}/assets/figures/minimax-h3-modular-pipeline/fig3-e2e-measured.svg)

| Environment field | Pinned value |
|---|---|
| Source / model | vLLM-Omni `072bfc02`；MiniMax-H3 snapshot `42ed227e` |
| Installed runtime | vLLM 0.27.0；vLLM-Omni package `0.27.0rc2.dev80+g20e3655f5`；PyTorch 2.13.0+cu129 |
| Hardware | 4×H200，driver 570.133.20；raw cluster label `NVIDIA L20X`，143,771 MiB/device |
| Topology | U4/Ring1/DiT-TP1，text-encoder TP4，VAE-PP4 tile |
| Precision / backend | BF16、无 quantization；CUDNN attention；regional compile，非 eager |
| Workload | T2VA、1344×768、124 帧 / 24 FPS、requested 5.0 s、50 点 / 49 次 evaluation |
| Samples | 一次 cold + 三次 warm profiled；ambient/warm page cache；1 Hz resource sampling |

Profiled process-start→first-`/health` startup 为 **149.880 s**（单次示意，不是重复
startup benchmark）：imports/CLI/config 50.741 s、worker+NCCL setup 37.000 s、model
initialization/load/placement 60.000 s、orchestrator/API readiness 2.138 s。Cold/compile
request 为 114.686 s。Warm complete-client receipt 为 **85.845 ± 0.161 s**
（mean ± sample SD，n=3）：

| Warm receipt bucket | Mean ± sample SD | Warm mean 占比 | Boundary |
|---|---:|---:|---|
| Prompt / text encoder | 0.056 ± 0.000 s | 0.06% | direct synchronized span |
| Denoise | 80.616 ± 0.074 s | 93.91% | direct synchronized span，49 次 evaluation |
| Video + audio VAE decode | 1.859 ± 0.004 s | 2.17% | direct synchronized span |
| Engine / IPC / output | 1.956 ± 0.115 s | 2.28% | server inference 内的 signed residual |
| CPU MP4 encode + mux | 1.326 ± 0.052 s | 1.54% | direct server span |
| HTTP transport | 0.033 ± 0.017 s | 0.04% | client-total 减 server inference |
| **Complete client receipt** | **85.845 ± 0.161 s** | **100%** | request start → 完整 MP4 body |

每条 request stack 都闭合到 client 收完 MP4 body 的时间：

```text
T_client = T_prompt + T_denoise + T_decode
         + T_engine_residual + T_mp4 + T_http_residual
```

Prompt、denoise、decode 与 CPU MP4 encoding 是 direct span；
`T_engine_residual` 和 `T_http_residual` 只是 accounting bucket，不是推断出来的 kernel。
12 条 profiled/control request 全部返回 HTTP 200，并通过 H.264、1344×768、124 帧/
24 FPS、stereo 32 kHz AAC 校验；canonical warm container 字节一致，sampled device
memory 峰值为 100,872 MiB。

四点/三次 evaluation diagnostic 的 warm mean 为 **10.375 ± 0.057 s**：denoise
4.947 s、VAE decode 1.849 s、engine residual 2.102 s、CPU MP4 1.390 s、prompt
0.051 s、HTTP 0.036 s。它只用于展示 schedule sensitivity，不是 Turbo，没有 quality
comparison，也不能称为 speedup。

> [!CAUTION]
> 三条 unprofiled control 不稳定，范围为 86.247–104.380 s（sample SD 9.111 s）。
> 比较 median 会得到表面上的 −8.48% profiler “overhead”，但符号错误，并超过协议的
> 5% disclosure threshold。因此本文**不**声称 profiling 加速模型，也不声称 observer
> overhead 已被定量；decomposition 与 85.845 s 只描述 profiled condition。

完整、可复核的
[evidence bundle](https://github.com/hsliuustc0106/vllm-omni-cookbook/tree/fe7ac7ab1aaca4192051acaacc399c93cdf14059/blog/assets/figures/minimax-h3-modular-pipeline/evidence/2026-08-24-h200-4gpu)
保存 frozen manifest、exact commands/harness、raw logs 与 response headers、health polls、
GPU/host samples、per-request signed residual、output hashes/`ffprobe` result 和 plotting
SVG。Open [PR #5810](https://github.com/vllm-project/vllm-omni/pull/5810) 仍然只是
open experimental context，不是本组 measurement 的来源。

## 服务模式 {#serving-modes}

先决定装哪些权重，再决定 performance topology；就像先决定卡车要带哪些工具，再安排
几位司机分工。FL2VA tab 是上面的四张 H200 实测 topology；combined 与 Ref2VA tab
保留 current recipe startup variant，但没有被本实验 benchmark。

{% include usage-cookbook.html modes=page.usage %}

每个 tab 都附带三条提醒：

1. **合并服务首先是容量选择。** 在这套 no-offload profile 中，两套 DiT 都常驻；
   “四张 GPU”本身不能保证 HBM 足够。
2. **Storage、host RAM 与 HBM 是三份预算。** 只选一个分区会少下载一份 checkpoint、
   少加载一套 task-specific weight，但 host/HBM 的精确差值属于某个 measured topology。
3. **Backend 与 compile flag 需要硬件证据。** 实测 FL2VA tab 在 H200 上固定 CUDNN；
   pinned recipe 负责 Blackwell default 和其它 profile。两套 flag 都不能移植到未经
   验证的卡上。

## 怎么选 {#decision-cards}

Operator 的核心决定，是 endpoint 承诺提供哪些任务。它像发布菜单：窄菜单更容易做
capacity planning，合并菜单则避免维护两套 endpoint。

{% include decision-cards.html items=page.decisions %}

## 限制与后续 {#limitations}

这里描述的是 shipped 起跑线，不是 MiniMax-H3 优化故事的全部。Baseline 地图的价值，
正是让后续捷径可以指出自己修改了哪个部件，而不用每篇文章重新画整台机器。

- 在 pinned baseline，H3 每个 diffusion batch 执行一个 generation request。Open #5810
  的 step execution/continuous batching 不能描述成 shipped。
- H3 是 classifier-free-guidance distilled model，因此 `--cfg-parallel-size` 必须保持 1；
  它没有 negative branch 可供并行。
- FL2VA-only 命令是四张 H200 实测 profile。Combined 与 Ref2VA 命令是 current
  recipe startup variant，不是本次 measurement；combined serving 需要容纳两套 DiT。
- Consumer、ROCm、offload、quantization、caching、attention tuning 与 Turbo 都有
  不同证据，属于各自 guide/post。
- Ref2VA reference 组合与 upload limit 会随 serving API 演进；应使用 pinned/current recipe，
  不要复制旧 PR body。
- E2E decomposition 是 profiled-condition measurement。Unprofiled control 不稳定，
  因此不能给出 observer-overhead claim；四点 diagnostic 也没有 quality evidence。
- 本文是
  [`cookbook#37`](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/37)
  规划系列的 Blog 1。Series metadata/navigation 是独立 site change，本文不会静默引入。

## 参考 {#references}

下面这组链接是设计的证据链：merged anchor 说明什么已经 shipped，pinned file 说明当前
行为，open work 则始终明确标成 open。

- [PR #5691 — Add MiniMax H3 diffusion support](https://github.com/vllm-project/vllm-omni/pull/5691)（merged）
- [PR #5720 — Add MiniMax-H3 modular pipeline support](https://github.com/vllm-project/vllm-omni/pull/5720)（merged）
- [PR #5752 — Align MiniMax H3 official input matrix](https://github.com/vllm-project/vllm-omni/pull/5752)（merged）
- [PR #5824 — Fail encoder load when a weight or fused shard is missing](https://github.com/vllm-project/vllm-omni/pull/5824)（merged）
- [PR #5810 — MiniMax-H3 continuous batching](https://github.com/vllm-project/vllm-omni/pull/5810)（open；仅作为实验背景）
- [Pinned MiniMax-H3 recipe](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/recipes/MiniMaxAI/MiniMax-H3.md)
- [Pinned pipeline implementation](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py) · [sigma construction](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/time_request.py) · [denoise loop](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/denoise_loop.py)
- [四张 H200 E2E evidence bundle](https://github.com/hsliuustc0106/vllm-omni-cookbook/tree/fe7ac7ab1aaca4192051acaacc399c93cdf14059/blog/assets/figures/minimax-h3-modular-pipeline/evidence/2026-08-24-h200-4gpu) — manifest、commands、harness、raw logs/headers/samples、results、hashes 与 media validation
- [Blog 1 计划与 E2E 实验契约](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/39) · [系列 RFC](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/37)
