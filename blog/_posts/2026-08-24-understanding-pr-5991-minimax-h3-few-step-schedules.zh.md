---
layout: post
title: '在 vLLM-Omni 中服务 MiniMax-H3（2）：“四步”背后的三种契约'
date: 2026-08-24 19:00:00 +0800
author: hsliuustc0106
summary: >-
  讲清 MiniMax-H3 的短均匀请求、checkpoint 固定 DMD2 调度与 Turbo LoRA
  为何采用不同的 sigma 点和去噪器评估契约，即使三者都可能被称为“四步”。
tags: [MiniMax-H3, DMD2]
category: PR Analysis
feature: lora
math: true
lang: zh
pair: /2026-08-24-understanding-pr-5991-minimax-h3-few-step-schedules/
permalink: /zh/2026-08-24-understanding-pr-5991-minimax-h3-few-step-schedules/
redirect_from:
  - /zh/2026-08-24-understanding-pr-6476-minimax-h3-turbo-lora/
usage:
  - label: "下载"
    blurb: "只支持 v1.0 一个文件"
    title: "hf download · Turbo v1.0（唯一受支持的 artifact）"
    code: |
      export TURBO_DIR=/path/to/minimax-h3-turbo
      export TURBO_FILE=minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors
      hf download lightx2v/Minimax-h3-Turbo "${TURBO_FILE}" --local-dir "${TURBO_DIR}"
      export TURBO_LORA="${TURBO_DIR}/${TURBO_FILE}"
    note: >-
      8-step、ComfyUI、Ref2VA 和 v1.1 均不支持；唯一受支持的是原生
      Diffusers 四 NFE FL2VA/T2VA v1.0 文件。
  - label: "启动服务"
    blurb: "非 offload 配置 + 两个 LoRA flag"
    title: "vllm serve · 4 卡，预加载 Turbo"
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
        --vae-use-tiling \
        --task-type fl2va \
        --lora-backend peft \
        --lora-path "${TURBO_LORA}"
    note: >-
      --lora-path 只做预加载，每个请求仍要显式激活适配器。Turbo 拒绝
      model-level CPU offload、layerwise offload 和 DLO，因此要从非
      offload 的命令启动。
  - label: "发请求"
    blurb: "按请求激活 Turbo"
    title: "curl · 按发布的 Turbo 采样契约发送 T2VA 请求"
    code: |
      curl -sS -X POST "http://127.0.0.1:${PORT}/v1/videos/sync" \
        -F 'prompt=In a snowy blue-purple forest, Ori carefully walks past a sleeping giant.' \
        -F 'width=1344' \
        -F 'height=768' \
        -F 'aspect_ratio=16:9' \
        -F 'fps=24' \
        -F 'seed=1101' \
        -F 'num_inference_steps=5' \
        -F 'flow_shift=6' \
        -F 'extra_params={"task":"t2va","duration":8.7,"audio_flow_shift":3.0}' \
        -F "lora={\"name\":\"h3-turbo-v1.0\",\"path\":\"${TURBO_LORA}\",\"scale\":1.0}" \
        -o t2va_turbo.mp4
    note: >-
      5 个 sigma 点产生 artifact 需要的 4 次去噪器评估；视频 flow shift
      为 6，音频 flow shift 为 3。FL2VA 要改 task 并添加 input_reference。
      非法的点数或 shift 会直接失败。
decisions:
  - when: "蒸馏 checkpoint 发布了 `base_schedule`"
    pick: "以 checkpoint 元数据为准"
    why: "通常省略 `num_inference_steps`；如显式填写，必须等于区间数，而不是边界数。"
  - when: "已发布的基础 checkpoint 需要降低延迟"
    pick: "使用受支持的 Turbo LoRA 契约"
    why: "激活唯一受支持的 v1.0 FL2VA/T2VA 适配器，并请求 5 个 sigma 点及视频/音频 shift 6/3。"
  - when: "质量或复现需要基础参照"
    pick: "使用普通 50 点路径"
    why: "已发布基础 checkpoint 的默认值是 50 个 sigma 点和 49 次去噪器评估。"
  - when: "只需要诊断对照"
    pick: "不带蒸馏的短均匀请求"
    why: "`num_inference_steps=4` 表示 4 个均匀 sigma 点和 3 次去噪器评估；不要把它写成保质快速路径。"
---

## TL;DR {#tldr}

**MiniMax-H3 目前有三种都可能被叫作“四步”的不同契约：4 个均匀生成的
sigma 点、蒸馏 checkpoint 固定的 4 个区间，或 Turbo LoRA 要求的 4 次去噪器
评估。** 想象一条有 5 个车站、站间有 4 段行程的路线：数车站和数行程会得到
不同数字，描述的却是同一条路线。这里，**sigma 边界（sigma boundary）**是一个
预定噪声级，**NFE（number of function evaluations）**则是相邻边界之间调用
去噪器（denoiser）模型的次数。

[PR #5991](https://github.com/vllm-project/vllm-omni/pull/5991) 增加的是
checkpoint 原生路径：DMD2 蒸馏的 FL2VA checkpoint 可以发布训练时使用的精确
连续噪声位置。[PR #6476](https://github.com/vllm-project/vllm-omni/pull/6476)
后来增加的是另一条运行时路径：已发布的基础 checkpoint 保持不变，请求激活唯一
受支持的 Turbo LoRA 及其采样契约。**#5991 没有实现 Turbo LoRA；两条路径也都
不等于从基础轨迹里随意删掉若干次调用。**

| 路径 | 用户/checkpoint 输入 | 实际使用的 sigma 点或边界 | 去噪器评估（NFE） | 数字的含义 |
|---|---|---:|---:|---|
| 普通均匀基础路径 | 省略请求值或 `num_inference_steps=50` | 经模态 shift 后的 50 个均匀生成点 | 49 | 在这条路径上，请求字段控制 **sigma 点数量**。 |
| 普通均匀短请求 | 请求 `num_inference_steps=4` | 4 个均匀生成点 | 3 | 这**不是**四 NFE 请求。 |
| #5991 的 checkpoint 固定 DMD2 路径 | 元数据 `base_schedule: [1.0, 0.7, 0.4, 0.15, 0.0]`；请求省略或显式写 `4` | 5 个训练时的精确边界 | 4 | `DMD2SigmaSchedule.num_inference_steps` 数的是 4 个区间。 |
| #6476 已发布的 Turbo LoRA 路径 | 请求 `num_inference_steps=5`；激活受支持适配器 | 使用 Turbo 指定 shift 的 5 个均匀路径点 | 4 | legacy Turbo 集成校验 5 个点，因为它们产生 4 NFE。 |

> [!IMPORTANT]
> 本文始终区分 **sigma 点/边界**与**去噪器评估（NFE）**。只看 API 字段名
> 无法判断它控制哪一个量；当前 checkpoint 和适配器路径才决定语义。

## 为什么蒸馏模型不能随便使用短均匀调度 {#why-not-uniform}

蒸馏 checkpoint 学到的是一条特定快线，并不是从普通线路里任意删站的所有组合。
类比失效的地方在训练本身：噪声位置是输入模型的连续数值，改动一个边界会改变
去噪器必须处理的状态，而不只是改了一张时刻表。

已发布的基础模型按普通轨迹训练。让它只走少量均匀 sigma 点，确实能减少去噪器
调用，但不会凭空得到跨越更大区间所需的权重。蒸馏 checkpoint 或少 NFE 适配器
补上的，正是针对一条特定轨迹学到的能力。

“Turbo 跳过了 49 次去噪器评估中的 45 次”可以作为面向操作者的简写，但不是
实现机制。Turbo 并非从基础模型的 49 次评估轨迹里随意挑出 4 个位置；它的适配器
针对另一条四评估轨迹训练。

## 基础路径：50 个 sigma 点，49 次去噪器评估 {#baseline-path}

普通 H3 路径先在尺子上画等距刻度，再分别为视频和音频弯曲这把尺子。起始刻度
是均匀的；模态专属的时间 shift 会移动内部数值，但点数和首尾端点不变。

在固定的上游快照
[`072bfc02`](https://github.com/vllm-project/vllm-omni/commit/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65)
中，[`time_request.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/time_request.py)
用均匀的 `linspace(1.0, 0.0, num_steps)` 生成普通基础位置。对基础位置 $u$
和正的 shift scale $k$，H3 的映射是：

$$
s_k(u) = \frac{k u}{1 + (k - 1)u}
$$

随后，[`denoise_loop.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/denoise_loop.py)
在每一对相邻边界之间调用一次联合视频/音频去噪器：

$$
\mathrm{NFE} = \lvert\mathrm{sigma\ boundaries}\rvert - 1
$$

因此，默认 50 个 sigma 点会产生 49 NFE；普通请求
`num_inference_steps=4` 则产生 4 个点、仅 3 NFE。字段名沿用了历史命名；
在均匀路径上，它的值控制的是点数。

## PR #5991 增加了什么：由 checkpoint 拥有轨迹 {#checkpoint-dmd2}

#5991 让蒸馏 checkpoint 把路线卡放进包裹里，服务端直接读取训练边界，而不是
重新发明一条均匀路线。在 H3 语境下，“DMD2”实际表示：checkpoint 针对一小组
固定的连续噪声位置完成训练，并把这组位置发布为元数据。

契约位于当前 partition 的 `model_index.json` 中的 `_minimax_h3` 下。下面是合并
PR 给出的说明性元数据片段；它**不**表示某个特定路径上已有可下载 checkpoint：

```json
{
  "_minimax_h3": {
    "partition": "fl2va",
    "tasks": ["t2va", "fl2va"],
    "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
    "base_schedule": [1.0, 0.7, 0.4, 0.15, 0.0]
  }
}
```

共享的
[`DMD2SigmaSchedule`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/sched/sigma_schedule.py)
会 fail-closed；调度必须满足：

- 至少包含 2 个有限数位置；
- 精确从 `1.0` 开始、以 `0.0` 结束；
- 严格递减。

这个类有意与 `DMD2Config.denoising_timesteps` 分开：H3 保存的是 `[0, 1]`
中的连续 rectified-flow 位置，不是 scheduler-backed pipeline 使用的整数 scheduler
timestep。

缺少 `base_schedule` 表示“使用普通均匀路径”。显式给出空列表则是错误元数据，
会直接报错而不是悄悄回退。5 个边界描述 4 个区间，因此上述示例调度会报告
`num_inference_steps == 4`。

[`pipeline_minimax_h3.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py)
中的请求契约也跟随区间数。用户通常应省略 `num_inference_steps`；显式发送 `4`
与 4 个区间一致，发送 `5` 或 `50` 则会被拒绝。完整的 5 边界序列仍会传给去噪循环。
#5991 改变的是位置的来源及其校验方式，并没有替换 H3 的 DiT 权重 loader 或
去噪 solver。

## 一条基础调度，两条模态轨迹 {#modality-shifts}

视频与音频共享一组路标，却走曲率不同的车道，就像两辆车经过相同出口、采用不同
加速曲线。共享所有权保证两个模态按基础位置对齐；各自的 shift scale 再把位置调整
到对应模态。

对说明性的 5 边界调度，上述公式给出：

| 边界索引 | 基础位置 $u$ | 视频 sigma，$k=12$ | 音频 sigma，$k=3$ |
|---:|---:|---:|---:|
| 0 | 1.0 | 1.0 | 1.0 |
| 1 | 0.7 | 0.9655172 | 0.875 |
| 2 | 0.4 | 0.8888889 | 0.6666667 |
| 3 | 0.15 | 0.6792453 | 0.3461539 |
| 4 | 0.0 | 0.0 | 0.0 |

调度所有权还限制在各 partition 内。combined 服务为 FL2VA 与 Ref2VA 分别保留
元数据，再按当前任务选择调度。因此，蒸馏 FL2VA partition 不会悄悄强迫普通
Ref2VA partition 使用同一条四区间轨迹。这个边界来自 #5991 review，并由固定快照
中的 CPU 契约测试覆盖。

## PR #6476 增加了什么：运行时 Turbo LoRA {#runtime-turbo}

#6476 走的是另一条路：在已发布基础 checkpoint 旁预加载一位专用司机，再由请求
决定是否启用。基础 checkpoint 元数据没有被替换；适配器的权重/布局翻译也独立于
#5991 的 checkpoint 调度类。

现有 LoRA manager 把 H3 专属转换交给模型自己的 loader。这个边界很重要：发布的
适配器与原生 H3 transformer 指向同一组目标，却使用不同的名字和布局：

- Diffusers 名字会转换成原生 transformer 与 token-refiner 名字，fused FFN 行
  重新恢复为 H3 的 `[gate; up]` 顺序；
- 分离的 Q/K/V 适配器绑定到 H3 的 packed QKV 投影；fused LoRA-B 张量按全局
  输出行切分，让每个 tensor-parallel rank 得到正确权重；
- loader 在修改 wrapper 前校验完整元数据、rank/alpha、目标集合和全局 A/B shape。
  激活是事务性的：绑定或校验失败会清空适配器状态，不会留下半激活模型。

适配器在服务启动时预加载，但按请求激活。请求 5 个点是有意设计：在普通均匀路径
上，5 个 sigma 点产生 4 NFE。若改写为 `num_inference_steps=4`，只会产生 4 个
均匀点和 3 NFE，因此 Turbo 运行时会拒绝。

## 运行已发布的 Turbo 路径 {#turbo-usage}

下面合并后的工作流使用唯一受支持的 artifact：
`minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors`。先下载它，在非
offload 的 FL2VA 服务中预加载，再按 5 个 sigma 点和 shift 6/3 的契约，在
T2VA 请求上激活。

{% include usage-cookbook.html modes=page.usage %}

命令旁必须同时保留这些已发布限制：

- 只接受原生 Diffusers 四 NFE FL2VA/T2VA v1.0 artifact；Ref2VA、八 NFE
  版本、ComfyUI 布局和 v1.1 均不支持；
- 只支持动态执行，不支持 prefusion；
- 拒绝 model-level CPU offload、layerwise offload 与 distributed layerwise
  offload（DLO）；
- 同时只能激活一个 LoRA，因此 Turbo 不能与第二个风格或身份适配器组合；
- 文件名、元数据、rank/alpha 与目标 shape 都必须符合声明的 artifact 契约；
- legacy 请求携带适配器路径。公开端点应提供白名单内的名字到路径映射，而不是
  允许客户端任意指定路径或触发下载解析。

## 同一句话，不同的所有权 {#contract-ownership}

避免配置错误最简单的方法是先问“轨迹归谁所有”，就像先确认导航来自车辆、道路
管理方还是司机。H3 中的所有权同时决定位置本身，以及请求字段的解释方式。

| 路径 | 谁拥有位置？ | 请求可以控制什么 | 失败方式 |
|---|---|---|---|
| 普通基础路径 | 请求/pipeline | 生成的 sigma 点数量 | 少点请求可以运行，但基础权重没有被训练成这条捷径上的保质权重。 |
| checkpoint 固定 DMD2 | 当前 partition 元数据 | 通常不控制；可以用显式区间数确认元数据 | 显式数量不匹配会被拒绝。 |
| 运行时 Turbo LoRA | 受支持的适配器契约与请求 | 适配器激活及精确的五点、shift 6/3 契约 | 任务、点数、shift、artifact、offload 或组合错误都会被拒绝。 |

checkpoint 固定 DMD2 支持与运行时 Turbo 支持不是可以互换的打包格式。目前没有
具体 artifact 通过两条路径共同验证，而且请求契约可能直接冲突：说明性 checkpoint
期望显式区间数 `4`，已发布 Turbo 路径却要求点数 `5`。

## 四次去噪器评估为什么不是免费午餐 {#not-free}

减少去噪器调用，就像把过河的 49 个小跳变成 4 个大跳：路线确实更短，但只有为
这些落点专门训练过的人才能可靠完成。这个类比不承诺输出相等——训练方法、artifact、
prompt 与 workload 仍会共同决定质量。

普通短均匀请求可以作为诊断对照，因为它隔离了减少去噪器调用的成本；但它不是
等质量基线。#6476 的证据显示，不加 Turbo 适配器时输出会明显退化，加适配器后
视频/音频恢复连贯；但这并不构成对所有 prompt 的普遍量化等价保证。

#6476 作者的测量把“大幅减少调用次数”和“动态适配器成本”这两个效应分开。环境为
4×H200、USP4/Ring1、VAE patch-parallel 4、text-encoder TP1、regional compile
与 FlashAttention；768×1344 T2VA、107 帧/24 FPS；prompt 与 seed 相同；两次
完整 shape warmup 后测 5 次中位数。

| 路径 | LoRA 执行 | NFE | Stage-0 p50 |
|---|---|---:|---:|
| 基础参照 | 无 | 49 | 68.388 s |
| 短路径诊断对照 | 无 | 4 | 8.967 s |
| Turbo | 动态 | 4 | 9.688 s |

Turbo 相对 49-NFE 参照**快 7.06 倍**，但动态 LoRA 工作使它比同调度无 LoRA
对照**慢 8.05%**。正确解读是：延迟下降主要来自把去噪器评估从 49 次降到 4 次；
适配器则在保留下来的每次评估上增加工作。短路径对照的输出明显退化，因此它只是
计算量对照，不是等质量替代方案。

下面的本地验证在另一种拓扑上隔离第二个效应：当两个请求已经同为 5 个 sigma 点/
4 NFE 时，激活适配器究竟增加了什么工作？#5991 本身仍是调度契约，没有发布可
命名的公开蒸馏 checkpoint benchmark。

## 本地验证：Turbo 开销落在哪里 {#local-validation}

公平的 LoRA 对比要把道路固定，只改变车上多出来的负载。这里 Base 与 Turbo 使用
相同的 sigma 点、NFE、prompt、seed、shape、server 和热 cache；唯一变量是请求时
是否激活适配器。下面是固定环境的本地抽测，不是对所有部署的承诺。

### 同调度 A/B

Final-path 运行使用上游 SHA `072bfc02` 对应的 vLLM-Omni
`0.27.0rc2.dev159+g072bfc02d`，venv 与运行时 cache 均在 node-local 存储。
硬件是绑定 NUMA 0 的 2×L20X：eager TP2、text-encoder TP2、VAE
patch-parallel 2/tile、CUDNN attention；workload 为 1344×768 T2VA、请求
4.0 s/107 帧、seed 1101、5 个 sigma 点/4 NFE、视频/音频 shift 6/3。
一次 Base 与一次 Turbo warmup 后，按 `A B B A A B` 顺序测量，每种条件 n=3。
准备时间 8.621 s、进程到 ready 88.286 s，均单独记录。

| 条件 | Stage-0 mean ± sample SD | Stage-0 median | Diffuse mean ± sample SD | Diffuse median | Client median | Stage-0 CV |
|---|---:|---:|---:|---:|---:|---:|
| Base 对照 | 15.108 ± 0.024 s | 15.099 s | 10.211 ± 0.009 s | 10.206 s | 16.246 s | 0.16% |
| Turbo LoRA | 16.398 ± 0.072 s | 16.434 s | 11.435 ± 0.015 s | 11.444 s | 17.548 s | 0.44% |

Turbo median 开销为 **stage-0 +8.85%**、**diffuse +12.13%**、**client wall
+8.01%**。两种条件的请求峰值同为 76,754 MiB，因为 LoRA buffer 已预分配。
所有请求均返回 HTTP 200 并完整解码；同一条件内输出逐字节确定性一致，Base 与
Turbo 输出则不同。

### nsys 看到了什么

Kernel trace 的非调度条件与第 1 篇对齐：FL2VA T2VA、
Ulysses4/Ring1/DiT-TP1、text-encoder TP4、VAE patch-parallel 4/tile、
BF16、CUDNN、regional compile、1344×768、请求 5.0 s/124 帧、seed 1101。
两侧仍保持 5 个 sigma 点/4 NFE。GPU 4–7 固定在 NUMA 1，而不复用第 1 篇跨
NUMA 的物理卡位。每种条件先完成一次 compile/warmup，再由 Nsight Systems
2026.1.3 捕获一个请求；因此这些受 observer 影响的 span 用于解释机制，不构成
另一条延迟 headline。

| 直接同步 span | Base | Turbo | Delta |
|---|---:|---:|---:|
| Stage-0 | 10.332 s | 11.200 s | +8.39% |
| Pipeline diffuse | 6.569 s | 7.331 s | +11.61% |
| Pipeline decode | 1.907 s | 1.920 s | +0.69% |

可见 kernel 证据指向新增的低秩投影与 layout 工作，而不是不同的去噪器调用数：

| 每设备 Turbo-only 可见特征 | Unique kernels | Launches | Visible time | 示例 |
|---|---:|---:|---:|---|
| Rank-128 / `badd` GEMM | 4 | 439 | 38.122 ms | `nvjet_tst_128x288...badd`、`nvjet_tst_256x160...badd` |
| Fused copy/slice | 3 | 157 | 45.220 ms | `triton_poi_fused_copy_slice_{0,1,4}` |

与此同时，匹配到的 core kernel 保持相同 launch 数和近乎相同的可见时间：两组
主 GEMM 在两种条件下均为每设备 5,292 与 1,764 次，short-SDPA 为 1,764 次，
LayerNorm 为 7,056 次。这正是 dynamic LoRA 的架构特征：基础路径保留，再在其
周围增加低秩投影和 packed-layout 处理。

> [!CAUTION]
> 在这台 Hopper 机器上，nsys node mode 覆盖 host-launched CUDA graph node，
> 但不覆盖所有 device-launched graph node。Base 与 Turbo 的 graph 覆盖不同，
> 因而 aggregate kernel/category total 不是完整 workload total。可信对比是上面的
> 直接同步 span、launch 数相同的匹配 kernel，以及保守筛选出的 Turbo-only 特征。

## 怎么选 {#how-to-choose}

选择与学习 artifact 所拥有轨迹相匹配的路径，就像一把锁要用对应钥匙。基础
checkpoint、checkpoint 原生蒸馏版本和运行时适配器是三种不同运维产品，即使
面向操作者看到的 NFE 数相同。

{% include decision-cards.html items=page.decisions %}

## 兼容性与部署安全 {#compatibility-safety}

把每条快速路径看成狭窄的运行包线，就像桥上标明车型与载重限制。通过校验只表示
请求符合已实现契约，并不会把相邻 artifact 或特性自动纳入支持范围。

- checkpoint 原生蒸馏版本应携带训练时的精确 `base_schedule`；不要用新生成的
  均匀调度替换它。
- 运行时 Turbo 请求同时需要适配器激活与精确的五点采样设置。只预加载并不等于
  已激活。
- 不要把已发布 Turbo 路径与 Ref2VA、prefusion、任何受支持的 offload 模式或
  第二个 LoRA 组合。
- 不要因为任意 LightX2V artifact 名称含有 “Turbo” 或 “four-step” 就认定兼容。
- 不要向不受信任客户端暴露请求指定的文件系统路径或下载路径；应把批准的公开名称
  映射到服务端管理的路径。
- 在没有验证具体权重、元数据和轨迹前，不要声称 DMD2 checkpoint 与 Turbo LoRA
  可以相互转换。

## 限制与后续 {#limitations}

本文画出已发布边界，而不填造缺失产品，就像地图会标记尚未开放的道路，而不是虚构
一条穿过去的路线。以下省略是有意为之，目的是保证示例可部署。

- 在固定快照中，上游没有记录一个命名、稳定、公开可访问的 checkpoint 原生 DMD2
  FL2VA artifact。因此 #5991 部分只展示元数据，不给出 copy-ready 启动命令。
- 上面的本地 A/B 只隔离同调度下的适配器开销；它不是 #6476 的 49-NFE 加速对比，
  不是质量对比，也不是普遍延迟结论。
- [#6473](https://github.com/vllm-project/vllm-omni/pull/6473) 与
  [#6017](https://github.com/vllm-project/vllm-omni/pull/6017) 是 draft、
  未发布的 LoRA runtime 方向，本文不把其提议行为写成可用功能。
- Ref2VA Turbo、任意 LightX2V artifact、prefusion、适配器组合、quantization、
  TeaCache、Cache-DiT、continuous batching、VAE 优化和 Super Acceleration
  均不在本文范围内。

## 证据台账 {#evidence-ledger}

下面的结论都带着“收据”，而不只依赖“已合并”三个字，就像审计轨迹会同时记录规则
和真正执行规则的检查。可固定版本的源码链接都指向本次审阅的上游快照。

| 结论 | 来源 | 发布状态 | 独立证据 |
|---|---|---|---|
| 蒸馏 checkpoint 可以固定精确的连续调度位置 | [#5991](https://github.com/vllm-project/vllm-omni/pull/5991) + [`sigma_schedule.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/sched/sigma_schedule.py) | 已合并/已发布 | [`test_dmd2_sigma_schedule.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/tests/diffusion/sched/test_dmd2_sigma_schedule.py) |
| 视频/音频使用同一基础调度的不同 shift 版本 | [#5991](https://github.com/vllm-project/vllm-omni/pull/5991) + [`time_request.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/time_request.py) | 已合并/已发布 | CPU 调度参考值测试 |
| 显式 checkpoint 调度数量不匹配会失败 | [`pipeline_minimax_h3.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py) | 已发布 | [`test_minimax_h3_contract.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/tests/diffusion/models/minimax_h3/test_minimax_h3_contract.py) |
| 受支持 Turbo artifact 会映射并绑定到原生 H3 | [#6476](https://github.com/vllm-project/vllm-omni/pull/6476) + [`lora.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/lora.py) | 已合并/已发布 | CPU LoRA 测试 + PR 全模型证据 |
| Turbo 要求五点/四 NFE 与 shift 6/3 | [当前 pipeline](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py) + [固定 recipe](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/recipes/MiniMaxAI/MiniMax-H3.md#turbo-lora) | 已合并/已发布 | CPU 校验 + #6476 端到端证据 |
| Turbo 工作流、本地抽测与可见 LoRA kernel 特征 | [上面的合并 Turbo 工作流](#turbo-usage) + [本地验证](#local-validation) | 已发布工作流与固定环境本地验证；不是普遍结论 | 充分预热的 n=3 A/B + 四设备 nsys，并保留 CUDA graph 覆盖 caveat |
| 模型声明/通用化 LoRA runtime | [#6473](https://github.com/vllm-project/vllm-omni/pull/6473) / [#6017](https://github.com/vllm-project/vllm-omni/pull/6017) | Draft/未发布 | 不描述为可用功能 |

## 参考 {#references}

下面的链接是两条已发布路径的源码地图，就像图例把文章里的简写还原成可审阅代码
与证据。

- [Blog 2 规划 issue #40](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/40) · [MiniMax-H3 系列 RFC #37](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/37)
- [PR #5991 — 为 MiniMax-H3 T2VA 增加蒸馏四 NFE sigma 调度支持](https://github.com/vllm-project/vllm-omni/pull/5991)（已合并）
- [PR #6476 — 用 legacy manager 支持 MiniMax-H3 Turbo LoRA](https://github.com/vllm-project/vllm-omni/pull/6476)（已合并）
- [第 1 篇 — MiniMax-H3 模块化 pipeline]({{ site.baseurl }}/zh/2026-08-24-understanding-pr-5720-minimax-h3-modular-pipeline/)
- 固定源码：[`sigma_schedule.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/sched/sigma_schedule.py) · [`time_request.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/time_request.py) · [`pipeline_minimax_h3.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py) · [`denoise_loop.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/denoise_loop.py) · [`lora.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/lora.py)
- [固定 MiniMax-H3 recipe 的 Turbo LoRA 小节](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/recipes/MiniMaxAI/MiniMax-H3.md#turbo-lora)
- [Draft #6473 — 模型声明 LoRA runtime](https://github.com/vllm-project/vllm-omni/pull/6473) · [draft #6017 — 通用化 LoRA 加载/组合](https://github.com/vllm-project/vllm-omni/pull/6017)
