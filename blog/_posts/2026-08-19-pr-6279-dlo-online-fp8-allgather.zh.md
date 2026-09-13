---
layout: post
title: "理解 PR #6279：DLO AllGather 路径上的 online FP8"
date: 2026-08-19 12:00:00 +0800
author: hsliuustc0106
summary: >-
  PR #6279 让 per-tensor online FP8 走通 DLO AllGather：finalize 后的 FP8
  权重与 scale 按 physical layout 分片并在设备端重建；启动期 host 内存的
  代价仍然显式存在。
tags: [MiniMax-H3, FP8, H100, DLO]
category: PR Analysis
feature: offloader
lang: zh
pair: /2026-08-19-pr-6279-dlo-online-fp8-allgather/
permalink: /zh/2026-08-19-pr-6279-dlo-online-fp8-allgather/
usage:
  - label: "Serve · DP2/SP2"
    blurb: "四卡，两个两 rank 的 DLO group"
    title: "vllm serve · MiniMax-H3 Ref2VA，online FP8 + DLO AllGather"
    code: |
      MODEL=/path/to/MiniMax-H3/Ref2VA

      CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$MODEL" \
        --omni \
        --task-type ref2va \
        --num-gpus 4 \
        --usp 2 \
        --quantization fp8 \
        --enable-distributed-layerwise-offload
    note: >-
      所有 rank 必须使用相同的 denoising step 数、按 lockstep 进入权重
      collective——一个 scheduler、一个同步 wave。
  - label: "Serve · DP4/SP1"
    blurb: "四卡，一个 DLO group"
    title: "vllm serve · 同一任务改用 --usp 1"
    code: |
      MODEL=/path/to/MiniMax-H3/Ref2VA

      CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$MODEL" \
        --omni \
        --task-type ref2va \
        --num-gpus 4 \
        --usp 1 \
        --quantization fp8 \
        --enable-distributed-layerwise-offload
  - label: "独立副本"
    blurb: "保留 online FP8，去掉 collective"
    title: "vllm serve · online FP8 + rank-local host tensor"
    code: |
      MODEL=/path/to/MiniMax-H3/Ref2VA

      CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL" \
        --omni \
        --task-type ref2va \
        --num-gpus 2 \
        --quantization fp8 \
        --enable-distributed-layerwise-offload \
        --dlo-no-use-allgather
    note: >-
      每个 rank 保留完整的 rank-local host 拷贝而不是 DLO shard——单副本
      host 内存上升，调度保持独立。
decisions:
  - when: "一个同步任务同时想要 online FP8 与 DLO"
    pick: "--quantization fp8 + DLO AllGather"
    why: "finalize 后的 per-tensor FP8 权重与 scale 由 DLO 分片、逐层重建；host PSS 峰值相对 native BF16 下降 39.0%（DP2/SP2）/ 15.0%（DP4/SP1）。"
  - when: "多台副本各自独立调度请求"
    pick: "--dlo-no-use-allgather"
    why: "AllGather 是同步的权重 collective：每个 rank 必须以相同的 block 顺序进入同一个 wave。"
  - when: "fidelity 或裸延迟优先"
    pick: "切换之前先测量"
    why: "配对 H100 实测中 native BF16 更快（9.2% / 2.7%），fidelity 领先约 0.005 SSIM / 3.84 dB PSNR。"
  - when: "启动期 host 内存是硬约束"
    pick: "保持 BF16-mmap，或等 runtime cache"
    why: "每个 rank 在保留 DLO shard 之前，仍要经 ordinary loader 物化 finalize 后的 FP8 模型；cache 契约由 #6231 跟踪。"
---

## 摘要 {#tldr}

**PR #6279 让 per-tensor online FP8 与 DLO 的 AllGather 路径兼容——就像两套
原本互斥的行李安检规则，现在承认对方封好的箱子。** ordinary loader 先把
BF16 checkpoint 变成 finalize 后的 FP8 权重与 scale；DLO 再把这些 runtime
tensor 分片、传输 FP8 shard，并在设备端逐层重建。结果是稳态 host 与 device
内存都低于 native BF16，但启动期仍有一个 ordinary-loader 峰值——因为
direct checkpoint mmap 无法完成 online 方法的量化。

| Precision | Layout | Wave latency | Throughput / device | Peak GPU / device | Host PSS peak |
|---|---|---:|---:|---:|---:|
| Online FP8 | DP2/SP2 | 737.54 s | 0.000678 req/s | 22.11 GiB | 220.94 GiB |
| Native BF16 | DP2/SP2 | 669.98 s | 0.000746 req/s | 24.04 GiB | 362.49 GiB |
| Online FP8 | DP4/SP1 | 1227.72 s | 0.000815 req/s | 21.79 GiB | 221.60 GiB |
| Native BF16 | DP4/SP1 | 1194.88 s | 0.000837 req/s | 23.53 GiB | 260.77 GiB |

这些都是 MiniMax-H3 Ref2VA 在四张 H100 上的单次运行测量；完整表格、RSS
数值、fidelity 指标、命令与注意事项见
[PR validation comment](https://github.com/vllm-project/vllm-omni/pull/6279#issuecomment-5328282759)。

## 背景 {#background}

Diffusion transformer 大到"每张 GPU 都驻留全部层"本身就可能成为瓶颈资源。
DLO 解决的是这个容量问题：权重常驻 host，每个时刻只把当前层搬进两个可复用
的 device buffer 之一。开启 AllGather 后，每个 rank 只保留 host 权重的一个
shard，在 block 运行前通过 collective 重建完整的一层。

Online FP8 是天然搭档：BF16 checkpoint 仍是唯一事实来源，但符合条件的
linear 权重在加载时被量化，serving 模型持有 FP8 权重加 scale。在 #6279
之前，DLO AllGather 的 gate 会拒绝所有 online quantization 方法。安全的
direct-mmap 路径无法"先映射 BF16 tensor、之后再量化"，因为最终 runtime
表示可能改变 dtype、shape、packing 与 stride。

实际症状是一个硬性互斥：用户必须在 online FP8 与 DLO AllGather 拓扑之间
二选一——即使 finalize 后的 per-tensor FP8 表示早已满足 DLO 的重建契约。

## 这个 PR 做了什么 {#what-the-pr-does}

新路径刻意收窄：

```text
BF16 checkpoint on disk
        │
        ▼
ordinary loader + online quantizer
        │
        ▼
FP8 weight + scale
        │
        ▼
1 / DLO-group-size host shard per rank
        │  H2D copy stream
        ▼
AllGather communication stream
        │
        ▼
full FP8 layer in one of two GPU slots
```

BF16 checkpoint 是量化的输入，不是 DLO AllGather 搬运的 payload。配套的
scale 跟随 FP8 runtime tensor，一起记录在重建 metadata 里。

<iframe
  src="{{ site.baseurl }}/assets/figures/pr-6279-dlo-online-fp8/dlo-online-fp8-allgather.html"
  title="DLO AllGather online FP8 交互式图解"
  loading="lazy"
  style="display:block;width:100%;height:720px;border:1px solid #d0d7de;border-radius:8px;margin:16px 0;">
</iframe>

交互式走查可以在 DP2/SP2 与 DP4/SP1 之间切换，逐步演示 source 加载、量化、
host 分片、H2D、AllGather、double-buffered compute、释放与复用；它还区分了
持久存在的 BF16 文件、瞬态 BF16 staging 与被保留的 FP8 DLO payload。

## 关键改动 {#key-changes}

### 只放行一种经过验证的 online 方法 {#allow-one-validated-online-method}

[`diffusers_loader.py`](https://github.com/vllm-project/vllm-omni/blob/284e05c88b7b46be9fae6d822bf22075840cbfbb/vllm_omni/diffusion/model_loader/diffusers_loader.py)
现在会检查实际的 online quantization 方法。只有
`Fp8PerTensorOnlineLinearMethod` 被放行进入 DLO AllGather；其余 online
方法保持 fail-closed，直到它们的 runtime layout 通过验证。ordinary
loader 仍然先运行，因此量化与 scale 生成会在 DLO 开始分片之前完成。

这保留了一条有用的安全边界：放行一种已知 layout，不等于假设每个 online
quantizer 都产出 DLO 兼容的 tensor。

### 保留 FP8 的 physical layout {#preserve-the-physical-fp8-layout}

Online Cutlass FP8 权重可能是转置过的、非连续的 view。如果按 logical
order 打平，就会悄悄改变 scaled matmul kernel 期望的布局。因此 DLO 记录
每个 runtime tensor 的 dtype、shape、stride 与 physical offset，必要时按
物理存储顺序打包，并在 AllGather 之后重建出记录的那个 view。

相关 backend 逻辑位于
[`distributed_layerwise_backend.py`](https://github.com/vllm-project/vllm-omni/blob/284e05c88b7b46be9fae6d822bf22075840cbfbb/vllm_omni/diffusion/offloader/distributed_layerwise_backend.py)。
回归测试
[`test_allgather_reconstructs_online_fp8_weight_and_scale`](https://github.com/vllm-project/vllm-omni/blob/284e05c88b7b46be9fae6d822bf22075840cbfbb/tests/diffusion/offloader/test_distributed_layerwise_backend.py)
检查 finalize 后的 FP8 权重、scale、dtype、shape 与转置 stride。

### collective 语义保持不变 {#keep-the-collective-semantics-unchanged}

这个 PR 没有增加请求期的 collective。AllGather 仍然是与请求无关的权重
操作：所有参与 rank 必须以相同的 block 顺序进入同一个 wave。两个 device
buffer 依然交替使用，compute stream 运行当前 FP8 block 的同时，copy 与
communication stream 在准备下一个。

## 实测影响 {#measured-impact}

验证 workload 是 MiniMax-H3 Ref2VA、四张 H100、vLLM 0.27.0、CUDA 12.9、
1344×768 输出、24 FPS、五秒视频、seed 0、50 个 denoising step。当前 PR
head 与冻结的 native-BF16 baseline 对比；DP2/SP2 与 DP4/SP1 使用按 DP
组大小对齐的并发 wave。每设备吞吐 = 完成请求数 ÷ wave 墙钟时间 ÷ 4。

| Precision | Configuration | Wave latency | Throughput / device | Peak GPU / device | Host RSS peak | Host PSS peak | SSIM / PSNR |
|---|---|---:|---:|---:|---:|---:|---:|
| Online FP8 | DP2/SP2 | 737.54 s | 0.000678 req/s | 22.11 GiB | 239.45 GiB | 220.94 GiB | 0.975421–0.975424 / 38.8108–38.8121 dB |
| Native BF16 | DP2/SP2 | 669.98 s | 0.000746 req/s | 24.04 GiB | 377.33 GiB | 362.49 GiB | 0.980584–0.980589 / 42.6535–42.6550 dB |
| Online FP8 | DP4/SP1 | 1227.72 s | 0.000815 req/s | 21.79 GiB | 285.73 GiB | 221.60 GiB | 0.975421–0.975424 / 38.8108–38.8121 dB |
| Native BF16 | DP4/SP1 | 1194.88 s | 0.000837 req/s | 23.53 GiB | 293.39 GiB | 260.77 GiB | 0.980584–0.980589 / 42.6535–42.6550 dB |

三条结论比任何一行数字更重要：

- 本次运行中 native BF16 更快：DP2/SP2 快 9.2%，DP4/SP1 快 2.7%。DLO
  通信与 online-loader 路径意味着"权重字节减半"不会自动让每个 diffusion
  GEMM 变快。
- FP8 的 host PSS 低于 BF16：DP2/SP2 低 39.0%，DP4/SP1 低 15.0%。DP2 差距
  更大，是因为该拓扑下 DLO group 在 host 侧保留两份 DiT 拷贝，BF16
  payload 复制得更重。
- 在这组配对对比中 BF16 的 fidelity 更高，约 0.005 SSIM、3.84 dB PSNR。
  Online FP8 达到的是 runtime 兼容，不是数值等价或生产级 quality parity
  的声明。

Host RSS 是 API server、四个 diffusion worker 与 resource tracker 之和的
峰值，按 1 Hz 采样。RSS 对共享 page 每个 mapping 计一次；PSS 是物理 host
内存更好的估计。所有输出都通过了视频与音频 metadata 校验：1344×768、
24 FPS、124 帧、32-kHz 立体声 AAC。

### cache 后续实验改变了什么 {#what-the-cache-follow-up-changes}

下面这个只看内存的后续实验，是一个探索性的 Phase-I normalized-FP8 cache
测量，不是对 #6279 的第二次 benchmark。它回答的是"为什么 #6279 之后仍然
需要 cache"：cache 命中可以跳过 cached transformer 的 ordinary online
quantization 物化过程，但 partial cache 不保证整个进程的 PSS 更低。

| Path | DP2/SP2 request PSS | DP4/SP1 request PSS |
|---|---:|---:|
| Ordinary online FP8 | 188.71 GiB | 145.07 GiB |
| Normalized FP8 cache hit | 213.33 GiB | 171.85 GiB |
| Native BF16 | 318.48 GiB | 222.93 GiB |

在这个原型里，只有 transformer 被规范化进 FP8 cache；text encoder 等组件
仍是 BF16。cache 的 file-backed page 同样计入 RSS/PSS。诚实的结论是：

1. #6279 把 finalize 后的 runtime payload 从 BF16 降到 FP8，并验证了
   AllGather 重建路径。
2. cache 命中可以为被缓存的组件移除完整的 BF16/online-quantization 启动
   路径。
3. 端到端的 host 内存收益，需要其余大组件的 cache/量化覆盖一致，以及口径
   一致的 PSS 记账。

完整的内存分解与原始数值记录在
[follow-up PR comment](https://github.com/vllm-project/vllm-omni/pull/6279#issuecomment-5337093776)。

## 怎么用 {#how-to-use-it}

默认的 DLO AllGather 路径现在接受 per-tensor online FP8：

```bash
MODEL=/path/to/MiniMax-H3/Ref2VA

# Four GPUs, DP2/SP2: two DLO groups, two ranks per group.
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$MODEL" \
  --omni \
  --task-type ref2va \
  --num-gpus 4 \
  --usp 2 \
  --quantization fp8 \
  --enable-distributed-layerwise-offload
```

同样四张卡跑 DP4/SP1 时改用 `--usp 1`。DLO 的 AllGather 默认随
distributed-layerwise-offload 配置开启。MiniMax-H3 recipe 更详细地记录了
组件范围与 ordinary-loader 启动权衡：

[MiniMax-H3 online FP8 recipe](https://github.com/vllm-project/vllm-omni/blob/284e05c88b7b46be9fae6d822bf22075840cbfbb/recipes/MiniMaxAI/MiniMax-H3.md)

如果独立副本无法遵守同步 request-wave 契约，使用
`--dlo-no-use-allgather`。它保留 online FP8 runtime 路径，但改用
rank-local host tensor 而不是 DLO 的分片 collective 路径。

## 局限与后续 {#limitations--follow-ups}

- **启动内存仍是主要缺口。** Direct checkpoint mmap 无法在线量化 BF16
  tensor。每个 rank 会临时运行 ordinary loader、物化 finalize 后的 FP8
  模型，然后才保留自己的 DLO shard。
- **allowlist 刻意收窄。** 其他 online quantizer（包括未通过验证的
  block/group layout）保持 fail-closed。
- **AllGather 要求同步 wave。** 各 rank 必须使用相同的 denoising step 数，
  并按 lockstep 进入权重 collective。独立调度的副本请选择 no-AllGather。
- **DLO AllGather 不是 TP group。** TP-aware 与 HSDP layout 有各自的兼容
  边界；HSDP 加 DLO AllGather 会被拒绝，以避免二次分片。
- **Phase-I cache 工作属于后续范围。** runtime-cache 兼容契约由
  [#6231](https://github.com/vllm-project/vllm-omni/issues/6231) 跟踪，
  包括 transformed layout、cache miss、source fingerprint 与跨 DP/SP
  复用。
- **质量评估需要更大范围。** 上面的测量是"一个五秒、50-step 请求矩阵"。
  它们确立的是系统行为与该 workload 相对 BF16 的 fidelity，不是普遍的
  prompt 对齐或人评质量 parity。

## 参考 {#references}

- [PR #6279 — Support online FP8 with DLO AllGather](https://github.com/vllm-project/vllm-omni/pull/6279)
- [PR #6279 benchmark comment](https://github.com/vllm-project/vllm-omni/pull/6279#issuecomment-5328282759)
- [PR #6279 host-memory follow-up](https://github.com/vllm-project/vllm-omni/pull/6279#issuecomment-5337093776)
- [DLO user guide](https://github.com/vllm-project/vllm-omni/blob/284e05c88b7b46be9fae6d822bf22075840cbfbb/docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md)
- [FP8 quantization guide](https://github.com/vllm-project/vllm-omni/blob/284e05c88b7b46be9fae6d822bf22075840cbfbb/docs/user_guide/quantization/fp8.md)
- [RFC #6231 — DLO runtime-cache compatibility](https://github.com/vllm-project/vllm-omni/issues/6231)
- [Online quantization deep dive]({{ site.baseurl }}/2026-08-18-online-quantization-fp8/)
