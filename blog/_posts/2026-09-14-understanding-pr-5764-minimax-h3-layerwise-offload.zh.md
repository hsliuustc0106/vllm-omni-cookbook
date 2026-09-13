---
layout: post
title: "在 vLLM-Omni 中服务 MiniMax-H3（4）：layerwise offload——整套模型塞进两张工作站显卡（PR #5764）"
date: 2026-09-14 12:00:00 +0800
author: hsliuustc0106
summary: >-
  PR #5764 让 MiniMax-H3 跑在两张工作站显卡上：DLO 常驻少量 DiT
  block，其余权重连同 51.5 GB encoder 一起从 pinned host memory 流式搬运——
  2× RTX 5090 上 1344×768、50 步视频跑通，单卡采样峰值约 22.6 GiB。
tags: [MiniMax-H3, RTX-5090, DLO]
category: PR Analysis
feature: offloader
lang: zh
pair: /2026-09-14-understanding-pr-5764-minimax-h3-layerwise-offload/
permalink: /zh/2026-09-14-understanding-pr-5764-minimax-h3-layerwise-offload/
usage:
  - label: "Serve · 2× RTX 5090"
    blurb: "TP2 + 20 个常驻 DiT block，1344×768"
    title: "vllm serve · MiniMax-H3 FL2VA 开启 DLO"
    code: |
      CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/MiniMax-H3/FL2VA \
        --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
        --task-type fl2va \
        --num-gpus 2 --tensor-parallel-size 2 --text-encoder-tp-size 2 \
        --usp 1 --ring 1 --vae-patch-parallel-size 2 \
        --vae-parallel-mode tile --vae-use-tiling \
        --enable-distributed-layerwise-offload --dlo-no-use-allgather \
        --dlo-resident-layers 20 --enforce-eager \
        --diffusion-attention-backend CUDNN_ATTN
    note: >-
      内存优先配置。该 shape 的两 rank B300 容量实测每 rank 峰值
      27,726 MiB——上调常驻层数前先在目标卡上复测。
  - label: "Serve · 1× RTX 5090"
    blurb: "单卡，12 个常驻 block"
    title: "vllm serve · 单卡 DLO"
    code: |
      CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/MiniMax-H3/FL2VA \
        --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
        --task-type fl2va \
        --num-gpus 1 --tensor-parallel-size 1 --text-encoder-tp-size 1 \
        --usp 1 --ring 1 --vae-patch-parallel-size 1 \
        --vae-parallel-mode tile --vae-use-tiling \
        --enable-distributed-layerwise-offload --dlo-no-use-allgather \
        --dlo-resident-layers 12 --enforce-eager \
        --diffusion-attention-backend CUDNN_ATTN
    note: >-
      同拓扑的 50 步 B300 显存实测峰值 26.50 GiB——容量 proxy，不是消费卡
      延迟结论。
  - label: "Offline · 全任务"
    blurb: "T2VA、FL2VA、Ref2VA 一键脚本"
    title: "run_h3_2gpu_all_tasks.sh"
    code: |
      RUN_ROOT=/path/to/run-root \
      MODEL_ROOT=/path/to/MiniMax-H3 \
      GPU_IDS=0,1 \
      PROFILE=rtx5090 \
      bash examples/offline_inference/minimax_h3/run_h3_2gpu_all_tasks.sh
    note: >-
      PROFILE=rtx4090 选择保守的 24 GB 默认值（1024×576、12 个常驻层）。
      DLO_RESIDENT_LAYERS=N 可覆盖任一 profile。
decisions:
  - when: "手里只有两张 24–32 GB 工作站卡"
    pick: "DLO + --dlo-no-use-allgather"
    why: "TP-local streaming 让每个 rank 只在 host 保留自己的 shard，从不逐 rank 重建完整 DiT block；已验证 shape（1344×768、50 步）在 32 GB 卡上采样峰值约 22.6 GiB。"
  - when: "多个同步副本共享一个快速 P2P 域"
    pick: "DLO AllGather（不加 --dlo-no-use-allgather）"
    why: "collective 重建取代每 rank 的完整 host 拷贝，且已可与 online FP8 组合——见 #6279 文（DP2/SP2 host PSS −39.0%）。"
  - when: "在延迟与 HBM 余量之间调优"
    pick: "--dlo-resident-layers N"
    why: "前 N 个 DiT block 常驻设备；从 20（2×32 GB）或 12（24 GB / 单卡）起步。调高不会降低 host RAM——常驻层保留 pinned CPU master 拷贝。"
  - when: "同一台 host 跑多台独立引擎"
    pick: "Host Weight Runtime"
    why: "共享 final-layout host artifact，而不是每台引擎一份完整拷贝——#6591 文实测两台 TP2 引擎 pair PSS −36.6%。"
  - when: "启动时间敏感"
    pick: "记住 mmap 的门"
    why: "在 mmap 能完成 grouped-QKV 与 fused-MLP 变换之前，H3 在 DLO 下坚持走 ordinary loader——这是本 PR review 里的显式 opt-out。"
  - when: "DP 副本各自服务一个请求"
    pick: "AllGather wave + preflight 检查"
    why: "#5864 在派发前拒绝不兼容的 wave（shape/CFG/steps/LoRA/extra_args）；no-AllGather 路径保持每副本一个请求。"
---

## 摘要 {#tldr}

**PR #5764 是容量特性，不是速度特性：它让 MiniMax-H3——66.3 GB 的 DiT 加
51.5 GB 的 Qwen3-VL encoder，再加视频/音频 VAE——跑在两张工作站显卡上，办法
是把绝大部分权重放在 pinned host memory，一次一层地流式搬上卡。** 想象一位
厨师，操作台（显存）很小，但旁边就有一间步入式冷库（host RAM）：他不必把
所有食材都摆上操作台，只需要拿出当前步骤要用的那几样，再把永远常用的几样
留在手边。

| Profile | GPUs | Starting shape | Resident DiT blocks | Attention | Validation |
|---|---:|---:|---:|---|---|
| `rtx5090` | 2 × 32 GB | 1344×768 | 20 | `CUDNN_ATTN` | 目标硬件实测 |
| `rtx4090` | 2 × 24 GB | 1024×576 | 12 | `CUDNN_ATTN` | 容量 proxy 起点值 |

在 2 × RTX 5090 上，一个完整的 50 步 T2VA 请求（1344×768）以客户端计时
**8 分 38 秒**跑完，每张 GPU 采样峰值约 **22.6 GiB**，H.264 + 32 kHz 立体声
AAC 输出通过完整 `ffmpeg` 解码校验。这就是这笔交易的价码：视频能装下、结果
正确，但容量是用带宽换来的——这是一个内存优先配置；也正因如此，这个 PR 在
review 历史里被从 perf 改名成了 feature。

## 背景 {#background}

**这一节要说明的是：在此之前 MiniMax-H3 只能是数据中心显卡的模型——因为无
论走哪条已有路径，权重都放不进工作站卡。** 如果把常驻路径比作"租下一整间仓
库、把所有货一次性上架"，把纯 sequence parallelism 比作"给同一间仓库多雇几
个人手"，那么当每个人的背包（24–32 GB HBM）比库存还小时，两条路都救不了你。

已有的选项各有各的墙：

- **常驻执行**把整个模型装进 HBM。MiniMax-H3 光 DiT（66.3 GB）就超过一张
  32 GB 的卡。
- **纯 Ulysses sequence parallelism** 把 activation 切到各 rank，但权重在每个
  rank 上仍是复制的——容量问题原封不动。
- **Distributed layerwise offload（DLO）已经存在**，但它围绕"block 等大"的
  transformer 和 AllGather 重建路径构建。MiniMax-H3 两个假设都打破：它的
  `token_refiner.blocks` 混着 ~1231 MB 和 ~239 MB 两种大小，51.5 GB 的 text
  encoder 还需要自己的 staging 方案。

PR #5764（[2026-08-06 合并](https://github.com/vllm-project/vllm-omni/pull/5764)，commit
[`1c2a81f`](https://github.com/vllm-project/vllm-omni/commit/1c2a81f6d84aea4fff53bd2f894c2a287c237245)）
就是让 DLO 认识 MiniMax-H3 的那个 PR。同一窗口还有两个兄弟修复——#5802 修
不等大 block 的崩溃，#5864 修 DP 并发请求的正确性——本文把三者当作一个故事
讲。

## PR #5764 改了什么 {#key-changes}

**这个 PR 教会 offloader 三件事：把挑中的少数层留在设备上、只流式搬运每个
rank 自己的 shard、按需 stage 非 DiT 的巨型组件。** 每一件事都对应后文 serve
命令里看得见的一个开关。

![MiniMax-H3 DLO 布局：host pinned memory 流式输出 DiT shard、encoder 与 VAE block；每张 GPU 常驻若干 block 外加两个轮换的 stream slot]({{ site.baseurl }}/assets/figures/pr-5764-minimax-h3-layerwise-offload/fig1-dlo-layout.svg)

### 显式模块常驻 {#module-residency}

`--dlo-resident-layers N` 让前 N 个 DiT block 整个请求期间常驻设备，而不是
每步都流式搬运。哪些路径算"可以 pin 的 DiT block"由模型通过新的
`resident_dit_paths` 字段在
[`OffloadPlan`](https://github.com/vllm-project/vllm-omni/blob/1c2a81f6d84aea4fff53bd2f894c2a287c237245/vllm_omni/diffusion/offloader/offload_plan.py)
里声明——消费级显卡的调优开关因此不会误伤辅助或双 DiT。新的
[`module_residency.py`](https://github.com/vllm-project/vllm-omni/blob/1c2a81f6d84aea4fff53bd2f894c2a287c237245/vllm_omni/diffusion/offloader/module_residency.py)
提供 `PinnedModuleStager`（基于不可变 pinned CPU snapshot stage 模块，不把
设备权重拷回 CPU）和 `PinnedResidentLayerGroup`。reviewer 要求并落地了两个
防御：设了 `--dlo-resident-layers` 但模型没有声明 resident 路径时给出警告；
空 block 列表时让不支持模型的 DiT 整体保持常驻，而不是对着空列表注册 hook。

### 不走 AllGather 的 TP-local streaming {#tp-local-streaming}

AllGather 模式会在每个 rank 上重建*完整*的 DiT block——每个 rank 在 host 持
有 shard，由 collective 拼出整层。对工作站上的两张消费卡，本 PR 改为直接流
式搬运每个 rank 的 tensor-parallel shard（`--dlo-no-use-allgather`）：没有
整块重建、没有 lockstep collective，rank 保留 rank-local host 拷贝。RTX
recipe 用的就是这一模式。塑造它的同一场 review 还把 DP 并发限制在 AllGather
路径上，因为 no-AllGather 的 forward 每个副本只接受一个 prompt。

### encoder 与 VAE 的 staging {#staging}

51.5 GB 的 Qwen3-VL encoder 和 VAE decoder 采用 staging 而非常驻，H3 必需的
encoder 则留在设备上。review 过程中，这部分从模型专属 hook 重构成了
offloader 直接读的声明式 plan：`OffloadPlan.encoder_block_attrs` 把 encoder
路径映射到 rank-local block 列表（用普通 layerwise hook 流式搬运，绝不进
DiT AllGather 组），`on_demand_component_paths` 让 CPU staging 按模型
opt-in。消费级路径还获得了跨两卡的 VAE patch parallelism（tile 解码）与
cuDNN attention，并给 prefix-KV 快路径加了守卫，让 ring attention 永远拿到
显式 mask。

### reviewer 改变了什么 {#review-story}

三个 review 决定值得记住，因为它们解释了代码今天的形状：

- **H3 显式留在 ordinary loader。** 一条 P1 review 指出 mmap 路径对
  MiniMax-H3 的 grouped-QKV 权重变换没有 producer——DLO+AllGather 下会装上
  原始 checkpoint 权重。作者没有依赖"恰好缺个 producer"，而是加上显式的
  `_supports_mmap_loading` opt-out，让 loader 与 backend 共用同一道安全门，
  直到 transformed mmap loading 出现。
- **DLO 的 CLI 字段要在 deploy 路径里活下来。** `--dlo-resident-layers`
  等旗标在 registered/deploy-config pipeline 里会被静默丢弃；修复把所有
  DLO 字段加进 `StageDeployConfig`。
- **PR 从 perf 改名为 feature。** reviewer 指出证据只有"旧 commit 上的一次
  未预热运行加一个 B300 proxy"——没有 main-vs-PR 的延迟或质量对比——要求要么
  补真实 benchmark，要么诚实改名。作者选择了改名。

## 同一周的两个正确性修复 {#bugfixes}

**两个修复回答的是同一个问题：真实模型撞上 DLO 的简化假设时，什么会碎？**
假设是"所有 transformer block 等大"和"一次一个请求就够了"——MiniMax-H3 把
两个都打破了。

[PR #5802](https://github.com/vllm-project/vllm-omni/pull/5802)（2026-08-05
合并）修复不等大 block 下的 AllGather 崩溃。两个共享 GPU buffer 按所有组中
*最大*的 block 分配，而 `prefetch_layer` 把整个 max-size buffer 交给
`all_gather_into_tensor`，输入却是*当前*（更小的）block 的 shard——对任何
更小的 block，`output.numel() != dp_size * input.numel()`，契约检查在
`enable()` 的第一次 prefetch 就失败。修复只是一刀切片——使用早已算好的逐
block AllGather 输出大小——而逼出它的复现用例是 8× 昇腾 NPU、USP=8 的
MiniMax-H3 FL2VA。

[PR #5864](https://github.com/vllm-project/vllm-omni/pull/5864)（2026-08-08
合并）让 DP 并发请求真正可用。DLO+AllGather 下最多 DP-size 个请求组成一个
wave——但多进程结果路径假设只有一个 rank-0 结果队列，一个非法或控制流不兼容
的请求可能在某个副本上失败，而另一个副本已进入 AllGather，wave 就挂死。修复
加入了 wave *preflight*（shape、CFG、denoising 步数、LoRA、以 canonical
signature 比较的 `extra_args`、非空 prompt），在 worker 派发前拒绝整个
wave；结果改走广播 message queue，用同一个 `wave_id` 标记所有 rank。
`extra_args` 检查比看上去更重要：#5764 的第一版读了错误的属性（普通请求恒为
`None`），不同的 pipeline 专属 schedule 能溜过守卫，让 DP rank 进入不同的
AllGather 序列——那是挂死，连报错都没有。

## 实测影响 {#measured-impact}

**诚实的数字只有一组，而且它是验证，不是 benchmark。** 在 commit `ae6577ea`
处，2 × RTX 5090 上一个完整的 50 步 T2VA 请求：

| Shape | Frames | Client E2E | Sampled peak/GPU | Output |
|---:|---:|---:|---:|---|
| 1344×768 | 124 @ 24 FPS | 8 min 38 s | ~22.6 GiB | H.264 + 32 kHz stereo AAC；完整 `ffmpeg` 解码通过 |

环境：vLLM 0.26.0、vLLM-Omni `0.26.1.dev14+gae6577ea`、PyTorch
2.11.0+cu130。注意事项是结论的一部分：单次端到端运行，不是预热后的多轮
benchmark；显存值是 `nvidia-smi` 采样峰值，不是 CUDA allocator 高水位；也没
有 main-vs-PR 的 A/B 延迟对比——这正是 PR 改名为 feature 的原因。

容量规划数字来自
[RTX 5090 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-5090.md)：

| Resource | Requirement |
|---|---|
| GPU HBM | 每卡 32 GiB（5090）；24 GB profile 用 1024×576 + 12 个常驻 block |
| Checkpoint 存储 | 每分区 135 GiB（`FL2VA` 与 `Ref2VA` 分开；同时只起一个 server） |
| 系统 RAM | 最低 200 GiB，推荐 384 GiB |

两个 proxy 实测锚定了常驻层默认值：单 rank 拓扑（12 常驻）的 50 步 B300 显存
实测峰值 26.50 GiB；两 rank TP2（20 常驻、1344×768、50 步）每 rank 峰值
27,726 MiB。两者都被明确标注为内存/正确性 proxy，不是消费卡延迟结论。一个
反直觉的行为值得知道：调高 `--dlo-resident-layers` 改善延迟，但**不会**降低
host RAM，因为常驻层保留 pinned CPU master 拷贝。

至于 DLO 相对常驻部署在数据中心规模下的代价，本博客最接近的配对测量是
[#6279 online FP8 文]({{ site.baseurl }}/zh/2026-08-19-pr-6279-dlo-online-fp8-allgather/)里的四卡 H100 矩阵。

## 怎么用 {#how-to-use}

{% include usage-cookbook.html modes=page.usage %}

跑 `Ref2VA` 时，先停掉 `FL2VA` server，再用 `Ref2VA` 分区重启同一条命令
——参考视频数量与 prompt 长度会推高 activation 显存，请从一个请求一次开始。
在模块化 pipeline（[Blog 1]({{ site.baseurl }}/zh/2026-08-24-understanding-pr-5720-minimax-h3-modular-pipeline/)）之后，
`--task-type fl2va` / `--task-type ref2va` 保持了这些 recipe 依赖的单分区行为。
完整的容量表与两条 serve 命令见上游
[RTX 5090 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-5090.md)。

一条接口说明，因为这个领域在本 PR 合并之后又动了：offload 家族正在
[RFC #6648](https://github.com/vllm-project/vllm-omni/issues/6648) 下统一。
[#5929](https://github.com/vllm-project/vllm-omni/pull/5929)（2026-09-05 合并）
引入了统一语法——
`--diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"],"layer_options":{"dit":{"weight_transfer":"rank-local","resident_layers":20}}}'`
——`components` 选择搬什么、`mode` 选整模块换入还是逐层流式、`weight_transfer`
选 rank-local 还是 AllGather；上文使用的 legacy DLO 旗标
（`--enable-distributed-layerwise-offload`、`--dlo-no-use-allgather`、
`--dlo-resident-layers`）保留为文档化的 compatibility alias，行为不变。
[#7209](https://github.com/vllm-project/vllm-omni/pull/7209)（2026-09-09 合并）
接着把拓扑解析集中到一个纯函数 `resolve_offload_plan()`——接受同样的配置，
但非法配置现在会在任何组件被搬动之前失败，而不是中途失败。对
MiniMax-H3 的全拓扑双卡 recipe，上游
[user guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md)
仍然指向本文使用的 compatibility 旗标；Host Weight Runtime 目前还不能用新
config 表达——上面的命令与当前上游 recipe 一致。

## 怎么选 {#decision-cards}

{% include decision-cards.html items=page.decisions %}

## 局限与后续 {#limitations}

- **这是容量，不是速度。** 五秒 50 步的视频要 8 分 38 秒，这就是流式的价
  码；降低常驻层数进一步省 HBM、增加 CPU 到 GPU 的传输时间。没有 A/B 延迟
  benchmark。
- **24 GB（`rtx4090`）profile 是容量 proxy**——在 B300 显存实测上验证，没有
  在目标 4090 硬件上验证，起点是 1024×576。
- **host RAM 才是真正的 footprint。** 每分区最低 200 GiB；常驻层保留
  pinned CPU master，这个开关换的是 HBM，不是 host 内存。同一台 host 上的
  多台独立引擎请看
  [Host Weight Runtime]({{ site.baseurl }}/zh/2026-08-26-understanding-pr-6591-host-weight-runtime/)。
- **H3 在 DLO 下没有 mmap 快启动。** 在 checkpoint mmap 能完成 grouped-QKV
  与 fused-MLP 变换之前，H3 走 ordinary loader——启动阶段先物化权重再分片。
- **Eager 执行 + cuDNN attention** 是已验证消费路径的一部分；这套配置刻意
  保守。
- **仅有单次运行验证**，显存是采样（非 allocator）峰值——把上面每个数字当
  边界条件，不要当分布。
- 这是 MiniMax-H3 8 篇优化系列的 Blog 4，由
  [系列 RFC #37](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/37)
  跟踪；它是"offload"主题的核心篇，两翼是
  [Host Weight Runtime 文]({{ site.baseurl }}/zh/2026-08-26-understanding-pr-6591-host-weight-runtime/)（host 内存共享）与
  [online FP8 + DLO 文]({{ site.baseurl }}/zh/2026-08-19-pr-6279-dlo-online-fp8-allgather/)（量化 payload）。
  延伸阅读：[Blog 1——模块化 pipeline]({{ site.baseurl }}/zh/2026-08-24-understanding-pr-5720-minimax-h3-modular-pipeline/)、
  [Blog 2——"四步"的三份契约]({{ site.baseurl }}/zh/2026-08-24-understanding-pr-5991-minimax-h3-few-step-schedules/)、
  [Blog 3——mask-free packed padding]({{ site.baseurl }}/zh/2026-09-12-understanding-pr-6542-trtllm-mask-free-packed-padding/)。

## 参考 {#references}

- [PR #5764 — feat(minimax-h3): enable RTX 4090/5090 support with DLO](https://github.com/vllm-project/vllm-omni/pull/5764)（2026-08-06 合并，commit [`1c2a81f`](https://github.com/vllm-project/vllm-omni/commit/1c2a81f6d84aea4fff53bd2f894c2a287c237245)）
- [PR #5802 — Fix DLO AllGather size mismatch for heterogeneous blocks](https://github.com/vllm-project/vllm-omni/pull/5802)（2026-08-05 合并）
- [PR #5864 — Fix DLO DP concurrent request execution](https://github.com/vllm-project/vllm-omni/pull/5864)（2026-08-08 合并）
- [MiniMax-H3 RTX 5090 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-5090.md)（上游，当前版）
- [MiniMax-H3 recipe 汇总](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md)（上游，当前版）
- [双卡全任务脚本](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/minimax_h3/run_h3_2gpu_all_tasks.sh)（上游，当前版）
- [DLO user guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md)（上游，当前版——同时记录新的 `diffusion_offload_config` 与 compatibility 旗标）
- [RFC #6648 — Unify the offloader protocol and user interface](https://github.com/vllm-project/vllm-omni/issues/6648)（OPEN；J0 = #5929、J1 = #7209 已合并）
- [Host Weight Runtime 文 — PR #6591]({{ site.baseurl }}/zh/2026-08-26-understanding-pr-6591-host-weight-runtime/)
- [Online FP8 with DLO AllGather 文 — PR #6279]({{ site.baseurl }}/zh/2026-08-19-pr-6279-dlo-online-fp8-allgather/)
- [系列 RFC #37 — MiniMax-H3 optimization blog series](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/37)
