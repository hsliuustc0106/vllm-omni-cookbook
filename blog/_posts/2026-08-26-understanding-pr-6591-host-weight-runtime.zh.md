---
layout: post
title: "理解 PR #6591：Host Weight Runtime——一套 TP2 权重，多台独立引擎"
date: 2026-08-26 12:00:00 +0800
author: hsliuustc0106
summary: >-
  Host Weight Runtime 让多台独立 TP2 DLO 引擎共享两份 final-layout host
  artifact。4×H200 实测 pair PSS 从 388.06 降至 245.98 GiB，CPU copy
  rank-work 减少 58.4%。
tags: [MiniMax-H3, H200, DLO]
category: PR Analysis
feature: offloader
lang: zh
pair: /2026-08-26-understanding-pr-6591-host-weight-runtime/
permalink: /zh/2026-08-26-understanding-pr-6591-host-weight-runtime/
usage:
  - label: "填充 · preferred"
    blurb: "第一组匹配的 TP2 cohort"
    title: "MiniMax-H3 · 生成两份 TP-coordinate artifact"
    code: |
      export MODEL=/path/to/MiniMax-H3/FL2VA
      export HWR_ROOT=/var/cache/vllm-omni/hwr

      CUDA_VISIBLE_DEVICES=0,1 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      vllm serve "${MODEL}" --omni \
        --host 0.0.0.0 --port 8000 --trust-remote-code \
        --task-type fl2va \
        --num-gpus 2 --tensor-parallel-size 2 \
        --usp 1 --ring 1 \
        --text-encoder-tp-size 2 \
        --vae-patch-parallel-size 2 \
        --vae-parallel-mode tile --vae-use-tiling \
        --enable-distributed-layerwise-offload \
        --dlo-no-use-allgather --dlo-resident-layers 20 \
        --host-weight-runtime-mode preferred \
        --host-weight-runtime-root "${HWR_ROOT}" \
        --dlo-host-registration-limit-gib 80 \
        --enforce-eager \
        --diffusion-attention-backend CUDNN_ATTN
    note: >-
      等待服务健康。artifact 在启动阶段发布，不需要先发推理请求。严格 prewarm
      流程会在切到 required 前干净关闭这组 cohort。
  - label: "共享 · 第二台引擎"
    blurb: "同一 root，独立请求"
    title: "MiniMax-H3 · Engine B 复用同一组 TP0/TP1 artifact"
    code: |
      export MODEL=/path/to/MiniMax-H3/FL2VA
      export HWR_ROOT=/var/cache/vllm-omni/hwr

      CUDA_VISIBLE_DEVICES=2,3 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      vllm serve "${MODEL}" --omni \
        --host 0.0.0.0 --port 8001 --trust-remote-code \
        --task-type fl2va \
        --num-gpus 2 --tensor-parallel-size 2 \
        --usp 1 --ring 1 \
        --text-encoder-tp-size 2 \
        --vae-patch-parallel-size 2 \
        --vae-parallel-mode tile --vae-use-tiling \
        --enable-distributed-layerwise-offload \
        --dlo-no-use-allgather --dlo-resident-layers 20 \
        --host-weight-runtime-mode preferred \
        --host-weight-runtime-root "${HWR_ROOT}" \
        --dlo-host-registration-limit-gib 80 \
        --enforce-eager \
        --diffusion-attention-backend CUDNN_ATTN
    note: >-
      两台引擎共享 artifact 文件和 OS page cache，不共享 process group。
      如需路由请求，请使用外部 router；HWR 本身不提供 router。
  - label: "强制命中 · required"
    blurb: "miss 时失败，不现场重建"
    title: "MiniMax-H3 · 要求预先填充的精确 identity"
    code: |
      # Use the same model revision, TP/SP layout, root, and serving flags as
      # the preferred producer cohort; change only the policy.
      vllm serve /path/to/MiniMax-H3/FL2VA --omni \
        --num-gpus 2 --tensor-parallel-size 2 \
        --text-encoder-tp-size 2 \
        --vae-patch-parallel-size 2 \
        --vae-parallel-mode tile --vae-use-tiling \
        --enable-distributed-layerwise-offload \
        --dlo-no-use-allgather --dlo-resident-layers 20 \
        --host-weight-runtime-mode required \
        --host-weight-runtime-root /var/cache/vllm-omni/hwr \
        --dlo-host-registration-limit-gib 80
    note: >-
      required 只消费现有 artifact。空 store、损坏 artifact 或 semantic
      identity 不匹配都会让启动失败，而不是回到 canonical loader。
decisions:
  - when: "同一节点运行多台独立 TP2 引擎"
    pick: "no-AllGather + HWR preferred"
    why: "相同 TP coordinate 映射同一 final-layout artifact，不需要把多台引擎合并成一个调度与故障域。"
  - when: "一个同步 DP×TP job 位于快速 P2P 域"
    pick: "考虑 DLO AllGather"
    why: "这些 rank 本来就执行同一套 collective 序列，持久化 host shard 可能比 final-layout cache 更简单。"
  - when: "TP1 checkpoint mmap 已兼容"
    pick: "先保持 HWR disabled"
    why: "TP1 control 启动快 96.96 s；HWR 不是天然更快的 checkpoint loader。"
  - when: "rollout 必须证明每次都是 cache hit"
    pick: "先 preferred 填充，再切 required"
    why: "required 不会填充空 store，陈旧或不兼容 identity 会在启动时显式失败。"
  - when: "registration 不支持或超出 budget"
    pick: "保留 HWR，接受 bounded staging"
    why: "file-backed sharing 仍然成立，只是每个 streamed block 会重新出现 mmap-to-pinned CPU copy。"
  - when: "online quantization、HSDP 或 dynamic LoRA 改变权重"
    pick: "使用当前 canonical path"
    why: "已发布 consumer 是 MiniMax-H3 BF16 no-AllGather 的精确 final-layout path，不是任意 transformed-weight cache。"
---

## 摘要 {#tldr}

**Host Weight Runtime（HWR）让多台独立引擎复用同一套精确的 runtime-ready
CPU 权重——就像两间厨房共用一个封存完好的中央仓库，而不是各自再租一间库房。**
对于两台 MiniMax-H3 `DP1×TP2` 引擎，仓库里是两份 artifact，每个 TP
coordinate 一份；两台引擎的请求节奏仍然互不约束。

在四张 H200 上，registered HWR 把 pair 的 proportional set size（PSS）从
**388.06 降到 245.98 GiB**，让第二台 TP2 引擎快 **25.73 s** ready，并且
相对 bounded-staging fallback 减少 **58.4%** 的 profiled CPU copy
rank-work。H2D payload、HBM、compute 与 NVLink traffic 基本不变。

| Decision metric | Control | Registered HWR | Result |
|---|---:|---:|---:|
| Pair PSS | 388.06 GiB existing path | 245.98 GiB | **−142.08 GiB (−36.6%)** |
| Warm second-engine startup, n=3 | 176.18 s HWR-disabled | 150.45 s | **−25.73 s (−14.6%)** |
| Pair wall, n=3 | 34.61 s HWR staged | 28.86 s | −16.6% mean; directional |
| Profile CPU `aten::copy_` | 33.91 s staged | 14.10 s | **−19.82 s (−58.4%)** |
| H2D / NVLink / peak HBM | reference | unchanged | transport bytes did not move |

> [!IMPORTANT]
> HWR 默认关闭，而且收益依赖 workload。配套 TP1 实验中，direct checkpoint
> mmap 的启动更快；TP2 latency 的保守区间也因第一轮 staged outlier 跨过零。
> 因此强结论是共享、startup、PSS、CPU-copy attribution 与稳态输出一致，
> 不是“所有请求都更快”。

## 背景 {#background}

**用户最先看到的是 host memory 随独立引擎数量增长——就像每增加一辆配送车，
就把同一批货再复制到一座新仓库。** DLO 只在 HBM 里轮换两个 block buffer，
所以 device memory 能装下；host 侧却可能因为每个 worker 都保留 private
runtime weights 而先触顶。

[PR #6213](https://github.com/vllm-project/vllm-omni/pull/6213) 已经解决了
compatible TP1 checkpoint 的第一阶段问题：loader 保留只读 checkpoint
`mmap` view，多个 worker 通过 OS page cache 共用物理文件页。这个路径面向
**checkpoint layout**，model-specific transform 可以延迟到 block packing。

TP2 不一样。ordinary loader 会把 tensor 切分并转换为 rank-local final
layout。TP rank 0 与 rank 1 拥有不同 bytes，而且它们未必能直接指向 raw
checkpoint。HWR 之前，两台独立 TP2 引擎等价于：

```text
Engine A: private TP0 runtime weights + private TP1 runtime weights
Engine B: private TP0 runtime weights + private TP1 runtime weights
```

为什么不把两台引擎直接放进一个 DLO AllGather group？因为 AllGather 要求
成员按相同逻辑顺序发出同一个 weight collective。两台接收不同请求的引擎，
可能分别处在不同 denoising step、不同 block，或者一台空闲。NVLink 能让
**已经匹配的 collective** 更快，却不能把独立 schedule 变成同一个 schedule。
replica routing 应由 vLLM Router、Dynamo 或 operator 负责，而不是 offloader。

[HWR RFC #6414](https://github.com/vllm-project/vllm-omni/issues/6414)
因此把问题拆成三层：loader 需要哪种精确 representation、runtime 去哪里取得、
以及 DLO 如何搬到 device。

## Host Weight Runtime 是什么 {#mental-model}

**HWR 是一套有版本的目录与仓库契约，不是 GPU copy engine——像图书目录保证拿到
正确版本，而快递系统另行决定怎么运输。** loader 定义 semantic identity；
store 发布并校验 immutable file；lease 负责进程内 mapping；DLO 负责
registration、staging、H2D stream 与 teardown。

![两台独立 TP2 引擎按相同 TP coordinate 映射同一 HWR artifact，同时保持独立请求 schedule]({{ site.baseurl }}/assets/figures/host-weight-runtime/fig1-architecture.svg)

[PR #6419](https://github.com/vllm-project/vllm-omni/pull/6419) 落下四个
核心 contract：

- `HostWeightStore`：exact lookup、one-builder coordination、hash、validation、
  atomic publication、quarantine 与文件生命周期。
- `HostWeightLease`：一个 consumer process 的稳定 tensor view、mapped range、
  file descriptor 与 shared artifact lock。
- `WeightProducer`：通过 store-scoped writer 生成一种声明过的 final
  representation。
- `WeightRestorer`：先做无 mutation 的 plan，再且只再 commit 一次 model
  rebinding。

artifact identity 包含 immutable source fingerprint、component ownership、
representation、mixed-precision policy、producer/restorer ABI 和会影响 bytes 的
parallel layout。replicated weight 的 DP rank 被排除；会改变 tensor bytes 的 TP
size/rank 被纳入；SP semantics 改变时也会被 guard。registration policy 与
device ID 不属于 identity，因为它们只是搬运同一份 representation。

本次 TP2 实验生成两份 30.86-GiB artifact：

```text
artifact(tp_rank=0) ← Engine A rank 0 + Engine B rank 0
artifact(tp_rank=1) ← Engine A rank 1 + Engine B rank 1
```

每个进程仍有自己的 virtual mapping 与 lease。共享的是 kernel 里的物理
file-backed page，不是一个跨进程 Python tensor，也不是一份跨进程 CUDA
registration。

## 这组 PR 分别做了什么 {#key-changes}

**这个特性不是一个巨型 cache patch，而是分开验收仓库、目录、装配线与运输车。**
这种拆分让 storage semantics 可复用，也避免 CUDA 或 DLO 反过来成为 model
identity 的 owner。

| PR | Responsibility | What it deliberately does not own |
|---|---|---|
| [#6419](https://github.com/vllm-project/vllm-omni/pull/6419) | Neutral runtime, store, lease, exact identity, atomic local filesystem lifecycle | Diffusion, BF16 policy, CUDA, H2D |
| [#6427](https://github.com/vllm-project/vllm-omni/pull/6427) | Explicit post-load publication for producers that need a finalized canonical model | Restoring or mutating the model serving the current cold startup |
| [#6445](https://github.com/vllm-project/vllm-omni/pull/6445) | MiniMax-H3 final-layout BF16 producer/restorer and TP/SP semantic identity | Loader activation and transport |
| [#6486](https://github.com/vllm-project/vllm-omni/pull/6486) | `disabled`/`preferred`/`required`, warm restore, ordinary-DiT skip, transactional lease handoff, bounded staging | CUDA registration |
| [#6591](https://github.com/vllm-project/vllm-omni/pull/6591) | Registered shared mmap → rotating HBM buffers, rollback, ordered teardown, writer/fsync/source-digest optimizations | AllGather, checkpoint-mmap registration, request orchestration |

当 DLO disabled、HWR disabled 或 AllGather enabled 时，loader 会在 identity 和
filesystem work 之前直接挡住 HWR：

```python
# host_weight_loader.py — zero-interaction precedence
if mode is RuntimeMode.DISABLED or not dist_offload or use_allgather:
    return None
```

见固定版本的
[`host_weight_loader.py`](https://github.com/vllm-project/vllm-omni/blob/497c537c6f70e44f376b491bf7b50395cf2cba5d/vllm_omni/diffusion/model_loader/host_weight_loader.py#L103-L151)。
eligible warm hit 会 plan 并 commit 精确 restore；`preferred` miss 走 canonical
loader 并给下次启动发布；`required` miss 直接失败。

transport 在更后面才选择：

```python
# distributed_layerwise_backend.py — same lease, two safe transfer outcomes
self._using_registered_mmap = self._try_register_hwr_mmap(source_tensors)
hook.registered_mmap = self._using_registered_mmap
```

完整 registration 与 unregister-before-lease-close 逻辑在
[`distributed_layerwise_backend.py`](https://github.com/vllm-project/vllm-omni/blob/497c537c6f70e44f376b491bf7b50395cf2cba5d/vllm_omni/diffusion/offloader/distributed_layerwise_backend.py#L1039-L1150)。

## 数据如何移动 {#dataflow}

**Registered mmap 去掉了一次反复发生的 host relay——像卡车直接从共享月台装货，
不再让每个 worker 先把货搬进自己的中转间。** 卡车仍把同样的 bytes 送入同一
HBM buffer，GPU kernel 仍然读取 HBM，而不是直接读取 host memory。

bounded fallback：

```text
shared read-only final-layout mmap
  → CPU copy / pack into two private pinned staging slots
  → H2D into two rotating HBM block buffers
  → GPU kernel
```

registered path：

```text
shared read-only final-layout mmap registered with CUDA
  → H2D into the same two rotating HBM block buffers
  → GPU kernel
```

这不是 zero-copy GPU execution。registration 会在 worker 生命周期内
page-lock 完整 mapping，让 asynchronous H2D 直接使用这些页；它不会消除 H2D
payload，不会删掉 HBM buffer，也不会删除 ordinary TP collective。

registration 是 all-or-nothing。read-only registration 不支持、正数 budget
小于完整 page-aligned mapping，或者能安全 rollback 的 registration error，都会
选择 bounded staging。如果 rollback/unregistration 不能安全释放 platform
ownership，则启动或 teardown 会失败，而不是关闭仍被 CUDA 持有的 mapping。

## 实验设置 {#setup}

**实验固定了货物、司机、道路与目的地，只改变 host backing 与 registration。**
这让 PSS、startup 与 copy-profile 的差异可以回到 HWR transport 本身。

| Field | Frozen value |
|---|---|
| vLLM / vLLM-Omni | vLLM 0.27.0; source `497c537c`; installed distribution metadata `0.27.0rc2.dev60+ge2721dc97` |
| Model | MiniMax-H3 FL2VA snapshot `42ed227e` |
| Hardware | 4×H200, one NUMA-0 domain, NV18 between every selected pair |
| Internal GPU label | `NVIDIA L20X`; this cluster label denotes the cookbook's H200 SKU |
| Engines | A on GPUs 0–1, B on GPUs 2–3; each `DP1×TP2` |
| DLO | no-AllGather, 20 resident main-DiT blocks, two rotating device buffers |
| Other parallelism | text encoder TP2; VAE tile patch parallel 2; SP1; PP1; CFG1 |
| Workload | T2VA, 1344×768, 8.7 s, seed 1101, three sigma points / two denoiser evaluations |
| Timing | one excluded warmup; three concurrent pair waves; observer quiesced before timing |

完整 compact evidence package 包含 report、summary、profile aggregate、command、
test plan 与 SHA manifest：
[`2026-08-26-h200-tp-validation`](https://github.com/hsliuustc0106/vllm-omni-cookbook/tree/main/blog/assets/figures/host-weight-runtime/evidence/2026-08-26-h200-tp-validation)。
raw torch trace 每个 TP rank 约 866 MiB，因此没有复制进 cookbook。

## 实测影响 {#measured-impact}

**HWR 最清楚的收益是让第二台引擎装得下、启动更快，同时不改变 GPU 工作——像
共享仓库，但每辆车仍走原来的路线。** registered HWR 保留一套物理
final-layout working set、删除 private staging slot，并保持稳态输出一致。

![TP2 Host Weight Runtime 的 PSS、第二台引擎 startup、pair latency 与 profile attribution]({{ site.baseurl }}/assets/figures/host-weight-runtime/fig2-tp2-results.svg)

### Rank-matched sharing 与 memory

| Mode | Pair PSS | `Private_Dirty` | `Pss_File` | `Pss_Shmem` |
|---|---:|---:|---:|---:|
| Existing host tensors | 388.06 GiB | 387.17 GiB | 0.92 GiB | 366.10 GiB |
| HWR + bounded staging | 254.08 GiB | 191.38 GiB | 62.73 GiB | 170.10 GiB |
| HWR + registered mmap | **245.98 GiB** | **183.25 GiB** | 62.75 GiB | **162.10 GiB** |

两种 HWR mode 都让两个 TP-rank-0 worker 共享七个相同 payload inode，也让两个
TP-rank-1 worker 共享另外七个；cross-rank shared inode 为零。registered mmap
相对 staging 再省 8.10 GiB PSS，来源是四个 worker 不再分配 private two-slot
staging。

### Warm startup

| Second-engine process-to-ready | Samples | Mean ± std | Registered delta |
|---|---|---:|---:|
| HWR disabled | 172.18, 182.88, 173.48 s | 176.18 ± 5.84 s | — |
| HWR registered | 149.56, 150.64, 151.14 s | **150.45 ± 0.81 s** | **−25.73 s (−14.6%)** |

registered minus disabled 的保守 95% interval 为 `[−40.37, −11.09] s`。第一台
warm registered Engine A 是 151.45 s，几乎等于 Engine B 的 150.45-s mean：
真正加速两台 consumer 的是已经发布的 artifact，不是“Engine A 还活着”。

### Concurrent request waves

| HWR transport | Pair-wall samples | Mean ± std | Combined throughput |
|---|---|---:|---:|
| Bounded staging | 39.812, 32.053, 31.969 s | 34.611 ± 4.504 s | 0.0584 req/s |
| Registered mmap | 28.819, 28.871, 28.889 s | **28.859 ± 0.036 s** | **0.0693 req/s** |

registered mean 低 16.6%，但保守 interval 是 `[−16.94, +5.44] s`，因为第一轮
staged 是 outlier，不能把它写成确定的 request-latency gain。profile attribution
更稳定：

| Per TP2 engine profile | HWR staged | HWR registered | Difference |
|---|---:|---:|---:|
| CPU `aten::copy_` rank-work | 33.914 s | **14.096 s** | **−19.818 s (−58.4%)** |
| H2D payload | 171.943 GiB | 171.943 GiB | unchanged |
| H2D device time | 3.356 s | 3.366 s | +0.010 s |
| H2D operations | 4,361 | 5,921 | +1,560 tensor-level copies |
| Compute-kernel rank-work | 39.539 s | 39.636 s | +0.097 s |
| NCCL-kernel rank-work | 1.252 s | 1.192 s | −0.059 s |
| NVLink Tx / Rx | 263.058 / 263.058 GiB | 263.058 / 263.058 GiB | unchanged |
| Peak HBM per GPU | 30,286 MiB | 30,286 MiB | unchanged |

registration 删除 recurrent host copy，却不减少 model bytes。额外 1,560 个
operation 来自 tensor-level direct H2D fragmentation，aggregate device-copy
time 仍保持不变。

### Correctness 与 capacity boundary

16 个 registered/staged 稳态 measured/profile output 只有一组完整 video/audio
digest：video `[209, 768, 1344, 3]`，stereo audio `[1, 2, 278400]`，均为
`float32`。被排除的 first warmup 在两种 mode 之间相互匹配，但 audio hash 与
稳态请求不同；video 保持一致。因此 parity gate 是稳态 output。

existing-path pair 是 **capacity evidence，不是 latency control**。它达到
388.06 GiB PSS，完成一轮 803.91-s pair wave，下一轮在 GPU work 之后让 parent
process 长时间消耗 kernel time，最终命中默认 600-s async-output watchdog。
缩短后的 retry 达到 913.23/921.00 GiB，即 container memory limit 的 99.16%。
这里不声称 existing path 有 n=3 latency 或 output parity。

## TP1 反例 {#tp1}

**共享 artifact 并不天然更快——像原始出版社文件已经能被两个人直接打开，
却又先为图书馆重新装订一遍。** compatible TP1 MiniMax-H3 已经有 PR #6213 的
checkpoint-mmap path；final-layout HWR 会在一个已经能共享 raw page 的场景里
额外执行 identity、validation 与 restoration。

| TP1 metric | Checkpoint mmap | Registered HWR | Result |
|---|---:|---:|---:|
| Second-engine process-to-ready, n=3 | **106.66 s** | 203.62 s | HWR **+96.96 s (+90.9%)** |
| Two-engine PSS | 203.51 GiB | **195.57 GiB** | HWR −7.94 GiB |
| Shared payload | raw checkpoint pages | one 61.73-GiB final artifact | both share file pages |

TP1 transport stress 中 registered transport 仍能删除 recurrent CPU staging，
但 HWR 不是 startup optimization。对 TP1，除非 final-layout representation
带来另一项明确收益，否则先保持 HWR disabled。

## 收益与代价 {#tradeoffs}

**HWR 用一份持久共享 representation 替换反复 private work——像购买公共冷库，
能省食材，但也需要场地、电力与库存规则。** 是否值得，取决于固定成本能否在足够
多的等价 worker 与 startup 上摊薄。

| Pros | Costs and risks |
|---|---|
| One immutable page-cache working set for equivalent workers | Cold canonical load plus artifact writing and validation |
| Runtime-ready TP-coordinate artifacts, not raw checkpoint guesses | Roughly one final-layout model copy on local disk per identity |
| Faster warm TP2 startup in the measured topology | A second, versioned model reconstruction path to validate and maintain |
| Registered mmap removes private staging and recurrent CPU copies | Full mapped range is registered/page-locked for each worker lifetime |
| Exact identity prevents TP/SP/revision/ABI aliasing | Layout, model revision, or producer ABI changes create a new identity |
| `preferred` retains canonical fallback for typed recoverable failures | `required` intentionally fails on a miss or invalid artifact |
| H2D, HBM, and TP communication remain unchanged | No automatic eviction or cross-node coherence in V1 |

store 必须位于经过验证的 node-local filesystem。remote filesystem 会被拒绝，
因为它的 page-cache 与 advisory-lock semantics 不满足本地 contract。`tmpfs`
虽然合法，却会直接消耗 host memory，还可能使用 swap，改变 disk-backed sharing
的容量故事。

## 如何启用 {#how-to-use}

**启用 HWR 分两步：先用完全相同的 topology 填仓，再让 matching worker 从同一个
node-local root 获取 lease。** 普通部署用 `preferred`；只有 store 已知完整时才用
`required`。

{% include usage-cookbook.html modes=page.usage %}

三种 mode 有意提供不同 availability contract：

| Mode | Exact local hit | Miss / recoverable store problem |
|---|---|---|
| `disabled` | HWR is not constructed or probed | existing checkpoint-mmap or ordinary-loader path |
| `preferred` | restore exact final-layout lease | canonical load; serve current model; attempt publication for future starts |
| `required` | restore exact final-layout lease | fail startup |

producer 与 consumer 要使用同一个 immutable model revision、dtype、TP/SP layout
和 root。replicated artifact identity 不包含 DP rank/size，因此等价 deployment
replica 可以共享。TP2 population 需要 matching TP2 cohort 来创建两份
TP-coordinate artifact。V1 是 node-local，每个 node/storage domain 都要重复填充。

## Registration policy 与验证 {#registration-policy}

**Registration 像带容量许可证的快速月台；许可证不支持或额度不足时，仓库仍共享，
只是货物重新经过 staging room。** eligible warm HWR hit 会在已有 pinned-memory
policy 下自动尝试 registration。

`--dlo-host-registration-limit-gib` 是覆盖完整 page-aligned mapping 的
**per-worker ceiling**。零表示不额外限制，不表示关闭 registration。本次 TP2 worker
各自 mapping 30.86 GiB，benchmark 使用 80-GiB ceiling。正数 budget 小于完整
mapping 时，会在任何 partial registration 前选择 bounded staging。

warm registered path 的日志应包含：

```text
DLO host-weight plan active (rank-local, host_weight_runtime):
  skipping ordinary materialization for ['transformer.']
DLO consuming final-layout Host Weight Runtime lease ...
Registered 30.86 GiB of HWR mmap in 7 range(s) for direct H2D ...
...
Unregistered HWR mmap ranges
Released Host Weight Runtime lease ...
```

如果日志显示 `using bounded host staging`，共享仍然有效，只是 direct transport
优化未启用。shutdown 会先 drain pending H2D、释放 registration 与 model/hook
reference，最后关闭 lease。

节点 memory 要比较 aggregate process-tree **PSS**，不要相加 RSS。RSS 会在每个
mapping process 都把同一个物理 file page 记一次；PSS 才会按映射者分摊。

## 如何选择 {#how-to-choose}

**先按 scheduling domain 选择，再考虑 storage——就像先决定火车是否共用一张
时刻表，再选择仓库。** 独立引擎需要 shared file 而不是 shared collective；
同步 rank 往往可以直接使用 AllGather。

{% include decision-cards.html items=page.decisions %}

快速 NVLink/NVSwitch 不会消除这个差异。在本次 H200 topology 中，staged 与
registered HWR 每个 profiled engine 都产生 263.058 GiB Tx 与 Rx，因为 ordinary
TP communication 没变。HWR 节省的是 host work，同时保留每台引擎原有 TP group。

## 运维清单 {#operations}

**把 HWR 当作本地 artifact store，而不是临时 Python cache——需要稳定地址、足够
disk 与明确 lifecycle。** 健康部署应该在服务流量前就能观察 identity、capacity
与 fallback。

1. 将 `host_weight_runtime_root` 放在所有 storage-domain worker 可见的持久
   node-local disk；不要使用 NFS、Ceph 或 process-private 临时目录。
2. 分开预算 artifact disk 与 host PSS。MiniMax-H3 TP2 的两份 artifact 总计
   61.73 GiB；新 revision/layout 会生成新 identity。
3. 用精确 serving TP/SP cohort 填充。`required` 不能 bootstrap 空 store。
4. 按 worker 预算 registration。full registration 用长期 page locking 换掉
   private staging；较小 ceiling 会安全 fallback。
5. 监控 cgroup memory，不要只看 host-wide `free`。capacity probe 在物理 host
   仍有 RAM 时已经达到 container limit 的 99.16%。
6. 修改 model revision、producer/restorer ABI、dtype 或 semantic parallel layout
   后，重新运行 `preferred`。
7. 显式清理 store 前先停掉 consumer。V1 有 capacity check 与 quarantine，但没有
   automatic eviction 或 public prewarm/cleanup CLI。

router 可以把请求分给 Engine A/B，但 host-page sharing 不依赖 router。反过来，
两台引擎使用同一个 root，也不会自动得到 routing、health coordination 或共享
failure domain。

## 限制与后续 {#limitations}

**已发布路径是一条精确而狭窄的桥，不是万能 cache——像先认证一条车道，而不是
一次铺完所有道路。** 不支持的 semantics 会失败或 fallback，而不是复用错误 bytes。

- 当前 active consumer 是 eligible MiniMax-H3 no-AllGather DLO 的
  final-layout BF16-with-preserved-FP32。
- DLO AllGather、DLO disabled 或 HWR disabled 不会选择 final-layout HWR。
- online quantization 需要 representation-specific producer 来处理 scale 与
  physical layout。dynamic LoRA 保持 overlay；static merged adapter 需要独立
  identity 与显式支持。
- 当前 consumer 不支持 HSDP 与 non-default load format。
- store 是 node-local。remote provider、cross-node coordination、automatic
  eviction 与 enforceable producer cancellation 都属于未来工作。
- GPU 被重新分配后，按用户要求缩短实验，未完成可选的 TP2
  `dlo_resident_layers=0` scaling profile。因此本文证明 recipe-placement sharing
  与 transport，不声称 TP2 per-layer scaling curve。
- existing-path capacity attempt 没有形成 n=3 latency 或 output-parity row；
  本地 correctness claim 是 registered/staged 稳态 parity。

[RFC #6414](https://github.com/vllm-project/vllm-omni/issues/6414) 继续追踪
FP8 producer 与可能的未来 AllGather consumer。它们应扩展统一 store/lease
contract，而不是重新创建另一套 cache format。

## 参考 {#references}

**下面是本文的收据：每项 architecture 与 performance claim 都能回到 merged PR、
current guide 或保留的 benchmark artifact。** audit code 时使用 pinned link，
实际部署时以 current guide 为准。

- [PR #6213 — Loader-owned checkpoint host-weight plans for TP1 DLO](https://github.com/vllm-project/vllm-omni/pull/6213)
- [PR #6419 — Host Weight Runtime foundation](https://github.com/vllm-project/vllm-omni/pull/6419)
- [PR #6427 — Explicit post-load HWR publication](https://github.com/vllm-project/vllm-omni/pull/6427)
- [PR #6445 — Final-layout BF16 HWR artifacts](https://github.com/vllm-project/vllm-omni/pull/6445)
- [PR #6486 — no-AllGather DLO HWR consumer](https://github.com/vllm-project/vllm-omni/pull/6486)
- [PR #6591 — Registered HWR mmap for direct DLO H2D](https://github.com/vllm-project/vllm-omni/pull/6591)
- [RFC #6414 — reusable runtime-ready host artifacts](https://github.com/vllm-project/vllm-omni/issues/6414)
- [Current DLO user guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md)
- [Current HWR module design](https://github.com/vllm-project/vllm-omni/blob/main/docs/design/module/host_weight_runtime.md)
- [PR #6213 前序文章]({{ site.baseurl }}/2026-08-16-pr-6213-loader-owned-host-weight-plans/)
- [2026-08-26 H200 TP validation evidence](https://github.com/hsliuustc0106/vllm-omni-cookbook/tree/main/blog/assets/figures/host-weight-runtime/evidence/2026-08-26-h200-tp-validation)
