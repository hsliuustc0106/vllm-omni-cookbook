---
layout: post
title: "在 vLLM-Omni 中服务 MiniMax-H3（5）：内核篇——TRTLLM 成为默认后端，Q/K 归一化+RoPE 融合为单个 kernel（PR #5779 + #5990）"
date: 2026-09-15 12:00:00 +0800
author: hsliuustc0106
summary: >-
  PR #5779 让 TRTLLM attention 读懂 MiniMax-H3 的 packed 序列，并成为
  数据中心 Blackwell 的默认后端；PR #5990 把 Q/K RMSNorm+RoPE 融合成单个
  Triton kernel——B300 上 steady denoising 再降 3.18%。
tags: [MiniMax-H3, TRTLLM, B300]
category: PR Analysis
feature: kernels
lang: zh
pair: /2026-09-15-understanding-pr-5779-5990-minimax-h3-kernels/
permalink: /zh/2026-09-15-understanding-pr-5779-5990-minimax-h3-kernels/
usage:
  - label: "Serve · 4× B300 默认"
    blurb: "无需 attention flag——TRTLLM 自动选中"
    title: "vllm serve · 四卡吞吐 profile"
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
      自 PR #5779 起，MiniMax-H3 声明自己的
      packed-sequence contract，平台在数据中心 Blackwell（sm_100/sm_103）
      上自动选择 dense BF16 TRTLLM_ATTN——不需要任何 attention flag。记录
      性能前确认日志出现 "Defaulting to diffusion attention backend
      TRTLLM_ATTN"。不要加 --enforce-eager；首个请求包含 regional 编译，
      测稳态前先预热一次。H3 是 CFG-distilled 模型：--cfg-parallel-size
      必须保持 1。
  - label: "Serve · FA4 对照"
    blurb: "显式 FLASH_ATTN 基线"
    title: "vllm serve · Blackwell 上的 FlashAttention-4"
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
        --diffusion-attention-backend FLASH_ATTN
    note: >-
      FA4 仍是一等对照项。先装可选依赖（"uv pip install -e '.[fa4]'"）——
      官方镜像没有它时会回退到 Hopper-only kernel，在 Blackwell 上直接报
      "no kernel image is available"（#5779 基线调试踩过的坑）。确认日志
      出现 "Using CuTe FlashAttention-4 on Blackwell"。
  - label: "Serve · SAGE + Skip-Softmax"
    blurb: "有损开关，per-role 保护"
    title: "vllm serve · TRTLLM 量化 + 稀疏"
    code: |
      vllm serve "${MODEL}" \
        --omni --host 0.0.0.0 --port "${PORT}" --trust-remote-code \
        --num-gpus 4 --usp 4 --ring 1 \
        --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
        --diffusion-attention-config '{
          "default": {
            "backend": "TRTLLM_ATTN",
            "quant": {
              "dtype_qk": "fp8_e4m3",
              "q_block_size": 1,
              "k_block_size": 16
            },
            "skip_softmax": {
              "threshold": 0.05,
              "disabled_until_timestep": 0.97
            }
          },
          "per_role": {
            "minimax_h3.token_refiner": {
              "backend": "TRTLLM_ATTN"
            }
          }
        }'
    note: >-
      两个优化都有损且效果会叠加——采纳前用同 prompt、同 seed 与 dense
      输出对比。以上是上游 recipe 的保守起点：50 步时 0.97 的 cutoff 让前
      14 / 49 次 denoiser forward 保持 dense。per_role 让 14-token 的
      token refiner 保持 dense；per-role 配置不继承 default 的 quant 与
      skip_softmax。B200 额外支持 int8 Q/K，精度比 FP8 更好。
decisions:
  - when: "数据中心 Blackwell（B200/B300，sm_100/sm_103）"
    pick: "保持 TRTLLM 默认"
    why: "自 #5779 起模型声明 packed contract 后自动选中；上游 recipe 的稳定 A/B 显示 dense TRTLLM 与 FA4 相差 2% 以内，且 TRTLLM 是唯一能打开 Skip-Softmax 与 SAGE 两扇门的 backend。"
  - when: "需要 FA4 对照"
    pick: "--diffusion-attention-backend FLASH_ATTN"
    why: "装上 fa4 extra 后仍是一等选项。注意各 A/B 的离散度：一次配对 review 实测 FA4 慢 14.1%（diffusion），稳定 recipe 实测相差 2% 以内——单对基准对配置很敏感。"
  - when: "工作站 Blackwell（sm_120/sm_121）或 head_dim ≠ 128"
    pick: "留在 CUDNN_ATTN 路线"
    why: "TRTLLM 自动路由刻意跳过工作站 Blackwell 与非 128 的 head dim；这些 GPU 保持正常 fallback，需要 mask 的路径同理。"
  - when: "数据中心 BF16 上抠每一个百分点"
    pick: "什么都不用配——融合 Q/K prologue 自动生效"
    why: "#5990 在 Triton + CUDA + BF16 + head_dim 128 + rotary_dim 96（恰好 H3 的几何）时自动启用，其余场景静默回退 eager reference。作者 B300 实测 steady denoising −3.18%。"
  - when: "用保真度换速度"
    pick: "SAGE + Skip-Softmax，per-role 保护"
    why: "量化 Q/K、跳过可忽略的 softmax tile。用 per_role 保住 token refiner 的 dense 路径；并记住即使 dense 后端之间输出也略有差异：同 prompt 下 TRTLLM vs FA4 实测 PSNR 27.10 dB / SSIM 0.8880。"
---

## 摘要 {#tldr}

**这两个 PR 是 MiniMax-H3 故事的内核篇：PR #5779 让最快的 attention
backend 与 H3 的 packed 序列兼容——并让它成为数据中心 Blackwell 上的默认
选择；PR #5990 把每次 attention 调用前运行的两个小操作融合成单个 GPU
kernel，再买到 −3.18% 的 steady denoising 延迟。** 想象一间仓库：最好的
卡车（TRTLLM）以前 outright 拒收这家客户的货板，因为板上有填充块
（issue #5771）；第一个修复教卡车读货单（manifest）、只卸真实货箱；第二个
修复把装卸口的两个文书岗位合并成一个。

| 改动 | PR | 合并时间 | 4× B300 上的效果 |
|---|---|---|---|
| TRTLLM 读懂 H3 的 packed 布局并成为默认 | [#5779](https://github.com/vllm-project/vllm-omni/pull/5779) | 2026-08-06 | 一次配对实测中 dense TRTLLM 比 FA4 快 14.1%（diffusion）；稳定 A/B 说"相差 2% 以内" |
| Q/K RMSNorm + RoPE 融合为单个 Triton pass | [#5990](https://github.com/vllm-project/vllm-omni/pull/5990) | 2026-08-14 | steady denoising 109.687 s → 106.205 s（−3.18%，1.033×） |

今天这两件事都是自动的：数据中心 Blackwell 上 H3 自动选择 dense BF16
`TRTLLM_ATTN`；融合 prologue 在 H3 的精确几何（BF16、`head_dim=128`、
`rotary_dim=96`）上自动启用，其余场景走 eager 回退。第二个数字也是本章的
诚实课——一次内核级重写，端到端落在"1.033×"，因为它移除的前置开销相对其
服务的 attention 计算太小了。

## 背景 {#background}

本节交代这两个 PR 的位置：它们把"TRTLLM 根本跑不了 MiniMax-H3"变成
"TRTLLM 是默认选择、且周边算术已融合"——像先修路、再拓宽车道。[Blog 3](#limitations)
讲的是再后面一步（PR #6542 移除逐层 mask 重验证）；本文讲它之前的两步。
两步出自同一位作者（Bo Li），同一块 4× B300 测试床，上游都打着
`Kernel optimization` 标签。

**为什么非要 TRTLLM？** 在 Blackwell 上，`TRTLLM_ATTN`——FlashInfer 对
TensorRT-LLM 生成式 attention kernel 的内置分发——是唯一能开两个有损加速
的 diffusion backend：Skip-Softmax（稀疏 attention）与 SAGE（量化
attention）。而在 #5779 之前，想要它们就意味着跑不了 H3：

- **硬失败（[issue #5771](https://github.com/vllm-project/vllm-omni/issues/5771)）。** 给 H3 选 `TRTLLM_ATTN`，第一次 packed attention 调用就死在 `ValueError: TRTLLM_ATTN does not support attn_mask`。H3 *需要* 这张 mask，因为它把 packed 序列对齐到 64 行的整数倍：示例 workload 是 58,758 个有效 token 装进 58,816 行的张量，末尾 58 行填充位靠一张 prefix-valid mask（`arange(total) < used`）标记。而 TRTLLM 拒绝*任何* mask。
- **SAGE 下输出非有限值。** 若把那 58 个填充 token 当成又一段"有效"序列，dense attention 撑得住——但 SAGE 量化会照单压缩垃圾数值，可能产出非有限输出。
- **一个 14-token 的角落。** H3 还有第二个极小的 attention 站点（token refiner），14 个 token *比 SAGE 的一个 K 量化 block 还短*（`k_block_size=16`）；FlashInfer SAGE 对这种序列返回非有限值。
- **Skip-Softmax 无法开启闸门。** H3 不发布 denoise timestep，"前期保持 dense、后期稀疏化"的门控永远无法生效——路径只打一条警告、全程 dense。

再看 #5990 的对象：每次 attention 调用前，Q 和 K 各自要过两道工序——
RMSNorm（把每个 head 的 128 维向量归一到单位 RMS，再乘可学习权重）和部分
RoPE（按位置相关的角度旋转 128 维中的前 96 维）。eager 模式下这是一串小 kernel，每个都把中间结果写进 HBM——每个
tensor、每个 DiT block、50 步视频的 49 次 denoiser 评估里都要付一遍。

## PR #5779 改了什么 {#trtllm-refine}

本节是机制故事：backend 学会信任 packing 元数据而不是拒收——像一个收货
口，终于接受"真实货箱在前、填充块在后"的货单，只卸真实货箱，离开时再把
填充块绑回去、让货板恢复标准尺寸。

### 信任 packing 元数据，然后裁剪 {#packed-trim}

packed 路径的钥匙是随 `AttentionMetadata.extra` 传来的四个元数据字段——
`cu_seqlens_q`、`cu_seqlens_k`、`max_seqlen_q`、`max_seqlen_k`（cu_seqlens
= 累积序列长度，varlen kernel 用来定位 packed 批中文档边界的格式）。契约
严格且报错响亮：

- **四者要么全在、要么全无。** 缺字段直接 `Incomplete packed TRTLLM attention metadata; missing [...]`。
- **元数据必须覆盖输入。** `cu_seqlens` 不覆盖全部 Q/K/V token 就报 `must cover all Q/K/V tokens`。
- **只认结构性 padding。** mask 只有 prefix-valid——真实 token 连续在前、填充在后——才被接受；其余照旧硬拒绝。

契约满足后，backend 把 Q/K/V 裁剪到有效前缀，带累积长度调用 ragged
（varlen）TRTLLM kernel——填充行永远到不了量化或 attention——然后**用零
填充恢复对齐的物理形状**，因为紧随其后的 Ulysses all-to-all 期望定长
buffer。

![PR #5779 的 packed 路径：prefix-valid mask 与 cu_seqlens 元数据让 TRTLLM 把 Q/K/V 裁剪到 58,758 个有效 token，ragged FMHA 只算真实 token，再用零填充恢复 58,816 行的对齐形状交给 Ulysses all-to-all。下半部分是它防住的两个失败模式：SAGE 把 padding 垃圾值量化成非有限输出；14-token 的 token refiner 短于一个 SAGE K 量化 block。]({{ site.baseurl }}/assets/figures/pr-5779-5990-minimax-h3-kernels/fig1-packed-trim.svg)

给系列连续性留一个诚实标记：#5779 里 prefix-valid 检查仍然*每次调用都从
device tensor 上跑*（mask 求和、比较、读取）。这笔逐调用开销——约
499.7 µs 的 p50 间隙——正是几周后
[PR #6542，即 Blog 3]({{ site.baseurl }}/2026-09-12-understanding-pr-6542-trtllm-mask-free-packed-padding/)
通过让生产者在 host 侧发布边界而移除的东西。#5779 让路径正确；#6542 让它
免同步。

### 那个必须保持 dense 的 14-token 站点 {#token-refiner}

token refiner 的 14 个 token 填不满一个 16 的 SAGE K-量化 block，所以在
那里开 SAGE 不是不准——是非有限。#5779 让 backend 自身把任何短于一个量化
block 的序列路由到 dense TRTLLM kernel（单测断言这种序列*不会*调到 SAGE
量化器）。recipe 的 `per_role` 覆盖（`minimax_h3.token_refiner`）在给主
DiT 序列打开 SAGE 与 Skip-Softmax 时把同一意图钉死——注意语义：**per-role
配置不继承 `default` 的 `quant` 与 `skip_softmax`**，"这个站点保持纯
dense"正依赖于此。

### 发布 denoise timestep {#timestep}

Skip-Softmax 需要知道"denoising 走到哪一步了"才能决定何时稀疏化是安全
的——一台读不到室温的恒温器永远不会切换模式。H3 此前不发布 normalized
timestep，门控只能全程 dense。readiness 修复按正确约定接入（`t = 1 −
sigma`，sigma 为噪声水平），并恢复了非负 threshold 契约；现在
`record_denoise_step(idx, normalized_timestep=...)` 会穿透 forward
context。recipe 默认 cutoff（`0.97`）加 H3 的 flow shift 12 意味着：49 次
denoiser forward 的前 14 次保持 dense。

### reviewer 改变了什么 {#review-story}

"能跑"变"默认"就发生在 review 里：

- **"LGTM, let's make trtllm as the default backend for SM100"**——
  maintainer 的第一条评论。随后他准备了后续 commit
  （[`20cc23ae`](https://github.com/lishunyang12/vllm-omni/commit/20cc23ae)）：
  让 H3 *声明自己的 packed-sequence contract*——平台在支持的数据中心
  Blackwell（sm_100/sm_103）上、FlashInfer trtllm-gen kernel 可用时，
  自动选择 dense BF16 `TRTLLM_ATTN`。工作站 Blackwell（sm_120/sm_121）、
  不支持的 head dim、需要 mask 的路径保持正常 fallback——且默认不开
  Skip-Softmax 或量化。契约测试钉住 `MiniMaxH3Pipeline →
  attention_mask_free = True`。
- **FA4 基线是被调试出来的。** 官方 vLLM 0.26.0 镜像没有 `flash-attn-4`，
  `FLASH_ATTN` 回退到 Hopper-only kernel，在 SM103 上报 `no kernel image is
  available`；装上 `flash-attn-4[cu13]==4.0.0b18` 之后 "CuTe
  FlashAttention-4 on Blackwell" 才生效——信任你自己的 FA4 数字之前，值得
  先知道这一段。
- **readiness 修复**：上面的 timestep 约定、threshold 契约、重写的 DCO
  作者身份。

## PR #5990 改了什么 {#fused-prologue}

本节是另一个机制故事：每次 attention 调用前的两道工序合并成一个 GPU
kernel——像把照片流水线里"锐化"和"旋转"两个工位（此前各自打印成果、让下
个工位从公共抽屉重新取读）并成一个不下台面的工位。

### eager prologue 及其代价 {#eager-prologue}

对 Q、对 K 各一遍：eager 路径先跑 `F.rms_norm`（把整份归一化的
`[tokens, heads, 128]` tensor 写进 HBM），再跑 RoPE：把每个向量前 96 维拆
成两半、乘 cos/sin 表、加减、再用 `torch.cat` 把旋转后的两半与未动的最后
32 维拼回去——更多 kernel、更多物化的中间结果、更多 HBM 往返。在 H3 的几
何（BF16、`head_dim=128`、`rotary_dim=96`、48 的非交错对半）下，这套流程
每个 DiT block 跑两遍、每次 denoise 评估跑一遍——一笔小额税费，重复次数极
大。

### 单个 Triton pass {#one-pass}

新的
[`fused_qk_norm_rope.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/layers/fused_qk_norm_rope.py)
kernel 按（token，8 个 head 一组）启动一个 program。每个 program 一次读
入自己的 Q（或 K）切片，在 fp32 寄存器里算 RMS 归一（`rsqrt(mean(x²) +
eps)` × 权重），用索引算术找到每个元素的旋转搭档（非交错的半交换），乘
packed 表里的 cos/sin，一次写出。没有归一化副本、没有旋转中间积、没有
`cat`——HBM 流量只有一读一写。

kernel 之外还有两个支撑改动：

- **共享 rope 表。** H3 每次 forward 物化一份 packed 的 `[cos θ₀..₄₇,
  sin θ₀..₄₇]` 表（`_build_rope_table`），所有 block 共用——此前每个
  block 从原始频率各自重算 cos/sin。TeaCache 上下文提取器也换到同一张
  表，保证缓存与 serving 位级一致。
- **custom-op 边界。** 融合路径注册为 `vllm_omni::fused_qk_norm_rope`，
  带 fake（meta）实现，H3 依赖的 regional `torch.compile` 因此能 trace
  穿过它。

公开的 layer API 与模型无关；快路径门控很窄——Triton + CUDA + BF16 +
`head_dim=128` + `rotary_dim=96`——其余一切（其他 dtype、其他几何、CPU）
静默走 eager reference。

![PR #5990 之前，eager prologue 对 Q、对 K 各跑一遍：F.rms_norm 把归一化结果写进 HBM，RoPE 再拆分、乘法、cat 重组——每个 DiT block、每个 denoise forward，一段视频 49 遍。之后每个 tensor 一个 Triton kernel：fp32 寄存器内做 RMS 归一、对共享的 packed cos/sin 表做 RoPE 对交换——一读一写，零中间结果。]({{ site.baseurl }}/assets/figures/pr-5779-5990-minimax-h3-kernels/fig2-qk-fusion.svg)

### reviewer 问出的精度问题 {#accuracy}

reviewer 要求 SSIM/PSNR 证据，证明融合不会可见地改变视频。作者的回答是单
测：融合结果与 BF16 eager 参考实现相比，在序列长度 1、257、1024 上**最大
绝对误差 0.0625、平均绝对误差 0.00072–0.00077**——BF16 量级的舍入差异
——端到端生成视频与基线"几乎一致"。没有补 SSIM 表；引用这个 PR 时值得知道
这一点。

## 实测影响 {#measured-impact}

以下数字全部来自上游 PR 正文、review 评论与 recipe——作者报告的实测，不是
cookbook 基准。

**后端 A/B（#5779 review，配对实测）。** 4× B300 SM103，1248×768，209
帧，50 步，seed 1101，Ulysses4/Ring1/TP1，VAE tile4，regional compile；
排除一次编译预热（[来源：review 评论，commit
`20cc23ae`](https://github.com/vllm-project/vllm-omni/pull/5779#pullrequestreview-)）：

| Backend | Steady diffusion | Wall |
|---|---:|---:|
| CuTe FlashAttention-4（`FLASH_ATTN`） | 83.854 s | 88.558 s |
| Dense BF16 TRTLLM（`TRTLLM_ATTN`） | **71.990 s** | **76.176 s** |

这次运行里 TRTLLM **diffusion 快 14.1%、端到端快 14.0%**。但上游 recipe
记录的是：用四卡 profile 的*稳定*实测"dense `TRTLLM_ATTN` 与 FA4 相差
2% 以内"，而且 #5779 之前的文档明说 TRTLLM 排在 cuDNN 前面"不是因为它的
dense kernel 更快"。两个数字要放在一起读：默认决策的根基是 TRTLLM 打开了
Skip-Softmax/SAGE 两扇门、并拥有维护中的 packed 路径——不是一个保底的
dense 稳赢。单对实测对配置敏感；这个离散度本身就是数据。

**dense 输出跨后端不是逐位一致。** 同一 B300 节点上，50 步 FA4 运行模型
阶段 88.98 s；同 prompt/seed 下 TRTLLM 与 FA4 的编码视频对比为**平均 PSNR
27.10 dB、SSIM 0.8880**——场景、运动、构图相同，但像素不同。切换后端、或
在其上叠有损开关时，把这个差异算进预算。

**融合 Q/K prologue（#5990）。** 4× B300 SXM6，TP1/Ulysses4/Ring1，dense
BF16 `TRTLLM_ATTN`，1344×768，243 帧，50 个配置步，seed 0；一次预热加一
次测量请求：

| 指标 | 之前（eager prologue） | 之后（融合） |
|---|---:|---:|
| Steady denoising 延迟 | 109.687 s | **106.205 s** |

即 **−3.18%（1.033×）**。PR 附带前后 Nsight Systems 捕获，展示 prologue
窗口内 kernel 启动与内存流量的坍缩。系列要守住两条诚实注记：

- **融合退回的正是被融合工作本身的成本——不多退。** prologue 随 token
  线性增长；在约 58k token 下 attention kernel 本身主导整个 layer，所以
  即便消灭全部中间往返，也只落在 steady denoising 的约 3%。profiler 里
  kernel 局部的赢面再大，也不会越过它移除的工作线性传导到端到端——本 PR
  从未如此宣称。
- **一次预热、一次测量请求。** 上面的每个数字都是单 seed 上的单次观察，
  不是分布——与 Blog 4 给容量实测挂的方法论提醒相同。

## 怎么用 {#how-to-use}

没有需要开启的开关。TRTLLM 默认在数据中心 Blackwell 上自动生效，融合 Q/K
prologue 在 H3 几何上自动生效。操作员可见的面是选择对照后端与两个有损开
关：

{% include usage-cookbook.html modes=page.usage %}

如果你在复现 [issue #5771](https://github.com/vllm-project/vllm-omni/issues/5771)
（`attn_mask` 被拒），升过 #5779 就是 packed 路径本身的修复；相关的 B200
默认选择失败（[#6358](https://github.com/vllm-project/vllm-omni/issues/6358)
，见 Blog 3）后来由 #6542 修复。

## 怎么选 {#decision-cards}

{% include decision-cards.html items=page.decisions %}

## 局限与后续 {#limitations}

- **TRTLLM dense 仅限数据中心 Blackwell**（sm_100/sm_103，装有
  FlashInfer）。工作站 Blackwell 与非 128 head dim 保持 fallback 路线；
  [Blog 4]({{ site.baseurl }}/2026-09-14-understanding-pr-5764-minimax-h3-layerwise-offload/)
  的双卡 offload profile 刻意跑 cuDNN attention。
- **SAGE 与 Skip-Softmax 有损且叠加。** recipe 的保守值（threshold 0.05、
  cutoff 0.97）让 49 次 forward 的前 14 次保持 dense；在 B300 上验证它们
  的 PR 检查的是 HTTP 200 + 有效 MP4，不是感知质量表。
- **融合 kernel 刻意收窄。** BF16、`head_dim=128`、`rotary_dim=96`——
  H3 的几何，其余只经 eager 回退泛化。已发表的精度证据是单测误差统计，
  不是 SSIM。
- **#5990 的测量是单 seed、单请求、仅 steady denoising。** 没有吞吐、多
  请求或质量表随行。
- **本 PR 的路径里 mask 仍然存在。** #5779 每次调用从 device tensor 验证
  prefix-valid mask；#6542（Blog 3）是移除这笔成本的后续——两篇要当一个
  弧线读。
- **内核主题的其余部分上游仍未合并。** VAE 侧 kernel PR（#6030 堆叠
  tiling、#5937 更轻解码、#5979 regional compile、#6014、#5985）全部
  open，SM120 benchmark（#5852）同样；主题里其他已合并的内核工作（NPU
  SwiGLU 融合，#5801/#6167）属于系列的硬件篇要讲的 NPU 故事。
- 这是 MiniMax-H3 优化系列（8 篇，[RFC #37](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/37)）
  的 Blog 5，"kernel optimization"主题的核心篇。相关阅读：
  [Blog 1——模块化 pipeline]({{ site.baseurl }}/2026-08-24-understanding-pr-5720-minimax-h3-modular-pipeline/)、
  [Blog 2——"四步"契约]({{ site.baseurl }}/2026-08-24-understanding-pr-5991-minimax-h3-few-step-schedules/)、
  [Blog 3——mask-free packed padding]({{ site.baseurl }}/2026-09-12-understanding-pr-6542-trtllm-mask-free-packed-padding/)（#5779 packed 路径的直系后续）、
  [Blog 4——layerwise offload]({{ site.baseurl }}/2026-09-14-understanding-pr-5764-minimax-h3-layerwise-offload/)。

## 参考 {#references}

- [PR #5779 — \[Attention\] Refine TRTLLM attention support for MiniMax H3](https://github.com/vllm-project/vllm-omni/pull/5779)（2026-08-06 合并，commit [`d219d93`](https://github.com/vllm-project/vllm-omni/commit/d219d93bbb4db1a93b6c84e77f375838ebd8246e)）
- [Issue #5771 — TRTLLM attention 拒绝 MiniMax-H3 的结构性 padding mask](https://github.com/vllm-project/vllm-omni/issues/5771)
- [PR #5990 — \[Kernel\] Fuse Q/K RMSNorm and RoPE](https://github.com/vllm-project/vllm-omni/pull/5990)（2026-08-14 合并，commit [`596c16a`](https://github.com/vllm-project/vllm-omni/commit/596c16a550aa134faf7f3dcfa0f8adf513ccd9ce)）
- [Issue #5700 — MiniMax-H3 进度 ↔ issue/PR 映射](https://github.com/vllm-project/vllm-omni/issues/5700)（两个 PR 汇报的伞形 tracker）
- [`fused_qk_norm_rope.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/layers/fused_qk_norm_rope.py) 与 [`trtllm_attn.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/attention/backends/trtllm_attn.py)（上游，现行）
- [Attention 后端用户指南](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/attention_backends.md)（上游，现行——#5779 改写的自动路由表）
- [MiniMax-H3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md)（上游，现行——四卡 profile 与 SAGE/Skip-Softmax 起点值）
- [Blog 3——mask-free TRTLLM packed padding]({{ site.baseurl }}/2026-09-12-understanding-pr-6542-trtllm-mask-free-packed-padding/)
- [系列 RFC #37——MiniMax-H3 优化博客系列](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/37)
