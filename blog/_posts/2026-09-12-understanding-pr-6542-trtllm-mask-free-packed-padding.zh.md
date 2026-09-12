---
layout: post
title: "在 vLLM-Omni 中服务 MiniMax-H3（3）：mask-free packed padding——TRTLLM 注意力不再逐层重复校验（PR #6542）"
date: 2026-09-12 12:00:00 +0800
author: hsliuustc0106
summary: >-
  PR #6542 让 MiniMax-H3 的打包方在主机（host）上直接公布有效 token 边界，TRTLLM
  注意力据此裁剪 packed padding——不再做掩码运算、不再触发 CUDA 同步。4x B300
  实测每次注意力调用的间隔从 p50 499.7 µs 降到 0.35 µs。
tags: [MiniMax-H3, TRTLLM]
category: PR Analysis
feature: host_path
lang: zh
pair: /2026-09-12-understanding-pr-6542-trtllm-mask-free-packed-padding/
permalink: /zh/2026-09-12-understanding-pr-6542-trtllm-mask-free-packed-padding/
usage:
  - label: "启动服务"
    blurb: "4 卡 TRTLLM 注意力——快路径自动生效"
    title: "vllm serve · MiniMax-H3 with TRTLLM_ATTN"
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
        --diffusion-attention-backend TRTLLM_ATTN \
        --task-type fl2va
    note: >-
      没有任何新开关需要打开：只要后端是 TRTLLM_ATTN 且模型发布了
      packed-padding 元数据，每次 packed 注意力调用都会走 mask-free 路径。
      在数据中心 Blackwell（B200/B300，SM100/SM103）上 TRTLLM_ATTN 本来就
      是自动选择的默认后端。
  - label: "回退方案"
    blurb: "非结构性 padding 的掩码"
    title: "vllm serve · 支持 mask 的后端"
    code: |
      CUDA_VISIBLE_DEVICES=0,1,2,3 \
      vllm serve "${MODEL}" \
        --omni \
        --trust-remote-code \
        --num-gpus 4 \
        --usp 4 \
        --diffusion-attention-backend CUDNN_ATTN \
        --task-type fl2va
    note: >-
      快路径只覆盖生产方自有的 [real, pad] 后缀 padding。TRTLLM_ATTN 现在
      会拒绝一切非空 attn_mask；需要任意逐 token 掩码（块稀疏、分块布局）
      时请选择支持 mask 的后端，例如 CUDNN_ATTN 或 TORCH_SDPA。
decisions:
  - when: "在数据中心 Blackwell 上服务 H3"
    pick: "保留 TRTLLM_ATTN 默认值"
    why: "该平台本就自动选择它，且 #6542 修复了 issue #6358 的 packed-padding 拒绝——快路径无需 mask 直接运行。"
  - when: "你的 padding 永远是生产方构造的 [real, pad] 后缀"
    pick: "TRTLLM_ATTN + packed-padding 元数据"
    why: "生产方早已在主机上知道有效 token 边界，因此既不构造 mask，也不触发设备同步。"
  - when: "你需要任意逐 token 掩码"
    pick: "CUDNN_ATTN 或 TORCH_SDPA"
    why: "mask-free 不等于通用 mask 支持：TRTLLM_ATTN 按设计拒绝一切非空 attn_mask。"
  - when: "使用 ring attention（--ring > 1）"
    pick: "mask 行为不变"
    why: "Ring 为固定大小的 P2P 缓冲区保留对齐行，因此不会走 mask-free 路径；生产方仍会构造 mask。"
  - when: "连续批处理混合长短请求"
    pick: "保留默认值"
    why: "TRTLLM 现在声明支持多文档 packed-varlen，生产方会省略空的 padding 文档，边界严格递增。"
---

## 摘要 {#tldr}

**[PR #6542](https://github.com/vllm-project/vllm-omni/pull/6542) 删除了横在 MiniMax-H3 的 Ulysses all-to-all 与 TRTLLM 注意力内核之间的逐层掩码复检。** 模型的打包代码在 CPU（主机端）上本来就知道确切的有效 token 数；这个 PR 新增了 `PackedPaddingMetadata` 字段，让它能把该数字直接交给注意力后端。TRTLLM 注意力随后用普通的 Python 整数裁剪 Q/K/V 的 padding——不碰注意力掩码张量，不做 `nonzero` 搜索，不做 CUDA 标量回读（每一次回读都会强迫 GPU 停下来与 CPU 同步）。

| 指标（每次注意力调用） | 之前：packed-mask 校验 | 之后：packed-padding 快路径 |
|---|---:|---:|
| all-to-all → FMHA 间隔 p50 | 499.683 µs | 0.352 µs |
| all-to-all → FMHA 间隔 p95 | 698.308 µs | 0.384 µs |
| 中间 GPU 活动 | 244,925 | 0 |
| 中间 D2H 拷贝 | 97,970 | 0 |

以上为 4x NVIDIA B300 上的上游实测，MiniMax-H3 官方 starship 工作负载（1344×768、243 帧、49 次去噪更新），Nsight Systems 采样——细节与注意事项见[实测影响](#measured-impact)。本 PR 同时修复了 [#6358](https://github.com/vllm-project/vllm-omni/issues/6358)：在 TRTLLM 被自动选为默认后端的数据中心 Blackwell 上，旧的 mask 检查会直接拒绝 H3 的结构性 padding。

## 背景 {#background}

在这个 PR 之前，用默认设置在 B200 上跑 MiniMax-H3，结局是以下两者之一，都不好。要么服务端根本无法开始生成，报 `RuntimeError: Attention backend 'TRTLLM_ATTN' does not support attn_mask`（[issue #6358](https://github.com/vllm-project/vllm-omni/issues/6358)——TRTLLM 是数据中心 Blackwell 上的自动默认，所以不指定任何 flag 也会撞上）。要么在存在 packed TRTLLM 路径的分支上，每个去噪步骤都拖着一段隐藏的停顿：GPU 完成.all-to-all 交换后，每次注意力内核启动前都要空转约半毫秒。半毫秒听起来不多——但它按次计费：每次注意力调用、每个 transformer 层、每个视频请求的 49 次去噪更新中的每一次都要付。

根源是"谁知道什么"的错配。MiniMax-H3 把 packed 序列对齐到 64 行的倍数，所以每个注意力输入的布局是 `[real tokens, padding]`——好比一个托盘：真正的箱子后面塞空填充块，把托盘凑到标准尺寸。[PR #5779](https://github.com/vllm-project/vllm-omni/pull/5779) 让 TRTLLM 注意力对这个布局变得*正确*：后端检查 mask、从 Q/K/V 中裁掉 padding token、恢复带 padding 的输出形状。但这个检查**每次调用都在设备张量上做**：对 mask 求和（GPU 归约）、与预期前缀模式比较、用 `nonzero` 搜索边界、把 CUDA 标量读回主机——每次读取都是一次设备到主机的同步，会让流串行化。这就像仓库工人收到一份带着打印清单的托盘，却坚持在放行前把每个箱子开封重数——而且每个托盘、每天都这么干。清单（主机端已知的有效 token 数）一直都在，只是后端没有渠道接收它。

PR 对根因的表述：*"The producer already knows the valid-token boundary on the host, but `AttentionMetadata` had no explicit way to publish that trusted boundary together with canonical device-side cumulative lengths."*（生产方早已在主机上知道有效 token 边界，但 `AttentionMetadata` 没有显式的渠道，把这个可信边界连同规范的设备端累积长度一起发布出去。）于是每个层都在重建、复验一段早就由上游在 CPU 上算好的信息。

## PR #6542 改了什么 {#key-changes}

一句话概括这个想法：**让 packed 布局的生产方把主机侧的知识发布成一个小小的类型化记录，让后端自行选择是否信任它。** 具体是三个部件：

1. **新的元数据记录**——[`PackedPaddingMetadata`](https://github.com/vllm-project/vllm-omni/blob/51b7565f33017d74e2da7045327a65e8f048eaae/vllm_omni/diffusion/attention/backends/abstract.py)，位于 `attention/backends/abstract.py`：

   ```python
   @dataclass(frozen=True, slots=True)
   class PackedPaddingMetadata:
       """Producer-validated mask-free view of padding in a [real, pad] packing."""

       q_length: int          # 主机侧统计的真实 Q token 数
       kv_length: int         # 主机侧统计的真实 K/V token 数
       cu_seqlens_q: torch.Tensor   # 规范的 [0, q_length] 视图
       cu_seqlens_k: torch.Tensor   # 规范的 [0, kv_length] 视图
   ```

   累积长度张量（"cu_seqlens"——cumulative sequence lengths，varlen 内核用来在 packed batch 中定位每个文档的格式）被固定为规范的两元素 `[0, length]` 形状。这个形状选择很关键：消费方可以直接切 `cu_seqlens_kv[1:]` 得到内核想要的逐序列长度表，**全程不读设备标量**。

2. **能力契约**——后端声明 `supports_packed_mask_free()`。packed 生产方*只在*所选后端声明了该能力时才可以省略注意力掩码。MiniMax-H3 的 transformer 现在恰好在后端声明时发布 `packed_padding`（[`minimax_h3_transformer.py`](https://github.com/vllm-project/vllm-omni/blob/51b7565f33017d74e2da7045327a65e8f048eaae/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py)）；其他后端继续收到旧的 mask，行为与之前完全一致。

3. **TRTLLM 的 mask-free 路径**——[`trtllm_attn.py`](https://github.com/vllm-project/vllm-omni/blob/51b7565f33017d74e2da7045327a65e8f048eaae/vllm_omni/diffusion/attention/backends/trtllm_attn.py) 校验主机契约（全部在 Python 层：整数类型、边界、设备、dtype、形状），然后用普通切片裁剪 Q/K/V：

   ```python
   q = q[: packed_layout.q_tokens]   # valid_q_tokens 是 Python int
   k = k[: packed_layout.kv_tokens]  # 没有 .item()，没有 mask 归约
   v = v[: packed_layout.kv_tokens]
   ```

![之前：每次注意力调用都要在 Ulysses all-to-all 与 FMHA 内核之间做 mask 求和、前缀比较、nonzero 搜索和 10 次设备到主机同步——p50 间隔 499.7 µs。之后：后端直接读取生产方在主机侧发布的 PackedPaddingMetadata 整数并启动 FMHA——0.35 µs，零中间内核，零拷贝。]({{ site.baseurl }}/assets/figures/pr-6542-mask-free-packed-padding/fig1-attention-gap.svg)

注意这个类比的诚实边界：工人仍然*核对清单*（后端仍然校验元数据的类型、边界、设备和形状一致性——见[主机契约](#host-contract)）。消失的是在 GPU 上物理重数箱子这一步，而不是文书工作。

## 主机契约的细节 {#host-contract}

先说人话版本：后端信任生产方的数字，但前提是先确认这些数字*彼此自洽*——像边检官员信任你打印的行程单，但仍然核对日期对不对得上。`_prepare_packed_padding_layout` 会用响亮的 `ValueError` 拒绝：

- **非整数长度。** 每个长度必须是 Python `int`（`bool` 或 `torch.Tensor` 都会被拒绝）。这正是全部要义：一旦这些数字里有任何一个变成设备张量，消费它就意味着一次同步。
- **越界边界。** `q_length` 和 `kv_length` 必须落在 packed Q 和 K/V 序列范围之内。
- **不一致的最大值。** `max_seqlen_q`/`max_seqlen_k` 必须等于有效长度；独立发布的 `valid_kv_length`（既有的 `extra` 字段）必须与 `kv_length` 一致——两个事实来源不能互相矛盾。
- **非规范的累积长度。** 必须恰好两个元素、`int32`、与 Q/K 同设备。
- **错误的形状。** 快路径要求单一 packed batch（`physical_batch == 1`）。

在多请求侧，两处小改动让连续批处理走上同一条路径：

- TRTLLM 现在声明 `supports_multi_doc_packed_varlen()`，通用 packed 布局（[`_prepare_generic_packed_layout`](https://github.com/vllm-project/vllm-omni/blob/51b7565f33017d74e2da7045327a65e8f048eaae/vllm_omni/diffusion/attention/backends/trtllm_attn.py)）接受任意数量的真实文档，不再限于单个 `[real, pad]` 对。
- 批量生产方（[`batched_packing.py`](https://github.com/vllm-project/vllm-omni/blob/51b7565f33017d74e2da7045327a65e8f048eaae/vllm_omni/diffusion/models/minimax_h3/batched_packing.py)）过去对每个请求都发出*两个*累积边界——真实行一个、padding 尾巴一个——即使 64 行对齐让尾巴为空也照发。现在空文档被省略，因为"并非所有 varlen 内核都接受重复的内部边界"；契约测试断言边界严格递增。

最后，SAGE 量化路径顺带受益：判断序列是否够长以进行 SAGE 块量化，过去是 `torch.all(seq_lens >= block).item()`——又是一次同步。在快路径上主机本来就知道最小 KV 长度，这个检查变成一次普通的整数比较。

## 实测影响 {#measured-impact}

以下所有数字来自 [PR 作者的 Nsight Systems 采样](https://github.com/vllm-project/vllm-omni/pull/6542)——是上游测量，不是 cookbook 基准跑分。环境：4x NVIDIA B300 SXM6（每卡 267.7 GiB）、TP1/Ulysses4/Ring1、dense BF16 `TRTLLM_ATTN`、regional compile，MiniMax-H3 官方 starship 工作负载 1344×768、243 帧、49 次去噪更新；每次采样前先跑一次不采样的预热请求。

| 指标 | 之前：packed-mask 校验 | 之后：packed-padding 快路径 |
|---|---:|---:|
| 配对注意力样本数 | 9,797 | 9,680 |
| 间隔 p50 | 499.683 µs | 0.352 µs |
| 间隔 p95 | 698.308 µs | 0.384 µs |
| 中间 GPU 活动 | 244,925 | 0 |
| 中间 D2H 拷贝 | 97,970 | 0 |

"间隔"的口径是刻意收窄的：从**最后一个 Ulysses all-to-all 内核结束**到 **TRTLLM 主 FMHA 内核启动**。这正是 PR 要打击的杂务窗口。两点诚实说明：

- **这是间隔级测量，不是端到端加速声明。** PR 没有发布 E2E 时延差；它发布的是正确性锚点——4x B300 端到端跑完全部 49 次去噪更新，产出的 243 帧 MP4 的帧 SHA256 与此前验证过的 packed-fastpath 基线**完全一致**。
- 两个原始计数随样本数缩放：PR 之前每次注意力调用约 25 个中间 GPU 活动、恰好 10 次设备到主机拷贝。之后：两者皆零。

E2E 哈希之外的正确性覆盖：B300 上 37 个后端测试通过（唯一 skip 是该卡上刻意不支持的 INT8 SAGE 用例），159 个 CPU/模拟契约测试通过，一次真实的 TRTLLM 多文档交叉注意力运行与逐文档 SDPA 对比，相对 L2 误差 `0.000041`。

## 怎么用 {#how-to-use}

没有什么需要开启：当选中的后端声明了 mask-free packed-padding 能力、且模型的生产方发布了元数据时，快路径自动生效。对运维可见的只有后端选择本身。

{% include usage-cookbook.html modes=page.usage %}

如果你此前在 B200 上被 [#6358](https://github.com/vllm-project/vllm-omni/issues/6358) 的 `RuntimeError` 咬过，升级到本 PR 之后即是修复——通用能力检查不再在 packed TRTLLM 路径运行之前拒绝 MiniMax-H3 的结构性 padding mask。

## 怎么选 {#decision-cards}

{% include decision-cards.html items=page.decisions %}

## 限制与后续 {#limitations}

- **这不是任意 mask 支持。** 快路径只覆盖一种形状：生产方自有的、单 batch 的 `[real, pad]` 打包，padding 是后缀。TRTLLM_ATTN 现在拒绝*一切*非空 `attn_mask`——过去它只拒绝非前缀 mask；新的报错信息指向 packed-padding 出口或支持 mask 的后端。
- **后端仍然校验主机契约。** 撒谎的生产方——错误的 dtype、不在正确设备上的张量、互相矛盾的最大值——会以 `ValueError` 响亮失败，这正是设计好的失败方式：宁可抛异常，也不要悄悄裁错。
- **E2E 墙钟影响未单独报告。** 已发布的测量是 all-to-all → FMHA 间隔加上零中间活动计数，E2E 用作比特级一致的正确性检查（帧 SHA256 一致）。视频层面的墙钟差值仍待测量。
- **它在技术栈中的位置：** PR #5543（通用 mask 能力检查）rebase 到本 PR 之上——作者明确计划先合 #6542 再 rebase，这样 #5543 "可以校验由此产生的契约，而不必吞下这个实现"。
- 本文是 [系列 RFC #37](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/37) 规划的 MiniMax-H3 优化系列 8 篇中的第 3 篇，也是"非 GPU 热路径"主题的旗舰篇。站内相关阅读：[第 1 篇——模块化流水线](/zh/2026-08-24-understanding-pr-5720-minimax-h3-modular-pipeline/)、[第 2 篇——"四步"背后的三种契约](/zh/2026-08-24-understanding-pr-5991-minimax-h3-few-step-schedules/)。

## 参考 {#references}

- [PR #6542 — [perf] Add mask-free TRTLLM packed-padding path](https://github.com/vllm-project/vllm-omni/pull/6542)（2026-08-25 合入，commit [`51b7565`](https://github.com/vllm-project/vllm-omni/commit/51b7565f33017d74e2da7045327a65e8f048eaae)）
- [Issue #6358 — [Bug][B200] MiniMax-H3 B200 default TRTLLM_ATTN rejects packed padding attn_mask](https://github.com/vllm-project/vllm-omni/issues/6358)
- [PR #5779 — 本 PR 取代此前的 packed-mask 校验](https://github.com/vllm-project/vllm-omni/pull/5779)
- [PR #5543 — 通用 mask 能力检查，rebase 于 #6542 之上](https://github.com/vllm-project/vllm-omni/pull/5543)
- [`docs/design/feature/attention_backend_selection.md`](https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/attention_backend_selection.md)（上游后端选择设计）
- [系列 RFC #37 — MiniMax-H3 优化博客系列](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/37)
