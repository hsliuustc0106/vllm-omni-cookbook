---
layout: post
title: "深入理解 PR #6516 — SenseNova-U1.5-8B-MoT 与它的 8 步蒸馏 LoRA"
date: 2026-09-15 12:00:00 +0800
author: hsliuustc0106
summary: >-
  PR #6516 让 SenseNova-U1.5-8B-MoT 正式落地 vLLM-Omni：修好一个让 8 步蒸馏
  LoRA 静默失效的子串 bug，并给自回归 think 解码装上分页 KV + CUDA
  graph——think 端到端快一倍，1024² 八步出图约 0.9 秒。
tags: [SenseNova-U1.5, A800, H200]
category: PR Analysis
feature: lora
lang: zh
pair: /2026-09-15-understanding-pr-6516-sensenova-u15-distilled-lora/
permalink: /zh/2026-09-15-understanding-pr-6516-sensenova-u15-distilled-lora/
usage:
  - label: "离线 · 质量优先"
    blurb: "50 步 + CFG 4.0，1024×1024"
    title: "text_to_image.py · 全质量路径"
    code: |
      python examples/offline_inference/text_to_image/text_to_image.py \
        --model sensenova/SenseNova-U1.5-8B-MoT \
        --prompt "Close portrait of an elderly woman by a farmhouse window, warm natural light." \
        --width 1024 --height 1024 \
        --seed 42 --num-inference-steps 50 --cfg-scale 4.0 \
        --extra-body '{"think": false, "cfg_norm": "none", "timestep_shift": 3.0, "t_eps": 0.02}' \
        --output sensenova_u15_t2i.png
    note: >-
      上游建议开启 think 模式（"think": true）以获得更好的画质；它会走一遍
      PR #6516 加速过的自回归推理（reasoning）过程。
  - label: "离线 · 8 步 LoRA"
    blurb: "蒸馏少步路径，1024² 约 0.9 秒"
    title: "text_to_image.py · 8 步蒸馏路径"
    code: |
      python examples/offline_inference/text_to_image/text_to_image.py \
        --model sensenova/SenseNova-U1.5-8B-MoT \
        --lora-path SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors --lora-backend distill \
        --prompt "Close portrait of an elderly woman by a farmhouse window, warm natural light." \
        --width 1024 --height 1024 \
        --seed 42 --num-inference-steps 8 --cfg-scale 1.0 \
        --extra-body '{"think": false, "cfg_norm": "none", "timestep_shift": 3.0, "t_eps": 0.02}' \
        --output sensenova_u15_lora8.png
    note: >-
      这个 LoRA 必须配 --cfg-scale 1.0：它用 DMD 蒸馏、免 classifier-free
      guidance 运行；默认的 4.0 会把 guidance 施加两次，出图过曝、色带化。
  - label: "编辑 · 图生图"
    blurb: "25 步，think 开启"
    title: "image_edit.py · 油画风格化"
    code: |
      python examples/offline_inference/image_to_image/image_edit.py \
        --model sensenova/SenseNova-U1.5-8B-MoT \
        --prompt "Turn this into an oil painting" \
        --image input.png --resolution 1024 \
        --seed 42 --num-inference-steps 25 --cfg-scale 4.0 \
        --extra-args '{"think": true, "img_cfg_scale": 1.0, "cfg_norm": "none", "timestep_shift": 3.0}' \
        --output sensenova_u15_edit.png
  - label: "在线 · OpenAI API"
    blurb: "在线服务，文本 + 视觉"
    title: "vllm serve · omni 入口"
    code: |
      vllm serve sensenova/SenseNova-U1.5-8B-MoT --omni --port 8091

      python examples/online_serving/sensenova_u1/openai_chat_client.py \
        -s http://127.0.0.1:8091 -m img2text -i input.png -p "Describe this image."
    note: >-
      -s 接 base URL，客户端自己拼 /v1。合并时的在线冒烟：/health 正常，
      img2text 返回 1,921 字符的描述。
decisions:
  - when: "画质优先，延迟其次"
    pick: "50 步 + CFG 4.0（think 开）"
    why: "recipe 推荐的质量路径：PR 后 A800 上 1024² 约 10.97 s；think 模式额外走的推理过程被同一个 PR 提速约 4.6 倍。"
  - when: "交互延迟或批量吞吐"
    pick: "8 步蒸馏 LoRA + CFG 1.0"
    why: "A800 上 1024² 约 1.0 s 墙钟时间（H200 扩散阶段 624 ms）；融合本身实测免费（−0.23% vs 同步数对照），却改变了 1,048,576 像素中的 1,048,574 个——绝不要配 CFG 4.0。"
  - when: "理解型负载（think、图生文、对话）"
    pick: "保持分页解码开启（默认）"
    why: "CUDA graph 解码把 think 循环缩短 43–46%、GPU 利用率拉到 97.7%；VLLM_OMNI_SENSENOVA_PAGED_DECODE=0 留给调试与一致性比对。"
  - when: "显存空闲不足 ~40 GB"
    pick: "暂时别上这个模型"
    why: "bf16 下 1024² 峰值 34.3 GB、1536×2720 峰值 36.4 GB；上游只在 80 GB 级（A800、H200）验证过。"
  - when: "需要运行时换适配器"
    pick: "每个引擎一个融合配置"
    why: "蒸馏融合是单向的（没有卸载路径）、只作用于生成塔（generation tower），且动态 LoRA 管理会主动关闭 graph/缓存复用。"
  - when: "在 v0.26.x 部署并撞上 head_dtype 崩溃"
    pick: "升级到 v0.28+"
    why: "U1.5 服务在 v0.26.0 启动即崩（_DiffusionVllmModelConfig 缺 head_dtype）；#5877 修复。完整的 U1.5 支持最早随 v0.29 系列发布。"
---

## 摘要 {#tldr}

**PR #6516 让 SenseNova-U1.5-8B-MoT 成为 vLLM-Omni 的一等公民——而有趣的是"支持"这个词最终意味着什么。模型本来就能跑在现有的 U1 流水线上，但它配套的 8 步蒸馏 LoRA（distilled LoRA）加载时打印成功、实际却什么都没改；修这个静默失效（silent no-op）的过程中，又暴露出解码循环被主机端卡得太死，于是 PR 里长出了一套分页 KV + CUDA graph 的自回归 "think" 加速路径。** 打个比方：涡轮徽章装上了车，发动机却从没接通过它——而打开引擎盖，还发现油管本身也是拧着的。

| 指标（1× A800-80G，1024²，seed 42） | main | PR #6516 | Δ |
|---|---:|---:|---:|
| think 模式端到端，50 步（P50 of 10） | 27.747 s | 13.930 s | **−49.8%** |
| …其中 think 解码阶段 | 16.146 s | 3.534 s | **−78.1%** |
| …换 8 步蒸馏 LoRA 后总端到端 | — | 4.455 s | 8 步 vs 50 步 |
| T2I，50 步，CFG 4.0（n=5 中位数） | 11,726.9 ms | 10,973.6 ms | **−6.42%** |
| T2I，8 步 + 蒸馏 LoRA | 1,114.4 ms | 1,024.2 ms | −8.09%（见下） |
| 每次八步 T2I 生成的 `cudaLaunchKernel` | 24,244 | 7,782 | **−67.9%** |

8 步那一行要加脚注：`main` 上适配器被静默丢弃，所以那里的"之前"其实是 8 步的基础权重。真实的说法更窄也更好——融合后的适配器实测**免费**（相对同步数无 LoRA 对照 −0.23%），却让输出图像 1,048,576 个像素中的 1,048,574 个发生变化（MAE 58.62）：蒸馏的价值在步数本身，而应用它的代价为零。合并于 2026-09-01（commit
[`0288c3f`](https://github.com/vllm-project/vllm-omni/commit/0288c3f56eb4a4ff410194e586498e3bc8ff8362)），首次随 v0.29.0rc1 发布；关闭
[#6471](https://github.com/vllm-project/vllm-omni/issues/6471)。

> [!NOTE]
> 本文所有数字都来自 PR 评审线程的实测——作者在 1× A800-80G，评审人在 L20X 和 H200 上。本 cookbook 尚未跑过自己的 SenseNova 追踪，请把这些当作带 commit SHA 的上游报告数据，而不是 cookbook 基准。

## 背景 {#background}

**这一节要说明 PR 开启时真正坏掉的是什么：不是加载、不是运行——而是两个静默的问题，其中一个让一个"已支持"的功能完全不生效。** 像厨房里新到的咖啡机：通电、接单、出水，一切看起来都正常，只有并排对比两口才知道根本没萃出咖啡。

[SenseNova-U1.5-8B-MoT](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)
是商汤（SenseNova）的统一图像生成 + 理解模型——文生图、图生图、图生文、对话装在一个 checkpoint 里。MoT 指
Mixture-of-Transformers（混合变换器）：除理解侧权重外，checkpoint 还带一套生成专用投影（即 `*_mot_gen` 参数）。它不小——13 个分片、磁盘上 50.2 GB（其中 30.3 GB 是 fp32），bf16 加载后约 34 GB。

"支持"这部分确实简单，PR 对此也很坦诚：U1.5 保留 `model_type: neo_chat`，因此直接解析到现有 `SenseNovaU1Pipeline`，不需要 `--model-class-name`；相对 U1 只有 `config.json` 里两个字段翻转（`use_pixel_head` → `true`，流匹配头变成 `ConvDecoder`；`noise_scale_max_value` 8.0 → 16.0），而这两个字段 `SenseNovaU1Config` 本来就读。剩下的是两个真缺陷：

1. **8 步蒸馏 LoRA 是个静默 no-op。** 官方随模型发布
   `SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors`——一个 DMD 蒸馏适配器，用 8 个免 guidance 步换来 50 步画质。在 `main` 上，`--lora-backend distill` 被接受，然后警告
   `Pipeline does not support loading distilled LoRA weights for now`——接着用基础权重生成。PR 自己的 A/B 把这件事钉死：`main` 上带与不带 `--lora-path` 只差 +0.48%，在噪声范围内。什么都没生效。
2. **没人测过 think 模式的时间花在哪。** 模型的自回归推理（"think"）在流水线 forward 内部跑一个几百步的 token 循环——而这个循环在 vLLM-Omni 下从未被 profile 过。

还有一个早期采用者会踩的坑（读旧 issue 时值得知道）：v0.26.0 docker 上 U1.5 服务启动即崩（`'_DiffusionVllmModelConfig' object has no attribute 'head_dtype'`——
[#5795](https://github.com/vllm-project/vllm-omni/issues/5795)）。那是配置管道 bug，由
[#5877](https://github.com/vllm-project/vllm-omni/pull/5877) 修复（进 v0.28.0），与本 PR 无关——本 PR 是这个模型的*官方*支持、recipe 与 LoRA 路径。

## PR 改了什么 {#key-changes}

**三幕剧：让适配器真正生效；让同步数生成略快且更准；以及——在一场升级成 Nsight profile 评审的讨论之后——围绕分页 KV 缓存和 CUDA graph 重建自回归解码循环。** 如果第一幕是修涡轮的油路，第三幕就是发现整套供油系统原来是根浇水管。

### 第一幕 — 加载了却没生效的 LoRA {#act1-lora}

在 `SenseNovaU1Pipeline` 上实现 `load_lora_weights`（挂上共享的 `LoraLoaderMixin`）立刻撞出一个 loader bug，它解释了"两倍高度"之谜。融合投影（fused projection）的增量（delta）靠*堆叠参数映射*（stacked params mapping）拼装——规则声明"这个 checkpoint 名字贡献这个融合参数的那一片"。旧匹配器用子串包含：

```python
# vllm_omni/diffusion/lora/loader.py — 之前
if param_name not in base_key:   # ".qkv_proj" 确实 "in" "...qkv_proj_mot_gen"
    continue                     # …".qkv_proj_mot_gen" 也是——两条都命中
```

`.qkv_proj` 是 `.qkv_proj_mot_gen` 的子串，所以对一个生成塔参数，*两条规则都命中*，拼出来的 delta **是参数高度的两倍**。修复改为匹配名字尾部并在首个命中处停止：

```python
# vllm_omni/diffusion/lora/loader.py — 之后（PR #6516）
if not base_key.endswith(param_name):
    continue
...
break                            # 首条命中规则生效
```

流水线一侧声明映射时把更具体的 `_mot_gen` 模式排在前面，把 kohya 风格的 checkpoint 键名重命名（`lora_down`/`lora_up` → `lora_A`/`lora_B`），并以 fp32 折叠适配器——在 bf16 里缩放 `B` 会在矩阵乘前后各舍入一次。结果：588 个 LoRA 键融合进 168 个参数，通过每层自己的权重加载器做 TP 分片（先写进一份清零副本再相加——每个 rank 只碰自己的切片；这也顺手修掉了评审抓到的 TP=2 启动崩溃）。两个刻意的诚实细节：匹配*不到任何参数*的 LoRA 现在直接报错而不是静默跳过；融合后只保留一个哨兵值——fp32 状态字典约 1.5 GiB，否则会因融合是单向的而永久驻留。

![子串 bug：.qkv_proj 是 .qkv_proj_mot_gen 的子串，两条堆叠映射规则同时命中，融合增量高度是参数的两倍；PR #6516 之后按名字尾部匹配并在首个命中处停止]({{ site.baseurl }}/assets/figures/pr-6516-sensenova-u15-distilled-lora/fig1-lora-substring-bug.svg)

### 第二幕 — 一行 RMSNorm {#act2-rmsnorm}

`Qwen3RMSNorm.forward`（U1.5 的语言塔基于 Qwen3）用了 eager 类型转换链，在乘权重*之前*就舍入到 bf16。`F.rms_norm` 在内部保持 fp32 累加、只舍入一次：

```python
return F.rms_norm(hidden_states, self.weight.shape, self.weight,
                  self.variance_epsilon)
```

相对 float64 参考实现，12 组形状/精度组合的平均相对误差全部下降——bf16 4096×3584：1.890e-3 → **1.409e-3**；fp16 16384×8192：2.368e-4 → **1.761e-4**——同步数延迟还顺带改善约 6–8%。一个回归测试对照 float64 参考钉住了精度声明。

### 第三幕 — 发现浇水管的评审 {#act3-decode}

评审人的 Nsight Systems 剖析（一个 1024² 步、预热后的第三次 forward，L20X）重新定义了这个 PR：CFG-1 去噪路径为约 34.1 ms 的 GPU 工作发射 **1,877 个 kernel，却花掉 71–76 ms 主机墙钟时间**。kernel 平均 8.9 µs；每个 launch API 约 5 µs，发射间隙 31–35 µs。GEMM 只占端到端墙钟的 11–15%——这是主机发射受限（host-dispatch-bound），不是设备算力受限，"GEMM 调优在这里天花板很低"。

作者的修正归因进一步收窄：文生图 GPU 利用率 92.0%，但 **think 只有 68.5%**，空闲全在 AR 解码——8.7% `cudaLaunchKernel`、0.2% 同步、**约 86% 完全不在任何 CUDA 调用里**（Python/ATen 派发）。这排除了同步类修复，指向 CUDA Graph（把发射序列录一遍，之后整段重放）。拦路的是 KV 缓存：`DynamicCache.update` 每步用 `torch.cat` 拼长 K/V，解码根本没有可供捕获的静态形状。显而易见的修法——垫到桶宽再 mask 尾部——实测后被否：mask 每步花 7.96–11.55 ms，而无 mask 只要 1.03 ms。

答案是**分页 KV 缓存（paged KV cache）**：`flash_attn_varlen_func` 接受以*张量*传入的已用长度（`seqused_k`）和 `block_table`，于是缓冲保持桶宽、可捕获，而 kernel 只读有效前缀——一次捕获服务整个桶里的所有长度。在此之上，是被 profile 证明值得的三个小提升（3D RoPE 表每次 forward 建一次，而不是 1,008 次调用建 17 张不同的表；不再在后端前把 K/V 扩到 query 头数——那会把 GQA 形状藏起来、让 SDPA 的融合 kernel 判定永不触发；丢掉全零的解码 mask）。桶宽为 (512, 1024, 2048, 4096, 8192)，超过最后一档按步长增长；一个 think 请求通常在跨过 512 时重捕获一次。

![CUDA graph 为什么需要分页缓存：torch.cat 每步改变形状、无可捕获；垫桶加 mask 形状静态但 mask 每步 7.96–11.55 ms vs 无 mask 1.03 ms；分页缓存保持桶宽缓冲，flash_attn_varlen_func 经 seqused_k 只读有效前缀，一次捕获服务整桶]({{ site.baseurl }}/assets/figures/pr-6516-sensenova-u15-distilled-lora/fig2-paged-decode.svg)

这一幕的其余部分就是评审对话的代码化——包括*试过又放弃*的方案，这也是整个线程里最有用的一张表：

| 改动 | 结论 | 证据 |
|---|---|---|
| 每次 forward 只建一次 3D RoPE 表 | 保留 | 1,008 次调用只产出 17 张不同的表 |
| 后端之前不再展开 K/V | 保留 | 单独 +2.55%；mask 去掉后才划算 |
| 丢掉全零解码 mask | 保留 | 与上一行合计 −5.66% / −8.19% |
| CUDA graph 下的分页解码 | 保留 | **−29.93% / −31.27%**，仅切换开关 A/B，n=5 |
| `_repeated_blocks` 区域编译 | 放弃 | 发射数 19,974 → 19,974；撞 `recompile_limit (8)` 永久回退 eager，+2.9% |
| 移除强制的 `sdpa_fallback` | 放弃 | 初始化即死——其它后端反转布尔 mask，而 SenseNova 传的是加性浮点 mask |
| 移除 `.item()` 同步 | 放弃 | 只占空闲的 0.2% |
| 垫桶加 mask | 放弃 | 每步 7.96–11.55 ms vs 无 mask 1.03 ms |

评审还抓到初版的两处错误，都在 PR 内修复：`_generate_text`——T2T/I2T 的解码循环——调用 `_ar_step` 时*没有*带 `_generate_think` 创建的解码上下文，于是 recipe 声称图生文也能吃到分页/graph 加速、实际路径根本到不了；另一个 GQA 精度门禁断言对照参考实现 8/8 固定种子全胜，在一套受支持的 CUDA 栈上不成立（L20X + cu129 上 `kv_len=512` 时 7/8），改成了显式的不回归容差。

最后，PR 自己引入——又自己修掉——一个显存泄漏：解码 runner 最初按请求构建，每个请求捕获自己的 graph 并持有，每请求泄漏约 40 MiB 且不平缓。修复改为捕获进共享平台池、跨请求复用缓存与 runner（降到 0.2 MiB/请求），并在 sleep level 2 释放捕获。当动态 LoRA 管理器接管解码路径时会刻意禁用复用——请求间绑定的适配器不会改变复用检查能看到的东西（形状、精度、桶宽）。

## 关键改动 {#diff-walkthrough}

**整个 diff 20 个文件、约 2,000 行新增——但最后约 460 行里只有 86 行是实现，其余是测试（258 行）与 recipe/文档（116 行）。** 实现分成 LoRA 路径、解码路径和外围管道三块。

- [`vllm_omni/diffusion/lora/loader.py`](https://github.com/vllm-project/vllm-omni/blob/0288c3f56eb4a4ff410194e586498e3bc8ff8362/vllm_omni/diffusion/lora/loader.py) —
  `_prepare_lora_delta` 改为尾部匹配（`endswith`）并在首个命中处 break；补齐 mypy 注解。QwenImage 与 Wan2.2 的融合名互不重叠，所以它们选中的分片不变——有回归测试钉住。
- [`vllm_omni/diffusion/models/sensenova_u1/pipeline_sensenova_u1.py`](https://github.com/vllm-project/vllm-omni/blob/0288c3f56eb4a4ff410194e586498e3bc8ff8362/vllm_omni/diffusion/models/sensenova_u1/pipeline_sensenova_u1.py) —
  挂上 `LoraLoaderMixin`、`stacked_params_mapping`（`_mot_gen` 模式在前）、`load_lora_weights`，以及解码机制：`_decode_context()` 构建或复用 `(PagedDecodeCache, DecodeGraphRunner)`，`release_captured_graphs()` 在 sleep level 2 丢弃，`_ar_step()` 扩桶并重放 graph，`_generate_think` 与 `_generate_text` *都*在上下文内运行。`_warm_ar_decode()` 在启动时跑一次单 token 解码，因为引擎的哑请求是关 think 的 T2I，只练 prefill、从不练解码形状。
- [`vllm_omni/diffusion/models/sensenova_u1/paged_decode.py`](https://github.com/vllm-project/vllm-omni/blob/0288c3f56eb4a4ff410194e586498e3bc8ff8362/vllm_omni/diffusion/models/sensenova_u1/paged_decode.py) —
  新文件：分页缓存（block table、`seqused_k`、桶增长）与 CUDA graph runner，外加 `paged_decode_supported()` 与 `dynamic_lora_wrappers_present()` 两个门。
- [`vllm_omni/diffusion/models/sensenova_u1/sensenova_u1_transformer.py`](https://github.com/vllm-project/vllm-omni/blob/0288c3f56eb4a4ff410194e586498e3bc8ff8362/vllm_omni/diffusion/models/sensenova_u1/sensenova_u1_transformer.py) —
  `F.rms_norm`、提升的 3D RoPE（`_build_3d_rope`：t 占 head dim 一半，h/w 各四分之一）、K/V 保持真实头数、解码层里的单 token 分页注意力分支。
- [`vllm_omni/config/environment_variable_inventory.py`](https://github.com/vllm-project/vllm-omni/blob/0288c3f56eb4a4ff410194e586498e3bc8ff8362/vllm_omni/config/environment_variable_inventory.py) —
  注册 `VLLM_OMNI_SENSENOVA_PAGED_DECODE`（默认开；`0` 强制回退普通缓存）。
- 测试：蒸馏 LoRA 融合、loader 子串回归、RMSNorm float64 参考、分页解码端到端（桶增长、GQA 能力探针）等 16 处新增；`tests/diffusion/models/sensenova_u1/` 达到 86 通过，作者还用"逐段还原"验证过套件（每还原一段，恰好对应的测试变红）。

## 实测影响 {#measured-impact}

**有四组独立测量：作者的 A800、评审人的 L20X 与 H200 验证，以及一张把每个改动对应到每秒节省的按阶段归因表。** 像一张按部门开列的收据，而不是一个总数——每一笔节省从哪来清清楚楚。

整 PR、`main` → 合并头（作者，1× A800-80G，seed 42，n=5 中位数，同一会话）：

| 场景 | main | PR #6516 | Δ |
|---|---:|---:|---:|
| think 模式端到端 1024²，50 步（P50 of 10，think 开） | 27.747 s | 13.930 s | **−49.8%** |
| T2I 1024²，8 步 + 蒸馏 LoRA | 1,120.8 ms | 924.4 ms | **−17.52%**\* |
| T2I 1024²，50 步，CFG 4.0 | 11,862.3 ms | 10,560.7 ms | −10.97% |
| T2I 1536²，50 步，CFG 4.0 | 25,779.6 ms | 23,131.7 ms | −10.27% |

\* 面向用户的口径、非同口径对比：`main` 会静默丢弃适配器。

按阶段归因（作者，1× A800，think 开，2 次预热 + N=10，P50；该模型的 `stage_metrics` 只报一个 `diffusion` 阶段，因为整个 AR 循环跑在流水线 forward 内部）：

| 指标 | main（eager） | PR paged=0 | PR paged=1 | PR paged=1 + LoRA（8 步） |
|---|---:|---:|---:|---:|
| 端到端 | 27.747 s | 16.500 s | 13.930 s | **4.455 s** |
| AR prefill | 69.0 ms | 38.9 ms | 39.3 ms | 39.0 ms |
| think 解码 | 16.146 s | 6.213 s | 3.534 s | 3.522 s |
| 扩散执行 | 11.460 s | 10.251 s | 10.317 s | 0.854 s |

拆开看：`main → paged=0`（加载/融合、RMSNorm、RoPE 提升）是端到端 **−40.53%**（解码 −61.52%、扩散 −10.56%）；`paged=0 → paged=1`（分页 KV + CUDA graph）是端到端 **−15.57%**（解码 −43.11%）。各派发入口同样口径：图像编辑 −20.25%、图生文 −42.04%、文生文 −44.51%（仅切换开关）。

为什么是 graph，量化如下——三条臂，每条只差一个东西（think 开、8 步；kernel 时间来自 profile 运行、墙钟来自干净运行）：

| 臂 | KV 缓存 | 注意力 | graph | 墙钟 | kernel | 空闲 | GPU 利用率 |
|---|---|---|---|---:|---:|---:|---:|
| A `paged=0` | 精确长度 | SDPA flash | 无 | 8,259.5 ms | 5,179.2 ms | 3,080.3 ms | 62.71% |
| B | 分页 | `flash_attn_varlen` | 无 | 8,323.3 ms | 5,136.6 ms | 3,186.7 ms | 61.71% |
| C `paged=1` | 分页 | `flash_attn_varlen` | 有 | 5,196.9 ms | 5,076.9 ms | **120.0 ms** | **97.69%** |

![三条解码臂：只换分页缓存（A 到 B）毫无变化，把解码循环捕获成 CUDA graph（B 到 C）后空闲时间从 3,186.7 ms 塌缩到 120.0 ms——kernel 时间持平，省下的全是主机派发]({{ site.baseurl }}/assets/figures/pr-6516-sensenova-u15-distilled-lora/fig3-decode-arms.svg)

三条臂的 kernel 时间持平——消失的空闲是主机时间、不是设备工作；每次运行的 `think_chars` 都是 1,233。解码循环上 A→B 为 −1.10%、B→C 为 −45.79%：*分页缓存买到静态形状，graph 买到时间。*整条路径的主机 CPU 开销从 graph 之前的 37.29% 降到 2.31%（AR 窗口）/ 3.97%（扩散窗口）。两个窗口利用率都超过 96% 后，剩余构成以 GEMM 为主（AR 解码：GEMM 82.4%、elementwise 6.4%、attention 5.5%、norm 5.4%；扩散：78.9/9.2/6.8/4.9）——这就是为什么上游的下一根杠杆是量化与 TeaCache，而不是继续压发射数。

发射数——评审人最初定下的判据（一次 profile 的八步 T2I 生成）：`cudaLaunchKernel` **24,244 → 7,782（−67.9%）**，GPU kernel 25,924 → 8,232，GPU 利用率 86.1% → 93.1%。两腿的 `cudaGraphLaunch` 都是 0——T2I 根本不跑解码；这轮下降来自 RoPE 提升、不展开的 K/V 与去掉的 mask。

张量并行（2× A800，SYS/PCIe 无 NVLink）：think 20,678.3 → 5,221.0 ms（**−74.75%**），8 步 LoRA T2I 1,234.7 → 1,127.8 ms（−8.66%），输出位级一致；T2I 一腿从 TP 分片修复之后起测，因为那之前该配置启动即崩。评审侧 L20X（一次预热后三次测量）：PR 头上 graph 关 → 开，13.900 → 7.566 s（**−45.57%**）；NVML 峰值 TP=1 +366 MiB（+1.01%）、TP=2 每卡 +402 MiB（+2.02%），在 5% 回归门内。

H200 验证（评审人，1× H200 139 GiB，vLLM 0.28.0 / torch 2.13.0+cu130，seed 42，1024²，单次运行）：

| 场景 | 阶段延迟 | 峰值显存 |
|---|---:|---:|
| 50 步，CFG 4.0，think 关 | 9,017 ms | 34,384 MiB |
| 8 步，CFG 1.0，无 LoRA 对照 | 624.00 ms（77.85 ms/步） | 34,376 MiB |
| 8 步，CFG 1.0，蒸馏 LoRA | 623.85 ms（77.98 ms/步） | 34,364 MiB |
| 50 步，CFG 4.0，think 开，分页解码 | 81,622 ms | 34,594 MiB |

LoRA 那一行就是 no-op 修复的可见证据：**融合进 168 个参数**，同种子图像与无 LoRA 对照在 1,048,576 个像素中的 1,048,574 个上不同（MAE 58.62）——适配器不再是装饰。think 开那行含首请求编译（见下）。分页路径本身的显存代价、仅切换开关：调度钉在 8192 桶时峰值 +864 MiB（+2.46%），稳态 host RSS +23 MiB。graph 捕获发生时花 30–34 ms，且跨请求可复用——五个编辑请求共享一次捕获；一个 think 请求捕获两次（跨过 512 桶）。

融合的精度账本：RMSNorm 误差在全部 12 组形状/精度组合中下降（上表）；CFG-并行输出与单 GPU 位级一致；T2I 输出在泄漏修复前 commit 与合并头之间位级一致。唯一真实的漂移：分页 FA2 与 SDPA 相对 float64 参考*同样*准确（平均误差到四位有效数字相同），但彼此只吻合到 7.63e-06——在 382 步 argmax 循环里最终翻转一个 token，所以 think 模式的*文本*可能在两个后端间不同。首个不同步测得 top1–top2 间距 0.000 与 0.125，而 `|Δlogit|` 为 0.188 与 0.164——货真价实的贪心平局，不是损坏。全部四个 T2I 场景保持位级一致。

## 怎么用 {#how-to-use}

{% include usage-cookbook.html modes=page.usage %}

版本说明：完整的 U1.5 支持（本 PR）最早随
[v0.29.0rc1](https://github.com/vllm-project/vllm-omni/releases/tag/v0.29.0rc1)
（2026-09-10）发布。在 v0.26.x 上，服务会以背景一节的 `head_dtype` 报错崩溃——如果你卡在中间版本，v0.28.0 带配置修复。

首次运行前值得知道的三件事：

- **这个 LoRA 要配 CFG 1.0。** 它是 DMD 蒸馏、免 guidance 运行的；8 步配 CFG 4.0 会出过曝、色带化的图。适配器只作用于生成塔——理解行为不受影响。
- **分页解码默认开启**，并在设备或内置 `flash_attn_varlen_func` 不支持时自动回退普通缓存。`VLLM_OMNI_SENSENOVA_PAGED_DECODE=0` 用于调试或输出一致性比对时强制回退。
- **启动后第一个请求比稳态贵约 0.7 s**（实测分页开 718 ms、关 679 ms，编译缓存清空、三次取中位）——那是解码编译区域在热身，而且引擎现在启动时就尽力跑一次解码预热，不再把这件事留给第一个用户请求。

完整命令、硬件说明与两套验证环境（A800 与 H200）见上游
[SenseNova-U1.5 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/SenseNova/SenseNova-U1.5.md)。

## 怎么选 {#decision-cards}

{% include decision-cards.html items=page.decisions %}

## 局限与后续 {#limitations}

- **graph 捕获按请求发生，不按引擎。** 每个请求捕获自己的 graph（每个 30–34 ms；think 请求两次），因为捕获地址绑定缓存实例——复用省的是显存，不是捕获本身。再加上前面约 0.7 s 的首请求编译。
- **think 模式文本跨后端不是位级稳定。** 分页 FA2 与 SDPA 可能翻转贪心平局的 token（吻合度 7.63e-06）；图像不受影响。需要文本严格一致时用回退环境变量。相关的
  [#4636](https://github.com/vllm-project/vllm-omni/issues/4636) 跟踪 think 生成漂移破坏像素金标测试的问题。
- **蒸馏融合是单向的**——没有卸载、只作用生成塔、每次加载一个适配器文件，且运行时 LoRA 管理会禁用 graph/缓存复用。官方 U1 LoRA checkpoint
  （[#5642](https://github.com/vllm-project/vllm-omni/pull/5642)）仍然开着。
- **已验证硬件是 80 GB 级**（A800、H200、L20X）。bf16 下 1024² 峰值约 34 GB，40 GB 卡属于未测区域；更小的卡没有上游 recipe。
- **Preview checkpoint 的问题还开着。**
  [#5795](https://github.com/vllm-project/vllm-omni/issues/5795) 在 v0.26.0 上部署 `SenseNova-U1.5-8B-MoT-Preview` 撞了 `head_dtype` 崩溃；修复（#5877）进了 v0.28.0，但没人端到端确认过 Preview checkpoint——recipe 覆盖的是 `SenseNova-U1.5-8B-MoT`。
- **空闲时间如今是真·设备时间。** 两个窗口利用率 >96%、GEMM 占 kernel 时间约 79–82%，下一步的收益在量化与 TeaCache 式跳步，而不是继续压主机：SenseNova-U1 的按分支 TeaCache 开在
  [#6660](https://github.com/vllm-project/vllm-omni/pull/6660)
  （多分支 CFG 状态见
  [#5287](https://github.com/vllm-project/vllm-omni/pull/5287)），区域编译在
  [#4732](https://github.com/vllm-project/vllm-omni/pull/4732)——正是本 PR 测过并放弃的那个实验。家族内还开着：在线动态批
  （[#4156](https://github.com/vllm-project/vllm-omni/pull/4156)）、AR+DiT 分离
  （[#4033](https://github.com/vllm-project/vllm-omni/pull/4033)）、图像理解的流式输出
  （[#4049](https://github.com/vllm-project/vllm-omni/issues/4049)）。
- 蒸馏侧延伸阅读：[MiniMax-H3 少步调度一篇]({{ site.baseurl }}/2026-08-24-understanding-pr-5991-minimax-h3-few-step-schedules/) 讲了让"8 步、CFG 1"成为蒸馏属性而非采样技巧的 DMD2 契约家族。

## 参考 {#references}

- [PR #6516 — Support SenseNova-U1.5-8B-MoT and its distilled 8-step LoRA](https://github.com/vllm-project/vllm-omni/pull/6516)（2026-09-01 合并，commit [`0288c3f`](https://github.com/vllm-project/vllm-omni/commit/0288c3f56eb4a4ff410194e586498e3bc8ff8362)）
- [Issue #6471 — New Model: SenseNova-U1.5-8B-MoT](https://github.com/vllm-project/vllm-omni/issues/6471)（由本 PR 关闭）
- [Issue #5795 — 在 v0.26.0 上部署 SenseNova-U1.5-8B-MoT-Preview](https://github.com/vllm-project/vllm-omni/issues/5795)（仍开着；根因已由 v0.28.0 中的 #5877 修复）
- [PR #5877 — Fix SenseNova & use well-defined model configs](https://github.com/vllm-project/vllm-omni/pull/5877)（`head_dtype` 修复，2026-08-18 合并）
- [SenseNova-U1.5 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/SenseNova/SenseNova-U1.5.md)（上游，当前版——含 A800 与 H200 验证小节）
- [支持模型列表](https://github.com/vllm-project/vllm-omni/blob/main/docs/models/supported_models.md) · [diffusion 功能表](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion_features.md)（收录 U1.5 的上游文档）
- 评审线程实测：[评审人的 Nsight 剖析](https://github.com/vllm-project/vllm-omni/pull/6516#issuecomment-5386693054)、[作者的保留/放弃实验与整 PR A/B](https://github.com/vllm-project/vllm-omni/pull/6516#issuecomment-5398538652)、[L20X 复验](https://github.com/vllm-project/vllm-omni/pull/6516#issuecomment-5403824092)、[按阶段归因](https://github.com/vllm-project/vllm-omni/pull/6516#issuecomment-5469226403)、[CPU 开销与 kernel 构成回答](https://github.com/vllm-project/vllm-omni/pull/6516#issuecomment-5475061734)、[H200 验证](https://github.com/vllm-project/vllm-omni/pull/6516#issuecomment-5487475554)
- [MiniMax-H3 少步调度一篇 — PR #5991]({{ site.baseurl }}/2026-08-24-understanding-pr-5991-minimax-h3-few-step-schedules/)（DMD 蒸馏背景）
