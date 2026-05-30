<!--
知乎发布备忘（发布前可删除本注释块）

【推荐标题】
vLLM-Omni 优化 Wan2.2 图生视频：v0.16 → v0.20 三轮发版 H200 实测

【备选标题】
Wan2.2 I2V 在 vLLM-Omni 里怎么变快的？4×H200 延迟对比与优化路径

【封面图建议】
- 从 [index.md](index.md) 的 H200 retro 表格导出，或自制 832×480 / 720p 三版本 latency 对比图

【话题标签】
#vLLM #大模型推理 #视频生成 #Wan2.2 #扩散模型 #AIGC

【置顶评论模板】见文末
-->

# vLLM-Omni 优化 Wan2.2 图生视频：v0.16 → v0.20 三轮发版 H200 实测

**模型：** Wan-AI/Wan2.2-I2V-A14B-Diffusers  
**框架：** vLLM-Omni + vLLM  
**实测环境：** 4× NVIDIA H200  
**指标：** 单次请求端到端延迟 `latency_mean`（秒，越低越好）

---

## 写在前面

Wan2.2 是阿里通义万相系列的 Diffusion Transformer（DiT）视频模型，支持文生视频、图生视频（I2V）等任务。14B 的 I2V 版本采用 **双 Transformer MoE 级联**（低噪 / 高噪两个 expert），在线 serving 时每一发请求都要走完：文本编码 → 图像预处理 → VAE 编码 → DiT 去噪（4 steps）→ VAE 解码 → 返回视频。

vLLM-Omni 从 v0.16.0 开始逐步完善 Wan2.2 的在线 API 与多卡并行，到 v0.20.0 引入 fused DiT kernel、VAE/preprocess 优化，并合入 nightly 性能回归测试。

这篇文章回答三个问题：

1. **三轮稳定版之间，I2V 延迟到底差多少？**
2. **优化主要落在 pipeline 的哪几个阶段？**
3. **读者如何在自己的机器上复现对比？**

英文技术文档与完整数据表：  
https://github.com/hsliuustc0106/vllm-omni-cookbook/tree/main/diffusion/wan2.2

---

## 一、我们怎么测的

### 1.1 官方 CI 基准（2× H100）

从 v0.20.0 起，上游 vLLM-Omni 合入了固定 perf JSON：

`tests/dfx/perf/tests/test_wan22_i2v_vllm_omni.json`（PR #3063）

Nightly 在 **2× H100** 上跑三个 case，JSON 里写死了断言阈值（**不是**每个发版自动更新）：

| 配置 | 工作负载 | CI 基线 latency_mean |
|------|----------|----------------------|
| 单卡 | 832×480，81 帧，4 steps | **26.0 s** |
| USP2 + VAE 切分并行 + HSDP + slicing | 832×480，81 帧，4 steps | **21.6 s** |
| 同上 | 1280×720，121 帧，4 steps | **101.6 s** |

公共参数：`task=i2v`，并发 1，`seed=42`，`num-prompts=10`，开启 negative prompt。

### 1.2 本文 H200  retro 对比（4× H200）

为了对比 **v0.16.0 / v0.18.0 / v0.20.0** 三个 tag，我们在 **4× H200** 上跑了 workload **形状相同** 的 retro 测试（2026-05-20）：

- 随机 I2V 数据集
- 分辨率 / 帧数 / steps 与 CI JSON 一致
- API：`POST /v1/videos`
- 记录 `latency_mean`

**注意：** H200 数字与 CI 的 H100 阈值 **不可直接对比**；本文只比较 **同一硬件、同一 workload 形状** 下的发版差异。

---

## 二、核心结果（建议配一张表图）

### 2.1 三版本延迟对比

| 工作负载 | v0.16.0 | v0.18.0 | v0.20.0 | v0.18→v0.20 | v0.16→v0.20 |
|----------|---------|---------|---------|-------------|-------------|
| 单卡 · 832×480 | 31.33 s | 23.56 s | **22.17 s** | **−5.9%** | **−29.2%** |
| USP2 · 832×480 | 22.20 s | 20.26 s | **16.43 s** | **−18.9%** | **−26.0%** |
| USP2 · 1280×720 | 133.94 s | 93.67 s | **79.19 s** | **−15.5%** | **−40.9%** |

**一句话结论：**

- **v0.18 → v0.20**：多卡 480p 收益最大（**−18.9%**），主要来自 DiT fused kernel + VAE/preprocess。
- **v0.16 → v0.20**：单卡 480p 约 **−29%**；720p 约 **−41%**（见下方说明）。
- **720p 仍是成本大头**：即使 v0.20 也要 ~79 s/请求，分辨率与帧数决定下限。

### 2.2 720p 直观对比（可截图作正文配图）

```
v0.16.0  ████████████████████████████████  133.9 s
v0.18.0  ██████████████████████            93.7 s
v0.20.0  ██████████████████                79.2 s
```

> 发布时建议：把上表导入 Excel 做柱状图上传，或从 [index.md](index.md) retro 表格截图。

### 2.3 阅读说明（建议放在表格下方）

- **v0.16.0 的 USP2** 只有 `usp=2`、`use-hsdp`、`vae-use-slicing`，**没有** `vae-patch-parallel-size=2`（该 CLI 在 v0.16 之后才有）；当时可见 GPU 为 `0,1,3`（3 卡）。
- **v0.18 / v0.20** 使用完整 USP2 栈（含 VAE patch parallel、negative prompt、`random-request-config`），4 卡 `0,1,2,3`。
- v0.16 的 720p 行用 **3 条 prompt** 统计，其余行用 **10 条**；单次请求延迟仍具可比性。
- 版本栈配对：v0.16 + `vllm==0.16.0`，v0.18 + `vllm==0.18.0`，v0.20 + `vllm==0.20.0`。

---

## 三、一条 I2V 请求怎么走

每次 benchmark 请求的路径（在线 serving）：

```
客户端 → 文本编码 → 图像预处理 → VAE 编码输入
       → DiT 去噪（4 steps，MoE 双 expert）
       → VAE 解码 → 返回视频
```

在标准 **USP2 + VAE patch parallel + HSDP** 配置下：

- **480p**：时间大头在 **DiT 去噪**。
- **720p / 121 帧**：**VAE 与预处理**占比明显上升。

优化前先开 profiler：`--enable-diffusion-pipeline-profiler`，看 server log 里各 stage 耗时，**先优化最大的那块**。

---

## 四、优化路径：按发版拆解

### 4.1 v0.14 – v0.16：先把多卡 serving 搭起来

| 阶段 | 做了什么 | 意义 |
|------|----------|------|
| v0.14 | Wan2.2 T2V/I2V pipeline、Ulysses SP、CFG parallel、VAE patch parallel | 多卡 DiT / VAE 基础能力 |
| v0.16 | 在线 `/v1/videos` API、TP、HSDP | 生产可用的在线 serving |

没有这一阶段的并行栈，后面的 kernel fusion 无处发力。perf JSON 里的 `test_wan22_i2v_usp2_vae_patch2_hsdp_slicing` 就是「4 卡标准配方」。

### 4.2 v0.18.0：砍 serving 固定开销

| PR | 优化 | 信号 |
|----|------|------|
| #1715 | 降低在线 Wan2.2 进程间 IPC | 另一 workload 上 I2V **37.5 s → 31.0 s**（−17.5%） |
| #1504 | 多线程 safetensors 加载 | 冷启动更快 |
| #1979 | 对齐 offline / online 配置 | 减少「离线快、在线慢」 |

v0.18 是 **serving 效率** 发版。本文 H200 v0.18 retro 数字已经包含这一步。

### 4.3 v0.20.0：VAE / 预处理 + DiT kernel

**预处理与 VAE（对 720p 更敏感）：**

| PR | 内容 |
|----|------|
| #2963 | 去掉重复 video preprocess |
| #2852 | I2V 图像预处理阶段释放 GPU |
| #2391 | VAE FP32 → BF16 |

H200 信号：720p **93.67 s → 79.19 s**（与 DiT 优化叠加）。

**DiT 去噪环（480p USP2 收益最大）：**

| PR | 内容 |
|----|------|
| #2583 / #2585 | Fused RMSNorm、Fused AdaLayerNorm |
| #2393 | 更快的 RoPE |
| #2459 | 短 cross-attn 时跳过 Ulysses SP |
| #3327 | Flash Attention / CUBLAS 路径修复 |

H200 信号：USP2 @ 480p **20.26 s → 16.43 s**（**−18.9%**）。

**运行时与 CI：**

- #3232：rebase 到 vLLM v0.20.0（部分 delta 来自 vLLM 本体）
- #3063：合入 I2V perf JSON + nightly H100 回归

### 4.4 汇总：v0.18 → v0.20 在 H200 上

| 优化阶段 | H200 影响（标准 workload） |
|----------|---------------------------|
| v0.18 已有（IPC 等） | retro 基线 |
| 预处理 / VAE | 720p 贡献大（合计约 −15.5%） |
| DiT kernel | USP2 480p **−18.9%** |
| vLLM 0.20 rebase | 贯穿各 case |
| **合计** | **−5.9% / −18.9% / −15.5%** |

---

## 五、发版时间线

| 版本 | 时间 | I2V 性能要点 |
|------|------|--------------|
| v0.14.0 | 2026-01 | 首个稳定 Wan2.2 pipeline |
| v0.16.0 | 2026-02 | `/v1/videos` 在线 API；H200 retro **31.33 / 22.20 / 133.94 s** |
| v0.18.0 | 2026-03 | IPC、快速加载；H200 **23.56 / 20.26 / 93.67 s** |
| v0.20.0 | 2026-05 | Fused DiT + CI JSON；H200 **22.17 / 16.43 / 79.19 s** |
| v0.22.0 | 下一偶数版 | Pipeline parallel #2322、GPU FP8 #2920、VAE tiling #3111 |

---

## 六、如何复现（正文可精简，完整版放评论）

### 6.1 v0.20+ 官方 JSON（推荐入门）

```bash
cd /path/to/vllm-omni
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn

pytest -s tests/dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_wan22_i2v_vllm_omni.json
```

非 H100 机器需在 JSON 的 `benchmark_params` 里加 `"skip-performance-assertion": true`。

### 6.2 三版本 retro A/B

配置模板：`vllm-omni/benchmark_results/wan22_retro/config/`  
默认 `num-prompts=3`、`warmup-requests=0`（加快本地对比）。

```bash
# v0.20
bash benchmark_results/wan22_retro/v0.20.0/run_benchmark.sh

# v0.18（i2v 需 /v1/videos backend）
bash benchmark_results/wan22_retro/v0.18.0/run_benchmark.sh

# v0.16（同步 /v1/videos；跑完可用 cleanup.sh 清 GPU 进程）
bash benchmark_results/wan22_retro/v0.16.0/run_benchmark.sh
```

### 6.3 标准 serving 命令（与 perf JSON 一致）

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers --omni \
  --usp 2 --vae-patch-parallel-size 2 --use-hsdp --vae-use-slicing \
  --enable-diffusion-pipeline-profiler
```

---

## 七、给做优化的读者：六步 checklist

1. **固定 workload** — 复用 `test_wan22_i2v_vllm_omni.json`，不要自造分辨率。
2. **Profile 一条请求** — 开 `--enable-diffusion-pipeline-profiler`。
3. **判断瓶颈** — 预处理 / VAE / DiT / API-IPC。
4. **一次只改一点** — fused op、dtype、去重复 pass、SP 策略等。
5. **同硬件重跑** — 相同 GPU、`serve_args`、`CUDA_VISIBLE_DEVICES`。
6. **算 delta** — `(t_new - t_old) / t_old`。

---

## 八、总结

- vLLM-Omni 对 Wan2.2 I2V 的优化是 **分阶段** 的：先并行 serving（v0.14–0.16），再砍 IPC（v0.18），再在 v0.20 集中做 **VAE/preprocess + DiT kernel**。
- **v0.20 相对 v0.18**，多卡 480p 最值得看（**−18.9%**）；单卡 480p 稳步但幅度小（**−5.9%**）。
- **720p** 仍是线上成本核心；下一步 **v0.22** 的 pipeline parallel、FP8、VAE tiling 值得用 **同一套 JSON** 继续量。
- 官方 CI 的 26.0 / 21.6 / 101.6 s 是 **H100 阈值**；本文 H200 实测更快，但二者用途不同——CI 防回归，retro 看发版 delta。

---

## 附录：建议置顶评论

```
完整数据表与 index：
https://github.com/hsliuustc0106/vllm-omni-cookbook/tree/main/diffusion/wan2.2

上游 vLLM-Omni：
https://github.com/vllm-project/vllm-omni

Perf JSON：
https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_wan22_i2v_vllm_omni.json

Retro 结果目录（v0.16/v0.18/v0.20）：
vllm-omni/benchmark_results/wan22_retro/

有问题欢迎评论区交流，后续 v0.22 测完会更新一版。
```

---

## 附录：粘贴知乎编辑器的小技巧

1. **按 `---` 分段粘贴** — 每段单独进编辑器，避免一次粘贴格式错乱。
2. **表格** — 可先粘到飞书/Excel，再复制到知乎表格组件。
3. **代码块** — 知乎里用「代码块」格式；正文只保留短命令，长脚本放评论。
4. **删除文首 HTML 注释块** — 那是给你发布备忘用的，读者看不到也无妨，但知乎若原样显示则删掉更干净。
