# vLLM-Omni 优化 Wan2.2 图生视频：H200 实测与 v0.22 回归检查

- **模型：** Wan-AI/Wan2.2-I2V-A14B-Diffusers
- **框架：** vLLM-Omni + vLLM
- **实测环境：** NVIDIA H200
- **指标：** 单次请求端到端延迟 `latency_mean`（秒，越低越好）

Wan2.2 是通义万相系列的 Diffusion Transformer（DiT）视频模型，支持文生视频、图生视频（I2V）等任务。14B I2V 版本采用双 Transformer MoE 级联：一次在线请求会经过文本编码、图像预处理、VAE 编码、DiT 去噪、VAE 解码，再返回视频。

这篇文章主要整理两组数据：

1. v0.16.0 / v0.18.0 / v0.20.0 在 4× H200 上的 retro 对比。
2. v0.22.0 在标准 perf JSON 路径上的 H200 spot check，用来确认是否有明显性能回退。

完整英文数据表见：<https://github.com/hsliuustc0106/vllm-omni-cookbook/tree/main/diffusion/wan2.2>

---

## 一、测试口径

从 v0.20.0 起，上游 vLLM-Omni 合入了固定 perf JSON：

`tests/dfx/perf/tests/test_wan22_i2v_vllm_omni.json`

Nightly 在 2× H100 上跑三类 I2V 工作负载，CI 阈值是静态回归阈值，不是每个发版自动更新的性能结果：

| 配置 | 工作负载 | CI `latency_mean` 阈值 |
|------|----------|----------------------|
| 单卡 | 832×480，81 帧，4 steps | 26.0 s |
| USP2 + VAE patch parallel + HSDP + slicing | 832×480，81 帧，4 steps | 21.6 s |
| 同上 | 1280×720，121 帧，4 steps | 101.6 s |

H200 retro 数据在 2026-05-20 采集，用于比较 v0.16.0 / v0.18.0 / v0.20.0 三个 tag。v0.22.0 spot check 在 2026-06-08 采集，使用同一个标准 perf JSON 路径。

H100 CI 阈值和 H200 实测数字不能直接横向比较。本文只比较同一硬件、相近 workload 形状下的版本差异。

---

## 二、v0.16 到 v0.20：H200 retro 结果

| 工作负载 | v0.16.0 | v0.18.0 | v0.20.0 | v0.18→v0.20 | v0.16→v0.20 |
|----------|---------|---------|---------|-------------|-------------|
| 单卡，832×480，81f | 31.33 s | 23.56 s | **22.17 s** | −5.9% | −29.2% |
| USP2 + HSDP + slicing，832×480，81f | 22.20 s | 20.26 s | **16.43 s** | −18.9% | −26.0% |
| USP2 + HSDP + slicing，1280×720，121f | 133.94 s | 93.67 s | **79.19 s** | −15.5% | −40.9% |

几个关键观察：

- v0.18 → v0.20 收益最明显的是多卡 480p，延迟从 20.26 s 降到 16.43 s，约 −18.9%。
- 720p 仍然是成本大头，但 v0.16 → v0.20 从 133.94 s 降到 79.19 s，累计约 −40.9%。
- 单卡 480p 在 v0.20 继续小幅改善，说明优化不只来自多卡并行，也来自 preprocess、VAE 和 DiT kernel 路径。

数据口径上有三个注意点：

- v0.16 的 USP2 配置没有 `vae-patch-parallel-size=2`，当时可见 GPU 为 `0,1,3`。
- v0.18 / v0.20 使用完整 USP2 栈，GPU 为 `0,1,2,3`。
- v0.16 的 720p 行使用 3 条 prompt 统计，其余行使用 10 条 prompt。

---

## 三、v0.22 H200 spot check

v0.22.0 使用 `vllm==0.22.0`，沿用标准 `test_wan22_i2v_vllm_omni.json` 做 H200 spot check：

| 配置 | 工作负载 | v0.20.0 H200 retro | v0.22.0 spot check | Δ v0.20→v0.22 |
|------|----------|--------------------|--------------------|---------------|
| 单卡 | 832×480，81f，4 steps | 22.17 s | **20.46 s** | −7.7% |
| USP2 + HSDP + slicing | 832×480，81f，4 steps | 16.43 s | **16.52 s** | +0.5% |
| USP2 + HSDP + slicing | 1280×720，121f，4 steps | 79.19 s | **80.20 s** | +1.3% |

额外指标：

- throughput：0.0489 / 0.0605 / 0.0125 qps
- peak memory mean：76360 / 48135 / 56952 MB
- 三个被测 workload 均完成 10/10 请求
- 测试套件结果为 3 passed、1 skipped

结论是：v0.22 的标准 JSON 路径没有看到明显性能回退。单卡 480p 更快，多卡 480p 和 720p 与 v0.20 H200 retro 基本持平。

测试结束后，benchmark 结果已经写入，但 shutdown 阶段出现 diffusion worker / orchestrator cleanup 报错；没有观察到请求失败。

---

## 四、优化路径怎么理解

一条 Wan2.2 I2V 在线请求大致经过：

```text
客户端 → 文本编码 → 图像预处理 → VAE 编码
       → DiT 去噪（4 steps，双 expert）
       → VAE 解码 → 返回视频
```

按发版拆开看：

| 阶段 | 重点 | 影响 |
|------|------|------|
| v0.14–v0.16 | Wan2.2 pipeline、Ulysses SP、CFG parallel、VAE patch parallel、`/v1/videos` API、TP、HSDP | 先把多卡在线 serving 搭起来 |
| v0.18 | 降低在线 IPC 开销、多线程 safetensors 加载、对齐 offline / online 配置 | 减少 serving 固定开销 |
| v0.20 | 去重 video preprocess、I2V 预处理释放 GPU、VAE BF16、fused norm、RoPE、Flash Attention 路径修复 | VAE / preprocess 和 DiT kernel 同时受益 |
| v0.22 | 标准 JSON 路径 spot check | 没有看到明显 H200 回退 |

在 480p 多卡配置下，DiT 去噪路径的优化更容易体现；到 720p / 121 帧时，VAE 和预处理的占比会明显上升。

---

## 五、如何复现

推荐先用 v0.20+ 的官方 perf JSON 跑通：

```bash
cd /path/to/vllm-omni
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn

pytest -s tests/dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_wan22_i2v_vllm_omni.json
```

如果不是 H100，需要在 JSON 的 `benchmark_params` 中设置 `"skip-performance-assertion": true`，否则会触发 H100 阈值断言。

标准 serving 命令如下：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers --omni \
  --usp 2 --vae-patch-parallel-size 2 --use-hsdp --vae-use-slicing \
  --enable-diffusion-pipeline-profiler
```

做版本 A/B 时，尽量固定 GPU、`serve_args`、分辨率、帧数、steps、prompt 数和 warmup 策略。否则 delta 很容易混入硬件和 workload 差异。

---

## 六、总结

vLLM-Omni 对 Wan2.2 I2V 的优化是分阶段发生的：先完成在线 API 和多卡 serving，再降低 IPC 等固定开销，随后在 v0.20 集中优化 VAE / preprocess 和 DiT kernel。

H200 retro 数据显示，v0.20 相比 v0.18 在多卡 480p 上有约 −18.9% 的延迟改善；相比 v0.16，720p 累计改善约 −40.9%。v0.22 的 H200 spot check 沿用标准 perf JSON，没有看到明显回退：三组 latency 分别为 20.46 / 16.52 / 80.20 s。
