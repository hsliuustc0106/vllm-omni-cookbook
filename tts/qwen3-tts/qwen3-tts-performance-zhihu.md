<!--
知乎发布备忘（发布前可删除本注释块）

【推荐标题】
vLLM-Omni 优化 Qwen3-TTS：v0.20 → main 28ce618f（vLLM 0.21.0 + #3732）实测，高并发怎么快 10 倍？

【备选标题】
Qwen3-TTS 性能怎么测？TTFP / RTF / throughput、#3732 Code2Wav 与 #3662 高并发

【封面图建议】
- 三指标并排：c=64 下 TTFP / RTF / throughput 柱状图（v0.20 vs main 标准 vs main 高并发）
- c=64 TTFP 对比：v0.20（7.8s）vs main 标准（7.9s）vs main 高并发（351ms）
- #3732 A/B：c=1 TTFP 52→48 ms；c=8 92→81 ms；c=64 RTF 1.55→1.44

【话题标签】
#vLLM #大模型推理 #语音合成 #TTS #Qwen3 #AIGC

【置顶评论模板】见文末
-->

# vLLM-Omni 优化 Qwen3-TTS：v0.20 → main 28ce618f（vLLM 0.21.0 + #3732）实测，高并发怎么快 10 倍？

**模型：** Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice  
**框架：** vLLM-Omni + vLLM  
**实测环境：** 2× NVIDIA L20X  
**核心指标：** **TTFP**（首包延迟）、**RTF**（实时率）、**throughput**（音频吞吐）— 下文所有结论均围绕这三项展开

---

## 核心指标说明

Qwen3-TTS 性能看 **三个 metric**（与上游 `benchmarks/tts/README.md` 一致）：

| 指标 | 含义 | 越低/越高越好 | 主要场景 |
|------|------|---------------|----------|
| **TTFP** | 首包音频延迟（ms） | **越低越好** | 交互体验、流式播放；**c=1** 延迟 SLO |
| **RTF** | 墙钟时间 ÷ 生成音频时长 | **越低越好**（<1 = 快于实时） | 单请求端到端算力效率 |
| **throughput** | 每秒产出的音频时长（audio-s / s） | **越高越好** | **c≥8**  sustained load 容量 |

CI 对 latency / throughput 阶段记录 **median TTFP / RTF** 与 **audio_throughput**；本文 L20X retro 沿用同一口径。

**怎么读：**

- **c=1**：主要看 TTFP + RTF；throughput 无意义。
- **c≥8**：三项都要看 — TTFP 反映排队/首包体验，RTF 反映单请求效率，throughput 反映系统总产能。

---

## 写在前面

Qwen3-TTS 是通义千问系列的流式 TTS 模型，在 vLLM-Omni 里走 **两阶段 pipeline**：

```
Talker（文本 → 8 层 RVQ codec）→ Code2Wav（codec → 波形）
```

在线 serving 走 OpenAI 兼容接口 `POST /v1/audio/speech`，支持 **预设音色（CustomVoice）**、**音色设计（VoiceDesign）**、**声音克隆（Base）** 等任务。

v0.20.0 合入了通用 TTS benchmark（`test_tts.json`，PR #2835）；main 在 vLLM 0.21.0 上持续优化：**#3485**（延迟修复）、**#3662**（高并发 deploy）、**#3732**（Code2Wav CUDA graph 默认开启，2026-05-23 合入）。

这篇文章回答五个问题：

1. **v0.20.0 → main（`28ce618f`），TTFP / RTF / throughput 各改善多少？**
2. **#3732 在 c=1 / c=8 / c=64 分别带来什么？**
3. **为什么 c=8 是「并发悬崖」？**
4. **#3662 高并发 deploy 对三项指标分别解决了什么？**
5. **读者如何复现？**

英文数据表与完整 JSON：  
https://github.com/hsliuustc0106/vllm-omni-cookbook/tree/main/tts/qwen3-tts

---

## 一、我们怎么测的

### 1.1 官方 CI 基准（2× H100）

上游 perf JSON：`tests/dfx/perf/tests/test_tts.json`

| 阶段 | 并发 c | TTFP | RTF | Throughput |
|------|--------|-----:|----:|-----------:|
| latency | 1 | **150 ms** | **0.15** | — |
| throughput | 8 | **1500 ms** | **0.30** | 记录 |
| throughput | 16 / 64 | 更高 TTFP 阈值 | 更高 RTF 阈值 | 记录 |

任务类型：`default_voice`（预设音色）、`voice_design`（自然语言描述音色）、`voice_clone`（Base 模型 + 参考音频）。

### 1.2 本文 L20X retro（2× L20X，2026-05-24）

| 维度 | 设置 |
|------|------|
| 模型 | Qwen3-TTS-12Hz-1.7B-**CustomVoice** |
| GPU | `CUDA_VISIBLE_DEVICES=2,3`（Talker → 逻辑 GPU0，Code2Wav → 逻辑 GPU1） |
| 对比版本 | **v0.20.0 tag** vs **main `28ce618f`**（vLLM **0.21.0**，含 #3732） |
| 低延迟 | c=1，`num-prompts=3`，bundled smoke/design 数据集（**2026-05-24 复测**） |
| 高并发 | c=**8 / 16 / 64**，prompts **80 / 128 / 128**（与 CI throughput 一致） |
| Deploy | 默认 `qwen3_tts.yaml`；另测 main 专用 **`qwen3_tts_high_concurrency.yaml`**（#3662） |

**注意：**

- L20X 数字与 CI 的 H100 阈值 **不可直接对比**；本文只比较 **同一硬件、同一 workload** 下的版本差异。
- main 跑 benchmark 需 pin **`transformers==5.8.1`**（5.9.0 会打挂 Code2Wav）；本地 bench 需 **unset HTTP_PROXY**。

---

## 二、核心结果（TTFP / RTF / throughput）

**main（标准 deploy）** 列 = **`28ce618f`**，含 [#3732](https://github.com/vllm-project/vllm-omni/pull/3732) Code2Wav cudagraph 默认开启（`qwen3_tts.yaml` stage-1 `enforce_eager: false`）。

### 2.1 单请求（c=1）：TTFP + RTF

| 任务 | v0.20.0 TTFP | main TTFP | Δ TTFP | v0.20 RTF | main RTF | Δ RTF |
|------|-------------:|----------:|-------:|----------:|---------:|------:|
| default_voice | 59 ms | **48 ms** | **−19%** | 0.145 | 0.147 | ~持平 |
| voice_design | 63 ms | **46 ms** | **−27%** | 0.148 | 0.147 | ~持平 |

**一句话：** main + vLLM 0.21 在 **TTFP** 上稳定快 **~20%**；**RTF** 持平或略好。c=1 不看 throughput。

### 2.2 高并发 TTFP（ms，越低越好）

| 任务 | c | v0.20.0 | main（标准 deploy） | main（#3662 高并发） |
|------|--:|--------:|--------------------:|---------------------:|
| default_voice | 8 | 214 | **81** | 83 |
| default_voice | 16 | 1179 | 974 | **118** |
| default_voice | 64 | 7805 | 7861 | **351** |
| voice_design | 8 | 216 | **81** | 84 |
| voice_design | 16 | 1089 | 817 | **127** |
| voice_design | 64 | 7207 | 6743 | **386** |

### 2.3 高并发 RTF（越低越好）

| 任务 | c | v0.20.0 | main（标准） | main（#3662 高并发） |
|------|--:|--------:|-------------:|---------------------:|
| default_voice | 8 | 0.249 | **0.245** | 0.243 |
| default_voice | 16 | 0.436 | 0.426 | **0.357** |
| default_voice | 64 | 1.554 | 1.443 | **0.996** |
| voice_design | 8 | 0.251 | **0.244** | 0.250 |
| voice_design | 16 | 0.447 | 0.430 | **0.374** |
| voice_design | 64 | 1.641 | 1.557 | **1.093** |

### 2.4 高并发 throughput（audio-s / s，越高越好）

| 任务 | c | v0.20.0 | main（标准） | main（#3662 高并发） |
|------|--:|--------:|-------------:|---------------------:|
| default_voice | 8 | 19.8 | **31.4** | 29.2 |
| default_voice | 16 | 35.9 | 37.0 | **40.9** |
| default_voice | 64 | 36.0 | 36.9 | **60.8** |
| voice_design | 8 | 31.2 | **31.6** | 30.8 |
| voice_design | 16 | 34.9 | 36.0 | **41.6** |
| voice_design | 64 | 33.7 | 36.1 | **55.0** |

‡ **#3662 高并发列** 来自 **`e7644daa`** 复测（hiconc stage-1 仍为 `enforce_eager: true`）。

### 2.5 c=64 直观对比（建议作配图）

**TTFP：**

```
default_voice @ c=64  TTFP

v0.20.0          ████████████████████████████████████████  7805 ms
main 标准 deploy  ████████████████████████████████████████  7861 ms
main 高并发 #3662  ██                                       351 ms
```

**RTF + throughput（default_voice @ c=64）：**

| 版本 | RTF | throughput |
|------|----:|-----------:|
| v0.20.0 | 1.554 | 36.0 |
| main 标准 | 1.443 | 36.9 |
| main 高并发 | **0.996** | **60.8** |

### 2.6 三指标结论汇总

| 对比 | c=8 | c=16 | c=64 |
|------|-----|------|------|
| v0.20 → main（标准 deploy） | TTFP **−62%**；RTF ~持平；tp **+59%** | TTFP **−21–25%**；RTF ~持平 | 三项 ~持平 |
| main 标准 → main 高并发（#3662） | 三项 ~持平 | TTFP **−85–87%**，RTF **−14–16%**，tp **+12%** | TTFP **−94–96%**，RTF **−33–35%**，tp **+50–75%** |

**三句话结论：**

1. **v0.20 → main（标准 deploy，`28ce618f`）**：c=8 是最大赢家 — TTFP **−62%**，throughput **+59%**；c=64 三项几乎不变。
2. **main 标准 → main 高并发（#3662）**：c=8 三项持平；**c=16/64** 三项齐升 — TTFP **−85–96%**，RTF **−14–35%**，throughput **+12–75%**。
3. **做线上高并发，务必换 deploy 配置** — 默认 yaml 下 TTFP 爆炸、throughput 平台化（[#272](https://github.com/vllm-project/vllm-omni/issues/272)）。

### 2.7 #3732 增量：Code2Wav eager vs cudagraph（同 commit 完整 A/B）

[#3732](https://github.com/vllm-project/vllm-omni/pull/3732)（2026-05-23 合入）把 Code2Wav 内层 CUDA graph 与 `enforce_eager` 挂钩，并 flip 默认 deploy。**§2.1–2.4 的 main 列已包含 #3732**；下表是 **同 commit `28ce618f` 上仅切换 stage-1 eager/cudagraph** 的隔离 A/B（`main-post3732-eager/` vs `main-post3732/`）。

**default_voice：**

| c | TTFP eager | TTFP cudagraph | Δ TTFP | RTF eager | RTF cudagraph | Δ RTF | tp eager | tp cudagraph | Δ tp |
|--:|-----------:|---------------:|-------:|----------:|--------------:|------:|---------:|-------------:|-----:|
| 1 | 52 ms | **48 ms** | **−8%** | 0.153 | **0.147** | **−4%** | 6.6 | 6.8 | +3% |
| 8 | 92 ms | **81 ms** | **−12%** | 0.248 | **0.245** | −1% | 30.7 | **31.4** | **+3%** |
| 16 | 955 ms | 974 ms | ~持平 | 0.428 | 0.426 | ~持平 | 36.7 | 37.0 | +1% |
| 64 | 8085 ms | **7861 ms** | **−3%** | 1.549 | **1.443** | **−7%** | 36.5 | 36.9 | +1% |

**voice_design** 同趋势：c=1 TTFP **−14%**，c=8 **−13%**，c=64 RTF **−1%**。

| c | #3732 效果 | 能否替代 #3662？ |
|---|-----------|-----------------|
| **c=1** | TTFP **−8–14%**，RTF **−4%** | — |
| **c=8** | TTFP **−12%**，tp **+3%** | 否（悬崖仍在） |
| **c=16** | ~持平 | 否 |
| **c=64** | RTF **−7%**；TTFP 仅 **−3%** | **否** — std TTFP 仍 ~7.9 s，hiconc **351 ms** |

Opt-out：`--stage-overrides '{"1": {"enforce_eager": true}}'`

**PR 分支 c=10 微基准**（40 prompts，更极端对比）：TTFP **509 → 117 ms（−77%）**，RTF **0.30 → 0.21**，throughput **30.6 → 43.8 audio-s/s**。

---

## 三、一条 TTS 请求怎么走

```
客户端 POST /v1/audio/speech
    → Talker（AR 自回归，产出 codec token）
    → 共享内存 async_chunk 流式传给 Code2Wav
    → Code2Wav（codec → PCM 波形，流式返回）
```

**三项 metric 在 pipeline 上的对应关系：**

- **TTFP** — 从请求发出到收到第一帧 PCM 的时间；Talker 首 token + Code2Wav 首 chunk 的叠加。
- **RTF** — 整段音频生成完成的墙钟时间 ÷ 音频时长；两阶段串行/流水共同决定。
- **throughput** — 所有并发请求在 wall time 内产出的总音频时长；高并发下受 codec batch、GPU 调度影响最大。

低并发优先看 **TTFP**；高并发 **三项齐看** — 只看 TTFP 会漏掉「首包慢但吞吐还行」或「首包还行但 RTF/吞吐平台化」的情况。

---

## 四、为什么 c=8 是「悬崖」？

社区 issue [#272](https://github.com/vllm-project/vllm-omni/issues/272) 记录：Qwen3-TTS 在 **c=4 → c=8** 时 TTFP 会跳 **4–6×**，根因之一是 **Code2Wav 侧 codec batch size = 1** 的瓶颈。同时 **throughput 在 c=4–8 附近平台化** — TTFP 暴涨但产能不再线性增长。

本文 L20X 数据也印证这一点（default_voice）：

```
c=1   TTFP  ~47–59 ms   RTF ~0.15        throughput N/A
c=8   v0.20  214 ms      main 标准 81 ms   tp 19.8 → 31.4 (+59%)
c=64  v0.20  7805 ms     main 标准 ~7860 ms  tp ~37（标准 deploy 几乎救不回来）
      main 高并发 351 ms                    tp 60.8（#3662 三项齐升）
```

CI 在 **c=8** 设了 throughput 回归哨兵 — 同时记录 TTFP / RTF / throughput，就是为了抓这类退化。

---

## 五、优化路径：按 PR 拆解

### 5.1 v0.20.0：benchmark 与基础 serving

| PR | 内容 |
|----|------|
| #2835 | 通用 TTS benchmark（`test_tts.json`、`bench_tts.py`） |
| #2383 | Deploy YAML 自动加载 |
| #2341 | Code2Wav native decoder |

### 5.2 main：延迟修复（标准 deploy）

| PR | 内容 | retro 信号（TTFP / RTF / throughput） |
|----|------|--------------------------------------|
| #3485 | Qwen3-TTS 延迟回归修复 | c=1 TTFP **−19–27%**；c=8 TTFP **−62%**，tp **+59%**（含后续 #3732 的 main 列） |
| #3732 | Code2Wav cudagraph 默认开启 | 见 **§2.7**：c=1 TTFP **−8–14%**；c=8 **−12%**；c=64 RTF **−7%** |
| #2376 | Code2Wav CUDA graph | 解码路径加速 |
| #3232 | rebase vLLM 0.21.0 | 运行时栈升级 |

### 5.3 main：高并发专项（#3662）

**PR [#3662](https://github.com/vllm-project/vllm-omni/pull/3662)** 新增 `qwen3_tts_high_concurrency.yaml`：

| 配置项 | 默认 deploy | 高并发 deploy |
|--------|-------------|---------------|
| Talker `max_num_seqs` | 10 | **64** |
| Code-predictor prefix CUDA graphs | 关 | **开**（bucket 64） |
| Code2Wav graph capture sizes | 默认 | ** tuned**（25/73/97/169/325） |
| GPU 布局 | 可单卡 | **2 卡**（Talker + Code2Wav 分离） |

**仅 main 可用** — v0.20.0 tag 里没有这个 yaml。

retro 效果（default_voice）：c=16/64 上 **TTFP −85–96%**、**RTF −14–35%**、**throughput +12–75%**（见 §2.6）。

高并发 serve 示例：

```bash
export CUDA_VISIBLE_DEVICES=0,1
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni \
  --deploy-config vllm_omni/deploy/qwen3_tts_high_concurrency.yaml
```

### 5.4 #3732 技术细节

PR [#3732](https://github.com/vllm-project/vllm-omni/pull/3732) 实现 `_maybe_enable_decoder_cudagraph()`，当 stage-1 `enforce_eager: false` 时捕获 Code2Wav inner CUDA graph（11 shapes）。**完整 c=1/8/16/64 A/B 数据见 §2.7**。

与 #2376 的关系：#2376 提供 graph 基础设施；#3732 将其 **接入 deploy 默认值** 并支持 eager 回退。

## 六、发版 / 分支对照

| 来源 | vLLM-Omni | vLLM | 高并发 deploy |
|------|-----------|------|---------------|
| v0.20.0 tag | 0.20.0 | 0.20.0 | ❌ |
| main `28ce618f` | 0.20.1.dev175 | **0.21.0** | ✅ #3662 + **#3732** |

---

## 七、如何复现

### 7.1 官方 JSON（CI 同款）

```bash
cd /path/to/vllm-omni
export CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn
pytest -s tests/dfx/perf/scripts/run_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_tts.json
```

### 7.2 本文 retro 脚本

```bash
# v0.20.0 baseline
bash benchmark_results/qwen3_tts_retro/v0.20.0/run_benchmark.sh
bash benchmark_results/qwen3_tts_retro/v0.20.0/run_benchmark_throughput.sh

# #3732 A/B — eager vs cudagraph（28ce618f）
bash benchmark_results/qwen3_tts_retro/main-post3732-eager/run_benchmark.sh
bash benchmark_results/qwen3_tts_retro/main-post3732-eager/run_benchmark_throughput.sh

# cudagraph 默认
bash benchmark_results/qwen3_tts_retro/main-post3732/run_benchmark.sh
bash benchmark_results/qwen3_tts_retro/main-post3732/run_benchmark_throughput.sh

# 合并前 baseline（e7644daa）
bash benchmark_results/qwen3_tts_retro/main/run_benchmark.sh
bash benchmark_results/qwen3_tts_retro/main/run_benchmark_throughput.sh

# 高并发 + #3662 deploy（仅 main）
bash benchmark_results/qwen3_tts_retro/main/run_benchmark_hiconc_throughput.sh
```

配置与 JSON 产物：`vllm-omni/benchmark_results/qwen3_tts_retro/`

### 7.3 手动 smoke

```bash
python benchmarks/tts/bench_tts.py \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --task default_voice \
  --dataset-path benchmarks/build_dataset/seed_tts_smoke \
  --concurrency 1 8 16 64 \
  --num-prompts 20 \
  --output-dir ./results
```

---

## 八、给做 TTS serving 的读者：Checklist

1. **固定 workload** — 用 `test_tts.json` 或 retro config，不要自造 prompt 长度。
2. **分清 deploy** — 低延迟用默认 yaml；**c≥16 换高并发 yaml**。
3. **看对 metric** — c=1 看 **TTFP + RTF**；c≥8 **三项齐看**（TTFP / RTF / throughput）。
4. **盯 c=8** — CI 哨兵；若 TTFP 突然翻倍或 throughput 不涨，先查 codec batch / 并发配置。
5. **环境坑** — main 上 pin transformers 5.8.1；bench 时 unset 代理。
6. **同硬件 A/B** — 相同 GPU 对、相同 `num-prompts` / c。

---

## 九、总结

- **v0.20 → main（标准 deploy，`28ce618f`）**：c=1 **TTFP −19–27%**；c=8 **TTFP −62% + throughput +59%**；c=64 三项 ~持平。
- **#3732** 完整 A/B（c=1/8/16/64）：c=1 TTFP **−8–14%**；c=8 TTFP **−12%**；c=64 RTF **−7%**；c=64 仍需 **#3662** 解决 TTFP 悬崖。
- **#3662 高并发 profile** 才是 c=16/64 的钥匙：TTFP 从 **~7–8 s → ~0.35 s**，RTF 从 **~1.6 → ~1.0**，throughput 从 **~37 → ~61 audio-s/s**。
- TTS 优化要 **分场景、分 metric**：低延迟调 talker batch（TTFP）；高并发调 **S0=64 + prefix CUDA graphs + Code2Wav graphs**（三项齐升）。
- 与 [#3812](https://github.com/vllm-project/vllm-omni/issues/3812) 里报告的离线 Omni 2× 回归不同路径 — serving benchmark + 正确 deploy 是 apples-to-apples 对比方式。

---

## 附录：建议置顶评论

```
完整数据表（英文 index）：
https://github.com/hsliuustc0106/vllm-omni-cookbook/tree/main/tts/qwen3-tts

上游 vLLM-Omni：
https://github.com/vllm-project/vllm-omni

Perf JSON：
https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_tts.json

Retro 结果与脚本：
vllm-omni/benchmark_results/qwen3_tts_retro/

#3732 PR（Code2Wav cudagraph 默认开启）：
https://github.com/vllm-project/vllm-omni/pull/3732

高并发 deploy（#3662，main only）：
vllm_omni/deploy/qwen3_tts_high_concurrency.yaml

有问题欢迎评论区交流；voice_clone（Base + seed-tts）测完会再更一版。
```

---

## 附录：粘贴知乎编辑器的小技巧

1. **按 `---` 分段粘贴** — 避免一次粘贴格式错乱。
2. **表格** — 建议先粘飞书/Excel，再复制到知乎表格。
3. **代码块** — 正文保留短命令；完整 shell 脚本放置顶评论。
4. **删除文首 HTML 注释块** — 发布备忘，读者无需看到。
