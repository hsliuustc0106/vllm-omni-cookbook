<!--
知乎发布备忘（发布前可删除本注释块）

【推荐标题】
vLLM-Omni 优化 Qwen3-TTS：v0.20 → main e7644daa（vLLM 0.21.0）实测，高并发怎么快 10 倍？

【备选标题】
Qwen3-TTS 在 vLLM-Omni 里怎么测性能？TTFP / RTF / 并发悬崖与 #3662 高并发配置

【封面图建议】
- 三指标并排：c=64 下 TTFP / RTF / throughput 柱状图（v0.20 vs main 标准 vs main 高并发）
- c=64 TTFP 对比：v0.20（7.8s）vs main 标准（7.9s）vs main 高并发（351ms）
- c=8 并发悬崖：v0.20 214ms vs main 82ms 柱状图

【话题标签】
#vLLM #大模型推理 #语音合成 #TTS #Qwen3 #AIGC

【置顶评论模板】见文末
-->

# vLLM-Omni 优化 Qwen3-TTS：v0.20 → main e7644daa（vLLM 0.21.0）实测，高并发怎么快 10 倍？

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

v0.20.0 合入了通用 TTS benchmark（`test_tts.json`，PR #2835）；main 分支在 vLLM 0.21.0 上继续优化延迟与高并发（#3485、#3662）。

这篇文章回答四个问题：

1. **v0.20.0 → main，TTFP / RTF / throughput 各改善多少？**
2. **为什么 c=8 是「并发悬崖」？**
3. **#3662 高并发 deploy 对三项指标分别解决了什么？**
4. **读者如何复现？**

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

### 1.2 本文 L20X retro（2× L20X，2026-05-22）

| 维度 | 设置 |
|------|------|
| 模型 | Qwen3-TTS-12Hz-1.7B-**CustomVoice** |
| GPU | `CUDA_VISIBLE_DEVICES=2,3`（Talker → 逻辑 GPU0，Code2Wav → 逻辑 GPU1） |
| 对比版本 | **v0.20.0 tag** vs **main `e7644daa`**（vLLM **0.21.0**） |
| 低延迟 | c=1，`num-prompts=3`，bundled smoke/design 数据集 |
| 高并发 | c=**8 / 16 / 64**，prompts **80 / 128 / 128**（与 CI throughput 一致） |
| Deploy | 默认 `qwen3_tts.yaml`；另测 main 专用 **`qwen3_tts_high_concurrency.yaml`**（#3662） |

**注意：**

- L20X 数字与 CI 的 H100 阈值 **不可直接对比**；本文只比较 **同一硬件、同一 workload** 下的版本差异。
- main 跑 benchmark 需 pin **`transformers==5.8.1`**（5.9.0 会打挂 Code2Wav）；本地 bench 需 **unset HTTP_PROXY**。

---

## 二、核心结果（TTFP / RTF / throughput）

### 2.1 单请求（c=1）：TTFP + RTF

| 任务 | v0.20.0 TTFP | main TTFP | Δ TTFP | v0.20 RTF | main RTF | Δ RTF |
|------|-------------:|----------:|-------:|----------:|---------:|------:|
| default_voice | 59 ms | **47 ms** | **−21%** | 0.145 | 0.145 | ~持平 |
| voice_design | 63 ms | **47 ms** | **−25%** | 0.148 | 0.139 | **−6%** |

**一句话：** main + vLLM 0.21 在 **TTFP** 上稳定快 **~20%**；**RTF** 持平或略好。c=1 不看 throughput。

### 2.2 高并发 TTFP（ms，越低越好）

| 任务 | c | v0.20.0 | main（标准 deploy） | main（#3662 高并发） |
|------|--:|--------:|--------------------:|---------------------:|
| default_voice | 8 | 214 | **82** | 83 |
| default_voice | 16 | 1179 | 935 | **118** |
| default_voice | 64 | 7805 | 7852 | **351** |
| voice_design | 8 | 216 | **82** | 84 |
| voice_design | 16 | 1089 | 839 | **127** |
| voice_design | 64 | 7207 | 6908 | **386** |

### 2.3 高并发 RTF（越低越好）

| 任务 | c | v0.20.0 | main（标准） | main（#3662 高并发） |
|------|--:|--------:|-------------:|---------------------:|
| default_voice | 8 | 0.249 | **0.246** | 0.243 |
| default_voice | 16 | 0.436 | 0.423 | **0.357** |
| default_voice | 64 | 1.554 | 1.533 | **0.996** |
| voice_design | 8 | 0.251 | **0.251** | 0.250 |
| voice_design | 16 | 0.447 | 0.437 | **0.374** |
| voice_design | 64 | 1.641 | 1.628 | **1.093** |

### 2.4 高并发 throughput（audio-s / s，越高越好）

| 任务 | c | v0.20.0 | main（标准） | main（#3662 高并发） |
|------|--:|--------:|-------------:|---------------------:|
| default_voice | 8 | 19.8 | **28.4** | 29.2 |
| default_voice | 16 | 35.9 | 36.5 | **40.9** |
| default_voice | 64 | 36.0 | 36.6 | **60.8** |
| voice_design | 8 | 31.2 | 30.0 | 30.8 |
| voice_design | 16 | 34.9 | 35.3 | **41.6** |
| voice_design | 64 | 33.7 | 35.0 | **55.0** |

### 2.5 c=64 直观对比（建议作配图）

**TTFP：**

```
default_voice @ c=64  TTFP

v0.20.0          ████████████████████████████████████████  7805 ms
main 标准 deploy  ████████████████████████████████████████  7852 ms
main 高并发 #3662  ██                                       351 ms
```

**RTF + throughput（default_voice @ c=64）：**

| 版本 | RTF | throughput |
|------|----:|-----------:|
| v0.20.0 | 1.554 | 36.0 |
| main 标准 | 1.533 | 36.6 |
| main 高并发 | **0.996** | **60.8** |

### 2.6 三指标结论汇总

| 对比 | c=8 | c=16 | c=64 |
|------|-----|------|------|
| v0.20 → main（标准 deploy） | TTFP **−62%**；RTF ~持平；tp **+43%** | TTFP **−21%**；RTF ~持平 | 三项 ~持平 |
| main 标准 → main 高并发（#3662） | 三项 ~持平 | TTFP **−85–87%**，RTF **−14–16%**，tp **+12%** | TTFP **−94–96%**，RTF **−33–35%**，tp **+50–75%** |

**三句话结论：**

1. **v0.20 → main（标准 deploy）**：c=8 是最大赢家 — TTFP **−62%**，throughput **+43%**；c=64 三项几乎不变。
2. **main 标准 → main 高并发（#3662）**：c=8 三项持平；**c=16/64** 三项齐升 — TTFP **−85–96%**，RTF **−14–35%**，throughput **+12–75%**。
3. **做线上高并发，务必换 deploy 配置** — 默认 yaml 下 TTFP 爆炸、throughput 平台化（[#272](https://github.com/vllm-project/vllm-omni/issues/272)）。

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
c=8   v0.20  214 ms      main 标准 82 ms  tp 19.8 → 28.4 (+43%)
c=64  v0.20  7805 ms     main 标准 ~7850 ms  tp ~36（标准 deploy 几乎救不回来）
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
| #3485 | Qwen3-TTS 延迟回归修复 | c=1 TTFP **−21–25%**；c=8 TTFP **−62%**，tp **+43%** |
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

---

## 六、发版 / 分支对照

| 来源 | vLLM-Omni | vLLM | 高并发 deploy |
|------|-----------|------|---------------|
| v0.20.0 tag | 0.20.0 | 0.20.0 | ❌ |
| main `e7644daa` | 0.20.1.dev171 | **0.21.0** | ✅ #3662 |

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
# 低延迟 c=1
bash benchmark_results/qwen3_tts_retro/v0.20.0/run_benchmark.sh
bash benchmark_results/qwen3_tts_retro/main/run_benchmark.sh

# 高并发 c=8/16/64（标准 deploy）
bash benchmark_results/qwen3_tts_retro/v0.20.0/run_benchmark_throughput.sh
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

- **v0.20 → main（标准 deploy）**：c=1 **TTFP −21–25%**；c=8 **TTFP −62% + throughput +43%**；c=64 三项 ~持平。
- **#3662 高并发 profile** 才是 c=16/64 的钥匙：TTFP 从 **~7–8 s → ~0.35 s**，RTF 从 **~1.6 → ~1.0**，throughput 从 **~36 → ~61 audio-s/s**。
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
