# Qwen3-TTS

**Category:** TTS (Text-to-Speech)  
**Model:** `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (retro) · `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (voice_clone)  
**Recipe:** [Qwen3-TTS](https://github.com/vllm-project/vllm-omni/tree/main/recipes/qwen3-tts)  
**Zhihu draft (中文):** [qwen3-tts-performance-zhihu.md](qwen3-tts-performance-zhihu.md)  
**Benchmark docs:** [benchmarks/tts/README.md](https://github.com/vllm-project/vllm-omni/blob/main/benchmarks/tts/README.md)  
**Retro harness:** [benchmark_results/qwen3_tts_retro/](https://github.com/vllm-project/vllm-omni/blob/main/benchmark_results/qwen3_tts_retro/README.md)

---

## Key metrics

Qwen3-TTS perf is tracked on **three metrics** (see [benchmarks/tts/README.md](https://github.com/vllm-project/vllm-omni/blob/main/benchmarks/tts/README.md)):

| Metric | Definition | Lower / higher? | When it matters |
|--------|------------|-----------------|-----------------|
| **TTFP** | Time to first audio packet (ms) | **Lower** is better | Interactive / streaming UX; latency SLO at **c=1** |
| **RTF** | Wall time ÷ generated audio duration | **Lower** is better (<1 = faster than realtime) | End-to-end generation efficiency per request |
| **Throughput** | Total audio duration served per wall second (`audio_throughput`, audio-s / s) | **Higher** is better | Sustained load at **c≥8** |

CI uses **median** TTFP / RTF for latency and throughput phases; retro tables below follow the same convention.

---

## Performance tracks

| Track | Hardware | Source |
|-------|----------|--------|
| **CI latency / throughput** | 2× H100 (nightly) | [`test_tts.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_tts.json) |
| **v0.20 / main retro** | 2× L20X (measured) | [Tables below](#l20x-retro-latency-c1) · [Retro README](https://github.com/vllm-project/vllm-omni/blob/main/benchmark_results/qwen3_tts_retro/README.md) |

---

## Standardized perf test (CI)

| Task | Model | Phase | c | TTFP | RTF | Throughput |
|------|-------|-------|--:|-----:|----:|-----------:|
| voice_clone | Base | latency | 1 | **350 ms** | **0.25** | — |
| default_voice | CustomVoice | latency | 1 | **150 ms** | **0.15** | — |
| voice_design | CustomVoice | latency | 1 | **150 ms** | **0.15** | — |
| default_voice | CustomVoice | throughput | 8 | 1500 ms | 0.30 | tracked |
| voice_design | CustomVoice | throughput | 8 | 1500 ms | 0.35 | tracked |

Throughput CI also sweeps **c=16, 64** with **80 / 128 / 128** prompts — TTFP / RTF / throughput are all recorded; see `test_tts.json`.

```bash
cd /path/to/vllm-omni
export CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn
pytest -s tests/dfx/perf/scripts/run_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_tts.json
```

---

## L20X retro — latency (c=1)

Measured **2026-05-24** on **2× NVIDIA L20X** (`CUDA_VISIBLE_DEVICES=2,3`).  
Protocol: **`num-prompts=3`**, **`num-warmups=2`**, standard deploy (`qwen3_tts.yaml` + 2-GPU stage split).  
**main** = `28ce618f` (includes [#3732](https://github.com/vllm-project/vllm-omni/pull/3732) Code2Wav CUDA graph default-on).

| Task | v0.20.0 TTFP | main TTFP | Δ TTFP | v0.20.0 RTF | main RTF | Δ RTF |
|------|-------------:|----------:|-------:|------------:|---------:|------:|
| default_voice | **59 ms** | **48 ms** | **−19%** | 0.145 | 0.147 | ~flat |
| voice_design | **63 ms** | **46 ms** | **−27%** | 0.148 | 0.147 | ~flat |

**Takeaway (c=1):** **main + vllm 0.21.0** wins on **TTFP** (~21–25%); **RTF** flat to slightly better. Throughput is not meaningful at c=1.

---

## L20X retro — throughput (c=8 / 16 / 64)

CI throughput matrix: **80 / 128 / 128** prompts at **c=8 / 16 / 64**.  
All three metrics below: **TTFP**, **RTF**, **throughput** (median unless noted).

### TTFP (ms)

| Task | c | v0.20.0 | main (std) | main (hiconc‡) |
|------|--:|--------:|-----------:|---------------:|
| default_voice | 8 | 214 | **81** | 83 |
| default_voice | 16 | 1179 | 974 | **118** |
| default_voice | 64 | 7805 | 7861 | **351** |
| voice_design | 8 | 216 | **81** | 84 |
| voice_design | 16 | 1089 | 817 | **127** |
| voice_design | 64 | 7207 | 6743 | **386** |

### RTF

| Task | c | v0.20.0 | main (std) | main (hiconc‡) |
|------|--:|--------:|-----------:|---------------:|
| default_voice | 8 | 0.249 | **0.245** | 0.243 |
| default_voice | 16 | 0.436 | 0.426 | **0.357** |
| default_voice | 64 | 1.554 | 1.443 | **0.996** |
| voice_design | 8 | 0.251 | **0.244** | 0.250 |
| voice_design | 16 | 0.447 | 0.430 | **0.374** |
| voice_design | 64 | 1.641 | 1.557 | **1.093** |

### Audio throughput (audio duration / s)

| Task | c | v0.20.0 | main (std) | main (hiconc‡) |
|------|--:|--------:|-----------:|---------------:|
| default_voice | 8 | 19.8 | **31.4** | 29.2 |
| default_voice | 16 | 35.9 | 37.0 | **40.9** |
| default_voice | 64 | 36.0 | 36.9 | **60.8** |
| voice_design | 8 | 31.2 | **31.6** | 30.8 |
| voice_design | 16 | 34.9 | 36.0 | **41.6** |
| voice_design | 64 | 33.7 | 36.1 | **55.0** |

‡ **main (hiconc)** = [`qwen3_tts_high_concurrency.yaml`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/qwen3_tts_high_concurrency.yaml) ([#3662](https://github.com/vllm-project/vllm-omni/pull/3662)), measured at **`e7644daa`** (pre-#3732; hiconc stage-1 still `enforce_eager: true`).

**Takeaways (TTFP / RTF / throughput):**

| Comparison | c=8 | c=16 | c=64 |
|------------|-----|------|------|
| v0.20 → main (std) | TTFP **−62%**; RTF ~flat; tp **+59%** | TTFP **−21–25%**; RTF ~flat | TTFP ~flat; tp ~flat |
| main std → main hiconc | ~neutral on all three | TTFP **−85–88%**, RTF **−14–16%**, tp **+11–16%** | TTFP **−94–96%**, RTF **−31–35%**, tp **+50–75%** |

At **c≥16**, use the **high-concurrency deploy** on main — standard deploy still hits the codec-batching cliff ([#272](https://github.com/vllm-project/vllm-omni/issues/272)): TTFP spikes while throughput plateaus.

---

## #3732 — Code2Wav CUDA graph default-on (2026-05-23)

[#3732](https://github.com/vllm-project/vllm-omni/pull/3732) gates inner Code2Wav CUDA graph capture on stage `enforce_eager` and flips **`qwen3_tts.yaml` stage-1 default to `enforce_eager: false`**. Opt-out: `--stage-overrides '{"1": {"enforce_eager": true}}'`.

### Isolated A/B on `28ce618f` (same commit, std deploy)

Only variable: stage-1 **eager** vs **Code2Wav cudagraph** (`main-post3732-eager/` vs `main-post3732/`).

**default_voice:**

| c | TTFP eager | TTFP cudagraph | Δ TTFP | RTF eager | RTF cudagraph | Δ RTF | tp eager | tp cudagraph | Δ tp |
|--:|-----------:|---------------:|-------:|----------:|--------------:|------:|---------:|-------------:|-----:|
| 1 | 52 ms | **48 ms** | **−8%** | 0.153 | **0.147** | **−4%** | 6.6 | 6.8 | +3% |
| 8 | 92 ms | **81 ms** | **−12%** | 0.248 | **0.245** | −1% | 30.7 | **31.4** | **+3%** |
| 16 | 955 ms | 974 ms | ~flat | 0.428 | 0.426 | ~flat | 36.7 | 37.0 | +1% |
| 64 | 8085 ms | **7861 ms** | **−3%** | 1.549 | **1.443** | **−7%** | 36.5 | 36.9 | +1% |

**voice_design:** same pattern — c=1 TTFP **−14%**, c=8 TTFP **−13%**, c=64 RTF **−1%** (TTFP −1%).

**Takeaway:** #3732 helps at **all c levels**, but unevenly:

| c | #3732 impact |
|---|--------------|
| **c=1** | TTFP **−8–14%**, RTF **−4%** — modest but real on Code2Wav path |
| **c=8** | TTFP **−12–13%**, throughput **+3–4%** — largest TTFP win in std matrix |
| **c=16** | ~neutral (talker/queue dominates) |
| **c=64** | TTFP **−1–3%**, RTF **−1–7%** — efficiency gain, not cliff fix |

At **c=64**, #3732 does **not** replace [#3662](https://github.com/vllm-project/vllm-omni/pull/3662) hiconc (TTFP still ~7.9 s std vs **351 ms** hiconc).

### Commit-level delta (`e7644daa` → `28ce618f`)

Pre-merge main already ran Code2Wav eager by default, so commit-to-commit delta understates #3732 when talker-side wins ([#3485](https://github.com/vllm-project/vllm-omni/pull/3485)) overlap. See isolated A/B above for Code2Wav-only effect.

**Supplemental PR-review A/B** (c=10, 40 prompts): TTFP **509 → 117 ms (−77%)**, RTF **0.30 → 0.21**, throughput **30.6 → 43.8 audio-s/s**.

---

## Deploy profiles

| Profile | When | Talker `max_num_seqs` | Key opts |
|---------|------|----------------------:|----------|
| [`qwen3_tts.yaml`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/qwen3_tts.yaml) | Default / latency | 10 | async_chunk, low-latency talker batches |
| [`qwen3_tts_high_concurrency.yaml`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/qwen3_tts_high_concurrency.yaml) | c≥16 sustained load | **64** | prefix CUDA graphs, Code2Wav capture sizes |

**Standard serve:**

```bash
export CUDA_VISIBLE_DEVICES=0,1
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni --port 8000
```

**High-concurrency serve (main only):**

```bash
export CUDA_VISIBLE_DEVICES=0,1
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni --port 8000 \
  --deploy-config vllm_omni/deploy/qwen3_tts_high_concurrency.yaml
```

---

## Stacks (retro)

| Source | vLLM-Omni | vLLM | transformers | Notes |
|--------|-----------|------|--------------|-------|
| v0.20.0 tag | 0.20.0 | 0.20.0 | 5.8.1 | standard deploy only |
| main `e7644daa` | 0.20.1.dev171 | **0.21.0** | **5.8.1**† | pre-#3732; hiconc retro |
| main `28ce618f` | 0.20.1.dev175 | **0.21.0** | **5.8.1**† | std deploy incl. #3732 |

† Pin **`transformers==5.8.1`** on main for Qwen3-TTS benchmarks — **5.9.0** breaks Code2Wav (`create_causal_mask` API). Unset **`HTTP_PROXY`** when running local bench (breaks localhost client).

---

## Optimization index

| PR | Area | Retro impact (TTFP / RTF / throughput) |
|----|------|----------------------------------------|
| [#3732](https://github.com/vllm-project/vllm-omni/pull/3732) | Code2Wav cudagraph default-on | c=1 TTFP **−8–14%**; c=8 TTFP **−12%** + tp **+3%**; c=64 RTF **−7%** (isolated A/B) |
| [#3485](https://github.com/vllm-project/vllm-omni/pull/3485) | Latency regression fix | c=1 TTFP −19–27%; c=8 TTFP −62%, tp +59% (std deploy) |
| [#3662](https://github.com/vllm-project/vllm-omni/pull/3662) | High-concurrency serving | c=16/64 TTFP −85–96%, RTF −14–35%, tp +50–75% vs std main |
| [#2376](https://github.com/vllm-project/vllm-omni/pull/2376) | Code2Wav CUDA graphs | Decoder path |
| [#2341](https://github.com/vllm-project/vllm-omni/pull/2341) | Native Code2Wav decoder | Stage-1 refactor |
| [#2835](https://github.com/vllm-project/vllm-omni/pull/2835) | Universal TTS benchmark | `test_tts.json`, `bench_tts.py` |
| [#272](https://github.com/vllm-project/vllm-omni/issues/272) | Concurrency cliff | c=4→8 TTFP jump; hiconc mitigates at c≥16 |

---

## Reproduce retro

```bash
# v0.20.0 baseline
bash benchmark_results/qwen3_tts_retro/v0.20.0/run_benchmark.sh
bash benchmark_results/qwen3_tts_retro/v0.20.0/run_benchmark_throughput.sh

# #3732 isolated A/B — eager vs cudagraph (28ce618f)
bash benchmark_results/qwen3_tts_retro/main-post3732-eager/run_benchmark.sh
bash benchmark_results/qwen3_tts_retro/main-post3732-eager/run_benchmark_throughput.sh

# main post-#3732 cudagraph default
bash benchmark_results/qwen3_tts_retro/main-post3732/run_benchmark.sh
bash benchmark_results/qwen3_tts_retro/main-post3732/run_benchmark_throughput.sh

# Pre-#3732 main baseline (e7644daa)
bash benchmark_results/qwen3_tts_retro/main/run_benchmark.sh
bash benchmark_results/qwen3_tts_retro/main/run_benchmark_throughput.sh

# #3662 high-concurrency deploy (e7644daa)
bash benchmark_results/qwen3_tts_retro/main/run_benchmark_hiconc_throughput.sh

bash benchmark_results/qwen3_tts_retro/cleanup.sh
```

Manual smoke:

```bash
python benchmarks/tts/bench_tts.py \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --task default_voice \
  --dataset-path benchmarks/build_dataset/seed_tts_smoke \
  --concurrency 1 8 16 \
  --num-prompts 20 \
  --output-dir ./results
```

---

## Not yet measured

- **voice_clone** (`-Base`): needs [seed-tts](https://github.com/BytedanceSpeech/seed-tts-eval) dataset download
- **v0.18.0**: legacy `benchmarks/qwen3-tts/` path — not comparable to `test_tts.json`
- **stress** phase (`request_rate=2.0`)
