# Qwen3-TTS

**Category:** TTS (Text-to-Speech)  
**Model:** `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (retro) · `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (voice_clone)  
**Recipe:** [Qwen3-TTS](https://github.com/vllm-project/vllm-omni/tree/main/recipes/qwen3-tts)  
**Zhihu draft (中文):** [qwen3-tts-performance-zhihu.md](qwen3-tts-performance-zhihu.md)  
**Benchmark docs:** [benchmarks/tts/README.md](https://github.com/vllm-project/vllm-omni/blob/main/benchmarks/tts/README.md)  
**Retro harness:** [benchmark_results/qwen3_tts_retro/](https://github.com/vllm-project/vllm-omni/blob/main/benchmark_results/qwen3_tts_retro/README.md)

---

## Performance tracks

| Track | Hardware | Source |
|-------|----------|--------|
| **CI latency / throughput** | 2× H100 (nightly) | [`test_tts.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_tts.json) |
| **v0.20 / main retro** | 2× L20X (measured) | [Tables below](#l20x-retro-latency-c1) · [Retro README](https://github.com/vllm-project/vllm-omni/blob/main/benchmark_results/qwen3_tts_retro/README.md) |

---

## Standardized perf test (CI)

| Task | Model | Phase | c | CI baseline TTFP | CI baseline RTF |
|------|-------|-------|--:|-----------------:|----------------:|
| voice_clone | Base | latency | 1 | **350 ms** | **0.25** |
| default_voice | CustomVoice | latency | 1 | **150 ms** | **0.15** |
| voice_design | CustomVoice | latency | 1 | **150 ms** | **0.15** |
| default_voice | CustomVoice | throughput | 8 | 1500 ms | 0.30 |
| voice_design | CustomVoice | throughput | 8 | 1500 ms | 0.35 |

Throughput CI also sweeps **c=16, 64** with **80 / 128 / 128** prompts — see `test_tts.json`.

```bash
cd /path/to/vllm-omni
export CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn
pytest -s tests/dfx/perf/scripts/run_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_tts.json
```

---

## L20X retro — latency (c=1)

Measured **2026-05-22** on **2× NVIDIA L20X** (`CUDA_VISIBLE_DEVICES=2,3`).  
Protocol: **`num-prompts=3`**, **`num-warmups=2`**, standard deploy (`qwen3_tts.yaml` + 2-GPU stage split).

| Task | v0.20.0 TTFP | main (`e7644daa`) TTFP | Δ | v0.20.0 RTF | main RTF |
|------|-------------:|-----------------------:|--:|------------:|---------:|
| default_voice | **59 ms** | **47 ms** | **−21%** | 0.145 | 0.145 |
| voice_design | **63 ms** | **47 ms** | **−25%** | 0.148 | 0.139 |

**Takeaway:** **main + vllm 0.21.0** improves **TTFP ~21–25%** at c=1; **RTF** flat to slightly better.

---

## L20X retro — throughput (c=8 / 16 / 64)

CI throughput matrix: **80 / 128 / 128** prompts at **c=8 / 16 / 64**.  
Metric: **median** TTFP / RTF (lower is better); **audio_throughput** (higher is better).

### TTFP (ms)

| Task | c | v0.20.0 | main (std) | main (hiconc‡) |
|------|--:|--------:|-----------:|---------------:|
| default_voice | 8 | 214 | **82** | 83 |
| default_voice | 16 | 1179 | 935 | **118** |
| default_voice | 64 | 7805 | 7852 | **351** |
| voice_design | 8 | 216 | **82** | 84 |
| voice_design | 16 | 1089 | 839 | **127** |
| voice_design | 64 | 7207 | 6908 | **386** |

### RTF

| Task | c | v0.20.0 | main (std) | main (hiconc‡) |
|------|--:|--------:|-----------:|---------------:|
| default_voice | 8 | 0.249 | **0.246** | 0.243 |
| default_voice | 16 | 0.436 | 0.423 | **0.357** |
| default_voice | 64 | 1.554 | 1.533 | **0.996** |
| voice_design | 8 | 0.251 | **0.251** | 0.250 |
| voice_design | 16 | 0.447 | 0.437 | **0.374** |
| voice_design | 64 | 1.641 | 1.628 | **1.093** |

### Audio throughput (audio duration / s)

| Task | c | v0.20.0 | main (std) | main (hiconc‡) |
|------|--:|--------:|-----------:|---------------:|
| default_voice | 8 | 19.8 | **28.4** | 29.2 |
| default_voice | 16 | 35.9 | 36.5 | **40.9** |
| default_voice | 64 | 36.0 | 36.6 | **60.8** |
| voice_design | 8 | 31.2 | 30.0 | 30.8 |
| voice_design | 16 | 34.9 | 35.3 | **41.6** |
| voice_design | 64 | 33.7 | 35.0 | **55.0** |

‡ **main (hiconc)** = [`qwen3_tts_high_concurrency.yaml`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/qwen3_tts_high_concurrency.yaml) ([#3662](https://github.com/vllm-project/vllm-omni/pull/3662)): S0=64 talker, code-predictor prefix CUDA graphs, tuned Code2Wav graph buckets. **main-only** — not on v0.20.0 tag.

**Takeaways:**

| Comparison | c=8 | c=16 | c=64 |
|------------|-----|------|------|
| v0.20 → main (std deploy) | TTFP **−62%** | TTFP **−21%** | ~parity |
| main std → main hiconc | ~neutral | TTFP **−85–87%**, RTF **−14–16%** | TTFP **−94–96%**, RTF **−33–35%**, audio_tp **+50–75%** |

At **c≥16**, use the **high-concurrency deploy** on main — standard deploy still hits the codec-batching cliff ([#272](https://github.com/vllm-project/vllm-omni/issues/272)).

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
| main `e7644daa` | 0.20.1.dev171 | **0.21.0** | **5.8.1**† | std + hiconc deploy |

† Pin **`transformers==5.8.1`** on main for Qwen3-TTS benchmarks — **5.9.0** breaks Code2Wav (`create_causal_mask` API). Unset **`HTTP_PROXY`** when running local bench (breaks localhost client).

---

## Optimization index

| PR | Area | Retro impact |
|----|------|--------------|
| [#3485](https://github.com/vllm-project/vllm-omni/pull/3485) | Latency regression fix | c=1 TTFP −21–25%; c=8 TTFP −62% (std deploy) |
| [#3662](https://github.com/vllm-project/vllm-omni/pull/3662) | High-concurrency serving | c=16/64 TTFP −85–96% vs std main; hiconc deploy |
| [#2376](https://github.com/vllm-project/vllm-omni/pull/2376) | Code2Wav CUDA graphs | Decoder path |
| [#2341](https://github.com/vllm-project/vllm-omni/pull/2341) | Native Code2Wav decoder | Stage-1 refactor |
| [#2835](https://github.com/vllm-project/vllm-omni/pull/2835) | Universal TTS benchmark | `test_tts.json`, `bench_tts.py` |
| [#272](https://github.com/vllm-project/vllm-omni/issues/272) | Concurrency cliff | c=4→8 TTFP jump; hiconc mitigates at c≥16 |

---

## Reproduce retro

```bash
# Latency c=1
bash benchmark_results/qwen3_tts_retro/v0.20.0/run_benchmark.sh
bash benchmark_results/qwen3_tts_retro/main/run_benchmark.sh

# Throughput c=8/16/64 (standard deploy)
bash benchmark_results/qwen3_tts_retro/v0.20.0/run_benchmark_throughput.sh
bash benchmark_results/qwen3_tts_retro/main/run_benchmark_throughput.sh

# Throughput with #3662 high-concurrency deploy (main only)
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
