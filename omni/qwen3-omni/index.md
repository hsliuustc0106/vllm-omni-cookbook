# Qwen3-Omni

**Category:** Omni (multimodal understanding + text/audio output)  
**Model:** `Qwen/Qwen3-Omni-30B-A3B-Instruct`  
**Recipe:** [Qwen/Qwen3-Omni.md](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen3-Omni.md)  
**Design doc:** [qwen3_omni_tts_performance_optimization.md](https://github.com/vllm-project/vllm-omni/blob/main/docs/design/qwen3_omni_tts_performance_optimization.md)  
**Retro harness:** [benchmark_results/qwen3_omni_retro/](https://github.com/vllm-project/vllm-omni/blob/main/benchmark_results/qwen3_omni_retro/README.md)

Three-stage pipeline: **Thinker** → **Talker** (+ code predictor) → **Code2Wav**.

---

## Key metrics

| Metric | Definition | Lower / higher? | When it matters |
|--------|------------|-----------------|-----------------|
| **TTFT** | Time to first text token (ms) | **Lower** | Text streaming UX |
| **TTFP** | Time to first audio packet (ms) | **Lower** | Speech output latency |
| **TPOT** | Time per output token (excl. first) | **Lower** | Decode efficiency |
| **RTF** | Wall time ÷ generated audio duration | **Lower** | End-to-end speech efficiency |
| **E2EL** | End-to-end request latency | **Lower** | Total user wait |
| **Throughput** | Audio duration / s, output tok/s | **Higher** | Sustained load |

CI tracks **mean** TTFT / TTFP / RTF in perf JSON; retro reports **mean** where available.

---

## Performance tracks

| Track | Hardware | Source |
|-------|----------|--------|
| **CI perf** | 2× H100 (nightly) | [`test_qwen3_omni_async_chunk.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_qwen3_omni_async_chunk.json) |
| **H200 full sweep (v0.22.0)** | 2× H200 (measured 2026-06-04) | [Table below](#h200-full-sweep--v0220) |
| **H200 retro** | 2× H200 (v0.18 / v0.20 / v0.22) | [Table below](#h200-retro-comparison) |
| **HF comparison** | A100 (design doc) | vLLM-Omni vs HF Transformers offline |

---

## Standardized perf test (CI)

Primary workload: **`random`** text+audio, `random_input_len=2500`, `random_output_len=900`, `--async-chunk`.

Measured on **2× H200** (`CUDA_VISIBLE_DEVICES=0,1`), vllm-omni main ~v0.22.0-pre (2026-06-04).

| Workload | c | prompts | mean TTFT | mean TTFP | mean RTF | E2EL | audio thr. |
|----------|--:|--------:|----------:|----------:|---------:|-----:|-----------:|
| text+audio | 1 | 4 | **101 ms** | **241 ms** | **0.132** | 23.0 s | 8.0 audio-s/s |
| text+audio | 4 | 16 | 165 ms | 410 ms | 0.193 | 44.4 s | 19.8 audio-s/s |
| text+audio | 8 | 32 | 254 ms | 537 ms | 0.257 | 40.8 s | 31.5 audio-s/s |
| text+audio | 16 | 64 | 418 ms | 802 ms | 0.361 | 53.2 s | 45.2 audio-s/s |
| text+audio | 32 | 128 | 666 ms | 1389 ms | 0.526 | 83.5 s | 60.5 audio-s/s |

CI thresholds (from [`test_qwen3_omni_async_chunk.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_qwen3_omni_async_chunk.json)): TTFT ≤ [400, 300, 400, 600, 900] ms; RTF ≤ [0.15, 0.30, 0.35, 0.50, 0.75]. Actuals are **well below** thresholds at all concurrencies.

Also sweeps **`random-mm`** (audio/image/video inputs) at fixed request rates — see upstream JSON.

```bash
cd /path/to/vllm-omni
export CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn
pytest -s tests/dfx/perf/scripts/run_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_qwen3_omni_async_chunk.json
```

---

## H200 full sweep — v0.22.0

**Protocol:** 2× H200 (`CUDA_VISIBLE_DEVICES=0,1`), vllm-omni main ~v0.22.0-pre (2026-06-04), text+audio `random` 2500/900, `--async-chunk`. Thinker on GPU0, Talker+Code2Wav on GPU1.

| c | prompts | TTFT (ms) | TTFP (ms) | TPOT (ms) | RTF | E2EL (s) | audio thr. (s/s) | audio dur. (s) |
|--:|--------:|----------:|----------:|----------:|----:|---------:|-----------------:|---------------:|
| 1 | 4 | **101** | **241** | **9** | **0.132** | 23.0 | 8.0 | 185 |
| 4 | 16 | 165 | 410 | 12 | 0.193 | 44.4 | 19.8 | 241 |
| 8 | 32 | 254 | 537 | 24 | 0.257 | 40.8 | 31.5 | 177 |
| 16 | 64 | 418 | 802 | 25 | 0.361 | 53.2 | 45.2 | 166 |
| 32 | 128 | 666 | 1389 | 33 | 0.526 | 83.5 | 60.5 | 179 |

Result files: `vllm-omni/tests/test_result/result_test_qwen3_omni_chunk_random_{c}_in2500_out900_20260604-*.json` (c = 1_4 / 4_16 / 8_32 / 16_64 / 32_128).

---

## H200 retro comparison

**Protocol:** 2× H200 (`CUDA_VISIBLE_DEVICES=0,1`), text+audio `random` 2500/900, `--async-chunk`. Thinker GPU0, Talker+Code2Wav GPU1. v0.18/v0.20 measured 2026-05-24; v0.22.0 measured 2026-06-04.

### c=1 latency (2500/900, async-chunk) — primary anchor

| Metric | v0.18.0 | v0.20.0 | v0.22.0 † | Δ v0.18→v0.20 | Δ v0.20→v0.22 |
|--------|--------:|--------:|----------:|--------------:|-------------:|
| TTFT (mean) | 248 ms | 721 ms | **101 ms** | +191% | **−86%** |
| TTFP (mean) | **736 ms** | 1325 ms | **241 ms** | +80% | **−82%** |
| TPOT (mean) | 20 ms | 31 ms | **9 ms** | +55% | **−71%** |
| RTF (mean) | **0.157** | 0.175 | **0.132** | +11% | **−25%** |
| E2EL (mean) | 37.7 s | **20.1 s** | 23.0 s | −47% | +14% ‡ |
| audio_throughput | 6.3 audio-s/s | 6.0 | **8.0** | −5% | **+33%** |
| mean audio duration | 239 s | 121 s | 185 s | −49% | +53% ‡ |

† v0.22.0-pre, measured on main ~2 days before the v0.22.0 tag (2026-06-04).  
‡ E2EL and audio-duration deltas are **not apples-to-apples** — random decode produces different output lengths per run. TTFP and RTF are the reliable cross-run anchors.

**Takeaways (c=1, same workload config):**
- **v0.18→v0.20:** TTFP regressed ~80% on this workload; E2EL improved because the run generated ~half the audio.
- **v0.20→v0.22:** Massive recovery — TTFP **−82%** (1325→241 ms, [#4054](https://github.com/vllm-project/vllm-omni/pull/4054)); TTFT **−86%**, TPOT **−71%**, RTF **−25%**. Audio throughput +33%.

### c=10 throughput — unstable on H200 (2026-05-24, v0.20.0 only)

| Metric | v0.20.0 | Notes |
|--------|--------:|-------|
| TTFP @ c=10 | 9201 ms (13/40 ok) | server crashed mid-run |
| RTF @ c=10 | 0.714 | partial only |
| E2EL @ c=10 | 110 s | partial only |
| audio_throughput @ c=10 | 11.2 audio-s/s | partial only |

Server exited under c=10 load before c=4 phase; treat as **indicative only**. High-concurrency sweep for v0.22.0 is in the [full sweep table above](#h200-full-sweep--v0220) (c=4/8/16/32).

### v0.18.0 blockers (resolved for c=1)

1. **flashinfer** version skew — `FLASHINFER_DISABLE_VERSION_CHECK=1` in run script
2. **`openai-chat-omni` backend** — PATH wrapper to `python -m vllm_omni.entrypoints.cli.main`
3. **GPU memory at init** — `stage_overrides` lower utilization; ensure clean GPUs before run
4. **async-chunk required** — without it TTFP ~45 s (non-comparable)

Harness: `benchmark_results/qwen3_omni_retro/` · Config: `qwen3_omni_retro_v018.json` / `qwen3_omni_retro_v020.json`

```bash
bash benchmark_results/qwen3_omni_retro/v0.18.0/run_benchmark.sh
bash benchmark_results/qwen3_omni_retro/v0.20.0/run_benchmark.sh
```

---

## Deploy profile

Default: [`qwen3_omni_moe.yaml`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/qwen3_omni_moe.yaml) — Thinker on GPU0, Talker+Code2Wav on GPU1 (verified 2× H100/H200).

```bash
export CUDA_VISIBLE_DEVICES=0,1
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --async-chunk --port 8000
```

Post-[#3732](https://github.com/vllm-project/vllm-omni/pull/3732): Code2Wav inner cudagraph enabled when stage-2 `enforce_eager: false`.

---

## Optimization index

| PR / doc | Release | Area | Notes |
|----------|---------|------|-------|
| [#4054](https://github.com/vllm-project/vllm-omni/pull/4054) | v0.22 | TTFP: initial_codec_chunk_frames | Primary driver of TTFP −82% (c=1: 1325→241 ms) |
| [#3592](https://github.com/vllm-project/vllm-omni/pull/3592) | v0.22 | Stage 1 bounded-K active-stream window | Throughput at high concurrency |
| [#3734](https://github.com/vllm-project/vllm-omni/pull/3734) | v0.22 | AR prefix cache CPU staging dedup | Reduces TTFT/TPOT overhead |
| [#3575](https://github.com/vllm-project/vllm-omni/pull/3575) | v0.22 | Performance regression fix | Fixes v0.20→main TTFP/TTFT regression |
| [#3732](https://github.com/vllm-project/vllm-omni/pull/3732) | v0.20+ | Code2Wav cudagraph | Qwen3-Omni c=10 A/B: E2EL −8%, output tok/s +8% |
| Design doc | v0.20 | Batching + CUDA graph + async chunk | HF vs vLLM-Omni on A100 (E2EL −93%, TTFP −99.7%) |
| `test_qwen3_omni_no_async_chunk.json` | — | async-chunk off | Higher TTFP baselines — use for regression only |

---

## Reproduce retro

```bash
bash benchmark_results/qwen3_omni_retro/v0.18.0/run_benchmark.sh
bash benchmark_results/qwen3_omni_retro/v0.20.0/run_benchmark.sh
bash benchmark_results/qwen3_omni_retro/cleanup.sh
```

For v0.22.0 (full concurrency sweep):

```bash
export CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn
pytest -s tests/dfx/perf/scripts/run_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_qwen3_omni_async_chunk.json
```

---

## Not yet measured

- c=10 throughput stability at v0.22.0 under sustained load
- `random-mm` multimodal-input workloads (audio/image/video) — sweep in `test_qwen3_omni_async_chunk.json`
- `--no-async-chunk` vs `--async-chunk` A/B on same hardware
- v0.18 retro (no comparable upstream JSON for v0.18 workload)
