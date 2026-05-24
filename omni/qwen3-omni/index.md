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

CI tracks **mean** TTFT / TTFP / RTF in perf JSON; retro will report **median** + **mean** where available.

---

## Performance tracks

| Track | Hardware | Source |
|-------|----------|--------|
| **CI perf** | 2× H100 (nightly) | [`test_qwen3_omni_async_chunk.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_qwen3_omni_async_chunk.json) |
| **L20X retro** | 2× L20X (in progress) | [Retro README](https://github.com/vllm-project/vllm-omni/blob/main/benchmark_results/qwen3_omni_retro/README.md) |
| **HF comparison** | A100 (design doc) | vLLM-Omni vs HF Transformers offline |

---

## Standardized perf test (CI)

Primary workload: **`random`** text+audio, `random_input_len=2500`, `random_output_len=900`, `--async-chunk`.

| Workload | c | prompts | CI baseline (mean TTFT) | CI baseline (mean TTFP) | CI baseline (mean RTF) |
|----------|--:|--------:|------------------------:|------------------------:|-------------------------:|
| text+audio | 1 | 4 | 1000 ms | 1000 ms | 0.20 |
| text+audio | 4 | 16 | 3000 ms | 3000 ms | 0.35 |
| text+audio | 8 | 32 | 5000 ms | 5000 ms | 0.60 |
| text+audio | 16 | 64 | 7000 ms | 7000 ms | 0.85 |
| text+audio | 32 | 128 | 9000 ms | 9000 ms | 0.90 |

Also sweeps **`random-mm`** (audio/image/video inputs) at fixed request rates — see upstream JSON.

```bash
cd /path/to/vllm-omni
export CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn
pytest -s tests/dfx/perf/scripts/run_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_qwen3_omni_async_chunk.json
```

---

## L20X retro — v0.18.0 vs v0.20.0

**Protocol:** shared workload on 2× L20X (`CUDA_VISIBLE_DEVICES=2,3`) — text+audio random **2500/900**, `--async-chunk`, deploy profile Thinker GPU0 / Talker+Code2Wav GPU1.

| Phase | c | prompts |
|-------|--:|--------:|
| latency | 1 | 4 |
| throughput | 10 | 40 |
| throughput | 4 | 16 |

**Note:** Tag-native CI JSON differs (v0.18 used 100/100 tokens; v0.20 uses 2500/900). Retro uses **v0.20 workload on both tags** for fair comparison.

### c=1 latency (2500/900) — primary anchor

| Metric | v0.18.0 | v0.20.0 | Δ (v0.20 vs v0.18) |
|--------|--------:|--------:|-------------------:|
| TTFT (mean) | _blocked_ | **721 ms** | — |
| TTFP (mean) | _blocked_ | **1325 ms** | — |
| TPOT (mean) | _blocked_ | **31 ms** | — |
| RTF (mean) | _blocked_ | **0.175** | — |
| E2EL (mean) | _blocked_ | **20.1 s** | — |
| audio_throughput | _blocked_ | **6.0 audio-s/s** | — |

v0.20.0 run: `v0.20.0` tag · 2× L20X · 4/4 prompts completed · harness `result_test_qwen3_omni_chunk_random_1_4_in2500_out900_20260524-150548.json`

### c=10 throughput — unstable on L20X (2026-05-24)

| Metric | v0.18.0 | v0.20.0 | Notes |
|--------|--------:|--------:|-------|
| TTFP @ c=10 | _blocked_ | 9201 ms (13/40 ok) | v0.20 server crashed mid-run |
| RTF @ c=10 | _blocked_ | 0.714 | partial only |
| E2EL @ c=10 | _blocked_ | 110 s | partial only |
| audio_throughput @ c=10 | _blocked_ | 11.2 audio-s/s | partial only |

Server exited under c=10 load before c=4 phase; treat throughput rows as **indicative only** until stability is fixed.

### v0.18.0 blockers (same hardware)

1. **flashinfer** version skew in v018 venv (`flashinfer 0.6.6` vs `flashinfer-cubin 0.6.8.post1`) — workaround: `FLASHINFER_DISABLE_VERSION_CHECK=1`
2. **`vllm` CLI** does not register `openai-chat-omni` backend — workaround: PATH wrapper to `python -m vllm_omni.entrypoints.cli.main`
3. **Server init crash** on 2500/900 + async-chunk (stage worker died during init); native 100/100 also failed stage init on L20X

Harness: `benchmark_results/qwen3_omni_retro/` · Config: `qwen3_omni_retro_v018.json` / `qwen3_omni_retro_v020.json`

```bash
bash benchmark_results/qwen3_omni_retro/v0.18.0/run_benchmark.sh
bash benchmark_results/qwen3_omni_retro/v0.20.0/run_benchmark.sh
```

---

## Deploy profile

Default: [`qwen3_omni_moe.yaml`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/deploy/qwen3_omni_moe.yaml) — Thinker on GPU0, Talker+Code2Wav on GPU1 (verified 2× H100).

```bash
export CUDA_VISIBLE_DEVICES=0,1
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --async-chunk --port 8000
```

Post-[#3732](https://github.com/vllm-project/vllm-omni/pull/3732): Code2Wav inner cudagraph enabled when stage-2 `enforce_eager: false`.

---

## Optimization index

| PR / doc | Area | Notes |
|----------|------|-------|
| [#3732](https://github.com/vllm-project/vllm-omni/pull/3732) | Code2Wav cudagraph | Qwen3-Omni c=10 A/B: E2EL −8%, output tok/s +8% |
| Design doc | Batching + CUDA graph + async chunk | HF vs vLLM-Omni on A100 (E2EL −93%, TTFP −99.7%) |
| `test_qwen3_omni_no_async_chunk.json` | async-chunk off | Higher TTFP baselines — use for regression only |

---

## Reproduce retro

```bash
bash benchmark_results/qwen3_omni_retro/v0.18.0/run_benchmark.sh
bash benchmark_results/qwen3_omni_retro/v0.20.0/run_benchmark.sh
bash benchmark_results/qwen3_omni_retro/cleanup.sh
```

---

## Not yet measured

- v0.20.0 vs main tag-to-tag on L20X
- `random-mm` multimodal-input workloads (audio/image/video)
- `--no-async-chunk` vs `--async-chunk` A/B on same hardware
- v0.18 retro (no comparable upstream JSON)
