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

## L20X retro — status

**In progress.** Phase-1 subset (2× L20X, `CUDA_VISIBLE_DEVICES=2,3`, `--async-chunk`):

| Phase | c | prompts | Workload |
|-------|--:|--------:|----------|
| latency | 1 | 4 | text+audio random 2500/900 |
| throughput | 10 | 40 | text+audio (matches [#3732](https://github.com/vllm-project/vllm-omni/pull/3732) PR smoke) |
| throughput | 4 | 16 | text+audio mid-concurrency |

Results table: _pending first run._

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
bash benchmark_results/qwen3_omni_retro/main/run_benchmark.sh
bash benchmark_results/qwen3_omni_retro/main/run_benchmark_throughput.sh
bash benchmark_results/qwen3_omni_retro/cleanup.sh
```

---

## Not yet measured

- v0.20.0 vs main tag-to-tag on L20X
- `random-mm` multimodal-input workloads (audio/image/video)
- `--no-async-chunk` vs `--async-chunk` A/B on same hardware
- v0.18 retro (no comparable upstream JSON)
