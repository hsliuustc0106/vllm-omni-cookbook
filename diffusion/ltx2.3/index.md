# LTX-2.3

**Category:** Diffusion (text-to-video with synchronized audio)

**Model:** `dg845/LTX-2.3-Diffusers`

**Pipeline:** `LTX23Pipeline`

**Recipe:** [LTX-2.3](https://github.com/vllm-project/vllm-omni/blob/main/recipes/LTX/LTX-2.3.md)

**Perf JSON:** [`test_ltx2_3_vllm_omni.json`](https://github.com/vllm-project/vllm-omni/blob/514ad762faf73fd684c4e551fdf03dc2c7ffbba9/tests/dfx/perf/tests/test_ltx2_3_vllm_omni.json) ([PR #4464](https://github.com/vllm-project/vllm-omni/pull/4464))

LTX-2.3 is tracked here as a newly onboarded video diffusion model. Current
public evidence is a v0.23-line maintenance snapshot rather than an even-release
retro table, so the tables below are deliberately labeled as watchlist evidence
until v0.24.0 or another even-release comparison is measured.

Checkpoint note: use `dg845/LTX-2.3-Diffusers` as the runnable Diffusers-layout
model id in cookbook commands. Treat upstream `Lightricks/LTX-2.3` as a
reference checkpoint unless loader support for that exact layout has been
re-validated.

---

## Key metrics

| Metric | Definition | Lower / higher? | When it matters |
|--------|------------|-----------------|-----------------|
| `stage_durations_mean.stage_0_gen_ms` | Server-side generation time for the diffusion stage | Lower | Main CI guard for the LTX-2.3 T2V pipeline |
| `latency_mean` | Client-observed `/v1/videos` request latency | Lower | Endpoint observation; includes async job polling and result fetch |
| `throughput_qps` | Completed video requests per second | Higher | Endpoint observation; not the active regression assertion |
| `peak_memory_mb_mean` | Mean peak memory reported by the benchmark | Lower | Memory regression guard |

---

## Performance tracks

| Track | Hardware | Source |
|-------|----------|--------|
| **L2 mock T2V contract** | CPU/mock | [PR #4440](https://github.com/vllm-project/vllm-omni/pull/4440) |
| **L4 accuracy** | Full model, nightly | [`test_ltx2_3_video_similarity.py`](https://github.com/vllm-project/vllm-omni/blob/main/tests/e2e/accuracy/test_ltx2_3_video_similarity.py) |
| **L4 serving performance** | v0.23 validation environment | [`test_ltx2_3_vllm_omni.json`](https://github.com/vllm-project/vllm-omni/blob/514ad762faf73fd684c4e551fdf03dc2c7ffbba9/tests/dfx/perf/tests/test_ltx2_3_vllm_omni.json) |
| **I2V public path** | Functional / docs coverage | [PR #4381](https://github.com/vllm-project/vllm-omni/pull/4381) |
| **CFG input-prep micro-profile** | CUDA operator micro-profile | [PR #4507](https://github.com/vllm-project/vllm-omni/pull/4507) |

---

## Standardized T2V perf test

| Item | Value |
|------|-------|
| Config | [`test_ltx2_3_vllm_omni.json`](https://github.com/vllm-project/vllm-omni/blob/514ad762faf73fd684c4e551fdf03dc2c7ffbba9/tests/dfx/perf/tests/test_ltx2_3_vllm_omni.json) |
| Runner | [`run_diffusion_benchmark.py`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/scripts/run_diffusion_benchmark.py) |
| Endpoint | `/v1/videos` |
| Workload | 512x384, 25 frames, 24 fps, 20 steps |
| Requests | 3 successful measured requests, `max_concurrency=1` |
| Warmup | compile guard warms 2 requests at 25 frames, 20 steps |
| Runtime in current evidence | `vllm==0.23.0` |
| Evidence head | [PR #4464](https://github.com/vllm-project/vllm-omni/pull/4464) head `514ad762` |

```bash
cd /path/to/vllm-omni
pytest -q -s tests/dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_ltx2_3_vllm_omni.json \
  --assert-baseline
```

Current PR-head assertion result from PR #4464:

```text
2 passed, 17 warnings in 319.76s (0:05:19)
```

| Case | Asserted metric | Configured baseline | Measured result | Assertion threshold |
|------|-----------------|--------------------:|----------------:|--------------------:|
| eager / no graph | `stage_durations_mean.stage_0_gen_ms` | 5339.81 ms | 5537.3076 ms | <= 5873.791 ms |
| eager / no graph | `peak_memory_mb_mean` | 73291 MB | 73290.67 MB | <= 80620.1 MB |
| torch.compile / graph | `stage_durations_mean.stage_0_gen_ms` | 4092.05 ms | 4025.3634 ms | <= 4501.255 ms |
| torch.compile / graph | `peak_memory_mb_mean` | 73290 MB | 73290.00 MB | <= 80619 MB |

Observed `/v1/videos` endpoint output for the same runs:

| Case | `throughput_qps` | `latency_mean` |
|------|-----------------:|---------------:|
| eager / no graph | 0.1663 | 6.0145 s |
| torch.compile / graph | 0.1651 | 6.0576 s |

The benchmark uses server-side `stage_durations_mean.stage_0_gen_ms` as the
regression signal because `/v1/videos` top-level latency also includes polling
and result-fetch overhead.

---

## v0.22.0 status

No even-release LTX-2.3 retro table has been published yet. The model is listed
in `SUMMARY.md` as tracked, with the first formal release-to-release comparison
left for a future even release once matching artifacts exist.

---

## Current optimization summary

| Phase | Release / window | Focus | Notable PRs |
|-------|------------------|-------|-------------|
| Auxiliary module placement | v0.22 window | Keep LTX-2.3 auxiliary modules resident by default; preserve explicit offload semantics | [#4144](https://github.com/vllm-project/vllm-omni/pull/4144) |
| Offload correctness | v0.22 window | Register the RMSNorm no-affine identity weight as a non-persistent buffer | [#4278](https://github.com/vllm-project/vllm-omni/pull/4278) |
| L2 guard split | v0.23 window | CPU/mock shape and metadata guard without runner or worker init | [#4440](https://github.com/vllm-project/vllm-omni/pull/4440) |
| L4 perf guard | v0.23 window | Shape-correct warmup, measured request propagation, stage-metric assertions | [#4464](https://github.com/vllm-project/vllm-omni/pull/4464) |
| I2V public path | v0.23 window | First-frame-conditioned I2V path and public docs/examples | [#4381](https://github.com/vllm-project/vllm-omni/pull/4381) |
| CFG input prep | v0.23 window | Cast video/audio latents before CFG duplication | [#4507](https://github.com/vllm-project/vllm-omni/pull/4507) |

### CFG input-prep micro-profile

PR #4507 reports a narrow CUDA operator micro-profile on LTX-2.3 512x384,
25-frame latent-scale shapes:

| Path | Target ops device time |
|------|-----------------------:|
| old path, `aten::cat` + cast `aten::copy_` | 13.54 ms |
| new path, `aten::cat` + cast `aten::copy_` | 9.83 ms |

That is a 27.5% reduction in the profiled target ops. Alternating CUDA-event
timing over 12 rounds showed a smaller local input-prep gain: 0.6480 ms to
0.6330 ms per 20 input-prep iterations, or 2.3%.

---

## Serve command

Minimal T2V serving command matching the tracked pipeline:

```bash
vllm serve dg845/LTX-2.3-Diffusers \
  --omni \
  --model-class-name LTX23Pipeline
```

Use the upstream recipe for full deployment options and examples:
[recipes/LTX/LTX-2.3.md](https://github.com/vllm-project/vllm-omni/blob/main/recipes/LTX/LTX-2.3.md).

---

## Release index

| Release | Date | LTX-2.3 perf highlight |
|---------|------|------------------------|
| [v0.22.0](https://github.com/vllm-project/vllm-omni/releases) | upcoming | Tracked model; no even-release retro table yet |
| v0.23 line | post-v0.22 | L4 T2V perf guard, I2V public path, L2 mock split, CFG input-prep micro-profile |
