# LTX-2.3

**Category:** Diffusion (text-to-video with synchronized audio)

**Model:** `dg845/LTX-2.3-Diffusers`

**Pipeline:** `LTX23Pipeline`

**Recipe:** [LTX-2.3](https://github.com/vllm-project/vllm-omni/blob/main/recipes/LTX/LTX-2.3.md)

**Perf JSON:** [`test_ltx2_3_vllm_omni.json`](https://github.com/vllm-project/vllm-omni/blob/514ad762faf73fd684c4e551fdf03dc2c7ffbba9/tests/dfx/perf/tests/test_ltx2_3_vllm_omni.json) ([PR #4464](https://github.com/vllm-project/vllm-omni/pull/4464))

LTX-2.3 is tracked here as a newly onboarded video diffusion model. Current
public evidence combines a v0.23-line maintenance snapshot with v0.24-line
regression triage and PR-head recovery validation from the PR discussion, rather
than a formal release-to-release retro table. The tables below are deliberately
labeled as watchlist evidence until matching even-release comparison artifacts
are available.

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

Note: upstream PRs sometimes call the real-weight validation tier "L4". That is
a validation level, not NVIDIA L4 hardware.

---

## Performance tracks

| Track | Hardware | Source |
|-------|----------|--------|
| **L2 mock T2V contract** | CPU/mock | [PR #4440](https://github.com/vllm-project/vllm-omni/pull/4440) |
| **Full-model accuracy** | Real weights, nightly | [`test_ltx2_3_video_similarity.py`](https://github.com/vllm-project/vllm-omni/blob/main/tests/e2e/accuracy/test_ltx2_3_video_similarity.py) |
| **Full-model serving performance** | v0.23 validation environment | [`test_ltx2_3_vllm_omni.json`](https://github.com/vllm-project/vllm-omni/blob/514ad762faf73fd684c4e551fdf03dc2c7ffbba9/tests/dfx/perf/tests/test_ltx2_3_vllm_omni.json) |
| **I2V public path** | Functional / docs coverage | [PR #4381](https://github.com/vllm-project/vllm-omni/pull/4381) |
| **CFG input-prep micro-profile** | CUDA operator micro-profile | [PR #4507](https://github.com/vllm-project/vllm-omni/pull/4507) |
| **v0.24 RMSNorm recovery validation** | Discussion / PR-head validation | [PR #4956](https://github.com/vllm-project/vllm-omni/pull/4956), [cookbook PR note](https://github.com/hsliuustc0106/vllm-omni-cookbook/pull/12#issuecomment-4912471061) |

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
# Run from the vllm-omni repository root.
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

## v0.24 RMSNorm regression and PR-head recovery

This section records the PR discussion and PR-head validation snapshot for the
v0.24-line LTX-2.3 RMSNorm regression. It is not a formal cookbook release row
because the numbers were posted as triage evidence, not as a completed
even-release retro with a full artifact bundle.

Discussion-level e2e / QPS comparison:

| Version / mode | e2e latency | QPS | Source |
|----------------|------------:|----:|--------|
| v0.23.0 eager | 6014.2 ms | 0.1663 | [comment](https://github.com/hsliuustc0106/vllm-omni-cookbook/pull/12#issuecomment-4902299472) |
| v0.23.0 torch.compile | 4010 ms | 0.2493 | [comment](https://github.com/hsliuustc0106/vllm-omni-cookbook/pull/12#issuecomment-4902299472) |
| v0.24.0 eager | 6712.5 ms | 0.1489 | [comment](https://github.com/hsliuustc0106/vllm-omni-cookbook/pull/12#issuecomment-4902299472) |
| v0.24.0 torch.compile | 6014.09 ms | 0.16627 | [comment](https://github.com/hsliuustc0106/vllm-omni-cookbook/pull/12#issuecomment-4902299472) |

Treat the e2e / QPS rows as endpoint observations. The `/v1/videos` path uses
async job polling and result fetch, so a single polling tail can make
`latency_mean` and `throughput_qps` look worse even when server-side generation
time is stable. The regression attribution below uses
`stage_durations_mean.stage_0_gen_ms` and module profile measurements.

RMSNorm was identified as the v0.24 regression source. v0.23 started LTX-2.3
with `vllm_c,native` RMSNorm priority, while v0.24 selected the native path by
default. [PR #4956](https://github.com/vllm-project/vllm-omni/pull/4956) adds an
LTX-2.3-specific override for `rms_norm` and `fused_add_rms_norm`.

| Scenario | `stage_durations_mean.stage_0_gen_ms` | Interpretation |
|----------|--------------------------------------:|----------------|
| v0.23.0 compile | 4215.94 ms | Reference compile-path result |
| v0.24.0 original compile | 4710.10 ms | Regressed under native RMSNorm default |
| v0.24.0 compile with `vllm_c,native` | 4169.64 ms | Compile path recovers with the LTX override |
| v0.23.0 eager | 5014.2 ms | Reference eager result in PR #4956 |
| v0.24.0 original eager | 6712.5 ms | Eager path still regressed |
| v0.24.0 eager with `vllm_c,native` | 5007.3 ms | Eager path recovers with the LTX override |

The module profile shows the regression sitting inside transformer blocks, with
the RMSNorm calls accounting for the block-level slowdown under the v0.24 native
RMSNorm default. PR #4956 moves the same calls back to the `vllm_c,native`
priority and brings the profiled block below the v0.23 reference.

| Profile item | v0.23 | v0.24 native | PR #4956 override |
|--------------|------:|-------------:|------------------:|
| `transformer.forward` | 5118.1 ms | 5701.5 ms | 5007.3 ms |
| `block24.forward` | 151.5 ms | 167.5 ms | 147.6 ms |

| Block-24 RMSNorm item | v0.23 | v0.24 native | PR #4956 override | Native delta | Override delta |
|-----------------------|------:|-------------:|------------------:|-------------:|---------------:|
| `norm1` | 0.840 ms | 2.257 ms | 0.788 ms | +1.416 ms | -0.053 ms |
| `norm2` | 1.303 ms | 2.562 ms | 1.026 ms | +1.259 ms | -0.277 ms |
| `norm3` | 1.170 ms | 2.514 ms | 1.078 ms | +1.344 ms | -0.093 ms |
| `audio_norm1` | 1.166 ms | 2.279 ms | 1.006 ms | +1.113 ms | -0.161 ms |
| `audio_norm2` | 1.113 ms | 2.218 ms | 1.020 ms | +1.105 ms | -0.093 ms |
| `audio_norm3` | 1.045 ms | 2.073 ms | 0.837 ms | +1.028 ms | -0.208 ms |
| `audio_to_video_norm` | 1.114 ms | 2.417 ms | 1.035 ms | +1.304 ms | -0.079 ms |
| `video_to_audio_norm` | 0.692 ms | 1.619 ms | 0.631 ms | +0.927 ms | -0.061 ms |

Related PR status:

| PR | Status | LTX-2.3 relevance |
|----|--------|-------------------|
| [#4079](https://github.com/vllm-project/vllm-omni/pull/4079) | merged | Related diffusion request-level batching change; A/B check in the discussion did not identify it as the regression source |
| [#4739](https://github.com/vllm-project/vllm-omni/pull/4739) | merged | LTX-2.3 I2V support and shared T2V/I2V refactor; A/B check in the discussion did not identify it as the regression source |
| [#4956](https://github.com/vllm-project/vllm-omni/pull/4956) | open | Restores compile and eager PR-head validation by forcing LTX-2.3 RMSNorm IR ops to prefer `vllm_c` |
| [#4507](https://github.com/vllm-project/vllm-omni/pull/4507) | open | Incremental CFG input-prep optimization; still tracked separately from the v0.24 RMSNorm regression |

SGLang 0.5.14 was also posted as an external comparison point in the PR
discussion: 14025 ms / 0.0712 QPS with warmup, and 19486 ms without warmup.
Because the comment does not restate the full workload, hardware, or commit
metadata, treat it as discussion context rather than a cookbook baseline.

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
| Full-model perf guard | v0.23 window | Shape-correct warmup, measured request propagation, stage-metric assertions | [#4464](https://github.com/vllm-project/vllm-omni/pull/4464) |
| I2V public path | v0.23 window | First-frame-conditioned I2V path and public docs/examples | [#4381](https://github.com/vllm-project/vllm-omni/pull/4381) |
| Request-level batching check | v0.24 window | Related diffusion batching change; not identified as the LTX-2.3 regression source in A/B discussion | [#4079](https://github.com/vllm-project/vllm-omni/pull/4079) |
| I2V shared-pipeline refactor | v0.24 window | Adds LTX-2.3 I2V support and shared T2V/I2V stages; not identified as the regression source in A/B discussion | [#4739](https://github.com/vllm-project/vllm-omni/pull/4739) |
| RMSNorm priority override | v0.24 window | Restores compile and eager PR-head validation by preferring `vllm_c` for LTX-2.3 RMSNorm IR ops | [#4956](https://github.com/vllm-project/vllm-omni/pull/4956) |
| CFG input prep | v0.24 window | Cast video/audio latents before CFG duplication; incremental optimization still tracked separately | [#4507](https://github.com/vllm-project/vllm-omni/pull/4507) |

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
| v0.23 line | post-v0.22 | Full-model T2V perf guard, I2V public path, L2 mock split, CFG input-prep micro-profile |
| v0.24 line | current watchlist | RMSNorm priority regression triage; compile and eager PR-head recovery in [#4956](https://github.com/vllm-project/vllm-omni/pull/4956) |
