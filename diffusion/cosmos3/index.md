# Cosmos3

**Category:** Diffusion / world model (image + video generation) \
**Model:** `nvidia/Cosmos3-Nano` \
**Recipe:** [Cosmos3-Nano](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-Nano.md) · [Cosmos3-Super](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-Super.md)

Cosmos3-Nano is a world-model generation pipeline covering text-to-image (T2I), text-to-video (T2V), image-to-video (I2V), and video-to-video (V2V). vLLM-Omni serves the modes through a single `Cosmos3OmniDiffusersPipeline`; the endpoint and request reference determine the task.

This cookbook tracks Cosmos3 starting with the v0.24.0 local validation run. The first dataset is intended as a reusable performance baseline for the official demo workload rather than a release-over-release regression conclusion.

---

## Performance tracks

| Track | Hardware | Source |
|-------|----------|--------|
| **Official demo perf** | 2× H200 local validation | `tests/dfx/perf/tests/test_cosmos3_vllm_omni.json` |
| **Smoke / correctness** | 1× H100 | `tests/e2e/online_serving/test_cosmos3.py`, `tests/e2e/accuracy/test_cosmos3_similarity.py` |

---

## v0.24.0 local validation

Measured **2026-07-10** with `vllm==0.24.0`, local `nvidia/Cosmos3-Nano` snapshot, and the new Cosmos3 perf JSON. Metric: **`latency_mean`** (seconds, lower is better).

| Task | Endpoint | Workload | Parallelism | Completed | `latency_mean` | Throughput | Peak memory mean |
|------|----------|----------|-------------|-----------|----------------|------------|------------------|
| T2I | `/v1/images/generations` | 1024×1024, 50 steps | CFG2 + HSDP2 + VAE-pp2 | 3/3 | **7.61 s** | 0.1314 qps | not reported |
| T2V | `/v1/videos` | 1280×720, 189f, 35 steps, 24 fps | CFG2 + HSDP2 + VAE-pp2 | 3/3 | **482.98 s** | 0.00207 qps | 27424 MB |
| I2V | `/v1/videos` | 1280×720, 189f, 35 steps, 24 fps | CFG2 + HSDP2 + VAE-pp2 | 3/3 | **483.00 s** | 0.00207 qps | 27684 MB |
| V2V | `/v1/videos` | 1280×720, 189f, 35 steps, 24 fps | CFG2 + HSDP2 + VAE-pp2 | 3/3 | **483.38 s** | 0.00207 qps | 27688 MB |

Parallel config:

```bash
--cfg-parallel-size 2 \
--use-hsdp --hsdp-shard-size 2 \
--vae-patch-parallel-size 2 --vae-use-tiling \
--model-class-name Cosmos3OmniDiffusersPipeline \
--no-guardrails \
--enable-diffusion-pipeline-profiler
```

Artifacts:

- `vllm-omni/tests/dfx/perf/results/cosmos3_v024/diffusion_result_test_cosmos3_vllm_omni_20260710-074221.json`
- logs under `vllm-omni/tests/dfx/perf/results/cosmos3_v024/logs/`

Notes:

- T2I did not report peak memory through `/v1/images/generations`; video tasks did report memory.
- The selected local snapshot includes `sound_tokenizer/diffusion_pytorch_model.safetensors`; incomplete snapshots can fail during pipeline initialization when sound capability is detected.
- Video tasks use synthetic image/video references for I2V/V2V to avoid external asset downloads during perf runs.

---

## Reproduce

```bash
cd /path/to/vllm-omni
export CUDA_VISIBLE_DEVICES=6,7
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DIFFUSION_BENCHMARK_DIR=tests/dfx/perf/results/cosmos3_v024

python -m pytest -s tests/dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_cosmos3_vllm_omni.json
```

---

## v0.26.0 plan

For the next even release, reuse the same four official-demo workloads and add:

| Area | Plan |
|------|------|
| Baseline repeat | Re-run T2I/T2V/I2V/V2V with the same 2× H200 parallel config |
| Scaling | Add optional 4× H200 comparison using CFG2 + Ulysses/HSDP if recipe support is confirmed |
| Memory | Ensure image endpoint reports peak memory, or document a separate sampler |
| Quality | Pair perf results with smoke outputs for T2I and video modes |
