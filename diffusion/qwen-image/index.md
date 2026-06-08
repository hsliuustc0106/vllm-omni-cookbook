# Qwen-Image

**Category:** Diffusion (text-to-image)  
**Model:** `Qwen/Qwen-Image`  
**Recipe:** [Qwen-Image](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen-Image.md)  
**Serving dashboard:** [qwen_image_serving_performance.md](https://github.com/vllm-project/vllm-omni/blob/main/benchmarks/diffusion/performance_dashboard/qwen_image_serving_performance.md)

Retro benchmark harness (v0.18 / v0.20): `vllm-omni/benchmark_results/qwen_image_retro/`.

---

## Performance tracks

| Track | Hardware | Source |
|-------|----------|--------|
| **Standardized T2I (CI)** | 2× H100 (nightly) | [`test_qwen_image_vllm_omni.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_qwen_image_vllm_omni.json) |
| **v0.18 / v0.20 / main retro** | 4× H200 (measured) | [Table below](#h200-retro-comparison) · [Retro README](https://github.com/vllm-project/vllm-omni/blob/main/benchmark_results/qwen_image_retro/README.md) |

---

## Standardized perf test (CI)

| `test_name` | Workload | CI `latency_mean` (H100) |
|-------------|----------|--------------------------|
| `test_qwen_image_single_device` | 512×512, 20 steps | **3.50 s** |
| same | 1536×1536, 35 steps | **27.0 s** |
| `test_qwen_image_ulysses2_cfg2_vae_patch4` | 1536×1536, 35 steps | **9.1 s** |

Full JSON includes `step_execution` and `cache_dit` cases — retro subset uses **single_device + ulysses2** only.

```bash
cd /path/to/vllm-omni
export CUDA_VISIBLE_DEVICES=0,1,2,3 DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN VLLM_WORKER_MULTIPROC_METHOD=spawn
pytest -s tests/dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_qwen_image_vllm_omni.json
```

---

## H200 retro comparison

Measured **2026-05-22** on **4× NVIDIA H200**. Metric: **`latency_mean`** (seconds, lower is better).

Protocol: **`num-prompts=3`**, **`warmup-requests=1`** (first request excluded from measurement — avoids `torch.compile` / resolution-transition inflation).

| Config | Workload | v0.18.0 | v0.20.0 | main (`89e389cf`) | Δ v0.20→main |
|--------|----------|---------|---------|-------------------|--------------|
| Single device | 512×512, 20 steps | **2.20** | 2.39 | 2.40 | +0.4% |
| Single device | 1536×1536, 35 steps | **23.96** | 24.48 | 24.46 | −0.1% |
| Ulysses2 + CFG2 + VAE-pp4 | 1536×1536, 35 steps | **8.16** | 8.42 | 8.56 | +1.7% |

**Takeaway:** Latest **main** tracks **v0.20** within ~2% on these workloads. **v0.18** remains marginally fastest on this H200 retro (~2–9% vs v0.20/main).

**Stacks:**

| Source | vLLM-Omni | vLLM | API path |
|--------|-----------|------|----------|
| v0.18.0 tag | 0.18.0 | 0.18.0 | `vllm-omni` |
| v0.20.0 tag | 0.20.0 | 0.20.0 | `vllm-omni` |
| main `89e389cf` | 0.20.1.dev167 | 0.21.0 | `/v1/chat/completions` |

**Artifacts (warmup=1 reruns):**

- `v0.18.0/benchmark_results_qwen_image_retro_v018_rerun_20260522-013621.json`
- `v0.20.0/diffusion_result_qwen_image_retro_v020_rerun_20260522-020009.json`
- `main/diffusion_result_qwen_image_retro_main_rerun_20260522-101723.json`

Retro runners: `benchmark_results/qwen_image_retro/{v0.18.0,v0.20.0,main}/run_benchmark_rerun.sh`.

> **Note:** v0.16.0 is omitted — no upstream perf JSON, different `openai` backend, SDPA fallback, and incomplete parallel CLI. Not comparable to v0.18/v0.20.

---

## Serve commands (matches perf JSON)

**Single device:**

```bash
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
vllm serve Qwen/Qwen-Image --omni --enable-diffusion-pipeline-profiler
```

**Ulysses2 + CFG2 + VAE-pp4 (4 GPUs):**

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve Qwen/Qwen-Image --omni \
  --ulysses-degree 2 --cfg-parallel-size 2 --vae-patch-parallel-size 4 --vae-use-tiling \
  --enable-diffusion-pipeline-profiler
```

---

## Related models

| Model | Perf JSON |
|-------|-----------|
| Qwen-Image-Edit | `test_qwen_image_edit_vllm_omni.json` |
| Qwen-Image-Edit-2509 | `test_qwen_image_edit_2509_vllm_omni.json` |
| Qwen-Image-Layered | [qwen-image-layered](../qwen-image-layered/index.md) · `test_qwen_image_layered_vllm_omni.json` |

Use the same `diffusion-perf-cookbook` skill to scaffold retro for these models.
