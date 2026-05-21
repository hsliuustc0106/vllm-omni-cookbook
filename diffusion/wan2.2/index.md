# WAN2.2

**Category:** Diffusion (image / video generation)  
**Models:** `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, `Wan-AI/Wan2.2-I2V-A14B-Diffusers`, `Wan-AI/Wan2.2-TI2V-5B-Diffusers`, Wan2.2-S2V

**Blog (narrative):** [Optimizing Wan2.2 I2V in vLLM-Omni](wan22-i2v-performance.html) — pipeline walkthrough, release comparison, and reproduction guide.  
**Zhihu draft (中文):** [wan22-i2v-performance-zhihu.md](wan22-i2v-performance-zhihu.md)

**Recipes:** [I2V](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Wan-AI/Wan2.2-I2V.md) · [S2V](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Wan-AI/Wan2.2-S2V.md) · [T2V dashboard](https://github.com/vllm-project/vllm-omni/blob/main/benchmarks/diffusion/performance_dashboard/wan_2_2_serving_performance.md)

Wan2.2 is a Diffusion Transformer (DiT) family for T2V, I2V, TI2V, and S2V. The 14B variants use a **dual-transformer MoE cascade** (low-noise / high-noise experts).

This cookbook **does not fork** upstream benchmark JSON. When comparing releases or hardware, record **CFG**, **USP**, **TP**, **HSDP**, and **VAE patch parallel**.

---

## Performance tracks

| Track | Hardware | Source |
|-------|----------|--------|
| **Standardized I2V (CI)** | 2× H100 80GB (nightly) | [`test_wan22_i2v_vllm_omni.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_wan22_i2v_vllm_omni.json) |
| **v0.16 / v0.18 / v0.20 retro** | 4× H200 (measured) | [Table below](#h200-retro-comparison) · [blog](wan22-i2v-performance.html#results) |
| **T2V serving dashboard** | A100-SXM4-80GB | [`wan_2_2_serving_performance.md`](https://github.com/vllm-project/vllm-omni/blob/main/benchmarks/diffusion/performance_dashboard/wan_2_2_serving_performance.md) |
| **NPU** | 8× Ascend A2 / A3 | [I2V recipe (NPU)](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Wan-AI/Wan2.2-I2V.md#npu) |

---

## Standardized I2V perf test

| Item | Value |
|------|--------|
| **Config** | [`test_wan22_i2v_vllm_omni.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_wan22_i2v_vllm_omni.json) ([#3063](https://github.com/vllm-project/vllm-omni/pull/3063), first in **v0.20.0**) |
| **Runner** | [`run_diffusion_benchmark.py`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/scripts/run_diffusion_benchmark.py) |
| **Nightly** | `.buildkite/test-nightly.yml` → “Diffusion X2V · Perf Test”, **2× H100**, rolling `main` |
| **CI baselines** | **26.0 / 21.6 / 101.6 s** — static H100 thresholds, not auto-updated per release |

| `test_name` | Serve args | Workload | CI `latency_mean` |
|-------------|------------|----------|-------------------|
| `test_wan22_i2v_single_device` | profiler only | 832×480, 81f, 4 steps | **26.0 s** |
| `test_wan22_i2v_usp2_vae_patch2_hsdp_slicing` | `usp=2`, `vae-patch-parallel-size=2`, `use-hsdp`, `vae-use-slicing` | 832×480, 81f, 4 steps | **21.6 s** |
| same | same | 1280×720, 121f, 4 steps | **101.6 s** |

Shared: `task=i2v`, `max-concurrency=1`, `seed=42`. CI uses `num-prompts=10`; local retro configs default to **`num-prompts=3`**, **`warmup-requests=0`**.

```bash
cd /path/to/vllm-omni
export CUDA_VISIBLE_DEVICES=0,1,2,3 DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN VLLM_WORKER_MULTIPROC_METHOD=spawn
pytest -s tests/dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_wan22_i2v_vllm_omni.json
```

On non-H100 hardware, set `"skip-performance-assertion": true` per `benchmark_params` in the **vllm-omni** repo.

---

## H200 retro comparison

Measured **2026-05-20** on **4× NVIDIA H200** (L20X SKU). Metric: **`latency_mean`** (seconds, lower is better). Full methodology: [blog § Results](wan22-i2v-performance.html#results).

| Config | Workload | v0.16.0 | v0.18.0 | v0.20.0 | Δ v0.18→v0.20 | Δ v0.16→v0.20 |
|--------|----------|---------|---------|---------|---------------|---------------|
| Single device | 832×480, 81f, 4 steps | **31.33** | **23.56** | **22.17** | −5.9% | −29.2% |
| USP2 + HSDP + slicing † | 832×480, 81f, 4 steps | **22.20** | **20.26** ‡ | **16.43** ‡ | −18.9% | −26.0% |
| USP2 + HSDP + slicing † | 1280×720, 121f, 4 steps | **133.94** § | **93.67** ‡ | **79.19** ‡ | −15.5% | −40.9% |

† v0.16: no `vae-patch-parallel-size=2`; GPUs `0,1,3`. ‡ v0.18/v0.20: full USP2 stack; GPUs `0,1,2,3`. § v0.16 720p: 3 prompts; others: 10.

**Stacks:** v0.16 → `vllm==0.16.0`; v0.18 → `vllm==0.18.0`; v0.20 → `vllm==0.20.0`.  
**Artifacts:** `vllm-omni/benchmark_results/wan22_retro/{v0.16.0,v0.18.0,v0.20.0}/` · configs: `.../wan22_retro/config/`.

Retro runners: `benchmark_results/wan22_retro/v0.{16,18,20}.0/run_benchmark.sh` (v0.18/v0.16 use `run_retro.py` for `/v1/videos` backend).

---

## Optimization guide (summary)

End-to-end path: **API → text encode → preprocess → VAE encode → DiT (4 steps) → VAE decode → video**.

| Phase | Release | Focus | Notable PRs |
|-------|---------|-------|-------------|
| Parallel stack | v0.14–v0.16 | SP, VAE-pp, `/v1/videos`, HSDP | #966, #756, #1073, #1339 |
| Serving overhead | v0.18 | IPC, load | #1715, #1504 |
| Preprocess / VAE | v0.20 | Dedup, BF16, GPU overlap | #2963, #2852, #2391 |
| DiT kernels | v0.20 | Fused norms, RoPE, FA fix | #2583, #2585, #2393, #2459, #3327 |
| Runtime + CI | v0.20 | vLLM 0.20 rebase, perf JSON | #3232, #3063 |

Step-by-step walkthrough, profiler checklist, and PR deep-dives: **[blog post](wan22-i2v-performance.html)**.

---

## Release index

| Release | Date | I2V perf highlight |
|---------|------|-------------------|
| [v0.21.0](https://github.com/vllm-project/vllm-omni/releases) | upcoming | Pipeline parallel #2322, FP8 #2920 |
| [v0.20.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.20.0) | 2026-05-07 | Fused DiT + CI JSON; H200 **22.17 / 16.43 / 79.19 s** |
| [v0.18.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.18.0) | 2026-03-28 | IPC −17.5% (other workload); H200 **23.56 / 20.26 / 93.67 s** |
| [v0.16.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.16.0) | 2026-02-28 | `/v1/videos` API; H200 retro **31.33 / 22.20 / 133.94 s** |
| [v0.14.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.14.0) | 2026-01-31 | First stable Wan2.2 pipelines |

Full PR lists per release: [blog § Release timeline](wan22-i2v-performance.html#timeline).

---

## Serve commands (GPU, matches perf JSON)

**Single device:**

```bash
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers --omni --enable-diffusion-pipeline-profiler
```

**USP2 + VAE-pp2 + HSDP + slicing (4 GPUs):**

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers --omni \
  --usp 2 --vae-patch-parallel-size 2 --use-hsdp --vae-use-slicing \
  --enable-diffusion-pipeline-profiler
```

8-card production and NPU commands: [Wan2.2 I2V recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Wan-AI/Wan2.2-I2V.md).

---

## Related tests

| Path | Role |
|------|------|
| `tests/e2e/online_serving/test_wan22_expansion.py` | Correctness |
| `tests/e2e/accuracy/wan22_i2v/` | Quality (SSIM/PSNR) |
| `tests/dfx/stability/tests/test_wan22.json` | Long-run stability |
