# Qwen-Image-Edit

**Category:** Diffusion (image-to-image editing)  
**Model:** `Qwen/Qwen-Image-Edit-2511` (tracked checkpoint)  
**Older versions:** [`Qwen/Qwen-Image-Edit`](https://huggingface.co/Qwen/Qwen-Image-Edit), [`Qwen/Qwen-Image-Edit-2509`](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) — **2511** is the latest release with stable multi-image input support.  
**Recipe:** [Qwen-Image-Edit](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen-Image-Edit.md)  
**Perf JSON (2511):** [`test_qwen_image_edit_2511_vllm_omni.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_qwen_image_edit_2511_vllm_omni.json)  
**Perf JSON (2509, v0.20 retro):** [`test_qwen_image_edit_2509_vllm_omni.json`](https://github.com/vllm-project/vllm-omni/blob/v0.20.0/tests/dfx/perf/tests/test_qwen_image_edit_2509_vllm_omni.json)

---

## Performance tracks

| Track | Hardware | Source |
|-------|----------|--------|
| **Standardized I2I (CI)** | 4× H100 (nightly) | [`test_qwen_image_edit_2511_vllm_omni.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_qwen_image_edit_2511_vllm_omni.json) |
| **Manual cross-version retro tests** | 4× H200 (measured) | [Table below](#h200-retro-comparison) |

---

## Standardized perf test (CI)

| `test_name` | Workload | CI `latency_mean` (H100) |
|-------------|----------|--------------------------|
| `test_qwen_image_edit_2511_single_device` | 512×512, 20 steps, 2 input images | — |
| same | 1536×1536, 35 steps, 2 input images | — |
| `test_qwen_image_edit_2511_ulysses2_cfg2_vae_patch4` | 1536×1536, 35 steps, 2 input images | — |

Full JSON includes `ulysses2_cfg2_cache_dit` cases — retro subset uses **single_device + ulysses2_cfg2_vae_patch4** only (same split as [Qwen-Image](../qwen-image/index.md#standardized-perf-test-ci)).

```bash
cd /path/to/vllm-omni
export CUDA_VISIBLE_DEVICES=0,1,2,3
pytest -s tests/dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_qwen_image_edit_2511_vllm_omni.json
```

---

## H200 retro comparison

Measured on **4× NVIDIA H200**. Metric: **`latency_mean`** (seconds, lower is better).

Protocol: the same as [`tests/dfx/perf/tests/test_qwen_image_edit_2511_vllm_omni.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_qwen_image_edit_2511_vllm_omni.json) — **`warmup-requests=1`** (first request excluded from measurement), then **`num-prompts=10`** measured prompts per workload (`512×512` / `1536×1536`, **2 input images**, negative prompt enabled).

| Config | Workload | v0.20.0† | v0.22.0 (`ee336015`) | v0.24.0 (`dda88a6f`) |
|--------|----------|----------|----------------------|---------------------|
| Single device | 512×512, 20 steps, 2 img | 14.69 | 14.51 | **14.51** |
| Single device | 1536×1536, 35 steps, 2 img | 57.68 | 57.23 | **57.05** |
| Ulysses2 + CFG2 + VAE-pp4 | 1536×1536, 35 steps, 2 img | 19.05 | 19.03 | **19.14** |

| Config | Workload | Δ v0.20→v0.22 | Δ v0.22→v0.24 |
|--------|----------|---------------|---------------|
| Single device | 512×512, 20 steps, 2 img | −1.2% | 0.0% |
| Single device | 1536×1536, 35 steps, 2 img | −0.8% | −0.3% |
| Ulysses2 + CFG2 + VAE-pp4 | 1536×1536, 35 steps, 2 img | −0.1% | +0.6% |

† **v0.20.0** retro runs used checkpoint **`Qwen/Qwen-Image-Edit-2509`**, not **2511** — early deltas compare different model weights on the same standardized workloads. But since they use the same pipeline and only differ in weights, the comparison is meaningful.

**Takeaway:** In recent versions, Qwen Image Edit 2511 is within ~1% performance change for these H200 i2i workloads; serving-stack parity is already tight despite the checkpoint mismatch.

---

## Pipeline profiler (1536×1536)

Diffusion pipeline profiler at the same **2511** inputs as the retro benchmark (`1536×1536`, 35 steps, 2 input images). Covers the two parallelism configs in the retro table (512×512 omitted).

| v0.22.0 (`ee336015`)      | GPU | GPU busy (%) | Comm / overhead (%) |
|---------------------------|-----|--------------|---------------------|
| Single device             | 0 | 98.63 | 1.37 |
| Ulysses2 + CFG2 + VAE-pp4 | 0 | 76.39 | 23.61 |
| same | 1 | 74.54 | 25.46 |
| same | 2 | 75.32 | 24.68 |
| same | 3 | 77.49 | 22.51 |

| v0.24.0 (`dda88a6f`)        | GPU | GPU busy (%) | Comm / overhead (%) |
|-----------------------------|-----|--------------|---------------------|
| Single device               | 0 | 97.60 | 2.40 |
| Ulysses2 + CFG2 + VAE-pp4   | 0 | 63.73 | 36.27 |
| same | 1 | 61.75 | 38.25 |
| same | 2 | 63.04 | 36.96 |
| same | 3 | 64.60 | 35.40 |

| Version              | Single device | Ulysses2 + CFG2 + VAE-pp4 |
|----------------------|---------------|---------------------------|
| v0.22.0 (`ee336015`) | 18.3 | 17.3 |
| v0.24.0 (`dda88a6f`) | 18.3 | 17.4 |

---

## Serve commands (matches perf JSON)

**Single device:**

```bash
vllm serve Qwen/Qwen-Image-Edit-2511 --omni --enable-diffusion-pipeline-profiler
```

**Ulysses2 + CFG2 + VAE-pp4 (4 GPUs):**

```bash
vllm serve Qwen/Qwen-Image-Edit-2511 --omni \
  --ulysses-degree 2 --cfg-parallel-size 2 --vae-patch-parallel-size 4 --vae-use-tiling \
  --enable-diffusion-pipeline-profiler
```

---

## Related models

| Model | Ledger / perf JSON |
|-------|-------------------|
| Qwen-Image (T2I) | [qwen-image](../qwen-image/index.md) · `test_qwen_image_vllm_omni.json` |
| Qwen-Image-Edit (legacy) | `test_qwen_image_edit_vllm_omni.json` |
| Qwen-Image-Layered | `test_qwen_image_layered_vllm_omni.json` |
