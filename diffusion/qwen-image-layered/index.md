# Qwen-Image-Layered

**Category:** Diffusion (image-to-layered-image decomposition)  
**Model:** `Qwen/Qwen-Image-Layered`  
**Recipe:** `recipes/Qwen/Qwen-Image-Layered.md` (TODO)  
**Perf JSON:** `[tests/dfx/perf/tests/test_qwen_image_layered_vllm_omni.json](tests/dfx/perf/tests/test_qwen_image_layered_vllm_omni.json)`

`Qwen-Image-Layered` decomposes an input image into multiple RGBA layer images. The important request knobs are `layers`, `resolution`, `num_inference_steps`, `true_cfg_scale`, `negative_prompt`, and `seed`.

---

## Performance tracks


| Track                                | Hardware                                           | Source                                                                  |
| ------------------------------------ | -------------------------------------------------- | ----------------------------------------------------------------------- |
| **Standardized I2I (CI)**            | 1x H100 (nightly)                                  | `tests/dfx/perf/tests/test_qwen_image_layered_vllm_omni.json`           |
| **Feature A/B (manual exploratory)** | current dev GPU box (verify SKU before publishing) | `tests/dfx/perf/results/qwen_image_layered_feature_ab_20260526-100844/` |
| **H200 retro comparison**            | 4x H200 (measured)                                 | Table below                                                             |


---

## Standardized perf test (CI)


| `test_name`                             | Workload                                  | CI `latency_mean` (H100) |
| --------------------------------------- | ----------------------------------------- | ------------------------ |
| `test_qwen_image_layered_single_device` | 640x640, 20 steps, i2i, negative prompt   | -                        |
| same                                    | 1024x1024, 35 steps, i2i, negative prompt | -                        |


The current standardized perf JSON is intentionally narrow: it only covers the default single-device baseline. It does not include Cache-DiT, Ulysses, CFG-Parallel, HSDP, or VAE parallel variants for Layered yet.

```bash
cd /path/to/vllm-omni
export DIFFUSION_BENCHMARK_DIR=tests/dfx/perf/results
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
export CACHE_DIT_VERSION=1.3.0
pytest -s -v tests/dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_qwen_image_layered_vllm_omni.json
```

---

## H200 retro comparison

Measured **2026-06-08** on **4x NVIDIA H200**. Metric: `**latency_mean`** (seconds, lower is better).

Protocol: `**num-prompts=3**`, `**warmup-requests=1**`, `**warmup-num-inference-steps=1**`. Keep `max-concurrency=1`, `seed=42`, and negative prompt enabled. Use the same checkpoint, `Qwen/Qwen-Image-Layered`, across all versions.


| Config              | Workload                 | v0.20.0 | v0.22.0   | Delta v0.20->v0.22 |
| ------------------- | ------------------------ | ------- | --------- | ------------------ |
| Single device       | 640x640, 20 steps, i2i   | 14.71   | **14.47** | -1.6%              |
| Single device       | 1024x1024, 35 steps, i2i | 25.11   | **24.74** | -1.5%              |
| CacheDiT + Ulysses2 | 1024x1024, 35 steps, i2i | 6.78    | **6.62**  | -2.4%              |


**Takeaway:** v0.22.0 is slightly faster than v0.20.0 on all three measured Qwen-Image-Layered workloads, with roughly 1.5-2.4% lower mean latency. The largest absolute gain is in the 1024x1024 single-device path, where `diffuse` drops from 24.39 s to 24.01 s.

**Stacks:**


| Source            | vLLM-Omni | vLLM   | API path                      |
| ----------------- | --------- | ------ | ----------------------------- |
| v0.20.0 `4a24a51` | 0.20.0    | 0.20.0 | `vllm-omni` benchmark backend |
| v0.22.0 `963ba1a` | 0.22.0    | 0.22.0 | `/v1/chat/completions`        |


---

## Exploratory feature A/B

Manual run from `tests/dfx/perf/results/qwen_image_layered_feature_ab_20260526-100844/`. This is useful as an early signal, but should not be treated as the official H200 retro table until rerun on the target hardware and stack.


| Config                     | Workload          | `latency_mean` | `diffuse` mean | Throughput | Peak memory | Notes                         |
| -------------------------- | ----------------- | -------------- | -------------- | ---------- | ----------- | ----------------------------- |
| Default, 1 GPU             | 640x640, 20 steps | 5.120 s        | 4.268 s        | 0.195 qps  | 63,180 MB   | Baseline                      |
| CacheDiT, 1 GPU            | 640x640, 20 steps | 7.852 s        | 6.994 s        | 0.127 qps  | 62,920 MB   | Slower on this small workload |
| CacheDiT + Ulysses2, 2 GPU | 640x640, 20 steps | 5.088 s        | 4.221 s        | 0.197 qps  | 63,180 MB   | Roughly baseline latency      |
| HSDP2 standalone, 2 GPU    | 640x640, 20 steps | 21.582 s       | 20.752 s       | 0.046 qps  | 45,886 MB   | Saves memory, much slower     |


Known incompatibility: `CacheDiT + HSDP` failed because cache-dit does not support the `FSDPQwenImageTransformer2DModel` wrapper. Keep HSDP as a standalone memory-strategy comparison unless cache-dit gains support for the wrapped transformer class.

---

## Serve commands

**Single device:**

```bash
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
vllm serve Qwen/Qwen-Image-Layered --omni \
  --enable-diffusion-pipeline-profiler
```

**CacheDiT, single device:**

```bash
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
vllm serve Qwen/Qwen-Image-Layered --omni \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.24,"max_continuous_cached_steps":3,"enable_taylorseer":false,"taylorseer_order":1,"scm_steps_mask_policy":null,"scm_steps_policy":"dynamic"}' \
  --enable-diffusion-pipeline-profiler
```

**CacheDiT + Ulysses2, 2 GPUs:**

```bash
export CUDA_VISIBLE_DEVICES=0,1
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
vllm serve Qwen/Qwen-Image-Layered --omni \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.24,"max_continuous_cached_steps":3,"enable_taylorseer":false,"taylorseer_order":1,"scm_steps_mask_policy":null,"scm_steps_policy":"dynamic"}' \
  --ulysses-degree 2 \
  --enable-diffusion-pipeline-profiler
```

**CacheDiT + CFG-Parallel2, 2 GPUs:**

```bash
export CUDA_VISIBLE_DEVICES=0,1
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
vllm serve Qwen/Qwen-Image-Layered --omni \
  --cache-backend cache_dit \
  --cfg-parallel-size 2 \
  --enable-diffusion-pipeline-profiler
```

**HSDP2 standalone, 2 GPUs:**

```bash
export CUDA_VISIBLE_DEVICES=0,1
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
vllm serve Qwen/Qwen-Image-Layered --omni \
  --use-hsdp --hsdp-shard-size 2 \
  --enable-diffusion-pipeline-profiler
```

**Layerwise offload with layer-count guard:**

```bash
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
vllm serve Qwen/Qwen-Image-Layered --omni \
  --enable-layerwise-offload \
  --enable-diffusion-pipeline-profiler
```

---

## Suggested request shape

For standardized online serving benchmarks, use `/v1/chat/completions` with the random i2i dataset via `benchmarks/diffusion/diffusion_benchmark_serving.py`. For product-style validation of layered semantics, use `/v1/images/edits` with `layers` and `resolution`.

```bash
python -u benchmarks/diffusion/diffusion_benchmark_serving.py \
  --host 127.0.0.1 --port 8192 \
  --model Qwen/Qwen-Image-Layered \
  --endpoint /v1/chat/completions \
  --dataset random --task i2i \
  --width 640 --height 640 \
  --num-inference-steps 20 \
  --num-prompts 10 \
  --max-concurrency 1 \
  --warmup-requests 1 \
  --warmup-num-inference-steps 1 \
  --seed 42 \
  --enable-negative-prompt \
  --output-file /tmp/qwen_image_layered_640x640_steps20.json
```

## Related models


| Model                  | Ledger / perf JSON                                                                |
| ---------------------- | --------------------------------------------------------------------------------- |
| Qwen-Image             | `diffusion/qwen-image/index.md` · `test_qwen_image_vllm_omni.json`                |
| Qwen-Image-Edit        | `diffusion/qwen-image-edit/index.md` · `test_qwen_image_edit_2511_vllm_omni.json` |
| Qwen-Image-Edit legacy | `test_qwen_image_edit_vllm_omni.json`, `test_qwen_image_edit_2509_vllm_omni.json` |
