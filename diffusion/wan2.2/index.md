# WAN2.2

**Category:** Diffusion (image / video generation)  
**Models:** `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, `Wan-AI/Wan2.2-I2V-A14B-Diffusers`, `Wan-AI/Wan2.2-TI2V-5B-Diffusers`, Wan2.2-S2V  
**Recipes (vLLM-Omni):**

- [Wan2.2 I2V](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Wan-AI/Wan2.2-I2V.md)
- [Wan2.2 S2V](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Wan-AI/Wan2.2-S2V.md)
- [Serving performance dashboard](https://github.com/vllm-project/vllm-omni/blob/main/benchmarks/diffusion/performance_dashboard/wan_2_2_serving_performance.md)

Diffusion Transformer (DiT) family for text-to-video, image-to-video, text-image-to-video, and speech-to-video. The 14B variants use a **dual-transformer MoE cascade** (low-noise / high-noise experts).

## Hardware Baselines

| Track | Hardware | Primary workloads |
|-------|----------|-------------------|
| **GPU nightly** | NVIDIA H100 80GB | I2V perf regression (`test_wan22_i2v_vllm_omni.json`) |
| **GPU dashboard** | NVIDIA A100-SXM4-80GB | T2V serving (`wan_2_2_serving_performance.md`) |
| **NPU** | 8× Ascend A2 / A3 | I2V/T2V with mindie-sd + MXFP8 ([recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Wan-AI/Wan2.2-I2V.md#npu)) |

Record **CFG**, **USP** (`--usp`), **TP**, **HSDP**, and **VAE patch parallel** when comparing numbers.

---

## v0.20.0 (2026-05-01) — Baseline

First stable release tracked in this cookbook. Performance tables below use **CI baselines** and the **official T2V dashboard** from vLLM-Omni main (May 2026).

### Performance — I2V (GPU, nightly CI)

Model: `Wan-AI/Wan2.2-I2V-A14B-Diffusers`  
Workload: random dataset, **832×480**, **81 frames**, **4 steps**, concurrency **1**, negative prompt enabled.

| Config | E2E latency (mean) | Throughput | Peak GPU memory (mean) | Delta vs single-GPU |
|--------|-------------------|------------|------------------------|---------------------|
| Single device | **26.0 s** | 0.034 qps | ~80 GB | — |
| USP=2, VAE-pp=2, HSDP, VAE slicing | **21.6 s** | 0.042 qps | ~55 GB | **−17%** latency |

Same parallel config at **1280×720**, **121 frames**, **4 steps**:

| Config | E2E latency (mean) | Throughput | Peak GPU memory (mean) |
|--------|-------------------|------------|------------------------|
| USP=2, VAE-pp=2, HSDP, VAE slicing | **101.6 s** | 0.0085 qps | ~65 GB |

Source: [`tests/dfx/perf/tests/test_wan22_i2v_vllm_omni.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_wan22_i2v_vllm_omni.json) (nightly via [PR #3063](https://github.com/vllm-project/vllm-omni/pull/3063)).

### Performance — T2V (GPU, serving dashboard)

Model: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`  
Config: **CFG=2**, **USP=2**, **TP=1**, **HSDP=On**, concurrency **1**, FlashAttention.

| Dataset | Resolution / steps | VAE parallel | Mean latency | Notes |
|---------|-------------------|--------------|--------------|-------|
| A | 480p, 3 steps | 1 | 24.68 s | |
| A | 480p, 3 steps | **4** | **21.68 s** | −12% vs VAE-pp=1 |
| B | 720p, 6 steps | 1 | 124.66 s | |
| B | 720p, 6 steps | **4** | **117.44 s** | −6% vs VAE-pp=1 |
| C (mix) | mixed | 1 | 79.22 s (mean) | P99 124.26 s |
| C (mix) | mixed | **4** | **74.50 s** (mean) | P99 117.71 s |

Source: [`wan_2_2_serving_performance.md`](https://github.com/vllm-project/vllm-omni/blob/main/benchmarks/diffusion/performance_dashboard/wan_2_2_serving_performance.md).

### Performance — NPU (recipe guidance)

Not captured in cookbook CI yet. The Ascend recipe documents:

- **Laser Attention** (`MINDIE_SD_FA_TYPE=ascend_laser_attention`): up to **~40%** improvement at 720p (tested workloads).
- **MXFP8 W8A8** ([#3140](https://github.com/vllm-project/vllm-omni/pull/3140)): validated for T2V / I2V / TI2V on Ascend NPU.

### Optimization Notes

Cumulative improvements landing in and before v0.20.0, grouped by theme.

#### Parallelism & scaling

| PR | Merged | Summary |
|----|--------|---------|
| [#964](https://github.com/vllm-project/vllm-omni/pull/964) | 2026-02-05 | Tensor parallelism for 14B DiT |
| [#966](https://github.com/vllm-project/vllm-omni/pull/966) | 2026-01-27 | Ulysses sequence parallelism |
| [#851](https://github.com/vllm-project/vllm-omni/pull/851) | 2026-01-30 | CFG parallel abstraction |
| [#2459](https://github.com/vllm-project/vllm-omni/pull/2459) | 2026-04-03 | Skip Ulysses SP on short cross-attention |
| [#1339](https://github.com/vllm-project/vllm-omni/pull/1339) | 2026-02-26 | HSDP for diffusion models |
| [#756](https://github.com/vllm-project/vllm-omni/pull/756) | 2026-02-10 | VAE patch parallelism |
| [#2322](https://github.com/vllm-project/vllm-omni/pull/2322) | 2026-05-19 | Pipeline parallelism integrated into Wan 2.2 |
| [#2969](https://github.com/vllm-project/vllm-omni/pull/2969) | 2026-04-21 | NPU VAE parallel: `gather` → `all_gather` (−480ms–1.5s HCCL stall) |

#### Memory & caching

| PR | Merged | Summary |
|----|--------|---------|
| [#980](https://github.com/vllm-project/vllm-omni/pull/980) | 2026-01-29 | Conditional dual-transformer loading (MoE experts) |
| [#1021](https://github.com/vllm-project/vllm-omni/pull/1021) | 2026-01-28 | Cache-DiT step caching |
| [#858](https://github.com/vllm-project/vllm-omni/pull/858) | 2026-01-30 | Layerwise CPU offloading |
| [#3224](https://github.com/vllm-project/vllm-omni/pull/3224) | 2026-05-05 | Offload transformer-1 after switch to transformer-2 |
| [#1392](https://github.com/vllm-project/vllm-omni/pull/1392) | 2026-03-22 | cache-dit fix for TI2V-5B (single transformer) |

#### Kernels & operators

| PR | Merged | Platform | Summary |
|----|--------|----------|---------|
| [#2583](https://github.com/vllm-project/vllm-omni/pull/2583) | 2026-04-16 | GPU | Fused RMSNorm |
| [#2585](https://github.com/vllm-project/vllm-omni/pull/2585) | 2026-04-15 | GPU | Fused AdaLayerNorm |
| [#2393](https://github.com/vllm-project/vllm-omni/pull/2393) | 2026-04-02 | Both | Rotary embedding optimization |
| [#2391](https://github.com/vllm-project/vllm-omni/pull/2391) | 2026-04-08 | Both | I2V VAE FP32 → BF16 |
| [#3327](https://github.com/vllm-project/vllm-omni/pull/3327) | 2026-05-04 | GPU | Flash Attention / CUBLAS fix |
| [#2576](https://github.com/vllm-project/vllm-omni/pull/2576) | 2026-04-18 | NPU | RMSNorm via mindie-sd |
| [#2575](https://github.com/vllm-project/vllm-omni/pull/2575) | 2026-04-16 | NPU | AdaLayerNorm on NPU |
| [#3067](https://github.com/vllm-project/vllm-omni/pull/3067) | 2026-04-23 | NPU | Fused RMSNorm replaces `WanRMS_norm` |
| [#2571](https://github.com/vllm-project/vllm-omni/pull/2571) | 2026-04-23 | NPU | mindiesd fused RoPE + cache |
| [#3400](https://github.com/vllm-project/vllm-omni/pull/3400) | 2026-05-07 | NPU | WanRMS_norm patch reliability fix |
| [#3140](https://github.com/vllm-project/vllm-omni/pull/3140) | 2026-05-13 | NPU | MXFP8 W8A8 online/offline quant (T2V/I2V/TI2V) |
| [#2640](https://github.com/vllm-project/vllm-omni/pull/2640) | 2026-05-13 | NPU | Online FP8 quantization for FA |
| [#3463](https://github.com/vllm-project/vllm-omni/pull/3463) | 2026-05-13 | ROCm | wan22 platform fixes |

#### Serving & pipeline

| PR | Merged | Summary |
|----|--------|---------|
| [#1073](https://github.com/vllm-project/vllm-omni/pull/1073) | 2026-02-12 | Online `/v1/videos` API |
| [#1715](https://github.com/vllm-project/vllm-omni/pull/1715) | 2026-03-09 | Reduce IPC overhead — **37.5s → 31.0s** e2e on one I2V online config (−17.5%) |
| [#1504](https://github.com/vllm-project/vllm-omni/pull/1504) | 2026-03-02 | Multi-thread safetensors load (~5min → faster for 14B I2V) |
| [#2963](https://github.com/vllm-project/vllm-omni/pull/2963) | 2026-04-21 | Remove duplicate video preprocess |
| [#2852](https://github.com/vllm-project/vllm-omni/pull/2852) | 2026-04-20 | Free GPU during I2V image preprocess |
| [#2672](https://github.com/vllm-project/vllm-omni/pull/2672) | 2026-04-17 | Pipeline refactor + unit tests |
| [#2751](https://github.com/vllm-project/vllm-omni/pull/2751) | 2026-05-06 | Wan2.2-S2V modeling |
| [#2134](https://github.com/vllm-project/vllm-omni/pull/2134) | 2026-04-13 | LightX2V offline conversion path for I2V |

#### Foundation

| PR | Merged | Summary |
|----|--------|---------|
| [#202](https://github.com/vllm-project/vllm-omni/pull/202) | 2025-12-11 | T2V pipeline |
| [#329](https://github.com/vllm-project/vllm-omni/pull/329) | 2025-12-25 | I2V + TI2V pipelines |
| [#1279](https://github.com/vllm-project/vllm-omni/pull/1279) | 2026-02-26 | Irregular output shapes |

### GPU vs NPU quick reference

| Dimension | GPU (CUDA) | NPU (Ascend) |
|-----------|------------|--------------|
| Fused ops | In-repo RMSNorm / AdaLN; Triton AdaLN planned ([#3270](https://github.com/vllm-project/vllm-omni/pull/3270)) | **mindie-sd** + Laser Attention |
| Recommended 8-card I2V (official + CFG) | `--cfg-parallel-size 2 --usp 4` | `--cfg 2 --usp 4` |
| Extra env | — | `MINDIE_SD_FA_TYPE=ascend_laser_attention`, `MULTI_STREAM_MEMORY_REUSE=2` (HSDP) |
| Quantization | FP8 wiring in progress | **MXFP8 validated** ([#3140](https://github.com/vllm-project/vllm-omni/pull/3140)) |

**8× GPU (official I2V + CFG):**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers --omni \
  --use-hsdp --cfg-parallel-size 2 --usp 4 \
  --vae-patch-parallel-size 8 --vae-use-tiling
```

**8× NPU (official I2V + CFG):**

```bash
export MINDIE_SD_FA_TYPE=ascend_laser_attention
export MULTI_STREAM_MEMORY_REUSE=2

vllm serve --omni Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --use-hsdp --usp 4 --cfg 2 \
  --vae-patch-parallel-size 8 --vae-use-tiling
```

### In progress (not in v0.20.0 baseline)

| PR | Status | Topic |
|----|--------|-------|
| [#3127](https://github.com/vllm-project/vllm-omni/pull/3127) | Open | Remove redundant `empty_cache` on NPU |
| [#3145](https://github.com/vllm-project/vllm-omni/pull/3145) | Open | VAE `blend_v` / `blend_h` optimization |
| [#3270](https://github.com/vllm-project/vllm-omni/pull/3270) | WIP | Triton fused AdaLN |
| [#3111](https://github.com/vllm-project/vllm-omni/pull/3111) | Open | VAE tiling interfaces |
| [#2920](https://github.com/vllm-project/vllm-omni/pull/2920) | Open | Online FP8 `quant_config` wiring (GPU) |

### Figures

_No figures for baseline release. Add charts under `assets/` when publishing release posts._
