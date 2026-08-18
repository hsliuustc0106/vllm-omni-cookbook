---
layout: post
title: "Online Quantization in vLLM-Omni — FP8 Without a Quantized Checkpoint"
date: 2026-08-18 12:00:00 +0800
author: hsliuustc0106
summary: >-
  How vLLM-Omni turns any BF16 DiT checkpoint into an FP8-served model at load
  time: the config-to-GEMM pipeline, the two scales with different lifetimes,
  what really happens to VRAM, and where the validated recipes stand.
tags: [Qwen-Image, LTX-2, H100, B200]
category: Feature Deep Dive
feature: quantization
math: true
---

## TL;DR

**Online quantization means vLLM-Omni computes the quantized weights and their
scales while loading the model. Pass one flag against the ordinary BF16
checkpoint — no calibration pass, no preprocessing tool, no second artifact on
disk — and every eligible linear ends up as FP8 (W8A8) in device memory with
roughly half the weight bytes.** Activations are quantized on the fly at every
GEMM, which is what makes the whole scheme calibration-free.

| Metric | BF16 | FP8 | Δ | Setup |
|--------|------|-----|---|-------|
| Peak VRAM, Qwen-Image-2512 | 99,000 MiB | 85,760 MiB (ModelOpt FP8) | **−13.4%** | 2× B200, TP=2, 1024×1024, 20 steps |
| LTX-2.5 on one 80 GB H100 | two-stage 1920×1088 ≈ 114 GB — does not fit | distilled one-stage 960×544, `--quantization fp8` — fits | **fit/no-fit** | 1× H100 80 GB |

The Qwen-Image row was measured with the offline ModelOpt FP8 checkpoint (same
FP8 weight residency that online quantization produces); the LTX-2.5 row is the
genuinely online recipe. Both are cited in [Measured impact](#measured-impact).

## Background

The symptom is capacity. Diffusion transformers for image and video are large
enough that a fully-resident BF16 copy needs most of a datacenter GPU — LTX-2.5
at 1920×1088 peaks around **114 GB**, so the canonical two-stage configuration
does not fit an 80 GB H100 at all
([recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/LTX/LTX-2.5.md)).

Before online quantization, the memory-saving paths all cost you a checkpoint
round-trip: you run an offline quantizer (ModelOpt, msModelSlim, the MXFP4 merge
tools), publish a second multi-gigabyte artifact next to the BF16 original, and
keep the two in sync. If nobody has published a quantized checkpoint for your
model — or for the revision you want to serve — you are the one running the
tooling.

Online quantization removes that step: the BF16 checkpoint you already
downloaded is the input, and the quantized weights exist only in memory. The
[online quantization
guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/quantization/online.md)
lists the validated surface: FP8 W8A8 for the Qwen-Image family and a growing
set of DiTs (LTX-2, Z-Image, FLUX.1/2, HunyuanImage-3.0, HunyuanVideo-1.5,
Cosmos3, MiniMax-H3), Int8 W8A8 on Qwen-Image and Z-Image, and MXFP8/MXFP4 for
Wan2.2 on Ascend NPU and Intel XPU. Wan2.2 FP8 and the omni/TTS stages are
explicitly *not* validated yet.

## How it works

### One flag, three phases

![Figure 1 — the online FP8 pipeline: configure, layerwise load, steady-state serving]({{ site.baseurl }}/assets/figures/online-quantization-fp8/fig1.png)

**Configure.** `quantization="fp8"` (or `--quantization fp8`) goes through
`build_quant_config()` in
[`vllm_omni/quantization/factory.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/quantization/factory.py).
Because `fp8` has no Omni-specific override there, it resolves through upstream
vLLM's registry to `Fp8Config(activation_scheme="dynamic")` — dynamic is the
default, which is where "no calibration" comes from. Scope is controllable two
ways: a plain config is global and reaches every quantization-aware component
(eligible text encoders included), while a dict like
`{"transformer": {"method": "fp8"}, "vae": None}` builds a
`ComponentQuantizationConfig` that routes per layer by longest prefix. Norm and
modulation layers never receive quantization configs at all
(`safe_quant_config`, guarding the shift/scale/gate values they produce), and
`ignored_layers` names sensitive linears that stay in BF16 — Qwen-Image's
`img_mlp` is the canonical example.

**Load.** Weights start on the `meta` device and stream in layer by layer from
the BF16 safetensors. Each linear is materialized in BF16, quantized to FP8
E4M3 with its scale, and then *replaced* — `layer.weight` becomes the qweight
and the BF16 storage is released (`torch.accelerator.empty_cache()` after the
sweep). In the CPU-offload path,
[`_stream_online_quant_weights_to_cpu`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/model_loader/diffusers_loader.py)
moves each layer to host memory the moment it finishes, specifically "to bound
accelerator residency during startup instead of retaining the entire quantized
model until loading ends". So the load-time peak is roughly *accumulated FP8
plus one BF16 layer* — never the full BF16 model.

**Serve.** What remains resident is FP8 weights plus frozen weight scales —
about half the weight bytes. Every GEMM quantizes its incoming activations on
the fly and dispatches to a scaled FP8 kernel: CUTLASS on Hopper/Ada,
FlashInfer on Blackwell, or the optional `quack` fused kernel on datacenter
Blackwell (see [below](#memory-savings-do-not-buy-speed)).

### Two scales, two lifetimes

![Figure 2 — the weight scale is computed once at load and frozen; the activation scale is recomputed every forward pass]({{ site.baseurl }}/assets/figures/online-quantization-fp8/fig2.png)

FP8 W8A8 is really two independent quantization decisions. The **weight scale**
is computed exactly once, at load time, from the weight tensor's own range —
per-tensor $s_w = \max|W| / 448$ (448 is E4M3's largest normal value), or one
scale per 128×128 tile if you pass `weight_block_size`. It is frozen afterward
and deterministic across every request and every denoising step. The
**activation scale** $s_a$ is the "dynamic" in `activation_scheme="dynamic"`:
recomputed from each batch's amax at every GEMM, every step.

Each linear then computes

$$
y \;=\; \underbrace{(a_8 \, W_8^{\!\top})}_{\text{FP8 tensor cores}} \cdot s_a \cdot s_w \;+\; b
$$

with the dequantization folded into the GEMM epilogue — a BF16 weight is never
materialized again.

The asymmetry is not a detail; it is why diffusion models sometimes need
`ignored_layers`. A weight tensor's range is fixed, so one static scale
describes it perfectly. But a denoising trajectory's *activation* ranges shift
across timesteps, and small per-layer errors compound through deep DiT blocks —
Qwen-Image's image-stream MLPs (`img_mlp`) are the documented sensitive spot.
The [FP8 guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/quantization/fp8.md)'s
workflow is exactly this: compare against a BF16 baseline at the same seed, and
skip the layers that regress.

### What actually happens to VRAM

![Figure 3 — schematic VRAM timeline: the BF16 copy is a per-layer transient; only FP8 weights stay resident]({{ site.baseurl }}/assets/figures/online-quantization-fp8/fig3.png)

A common misconception is that online quantization "only saves activation
memory, since the full BF16 weights need to be stored locally." Half of that is
true: the BF16 *checkpoint* must stay on disk, and every boot re-reads and
re-quantizes it. But in device memory the steady state is the other way around —
the parameter *is* the FP8 tensor after `process_weights_after_loading`
([`hsdp_fp8.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/quantization/hsdp_fp8.py)
documents the `layer.weight = qweight.t()` handoff that FSDP2/HSDP sharding then
reconciles), and the loader releases the old BF16 storages explicitly.

So the guaranteed saving is the deterministic one — **2 bytes → 1 byte per
element on every quantized linear** — while activations, attention workspace,
the VAE, and any component without quantizable layers are untouched. That is
why measured *peak* VRAM drops by less than the weight halving: in the
Qwen-Image-2512 recipe below, weights halve on quantized layers but the peak
moves only 99 → 86 GiB, because everything else in the peak stays.

### Memory savings do not buy speed

FP8 halves weight bytes, but tensor-core throughput at small video-DiT GEMM
shapes does not automatically follow. On Blackwell the FP8 path runs through
FlashInfer, which applies bias as a *separate* kernel after the GEMM — the
[FP8 guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/quantization/fp8.md)
notes HunyuanVideo-1.5 FP8 running slower than BF16 until the optional `quack`
package fuses `alpha * (A @ B) + bias` into one CuteDSL tcgen05 GEMM
(datacenter Blackwell only; auto-enabled once installed; compile cache at
`~/.cache/vllm-omni/quack`). Treat memory as the primary win and measure speed
for your model.

## The code path, file by file

- [`vllm_omni/quantization/factory.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/quantization/factory.py) —
  `build_quant_config()` turns the flag (or dict) into a `QuantizationConfig`;
  Omni-specific builders exist only for int8/bitsandbytes/mxfp8/mxfp4/INC —
  `fp8` deliberately falls through to upstream vLLM. `resolve_quant_config_from_disk()`
  reconciles your flag against `quantization_config` in the checkpoint's
  config.json: auto-detects pre-quantized checkpoints, switches to offline mode
  if the disk marks serialized weights (never double-quantizes), and raises on a
  genuine method mismatch.
- [`vllm_omni/quantization/component_config.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/quantization/component_config.py) —
  `ComponentQuantizationConfig` longest-prefix routing; `safe_quant_config`
  keeps norm/modulation layers out of every non-pre-quantized scheme.
- [`vllm_omni/diffusion/model_loader/diffusers_loader.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/model_loader/diffusers_loader.py) —
  the layerwise streaming loader: meta-device skeleton, per-layer
  materialize→quantize→offload, `empty_cache()` after the sweep, and the
  load-on-device-then-offload ordering (`offload_after_quant`) for CPU-offload
  deployments.
- [`vllm_omni/diffusion/quantization/hsdp_fp8.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/quantization/hsdp_fp8.py) —
  makes online-FP8 linears FSDP2-shardable: keeps the parameter as contiguous
  row-major storage and injects the CUTLASS-required `.t()` view at the GEMM
  call site instead.
- [`vllm_omni/diffusion/models/diffusers_adapter/quantization_utils.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/diffusers_adapter/quantization_utils.py) —
  courtesy-conversion of an online fp8/int8 config to TorchAO dynamic
  quantization for the Diffusers backend (transformer/transformer_2 only;
  serialized checkpoints, static schemes, `weight_block_size`, and
  `ignored_layers` are not mapped).

## Measured impact

All numbers are from the upstream recipes — this post adds no new measurements.

**LTX-2.5, one H100 80 GB — genuinely online FP8**
([recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/LTX/LTX-2.5.md)):
the BF16 `Lightricks/LTX-2.5-Diffusers` checkpoint served with
`--quantization fp8` on the distilled one-stage pipeline at 960×544 is the
documented way to run LTX-2.5 on that GPU; the BF16 two-stage configuration at
1920×1088 needs ≈114 GB and does not fit. This is the load path this post
describes, end to end.

**Qwen-Image-2512, 2× B200 — FP8 residency, offline checkpoint**
([recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen-Image.md),
100 sequential requests, concurrency 1, 1024×1024, 20 denoising steps, TP=2):

| Config | Mean | Peak VRAM |
|---|---:|---:|
| BF16 | 2.696 s | 99,000 MiB |
| ModelOpt FP8 (CUTLASS) | 2.663 s | 85,760 MiB |
| ModelOpt mixed FP8/NVFP4 (CUTLASS) | 2.584 s | 84,412 MiB |

The caveat that matters: these rows were measured with **pre-quantized ModelOpt
checkpoints**, not the online flag — the recipe documents the offline artifact
because that is what was benchmarked. The steady-state residency is the same
FP8 weights-plus-scales that online quantization produces (the online path just
builds them at load), so the memory column is the honest reference for what FP8
buys on this model; treat the latency column as ModelOpt-specific until an
online-flag run is recorded. If you reproduce it, the cookbook wants the ledger
entry.

## How to use it

Global, from the BF16 checkpoint:

```bash
vllm serve Lightricks/LTX-2.5-Diffusers \
  --omni \
  --model-class-name LTX2DistilledOneStagePipeline \
  --quantization fp8
```

```python
from vllm_omni import Omni

omni = Omni(model="Qwen/Qwen-Image", quantization="fp8")
# or keep sensitive layers in BF16:
omni = Omni(model="Qwen/Qwen-Image",
            quantization_config={"method": "fp8",
                                 "ignored_layers": ["img_mlp"]})
```

Scope to one component, or mix methods:

```python
from vllm_omni.quantization import build_quant_config

config = build_quant_config({
    "transformer": {"method": "fp8"},
    "text_encoder": {"method": "fp8"},
    "vae": None,
})
```

Quality workflow, per the guides: compare against a BF16 baseline with the same
seed and generation parameters before trusting a new model; document any
required `ignored_layers`. FP8 also composes with cache acceleration
(`cache_backend="tea_cache"`). On datacenter Blackwell, `pip install
vllm-omni[quack]` is the one-line speed recovery; nothing else changes.

## Limitations & follow-ups

- **Validation surface.** Wan2.2 FP8 and the omni/TTS stages (Qwen3-Omni
  thinker, Qwen3-TTS) are not validated for online quantization — the omni
  path with evidence is the ModelOpt pre-quantized checkpoint. BAGEL and
  GLM-Image need explicit per-stage routing before enabling. On Ascend NPU
  there is no FP8; the validated online paths there are Int8 and MXFP8/MXFP4.
- **Startup cost.** The BF16 checkpoint stays on disk and re-quantizes on every
  boot — the price of skipping the offline step. If disk footprint or load time
  dominates your constraints, a serialized quantized checkpoint (or a published
  ModelOpt one) is the better trade; `resolve_quant_config_from_disk` will pick
  it up automatically.
- **Offloader interaction.** Online quantization skips the loader-owned
  host-weight-plan fast path (`host_weight_plan.py` returns "online quantization
  requires the ordinary loader"); layerwise offload still works through the
  ordinary loader.
- **Diffusers backend.** The TorchAO courtesy-conversion covers only
  transformer components with the dynamic scheme; everything else needs a
  Diffusers-native `quantization_config` via `diffusers_load_kwargs`.
- **Quality is per-model.** The method is validated per family with specific
  `ignored_layers` guidance; there is no universal "FP8 is free" claim. Diffusion
  trajectories shift activation ranges across timesteps, which is exactly where
  dynamic scaling helps and where sensitive layers may still need to stay BF16.

## References

- [Online quantization guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/quantization/online.md) — hardware/method support matrix
- [FP8 guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/quantization/fp8.md) — validated models, `ignored_layers`, quack
- [LTX-2.5 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/LTX/LTX-2.5.md) — the online-FP8 H100 configuration
- [Qwen-Image recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen-Image.md) — BF16 vs ModelOpt FP8 peak-VRAM table
- [Quantization overview](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/quantization/overview.md) — how the cookbook validates quantized outputs
- Figure sources: [`fig1.svg`](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/blog/assets/figures/online-quantization-fp8/fig1.svg) · [`fig2.svg`](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/blog/assets/figures/online-quantization-fp8/fig2.svg) · [`fig3.svg`](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/blog/assets/figures/online-quantization-fp8/fig3.svg)
