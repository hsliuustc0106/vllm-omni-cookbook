---
layout: post
title: "Understanding PR #6279 — Online FP8 with DLO AllGather"
date: 2026-08-19 12:00:00 +0800
author: hsliuustc0106
summary: >-
  PR #6279 lets per-tensor online FP8 run through DLO AllGather: finalized FP8
  weights and scales are sharded and reconstructed with their physical layout,
  while the remaining startup host-memory trade-off stays explicit.
tags: [MiniMax-H3, FP8, H100, DLO]
category: PR Analysis
feature: offloader
---

## TL;DR

**PR #6279 makes per-tensor online FP8 compatible with DLO's AllGather path.**
The ordinary loader first turns the BF16 checkpoint into finalized FP8 weights
and scales; DLO then shards those runtime tensors, transfers FP8 shards, and
reconstructs each layer on the device. The result is lower steady-state host and
device memory than native BF16, but startup still has an ordinary-loader peak
because direct checkpoint mmap cannot quantize an online method.

| Precision | Layout | Wave latency | Throughput / device | Peak GPU / device | Host PSS peak |
|---|---|---:|---:|---:|---:|
| Online FP8 | DP2/SP2 | 737.54 s | 0.000678 req/s | 22.11 GiB | 220.94 GiB |
| Native BF16 | DP2/SP2 | 669.98 s | 0.000746 req/s | 24.04 GiB | 362.49 GiB |
| Online FP8 | DP4/SP1 | 1227.72 s | 0.000815 req/s | 21.79 GiB | 221.60 GiB |
| Native BF16 | DP4/SP1 | 1194.88 s | 0.000837 req/s | 23.53 GiB | 260.77 GiB |

These are single-run MiniMax-H3 Ref2VA measurements on four H100 GPUs; the
complete table, RSS values, fidelity metrics, commands, and caveats are in the
[PR validation comment](https://github.com/vllm-project/vllm-omni/pull/6279#issuecomment-5328282759).

## Background

Diffusion transformers are large enough that keeping every layer resident on
every GPU can be the limiting resource. DLO addresses that capacity problem by
keeping weight storage on the host and moving only the current layer into one of
two reusable device buffers. With AllGather enabled, each rank keeps a shard of
the host weights and reconstructs a full layer collectively before the block
runs.

Online FP8 is a natural companion: a BF16 checkpoint remains the source of
truth, but eligible linear weights are quantized at load time and the serving
model retains FP8 weights plus scales. Before #6279, the DLO AllGather gate
rejected all online quantization methods. The safe direct-mmap path cannot simply
map a BF16 tensor and defer quantization, because the final runtime representation
may change dtype, shape, packing, and stride.

The practical symptom was a hard incompatibility: users had to choose between
online FP8 and the DLO AllGather topology, even when the finalized per-tensor
FP8 representation was already compatible with DLO's reconstruction contract.

## What the PR does

The new path is deliberately narrow:

```text
BF16 checkpoint on disk
        │
        ▼
ordinary loader + online quantizer
        │
        ▼
FP8 weight + scale
        │
        ▼
1 / DLO-group-size host shard per rank
        │  H2D copy stream
        ▼
AllGather communication stream
        │
        ▼
full FP8 layer in one of two GPU slots
```

The BF16 checkpoint is the input to quantization; it is not the payload moved by
the DLO AllGather. The associated scale travels with the FP8 runtime tensor in
the reconstruction metadata.

<iframe
  src="{{ site.baseurl }}/assets/figures/pr-6279-dlo-online-fp8/dlo-online-fp8-allgather.html"
  title="Interactive DLO AllGather online FP8 visualization"
  loading="lazy"
  style="display:block;width:100%;height:720px;border:1px solid #d0d7de;border-radius:8px;margin:16px 0;">
</iframe>

The interactive walkthrough switches between DP2/SP2 and DP4/SP1 and steps
through source loading, quantization, host sharding, H2D, AllGather,
double-buffered compute, release, and reuse. It also distinguishes the
persistent BF16 file from transient BF16 staging and the retained FP8 DLO
payload.

## Key changes

### Allow one validated online method

[`diffusers_loader.py`](https://github.com/vllm-project/vllm-omni/blob/284e05c88b7b46be9fae6d822bf22075840cbfbb/vllm_omni/diffusion/model_loader/diffusers_loader.py)
now checks the actual online quantization method. Only
`Fp8PerTensorOnlineLinearMethod` is allowed through DLO AllGather; other online
methods remain fail-closed until their runtime layouts are validated. The
ordinary loader still runs first, so quantization and scale generation finish
before DLO starts sharding.

This preserves a useful safety boundary: allowing one known layout is different
from assuming that every online quantizer produces a DLO-compatible tensor.

### Preserve the physical FP8 layout

Online Cutlass FP8 weights can be transposed, non-contiguous views. Flattening
them in logical order would silently change the layout expected by the scaled
matrix-multiply kernel. DLO therefore records each runtime tensor's dtype, shape,
stride, and physical offset, packs physical storage order when needed, and
reconstructs the recorded view after AllGather.

The relevant backend logic is in
[`distributed_layerwise_backend.py`](https://github.com/vllm-project/vllm-omni/blob/284e05c88b7b46be9fae6d822bf22075840cbfbb/vllm_omni/diffusion/offloader/distributed_layerwise_backend.py).
The regression test
[`test_allgather_reconstructs_online_fp8_weight_and_scale`](https://github.com/vllm-project/vllm-omni/blob/284e05c88b7b46be9fae6d822bf22075840cbfbb/tests/diffusion/offloader/test_distributed_layerwise_backend.py)
checks the finalized FP8 weight, scale, dtype, shape, and transposed stride.

### Keep the collective semantics unchanged

The PR does not add a request-time collective. AllGather remains a
request-independent weight operation: every participating rank must enter the
same wave in the same block order. The two device buffers still alternate, so
the compute stream can run the current FP8 block while the copy and communication
streams prepare the next one.

## Measured impact

The validation workload was MiniMax-H3 Ref2VA on four H100 GPUs with vLLM
0.27.0, CUDA 12.9, a 1344×768 output, 24 FPS, five-second video, seed 0, and
50 denoising steps. The current PR head was compared with a frozen native-BF16
baseline; DP2/SP2 and DP4/SP1 used a concurrent wave sized to the DP group.
Throughput per device is completed requests divided by wave wall time and four
GPUs.

| Precision | Configuration | Wave latency | Throughput / device | Peak GPU / device | Host RSS peak | Host PSS peak | SSIM / PSNR |
|---|---|---:|---:|---:|---:|---:|---:|
| Online FP8 | DP2/SP2 | 737.54 s | 0.000678 req/s | 22.11 GiB | 239.45 GiB | 220.94 GiB | 0.975421–0.975424 / 38.8108–38.8121 dB |
| Native BF16 | DP2/SP2 | 669.98 s | 0.000746 req/s | 24.04 GiB | 377.33 GiB | 362.49 GiB | 0.980584–0.980589 / 42.6535–42.6550 dB |
| Online FP8 | DP4/SP1 | 1227.72 s | 0.000815 req/s | 21.79 GiB | 285.73 GiB | 221.60 GiB | 0.975421–0.975424 / 38.8108–38.8121 dB |
| Native BF16 | DP4/SP1 | 1194.88 s | 0.000837 req/s | 23.53 GiB | 293.39 GiB | 260.77 GiB | 0.980584–0.980589 / 42.6535–42.6550 dB |

Three conclusions matter more than any one row:

- Native BF16 was faster in this run: by 9.2% for DP2/SP2 and 2.7% for
  DP4/SP1. DLO communication and the online-loader path mean that halving
  weight bytes does not automatically make every diffusion GEMM faster.
- FP8 used less host PSS than BF16: 39.0% lower for DP2/SP2 and 15.0% lower
  for DP4/SP1. The DP2 gap is larger because the DLO group retains two DiT
  copies in that topology, so the BF16 payload is replicated more heavily.
- BF16 had higher fidelity in this paired comparison, by approximately 0.005
  SSIM and 3.84 dB PSNR. Online FP8 is runtime-compatible, but this result is
  not a claim of numerical equality or production-quality parity.

Host RSS is the peak sum of the API server, four diffusion workers, and the
resource tracker sampled at 1 Hz. RSS counts shared pages once per mapping;
PSS is the better estimate of physical host memory. All outputs passed video and
audio metadata validation: 1344×768, 24 FPS, 124 frames, and 32-kHz stereo AAC.

### What the cache follow-up changes

The short memory-only follow-up below is an exploratory Phase-I normalized-FP8
cache measurement, not a second benchmark of #6279. It isolates why the cache
is still needed even after #6279: a cache hit can skip the ordinary online
quantization materialization for the cached transformer, but a partial cache
does not guarantee lower whole-process PSS.

| Path | DP2/SP2 request PSS | DP4/SP1 request PSS |
|---|---:|---:|
| Ordinary online FP8 | 188.71 GiB | 145.07 GiB |
| Normalized FP8 cache hit | 213.33 GiB | 171.85 GiB |
| Native BF16 | 318.48 GiB | 222.93 GiB |

In this prototype, only the transformer was normalized into the FP8 cache; the
text encoder and other components remained BF16. The cache's file-backed pages
also appear in RSS/PSS. The honest conclusion is therefore:

1. #6279 lowers the finalized runtime payload from BF16 to FP8 and proves the
   AllGather reconstruction path.
2. A cache hit can remove the full BF16/online-quantization startup path for the
   cached component.
3. An end-to-end host-memory win requires consistent cache or quantization
   coverage for the other large components and apples-to-apples PSS accounting.

The full memory decomposition and raw values are recorded in the
[follow-up PR comment](https://github.com/vllm-project/vllm-omni/pull/6279#issuecomment-5337093776).

## How to use it

The default DLO AllGather path now accepts per-tensor online FP8:

```bash
MODEL=/path/to/MiniMax-H3/Ref2VA

# Four GPUs, DP2/SP2: two DLO groups, two ranks per group.
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$MODEL" \
  --omni \
  --task-type ref2va \
  --num-gpus 4 \
  --usp 2 \
  --quantization fp8 \
  --enable-distributed-layerwise-offload
```

For DP4/SP1 on the same four GPUs, use `--usp 1`. DLO's default AllGather
setting is enabled by the distributed-layerwise-offload configuration. The
MiniMax-H3 recipe documents the component scope and the ordinary-loader startup
trade-off in more detail:

[MiniMax-H3 online FP8 recipe](https://github.com/vllm-project/vllm-omni/blob/284e05c88b7b46be9fae6d822bf22075840cbfbb/recipes/MiniMaxAI/MiniMax-H3.md)

If independent replicas cannot obey the synchronized request-wave contract,
use `--dlo-no-use-allgather`. That keeps the online FP8 runtime path but uses
rank-local host tensors rather than DLO's sharded collective path.

## Limitations & follow-ups

- **Startup memory remains the main gap.** Direct checkpoint mmap does not
  quantize BF16 tensors online. Each rank temporarily runs the ordinary loader
  and materializes the finalized FP8 model before retaining its DLO shard.
- **The allowlist is intentionally narrow.** Other online quantizers, including
  block/group layouts not covered by the validated method, remain fail-closed.
- **AllGather requires synchronized waves.** Ranks must use the same denoising
  step count and enter the weight collective in lockstep. Choose no-AllGather
  for independently scheduled replicas.
- **DLO AllGather is not a TP group.** TP-aware and HSDP layouts have separate
  compatibility boundaries; HSDP plus DLO AllGather is rejected to avoid
  double-sharding.
- **Phase-I cache work is follow-up scope.** The runtime-cache compatibility
  contract is tracked in [#6231](https://github.com/vllm-project/vllm-omni/issues/6231),
  including transformed layouts, cache misses, source fingerprints, and
  cross-DP/SP reuse.
- **Quality needs a broader evaluation.** The measurements above are one
  five-second, 50-step request matrix. They establish systems behavior and
  fidelity-to-BF16 for this workload, not general prompt alignment or human
  quality parity.

## References

- [PR #6279 — Support online FP8 with DLO AllGather](https://github.com/vllm-project/vllm-omni/pull/6279)
- [PR #6279 benchmark comment](https://github.com/vllm-project/vllm-omni/pull/6279#issuecomment-5328282759)
- [PR #6279 host-memory follow-up](https://github.com/vllm-project/vllm-omni/pull/6279#issuecomment-5337093776)
- [DLO user guide](https://github.com/vllm-project/vllm-omni/blob/284e05c88b7b46be9fae6d822bf22075840cbfbb/docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md)
- [FP8 quantization guide](https://github.com/vllm-project/vllm-omni/blob/284e05c88b7b46be9fae6d822bf22075840cbfbb/docs/user_guide/quantization/fp8.md)
- [RFC #6231 — DLO runtime-cache compatibility](https://github.com/vllm-project/vllm-omni/issues/6231)
- [Online quantization deep dive](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/blog/_posts/2026-08-18-online-quantization-fp8.md)
