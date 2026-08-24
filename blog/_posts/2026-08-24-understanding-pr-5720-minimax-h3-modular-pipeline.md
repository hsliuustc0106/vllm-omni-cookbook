---
layout: post
title: "Serving MiniMax-H3 in vLLM-Omni (1): modular pipeline — two DiTs, one shared stack"
date: 2026-08-24 18:00:00 +0800
author: hsliuustc0106
summary: >-
  How MiniMax-H3 routes two task-specific DiTs around one shared component
  stack, plus a profiled 4×H200 E2E decomposition: 85.85 s warm at 50 sigma
  points, including 80.62 s of denoising.
tags: [MiniMax-H3, H200]
category: PR Analysis
feature: pipeline
lang: en
pair: /zh/2026-08-24-understanding-pr-5720-minimax-h3-modular-pipeline/
usage:
  - label: "Combined"
    blurb: "all three request tasks"
    title: "vllm serve · both MiniMax-H3 partitions"
    code: |
      export MODEL=MiniMaxAI/MiniMax-H3
      export PORT=8091

      CUDA_VISIBLE_DEVICES=0,1,2,3 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
      vllm serve "${MODEL}" \
        --omni \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --trust-remote-code \
        --num-gpus 4 \
        --usp 4 \
        --ring 1 \
        --vae-patch-parallel-size 4 \
        --vae-parallel-mode tile \
        --vae-use-tiling
    note: >-
      This is the pinned recipe's high-memory four-GPU combined profile, not a
      generic four-GPU promise. Both DiTs remain resident; verify capacity for
      the selected hardware before launch.
  - label: "FL2VA only"
    blurb: "T2VA + frame-conditioned FL2VA"
    title: "vllm serve · select the FL2VA partition at startup"
    code: |
      export MODEL=MiniMaxAI/MiniMax-H3
      export PORT=8091

      CUDA_VISIBLE_DEVICES=0,1,2,3 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
      vllm serve "${MODEL}" \
        --omni \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --trust-remote-code \
        --task-type fl2va \
        --num-gpus 4 \
        --usp 4 \
        --ring 1 \
        --text-encoder-tp-size 4 \
        --vae-patch-parallel-size 4 \
        --vae-parallel-mode tile \
        --vae-use-tiling \
        --diffusion-attention-backend CUDNN_ATTN
    note: >-
      This is the measured 4×H200 low-latency topology. It downloads and loads
      only FL2VA; task=t2va and task=fl2va are accepted, while task=ref2va is
      rejected because that DiT is absent.
  - label: "Ref2VA only"
    blurb: "reference-conditioned generation"
    title: "vllm serve · select the Ref2VA partition at startup"
    code: |
      export MODEL=MiniMaxAI/MiniMax-H3
      export PORT=8091

      CUDA_VISIBLE_DEVICES=0,1,2,3 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
      vllm serve "${MODEL}" \
        --omni \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --trust-remote-code \
        --task-type ref2va \
        --num-gpus 4 \
        --usp 4 \
        --ring 1 \
        --vae-patch-parallel-size 4 \
        --vae-parallel-mode tile \
        --vae-use-tiling
    note: >-
      Downloads and loads only Ref2VA. Use task=ref2va and a supported visual
      reference combination; T2VA and FL2VA are not available in this server.
decisions:
  - when: "One endpoint must accept every H3 task"
    pick: "Combined startup"
    why: "It keeps both task-specific DiTs available and shares the tokenizer, encoder, and VAEs once."
  - when: "Only text or keyframes drive generation"
    pick: "FL2VA-only startup"
    why: "T2VA and FL2VA use the same partition; skipping Ref2VA avoids downloading and loading an unused DiT."
  - when: "Every request is reference-conditioned"
    pick: "Ref2VA-only startup"
    why: "It narrows the server contract to Ref2VA and rejects accidental T2VA/FL2VA routing."
  - when: "You are profiling where time goes"
    pick: "Pin one partition and one task"
    why: "A fixed workload makes startup, prompt, denoise, decode, CPU encoding, and transport boundaries reviewable."
  - when: "Capacity is uncertain"
    pick: "Budget storage, host RAM, and HBM separately"
    why: "A smaller download is not automatically the same amount of GPU or host-memory savings."
---

## TL;DR {#tldr}

**[PR #5720](https://github.com/vllm-project/vllm-omni/pull/5720)
turns MiniMax-H3 into one workshop with two specialist engines: the FL2VA DiT
serves text-to-video+audio (T2VA) and frame-conditioned FL2VA, the Ref2VA DiT
serves reference-conditioned requests, and both reuse one tokenizer,
processor, Qwen3-VL encoder, video VAE, and audio VAE stack.** A combined
server loads both DiTs and selects one with `extra_params.task`; a
task-selected server loads only one partition. The analogy stops at capacity:
sharing components once does not mean both DiTs are small, and storage, host
RAM, and GPU HBM remain separate budgets.

This post explains the shipped architecture at
[`072bfc02`](https://github.com/vllm-project/vllm-omni/commit/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65).
On 4×H200, a profiled FL2VA-only 1344×768 T2VA workload with 124 frames and
50 sigma points / 49 denoiser evaluations measured **85.845 ± 0.161 s** warm
client receipt time (mean ± sample SD, n=3): **80.616 s denoise**, 1.859 s VAE
decode, and 3.370 s across prompt encoding, engine/IPC residual, CPU MP4
encoding, and HTTP residual. These values describe that fully pinned profiled
condition—not every H3 request. The unprofiled controls were nonstationary, so
profiler overhead remains unresolved and is disclosed below.

| Baseline fact | Shipped behavior |
|---|---|
| Checkpoint partitions | FL2VA and Ref2VA |
| Request tasks | `t2va`, `fl2va`, `ref2va` |
| Shared components | tokenizer, processor, retained Qwen3-VL encoder, video VAE, audio VAE |
| Ordinary default schedule | 50 sigma points → 49 denoiser evaluations |
| Checkpoint storage | about 135 GiB per partition; roughly 270 GiB for both, per the pinned recipe |
| Profiled 4×H200 warm E2E | 85.845 ± 0.161 s; 80.616 s denoise (1344×768, 124 frames, 50 points / 49 evaluations) |

## Background {#background}

The user-visible problem was not that MiniMax-H3 lacked two task families; it
was that they looked like two complete machines when most of their machinery
was the same. Imagine two delivery routes that need different engines but use
the same loading dock, maps, and packaging line: duplicating the whole depot
would make operating both routes needlessly awkward.

[PR #5691](https://github.com/vllm-project/vllm-omni/pull/5691) introduced the
original MiniMax-H3 diffusion pipeline. The checkpoint publishes FL2VA and
Ref2VA partitions, but their task-specific difference is the diffusion
transformer (DiT)—the large denoising network—while tokenization, multimodal
prompt encoding, and video/audio compression are common. PR #5720 made that
structure explicit: a repository-root model can build both DiTs around one
shared component stack, while `--task-type` keeps single-partition deployment
available.

Two later merged changes define the baseline described here:
[PR #5752](https://github.com/vllm-project/vllm-omni/pull/5752) aligned the
official FL2VA/Ref2VA input matrix and validation boundaries, and
[PR #5824](https://github.com/vllm-project/vllm-omni/pull/5824) made the
Qwen3-VL encoder loader fail closed when a parameter or fused source shard is
missing. The result is not just a diagram: startup rejects an incomplete
checkpoint, and requests reject task/input combinations that the loaded
partition cannot serve.

## What PR #5720 changed {#key-changes}

PR #5720 is the switchboard change in this four-PR baseline: #5691 built the
original workshop, #5720 made its common tools shareable and installed the
task switch, #5752 tightened the input rulebook, and #5824 later sealed the
encoder inventory. That division matters because the series title centers the
modular pipeline while this post describes the shipped system around it.

| Anchor | Its distinct contribution to this baseline |
|---|---|
| [#5691](https://github.com/vllm-project/vllm-omni/pull/5691) | Added the original MiniMax-H3 diffusion model, packed joint video/audio denoising, encoder, VAEs, registry, and serving integration. |
| **[#5720](https://github.com/vllm-project/vllm-omni/pull/5720)** | Propagated model-defined `task_type` into the diffusion stage; added combined versus task-selected download/construction; loaded two task-specific DiTs around shared components; routed each request to the active DiT. |
| [#5752](https://github.com/vllm-project/vllm-omni/pull/5752) | Aligned the official FL2VA/Ref2VA input and validation matrix after the modular topology existed. |
| [#5824](https://github.com/vllm-project/vllm-omni/pull/5824) | Made current encoder weight loading fail closed for missing parameters or fused source shards. |

At the pinned source level, #5720's responsibilities now appear in three
places:

- [`MINIMAX_H3_DOWNLOAD_PATTERNS` and `_minimax_h3_partition_for_task`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py#L145-L177)
  decide whether snapshot download needs both partitions or one.
- [Pipeline construction](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py#L660-L825)
  registers one or two transformer weight sources while constructing the
  tokenizer, processor, encoder, and VAEs once.
- [`_transformer_for_task` and `_resolve_task`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py#L856-L891)
  enforce request-to-loaded-partition routing.

## Mental model: one workshop, two engines {#mental-model}

The simplest useful model is a workshop with a shared front desk and finishing
line, plus two engines in the middle. The front desk understands the request;
one engine performs repeated denoising; the finishing line turns latents back
into synchronized video and audio.

![MiniMax-H3 modular architecture: shared encoding and VAE components around FL2VA and Ref2VA DiTs]({{ site.baseurl }}/assets/figures/minimax-h3-modular-pipeline/fig1-architecture.svg)

The mapping is precise:

- **Front desk:** tokenizer, Qwen3-VL processor, and the retained Qwen3-VL
  text/vision encoder build the context seen by the DiT.
- **Two engines:** `transformer` is the FL2VA DiT;
  `transformers_ref` is the Ref2VA DiT in combined mode.
- **Finishing line:** the shared video VAE and audio VAE decode the final
  latents; the serving layer packages them as H.264 video plus synchronized
  stereo audio.

Where the analogy breaks: reference images, videos, and audio can require VAE
encoding *before* denoising, and distributed placement can shard or move
components. Those details matter to memory and latency, but the later
parallelism/offload posts own them; this post keeps the logical ownership
clear.

## Architecture and dataflow {#architecture-dataflow}

Startup decides which engines exist; the request then passes through shared
preparation, one selected engine, and shared decoding. It is an assembly line
with a switch in the middle, not two independent lines from raw input to MP4.

The startup resolver in
[`pipeline_minimax_h3.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py#L162-L177)
maps the model-level selection to a partition:

```python
if task in {"auto", "combined"}:
    return "combined"
if task in {"t2va", "fl2va"}:
    return "fl2va"
if task == "ref2va":
    return "ref2va"
```

That choice changes downloads and construction, not just a label:

| Startup mode | DiTs constructed | Shared-component source | Accepted request tasks |
|---|---|---|---|
| Combined (repository model, no `--task-type`) | FL2VA + Ref2VA | FL2VA; Ref2VA contributes its model index and transformer | `t2va`, `fl2va`, `ref2va` |
| `--task-type fl2va` (or direct `FL2VA` path) | FL2VA only | FL2VA | `t2va`, `fl2va` |
| `--task-type ref2va` (or direct `Ref2VA` path) | Ref2VA only | Ref2VA | `ref2va` |

The [pinned recipe](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/recipes/MiniMaxAI/MiniMax-H3.md#start-a-server)
puts each partition at about 134 GiB of BF16 safetensors (about 135 GiB on
disk). Combined serving therefore needs roughly 270 GiB of checkpoint storage;
task-selected serving downloads only the selected partition. That is a
**storage** statement. Host-loading peaks and GPU residency depend on loader,
topology, and offload settings and must be measured separately.

## How a request chooses its DiT {#task-routing}

There are two switches with different jobs: `--task-type` is the circuit
breaker installed when the server starts, while `extra_params.task` is the
button pressed for one request. A request cannot select a DiT that startup did
not load.

| Request task | Selected DiT | Required input shape |
|---|---|---|
| `t2va` | FL2VA | text only |
| `fl2va` | FL2VA | one or two image keyframes, with the shipped first/last-frame contract |
| `ref2va` | Ref2VA | a supported visual-reference combination; audio-only is rejected |

The current resolver also has an implicit fallback: a Ref2VA-only server
defaults to `ref2va`; otherwise video/audio input implies `ref2va`, image input
implies `fl2va`, and text-only input implies `t2va`. Explicit `task` values are
better operational practice because they make logs, rejected requests, and
profiling manifests unambiguous.

For example, the same combined endpoint can route a text-only request to the
FL2VA partition:

```bash
curl -sS -X POST "http://127.0.0.1:${PORT}/v1/videos/sync" \
  -F 'prompt=A quiet moonlit harbor with synchronized waves and wind.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":8.7,"audio_flow_shift":3.0}' \
  -o t2va.mp4
```

For FL2VA or Ref2VA, use the exact multipart reference fields and limits in
the [current recipe](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/recipes/MiniMaxAI/MiniMax-H3.md#http-api-examples),
not historical PR test commands.

## Pipeline stages {#pipeline-stages}

The request crosses three model stages—encode, diffuse, decode—but “encode” is
not one magic timing bucket. Like a kitchen ticket, the label only helps when
we say whether ingredient preparation and final packaging are written on the
same line.

| Stage | What happens | Frequency | Source boundary |
|---|---|---|---|
| Validate + prepare references | task/input checks, video trimming, shape and duration resolution | once per request | serving + pipeline preparation |
| Prompt/context encode | tokenizer/processor + retained Qwen3-VL encoder | once per request | `encode_prompt` |
| Reference encode | video/audio VAE turns conditions into latent rows when required | once per reference set | separate profiler targets, not automatically part of `encode_prompt` |
| Diffuse | selected DiT updates packed video/audio rows across adjacent sigma points | once per denoiser evaluation | `diffuse` |
| Decode | shared video VAE and audio VAE reconstruct final media | once per output | `decode` |
| Package response | host CPU encodes/muxes MP4 and HTTP transfers the body | once per response | outside the three model-stage totals |

The core implementation lives in four source areas:

- [`pipeline_minimax_h3.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py#L660-L825)
  constructs the selected DiTs and shared components; its
  [`forward`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py#L1731-L1998)
  validates, encodes, diffuses, and decodes.
- [`packed_sequence.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/packed_sequence.py)
  lays text, generated video, generated audio, and task-specific condition rows
  into the joint sequence consumed by the DiT.
- [`time_request.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/time_request.py#L33-L70)
  builds the ordinary uniform sigma points.
- [`denoise_loop.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/denoise_loop.py#L139-L203)
  executes one DiT evaluation for every interval between adjacent points.

That last distinction fixes a common terminology trap: ordinary
`num_inference_steps=N` creates **N sigma points and N−1 denoiser
evaluations**. The default `50` therefore means 49 evaluations. A request value
of `4` means four points and three evaluations; it is not the later Turbo
four-NFE contract, which requests five points. Turbo itself is outside this
baseline article.

## Where time goes—and how to measure it {#measured-profile}

A timing chart is a receipt for one order, not a price list for every meal.
Resolution, frame count, references, sigma count, topology, backend, compile
state, and warmup can all move the boundaries, so a percentage without that
context is not a reusable fact.

The experiment specified in
[`cookbook#39`](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/39)
was run through `gpu run` on four H200 GPUs (the cluster's raw `nvidia-smi`
label is `NVIDIA L20X`). It used FL2VA-only T2VA, Ulysses 4 / Ring 1 / DiT
TP1, text-encoder TP4, VAE patch-parallel 4/tile, BF16, CUDNN attention, and
regional `torch.compile`. The request was 1344×768, requested 5.0 s, 24 FPS,
124 aligned frames, seed 1101, and the ordinary 50 sigma points / 49 denoiser
evaluations. One cold request was excluded; the table reports three warm
profiled requests.

![Measured MiniMax-H3 four-H200 startup and complete-response timing decomposition]({{ site.baseurl }}/assets/figures/minimax-h3-modular-pipeline/fig3-e2e-measured.svg)

| Environment field | Pinned value |
|---|---|
| Source / model | vLLM-Omni `072bfc02`; MiniMax-H3 snapshot `42ed227e` |
| Installed runtime | vLLM 0.27.0; vLLM-Omni package `0.27.0rc2.dev80+g20e3655f5`; PyTorch 2.13.0+cu129 |
| Hardware | 4×H200, driver 570.133.20; raw cluster label `NVIDIA L20X`, 143,771 MiB/device |
| Topology | U4/Ring1/DiT-TP1, text-encoder TP4, VAE-PP4 tile |
| Precision / backend | BF16, no quantization; CUDNN attention; regional compile, not eager |
| Workload | T2VA, 1344×768, 124 frames / 24 FPS, 5.0 s requested, 50 points / 49 evaluations |
| Samples | one cold + three warm profiled; ambient/warm page cache; 1 Hz resource sampling |

Profiled process-start→first-`/health` startup was **149.880 s** (one
illustrative observation, not a repeated startup benchmark): 50.741 s imports/
CLI/config, 37.000 s worker+NCCL setup, 60.000 s model initialization/load/
placement, and 2.138 s orchestrator/API readiness. The cold/compile request was
114.686 s. Warm complete-client receipt was **85.845 ± 0.161 s** (mean ±
sample SD, n=3):

| Warm receipt bucket | Mean ± sample SD | Share of warm mean | Boundary |
|---|---:|---:|---|
| Prompt / text encoder | 0.056 ± 0.000 s | 0.06% | direct synchronized span |
| Denoise | 80.616 ± 0.074 s | 93.91% | direct synchronized span, 49 evaluations |
| Video + audio VAE decode | 1.859 ± 0.004 s | 2.17% | direct synchronized span |
| Engine / IPC / output | 1.956 ± 0.115 s | 2.28% | signed residual inside server inference |
| CPU MP4 encode + mux | 1.326 ± 0.052 s | 1.54% | direct server span |
| HTTP transport | 0.033 ± 0.017 s | 0.04% | client-total minus server inference |
| **Complete client receipt** | **85.845 ± 0.161 s** | **100%** | request start → complete MP4 body |

Every request stack closes to complete client receipt time:

```text
T_client = T_prompt + T_denoise + T_decode
         + T_engine_residual + T_mp4 + T_http_residual
```

Prompt, denoise, decode, and CPU MP4 encoding are direct spans.
`T_engine_residual` and `T_http_residual` are accounting buckets, not inferred
kernels. All 12 profiled/control requests returned HTTP 200 and passed H.264,
1344×768, 124-frame/24-FPS, stereo 32 kHz AAC validation; all canonical warm
containers were byte-identical, and peak sampled device memory was 100,872
MiB.

The four-point/three-evaluation diagnostic averaged **10.375 ± 0.057 s** warm:
4.947 s denoise, 1.849 s VAE decode, 2.102 s engine residual, 1.390 s CPU MP4,
0.051 s prompt, and 0.036 s HTTP. It exists only to demonstrate schedule
sensitivity. It is not Turbo, has no quality comparison, and is not a speedup
claim.

> [!CAUTION]
> The three unprofiled controls were nonstationary at 86.247–104.380 s
> (sample SD 9.111 s). A median comparison yields an apparent −8.48% profiler
> “overhead,” which has the wrong sign and exceeds the protocol's 5% disclosure
> threshold. We therefore do **not** claim that profiling speeds the model or
> that observer overhead is quantified. The decomposition and 85.845 s result
> describe the profiled condition only.

The complete reviewable
[evidence bundle](https://github.com/hsliuustc0106/vllm-omni-cookbook/tree/fe7ac7ab1aaca4192051acaacc399c93cdf14059/blog/assets/figures/minimax-h3-modular-pipeline/evidence/2026-08-24-h200-4gpu)
contains the frozen manifest, exact commands/harness, raw logs and response
headers, health polls, GPU/host samples, per-request signed residuals, output
hashes/`ffprobe` results, and plotting SVG. Open
[PR #5810](https://github.com/vllm-project/vllm-omni/pull/5810) remains open
experimental context and is not the source of these measurements.

## Serving modes {#serving-modes}

Choose the weights first, then choose the performance topology; it is like
choosing which tools go in the truck before deciding how many drivers share
the route. The FL2VA tab is the measured four-H200 topology above; the combined
and Ref2VA tabs preserve current recipe startup variants but are not benchmarked
by this experiment.

{% include usage-cookbook.html modes=page.usage %}

Three cautions travel with every tab:

1. **Combined is a capacity choice.** Both DiTs remain resident in this
   no-offload profile; “four GPUs” alone does not guarantee enough HBM.
2. **Storage, host RAM, and HBM are different.** Selecting one partition saves
   the unused checkpoint download and task-specific weight load, but the exact
   host/HBM delta belongs to a measured topology.
3. **Backend and compile flags are hardware evidence.** The measured FL2VA tab
   pins CUDNN on H200; the pinned recipe owns Blackwell defaults and other
   profiles. Do not transplant either flag set onto an unvalidated card.

## How to choose {#decision-cards}

The operator decision is mostly about which tasks the endpoint promises to
serve. Think of it as publishing a menu: a narrower menu is easier to capacity
plan, while the combined menu avoids maintaining separate endpoints.

{% include decision-cards.html items=page.decisions %}

## Limitations and follow-ups {#limitations}

This is the shipped starting line, not the whole MiniMax-H3 optimization
story. A baseline map is useful precisely because later shortcuts can point to
the component they change instead of redrawing the machine.

- At the pinned baseline, H3 executes one generation request per diffusion
  batch. Step execution/continuous batching in open #5810 is not described as
  shipped.
- H3 is classifier-free-guidance distilled, so `--cfg-parallel-size` must stay
  at 1; there is no negative branch to parallelize.
- The FL2VA-only command is the measured four-H200 profile. Combined and
  Ref2VA commands are current recipe startup variants, not measurements from
  this run; combined serving requires capacity for both DiTs.
- Consumer, ROCm, offload, quantization, caching, attention tuning, and Turbo
  paths have different evidence and belong to their own guides/posts.
- Ref2VA reference combinations and upload limits evolve with the serving API;
  use the pinned/current recipe rather than copying an old PR body.
- The E2E decomposition is a profiled-condition measurement. Nonstationary
  unprofiled controls prevent an observer-overhead claim, and the four-point
  diagnostic has no quality evidence.
- This is Blog 1 in the proposed series tracked by
  [`cookbook#37`](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/37).
  Series metadata/navigation is a separate site change and is not silently
  introduced here.

## References {#references}

These links are the receipt trail for the design: merged anchors establish
what shipped, pinned files establish current behavior, and open work stays
visibly open.

- [PR #5691 — Add MiniMax H3 diffusion support](https://github.com/vllm-project/vllm-omni/pull/5691) (merged)
- [PR #5720 — Add MiniMax-H3 modular pipeline support](https://github.com/vllm-project/vllm-omni/pull/5720) (merged)
- [PR #5752 — Align MiniMax H3 official input matrix](https://github.com/vllm-project/vllm-omni/pull/5752) (merged)
- [PR #5824 — Fail encoder load when a weight or fused shard is missing](https://github.com/vllm-project/vllm-omni/pull/5824) (merged)
- [PR #5810 — MiniMax-H3 continuous batching](https://github.com/vllm-project/vllm-omni/pull/5810) (open; experimental context only)
- [Pinned MiniMax-H3 recipe](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/recipes/MiniMaxAI/MiniMax-H3.md)
- [Pinned pipeline implementation](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py) · [sigma construction](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/time_request.py) · [denoise loop](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/denoise_loop.py)
- [Four-H200 E2E evidence bundle](https://github.com/hsliuustc0106/vllm-omni-cookbook/tree/fe7ac7ab1aaca4192051acaacc399c93cdf14059/blog/assets/figures/minimax-h3-modular-pipeline/evidence/2026-08-24-h200-4gpu) — manifest, commands, harness, raw logs/headers/samples, results, hashes, and media validation
- [Blog 1 plan and E2E experiment contract](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/39) · [series RFC](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/37)
