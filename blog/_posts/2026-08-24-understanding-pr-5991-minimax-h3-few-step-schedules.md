---
layout: post
title: 'Serving MiniMax-H3 in vLLM-Omni (2): why "four steps" has three contracts'
date: 2026-08-24 19:00:00 +0800
author: hsliuustc0106
summary: >-
  Why MiniMax-H3's short uniform request, checkpoint-pinned DMD2 schedule, and
  Turbo LoRA use different sigma-point and denoiser-evaluation contracts—even
  when each is called “four steps.”
tags: [MiniMax-H3, DMD2]
category: PR Analysis
feature: lora
math: true
lang: en
pair: /zh/2026-08-24-understanding-pr-5991-minimax-h3-few-step-schedules/
redirect_from:
  - /2026-08-24-understanding-pr-6476-minimax-h3-turbo-lora/
usage:
  - label: "Download"
    blurb: "only the v1.0 artifact"
    title: "hf download · Turbo v1.0 (the only supported file)"
    code: |
      export TURBO_DIR=/path/to/minimax-h3-turbo
      export TURBO_FILE=minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors
      hf download lightx2v/Minimax-h3-Turbo "${TURBO_FILE}" --local-dir "${TURBO_DIR}"
      export TURBO_LORA="${TURBO_DIR}/${TURBO_FILE}"
    note: >-
      The 8-step, ComfyUI, Ref2VA, and v1.1 artifacts are not supported — only
      the native Diffusers 4-NFE FL2VA/T2VA v1.0 file.
  - label: "Serve"
    blurb: "non-offloaded + two LoRA flags"
    title: "vllm serve · 4x GPU, Turbo preloaded"
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
        --vae-use-tiling \
        --task-type fl2va \
        --lora-backend peft \
        --lora-path "${TURBO_LORA}"
    note: >-
      --lora-path preloads the adapter; each request still activates it
      explicitly. Turbo rejects model-level CPU offload, layerwise offload,
      and DLO, so start from a non-offloaded command.
  - label: "Request"
    blurb: "activate Turbo per request"
    title: "curl · T2VA with the published Turbo sampling contract"
    code: |
      curl -sS -X POST "http://127.0.0.1:${PORT}/v1/videos/sync" \
        -F 'prompt=In a snowy blue-purple forest, Ori carefully walks past a sleeping giant.' \
        -F 'width=1344' \
        -F 'height=768' \
        -F 'aspect_ratio=16:9' \
        -F 'fps=24' \
        -F 'seed=1101' \
        -F 'num_inference_steps=5' \
        -F 'flow_shift=6' \
        -F 'extra_params={"task":"t2va","duration":8.7,"audio_flow_shift":3.0}' \
        -F "lora={\"name\":\"h3-turbo-v1.0\",\"path\":\"${TURBO_LORA}\",\"scale\":1.0}" \
        -o t2va_turbo.mp4
    note: >-
      Five sigma points produce the four denoiser evaluations the artifact
      expects; video flow shift 6, audio flow shift 3. For FL2VA change the
      task and add input_reference. Invalid point/shift values fail closed.
decisions:
  - when: "A distilled checkpoint publishes `base_schedule`"
    pick: "Let checkpoint metadata lead"
    why: "Normally omit `num_inference_steps`; if supplied, it must equal the number of intervals, not boundaries."
  - when: "The released base checkpoint needs lower latency"
    pick: "Use the supported Turbo LoRA contract"
    why: "Activate the exact v1.0 FL2VA/T2VA adapter and request five sigma points with video/audio shifts 6/3."
  - when: "Quality or reproducibility needs the base reference"
    pick: "Use the ordinary 50-point path"
    why: "The released base checkpoint's default is 50 sigma points and 49 denoiser evaluations."
  - when: "You only want a diagnostic control"
    pick: "Short uniform request without distillation"
    why: "`num_inference_steps=4` means four uniform sigma points and three denoiser evaluations; do not present it as a quality-preserving fast path."
---

## TL;DR {#tldr}

**MiniMax-H3 currently has three different contracts that people may call
“four steps”: four uniformly generated sigma points, four intervals pinned by
a distilled checkpoint, or four denoiser evaluations required by Turbo
LoRA.** Think of a route with five stations and four journeys between them:
counting stations and counting journeys produces different numbers even
though both describe the same route. Here a **sigma boundary** is one scheduled
noise level, while **NFE** (number of function evaluations) is the number of
denoiser model calls between adjacent boundaries.

[PR #5991](https://github.com/vllm-project/vllm-omni/pull/5991) added the
checkpoint-native path: a DMD2-distilled FL2VA checkpoint can publish the exact
continuous noise positions used during training. [PR #6476](https://github.com/vllm-project/vllm-omni/pull/6476)
later added a separate runtime path: the released base checkpoint stays intact,
and a request activates one supported Turbo LoRA plus its required sampling
contract. **#5991 did not implement Turbo LoRA, and neither path is equivalent
to arbitrarily dropping calls from the base trajectory.**

| Path | User/checkpoint input | Sigma points or boundaries used | Denoiser evaluations (NFE) | What the number means |
|---|---|---:|---:|---|
| Ordinary uniform base path | request omitted or `num_inference_steps=50` | 50 uniformly generated points after the modality shift | 49 | On this path the request field controls **sigma-point count**. |
| Ordinary uniform short request | request `num_inference_steps=4` | 4 uniformly generated points | 3 | This is **not** a four-NFE request. |
| Checkpoint-pinned DMD2 path from #5991 | metadata `base_schedule: [1.0, 0.7, 0.4, 0.15, 0.0]`; request omitted or explicitly `4` | 5 exact trained boundaries | 4 | `DMD2SigmaSchedule.num_inference_steps` counts the four intervals. |
| Shipped Turbo LoRA path from #6476 | request `num_inference_steps=5`; supported adapter active | 5 uniform-path points with Turbo-required shifts | 4 | The legacy Turbo integration validates five points because they produce four NFE. |

> [!IMPORTANT]
> This post always distinguishes **sigma points/boundaries** from **denoiser
> evaluations (NFE)**. The API field name alone does not tell you which quantity
> it controls; the active checkpoint and adapter path do.

## Why a distilled model cannot use an arbitrary short uniform schedule {#why-not-uniform}

A distilled checkpoint learns a particular express route, not every possible
way of removing stops from the local route. The analogy breaks at training:
noise positions are continuous numerical inputs to the model, so changing a
boundary changes the state the denoiser must handle rather than merely changing
a timetable.

The released base model is trained to follow its ordinary trajectory. Asking
that model for fewer uniformly spaced sigma points reduces denoiser calls, but
does not provide weights trained to bridge those larger gaps. A distilled
checkpoint or a few-NFE adapter supplies that missing learned behavior for a
specific trajectory.

“Turbo skips 45 of 49 denoiser evaluations” is useful shorthand for the
operator-visible result, but it is not the mechanism. Turbo does not select
four arbitrary entries from the base model's 49-evaluation trajectory; its
adapter was trained for a different four-evaluation route.

## The baseline: 50 sigma points, 49 denoiser evaluations {#baseline-path}

The ordinary H3 path draws evenly spaced marks on a ruler, then bends that
ruler separately for video and audio. The marks start uniform, but the
modality-specific time shift moves their interior values while preserving the
same number of points and the endpoints.

At the pinned upstream snapshot
[`072bfc02`](https://github.com/vllm-project/vllm-omni/commit/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65),
[`time_request.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/time_request.py)
constructs the ordinary base positions with a uniform `linspace(1.0, 0.0,
num_steps)`. For a base position $u$ and a positive shift scale $k$, H3 maps it
to:

$$
s_k(u) = \frac{k u}{1 + (k - 1)u}
$$

[`denoise_loop.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/denoise_loop.py)
then makes one joint video/audio denoiser call per adjacent pair:

$$
\mathrm{NFE} = \lvert\mathrm{sigma\ boundaries}\rvert - 1
$$

That is why the default 50 sigma points produce 49 NFE—and why an ordinary
request with `num_inference_steps=4` produces four points but only three NFE.
The field name is historical; on this uniform path its value controls point
count.

## What PR #5991 added: checkpoint ownership of the route {#checkpoint-dmd2}

#5991 lets a distilled checkpoint put its route card inside the package, so the
server reads the trained boundaries instead of inventing a new uniform route.
In H3, “DMD2” operationally means that the checkpoint is trained for a small,
fixed sequence of continuous noise positions and publishes that sequence as
metadata.

The contract lives under `_minimax_h3` in the active partition's
`model_index.json`. This is an illustrative metadata excerpt from the merged
PR—not a claim that a downloadable checkpoint exists at a particular path:

```json
{
  "_minimax_h3": {
    "partition": "fl2va",
    "tasks": ["t2va", "fl2va"],
    "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
    "base_schedule": [1.0, 0.7, 0.4, 0.15, 0.0]
  }
}
```

The shared
[`DMD2SigmaSchedule`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/sched/sigma_schedule.py)
fails closed unless the schedule:

- contains at least two finite positions;
- starts exactly at `1.0` and ends exactly at `0.0`;
- is strictly decreasing.

This class is deliberately separate from
`DMD2Config.denoising_timesteps`: H3 stores continuous rectified-flow positions
in `[0, 1]`, not the integer scheduler timesteps used by scheduler-backed
pipelines.

An absent `base_schedule` means “use the ordinary uniform path.” An explicitly
empty list is malformed and raises an error rather than silently falling back.
Five boundaries describe four intervals, so the schedule reports
`num_inference_steps == 4` for the example above.

The
[`pipeline_minimax_h3.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py)
request contract follows that interval count. Users should normally omit
`num_inference_steps`; if they explicitly send `4`, it agrees with the four
intervals, while `5` or `50` is rejected. The full five-boundary sequence still
goes to the denoise loop. #5991 changes where those positions come from and how
they are validated; it does not replace H3's DiT weight loader or denoising
solver.

## One base schedule, two modality trajectories {#modality-shifts}

Video and audio share one list of waypoints but take differently curved lanes,
like two vehicles following the same exits with different acceleration maps.
The shared ownership matters: the modalities stay aligned by base position,
while separate shift scales adapt those positions to each modality.

For the illustrative five-boundary schedule, the formula above produces:

| Boundary index | Base position $u$ | Video sigma, $k=12$ | Audio sigma, $k=3$ |
|---:|---:|---:|---:|
| 0 | 1.0 | 1.0 | 1.0 |
| 1 | 0.7 | 0.9655172 | 0.875 |
| 2 | 0.4 | 0.8888889 | 0.6666667 |
| 3 | 0.15 | 0.6792453 | 0.3461539 |
| 4 | 0.0 | 0.0 | 0.0 |

Schedule ownership is also partition-local. A combined server retains separate
metadata for FL2VA and Ref2VA, then selects the schedule for the active task.
A distilled FL2VA partition therefore cannot silently force a regular Ref2VA
partition onto the same four-interval route. This boundary was added during
#5991 review and is covered by the pinned CPU contract tests.

## What PR #6476 added: runtime Turbo LoRA {#runtime-turbo}

#6476 takes a different route: it preloads a specialized driver beside the
released base checkpoint, then a request chooses whether that driver is active.
The base checkpoint metadata is not replaced, and the adapter's weight/layout
translation is separate from #5991's checkpoint schedule class.

The existing LoRA manager delegates H3-specific conversion to a model-owned
loader. That boundary matters because the published adapter and native H3
transformer describe the same targets in different layouts:

- Diffusers names are translated to native transformer and token-refiner
  names, and fused FFN rows are restored to H3's `[gate; up]` order;
- separate Q/K/V adapters bind to H3's packed QKV projection, while fused
  LoRA-B tensors are sliced using global output rows so tensor-parallel ranks
  receive the right weights;
- the loader validates the full metadata, rank/alpha, target set, and global
  A/B shapes before mutating wrappers. Activation is transactional, so a
  binding or validator failure resets the adapter state rather than leaving a
  partially active model.

The adapter is preloaded at server startup but activated per request. The
five-point request is deliberate: on the ordinary uniform path, five sigma
points yield four NFE. Rewriting it as `num_inference_steps=4` would create only
four uniform points and three NFE, so the Turbo runtime rejects it.

## Run the shipped Turbo path {#turbo-usage}

The combined workflow below uses the only supported artifact,
`minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors`. Download it, preload
it into a non-offloaded FL2VA server, then activate it on the T2VA request with
five sigma points and shifts 6/3.

{% include usage-cookbook.html modes=page.usage %}

Keep these shipped restrictions beside the commands:

- only the native Diffusers four-NFE FL2VA/T2VA v1.0 artifact is accepted;
  Ref2VA, the eight-NFE release, ComfyUI layout, and v1.1 are unsupported;
- execution is dynamic only—no prefusion;
- model-level CPU offload, layerwise offload, and distributed layerwise offload
  (DLO) are rejected;
- only one LoRA can be active, so Turbo cannot be composed with a second style
  or identity adapter;
- filename, metadata, rank/alpha, and target shapes must match the declared
  artifact contract;
- the legacy request carries an adapter path. A public endpoint should expose
  an allowlisted name-to-path mapping rather than unrestricted client-supplied
  path or download resolution.

## Same phrase, different ownership {#contract-ownership}

The easiest way to avoid configuration mistakes is to ask who owns the route,
like checking whether directions came from the vehicle, the road authority, or
the driver. In H3, ownership determines both the positions and how the request
field is interpreted.

| Path | Who owns the positions? | What the request may control | Failure mode |
|---|---|---|---|
| Ordinary base | Request/pipeline | Number of generated sigma points | A low point count runs, but the base weights were not made quality-preserving for that shortcut. |
| Checkpoint-pinned DMD2 | Active partition metadata | Nothing normally; an explicit interval count may confirm the metadata | A mismatched explicit count is rejected. |
| Runtime Turbo LoRA | Supported adapter contract plus request | Adapter activation and the exact five-point, shift-6/3 contract | Wrong task, points, shifts, artifact, offload, or composition is rejected. |

Checkpoint-pinned DMD2 support and runtime Turbo support are not interchangeable
packaging formats. No concrete artifact has been validated through both paths,
and their request contracts can conflict: the illustrated checkpoint expects an
explicit interval count of `4`, while the shipped Turbo path requires a point
count of `5`.

## Why four denoiser evaluations are not free {#not-free}

Fewer denoiser calls are like crossing a river in four long jumps instead of 49
short ones: the route is shorter, but only a jumper trained for those exact
landings can do it reliably. The analogy does not promise equal output—the
training method, artifact, prompt, and workload still determine quality.

A plain short uniform request is a useful diagnostic control because it isolates
the cost of fewer denoiser calls. It is not an equivalent-quality baseline. The
#6476 evidence reports visibly degraded output without the Turbo adapter and
coherent video/audio restoration with it, but it does not establish universal
quantitative parity for every prompt.

The #6476 author measurement separates the large schedule effect from the
smaller dynamic-adapter cost. It used 4×H200, USP4/Ring1, VAE patch-parallel 4,
text-encoder TP1, regional compile and FlashAttention; 768×1344 T2VA at 107
frames/24 FPS; the same prompt and seed; and the median of five runs after two
full-shape warmups.

| Path | LoRA execution | NFE | Stage-0 p50 |
|---|---|---:|---:|
| Base reference | None | 49 | 68.388 s |
| Short diagnostic control | None | 4 | 8.967 s |
| Turbo | Dynamic | 4 | 9.688 s |

Turbo is **7.06× faster than the 49-NFE reference**, while its dynamic LoRA
work makes it **8.05% slower than the same-schedule no-LoRA control**. That is
the useful interpretation: most of the latency reduction comes from executing
four rather than 49 denoiser evaluations, while the adapter pays extra work on
each retained evaluation. The short control's visibly degraded output means it
is a compute control, not an equal-quality alternative.

The local validation below isolates that second effect on another topology:
once both requests already use five sigma points/four NFE, what work does
activating the adapter add? #5991 itself remains a schedule contract and does
not publish a named public distilled checkpoint benchmark.

## Local validation: where Turbo overhead appears {#local-validation}

A fair LoRA comparison keeps the road fixed and changes only the passenger,
like timing the same car and route with one extra load. Here Base and Turbo use
the same sigma points, NFE, prompt, seed, shape, server, and warm cache; only
request-time adapter activation changes. These are pinned local spot checks,
not a promise for every deployment.

### Same-schedule A/B

The final-path run used vLLM-Omni
`0.27.0rc2.dev159+g072bfc02d` at upstream SHA `072bfc02`, with its venv and
runtime caches on node-local storage. It ran on 2×L20X bound to NUMA 0: eager
TP2, text-encoder TP2, VAE patch-parallel 2/tile, CUDNN attention, 1344×768
T2VA, requested 4.0 s/107 frames, seed 1101, five sigma points/four NFE, and
video/audio shifts 6/3. One Base and one Turbo warmup preceded the measured
`A B B A A B` order, n=3 per condition. Preparation (8.621 s) and
process-to-readiness (88.286 s) were recorded separately.

| Condition | Stage-0 mean ± sample SD | Stage-0 median | Diffuse mean ± sample SD | Diffuse median | Client median | Stage-0 CV |
|---|---:|---:|---:|---:|---:|---:|
| Base control | 15.108 ± 0.024 s | 15.099 s | 10.211 ± 0.009 s | 10.206 s | 16.246 s | 0.16% |
| Turbo LoRA | 16.398 ± 0.072 s | 16.434 s | 11.435 ± 0.015 s | 11.444 s | 17.548 s | 0.44% |

Median Turbo overhead was **+8.85% stage-0**, **+12.13% diffuse**, and
**+8.01% client wall**. Both conditions reported the same 76,754 MiB request
peak because LoRA buffers were preallocated. All requests returned HTTP 200 and
fully decoded; outputs were byte-deterministic within each condition, while
Base and Turbo outputs differed.

### What nsys shows

The kernel trace aligned the non-schedule controls with Blog 1: FL2VA T2VA,
Ulysses4/Ring1/DiT-TP1, text-encoder TP4, VAE patch-parallel 4/tile, BF16,
CUDNN, regional compile, 1344×768, requested 5.0 s/124 frames, and seed 1101.
Both sides still used five sigma points/four NFE. GPUs 4–7 were kept on NUMA 1
rather than repeat Blog 1's cross-NUMA physical placement. One compile/warmup
request preceded one Nsight Systems 2026.1.3 trace per condition, so these
observer-affected spans explain mechanism rather than establish another latency
headline.

| Direct synchronized span | Base | Turbo | Delta |
|---|---:|---:|---:|
| Stage-0 | 10.332 s | 11.200 s | +8.39% |
| Pipeline diffuse | 6.569 s | 7.331 s | +11.61% |
| Pipeline decode | 1.907 s | 1.920 s | +0.69% |

The visible kernel evidence points to additive low-rank and layout work, not a
different denoiser-call count:

| Turbo-only visible signature, per device | Unique kernels | Launches | Visible time | Examples |
|---|---:|---:|---:|---|
| Rank-128 / `badd` GEMMs | 4 | 439 | 38.122 ms | `nvjet_tst_128x288...badd`, `nvjet_tst_256x160...badd` |
| Fused copy/slice | 3 | 157 | 45.220 ms | `triton_poi_fused_copy_slice_{0,1,4}` |

Meanwhile, matched core kernels kept equal launch counts and near-identical
visible time: the two main GEMM families ran 5,292 and 1,764 times/device in
both conditions, the short-SDPA kernel ran 1,764 times/device, and LayerNorm ran
7,056 times/device. That is the architectural signature of dynamic LoRA: the
base route remains, and low-rank projections plus packed-layout handling are
added around it.

> [!CAUTION]
> On this Hopper platform, nsys node mode covers host-launched CUDA graph nodes,
> not every device-launched graph node. Base and Turbo have different graph
> coverage, so aggregate kernel/category totals are not complete workload
> totals. The trustworthy comparisons are the synchronized spans, equal-count
> matched kernels, and conservative Turbo-only signatures above.

## How to choose {#how-to-choose}

Choose the path whose learned artifact owns the route, just as you use the key
made for a particular lock. A base checkpoint, checkpoint-native distilled
release, and runtime adapter are different operational products even when their
operator-facing NFE count matches.

{% include decision-cards.html items=page.decisions %}

## Compatibility and deployment safety {#compatibility-safety}

Treat each fast path as a narrow operating envelope, like a bridge with posted
vehicle and weight limits. Passing validation says the request matches the
implemented contract; it does not broaden the contract to nearby artifacts or
features.

- A checkpoint-native distilled release should carry its exact trained
  `base_schedule`; do not add a newly generated uniform replacement.
- A runtime Turbo request needs both adapter activation and the exact five-point
  sampling settings. Preloading alone does not activate the adapter.
- Do not combine the shipped Turbo path with Ref2VA, prefusion, any supported
  offload mode, or a second LoRA.
- Do not assume an arbitrary LightX2V artifact is compatible because its name
  contains “Turbo” or “four-step.”
- Do not expose request-supplied filesystem or download paths to untrusted
  clients; map approved public names to server-owned paths.
- Do not claim that a DMD2 checkpoint and Turbo LoRA can be converted into each
  other without validating the concrete weights, metadata, and trajectory.

## Limitations and follow-ups {#limitations}

This post draws the shipped boundary rather than filling in missing products,
like a map that labels an unopened road instead of inventing a route through
it. The omissions are intentional and keep examples deployable.

- Upstream does not document a named, stable, publicly accessible
  checkpoint-native DMD2 FL2VA artifact at the pinned snapshot. The #5991 half
  therefore shows metadata only, not a copy-ready serve command.
- The local A/B isolates same-schedule adapter overhead; it is not the #6476
  49-NFE speedup comparison, not a quality comparison, and not a universal
  latency claim.
- [#6473](https://github.com/vllm-project/vllm-omni/pull/6473) and
  [#6017](https://github.com/vllm-project/vllm-omni/pull/6017) are draft,
  unshipped LoRA-runtime directions. Their proposed behavior is not described
  as available.
- Ref2VA Turbo, arbitrary LightX2V artifacts, prefusion, adapter composition,
  quantization, TeaCache, Cache-DiT, continuous batching, VAE optimization, and
  Super Acceleration are outside this post.

## Evidence ledger {#evidence-ledger}

The claims below carry receipts rather than relying on the word “merged,” like
an audit trail that records both the rule and the check that exercises it. All
source links are pinned to the reviewed upstream snapshot where practical.

| Claim | Source | Shipped status | Independent evidence |
|---|---|---|---|
| A distilled checkpoint can pin exact continuous schedule positions | [#5991](https://github.com/vllm-project/vllm-omni/pull/5991) + [`sigma_schedule.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/sched/sigma_schedule.py) | Merged/shipped | [`test_dmd2_sigma_schedule.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/tests/diffusion/sched/test_dmd2_sigma_schedule.py) |
| Video/audio use separately shifted versions of one base schedule | [#5991](https://github.com/vllm-project/vllm-omni/pull/5991) + [`time_request.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/time_request.py) | Merged/shipped | CPU schedule reference-value tests |
| An explicit checkpoint-schedule mismatch fails | [`pipeline_minimax_h3.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py) | Shipped | [`test_minimax_h3_contract.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/tests/diffusion/models/minimax_h3/test_minimax_h3_contract.py) |
| The supported Turbo artifact maps and binds to native H3 | [#6476](https://github.com/vllm-project/vllm-omni/pull/6476) + [`lora.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/lora.py) | Merged/shipped | CPU LoRA tests + PR full-model evidence |
| Turbo requires five points/four NFE and shifts 6/3 | [current pipeline](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py) + [pinned recipe](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/recipes/MiniMaxAI/MiniMax-H3.md#turbo-lora) | Merged/shipped | CPU validation + #6476 end-to-end evidence |
| Turbo workflow, local spot check, and visible LoRA kernel signatures | [combined Turbo workflow above](#turbo-usage) + [local validation](#local-validation) | Shipped workflow plus pinned local validation; not universal | Warmed n=3 A/B + four-device nsys, with CUDA-graph coverage caveat |
| Model-declared/generalized LoRA runtime | [#6473](https://github.com/vllm-project/vllm-omni/pull/6473) / [#6017](https://github.com/vllm-project/vllm-omni/pull/6017) | Draft/unshipped | Not described as available |

## References {#references}

These links are the source map for the two shipped routes, like the legend that
turns the article's shorthand back into reviewable code and evidence.

- [Blog 2 planning issue #40](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/40) · [MiniMax-H3 series RFC #37](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/37)
- [PR #5991 — Add distilled four-NFE sigma schedule support for MiniMax-H3 T2VA](https://github.com/vllm-project/vllm-omni/pull/5991) (merged)
- [PR #6476 — Support MiniMax-H3 Turbo LoRA with the legacy manager](https://github.com/vllm-project/vllm-omni/pull/6476) (merged)
- [Part 1 — MiniMax-H3 modular pipeline]({{ site.baseurl }}/2026-08-24-understanding-pr-5720-minimax-h3-modular-pipeline/)
- Pinned source: [`sigma_schedule.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/sched/sigma_schedule.py) · [`time_request.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/time_request.py) · [`pipeline_minimax_h3.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py) · [`denoise_loop.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/denoise_loop.py) · [`lora.py`](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/vllm_omni/diffusion/models/minimax_h3/lora.py)
- [Pinned MiniMax-H3 recipe, Turbo LoRA section](https://github.com/vllm-project/vllm-omni/blob/072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65/recipes/MiniMaxAI/MiniMax-H3.md#turbo-lora)
- [Draft #6473 — model-declared LoRA runtime](https://github.com/vllm-project/vllm-omni/pull/6473) · [draft #6017 — generalized LoRA loading/composition](https://github.com/vllm-project/vllm-omni/pull/6017)
