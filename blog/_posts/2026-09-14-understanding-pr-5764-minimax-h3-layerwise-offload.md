---
layout: post
title: "Serving MiniMax-H3 in vLLM-Omni (4): layerwise offload — the whole model on two workstation GPUs (PR #5764)"
date: 2026-09-14 12:00:00 +0800
author: hsliuustc0106
summary: >-
  PR #5764 makes MiniMax-H3 run on two workstation GPUs: DLO keeps a few DiT
  blocks resident, streams the rest plus the 51.5 GB encoder from pinned host
  memory — a 1344×768 50-step video completes on 2× RTX 5090 at ~22.6 GiB per
  card.
tags: [MiniMax-H3, RTX-5090, DLO]
category: PR Analysis
feature: offloader
lang: en
pair: /zh/2026-09-14-understanding-pr-5764-minimax-h3-layerwise-offload/
usage:
  - label: "Serve · 2× RTX 5090"
    blurb: "TP2 + 20 resident DiT blocks, 1344×768"
    title: "vllm serve · MiniMax-H3 FL2VA with DLO"
    code: |
      CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/MiniMax-H3/FL2VA \
        --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
        --task-type fl2va \
        --num-gpus 2 --tensor-parallel-size 2 --text-encoder-tp-size 2 \
        --usp 1 --ring 1 --vae-patch-parallel-size 2 \
        --vae-parallel-mode tile --vae-use-tiling \
        --enable-distributed-layerwise-offload --dlo-no-use-allgather \
        --dlo-resident-layers 20 --enforce-eager \
        --diffusion-attention-backend CUDNN_ATTN
    note: >-
      Memory-first config. The two-rank B300 capacity run for this shape
      peaked at 27,726 MiB per rank — re-measure on the target cards before
      raising the resident count.
  - label: "Serve · 1× RTX 5090"
    blurb: "Single GPU, 12 resident blocks"
    title: "vllm serve · single-card DLO"
    code: |
      CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/MiniMax-H3/FL2VA \
        --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
        --task-type fl2va \
        --num-gpus 1 --tensor-parallel-size 1 --text-encoder-tp-size 1 \
        --usp 1 --ring 1 --vae-patch-parallel-size 1 \
        --vae-parallel-mode tile --vae-use-tiling \
        --enable-distributed-layerwise-offload --dlo-no-use-allgather \
        --dlo-resident-layers 12 --enforce-eager \
        --diffusion-attention-backend CUDNN_ATTN
    note: >-
      A 50-step B300 allocation test with this single-rank topology peaked
      at 26.50 GiB — a capacity proxy, not a consumer-GPU latency claim.
  - label: "Offline · all tasks"
    blurb: "T2VA, FL2VA, Ref2VA runner"
    title: "run_h3_2gpu_all_tasks.sh"
    code: |
      RUN_ROOT=/path/to/run-root \
      MODEL_ROOT=/path/to/MiniMax-H3 \
      GPU_IDS=0,1 \
      PROFILE=rtx5090 \
      bash examples/offline_inference/minimax_h3/run_h3_2gpu_all_tasks.sh
    note: >-
      PROFILE=rtx4090 selects the conservative 24 GB defaults
      (1024×576, 12 resident layers). DLO_RESIDENT_LAYERS=N overrides
      either profile.
decisions:
  - when: "Two 24–32 GB workstation cards are all you have"
    pick: "DLO + --dlo-no-use-allgather"
    why: "TP-local streaming keeps only the rank's shard in host memory and never reconstructs full DiT blocks per rank; the validated shape (1344×768, 50 steps) sampled ~22.6 GiB per 32 GB card."
  - when: "Several synchronized replicas share one fast P2P domain"
    pick: "DLO AllGather (no --dlo-no-use-allgather)"
    why: "Collective reconstruction replaces per-rank full host copies, and it now composes with online FP8 — see the #6279 post (−39.0% host PSS on DP2/SP2)."
  - when: "Tuning latency against HBM headroom"
    pick: "--dlo-resident-layers N"
    why: "The leading N DiT blocks stay on device; start at 20 (2×32 GB) or 12 (24 GB / single card). Host RAM does not drop when you raise it — resident layers keep pinned CPU master copies."
  - when: "Multiple independent engines on one host"
    pick: "Host Weight Runtime"
    why: "Share final-layout host artifacts instead of one full copy per engine — the #6591 post measured −36.6% pair PSS for two TP2 engines."
  - when: "Startup time matters"
    pick: "Know the mmap gate"
    why: "H3 stays on the regular loader under DLO until mmap can apply the grouped-QKV and fused-MLP transforms — the explicit opt-out from this PR's review."
  - when: "DP replicas each serve a request"
    pick: "AllGather waves, preflight-checked"
    why: "#5864 rejects incompatible waves (shape/CFG/steps/LoRA/extra_args) before dispatch; the no-AllGather path keeps one request per replica."
---

## TL;DR {#tldr}

**PR #5764 is a capacity feature, not a speed feature: it makes MiniMax-H3 —
66.3 GB of DiT plus a 51.5 GB Qwen3-VL encoder plus video/audio VAEs — run on
two workstation GPUs by keeping most of the weights in pinned host memory and
streaming them onto the card one layer at a time.** Think of a chef with a
small kitchen counter (GPU memory) and a walk-in refrigerator right next to it
(host RAM): instead of trying to fit every ingredient on the counter, the chef
brings out exactly what the current step needs, and keeps the handful of
always-used items within arm's reach.

| Profile | GPUs | Starting shape | Resident DiT blocks | Attention | Validation |
|---|---:|---:|---:|---|---|
| `rtx5090` | 2 × 32 GB | 1344×768 | 20 | `CUDNN_ATTN` | target-hardware validated |
| `rtx4090` | 2 × 24 GB | 1024×576 | 12 | `CUDNN_ATTN` | capacity-proxy starting point |

On 2 × RTX 5090, one full 50-step T2VA request at 1344×768 completed in
**8 min 38 s** client-side with a sampled peak of **~22.6 GiB per GPU** and a
clean `ffmpeg` decode of the H.264 + 32 kHz stereo AAC output. That is the
price of the trade: the video fits and is correct, but capacity was bought
with bandwidth — this is a memory-first configuration, and the PR's own review
history records it being retitled from a perf change to a feature change for
exactly that reason.

## Background {#background}

**This section establishes why MiniMax-H3 was previously a datacenter-GPU
model: the weights simply did not fit on workstation cards under any existing
path.** If the resident path is buying a warehouse to stock every shelf at
once, and pure sequence parallelism is hiring more staff for one shared
warehouse, neither helps when each staff member's backpack (24–32 GB of HBM)
is smaller than the inventory.

The existing options each hit a wall:

- **Resident execution** loads the whole model into HBM. MiniMax-H3's DiT
  alone (66.3 GB) exceeds a 32 GB card.
- **Pure Ulysses sequence parallelism** shards activations across ranks but
  keeps weights replicated per rank — the capacity problem remains.
- **Distributed layerwise offload (DLO)** existed, but was built around
  uniform-block transformers and the AllGather reconstruction path. MiniMax-H3
  breaks both assumptions: its `token_refiner.blocks` mixes block sizes
  (~1231 MB and ~239 MB), and its 51.5 GB text encoder needs its own staging
  story.

PR #5764 ([merged 2026-08-06](https://github.com/vllm-project/vllm-omni/pull/5764),
commit
[`1c2a81f`](https://github.com/vllm-project/vllm-omni/commit/1c2a81f6d84aea4fff53bd2f894c2a287c237245))
is the PR that makes DLO MiniMax-H3-aware. Two sibling fixes landed in the
same window — #5802 for the heterogeneous-block crash and #5864 for DP
concurrent-request correctness — and this post tells the three as one story.

## What PR #5764 changed {#key-changes}

**The PR teaches the offloader three new tricks: keep a chosen few layers on
the device, stream only each rank's own shard, and stage the non-DiT giants on
demand.** Each corresponds to a knob you can see in the serve commands later.

![MiniMax-H3 DLO layout: host pinned memory streams DiT shards, encoder and VAE blocks; each GPU keeps resident blocks plus two rotating stream slots]({{ site.baseurl }}/assets/figures/pr-5764-minimax-h3-layerwise-offload/fig1-dlo-layout.svg)

### Explicit module residency {#module-residency}

`--dlo-resident-layers N` keeps the *leading* N DiT blocks on the device for
the whole request, instead of streaming every block every step. Which paths
count as "DiT blocks you may pin" is model-declared via the new
`resident_dit_paths` field in
[`OffloadPlan`](https://github.com/vllm-project/vllm-omni/blob/1c2a81f6d84aea4fff53bd2f894c2a287c237245/vllm_omni/diffusion/offloader/offload_plan.py) —
so a consumer-GPU tuning knob cannot accidentally pin auxiliary or dual DiTs.
A new
[`module_residency.py`](https://github.com/vllm-project/vllm-omni/blob/1c2a81f6d84aea4fff53bd2f894c2a287c237245/vllm_omni/diffusion/offloader/module_residency.py)
adds `PinnedModuleStager` (stage a module from an immutable pinned CPU
snapshot, without copying device weights back to CPU) and
`PinnedResidentLayerGroup`. Reviewers asked for — and got — a warning when
`--dlo-resident-layers` is set but the model declares no resident paths, and
an empty-block guard that leaves unsupported models' DiTs fully resident
instead of registering hooks against nothing.

### TP-local streaming without AllGather {#tp-local-streaming}

The AllGather mode reconstructs a *full* DiT block on every rank — each rank
holds a shard on the host and the collective assembles the whole layer. For
two consumer cards on a workstation, the PR instead streams each rank's
tensor-parallel shard directly (`--dlo-no-use-allgather`): no full-block
reconstruction, no lockstep collective, and the rank keeps a rank-local host
copy. This is the mode the RTX recipes use. The same review that shaped it
also gated DP concurrency to the AllGather path only, because the
no-AllGather forward expects exactly one prompt per replica.

### Staging the encoder and the VAE {#staging}

The 51.5 GB Qwen3-VL encoder and the VAE decoder are staged rather than kept
resident, while the H3-required encoders stay on device. During review this
was refactored from model-specific hooks into the declarative plan the
offloader now reads: `OffloadPlan.encoder_block_attrs` maps encoder paths to
rank-local block lists (streamed with ordinary layerwise hooks, never with the
DiT AllGather group), and `on_demand_component_paths` makes CPU staging opt-in
per model. The consumer path also gets VAE patch parallelism across both GPUs
(tiled decode) and cuDNN attention, plus a guarded prefix-KV fast path so ring
attention always receives an explicit mask.

### What the reviewers changed {#review-story}

Three review decisions are worth knowing because they explain the shape of
the code today:

- **H3 explicitly stays on the regular loader.** A P1 review comment caught
  that the mmap path had no producer for MiniMax-H3's grouped-QKV weight
  transform — under DLO+AllGather it would install raw checkpoint weights.
  Rather than rely on a missing producer, the author added the explicit
  `_supports_mmap_loading` opt-out, so loader and backend share one safety
  gate until transformed mmap loading exists.
- **DLO CLI fields survive the deploy path.** `--dlo-resident-layers` and
  companion flags were silently dropped for registered/deploy-config
  pipelines; the fix adds all DLO fields to `StageDeployConfig`.
- **The PR was retitled from perf to feature.** A reviewer noted the evidence
  was one un-warmed run on an older commit plus a B300 proxy — no matched
  main-vs-PR latency or quality comparison — and asked for either real
  benchmarks or an honest retitle. The author retitled.

## Two correctness fixes in the same week {#bugfixes}

**Both fixes answer the same question: what breaks when a real model meets
DLO's simplifying assumptions?** The assumptions were "all transformer blocks
are the same size" and "one request at a time is fine" — MiniMax-H3 broke
both.

[PR #5802](https://github.com/vllm-project/vllm-omni/pull/5802) (merged
2026-08-05) fixed the AllGather crash on heterogeneous blocks. The two shared
GPU buffers are sized for the *largest* block across all groups, but
`prefetch_layer` handed the entire max-sized buffer to
`all_gather_into_tensor` while the input was the *current* (smaller) block's
shard — for any smaller block, `output.numel() != dp_size * input.numel()` and
the contract check fails during `enable()`'s first prefetch. The fix is one
slice — use the already-computed per-block AllGather output size — and the
reproducer that flushed it out was MiniMax-H3 FL2VA on 8× Ascend NPU with
USP=8.

[PR #5864](https://github.com/vllm-project/vllm-omni/pull/5864) (merged
2026-08-08) made DP concurrent requests actually work. With DLO+AllGather, up
to DP-size requests run as one wave — but the multiprocess result path
assumed a single rank-0 result queue, and one invalid or
control-flow-incompatible request could fail on one replica while another
entered AllGather, hanging the wave. The fix adds a wave *preflight* (shape,
CFG, denoising steps, LoRA, `extra_args` compared as a canonical signature,
non-empty prompt) that rejects the whole wave before worker dispatch, and
routes results through a broadcast message queue tagged with the same
`wave_id` on every rank. The `extra_args` check matters more than it looks:
#5764's first version read the wrong attribute (always `None` for normal
requests), so differing pipeline-specific schedules could pass the guard and
make DP ranks enter different AllGather sequences — a hang, not an error.

## Measured impact {#measured-impact}

**One honest number set exists, and it is a validation, not a benchmark.** At
commit `ae6577ea`, one full 50-step T2VA request on 2 × RTX 5090:

| Shape | Frames | Client E2E | Sampled peak/GPU | Output |
|---:|---:|---:|---:|---|
| 1344×768 | 124 @ 24 FPS | 8 min 38 s | ~22.6 GiB | H.264 + 32 kHz stereo AAC; full `ffmpeg` decode passed |

Environment: vLLM 0.26.0, vLLM-Omni `0.26.1.dev14+gae6577ea`, PyTorch
2.11.0+cu130. The caveats are part of the claim: single end-to-end run, not a
warmed multi-run benchmark; the memory value is a sampled `nvidia-smi` peak,
not a CUDA allocator high-water mark; and there is no main-vs-PR A/B latency
comparison — which is why the PR is titled a feature change.

The capacity planning numbers come from the
[RTX 5090 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-5090.md):

| Resource | Requirement |
|---|---|
| GPU HBM | 32 GiB per card (5090); 24 GB profile uses 1024×576 + 12 resident blocks |
| Checkpoint storage | 135 GiB per partition (`FL2VA` and `Ref2VA` are separate; one server at a time) |
| System RAM | 200 GiB minimum, 384 GiB recommended |

Two proxy measurements anchor the resident-layer defaults: a 50-step B300
allocation test with the single-rank topology (12 resident) peaked at
26.50 GiB, and the two-rank TP2 run (20 resident, 1344×768, 50 steps) peaked
at 27,726 MiB per rank. Both are explicitly labeled memory/correctness
proxies, not consumer-GPU latency claims. One non-obvious behavior worth
knowing: raising `--dlo-resident-layers` improves latency but does **not**
reduce host RAM, because resident layers retain their pinned CPU master
copies.

For what DLO costs versus a resident deployment at the datacenter scale, the
H100 four-GPU matrix in the [#6279 online-FP8 post]({{ site.baseurl }}/2026-08-19-pr-6279-dlo-online-fp8-allgather/) is the closest paired measurement on this blog.

## How to use it {#how-to-use}

{% include usage-cookbook.html modes=page.usage %}

For `Ref2VA`, stop the `FL2VA` server and restart the same command with the
`Ref2VA` partition — reference video count and prompt length raise activation
memory, so start with one request at a time. After the modular pipeline
([Blog 1]({{ site.baseurl }}/2026-08-24-understanding-pr-5720-minimax-h3-modular-pipeline/)),
`--task-type fl2va` / `--task-type ref2va` preserves the one-partition
behavior these recipes rely on. The full capacity tables and both serve
commands live in the upstream
[RTX 5090 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-5090.md).

One interface note, because this area moved after the PR merged: the offload
family has been unifying under
[RFC #6648](https://github.com/vllm-project/vllm-omni/issues/6648).
[#5929](https://github.com/vllm-project/vllm-omni/pull/5929) (merged
2026-09-05) introduced one grammar —
`--diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"],"layer_options":{"dit":{"weight_transfer":"rank-local","resident_layers":20}}}'` —
where `components` selects what moves, `mode` picks module-swap vs
layer-streaming, and `weight_transfer` picks rank-local vs AllGather; the
legacy DLO flags used above (`--enable-distributed-layerwise-offload`,
`--dlo-no-use-allgather`, `--dlo-resident-layers`) remain documented
compatibility aliases with identical behavior.
[#7209](https://github.com/vllm-project/vllm-omni/pull/7209) (merged
2026-09-09) then centralized topology resolution behind one pure
`resolve_offload_plan()` — same accepted configurations, but invalid ones now
fail before any component is moved rather than midway. For MiniMax-H3's
full-topology two-GPU recipes, the upstream
[user guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md)
still points at the compatibility flags used here, and Host Weight Runtime
cannot yet be expressed through the new config at all — the commands above
match current upstream recipes either way.

## How to choose {#decision-cards}

{% include decision-cards.html items=page.decisions %}

## Limitations and follow-ups {#limitations}

- **This is capacity, not speed.** 8 min 38 s for a five-second 50-step video
  is the streaming price; lower resident counts reduce HBM further and
  increase CPU-to-GPU transfer time. No A/B latency benchmark exists.
- **The 24 GB (`rtx4090`) profile is a capacity proxy** — validated on B300
  allocation runs, not on target 4090 hardware, and starting from 1024×576.
- **Host RAM is the real footprint.** 200 GiB minimum per partition; resident
  layers keep pinned CPU masters, so the knob trades HBM, not host memory.
  Multiple independent engines on one host should look at
  [Host Weight Runtime]({{ site.baseurl }}/2026-08-26-understanding-pr-6591-host-weight-runtime/).
- **No mmap fast-start under DLO for H3.** Until checkpoint mmap can apply the
  grouped-QKV and fused-MLP transforms, H3 uses the ordinary loader — startup
  includes materializing weights before sharding.
- **Eager execution + cuDNN attention** are part of the validated consumer
  path; the configuration is deliberately conservative.
- **Single-run validation only**, with sampled (not allocator) memory peaks —
  treat every number above as a boundary condition, not a distribution.
- This is Blog 4 in the 8-post MiniMax-H3 optimization series tracked by
  [series RFC #37](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/37);
  it is the core post of the "offload" topic, flanked by the
  [Host Weight Runtime post]({{ site.baseurl }}/2026-08-26-understanding-pr-6591-host-weight-runtime/) (host-memory sharing) and the
  [online FP8 + DLO post]({{ site.baseurl }}/2026-08-19-pr-6279-dlo-online-fp8-allgather/) (quantized payload).
  Related reads: [Blog 1 — the modular pipeline]({{ site.baseurl }}/2026-08-24-understanding-pr-5720-minimax-h3-modular-pipeline/),
  [Blog 2 — the "four steps" contracts]({{ site.baseurl }}/2026-08-24-understanding-pr-5991-minimax-h3-few-step-schedules/),
  [Blog 3 — mask-free packed padding]({{ site.baseurl }}/2026-09-12-understanding-pr-6542-trtllm-mask-free-packed-padding/).

## References {#references}

- [PR #5764 — feat(minimax-h3): enable RTX 4090/5090 support with DLO](https://github.com/vllm-project/vllm-omni/pull/5764) (merged 2026-08-06, commit [`1c2a81f`](https://github.com/vllm-project/vllm-omni/commit/1c2a81f6d84aea4fff53bd2f894c2a287c237245))
- [PR #5802 — Fix DLO AllGather size mismatch for heterogeneous blocks](https://github.com/vllm-project/vllm-omni/pull/5802) (merged 2026-08-05)
- [PR #5864 — Fix DLO DP concurrent request execution](https://github.com/vllm-project/vllm-omni/pull/5864) (merged 2026-08-08)
- [MiniMax-H3 RTX 5090 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-5090.md) (upstream, current)
- [MiniMax-H3 recipe hub](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md) (upstream, current)
- [All-task 2-GPU runner](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/minimax_h3/run_h3_2gpu_all_tasks.sh) (upstream, current)
- [DLO user guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md) (upstream, current — documents both the new `diffusion_offload_config` and the compatibility flags)
- [RFC #6648 — Unify the offloader protocol and user interface](https://github.com/vllm-project/vllm-omni/issues/6648) (open; J0 = #5929 and J1 = #7209 merged)
- [Host Weight Runtime post — PR #6591]({{ site.baseurl }}/2026-08-26-understanding-pr-6591-host-weight-runtime/)
- [Online FP8 with DLO AllGather post — PR #6279]({{ site.baseurl }}/2026-08-19-pr-6279-dlo-online-fp8-allgather/)
- [Series RFC #37 — MiniMax-H3 optimization blog series](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/37)
