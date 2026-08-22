---
layout: post
title: "Understanding PR #6213 — Loader-owned host-weight plans for DLO"
date: 2026-08-16 12:00:00 +0800
author: hsliuustc0106
summary: >-
  The diffusion loader now preflights the checkpoint and hands DLO one exact
  host-weight plan, so no-AllGather workers share checkpoint pages instead of
  private copies — node PSS −47% on a two-worker MiniMax-H3 node.
tags: [MiniMax-H3]
category: PR Analysis
feature: offloader
math: true
usage:
  - label: "Serve · DP2 no-AG"
    blurb: "the measured sharing config"
    title: "MiniMax-H3 · DP2 + no AllGather"
    code: |
      vllm serve /path/to/MiniMax-H3/FL2VA --omni \
        --enable-distributed-layerwise-offload \
        --data-parallel-size 2 \
        --dlo-no-use-allgather
    note: >-
      The loader picks direct mmap when the preflight proves the layout
      compatible and silently falls back to the ordinary loader otherwise.
  - label: "Offline · lifecycle"
    blurb: "lifecycle-managed run"
    title: "dlo_lifecycle.py · dlo-dp2-no-allgather"
    code: |
      python examples/offline_inference/minimax_h3/dlo_lifecycle.py \
        --mode dlo-dp2-no-allgather
    note: >-
      The repo's lifecycle-managed offline entry point for the same
      DP2 no-AllGather configuration.
decisions:
  - when: "TP=1 · several workers per node"
    pick: "DP + no-AllGather"
    why: "Workers keep file-backed checkpoint views and share pages through the OS page cache — the measured −47% node-PSS configuration."
  - when: "TP>1 · HSDP · online quantization"
    pick: "Ordinary-loader fallback"
    why: "The preflight fails closed before any mutation; DLO still runs, workers just hold private runtime weights."
  - when: "AllGather mode"
    pick: "Nothing changes"
    why: "The weight group already totals ≈ one persistent model copy in private shards; the source mapping is released after shard preparation."
  - when: "Measuring node memory"
    pick: "Sum PSS, not RSS"
    why: "RSS counts a shared page in every worker; node-level comparisons should use summed PSS."
---

## TL;DR

**The diffusion model loader is now the single owner of checkpoint semantics: it
preflights the checkpoint, builds one exact `HostWeightPlan`, and hands it to the
distributed layerwise offload (DLO) backend. Host storage and the transfer
protocol became independent decisions, so no-AllGather workers keep file-backed
views of the checkpoint and share its pages through the OS page cache instead of
holding private full-model copies.** On a two-worker MiniMax-H3 node this cut
node memory nearly in half, with byte-identical model output.

| Metric | Before (ordinary loader) | After (direct mmap) | Δ | Setup |
|--------|--------------------------|---------------------|---|-------|
| Two-worker node PSS | 283.56 GiB | 150.08 GiB | **−47.1%** | MiniMax-H3 FL2VA, BF16, one L20X node, DP=2 TP=1, no AllGather |
| Worker-0 PSS | 167.84 GiB | 101.43 GiB | −66.40 GiB | same |
| Worker-1 PSS | 115.73 GiB | 48.64 GiB | −67.09 GiB | same |

## Background

**User-visible symptom:** running multiple replicas of a large diffusion model on
one machine — pure data parallelism with per-worker layerwise offload — scales
host memory linearly with the replica count. A node that fits two MiniMax-H3
workers' device memory comfortably can still exhaust host RAM, because every
worker keeps its own complete copy of the weights even though the copies are
byte-identical:

$$
\text{private host memory} \;\approx\; \text{DP size} \times \text{model size}
$$

**Technical cause:** DLO's no-AllGather mode deliberately avoids a weight
collective — each rank streams complete blocks with H2D copies only — but its host
backing was coupled to that choice: without a collective, each process fell back
to the ordinary loader's private runtime tensors. And the machinery that decided
whether weights could be loaded as checkpoint `mmap` views was split between two
gates — one in the loader, one in the offload backend — kept in sync only by
comments. When those gates disagreed, a model could be left with parameters on
the `meta` device and no guard would fire.

## What the PR does

One owner, one plan, two independent decisions. The loader runs a **preflight**
(topology, key mapping, coverage, shapes, dtypes, custom-loader compatibility)
and either produces a complete `HostWeightPlan` — in which case ordinary weight
materialization is skipped — or falls back to the ordinary loader *before any
mutation*. The backend consumes the exact plan or refuses to run; it never
rescans checkpoint files or consults pipeline capability flags.

<div style="border:1px solid #d0d7de; border-radius:8px; padding:12px; font-family:monospace; font-size:12.5px; text-align:center; line-height:1.7;">
  <b>storage</b> (loader decides): direct mmap plan &nbsp;or&nbsp; ordinary loader<br>
  &nbsp;×&nbsp;<br>
  <b>transfer</b> (backend decides): AllGather shard+collective &nbsp;or&nbsp; rank-local H2D
</div>

In the new **rank-local + mmap** cell, each worker retains immutable
`safetensors` views of the checkpoint and packs one block at a time through two
pinned host staging slots (bounded by the largest streamed block). Workers on the
same node that map the same files share physical checkpoint pages through the OS
page cache — `Shared_Clean` 62.45 GiB, each worker charged `Pss_File` 31.20 GiB.
An interactive walkthrough of all five diagrams — ownership flow, preflight
ladder, storage×transfer matrix, runtime pipeline, and the measured node picture
— is here:

> [!TIP]
> Open the [interactive design visualization]({{ site.baseurl }}/assets/figures/pr-6213-dlo-host-weight-plans/pr-6213-dlo-host-weight-plans.html)
> (self-contained page; the blog serves it directly).

## Key changes

- [`host_weight_plan.py`](https://github.com/vllm-project/vllm-omni/blob/9d2bb23ff6/vllm_omni/diffusion/model_loader/host_weight_plan.py) — the plan builder. Every unsupported layout fails closed with a reason:
  ```python
  if tensor_parallel_size != 1:
      return HostWeightPlanResult(None, f"TP={tensor_parallel_size} requires the ordinary loader")
  ```
  Duplicates and missing checkpoint bindings are rejected; shape and dtype must match per tensor; a tensor with a custom `weight_loader` needs an adapter policy.
- [`checkpoint_adapters/direct_mmap.py`](https://github.com/vllm-project/vllm-omni/blob/9d2bb23ff6/vllm_omni/diffusion/model_loader/checkpoint_adapters/direct_mmap.py) — loader-side proofs for the two models whose loaders transform layouts: MiniMax-H3's grouped-QKV reorder (argument-identical to the regular loader's call) and Cosmos3's identity-at-TP1 contract.
- [`diffusers_loader.py`](https://github.com/vllm-project/vllm-omni/blob/9d2bb23ff6/vllm_omni/diffusion/model_loader/diffusers_loader.py) — builds the plan and skips materialization only when one exists:
  ```python
  _skip_load = self.host_weight_plan is not None
  ```
  The transfer is one-shot (`take_host_weight_plan()`), and component weight sources outside the DiT keep loading ordinarily.
- [`distributed_layerwise_backend.py`](https://github.com/vllm-project/vllm-omni/blob/9d2bb23ff6/vllm_omni/diffusion/offloader/distributed_layerwise_backend.py) — realizes the plan: DiT tensors become `mmap` views, non-persistent buffers are saved and restored around the meta conversion, `post_load_weights`/`validate_loaded_weights` are preserved, and the rank-local path packs blocks through the two shared staging slots. Adapter transforms — including strided ones — are applied only while packing, never on the raw views.
- [`offload_plan.py`](https://github.com/vllm-project/vllm-omni/blob/9d2bb23ff6/vllm_omni/diffusion/offloader/offload_plan.py) and [`pipeline_minimax_h3.py`](https://github.com/vllm-project/vllm-omni/blob/9d2bb23ff6/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py) — the `_supports_mmap_loading` pipeline flag is deleted; models no longer carry DLO capability markers.
- Tests (~680 added lines) assert the risky invariants directly: storage-pointer retention, mutation-free fallback, the exact grouped-QKV permutation, bounded staging, strided-transform round-trips, and that a loader plan can never be silently dropped.

## Measured impact

All numbers from the PR's validation (two-worker MiniMax-H3 FL2VA, BF16, one
L20X node, DP=2 TP=1, no AllGather, `/proc/<worker>/smaps_rollup`); mmap workers
were sampled after one completed request — the conservative point, once the
checkpoint working set had faulted in.

| Metric | Before | After | Δ | Evidence |
|--------|--------|-------|---|----------|
| Two-worker node PSS | 283.56 GiB | 150.08 GiB | −133.48 GiB (−47.1%) | [PR #6213 description](https://github.com/vllm-project/vllm-omni/pull/6213) |
| Worker-0 / Worker-1 PSS | 167.84 / 115.73 GiB | 101.43 / 48.64 GiB | −66.40 / −67.09 GiB | same |
| Private_Dirty (w0 / w1) | 167.53 / 115.40 GiB | 70.24 / 17.44 GiB | ≈ −97–98 GiB each | same |
| Persistent host staging per worker | full model copy | 2 × 1.2313 GiB slots | bounded | same |
| Planned tensors realized (E2E) | — | 535/535, DiT materialization skipped | — | same |

PSS arithmetic cross-checks: per worker, `Private_Dirty + Pss_File` reproduces
the reported PSS to ±0.01 GiB. Output was byte-identical to the ordinary path —
generated video `[107, 256, 256, 3]` and audio `[1, 2, 142400]` — with peak
device memory 13,226 MiB and clean shutdown.

## How to use it

Nothing new to enable — correctness never depends on choosing a storage mode.
The loader picks direct `mmap` when the preflight proves the layout compatible,
and silently falls back to the ordinary loader (with an observable reason in the
logs) otherwise. Pick a mode; commands are copy-ready. For a lifecycle-managed
offline run, the repo also ships
[`dlo_lifecycle.py`](https://github.com/vllm-project/vllm-omni/blob/9d2bb23ff6/examples/offline_inference/minimax_h3/dlo_lifecycle.py).

{% include usage-cookbook.html modes=page.usage %}

> [!NOTE]
> An effective DLO group size of one performs no collective, even with
> `dlo_use_allgather=True` — it automatically takes the rank-local path.

## How to choose

{% include decision-cards.html items=page.decisions %}

## Limitations & follow-ups

- Direct `mmap` is proven for TP=1, non-HSDP, non-online-quantized layouts; everything else fails closed to the ordinary loader (TP>1 and HSDP keep working through it).
- Page-cache sharing is **node-local** — each node has its own page cache; there is no cross-node sharing.
- AllGather mode is unchanged: across its weight-transfer group it already totals ≈ one persistent model copy in private shards, and the source mapping is released after shard preparation.
- RSS overstates sharing (it counts a page in every mapping process); node-level comparisons should use summed PSS.
- Follow-up [#6231](https://github.com/vllm-project/vllm-omni/issues/6231) defines the Phase B runtime-cache compatibility contract — sharing *transformed* runtime layouts (TP coordinates, quantization) built through the ordinary loader.

## References

- [PR #6213 — Loader-owned host-weight plans for DLO](https://github.com/vllm-project/vllm-omni/pull/6213)
- [RFC #6195 — Decouple DLO host-weight storage from DP request scheduling](https://github.com/vllm-project/vllm-omni/issues/6195)
- [Issue #6231 — DLO runtime-cache compatibility across parallelism dimensions](https://github.com/vllm-project/vllm-omni/issues/6231)
- [DLO feature design](https://github.com/vllm-project/vllm-omni/blob/9d2bb23ff6/docs/design/feature/offloader/distributed_layerwise_offload.md) (upstream repo)
- [Interactive design visualization]({{ site.baseurl }}/assets/figures/pr-6213-dlo-host-weight-plans/pr-6213-dlo-host-weight-plans.html) — the served copy this blog owns; the [visualization index](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/docs/visualizations/README.md) links back to it
