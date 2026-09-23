---
layout: post
title: "Understanding PR #6094 — Scheduler-managed diffusion KV cache control plane"
date: 2026-08-16 12:00:00 +0800
author: hsliuustc0106
summary: >-
  The diffusion stage gains a Scheduler-side KV control plane: workers export
  rank-local attention geometry, the engine builds native vLLM cache configs
  from a profiled budget, and CFG groups reserve blocks atomically with
  backpressure — the dense default is untouched.
description: >-
  The diffusion stage gains a Scheduler-side KV control plane: workers export rank-local attention geometry, the engine builds native vLLM cache configs from a profiled budget, and CFG groups reserve blocks atomically with backpressure — the dense default is untouched.
tags: [hunyuan-image3, kv-cache]
category: PR Analysis
---

## TL;DR

**PR #6094 (RFC #5244, part 2 of N) lands the control plane for
Scheduler-managed paged KV in the diffusion stage: Worker ranks export native
`FullAttentionSpec` geometry from the attention modules they actually loaded,
the Engine derives native vLLM `KVCacheConfig`s from a profiled memory budget,
and the Scheduler reserves blocks for every CFG group atomically — rolling
back, backpressuring, or failing single requests instead of crashing the
engine. The dense default path is untouched.** On HunyuanImage3 (2×A100 80GB,
TP=2 with expert parallel) model memory is byte-identical and E2E timing moves
within ~1% — the numbers establish *no regression*, not a speedup.

| Metric | Before (isolated `main`) | After (PR) | Δ | Setup |
|--------|--------------------------|------------|---|-------|
| E2E wall | 1942.7 ms | 1922.9 ms | −1.0% | HunyuanImage3, 2×A100 80GB, TP=2 + expert parallel, online FP8/Marlin, `TORCH_SDPA`, 512×512, 4 denoise steps |
| Stage generation | 1941.7 ms | 1921.9 ms | −1.0% | same |
| Model memory | 42.1744 GiB | 42.1744 GiB | 0 | same |

All three rows are the author-reported single-request smoke from the PR body;
the PR explicitly makes no performance claim. What the table buys is confidence
that wiring a whole control plane into startup and scheduling did not disturb
the default path.

## Background

**User-visible symptom:** the diffusion stage had no KV capacity management.
Serving HunyuanImage3 under load, there is no way to say "this much HBM is for
KV, admit requests until it's full, queue the rest." Concurrency is bounded by
guesswork; a request too large to ever serve sits in the queue indistinguishably
from one that merely needs to wait; and there is no foundation for the features
RFC #5244 wants next — sharing a conditioning prefix between requests, or
moving AR-talker KV into the DiT cache by page transfer.

**Technical cause:** diffusion attention modules own dense, per-layer KV
tensors allocated inside the pipeline. Nothing exports their geometry
(block size, KV heads, dtype, causal semantics) in a form a scheduler can do
arithmetic on, so the scheduler cannot account for KV at all. Building a
diffusion-only block pool and refcount system to fix that would fork the whole
cache lifecycle — exactly what the RFC decided to avoid.

**The bet instead:** reuse native vLLM 0.27's KV machinery end to end —
`KVCacheSpec`/`FullAttentionSpec`, `get_kv_cache_configs()`,
`KVCacheManager` + `BlockPool` — and keep Omni's additions to a thin,
diffusion-semantics-only facade: what a *CFG group* means (allocate all
sequences or none), what *prefix* vs *target* means (retained vs rewritten
spans inside one reservation), and how allocations travel to Workers.

## What the PR does

Three flows. The diagram is the reviewer's understanding of the head this post
describes (`c5f6159`):

![PR #6094 control plane overview]({{ site.baseurl }}/assets/figures/pr-6094-kv-control-plane/overview.png)

**Initialization — before the Scheduler exists.** Each Worker's attention
modules are now *opt-in* markers: `Attention(paged_kv_cache_role="primary")`
reports a native `FullAttentionSpec` with the rank-local KV heads it actually
holds (TP-correct by construction, since geometry comes from loaded modules
rather than being re-derived in the Scheduler). Two RPCs flow up: the specs,
and a profiled KV budget — Workers run the model's maximum advertised serving
shape (for HunyuanImage3: 1024×1024, CFG enabled, three reference images) under
native `memory_profiling`, so the budget already subtracts the activation peak.
The Engine then calls the native sizing chain
(`get_kv_cache_configs` → `generate_scheduler_kv_cache_config` →
`resolve_kv_cache_block_sizes`), which equalizes `num_blocks` to the smallest
rank, auto-fits `max_model_len = -1` down to what fits, and pushes each rank's
config back down. The Scheduler wraps the resulting config in
`DiffusionKVCacheManager` — a facade over the native `KVCacheManager` with
`enable_caching=False`, owning no second pool.

**Admission — every `schedule()` call.** `reserve_request` validates the CFG
group (contiguous sequence ids, `seq_len ≤ max_model_len`, no cross-attention
contexts yet), applies a deterministic never-fit check — the summed per-sequence
block requirement against the empty-pool baseline, independent of current load —
then allocates each sequence with native
`allocate_slots(full_sequence_must_fit=True)`, which is atomic: a `None`
return attaches zero blocks. Three outcomes: **backpressure** (pool busy —
return `None`, the request waits at the FIFO head and retries), **never-fit**
(upfront `DiffusionKVAdmissionError` — that one request finishes with an error
on its stream; the busy loop and the rest of the queue keep running), or
**success** (a `DiffusionKVMetadata` with per-sequence lengths and block ids).
Requests are also gated against the startup profile envelope — CFG count, max
`seq_len`, max `target_len` — so a request larger than what was profiled is
rejected with an actionable message instead of silently oversubscribing the
activation budget the pool was sized from.

Each sequence reserves its full first-step `seq_len`; the prefix/target split
describes spans *inside* that reservation:

<div style="border:1px solid #d0d7de; border-radius:8px; padding:12px; font-family:monospace; font-size:12.5px; text-align:center; line-height:1.7;">
  |&#8212;&#8212;&#8212;&#8212;&#8212;&#8212;&#8212;&#8212; seq_len (reserved) &#8212;&#8212;&#8212;&#8212;&#8212;&#8212;&#8212;&#8212;|<br>
  |&#8212; prefix_len (retained across denoise steps) &#8212;|&#8212; target_len (rewritten each step) &#8212;| suffix
</div>

For HunyuanImage3 the prefix is text plus reference-image tokens — identical at
every step — and the target is the generated latents. The metadata rides inside
the `NewRequestData` envelope to the Worker on both executor paths, including
the DLO+AllGather wave, where each DP rank picks a whole envelope so a request
and its block table can never separate.

**Terminals.** finish / cancel / error / `pop_request_state` / `close()` all
funnel into one idempotent native `free()` (reverse order, tail blocks first).
The scheduler is the only allocation lifecycle owner; workers never free.

## Key changes

- `vllm_omni/diffusion/diffusion_kv/initialization.py` (new) — the startup
  chain; returns the scheduler-side `KVCacheConfig`, block sizes, and the
  mutated `vllm_config` for the scheduler to consume.
- `vllm_omni/diffusion/diffusion_kv/manager.py` (new) — the facade:
  validation, the deterministic never-fit block-sum, the atomic CFG loop,
  rollback, generation counters, and terminal frees.
- `vllm_omni/diffusion/vllm_config.py` (new) — one shared builder for the
  native config views used by both the Engine and Workers (replacing the
  worker-local copy, and fixing `is_quantized` to match upstream's property
  semantics).
- `vllm_omni/diffusion/attention/layer.py` — the opt-in spec surface:
  `paged_kv_cache_role` / `paged_kv_cache_dtype`, and `get_kv_cache_spec()`
  returning `FullAttentionSpec` under the current-config context.
- `vllm_omni/diffusion/sched/base_scheduler.py` — reserve-on-admit with
  backpressure, admission errors surfaced as terminal per-request errors in the
  same `schedule()` output, and KV frees on every terminal path.
- `vllm_omni/diffusion/executor/` — three new control-plane RPCs
  (`get_kv_cache_specs`, `determine_available_kv_memory`,
  `set_kv_cache_configs`), and the DLO wave now sends whole `NewRequestData`
  envelopes.
- `vllm_omni/diffusion/worker/` — spec/memory discovery with rank-gathered
  failure containment (a failing rank cannot strand peers on the collective),
  and the paged-mode startup memory snapshot.
- `vllm_omni/diffusion/models/hunyuan_image3/` — `ImageKVCacheManager`
  becomes an `nn.Module` so its attention is discoverable, and opts in with
  bf16 KV.

## Measured impact

The table in the TL;DR is the whole quantitative story, and it is the right one
for a control-plane PR: **the dense default didn't move** (author-reported;
2×A100 80GB, TP=2 + EP, online FP8/Marlin, single-request smoke, byte-identical
memory). Test evidence: the diffusion KV/scheduler/worker/executor/Hunyuan
suite (282 passed, 2 deselected) and the config/engine suite (166 passed,
1 skipped) at the PR head, covering native conformance, atomic rollback,
capacity backpressure, metadata propagation, multi-rank failure handling, and
release on every terminal path. No paged *execution* exists yet, so there are
no paged performance numbers to report — that is the data-plane PR's job.

## How to use it

Nothing changes by default — `dense_legacy` skips every step above. The mode is
opt-in per diffusion stage:

```yaml
# stage engine args for the diffusion stage
diffusion_kv_mode: paged_scheduler
kv_cache_memory_bytes: null   # or an explicit byte budget to skip auto-sizing
gpu_memory_utilization: 0.9
max_model_len: -1             # auto-fit the admission bound to the pool
```

Today this is an integration surface, not a user feature: only HunyuanImage3
exports attention specs, and the physical data plane (cache tensors, block
tables, paged attention execution) lands in the follow-up PRs — the pool is
accounting until then. Other models fail fast at startup with "no cache-enabled
Attention modules" rather than running silently dense.

## Limitations & follow-ups

- **Deferred by design:** physical `kv_caches` / `BlockTable` / slot mapping,
  paged attention execution, prefix hashing and CFG prefix sharing,
  `DiffusionKVContext` cross-attention roles (rejected at admission for now),
  connector delayed-free, and AR→DiT page transfer.
- **Attention coverage:** `FullAttentionSpec` only, non-causal, self-attention
  KV; the backend `indexes_kv_by_block_stride` flag (default `False`) is the
  seam where paged kernels plug in.
- **Review round-trip:** the first review round's design questions (never-fit
  heuristic, profile-shape oversubscription, `≤ 0` budget messaging) were all
  addressed in follow-up commits at `c5f6159`; the shared-understanding diagram
  is posted [on the PR](https://github.com/vllm-project/vllm-omni/pull/6094#issuecomment-5305602538).
- **Next:** the data-plane PRs of RFC #5244 are where prefix/target actually
  starts saving compute and memory.

## References

- [PR #6094 — Add native KV cache initialization and Scheduler-managed block allocation](https://github.com/vllm-project/vllm-omni/pull/6094)
- [RFC #5244 — Scheduler-managed diffusion KV cache](https://github.com/vllm-project/vllm-omni/issues/5244)
- [Diagram + SVG source in this cookbook](https://github.com/hsliuustc0106/vllm-omni-cookbook/tree/main/docs/visualizations)
- [Stage config reference (docs/configuration/stage_configs.md, PR branch)](https://github.com/vllm-project/vllm-omni/pull/6094/files)
