---
layout: post
title: "Understanding diffusion sequence parallelism — Ulysses, Ring, Hybrid and AllGather-KV"
date: 2026-08-17 12:00:00 +0800
author: hsliuustc0106
summary: >-
  How vLLM-Omni shards diffusion token sequences across GPUs: Ulysses'
  head-for-sequence all-to-all swap, Ring attention's P2P K/V lap hidden
  under compute, the hybrid composition, and the AllGather-KV shortcut.
tags: [sequence-parallelism, multi-GPU]
category: PR Analysis
feature: parallelism
math: true
usage:
  - label: "Ulysses"
    blurb: "big images, NVLink"
    title: "Ulysses-SP · online serving (>= 2 GPUs)"
    code: |
      vllm serve Qwen/Qwen-Image --omni --port 8091 --usp 2
  - label: "UAA"
    blurb: "head count not divisible"
    title: "advanced_uaa · e.g. Z-Image-Turbo (30 heads)"
    code: |
      vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8091 --usp 4 --ulysses-mode advanced_uaa
    note: >-
      Experimental; in hybrid mode it adds the equal-global-S ring constraint
      plus a K/V-head-padding restriction.
  - label: "Ring"
    blurb: "very long sequences"
    title: "Ring · 2 GPUs"
    code: |
      vllm serve Qwen/Qwen-Image --omni --port 8091 --ring 2
  - label: "Hybrid"
    blurb: "compose u×r"
    title: "Hybrid · Ulysses 2 × Ring 2 = 4 GPUs"
    code: |
      vllm serve Qwen/Qwen-Image --omni --port 8091 --usp 2 --ring 2
  - label: "Python API"
    blurb: "offline Omni"
    title: "Omni · DiffusionParallelConfig"
    code: |
      from vllm_omni import Omni
      from vllm_omni.diffusion.data import DiffusionParallelConfig

      omni = Omni(model="Qwen/Qwen-Image",
                  parallel_config=DiffusionParallelConfig(ulysses_degree=2, ring_degree=2))
decisions:
  - when: "Big images on NVLink"
    pick: "Ulysses"
    why: "Full Q/K/V/O moves twice per layer (3 all-to-all + 1 reverse); kernel sees full sequence at H/P heads."
  - when: "Very long sequences, tight memory"
    pick: "Ring"
    why: "Q never moves; K/V take one P2P hop per step, overlapped with compute — 2 P2P x (P-1) steps per layer."
  - when: "Beyond ~8 GPUs"
    pick: "Hybrid u×r"
    why: "Ulysses set first, then a K/V ring lap — avoids oversharding heads on big nodes."
  - when: "Small K/V, cheap fabric"
    pick: "AllGather-KV"
    why: "Two all-gathers; K/V moves once, Q stays local — no divisibility constraints."
  - when: "Short sequences (below ~1024px)"
    pick: "Stay single-GPU"
    why: "Ring loop overhead dominates short sequences — the guide's own advice."
  - when: "Heads not divisible by u"
    pick: "advanced_uaa"
    why: "Ulysses-mode workaround (e.g. 30 heads on Z-Image-Turbo); experimental."
  - when: "Layerwise CPU offload"
    pick: "SP not available"
    why: "SP does not compose with layerwise CPU offloading; HSDP/Expert-Parallel combinations are unverified."
---

## TL;DR

**vLLM-Omni's diffusion stack keeps the token sequence sharded across GPUs for the
whole forward pass and pays a communication tax only inside attention, where four
pluggable strategies — Ulysses, Ring, Hybrid ×Ring and AllGather-KV — differ purely
in *how* they move Q/K/V between ranks.** Everything else in a DiT block (norms,
MLPs, projections) is per-token work that simply runs on an `S/P` shard. The
upstream user guide reports **1.5×–3.6× speedups on large images and videos at 2–8
GPUs**; those are upstream-doc numbers, not yet independently measured in this
cookbook.

| Claim | Value | Evidence |
|---|---|---|
| Speedup, large images/videos, 2–8 GPUs | 1.5×–3.6× | [upstream SP guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/parallelism/sequence_parallel.md) (not cookbook-measured) |
| SP world size | `ulysses_degree × ring_degree` (or `allgather_degree` alone) | [parallel_state.py](https://github.com/vllm-project/vllm-omni/blob/c588208cc/vllm_omni/diffusion/distributed/parallel_state.py) at `c588208cc` |
| Models with SP wired | 14 image + 9 video families | [diffusion_features.md](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion_features.md) |

An [interactive explainer]({{ site.baseurl }}/assets/figures/diffusion-sequence-parallelism/diffusion-sequence-parallelism-design.html)
accompanies this post — every figure below is a snapshot of one of its live
diagrams (phase steppers, an animated ring, a topology painter).

## Background

The symptom is sequence length. A diffusion transformer turns its input into
patches: a 2048×2048 image at typical patch sizes becomes tens of thousands of
tokens, and a few seconds of video an order of magnitude more. Attention cost
grows quadratically in that count, and a KV buffer sized for the full sequence
fits fewer and fewer denoising requests per GPU. Users see OOMs on
high-resolution jobs and step latencies that make 50-step samplers crawl.

The structural fact that makes this tractable: **attention is the only
sequence-global operation in a DiT block.** Norms, feed-forwards, modulation and
the QKV/O projections all process tokens independently. So if the sequence is
sharded `S/P` across `P` GPUs, everything except attention runs unchanged on a
quarter-length sequence — you only need a protocol for attention to exchange
the right shards at the right time. Sequence parallelism (SP) is that protocol;
vLLM-Omni calls it SP where diffusers calls the same idea Context Parallelism
(CP).

## The design: strategies are plug-ins around the kernel

Two pieces keep model code clean:

1. A declarative [`_sp_plan`](https://github.com/vllm-project/vllm-omni/blob/c588208cc/vllm_omni/diffusion/distributed/sp_plan.py)
   on each transformer — split inputs like `hidden_states` along the sequence
   dim before `forward()`, gather after `proj_out`. No manual sharding in the
   model body; 19 transformer files declare one today.
2. A [`ParallelAttentionStrategy`](https://github.com/vllm-project/vllm-omni/blob/c588208cc/vllm_omni/diffusion/attention/parallel/base.py)
   selected per-forward, with `pre_attention()` / `post_attention()` hooks that
   wrap **whatever kernel backend the model picked** — FA2/FA3/FA4, AITER,
   SDPA, TRT-LLM. Strategy and kernel are orthogonal axes:

```python
# attention/parallel/factory.py — selection order
if allgather_degree > 1: return AllGatherKVParallelAttention(...)
if ulysses_degree   > 1: return UlyssesParallelAttention(...)   # hybrid if ring>1
if ring_degree      > 1: return RingParallelAttention(...)
return NoParallelAttention()
```

## Group topology: one SP group, factored two ways

`sp_size = ulysses_degree × ring_degree`. At init,
[`set_seq_parallel_pg()`](https://github.com/vllm-project/vllm-omni/blob/c588208cc/vllm_omni/diffusion/distributed/parallel_state.py)
splits each SP group into an **Ulysses subgroup** and a **Ring subgroup**. With
the default `use_ulysses_low=True`, Ulysses groups are *contiguous rank chunks*
and Ring groups are *strided* — with `u=2, r=2` over ranks 0–3: Ulysses groups
{0,1}, {2,3} and Ring groups {0,2}, {1,3}. This factorization is exactly what
hybrid mode composes, and what pure mode degenerates to (`r=1` → one all-to-all
group; `u=1` → one ring of everything).

![Figure 1 — the topology painter: dashed cyan outlines are contiguous Ulysses subgroups, violet arrows the strided ring groups; presets flip between u2×r2, u4×r1 and u1×r4]({{ site.baseurl }}/assets/figures/diffusion-sequence-parallelism/fig1-topology.png)

## Ulysses: trade your sequence shard for a head shard

Each rank enters attention with `(B, S/P, H, D)` — a slice of tokens, *all*
`H` heads. One `dist.all_to_all_single` per tensor swaps the two dimensions:
rank *i* sends its token slice to every peer and receives the slices matching
its head subset, leaving `(B, S, H/P, D)` — the *full* sequence, a fraction of
the heads. Attention is then an ordinary local kernel over a
standard-shape input, and a reverse all-to-all restores the sequence shard.
Four collectives per attention layer: Q, K, V forward plus the output reverse
([`all_to_all_4D`](https://github.com/vllm-project/vllm-omni/blob/c588208cc/vllm_omni/diffusion/distributed/comm.py)).

Concretely, the all-to-all moves **P pieces per rank, per tensor**. Rank *i*'s
local `(S/P, H)` block is cut along heads into P groups, and **piece k —
rank *i*'s tokens × head group k — goes to rank k**. Symmetrically, rank *i*
receives from every rank *r* that rank's tokens × **rank *i*'s own head
group**. Stacking the P received pieces along the sequence axis is exactly
what turns `(S/P, H)` into `(S, H/P)`: after the swap, rank *i* owns every
token but only heads `[i·H/P, (i+1)·H/P)`.

![Figure 2 — top: the phase stepper, rank 0 going from S/4 tokens × 8 heads (phase 0) to the full stacked sequence × 2 heads after the all-to-all (phase 1), local kernel (phase 2), reverse (phase 3). Bottom: the send matrix — cell (r, k) is the one piece rank r sends to rank k (r's tokens × k's head group); rank 0's highlighted row is its four sends, the green-dashed column is what it receives back and stacks into the full sequence]({{ site.baseurl }}/assets/figures/diffusion-sequence-parallelism/fig2-ulysses.png)

Text conditioning rides along differently: joint Q/K/V are replicated on all
ranks, sliced **by heads** per Ulysses rank, concatenated to the image stream
only *after* the exchange, and the joint slice of the output is recombined with
a head-dimension `all_gather` on the way out.

Strict mode requires `H % P == 0` and evenly divisible shards. The experimental
`ulysses_mode="advanced_uaa"` lifts both — it all-gathers the per-rank sequence
lengths and passes them as `output_split_sizes` to the all-to-all, and pads the
head dim only inside the exchange. That is what lets Z-Image-Turbo's 30 heads
run `ulysses_degree=4`.

## Ring: Q never moves; K/V take a lap while you compute

No resharding at all. Each rank keeps its query shard and one K/V block; the
kernel rotates K/V blocks around the ring with P2P
[`isend/irecv`](https://github.com/vllm-project/vllm-omni/blob/c588208cc/vllm_omni/diffusion/distributed/comm.py),
posting the transfer **before** calling attention so the communication hides
under compute:

```python
# attention/backends/ring_pytorch_attn.py — the whole idea
for step in range(comm.world_size):
    if step + 1 != comm.world_size:
        next_k = comm.send_recv(k)          # isend → next, irecv ← prev
        next_v = comm.send_recv(v)
        comm.commit()                        # batch_isend_irecv
    block_out, block_lse = attn(q_local, k, v)   # overlap with transfer
    out, lse = update_out_and_lse(out, lse, block_out, block_lse)
    if step + 1 != comm.world_size:
        comm.wait(); k, v = next_k, next_v
```

Per-step partial outputs merge by online softmax — each rank keeps a running
`(out, lse)` pair in fp32 and folds every block into it:

$$
\text{out} \leftarrow \text{out}\cdot e^{\,\text{lse}-m} + \text{block\_out}\cdot e^{\,\text{lse}_b-m},
\qquad m = \max(\text{lse},\ \text{lse}_b)
$$

![Figure 3 — the animated ring: Q blocks (dashed, stationary) watch K/V chips rotate one hop per step; the right ledger tracks rank 0's (out, lse) accumulation; the amber prefix is the static text K/V concatenated at step 0 only]({{ site.baseurl }}/assets/figures/diffusion-sequence-parallelism/fig3-ring.png)

Three details worth knowing: joint/text K/V never rotate — they concatenate as
a static prefix at step 0 only (`joint_strategy="rear"` + causal is rejected
because the mask would hide the prefix); causal attention works by skipping
steps where `step > rank`; and the ring kernels pick FA4 on Blackwell, FA3
elsewhere on NVIDIA, AITER on ROCm, with a PyTorch-SDPA fallback (forced for
fp32). The output needs no post-processing — it is already sequence-sharded,
which is why `post_attention` is a no-op.

## Hybrid and AllGather-KV

**Hybrid ×Ring** runs the Ulysses all-to-all *first* inside each contiguous
subgroup — every rank now holds the full sequence for its head subset — and the
ring then rotates those full-sequence K/V tensors across the strided groups.
One rule keeps it honest: ring P2P buffers are fixed-shape, so every rank in a
ring group must end up with the same post-Ulysses `S_global`; UAA validates
this up front and raises instead of hanging the ring.

**AllGather-KV** is the minimalist alternative: `all_gather_into_tensor`
rebuilds full K/V on every rank, each rank attends with its local `S/P`
queries, and the mask/query-ranges are sliced to its window. Q never moves, so
there is no reverse step at all — two collectives per layer, non-causal only,
mutually exclusive with the other two modes.

![Figure 4 — hybrid (top): all-to-all inside {0,1}/{2,3}, then the ring lap on {0,2}/{1,3}; AllGather-KV (bottom): the gather bus rebuilds full K/V while every Q stays local]({{ site.baseurl }}/assets/figures/diffusion-sequence-parallelism/fig4-hybrid-allgather.png)

## Cheat sheet

| | Ulysses | Ring | Hybrid u×r | AllGather-KV |
|---|---|---|---|---|
| Collectives / layer | 3 all-to-all + 1 reverse | 2 P2P × (P−1) steps, overlapped | Ulysses set + ring lap | 2 all-gathers |
| What moves | full Q/K/V/O, twice | K/V, one hop/step | heads, then K/V lap | K/V once; Q local |
| Kernel sees | full S, H/P heads | S/P Q vs rotating blocks | full S, H/u heads | S/P Q vs full K/V |
| Constraints | H, S divisible by u (or UAA) | equal shards per ring peer | both, + equal `S_global` | — |
| Best for | big images, NVLink | very long sequences, tight memory | >8 GPU scale-out | small K/V, cheap fabric |

## How to use it

Pick a topology; commands are copy-ready:

{% include usage-cookbook.html modes=page.usage %}

GPU budget: `ulysses × ring × cfg × tp`. SP composes with TeaCache/Cache-DiT,
CFG-Parallel, TP and VAE patch-parallel; it does **not** compose with layerwise
CPU offloading, and HSDP/Expert-Parallel combinations are unverified.

## How to choose

The cheat sheet above is the full reference; the short version:

{% include decision-cards.html items=page.decisions %}

## Limitations & follow-ups

- The 1.5×–3.6× figures are upstream-doc claims; no cookbook ledger has
  measured SP on tracked models yet — an obvious follow-up retro.
- Not every model is wired: FLUX.1 family, GLM-Image, HunyuanImage3, Krea 2,
  SD3.5 and others are marked unsupported in the
  [feature matrix](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion_features.md).
  GLM-Image and HunyuanImage3 transformers even declare an `_sp_plan`, so the
  scaffolding exists — wiring and validation are the missing work.
- `advanced_uaa` is experimental, and in hybrid mode it adds the equal-`S_global`
  ring constraint plus a K/V-head-padding restriction.
- Ring loop overhead dominates short sequences — the guide's own advice is to
  stay single-GPU below ~1024px.

## References

- [Sequence Parallelism guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/parallelism/sequence_parallel.md) and [feature matrix](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion_features.md) (upstream docs)
- Strategy sources at `c588208cc`: [factory.py](https://github.com/vllm-project/vllm-omni/blob/c588208cc/vllm_omni/diffusion/attention/parallel/factory.py) · [ulysses.py](https://github.com/vllm-project/vllm-omni/blob/c588208cc/vllm_omni/diffusion/attention/parallel/ulysses.py) · [ring.py](https://github.com/vllm-project/vllm-omni/blob/c588208cc/vllm_omni/diffusion/attention/parallel/ring.py) · [allgather_kv.py](https://github.com/vllm-project/vllm-omni/blob/c588208cc/vllm_omni/diffusion/attention/parallel/allgather_kv.py)
- Communication primitives: [comm.py](https://github.com/vllm-project/vllm-omni/blob/c588208cc/vllm_omni/diffusion/distributed/comm.py) (`all_to_all_4D`, `RingComm`) · [parallel_state.py](https://github.com/vllm-project/vllm-omni/blob/c588208cc/vllm_omni/diffusion/distributed/parallel_state.py) · [sp_plan.py](https://github.com/vllm-project/vllm-omni/blob/c588208cc/vllm_omni/diffusion/distributed/sp_plan.py)
- Papers: [DeepSpeed Ulysses](https://arxiv.org/pdf/2309.14509) · [Ring Attention](https://arxiv.org/abs/2310.01889)
- [Interactive explainer]({{ site.baseurl }}/assets/figures/diffusion-sequence-parallelism/diffusion-sequence-parallelism-design.html) (this post's figures, live)
