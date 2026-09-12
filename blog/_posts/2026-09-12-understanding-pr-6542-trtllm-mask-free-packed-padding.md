---
layout: post
title: "Serving MiniMax-H3 in vLLM-Omni (3): mask-free packed padding — TRTLLM attention stops re-validating every layer (PR #6542)"
date: 2026-09-12 12:00:00 +0800
author: hsliuustc0106
summary: >-
  PR #6542 lets MiniMax-H3's producer publish the valid-token boundary on the
  host, so TRTLLM attention trims packed padding with no mask math and no CUDA
  syncs — the per-call gap drops from 499.7 µs to 0.35 µs p50 on 4x B300.
tags: [MiniMax-H3, TRTLLM]
category: PR Analysis
feature: host_path
lang: en
pair: /zh/2026-09-12-understanding-pr-6542-trtllm-mask-free-packed-padding/
usage:
  - label: "Serve"
    blurb: "TRTLLM attention on 4 GPUs — fast path is automatic"
    title: "vllm serve · MiniMax-H3 with TRTLLM_ATTN"
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
        --diffusion-attention-backend TRTLLM_ATTN \
        --task-type fl2va
    note: >-
      No new flag turns the fast path on: once the backend is TRTLLM_ATTN and
      the model publishes packed-padding metadata, every packed attention call
      runs mask-free. On datacenter Blackwell (B200/B300, SM100/SM103)
      TRTLLM_ATTN is already the auto-selected default.
  - label: "Fallback"
    blurb: "masks that are not structural padding"
    title: "vllm serve · mask-capable backend"
    code: |
      CUDA_VISIBLE_DEVICES=0,1,2,3 \
      vllm serve "${MODEL}" \
        --omni \
        --trust-remote-code \
        --num-gpus 4 \
        --usp 4 \
        --diffusion-attention-backend CUDNN_ATTN \
        --task-type fl2va
    note: >-
      The fast path only covers producer-owned [real, pad] suffix padding.
      TRTLLM_ATTN now rejects every nonempty attn_mask; for arbitrary
      per-token masks (block-sparse locality, tiled layouts) pick a
      mask-capable backend such as CUDNN_ATTN or TORCH_SDPA.
decisions:
  - when: "Serving H3 on datacenter Blackwell"
    pick: "Keep the TRTLLM_ATTN default"
    why: "It is auto-selected there, and #6542 fixes the packed-padding rejection from issue #6358 — the fast path runs mask-free."
  - when: "Your padding is always a [real, pad] suffix the producer built"
    pick: "TRTLLM_ATTN with packed-padding metadata"
    why: "The producer already knows the valid-token boundary on the host, so no mask is constructed and no device sync happens."
  - when: "You need arbitrary per-token masks"
    pick: "CUDNN_ATTN or TORCH_SDPA"
    why: "Mask-free is not general mask support: TRTLLM_ATTN rejects every nonempty attn_mask by design."
  - when: "Running with ring attention (--ring > 1)"
    pick: "Unchanged mask behavior"
    why: "Ring keeps aligned rows for fixed-size P2P buffers, so the mask-free path is not taken; the producer still builds the mask."
  - when: "Continuous batching mixes short and long requests"
    pick: "Keep the default"
    why: "TRTLLM now advertises multi-document packed-varlen support, and producers omit empty padding documents so boundaries stay strictly increasing."
---

## TL;DR {#tldr}

**[PR #6542](https://github.com/vllm-project/vllm-omni/pull/6542) removes the per-layer mask re-validation that sat between MiniMax-H3's Ulysses all-to-all and the TRTLLM attention kernel.** The model's packing code already knows on the CPU (the "host") exactly how many tokens are real; the PR adds a `PackedPaddingMetadata` field so it can hand that number to the attention backend directly. TRTLLM attention then trims the padding from Q/K/V using plain Python integers — no attention-mask tensor, no `nonzero` searches, no CUDA scalar reads (each of which forces the GPU to stop and sync with the CPU).

| Metric (per attention call) | Before: packed-mask validation | After: packed-padding fast path |
|---|---:|---:|
| All-to-all → FMHA gap, p50 | 499.683 µs | 0.352 µs |
| All-to-all → FMHA gap, p95 | 698.308 µs | 0.384 µs |
| Intermediate GPU activities | 244,925 | 0 |
| Intermediate D2H copies | 97,970 | 0 |

Upstream measurements on 4x NVIDIA B300, MiniMax-H3 official starship workload (1344×768, 243 frames, 49 denoise updates), Nsight Systems captures — details and caveats under [Measured impact](#measured-impact). The PR also fixes [#6358](https://github.com/vllm-project/vllm-omni/issues/6358): on datacenter Blackwell, where TRTLLM is the auto-selected default, the old mask check rejected H3's structural padding outright.

## Background {#background}

If you ran MiniMax-H3 on a B200 with default settings before this PR, one of two things happened, both bad. Either the server refused to start generating at all with `RuntimeError: Attention backend 'TRTLLM_ATTN' does not support attn_mask` ([issue #6358](https://github.com/vllm-project/vllm-omni/issues/6358) — TRTLLM is the auto-selected default on datacenter Blackwell, so you got this without pinning any flag). Or, on the branch where the packed TRTLLM path existed, every denoise step carried a hidden pause: the GPU finished its all-to-all exchange and then sat idle for roughly half a millisecond before each attention kernel could start. Half a millisecond does not sound like much — but it is charged on every attention call, in every transformer layer, in every one of the 49 denoise updates of a single video request.

The cause is a mismatch about *who knows what*. MiniMax-H3 aligns its packed sequence to a multiple of 64 rows, so each attention input is laid out as `[real tokens, padding]` — like a shipping pallet with real boxes followed by empty filler blocks so the pallet reaches a standard size. [PR #5779](https://github.com/vllm-project/vllm-omni/pull/5779) made TRTLLM attention *correct* for this layout by having the backend check the mask, trim the padded tokens off Q/K/V, and restore the padded output shape. But it did that check **from device tensors, on every call**: it summed the mask (a GPU reduction), compared it against an expected prefix pattern, ran `nonzero` searches to find the boundary, and read CUDA scalars back to the host — each read a device-to-host sync that serializes the stream. It is the equivalent of a warehouse worker who receives a pallet with a printed manifest, but insists on reopening and re-counting every box — before passing it on — and does so for every pallet, every day. The manifest (the host-known valid-token count) was always there; the backend just had no way to receive it.

The PR's framing of the root cause: *"The producer already knows the valid-token boundary on the host, but `AttentionMetadata` had no explicit way to publish that trusted boundary together with canonical device-side cumulative lengths."* So every layer reconstructed and re-validated information that was computed once, upstream, on the CPU.

## What PR #6542 changed {#key-changes}

The idea in one sentence: **let the producer of the packed layout publish its host-side knowledge as a small typed record, and let backends opt into trusting it.** Concretely, three moving parts:

1. **A new metadata record** — [`PackedPaddingMetadata`](https://github.com/vllm-project/vllm-omni/blob/51b7565f33017d74e2da7045327a65e8f048eaae/vllm_omni/diffusion/attention/backends/abstract.py) in `attention/backends/abstract.py`:

   ```python
   @dataclass(frozen=True, slots=True)
   class PackedPaddingMetadata:
       """Producer-validated mask-free view of padding in a [real, pad] packing."""

       q_length: int          # host-side count of real Q tokens
       kv_length: int         # host-side count of real K/V tokens
       cu_seqlens_q: torch.Tensor   # canonical [0, q_length] view
       cu_seqlens_k: torch.Tensor   # canonical [0, kv_length] view
   ```

   The cumulative-length tensors ("cu_seqlens" — cumulative sequence lengths, the format varlen kernels use to find each document inside a packed batch) are pinned to a canonical two-element `[0, length]` shape. That shape choice matters: a consumer can slice `cu_seqlens_kv[1:]` to get the per-sequence length list the kernel wants **without ever reading a device scalar**.

2. **A capability contract** — backends declare `supports_packed_mask_free()`. A packed producer may omit the attention mask *only* when the selected backend advertises it. MiniMax-H3's transformer now publishes `packed_padding` exactly when the backend does ([`minimax_h3_transformer.py`](https://github.com/vllm-project/vllm-omni/blob/51b7565f33017d74e2da7045327a65e8f048eaae/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py)); other backends keep receiving the old mask and behave exactly as before.

3. **A mask-free TRTLLM path** — [`trtllm_attn.py`](https://github.com/vllm-project/vllm-omni/blob/51b7565f33017d74e2da7045327a65e8f048eaae/vllm_omni/diffusion/attention/backends/trtllm_attn.py) validates the host contract (all in Python-land: integer types, bounds, devices, dtypes, shapes) and trims Q/K/V with plain slicing:

   ```python
   q = q[: packed_layout.q_tokens]   # valid_q_tokens is a Python int
   k = k[: packed_layout.kv_tokens]  # no .item(), no mask reduction
   v = v[: packed_layout.kv_tokens]
   ```

![Before: every attention call runs mask reduction, prefix comparison, nonzero searches, and ten device-to-host syncs between the Ulysses all-to-all and the FMHA kernel — a 499.7 µs p50 gap. After: the backend reads the producer's host-side `PackedPaddingMetadata` integers and starts FMHA directly — 0.35 µs, zero intermediate kernels, zero copies.]({{ site.baseurl }}/assets/figures/pr-6542-mask-free-packed-padding/fig1-attention-gap.svg)

Note the honesty of the analogy: the worker still *checks the manifest* (the backend still validates types, bounds, devices, and shape consistency of the metadata — see [the host contract](#host-contract)). What disappeared is the physical re-counting of boxes on GPU, not the paperwork.

## The host contract, in detail {#host-contract}

The first paragraph version: the backend trusts the producer's numbers, but only after checking they are *coherent* — like a border officer who trusts your printed itinerary but still checks the dates line up. `_prepare_packed_padding_layout` rejects, with loud `ValueError`s:

- **Non-integer lengths.** Every length must be a Python `int` (a `bool` or a `torch.Tensor` is rejected). This is the whole point: the moment one of these numbers is a device tensor, consuming it means a sync.
- **Out-of-bounds boundaries.** `q_length` and `kv_length` must land inside the packed Q and K/V sequences.
- **Inconsistent maxima.** `max_seqlen_q`/`max_seqlen_k` must equal the valid lengths, and an independently published `valid_kv_length` (the pre-existing `extra` field) must agree with `kv_length` — two sources of truth must not diverge.
- **Non-canonical cumulative lengths.** Exactly two elements, `int32`, on the same device as Q/K.
- **Wrong shape.** The fast path requires a single packed batch (`physical_batch == 1`).

On the multi-request side, two smaller changes make continuous batching work on the same path:

- TRTLLM now advertises `supports_multi_doc_packed_varlen()`, and the generic packed layout ([`_prepare_generic_packed_layout`](https://github.com/vllm-project/vllm-omni/blob/51b7565f33017d74e2da7045327a65e8f048eaae/vllm_omni/diffusion/attention/backends/trtllm_attn.py)) accepts any number of real documents, not just a single `[real, pad]` pair.
- The batched producer ([`batched_packing.py`](https://github.com/vllm-project/vllm-omni/blob/51b7565f33017d74e2da7045327a65e8f048eaae/vllm_omni/diffusion/models/minimax_h3/batched_packing.py)) used to emit *two* cumulative boundaries per request — one for the real rows, one for the padding tail — even when 64-row alignment left the tail empty. Empty documents are now omitted, because "not every varlen kernel accepts repeated interior boundaries"; a contract test asserts the boundaries are strictly increasing.

Finally, the SAGE quantization path gets a free win: deciding whether sequences are long enough for SAGE block quantization used to be `torch.all(seq_lens >= block).item()` — another sync. On the fast path the host already knows the minimum KV length, so the check is a plain integer comparison.

## Measured impact {#measured-impact}

All numbers below are from the [PR author's Nsight Systems captures](https://github.com/vllm-project/vllm-omni/pull/6542) — they are upstream measurements, not cookbook benchmark runs. Setup: 4x NVIDIA B300 SXM6 (267.7 GiB each), TP1/Ulysses4/Ring1, dense BF16 `TRTLLM_ATTN`, regional compile, MiniMax-H3 official starship workload at 1344×768, 243 frames, 49 denoise updates; each capture followed one unprofiled warmup request.

| Metric | Before: packed-mask validation | After: packed-padding fast path |
|---|---:|---:|
| Paired attention samples | 9,797 | 9,680 |
| Gap p50 | 499.683 µs | 0.352 µs |
| Gap p95 | 698.308 µs | 0.384 µs |
| Intermediate GPU activities | 244,925 | 0 |
| Intermediate D2H copies | 97,970 | 0 |

The "gap" is deliberately scoped: from the **end of the final Ulysses all-to-all kernel** to the **start of the main TRTLLM FMHA kernel**. That is the housekeeping window the PR attacks. Two honesty notes:

- **This is a gap-level measurement, not an end-to-end speedup claim.** The PR does not publish an E2E latency delta; what it does publish is a correctness anchor — the 4x B300 end-to-end run completed all 49 denoise updates and produced a valid 243-frame MP4 whose frame SHA256 **exactly matches** the previously validated packed-fastpath baseline.
- The two raw counts scale with samples: ~25 intermediate GPU activities and exactly 10 device-to-host copies per attention call before the PR. After: zero of each.

Correctness coverage beyond the E2E hash: 37 backend tests pass on B300 (the one skip is the intentionally unsupported INT8 SAGE case there), 159 CPU/mocked contract tests pass, and a real TRTLLM multi-document cross-attention run matched per-document SDPA with relative L2 error `0.000041`.

## How to use it {#how-to-use}

There is nothing to enable: the fast path engages automatically when the selected backend advertises mask-free packed-padding support and the model's producer publishes the metadata. The operator-visible surface is the backend choice itself.

{% include usage-cookbook.html modes=page.usage %}

If you were previously bitten by the [#6358](https://github.com/vllm-project/vllm-omni/issues/6358) `RuntimeError` on B200, upgrading past this PR is the fix — the generic capability check no longer rejects MiniMax-H3's structural padding mask before the packed TRTLLM path runs.

## How to choose {#decision-cards}

{% include decision-cards.html items=page.decisions %}

## Limitations and follow-ups {#limitations}

- **This is not arbitrary-mask support.** The fast path covers exactly one shape: a producer-owned, single-batch `[real, pad]` packing where padding is a suffix. TRTLLM_ATTN now rejects *every* nonempty `attn_mask` — previously it rejected non-prefix masks; the new message points at the packed-padding escape hatch or a mask-capable backend.
- **The backend still validates the host contract.** A producer that lies — wrong dtypes, off-device tensors, inconsistent maxima — fails loudly with `ValueError`, which is the intended failure mode: better an exception than a silently wrong trim.
- **E2E wall-clock impact is not separately reported.** The published measurement is the all-to-all → FMHA gap plus the zero-intermediate-activity counts, with E2E used as a bit-exact correctness check (frame SHA256 match). A wall-clock delta at the video level remains to be measured.
- **Where this sits in the stack:** PR #5543 (the generic mask-capability check) was rebased on top of this PR — the author's stated plan was to merge #6542 first, then rebase, so #5543 "can validate the resulting contract without absorbing this implementation."
- This is Blog 3 in the 8-post MiniMax-H3 optimization series tracked by [series RFC #37](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/37); it is the flagship of the "non-GPU hot path" topic. Related reads on this blog: [Blog 1 — the modular pipeline](/2026-08-24-understanding-pr-5720-minimax-h3-modular-pipeline/) and [Blog 2 — the "four steps" contracts](/2026-08-24-understanding-pr-5991-minimax-h3-few-step-schedules/).

## References {#references}

- [PR #6542 — [perf] Add mask-free TRTLLM packed-padding path](https://github.com/vllm-project/vllm-omni/pull/6542) (merged 2026-08-25, commit [`51b7565`](https://github.com/vllm-project/vllm-omni/commit/51b7565f33017d74e2da7045327a65e8f048eaae))
- [Issue #6358 — [Bug][B200] MiniMax-H3 B200 default TRTLLM_ATTN rejects packed padding attn_mask](https://github.com/vllm-project/vllm-omni/issues/6358)
- [PR #5779 — the prior packed-mask validation this PR replaces](https://github.com/vllm-project/vllm-omni/pull/5779)
- [PR #5543 — generic mask-capability check, rebased on top of #6542](https://github.com/vllm-project/vllm-omni/pull/5543)
- [`docs/design/feature/attention_backend_selection.md`](https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/attention_backend_selection.md) (upstream backend-selection design)
- [Series RFC #37 — MiniMax-H3 optimization blog series](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/37)
