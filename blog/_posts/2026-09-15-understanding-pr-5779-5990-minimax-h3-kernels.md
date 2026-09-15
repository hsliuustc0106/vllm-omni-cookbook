---
layout: post
title: "Serving MiniMax-H3 in vLLM-Omni (5): kernels — TRTLLM becomes the default and the Q/K prologue fuses into one pass (PRs #5779 + #5990)"
date: 2026-09-15 12:00:00 +0800
author: hsliuustc0106
summary: >-
  PR #5779 makes TRTLLM attention work on MiniMax-H3's packed sequences and
  the datacenter-Blackwell default; PR #5990 fuses Q/K RMSNorm+RoPE into one
  Triton pass — −3.18% steady denoising on B300.
tags: [MiniMax-H3, TRTLLM, B300]
category: PR Analysis
feature: kernels
lang: en
pair: /zh/2026-09-15-understanding-pr-5779-5990-minimax-h3-kernels/
usage:
  - label: "Serve · 4× B300 default"
    blurb: "no attention flag — TRTLLM is auto-selected"
    title: "vllm serve · four-GPU throughput profile"
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
      Since PR #5779, MiniMax-H3 declares its packed-sequence contract and the
      platform auto-selects dense BF16 TRTLLM_ATTN on datacenter Blackwell
      (sm_100/sm_103) — no attention flag. Confirm the log line "Defaulting to
      diffusion attention backend TRTLLM_ATTN" before recording measurements.
      Do not add --enforce-eager; the first request includes regional
      compilation, so warm once before measuring. H3 is CFG-distilled:
      --cfg-parallel-size stays 1.
  - label: "Serve · FA4 comparison"
    blurb: "explicit FLASH_ATTN baseline"
    title: "vllm serve · FlashAttention-4 on Blackwell"
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
        --diffusion-attention-backend FLASH_ATTN
    note: >-
      FA4 stays a first-class comparison point. Install the optional extra
      first ("uv pip install -e '.[fa4]'") — without it the official image
      falls back to a Hopper-only kernel that fails on Blackwell with "no
      kernel image is available" (the #5779 baseline debugging story). Confirm
      the log line "Using CuTe FlashAttention-4 on Blackwell".
  - label: "Serve · SAGE + Skip-Softmax"
    blurb: "lossy knobs, per-role guarded"
    title: "vllm serve · TRTLLM quantized + sparse"
    code: |
      vllm serve "${MODEL}" \
        --omni --host 0.0.0.0 --port "${PORT}" --trust-remote-code \
        --num-gpus 4 --usp 4 --ring 1 \
        --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
        --diffusion-attention-config '{
          "default": {
            "backend": "TRTLLM_ATTN",
            "quant": {
              "dtype_qk": "fp8_e4m3",
              "q_block_size": 1,
              "k_block_size": 16
            },
            "skip_softmax": {
              "threshold": 0.05,
              "disabled_until_timestep": 0.97
            }
          },
          "per_role": {
            "minimax_h3.token_refiner": {
              "backend": "TRTLLM_ATTN"
            }
          }
        }'
    note: >-
      Both optimizations are lossy and their effects compound — compare
      against dense output on the same prompt and seed before adopting. The
      values above are the upstream recipe's conservative starting point: at
      50 steps the 0.97 cutoff leaves the first 14 of 49 denoiser forwards
      dense. The per_role entry keeps the 14-token token refiner dense; a
      per-role spec does not inherit quant or skip_softmax from default. B200
      additionally supports int8 Q/K, which preserves accuracy better than
      FP8.
decisions:
  - when: "Datacenter Blackwell (B200/B300, sm_100/sm_103)"
    pick: "Keep the TRTLLM default"
    why: "Auto-selected since #5779 once the model declares its packed contract; the upstream recipe's stable A/B puts dense TRTLLM within 2% of FA4, and TRTLLM is the only backend that opens the Skip-Softmax and SAGE doors."
  - when: "You need the FA4 comparison point"
    pick: "--diffusion-attention-backend FLASH_ATTN"
    why: "Stays first-class with the fa4 extra installed. Note the spread in reported A/Bs: one matched review run had FA4 14.1% slower on diffusion, stable recipe runs say within 2% — single-pair benchmarks are config-sensitive."
  - when: "Workstation Blackwell (sm_120/sm_121) or head_dim ≠ 128"
    pick: "Stay on the CUDNN_ATTN route"
    why: "The TRTLLM auto-route deliberately skips workstation Blackwell and non-128 head dims; those GPUs keep their normal fallback, as do mask-using paths."
  - when: "Counting every percent on datacenter BF16"
    pick: "Nothing to configure — the fused Q/K prologue is automatic"
    why: "#5990 engages whenever Triton + CUDA + BF16 + head_dim 128 + rotary_dim 96 hold (exactly H3's geometry), and silently falls back to the eager reference everywhere else. −3.18% steady denoising in the author's B300 run."
  - when: "Trading fidelity for speed"
    pick: "SAGE + Skip-Softmax, per-role guarded"
    why: "Quantizes Q/K and skips negligible softmax tiles. Keep the token refiner dense via per_role, and remember even dense backends differ slightly: TRTLLM vs FA4 measured PSNR 27.10 dB / SSIM 0.8880 on the same prompt."
---

## TL;DR {#tldr}

**These two PRs are the kernel chapter of the MiniMax-H3 story: PR #5779 made
the fastest attention backend compatible with H3's packed sequences — and made
it the default on datacenter Blackwell — and PR #5990 fused the two small
operations that run before every attention call into a single GPU kernel,
buying a further −3.18% steady denoising latency.** Think of a warehouse where
the best truck (TRTLLM) used to refuse this customer's pallets outright
because they contain filler blocks (#5771); the first fix teaches the truck to
read the manifest and unload only real boxes, and the second fix merges the
two paperwork stations at the loading dock into one.

| Change | PR | Merged | Effect on 4× B300 |
|---|---|---|---|
| TRTLLM reads H3's packed layout, becomes the default | [#5779](https://github.com/vllm-project/vllm-omni/pull/5779) | 2026-08-06 | dense TRTLLM 14.1% faster diffusion than FA4 in one matched run; stable A/Bs say "within 2%" |
| Q/K RMSNorm + RoPE fused into one Triton pass | [#5990](https://github.com/vllm-project/vllm-omni/pull/5990) | 2026-08-14 | steady denoising 109.687 s → 106.205 s (−3.18%, 1.033×) |

Both are automatic today: on datacenter Blackwell, H3 auto-selects dense BF16
`TRTLLM_ATTN`, and the fused prologue engages on H3's exact geometry (BF16,
`head_dim=128`, `rotary_dim=96`) with an eager fallback everywhere else. The
second number is also the honesty lesson of this chapter — a kernel-level
rewrite that ends in "1.033×" end-to-end, because the prologue it removes is
small next to the attention math it precedes.

## Background {#background}

This section establishes where these two PRs sit: they are the two steps that
turned "TRTLLM attention cannot run MiniMax-H3 at all" into "TRTLLM is the
default and the surrounding arithmetic is fused" — like first paving a road,
then widening the driveway. [Blog 3](#limitations) covered the step *after*
that (PR #6542 removing the per-call mask re-validation); this post covers the
two steps before it, both by the same author (Bo Li) on the same 4× B300
testbed, both labeled `Kernel optimization` upstream.

**Why TRTLLM at all?** On Blackwell, `TRTLLM_ATTN` — FlashInfer's vendoring of
TensorRT-LLM's generated attention kernels — is the only diffusion backend
that can enable the two lossy accelerations Skip-Softmax (sparse attention)
and SAGE (quantized attention). Before #5779, wanting those meant being unable
to run H3:

- **Hard failure ([issue #5771](https://github.com/vllm-project/vllm-omni/issues/5771)).** Select `TRTLLM_ATTN` for H3 and the very first packed attention call dies with `ValueError: TRTLLM_ATTN does not support attn_mask`. H3 *needs* that mask because it aligns its packed sequence to a multiple of 64 rows: the example workload carries 58,758 valid tokens in a 58,816-row tensor, and the 58 filler rows at the end are marked by a prefix-valid mask (`arange(total) < used`). TRTLLM rejected *any* mask.
- **Non-finite output under SAGE.** If you instead treated the 58 filler tokens as one more "valid" sequence, dense attention survived — but SAGE quantization happily compressed garbage values and could produce non-finite output.
- **A 14-token corner.** H3 has a second, tiny attention site (the token refiner) whose 14 tokens are *shorter than one SAGE K quantization block* (`k_block_size=16`); FlashInfer SAGE returns non-finite output for such sequences.
- **Skip-Softmax couldn't gate.** H3 did not publish a denoise timestep, so the Skip-Softmax "stay dense early, sparsify late" gate could never engage — the path logged a warning and stayed dense.

And the target of #5990: before every attention call, each of Q and K passes
through two operations — RMSNorm (rescale each head's 128-dim vector to unit
RMS, then apply a learned weight) and a partial RoPE (rotate the first 96 of
the 128 dims by position-dependent angles). In eager mode that is a chain of
small kernels each writing intermediates to HBM — paid per tensor, per DiT
block, in each of the 49 denoiser evaluations of a 50-step video.

## What PR #5779 changed {#trtllm-refine}

This section is the mechanism story: the backend learns to trust the packing
metadata instead of refusing it — like a receiving dock that finally accepts
pallets whose manifest says "real boxes first, filler last", unloads only the
real boxes, and re-straps filler so the pallet leaves at the standard size
again.

### Trust the packing metadata, then trim {#packed-trim}

The packed path keys off four metadata fields that ride along in
`AttentionMetadata.extra` — `cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q`,
`max_seqlen_k` (cu_seqlens = cumulative sequence lengths, the varlen format
packed kernels use to find document boundaries). The contract is strict and
loud:

- **All four or none.** A partially populated set fails with `Incomplete
  packed TRTLLM attention metadata; missing [...]`.
- **The metadata must cover the inputs.** `cu_seqlens` that do not span every
  Q/K/V token fail with `must cover all Q/K/V tokens`.
- **Only structural padding qualifies.** A mask is accepted *only* if it is
  prefix-valid — real tokens form a contiguous prefix, filler is a suffix.
  Anything else keeps the original hard rejection.

With the contract satisfied, the backend trims Q/K/V to the valid prefix,
calls the ragged (varlen) TRTLLM kernel with the cumulative lengths — the
filler never reaches quantization or attention — and then **restores the
aligned physical shape with zero padding**, because the Ulysses all-to-all
that runs next expects fixed-size buffers.

![PR #5779's packed path: the prefix-valid mask and cu_seqlens metadata let TRTLLM trim Q/K/V to the 58,758 valid tokens, run the ragged FMHA on real tokens only, and zero-restore the 58,816-row aligned shape for the Ulysses all-to-all. The lower panel shows the two failure modes prevented: SAGE quantizing padding garbage into non-finite output, and the 14-token token refiner being shorter than one SAGE K quantization block.]({{ site.baseurl }}/assets/figures/pr-5779-5990-minimax-h3-kernels/fig1-packed-trim.svg)

One honesty marker for series continuity: in #5779 the prefix-valid check
still runs *from device tensors on every call* (mask sums, comparisons,
reads). That per-call cost — the ~499.7 µs p50 gap — is exactly what
[PR #6542, Blog 3]({{ site.baseurl }}/2026-09-12-understanding-pr-6542-trtllm-mask-free-packed-padding/)
removed a few weeks later by letting the producer publish the boundary on the
host. #5779 made the path correct; #6542 made it sync-free.

### The 14-token attention site that stays dense {#token-refiner}

The token refiner's 14 tokens cannot fill one SAGE K-quantization block of
16, so SAGE there is not merely inaccurate — it is non-finite. #5779 makes the
backend itself route any sequence shorter than one quantization block to the
dense TRTLLM kernel (a unit test asserts the SAGE quantizer is *not* invoked
for such sequences). The recipe's `per_role` override
(`minimax_h3.token_refiner`) pins the same intent explicitly when SAGE and
Skip-Softmax are enabled for the main DiT sequence — and note the semantics:
**a per-role spec does not inherit `quant` or `skip_softmax` from `default`**,
which is what "keep this site plain dense" relies on.

### Publishing the denoise timestep {#timestep}

Skip-Softmax needs to know "how far into denoising are we" to decide when
sparsification is safe — a thermostat that cannot read the room never switches
modes. H3 was not publishing a normalized timestep, so the gate stayed dense.
The readiness fixes wired it in with the correct convention (`t = 1 − sigma`,
sigma being the noise level) and restored the non-negative threshold
contract; `record_denoise_step(idx, normalized_timestep=...)` now surfaces
through the forward context. At the recipe's default cutoff (`0.97`) and H3's
flow shift of 12, the first 14 of 49 denoiser forwards stay dense.

### What the reviewers changed {#review-story}

The review is where "works" became "default":

- **"LGTM, let's make trtllm as the default backend for SM100"** — the
  maintainer's first comment. He then prepared the follow-up commit
  ([`20cc23ae`](https://github.com/lishunyang12/vllm-omni/commit/20cc23ae))
  that has H3 *declare its packed-sequence contract*: the platform
  auto-selects dense BF16 `TRTLLM_ATTN` on supported datacenter Blackwell
  (sm_100/sm_103) when FlashInfer's trtllm-gen kernel is available.
  Workstation Blackwell (sm_120/sm_121), unsupported head dims, and
  mask-using paths keep their normal fallback — and no Skip-Softmax or
  quantization is enabled by default. A contract test pins
  `MiniMaxH3Pipeline → attention_mask_free = True`.
- **The FA4 baseline had to be debugged into existence.** The official vLLM
  0.26.0 image lacked `flash-attn-4`, so `FLASH_ATTN` fell back to a
  Hopper-only kernel that failed on SM103 with `no kernel image is
  available`; only after installing `flash-attn-4[cu13]==4.0.0b18` did
  "CuTe FlashAttention-4 on Blackwell" engage — worth knowing before you
  trust your own FA4 numbers.
- **Readiness fixes**: the timestep convention above, the threshold contract,
  and a rewritten DCO author identity.

## What PR #5990 changed {#fused-prologue}

This section is the other mechanism story: the two operations before every
attention call become one GPU kernel — like merging the "sharpen" and
"rotate" stations of a photo pipeline, which previously each printed their
work and made the next station re-read it from a shared drawer, into a single
station that does both without putting anything down.

### The eager prologue, and what it costs {#eager-prologue}

Per Q and per K, the eager path runs `F.rms_norm` (writes a full normalized
`[tokens, heads, 128]` tensor to HBM), then RoPE: split each vector's first 96
dims into two halves, multiply by the cos/sin table, subtract/add, and
`torch.cat` the rotated halves back with the untouched last 32 dims — more
kernels, more materialized intermediates, more HBM round-trips. On H3's
geometry (BF16, `head_dim=128`, `rotary_dim=96`, non-interleaved halves of
48) this runs twice per DiT block, per denoise evaluation — a small tax
repeated a very large number of times.

### One Triton pass {#one-pass}

The new
[`fused_qk_norm_rope.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/layers/fused_qk_norm_rope.py)
kernel launches one program per (token, group of 8 heads). Each program loads
its Q (or K) slice once, computes the RMS normalization in fp32 registers
(`rsqrt(mean(x²) + eps)` × weight), locates each element's rotation partner
by index arithmetic (the non-interleaved half-swap), multiplies by cos/sin
from a packed table, and stores once. No normalized copy, no rotary-product
intermediates, no `cat` — the only HBM traffic is one read and one write.

Two supporting changes matter beyond the kernel itself:

- **A shared rope table.** H3 now materializes one packed
  `[cos θ₀..₄₇, sin θ₀..₄₇]` table per forward (`_build_rope_table`) and every
  block consumes it — previously each block re-derived cos/sin from raw
  frequencies. The TeaCache context extractor was moved to the same table so
  caching stays bit-consistent with serving.
- **A custom-op boundary.** The fused path is registered as
  `vllm_omni::fused_qk_norm_rope` with a fake (meta) implementation, so the
  regional `torch.compile` H3 relies on can trace through it.

The public layer API is model-independent; the fast path is gated narrowly —
Triton + CUDA + BF16 + `head_dim=128` + `rotary_dim=96` — and anything else
(other dtypes, other geometries, CPU) silently uses the eager reference.

![Before PR #5990, the eager prologue runs F.rms_norm, writes a normalized tensor to HBM, then splits, multiplies, and re-cats the RoPE rotation — per Q and per K, every DiT block, 49 forwards per video. After, one Triton kernel per tensor does the fp32 RMS normalization in registers and the RoPE pair swap against a shared packed cos/sin table: one read, one write, no intermediates.]({{ site.baseurl }}/assets/figures/pr-5779-5990-minimax-h3-kernels/fig2-qk-fusion.svg)

### The accuracy question reviewers asked {#accuracy}

The reviewer asked for SSIM/PSNR evidence that fusion does not visibly change
the video. The author's answer was the unit test: fused vs the BF16 eager
reference over sequence lengths 1, 257, and 1024 shows a **maximum absolute
error of 0.0625 and a mean absolute error of 0.00072–0.00077** — BF16-scale
rounding differences — and the end-to-end generated video is "nearly
identical" to the baseline. No SSIM table was added; that is worth knowing
when you quote this PR.

## Measured impact {#measured-impact}

All numbers below are from the upstream PR bodies, review comments, and
recipe — author-reported measurements, not cookbook benchmark runs.

**Backend A/B (#5779 review, matched pair).** 4× B300 SM103, 1248×768,
209 frames, 50 steps, seed 1101, Ulysses4/Ring1/TP1, VAE tile4, regional
compile; one compile warmup excluded ([source: review comment with commit
`20cc23ae`](https://github.com/vllm-project/vllm-omni/pull/5779#pullrequestreview-)):

| Backend | Steady diffusion | Wall |
|---|---:|---:|
| CuTe FlashAttention-4 (`FLASH_ATTN`) | 83.854 s | 88.558 s |
| Dense BF16 TRTLLM (`TRTLLM_ATTN`) | **71.990 s** | **76.176 s** |

TRTLLM is **14.1% faster on diffusion, 14.0% end-to-end** in this run. The
upstream recipe, however, records that *stable* measurements with its
four-GPU profile "put dense `TRTLLM_ATTN` and FA4 within 2% of each other",
and the pre-#5779 documentation was explicit that TRTLLM outranks cuDNN "not
because its dense kernel is faster". Read the two numbers together: the
default decision rests on TRTLLM unlocking Skip-Softmax/SAGE and owning the
maintained packed path — not on a guaranteed dense-kernel win. Single matched
pairs are config-sensitive; that spread is the data.

**Dense output is not bit-identical across backends.** On the same B300 node,
a 50-step FA4 run completed in 88.98 s (model stage); comparing the encoded
TRTLLM video against FA4 on the same prompt/seed gave **average PSNR
27.10 dB, SSIM 0.8880** — same scene, motion, and composition, but not the
same pixels. Budget for that when you switch backends or chase the lossy
knobs on top.

**Fused Q/K prologue (#5990).** 4× B300 SXM6, TP1/Ulysses4/Ring1, dense BF16
`TRTLLM_ATTN`, 1344×768, 243 frames, 50 configured steps, seed 0; one warmup
followed by one measured request:

| Metric | Before (eager prologue) | After (fused) |
|---|---:|---:|
| Steady denoising latency | 109.687 s | **106.205 s** |

That is **−3.18% (1.033×)**. The PR ships Nsight Systems captures before and
after showing the kernel-launch and memory-traffic collapse in the prologue
window. Two honesty notes the series holds onto:

- **Fusion refunds what the fused work cost — no more.** The prologue is
  linear in tokens; at ~58k tokens the attention kernel itself dominates the
  layer, so eliminating every intermediate round-trip still lands at ~3% of
  steady denoising. However large the kernel-local win looks in a profile,
  it does not translate linearly past the work it removes — and this PR never
  claims it does.
- **One warmup, one measured request.** Every number above is a single
  observation on one seed, not a distribution — the same methodology caveat
  Blog 4 attached to its capacity runs.

## How to use it {#how-to-use}

There is nothing to enable. The TRTLLM default engages automatically on
datacenter Blackwell, and the fused Q/K prologue engages automatically on
H3's geometry. The operator-visible surface is choosing a comparison backend
and the two lossy knobs:

{% include usage-cookbook.html modes=page.usage %}

If you were reproducing [issue #5771](https://github.com/vllm-project/vllm-omni/issues/5771)
(the `attn_mask` rejection), upgrading past #5779 is the fix for the packed
path itself; the related B200 default-selection failure
([#6358](https://github.com/vllm-project/vllm-omni/issues/6358), Blog 3) was
fixed later by #6542.

## How to choose {#decision-cards}

{% include decision-cards.html items=page.decisions %}

## Limitations and follow-ups {#limitations}

- **TRTLLM dense is datacenter-Blackwell-only** (sm_100/sm_103, FlashInfer
  present). Workstation Blackwell and non-128 head dims keep their fallback
  routes; the two-GPU offload profiles of
  [Blog 4]({{ site.baseurl }}/2026-09-14-understanding-pr-5764-minimax-h3-layerwise-offload/)
  deliberately run cuDNN attention.
- **SAGE and Skip-Softmax are lossy and compound.** The conservative recipe
  values (threshold 0.05, cutoff 0.97) leave the first 14 of 49 forwards
  dense; the PR that validated them on B300 checked HTTP-200-plus-valid-MP4,
  not perceptual quality tables.
- **The fused kernel is deliberately narrow.** BF16, `head_dim=128`,
  `rotary_dim=96` — H3's geometry, generalized only via the eager fallback.
  Its published accuracy evidence is the unit-test error stats, not SSIM.
- **#5990's measurement is one seed, one request, steady denoising only.** No
  throughput, multi-request, or quality table accompanies it.
- **The mask still exists in this PR's path.** #5779 validates the
  prefix-valid mask per call from device tensors; #6542 (Blog 3) is the
  follow-up that removed that cost — read the two posts as one arc.
- **The rest of the kernel topic is still open upstream.** The VAE-side
  kernel PRs (#6030 stacked tiling, #5937 leaner decode, #5979 regional
  compile, #6014, #5985) are all unmerged, as is the SM120 benchmark (#5852);
  the topic's other merged kernel work (NPU SwiGLU fusion, #5801/#6167)
  belongs to the NPU hardware story planned for the series' hardware post.
- This is Blog 5 in the 8-post MiniMax-H3 optimization series tracked by
  [series RFC #37](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/37);
  it is the core post of the "kernel optimization" topic. Related reads:
  [Blog 1 — the modular pipeline]({{ site.baseurl }}/2026-08-24-understanding-pr-5720-minimax-h3-modular-pipeline/),
  [Blog 2 — the "four steps" contracts]({{ site.baseurl }}/2026-08-24-understanding-pr-5991-minimax-h3-few-step-schedules/),
  [Blog 3 — mask-free packed padding]({{ site.baseurl }}/2026-09-12-understanding-pr-6542-trtllm-mask-free-packed-padding/)
  (the direct successor of #5779's packed path),
  [Blog 4 — layerwise offload]({{ site.baseurl }}/2026-09-14-understanding-pr-5764-minimax-h3-layerwise-offload/).

## References {#references}

- [PR #5779 — \[Attention\] Refine TRTLLM attention support for MiniMax H3](https://github.com/vllm-project/vllm-omni/pull/5779) (merged 2026-08-06, commit [`d219d93`](https://github.com/vllm-project/vllm-omni/commit/d219d93bbb4db1a93b6c84e77f375838ebd8246e))
- [Issue #5771 — TRTLLM attention rejects MiniMax-H3's structural padding mask](https://github.com/vllm-project/vllm-omni/issues/5771)
- [PR #5990 — \[Kernel\] Fuse Q/K RMSNorm and RoPE](https://github.com/vllm-project/vllm-omni/pull/5990) (merged 2026-08-14, commit [`596c16a`](https://github.com/vllm-project/vllm-omni/commit/596c16a550aa134faf7f3dcfa0f8adf513ccd9ce))
- [Issue #5700 — MiniMax-H3 progress ↔ issue/PR mapping](https://github.com/vllm-project/vllm-omni/issues/5700) (the umbrella tracker both PRs report into)
- [`fused_qk_norm_rope.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/layers/fused_qk_norm_rope.py) and [`trtllm_attn.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/attention/backends/trtllm_attn.py) (upstream, current)
- [Attention backends user guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/attention_backends.md) (upstream, current — the auto-route table #5779 rewrote)
- [MiniMax-H3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md) (upstream, current — the four-GPU profile and the SAGE/Skip-Softmax starting values)
- [Blog 3 — mask-free TRTLLM packed padding]({{ site.baseurl }}/2026-09-12-understanding-pr-6542-trtllm-mask-free-packed-padding/)
- [Series RFC #37 — MiniMax-H3 optimization blog series](https://github.com/hsliuustc0106/vllm-omni-cookbook/issues/37)
