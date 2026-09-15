---
layout: post
title: "Understanding PR #6516 — SenseNova-U1.5-8B-MoT and its distilled 8-step LoRA"
date: 2026-09-15 12:00:00 +0800
author: hsliuustc0106
summary: >-
  PR #6516 lands SenseNova-U1.5-8B-MoT: the distilled 8-step LoRA is fused for
  real (a substring bug silently dropped it), and paged-KV CUDA-graph decode
  cuts think-mode end-to-end by half — 1024² images in ~0.9 s at 8 steps.
tags: [SenseNova-U1.5, A800, H200]
category: PR Analysis
feature: lora
lang: en
pair: /zh/2026-09-15-understanding-pr-6516-sensenova-u15-distilled-lora/
usage:
  - label: "Offline · quality"
    blurb: "50 steps + CFG 4.0, 1024×1024"
    title: "text_to_image.py · full-quality path"
    code: |
      python examples/offline_inference/text_to_image/text_to_image.py \
        --model sensenova/SenseNova-U1.5-8B-MoT \
        --prompt "Close portrait of an elderly woman by a farmhouse window, warm natural light." \
        --width 1024 --height 1024 \
        --seed 42 --num-inference-steps 50 --cfg-scale 4.0 \
        --extra-body '{"think": false, "cfg_norm": "none", "timestep_shift": 3.0, "t_eps": 0.02}' \
        --output sensenova_u15_t2i.png
    note: >-
      Think mode (`"think": true`) is recommended upstream for higher image
      quality; it adds the autoregressive reasoning pass that PR #6516
      accelerated.
  - label: "Offline · 8-step LoRA"
    blurb: "distilled few-step, ~0.9 s at 1024²"
    title: "text_to_image.py · distilled 8-step path"
    code: |
      python examples/offline_inference/text_to_image/text_to_image.py \
        --model sensenova/SenseNova-U1.5-8B-MoT \
        --lora-path SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors --lora-backend distill \
        --prompt "Close portrait of an elderly woman by a farmhouse window, warm natural light." \
        --width 1024 --height 1024 \
        --seed 42 --num-inference-steps 8 --cfg-scale 1.0 \
        --extra-body '{"think": false, "cfg_norm": "none", "timestep_shift": 3.0, "t_eps": 0.02}' \
        --output sensenova_u15_lora8.png
    note: >-
      Use --cfg-scale 1.0 with this LoRA. It is distilled with DMD and runs
      without classifier-free guidance; the default 4.0 applies guidance twice
      and produces a blown-out, posterised image.
  - label: "Edit · image-to-image"
    blurb: "25 steps, think on"
    title: "image_edit.py · oil-painting pass"
    code: |
      python examples/offline_inference/image_to_image/image_edit.py \
        --model sensenova/SenseNova-U1.5-8B-MoT \
        --prompt "Turn this into an oil painting" \
        --image input.png --resolution 1024 \
        --seed 42 --num-inference-steps 25 --cfg-scale 4.0 \
        --extra-args '{"think": true, "img_cfg_scale": 1.0, "cfg_norm": "none", "timestep_shift": 3.0}' \
        --output sensenova_u15_edit.png
  - label: "Serve · OpenAI API"
    blurb: "online, text + vision"
    title: "vllm serve · omni entrypoint"
    code: |
      vllm serve sensenova/SenseNova-U1.5-8B-MoT --omni --port 8091

      python examples/online_serving/sensenova_u1/openai_chat_client.py \
        -s http://127.0.0.1:8091 -m img2text -i input.png -p "Describe this image."
    note: >-
      -s takes the base URL; the client appends /v1 itself. Online smoke at
      merge: /health OK, img2text returned a 1,921-character description.
decisions:
  - when: "Best image quality, latency secondary"
    pick: "50 steps + CFG 4.0 (think on)"
    why: "The quality path the recipe recommends: 10.97 s at 1024² on A800 post-PR; think mode adds the reasoning pass the same PR made ~4.6× faster."
  - when: "Interactive latency or batch throughput"
    pick: "8-step distilled LoRA + CFG 1.0"
    why: "~1.0 s wall at 1024² on A800 (624 ms diffusion stage on H200); the fusion itself is measured free (−0.23% vs same-step control) while changing 1,048,574 of 1,048,576 pixels — never pair it with CFG 4.0."
  - when: "Understanding-heavy workloads (think, img2text, chat)"
    pick: "Keep paged decode on (the default)"
    why: "CUDA-graph decode cuts the think loop 43–46% and lifts GPU busy to 97.7%; VLLM_OMNI_SENSENOVA_PAGED_DECODE=0 exists for debugging and parity checks."
  - when: "Less than ~40 GB of GPU memory free"
    pick: "Not this model (yet)"
    why: "34.3 GB peak at 1024² and 36.4 GB at 1536×2720 in bf16; validated on 80 GB-class cards (A800, H200) only."
  - when: "You need to swap adapters at runtime"
    pick: "One fused config per engine"
    why: "Distill fusion is one-way (no unload path), targets the generation tower only, and a dynamic LoRA manager intentionally disables graph/cache reuse."
  - when: "Deploying on v0.26.x and hitting a head_dtype crash"
    pick: "Upgrade to v0.28+"
    why: "U1.5 serving crashed on v0.26.0 (_DiffusionVllmModelConfig missing head_dtype); fixed by #5877. Full U1.5 support first ships in the v0.29 line."
---

## TL;DR {#tldr}

**PR #6516 makes SenseNova-U1.5-8B-MoT a first-class citizen in vLLM-Omni — and
the interesting part is what "support" turned out to require. The model already
ran on the existing U1 pipeline, but its distilled 8-step LoRA loaded with a
success message and then silently changed nothing; fixing that no-op exposed a
decode loop so host-bound that the PR grew a paged-KV CUDA-graph path for the
autoregressive "think" stage.** Think of a turbo badge bolted onto a car whose
engine never engaged it — and while opening the hood, finding the fuel line was
also kinked.

| Metric (1× A800-80G, 1024², seed 42) | main | PR #6516 | Δ |
|---|---:|---:|---:|
| Think-mode E2E, 50 steps (P50 of 10) | 27.747 s | 13.930 s | **−49.8%** |
| …think decode stage alone | 16.146 s | 3.534 s | **−78.1%** |
| …with 8-step distilled LoRA, total E2E | — | 4.455 s | 8 steps vs 50 |
| T2I, 50 steps, CFG 4.0 (n=5 median) | 11,726.9 ms | 10,973.6 ms | **−6.42%** |
| T2I, 8 steps + distilled LoRA | 1,114.4 ms | 1,024.2 ms | −8.09% (see below) |
| `cudaLaunchKernel` per 8-step T2I generation | 24,244 | 7,782 | **−67.9%** |

The 8-step row needs its asterisk: on `main` the adapter was silently dropped,
so "before" there is the base weights at 8 steps. The real claim is narrower
and better — the fused adapter is measured **free** (−0.23% against the
same-step no-LoRA control) while changing 1,048,574 of 1,048,576 output pixels
(MAE 58.62): the distillation's value is the step count, and it costs nothing
to apply. Merged 2026-09-01 (commit
[`0288c3f`](https://github.com/vllm-project/vllm-omni/commit/0288c3f56eb4a4ff410194e586498e3bc8ff8362)),
first shipped in v0.29.0rc1; closes
[#6471](https://github.com/vllm-project/vllm-omni/issues/6471).

> [!NOTE]
> Every number in this post is a PR-review-thread measurement — the author on
> 1× A800-80G, the reviewer on an L20X and an H200. This cookbook has not yet
> run its own SenseNova traces, so treat these as upstream-reported evidence
> with commit SHAs, not cookbook benchmarks.

## Background {#background}

**This section establishes what was actually broken when the PR opened: not
loading, not running — two silent problems, one of which made a shipped feature
a no-op.** Like a kitchen where the new espresso machine powers on, accepts
your order, and dispenses hot water: everything looks like it works, and only a
side-by-side taste test reveals nothing happened.

[SenseNova-U1.5-8B-MoT](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)
is SenseNova's unified image generation + understanding model — text2img,
img2img, img2text, and chat in one checkpoint. MoT means
Mixture-of-Transformers: alongside the understanding weights, the checkpoint
carries a second, generation-specialised set of projections (the `*_mot_gen`
parameters). It is big — 13 shards, 50.2 GB on disk (30.3 GB of it fp32),
loading to roughly 34 GB in bf16.

The "support" part was genuinely easy, and the PR is honest about it: U1.5 keeps
`model_type: neo_chat`, so it resolves on the existing `SenseNovaU1Pipeline`
with no `--model-class-name`; only two `config.json` fields flip versus U1
(`use_pixel_head` → `true`, turning the flow-matching head into a
`ConvDecoder`, and `noise_scale_max_value` 8.0 → 16.0), and both were already
read by `SenseNovaU1Config`. That left the two real defects:

1. **The distilled 8-step LoRA was a silent no-op.** The vendor ships
   `SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors`, a DMD-distilled adapter that
   buys 50-step quality in 8 guidance-free steps. On `main`,
   `--lora-backend distill` was accepted and then warned
   `Pipeline does not support loading distilled LoRA weights for now` — and
   generated with the base weights. The tell in the PR's own A/B: on `main`,
   runs with and without `--lora-path` differ by +0.48%, inside run-to-run
   noise. Nothing was applied.
2. **Nobody had measured where think-mode time went.** The model's autoregressive
   reasoning pass ("think") runs a hundreds-step token loop inside one pipeline
   forward — and that loop had never been profiled under vLLM-Omni.

There was also a trap for early adopters worth knowing when reading old issues:
on the v0.26.0 docker, U1.5 serving crashed at startup
(`'_DiffusionVllmModelConfig' object has no attribute 'head_dtype'` —
[#5795](https://github.com/vllm-project/vllm-omni/issues/5795)). That was a
config plumbing bug fixed by
[#5877](https://github.com/vllm-project/vllm-omni/pull/5877) (in v0.28.0), not
this PR — this PR is the model's *official* support, recipe, and LoRA path.

## What the PR does {#key-changes}

**Three acts: make the adapter actually apply, make equal-step generation
slightly faster and more accurate, and — after a review that escalated into an
Nsight profiling session — rebuild the autoregressive decode loop around a
paged KV cache and CUDA graphs.** If act one is fixing the turbo's fuel line,
act three is discovering the whole fuel system was a garden hose.

### Act 1 — the LoRA that loaded but never fired {#act1-lora}

Implementing `load_lora_weights` on `SenseNovaU1Pipeline` (via the shared
`LoraLoaderMixin`) immediately hit a loader bug that explains the "twice the
height" mystery. Deltas for fused projections are assembled by matching
*stacked params mappings* — rules that say "this checkpoint name contributes
that shard of this fused parameter". The old matcher used substring
containment:

```python
# vllm_omni/diffusion/lora/loader.py — before
if param_name not in base_key:   # ".qkv_proj" IS "in" "...qkv_proj_mot_gen"
    continue                     # …and so is ".qkv_proj_mot_gen" — both fire
```

`.qkv_proj` is a substring of `.qkv_proj_mot_gen`, so for a generation-tower
parameter *both* rules fired and the delta came out **twice the height of the
parameter**. The fix matches on the tail and stops at the first hit:

```python
# vllm_omni/diffusion/lora/loader.py — after (PR #6516)
if not base_key.endswith(param_name):
    continue
...
break                            # first matching rule wins
```

The pipeline side declares the mapping with the more specific `_mot_gen`
patterns first, renames the kohya-style checkpoint keys
(`lora_down`/`lora_up` → `lora_A`/`lora_B`), and folds the adapter in fp32 —
scaling `B` in bf16 would round once before the matmul and once after. The
result: 588 LoRA keys fuse into 168 parameters, TP-sharded through each
layer's own weight loader (into a zeroed copy, then added — so a rank only
ever touches its slice; this is also what fixed a TP=2 startup crash the
review caught). Two deliberate honesty details: a LoRA that matches *nothing*
now raises instead of no-op'ing, and only a sentinel is retained afterwards —
the fp32 state dict is ~1.5 GiB and would otherwise stay resident forever,
because fusion is one-way.

![The substring bug: .qkv_proj is contained in .qkv_proj_mot_gen, so both stacked-params rules fire and the fused delta comes out twice the parameter height; after PR #6516 the match is on the name tail and stops at the first hit]({{ site.baseurl }}/assets/figures/pr-6516-sensenova-u15-distilled-lora/fig1-lora-substring-bug.svg)

### Act 2 — one line of RMSNorm {#act2-rmsnorm}

`Qwen3RMSNorm.forward` (U1.5's language tower is Qwen3-based) used an eager
cast chain that rounded to bf16 *before* multiplying by the weight.
`F.rms_norm` keeps the accumulation in fp32 internally and rounds once:

```python
return F.rms_norm(hidden_states, self.weight.shape, self.weight,
                  self.variance_epsilon)
```

Against a float64 reference the mean relative error drops in all 12
shape/dtype combinations tested — bf16 4096×3584: 1.890e-3 → **1.409e-3**;
fp16 16384×8192: 2.368e-4 → **1.761e-4** — and equal-step latency improves
~6–8% as a side effect. A regression test pins the accuracy claim against the
float64 reference.

### Act 3 — the review that found the kinked hose {#act3-decode}

The reviewer's Nsight Systems profile (one 1024² step, third forward after
warmup, L20X) reframed the PR: the CFG-1 denoise path launches **1,877 kernels
for ~34.1 ms of GPU work in 71–76 ms of host wall time**. Kernels average
8.9 µs; ~5 µs go to each launch API and 31–35 µs between launches. GEMMs are
only 11–15% of end-to-end wall — this is host-dispatch-bound, not
device-compute-bound, and "GEMM tuning has a low ceiling here".

The author's corrected attribution narrowed it further: GPU busy is 92.0% for
text-to-image but **68.5% for think**, and the idle sits in AR decode — 8.7%
`cudaLaunchKernel`, 0.2% sync, and **~86% inside no CUDA call at all**
(Python/ATen dispatch). That rules out sync fixes and selects CUDA Graphs
(record the launch sequence once, replay it). The blocker was the KV cache:
`DynamicCache.update` grows K/V with `torch.cat` every step, so decode has no
static shape to capture. The obvious fix — pad to a bucket and mask the tail —
was measured and rejected: the mask costs 7.96–11.55 ms per step against
1.03 ms unmasked.

The answer is a **paged KV cache**: `flash_attn_varlen_func` accepts the used
length as a *tensor* (`seqused_k`) alongside a `block_table`, so the buffers
stay bucket-sized and capturable while the kernel reads only the valid prefix —
one capture serves every sequence length in the bucket. On top of it, three
cheap hoists that the profile justified (build the 3D RoPE tables once per
forward instead of 1,008 times for 17 distinct tables; stop expanding K/V to
the query head count before the backend, which hides the GQA shape from SDPA's
fused-kernel check; drop the all-zeros decode mask). Buckets are
(512, 1024, 2048, 4096, 8192) and grow in steps past the last entry; a think
request typically re-captures once when it crosses 512.

![Why CUDA graphs needed a paged cache: torch.cat growth changes shapes every step so nothing is capturable; bucket-plus-mask keeps shapes static but the mask costs 7.96-11.55 ms per step against 1.03 ms unmasked; the paged cache keeps bucket-sized buffers and lets flash_attn_varlen_func read only the valid prefix via seqused_k, so one capture serves the whole bucket]({{ site.baseurl }}/assets/figures/pr-6516-sensenova-u15-distilled-lora/fig2-paged-decode.svg)

Everything else in the act is the review conversation made concrete —
including what was *tried and dropped*, which is the most useful table in the
thread:

| Change | Verdict | Evidence |
|---|---|---|
| Build 3D RoPE tables once per forward | kept | 1,008 calls produced 17 distinct tables |
| Stop expanding K/V before the backend | kept | +2.55% alone; pays once the mask is gone |
| Drop the all-zeros decode mask | kept | −5.66% / −8.19% together with the row above |
| Paged decode under a CUDA graph | kept | **−29.93% / −31.27%**, switch-only A/B, n=5 |
| `_repeated_blocks` regional compile | dropped | launches 19,974 → 19,974; hits `recompile_limit (8)`, falls back to eager, +2.9% |
| Removing the forced `sdpa_fallback` | dropped | init dies — other backends invert a boolean mask, SenseNova passes an additive float one |
| Removing the `.item()` sync | dropped | 0.2% of the idle |
| Bucketing plus a mask | dropped | 7.96–11.55 ms/step vs 1.03 ms unmasked |

Review also caught two things the first version got wrong, both fixed in-PR:
`_generate_text` — the T2T/I2T decode loop — was calling `_ar_step` *without*
the decode context `_generate_think` creates, so the recipe claimed paged/graph
acceleration for image-to-text that the path never reached; and a GQA accuracy
gate asserting 8/8 fixed-seed wins over the reference did not hold across a
supported CUDA stack (7/8 at `kv_len=512` on L20X + cu129) and became an
explicit no-regression tolerance.

Finally, the PR introduced — and then fixed — its own VRAM leak: the decode
runner was originally built per request, so every request captured its own
graph and kept it, leaking ~40 MiB of device memory per request without
levelling off. The fix captures into the shared platform pool and reuses the
cache and runner across requests (down to 0.2 MiB/request), releasing captures
on sleep level 2. Reuse is deliberately disabled when a dynamic LoRA manager
holds the decode path — an adapter bound between requests changes nothing the
reuse check can see.

## Key changes {#diff-walkthrough}

**The diff is 20 files, ~2,000 added lines — but only 86 of the last ~460 were
implementation; the rest is tests (258), the recipe and docs (116).** The
implementation splits into the LoRA path, the decode path, and the plumbing
around them.

- [`vllm_omni/diffusion/lora/loader.py`](https://github.com/vllm-project/vllm-omni/blob/0288c3f56eb4a4ff410194e586498e3bc8ff8362/vllm_omni/diffusion/lora/loader.py) —
  `_prepare_lora_delta` matches on the tail (`endswith`) and breaks at the
  first hit; mypy annotations added. QwenImage and Wan2.2 fused names do not
  overlap, so they select the same shards as before — pinned by a regression
  test.
- [`vllm_omni/diffusion/models/sensenova_u1/pipeline_sensenova_u1.py`](https://github.com/vllm-project/vllm-omni/blob/0288c3f56eb4a4ff410194e586498e3bc8ff8362/vllm_omni/diffusion/models/sensenova_u1/pipeline_sensenova_u1.py) —
  gains `LoraLoaderMixin`, `stacked_params_mapping` (`_mot_gen` patterns
  first), `load_lora_weights`, and the decode machinery: `_decode_context()`
  builds-or-reuses `(PagedDecodeCache, DecodeGraphRunner)`,
  `release_captured_graphs()` drops them on sleep level 2, `_ar_step()` grows
  the bucket and replays the graph, and both `_generate_think` *and*
  `_generate_text` run inside the context. `_warm_ar_decode()` runs one
  single-token decode at startup because the engine's dummy request is a
  think-off T2I run that exercises prefill but never the decode shape.
- [`vllm_omni/diffusion/models/sensenova_u1/paged_decode.py`](https://github.com/vllm-project/vllm-omni/blob/0288c3f56eb4a4ff410194e586498e3bc8ff8362/vllm_omni/diffusion/models/sensenova_u1/paged_decode.py) —
  new: the paged cache (block table, `seqused_k`, bucket growth) and the CUDA
  graph runner, plus `paged_decode_supported()` and
  `dynamic_lora_wrappers_present()` gates.
- [`vllm_omni/diffusion/models/sensenova_u1/sensenova_u1_transformer.py`](https://github.com/vllm-project/vllm-omni/blob/0288c3f56eb4a4ff410194e586498e3bc8ff8362/vllm_omni/diffusion/models/sensenova_u1/sensenova_u1_transformer.py) —
  `F.rms_norm`, hoisted 3D RoPE (`_build_3d_rope`: t gets half the head dim,
  h/w a quarter each), K/V kept at their real head count, and the
  single-token paged attention branch in the decoder layer.
- [`vllm_omni/config/environment_variable_inventory.py`](https://github.com/vllm-project/vllm-omni/blob/0288c3f56eb4a4ff410194e586498e3bc8ff8362/vllm_omni/config/environment_variable_inventory.py) —
  registers `VLLM_OMNI_SENSENOVA_PAGED_DECODE` (default on; `0` forces the
  ordinary-cache fallback).
- Tests: 16 new files/hunks across distill-LoRA fusion, loader substring
  regression, RMSNorm float64 reference, paged decode end-to-end (bucket
  growth, GQA capability probes); `tests/diffusion/models/sensenova_u1/`
  stands at 86 passing, and the author verified the suite by reverting pieces
  (each revert turns exactly the matching tests red).

## Measured impact {#measured-impact}

**Four independent measurement sets exist: the author's A800 runs, the
reviewer's L20X and H200 validations, and a per-stage attribution that isolates
which change bought which second.** Like a receipt itemised by department
rather than one total — you can see exactly where each saving came from.

Whole-PR, `main` → merged head (author, 1× A800-80G, seed 42, n=5 medians, one
session):

| Case | main | PR #6516 | Δ |
|---|---:|---:|---:|
| Think-mode E2E 1024², 50 steps (P50 of 10, think on) | 27.747 s | 13.930 s | **−49.8%** |
| T2I 1024², 8 steps + distilled LoRA | 1,120.8 ms | 924.4 ms | **−17.52%**\* |
| T2I 1024², 50 steps, CFG 4.0 | 11,862.3 ms | 10,560.7 ms | −10.97% |
| T2I 1536², 50 steps, CFG 4.0 | 25,779.6 ms | 23,131.7 ms | −10.27% |

\* user-facing, not like-for-like: `main` silently drops the adapter.

Per-stage attribution (author, 1× A800, think on, 2 warmups + N=10, P50;
`stage_metrics` reports a single `diffusion` stage for this model because the
whole AR loop runs inside one pipeline forward):

| Metric | main (eager) | PR paged=0 | PR paged=1 | PR paged=1 + LoRA (8 steps) |
|---|---:|---:|---:|---:|
| E2E | 27.747 s | 16.500 s | 13.930 s | **4.455 s** |
| AR prefill | 69.0 ms | 38.9 ms | 39.3 ms | 39.0 ms |
| think decode | 16.146 s | 6.213 s | 3.534 s | 3.522 s |
| diffusion execution | 11.460 s | 10.251 s | 10.317 s | 0.854 s |

Isolated: `main → paged=0` (loading/fusion, RMSNorm, RoPE hoists) is
**−40.53% E2E** (decode −61.52%, diffusion −10.56%); `paged=0 → paged=1`
(paged KV + CUDA graph) is **−15.57% E2E** (decode −43.11%). The dispatch
entry points tell the same story per path: image-edit −20.25%, image-to-text
−42.04%, text-to-text −44.51% with the switch alone.

Why the graph, quantified — three arms differing by one thing each (think on,
8 steps; kernel time from a profiled run, wall from a clean one):

| Arm | KV cache | attention | graph | wall | kernel | idle | GPU busy |
|---|---|---|---|---:|---:|---:|---:|
| A `paged=0` | exact length | SDPA flash | no | 8,259.5 ms | 5,179.2 ms | 3,080.3 ms | 62.71% |
| B | paged | `flash_attn_varlen` | no | 8,323.3 ms | 5,136.6 ms | 3,186.7 ms | 61.71% |
| C `paged=1` | paged | `flash_attn_varlen` | yes | 5,196.9 ms | 5,076.9 ms | **120.0 ms** | **97.69%** |

![The three decode arms: switching to a paged cache alone (A to B) changes nothing, but capturing the decode loop into a CUDA graph (B to C) collapses idle time from 3,186.7 ms to 120.0 ms — kernel time is flat, so the entire win is host dispatch]({{ site.baseurl }}/assets/figures/pr-6516-sensenova-u15-distilled-lora/fig3-decode-arms.svg)

Kernel time is flat across arms — the idle that disappears is host time, not
device work; `think_chars` is 1,233 in every run. On the decode loop A→B is
−1.10% and B→C is −45.79%: *the paged cache buys the static shape, the graph
buys the time.* Host CPU overhead of the whole path drops from 37.29% before
the graph to 2.31% (AR window) / 3.97% (diffusion window). With both windows
above 96% busy, the remaining mix is GEMM-dominated (AR decode: 82.4% GEMM,
6.4% elementwise, 5.5% attention, 5.4% norm; diffusion: 78.9/9.2/6.8/4.9) —
which is why the next lever upstream is quantisation and TeaCache, not more
launch reduction.

Launch count, the reviewer's original criterion (one profiled 8-step T2I
generation): `cudaLaunchKernel` **24,244 → 7,782 (−67.9%)**, GPU kernels
25,924 → 8,232, GPU busy 86.1% → 93.1%. `cudaGraphLaunch` is 0 on both legs —
T2I never runs decode; this reduction is the RoPE hoist, uncompressed K/V, and
the dropped mask.

Tensor parallelism (2× A800 over SYS/PCIe, no NVLink): think 20,678.3 →
5,221.0 ms (**−74.75%**), 8-step LoRA T2I 1,234.7 → 1,127.8 ms (−8.66%),
bit-identical output; the T2I leg is measured from the TP-sharding fix
onward because the configuration crashed at startup before it. Reviewer-side
on L20X (three measured requests after one warmup): graph off → on at the PR
head, 13.900 → 7.566 s (**−45.57%**); NVML peak +366 MiB (+1.01%) at TP=1,
+402 MiB/GPU (+2.02%) at TP=2, inside the 5% regression gate.

H200 validation (reviewer, 1× H200 139 GiB, vLLM 0.28.0 / torch 2.13.0+cu130,
seed 42, 1024², single runs):

| Case | Stage latency | Peak GPU memory |
|---|---:|---:|
| 50 steps, CFG 4.0, think off | 9,017 ms | 34,384 MiB |
| 8 steps, CFG 1.0, no-LoRA control | 624.00 ms (77.85 ms/step) | 34,376 MiB |
| 8 steps, CFG 1.0, distilled LoRA | 623.85 ms (77.98 ms/step) | 34,364 MiB |
| 50 steps, CFG 4.0, think on, paged decode | 81,622 ms | 34,594 MiB |

The LoRA row is the no-op fix made visible: **fused into 168 parameters**, and
its same-seed image differs from the no-LoRA control in 1,048,574 of
1,048,576 pixels (MAE 58.62) — the adapter is not just decorative. The
think-on row carries the first-request compile (see below). Memory cost of the
paged path itself, switch-only: peak VRAM +864 MiB (+2.46%) with the schedule
pinned to the 8192 bucket, steady-state host RSS +23 MiB. Graph capture costs
30–34 ms when it happens, and reuse works across requests — five edit
requests shared one capture; a think request captures twice (the 512-bucket
crossing).

The accuracy ledger for the fusions: RMSNorm error drops in all 12
shape/dtype combos (table above); CFG-parallel output stays byte-identical to
single GPU; T2I output is bit-identical between the pre-leak-fix commit and
the merged head. The one real drift: paged FA2 and SDPA are *equally* accurate
against a float64 reference (mean error identical to four significant
figures) yet agree with each other only to 7.63e-06 — in a 382-step argmax
loop that eventually flips a token, so think-mode *text* can differ between
the two backends. The first differing steps measured top1–top2 margins of
0.000 and 0.125 against a `|Δlogit|` of 0.188 and 0.164 — genuine greedy-tie
breaks, not corruption. All four T2I cases stayed bit-identical.

## How to use it {#how-to-use}

{% include usage-cookbook.html modes=page.usage %}

Release note: full U1.5 support (this PR) first shipped in
[v0.29.0rc1](https://github.com/vllm-project/vllm-omni/releases/tag/v0.29.0rc1)
(2026-09-10). On v0.26.x, serving crashes with the `head_dtype` error from
Background — v0.28.0 carries the config fix if you are stuck between.

Three operating details worth knowing before your first run:

- **The LoRA wants CFG 1.0.** It is DMD-distilled to run guidance-free; CFG 4.0
  with 8 steps produces a blown-out, posterised image. The adapter targets the
  generation tower only — understanding behaviour is untouched.
- **Paged decode is on by default** and falls back to the ordinary cache when
  the device or the bundled `flash_attn_varlen_func` cannot support it.
  `VLLM_OMNI_SENSENOVA_PAGED_DECODE=0` forces the fallback for debugging or
  output-parity checks.
- **The first request after startup costs ~0.7 s more** than steady state
  (measured 718 ms with the paged path on, 679 ms with it off, compile caches
  cleared, median of 3) — that is the compiled decode region warming up, and
  the engine now runs a best-effort decode warmup at startup instead of
  leaving it to the first user request.

The full commands, hardware notes, and both validation environments (A800 and
H200) live in the upstream
[SenseNova-U1.5 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/SenseNova/SenseNova-U1.5.md).

## How to choose {#decision-cards}

{% include decision-cards.html items=page.decisions %}

## Limitations and follow-ups {#limitations}

- **Graph capture is per request, not per engine.** Each request captures its
  own graphs (30–34 ms each; a think request captures twice) because the
  captured addresses bind to the cache instance — reuse saves memory, not the
  capture itself. Plus the ~0.7 s first-request compile cost above.
- **Think-mode text is not bit-stable across backends.** Paged FA2 vs SDPA can
  flip greedy-tie tokens (7.63e-06 agreement); images are unaffected. If you
  need exact text parity, force the fallback env var. The related
  [#4636](https://github.com/vllm-project/vllm-omni/issues/4636) tracks
  think-mode generation drift breaking pixel-golden tests.
- **Distill fusion is one-way** — no unload, generation tower only, one
  adapter file per load, and a runtime LoRA manager disables graph/cache
  reuse. The official U1 LoRA checkpoints
  ([#5642](https://github.com/vllm-project/vllm-omni/pull/5642)) remain open.
- **Validated hardware is 80 GB-class** (A800, H200, L20X). ~34 GB peak at
  1024² in bf16 means 40 GB cards are untested territory; nothing smaller has
  an upstream recipe.
- **The Preview checkpoint question is open.**
  [#5795](https://github.com/vllm-project/vllm-omni/issues/5795) deployed
  `SenseNova-U1.5-8B-MoT-Preview` on v0.26.0 and hit the `head_dtype` crash;
  the fix (#5877) landed in v0.28.0 but nobody has confirmed the Preview
  checkpoint end-to-end — the recipe covers `SenseNova-U1.5-8B-MoT`.
- **Idle time is now genuinely device time.** With both windows >96% busy and
  GEMMs at ~79–82% of kernel time, the next wins are quantisation and
  TeaCache-style step skipping, not more host work: per-branch TeaCache for
  SenseNova-U1 is open in
  [#6660](https://github.com/vllm-project/vllm-omni/pull/6660)
  ([#5287](https://github.com/vllm-project/vllm-omni/pull/5287) for the
  multi-branch CFG state), regional compilation in
  [#4732](https://github.com/vllm-project/vllm-omni/pull/4732) — the
  experiment this PR measured and dropped. Also open for the family: online
  dynamic batching
  ([#4156](https://github.com/vllm-project/vllm-omni/pull/4156)), AR+DiT
  separation ([#4033](https://github.com/vllm-project/vllm-omni/pull/4033)),
  and streaming output for image understanding
  ([#4049](https://github.com/vllm-project/vllm-omni/issues/4049)).
- Related reading on the distillation side: the
  [MiniMax-H3 few-step schedules post]({{ site.baseurl }}/2026-08-24-understanding-pr-5991-minimax-h3-few-step-schedules/)
  covers the DMD2 contract family that makes "8 steps, CFG 1" a distillation
  property rather than a sampling trick.

## References {#references}

- [PR #6516 — Support SenseNova-U1.5-8B-MoT and its distilled 8-step LoRA](https://github.com/vllm-project/vllm-omni/pull/6516) (merged 2026-09-01, commit [`0288c3f`](https://github.com/vllm-project/vllm-omni/commit/0288c3f56eb4a4ff410194e586498e3bc8ff8362))
- [Issue #6471 — New Model: SenseNova-U1.5-8B-MoT](https://github.com/vllm-project/vllm-omni/issues/6471) (closed by this PR)
- [Issue #5795 — deploying SenseNova-U1.5-8B-MoT-Preview on v0.26.0](https://github.com/vllm-project/vllm-omni/issues/5795) (open; root cause fixed by #5877 in v0.28.0)
- [PR #5877 — Fix SenseNova & use well-defined model configs](https://github.com/vllm-project/vllm-omni/pull/5877) (the `head_dtype` fix, merged 2026-08-18)
- [SenseNova-U1.5 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/SenseNova/SenseNova-U1.5.md) (upstream, current — includes the A800 and H200 validation sections)
- [Supported models entry](https://github.com/vllm-project/vllm-omni/blob/main/docs/models/supported_models.md) · [diffusion features table](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion_features.md) (upstream docs listing U1.5)
- The review-thread measurements: the [reviewer's Nsight profile](https://github.com/vllm-project/vllm-omni/pull/6516#issuecomment-5386693054), the [author's kept/dropped experiments and whole-PR A/B](https://github.com/vllm-project/vllm-omni/pull/6516#issuecomment-5398538652), the [L20X re-verification](https://github.com/vllm-project/vllm-omni/pull/6516#issuecomment-5403824092), the [stage attribution](https://github.com/vllm-project/vllm-omni/pull/6516#issuecomment-5469226403), the [CPU-overhead and kernel-mix answer](https://github.com/vllm-project/vllm-omni/pull/6516#issuecomment-5475061734), and the [H200 validation](https://github.com/vllm-project/vllm-omni/pull/6516#issuecomment-5487475554)
- [MiniMax-H3 few-step schedules post — PR #5991]({{ site.baseurl }}/2026-08-24-understanding-pr-5991-minimax-h3-few-step-schedules/) (DMD distillation background)
