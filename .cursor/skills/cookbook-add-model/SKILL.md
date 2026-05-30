---
name: cookbook-add-model
description: >-
  Add a new model to vllm-omni-cookbook with a release-by-release improvement ledger
  (index.md). Use when onboarding a model not yet tracked under omni/ or diffusion/.
---

# Add a New Model Ledger

Create `{category}/{model}/index.md` as the model's **improvement timeline** — one `## vX.Y.Z` section per release going forward, each summarizing metrics, deltas, and optimization PRs for that release.

## Prerequisites

- Model is supported in [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) with a recipe under `recipes/`
- Upstream perf JSON exists under `tests/dfx/perf/tests/` **or** you will add a retro harness under `benchmark_results/{model_slug}_retro/` first (in vllm-omni, not here)

## Step 1 — Intake

Fill this checklist (copy into the new `index.md` draft or retro README in vllm-omni):

| Field | Example |
|-------|---------|
| `model_slug` | `qwen-image` |
| HuggingFace id | `Qwen/Qwen-Image` |
| Category | `diffusion` / `omni` |
| Task | `t2i`, `i2v`, `voice_clone`, `text+audio`, … |
| Upstream perf JSON | `tests/dfx/perf/tests/test_qwen_image_vllm_omni.json` |
| Primary metric | `latency_mean` / `median_audio_ttfp_ms` / `mean_ttfp` |
| CI hardware | 2× H100 |
| Retro hardware | 4× H200 |
| Release tags to compare | v0.18.0, v0.20.0, v0.22.0 (even minors only) |

## Step 2 — Scaffold cookbook dirs

```bash
cd vllm-omni-cookbook
CATEGORY=diffusion   # or omni
MODEL=qwen-image     # kebab-case folder name

mkdir -p "${CATEGORY}/${MODEL}/assets"
touch "${CATEGORY}/${MODEL}/assets/.gitkeep"
```

## Step 3 — Write `index.md`

Follow [references/templates.md](../vllm-omni-cookbook/references/templates.md#indexmd). Required sections:

1. **Header** — category, HF id, links to upstream recipe + perf JSON + retro README
2. **Key metrics** — definitions table (lower/higher better, when it matters)
3. **Performance tracks** — CI vs retro vs dashboard rows
4. **Standardized perf test** — CI baselines, reproduction command
5. **Retro comparison** — main table (can be “TBD” until measured)
6. **Optimization summary** — phase → release → PR links (start sparse)
7. **Serve commands** — minimal, match perf JSON

Copy tone and structure from the closest existing model:

| Category | Template |
|----------|----------|
| Diffusion video | `diffusion/wan2.2/index.md` |
| Diffusion image | `diffusion/qwen-image/index.md` |
| Omni | `omni/qwen3-omni/index.md` |

## Step 4 — Update root docs

- Add the model row to [README.md](../../README.md) “Models Tracked” table
- Add placeholder rows to [SUMMARY.md](../../SUMMARY.md) for the next **even** release section (`—` until measured)

## Step 5 — Optional narrative layer

Only after retro tables exist — see [cookbook-write-narrative](../cookbook-write-narrative/SKILL.md).

## Do not

- Add how-to deployment recipes (link upstream instead)
- Invent workloads not in upstream JSON without footnotes
- Claim deltas without measured numbers and commit SHAs
