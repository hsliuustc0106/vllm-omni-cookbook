---
name: cookbook-add-release
description: >-
  Append improvement summary for an even vLLM-Omni release (v0.22.0, …): add vX.Y.Z
  to each model index.md and update SUMMARY.md. Skip odd minors. Use when v0.22.0 ships
  or backfilling even-release summaries.
---

# Add a Release Improvement Summary

## Gate — even minor only

Update the cookbook **only** when the release minor version is **even** (v0.18.0, v0.20.0, v0.22.0, …).

| Minor (Y in v0.Y.Z) | Cookbook update? | Delta baseline |
|---------------------|------------------|----------------|
| Even (18, 20, 22) | **Yes** | Previous even release (20 → 18, 22 → 20) |
| Odd (19, 21, 23) | **No** — wait for next even | — |

Trigger: even tag `vX.Y.Z` is released **and** measured numbers exist (retro or CI).

Goal: document **this release's performance improvement** for each tracked model — what got better, by how much vs the **previous even** release, and which PRs explain it.

## Step 1 — Gather evidence

From each model's retro run in `vllm-omni`:

- `env_check.txt`: `git rev-parse --short HEAD`, `vllm` / `vllm_omni` versions
- Result JSON: primary metrics per `test_name`
- Previous release numbers for delta columns

Record footnotes for anything that breaks comparability (GPU count, missing serve flags, prompt count, warmup policy).

## Step 2 — Update each model `index.md`

Add a new section **below the header, above older releases** (newest first):

```markdown
## vX.Y.Z (YYYY-MM-DD)

### Performance

| Metric / config | Value | Δ vs vPREV |
|-----------------|------:|-----------:|
| ... | **X.XX** | **−Y.Y%** |

### Optimization notes

- [PR #NNNN](https://github.com/vllm-project/vllm-omni/pull/NNNN) — one-line effect
- ...

### Figures

![chart](assets/vX.Y.Z-latency.png)
```

Rules:

- Bold the metric values readers compare (`**26.0 s**`)
- Deltas are vs **immediately prior even release** on the **same hardware track** (not the last odd tag)
- Link PRs/issues; do not paste long changelogs
- If not measured yet, omit the section — use `—` in SUMMARY only

Also update the **retro comparison** anchor table if new tag columns were added.

## Step 3 — Update `SUMMARY.md`

Add `## vX.Y.Z (YYYY-MM-DD)` at the top (below any “upcoming” stub):

```markdown
## vX.Y.Z (YYYY-MM-DD)

| Model | Category | Key Metric | Value | Delta from vPREV |
|-------|----------|------------|-------|-------------------|
| WAN2.2 | diffusion | I2V E2E (4×H200) | **22.17 s** | **−5.9%** |
| ... | ... | ... | ... | ... |

### Highlights

- **Model:** one sentence tying metric to PR/theme + link to model index
```

One row per **headline metric** per model; extra configs go in model `index.md`.

## Step 4 — Assets

- Charts → `{category}/{model}/assets/vX.Y.Z-*.png`
- Reference in both release section and retro table if reused

## Step 5 — PR

Use checklist from [references/templates.md](../vllm-omni-cookbook/references/templates.md#pull-request).

Type: **Documentation improvement** / perf ledger update (not “New recipe”).

## Release index (optional compact block)

At bottom of `index.md`, keep a one-line-per-release index linking to anchors — see `diffusion/wan2.2/index.md`.
