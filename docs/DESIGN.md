# Performance Cookbook — Design

**Purpose:** Per-model performance improvement summary at each **even** vLLM-Omni stable release (v0.14, v0.16, v0.18, v0.20, v0.22, …). Odd minors are skipped.

## Structure

```
vllm-omni-cookbook/
├── README.md
├── SUMMARY.md              # cross-model snapshot per even release
├── omni/{model}/index.md   # per-model improvement timeline
├── diffusion/{model}/
│   ├── index.md
│   ├── assets/             # charts (optional)
│   └── *-zhihu.md          # Chinese republish draft (optional)
└── blog/                   # self-contained Jekyll site (see below)
    ├── _config.yml
    ├── Gemfile
    ├── _includes/custom-head.html  # MathJax on pages with math: true
    ├── _posts/             # YYYY-MM-DD-<slug>.md PR-analysis posts
    ├── assets/figures/     # per-post figures in <slug>/
    └── TEMPLATE.md         # canonical post template
```

## Model file format (`index.md`)

1. Header — model id, upstream recipe, perf JSON / retro harness links
2. Key metrics + CI / retro tracks
3. Retro comparison tables (with footnotes)
4. `## vX.Y.Z` sections per **even** release (newest first) — metrics, Δ vs prior even release, optimization PRs
5. Serve / reproduce commands

## SUMMARY.md format

One `## vX.Y.Z` block per even release: headline metric row per tracked model + short highlights linking to model indexes.

## Blog format (`blog/`)

Deep-dive "PR Analysis" posts explaining one important upstream PR/feature: TL;DR → Background → What the PR does → Key changes → Measured impact → How to use → Limitations → References.

- **Self-contained Jekyll site** (same stack as `vllm-project.github.io`: Minima remote theme pinned, `jekyll-feed`/`jekyll-seo-tag`/`jekyll-gfm-admonitions`), so `blog/` can move to a dedicated repo unchanged.
- Posts: `blog/_posts/YYYY-MM-DD-<slug>.md`, front matter per `blog/TEMPLATE.md` (`summary` ≤ 240 chars, `category: PR Analysis`, tags = model + area). No fixed cadence.
- Metrics come **only** from cookbook ledgers or upstream perf JSON; cookbook links are absolute GitHub URLs (the blog deploys standalone).
- Deployed to GitHub Pages by `.github/workflows/blog-pages.yml` on pushes touching `blog/**`.

## Out of scope

- Deployment how-tos (vLLM-Omni `recipes/`)
- Benchmark JSON and retro harnesses (vLLM-Omni `tests/dfx/perf/`, `benchmark_results/`)
- Odd-minor release sections

## Tracked models (v0.20 cookbook)

| Model | Path |
|-------|------|
| Qwen3-Omni | `omni/qwen3-omni/` |
| WAN2.2 | `diffusion/wan2.2/` |
| Qwen-Image | `diffusion/qwen-image/` |
