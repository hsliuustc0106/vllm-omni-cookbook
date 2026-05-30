# Performance Cookbook — Design

**Purpose:** Per-model performance improvement summary at each **even** vLLM-Omni stable release (v0.14, v0.16, v0.18, v0.20, v0.22, …). Odd minors are skipped.

## Structure

```
vllm-omni-cookbook/
├── README.md
├── SUMMARY.md              # cross-model snapshot per even release
├── omni/{model}/index.md   # per-model improvement timeline
└── diffusion/{model}/
    ├── index.md
    ├── assets/             # charts (optional)
    └── *-zhihu.md          # Chinese republish draft (optional)
```

## Model file format (`index.md`)

1. Header — model id, upstream recipe, perf JSON / retro harness links
2. Key metrics + CI / retro tracks
3. Retro comparison tables (with footnotes)
4. `## vX.Y.Z` sections per **even** release (newest first) — metrics, Δ vs prior even release, optimization PRs
5. Serve / reproduce commands

## SUMMARY.md format

One `## vX.Y.Z` block per even release: headline metric row per tracked model + short highlights linking to model indexes.

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
