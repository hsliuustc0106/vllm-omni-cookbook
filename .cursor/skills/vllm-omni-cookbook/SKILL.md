---
name: vllm-omni-cookbook
description: >-
  Contribute release-by-release performance summaries to vllm-omni-cookbook:
  per-model improvement ledgers (index.md) and cross-model release tables (SUMMARY.md).
  Use when documenting what improved at vX.Y.Z, updating perf deltas, or adding a new model ledger.
---

# vLLM-Omni Performance Cookbook — Contributor Guide

This repo summarizes **model performance improvements at every even stable release** — not deployment how-tos. vLLM-Omni owns recipes and benchmarks; here we publish **what changed, by how much, and why** (PR links).

**Even releases only:** add cookbook sections for v0.18.0, v0.20.0, v0.22.0, … Skip odd minors. Deltas compare to the **previous even** release.

Each `{category}/{model}/index.md` is the model's full improvement timeline. `SUMMARY.md` is the per-release cross-model snapshot.

## Before you start

1. Read [README.md](../../README.md) and [docs/DESIGN.md](../../docs/DESIGN.md).
2. Do **not** fork upstream perf JSON here — link or copy configs under `vllm-omni/benchmark_results/`.
3. Every comparison must record: commit SHAs, `vllm` + `vllm_omni` versions, GPU SKU/count, serve args, workload dimensions.
4. Retro hardware on this cluster is **NVIDIA H200** (internal logs may label it **L20X** — same SKU; document as H200 in the cookbook).

## Choose a workflow

| Goal | Skill |
|------|-------|
| First-time model in the cookbook | [cookbook-add-model](../cookbook-add-model/SKILL.md) |
| New even stable release (v0.22.0, …) | [cookbook-add-release](../cookbook-add-release/SKILL.md) |
| Optional Zhihu draft | [cookbook-write-narrative](../cookbook-write-narrative/SKILL.md) |

## Category → metrics

| Category | Folder | Primary metrics |
|----------|--------|-----------------|
| Omni | `omni/` | TTFT, TTFP, TPOT, RTF, E2EL, throughput |
| Diffusion | `diffusion/` | E2E latency (`latency_mean`), optional stage profiler |

## Repo layout

```
vllm-omni-cookbook/
├── SUMMARY.md                 # per-release cross-model snapshot
├── omni/{model}/index.md
└── diffusion/{model}/
    ├── index.md               # slim reference ledger
    └── *-performance-zhihu.md # optional Chinese draft
```

Canonical examples: `diffusion/wan2.2/`, `omni/qwen3-omni/`.

## PR checklist

- [ ] Numbers come from a completed retro run or CI JSON (not estimates)
- [ ] Footnotes explain non-comparable rows (missing CLI, different GPU count, sync vs async API)
- [ ] `SUMMARY.md` updated if adding a release section
- [ ] Links to upstream recipe, perf JSON, and relevant PRs
- [ ] No duplicate full PR lists in both `index.md` and Zhihu draft

## Templates

See [references/templates.md](references/templates.md) for `index.md`, `SUMMARY.md`, and PR body templates.
