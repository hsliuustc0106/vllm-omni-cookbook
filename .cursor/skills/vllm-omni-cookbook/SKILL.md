---
name: vllm-omni-cookbook
description: >-
  Contribute performance ledgers to vllm-omni-cookbook: add models, document
  release deltas, write Zhihu narratives. Use when contributing to the cookbook,
  updating SUMMARY.md, adding index.md ledgers, or documenting vLLM-Omni perf across releases.
---

# vLLM-Omni Performance Cookbook — Contributor Guide

This repo is a **performance ledger**, not a how-to cookbook. vLLM-Omni owns recipes and benchmark JSON; this repo records **measured deltas** across stable releases.

## Before you start

1. Read [README.md](../../README.md) and [docs/plans/2026-05-09-performance-cookbook-design.md](../../docs/plans/2026-05-09-performance-cookbook-design.md).
2. Do **not** fork upstream perf JSON here — link or copy configs under `vllm-omni/benchmark_results/`.
3. Every comparison must record: commit SHAs, `vllm` + `vllm_omni` versions, GPU SKU/count, serve args, workload dimensions.
4. Retro hardware on this cluster is **NVIDIA H200** (internal logs may label it **L20X** — same SKU; document as H200 in the cookbook).

## Choose a workflow

| Goal | Skill |
|------|-------|
| First-time model in the cookbook | [cookbook-add-model](../cookbook-add-model/SKILL.md) |
| New stable release (vX.Y.Z) for existing models | [cookbook-add-release](../cookbook-add-release/SKILL.md) |
| Optional Zhihu draft | [cookbook-write-narrative](../cookbook-write-narrative/SKILL.md) |

## Category → metrics

| Category | Folder | Primary metrics |
|----------|--------|-----------------|
| Omni | `omni/` | TTFT, TTFP, TPOT, RTF, E2EL, throughput |
| TTS | `tts/` | TTFP, RTF, throughput (note concurrency c) |
| Diffusion | `diffusion/` | E2E latency (`latency_mean`), optional stage profiler |

## Repo layout

```
vllm-omni-cookbook/
├── SUMMARY.md                 # cross-model release table (update every release)
├── omni/{model}/index.md
├── tts/{model}/index.md
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
