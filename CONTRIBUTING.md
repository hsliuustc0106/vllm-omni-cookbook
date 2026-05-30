# Contributing to vLLM-Omni Performance Cookbook

This repository tracks **measured performance and optimization deltas** for vLLM-Omni models across stable releases. It is **not** a how-to recipe book — deployment recipes and benchmark JSON live in [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni).

## Agent skills (recommended)

Cursor and compatible agents can load project skills from `.cursor/skills/`:

| Skill | When to use |
|-------|-------------|
| [vllm-omni-cookbook](.cursor/skills/vllm-omni-cookbook/SKILL.md) | Start here — overview and PR checklist |
| [cookbook-add-model](.cursor/skills/cookbook-add-model/SKILL.md) | First ledger for a new model |
| [cookbook-add-release](.cursor/skills/cookbook-add-release/SKILL.md) | New `vX.Y.Z` section + `SUMMARY.md` |
| [cookbook-write-narrative](.cursor/skills/cookbook-write-narrative/SKILL.md) | Optional Zhihu draft |

Templates: [.cursor/skills/vllm-omni-cookbook/references/templates.md](.cursor/skills/vllm-omni-cookbook/references/templates.md)

Claude Code users: see condensed stubs in [.claude/skills/](.claude/skills/).

## Getting started

1. Fork the repository
2. Create a branch (`git checkout -b docs/wan22-v021-retro`)
3. Run or cite benchmarks from vllm-omni (`benchmark_results/`, upstream perf JSON — see model `index.md` reproduce sections)
4. Update model `index.md` and `SUMMARY.md`
5. Open a pull request

## What to contribute

| Change | Files |
|--------|-------|
| New model | `{omni,tts,diffusion}/{model}/index.md`, `README.md`, `SUMMARY.md` placeholders |
| New release | Each tracked model's `index.md` + root `SUMMARY.md` |
| Retro numbers | Tables in `index.md` with SHAs, GPU SKU, footnotes |
| Optional Zhihu draft | `*-performance-zhihu.md` |

## Repository layout

```
omni/{model}/index.md      # omni-modal ledgers
tts/{model}/index.md       # TTS ledgers
diffusion/{model}/index.md # diffusion ledgers
SUMMARY.md                 # cross-model release overview
```

Each model folder may include `assets/` for charts.

## Evidence requirements

Every performance table must be traceable:

- Git commit SHA(s) for each compared release
- Installed `vllm` and `vllm_omni` versions
- GPU type and count (`CUDA_VISIBLE_DEVICES`)
- Upstream perf JSON path or `benchmark_results/{slug}_retro/` artifact
- Footnotes when rows are not apples-to-apples (missing CLI flags, different prompt counts, API backend changes)

Do **not** copy upstream perf JSON into this repo.

## Metrics by category

| Category | Folder | Primary metrics |
|----------|--------|-----------------|
| Omni | `omni/` | TTFT, TTFP, TPOT, RTF, E2EL |
| TTS | `tts/` | TTFP, RTF, throughput (note concurrency) |
| Diffusion | `diffusion/` | E2E latency (`latency_mean`) |

## Canonical examples

- [diffusion/wan2.2/index.md](diffusion/wan2.2/index.md) — fullest ledger + HTML + Zhihu
- [diffusion/qwen-image/index.md](diffusion/qwen-image/index.md) — compact T2I retro
- [omni/qwen3-omni/index.md](omni/qwen3-omni/index.md) — multi-stage omni metrics

## Pull request checklist

- [ ] Numbers from a completed retro run or CI JSON
- [ ] `SUMMARY.md` updated for release-level changes
- [ ] Links to upstream recipe, perf JSON, and relevant PRs
- [ ] Non-comparable benchmark rows footnoted
- [ ] PR template evidence table filled in

## Questions?

Open an issue before large structural changes. For benchmark methodology, see [docs/plans/2026-05-09-performance-cookbook-design.md](docs/plans/2026-05-09-performance-cookbook-design.md).
