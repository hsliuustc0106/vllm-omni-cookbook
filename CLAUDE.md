# vLLM-Omni Performance Cookbook — Guidance for Claude Code

Release-by-release **performance improvement summary** for vLLM-Omni models — not a how-to cookbook.

**Even releases only:** update on v0.18.0, v0.20.0, v0.22.0, … Skip odd minors. Deltas vs the previous even release.

Each `{category}/{model}/index.md` is one model's timeline: a `## vX.Y.Z` section per **even** stable release with measured metrics, delta vs the prior even release, and optimization notes (PR links). `SUMMARY.md` aggregates headline numbers across models for each even release.

## Repository Structure

```
├── README.md              # Overview and model listing
├── SUMMARY.md             # Cross-model release summary
├── omni/                  # Omni-modal models
│   └── qwen3-omni/
│       ├── index.md
│       └── assets/
└── diffusion/             # Diffusion models
    └── wan2.2/
        ├── index.md
        └── assets/
```

## Adding Performance Data for a New Release

Only when vLLM-Omni ships an **even** minor release (v0.22.0, …):

1. In each model's `index.md`, add `## vX.Y.Z (YYYY-MM-DD)` — delta vs the **previous even** release
2. Include: performance table, optimization notes (PR links), figures
3. Update `SUMMARY.md` with the cross-model release table

## Adding a New Model

```bash
mkdir -p <category>/<model-name>/assets
touch <category>/<model-name>/assets/.gitkeep
```

Write `<category>/<model-name>/index.md` following the same format as existing models.

## Contributor skills

Load `.cursor/skills/vllm-omni-cookbook/SKILL.md` first, then route to:

- `cookbook-add-model` — new model ledger
- `cookbook-add-release` — vX.Y.Z + SUMMARY.md
- `cookbook-write-narrative` — optional Zhihu draft

Claude Code stubs: `.claude/skills/*.md`

## Resources

- vLLM-Omni: https://github.com/vllm-project/vllm-omni
- vLLM-Omni Docs: https://docs.vllm.ai/projects/vllm-omni/en/latest/
