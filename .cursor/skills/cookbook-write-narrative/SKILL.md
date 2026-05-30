---
name: cookbook-write-narrative
description: >-
  Write optional Zhihu draft for vllm-omni-cookbook model folders. Use after retro
  tables exist in index.md, for Chinese republishing of performance ledgers.
---

# Write Zhihu Draft (Optional)

Only after `index.md` retro tables are complete. The ledger stays authoritative; the Zhihu draft explains *how* and *why* for a Chinese audience.

## Two layers

| File | Audience | Content |
|------|----------|---------|
| `index.md` | Maintainers, quick lookup | Tracks, tables, commands, links |
| `{model}-performance-zhihu.md` | Zhihu / 中文 | Paste-friendly narrative |

Do **not** duplicate full PR lists from `index.md` — summarize and link instead.

## Zhihu draft rules

Reference: `diffusion/wan2.2/wan22-i2v-performance-zhihu.md`

```markdown
<!--
标题建议：...
话题：#vLLM #扩散模型 ...
-->

# {中文标题}

**模型：** ... **环境：** ... **指标：** ...
```

- Full Chinese prose; keep English proper nouns (vLLM-Omni, USP, DiT, TTFP)
- ASCII charts OK; no HTML/Mermaid
- Tables: author copies to 飞书/Excel → 知乎 for formatting
- 置顶评论 template with GitHub links to cookbook + vllm-omni JSON
- After publish: add live Zhihu URL to `index.md` header and zhihu md footer
- Delete the HTML comment block at the top before publishing

## Wire up `index.md` header

```markdown
**Zhihu draft (中文):** [wan22-i2v-performance-zhihu.md](wan22-i2v-performance-zhihu.md)
```

## Optimization narrative structure

1. Map releases → PRs (search vllm-omni release notes + perf PRs)
2. Classify by pipeline stage: API/IPC → preprocess → VAE → DiT → runtime
3. Tie measured deltas to stages using profiler logs when available
4. Separate **CI H100 thresholds** from **local retro numbers** — never conflate
