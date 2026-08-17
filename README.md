# vLLM-Omni Blog

**Feature blog and performance tracing for [vLLM-Omni](https://github.com/vllm-project/vllm-omni)** — deep dives into important PRs, features, and new model support, plus tracing-driven analysis of where inference time goes and what each change costs or saves.

Read it live at **[hsliuustc0106.github.io/vllm-omni-cookbook](https://hsliuustc0106.github.io/vllm-omni-cookbook/)**.

Whenever an important PR, feature, or model is finalized, it gets a **PR Analysis** post explaining it for users: the problem it solves, the design, the key code changes, and the benchmark evidence. Posts have no fixed cadence — they publish when something important lands. The blog is a self-contained Jekyll site in [`blog/`](blog/) (same stack as [vllm.ai/blog](https://vllm.ai/blog)), redeployed to GitHub Pages automatically on every push that touches `blog/`. New posts start from [`blog/TEMPLATE.md`](blog/TEMPLATE.md).

Local preview (needs Ruby ≥ 3, e.g. `brew install ruby`):
`cd blog && bundle install && bundle exec jekyll serve`

Behind the posts, this repo maintains the **vLLM-Omni performance cookbook** — the two things every post is built on: **feature analysis** (what a PR does and how it works) and **performance tracing** (the measured numbers and where they come from). For each tracked model it records **what improved, by how much, and why** at every **even** stable release, measured and linked to the PRs behind each change. The cadence rules and cross-model numbers live in [SUMMARY.md](SUMMARY.md).

## How it is organized

| Layer | File | Role |
|-------|------|------|
| **Blog posts** | `blog/_posts/` | One deep-dive "PR Analysis" post per important PR/feature — the user-facing narrative |
| **Blog site** | [`blog/`](blog/) | Self-contained Jekyll site (config, template, figures) deployed to GitHub Pages |
| **Per model** | `{category}/{model}/index.md` | Cookbook ledger — one `## vX.Y.Z` section per **even** release (metrics, delta vs prior even release, optimization notes) |
| **Per release** | `SUMMARY.md` | Cross-model cookbook snapshot for each **even** release only |
| **Update notes** | [`notes/`](notes/) | Chronological feature, RFC, and PR updates with no fixed cadence |

vLLM-Omni owns deployment recipes and benchmark harnesses; this repo publishes the
feature blog, the performance-tracing history, and the related update notes.

## Models Tracked

| Model       | Category                          | Type                       |
|-------------|-----------------------------------|----------------------------|
| Qwen3-Omni  | [omni](omni/qwen3-omni/)          | Omni-modal / any-to-any    |
| VoxCPM2     | [omni](omni/voxcpm2/)             | Text-to-speech / voice clone |
| WAN2.2      | [diffusion](diffusion/wan2.2/)    | DiT image/video generation |
| Qwen-Image  | [diffusion](diffusion/qwen-image/) | Text-to-image (DiT)       |
| Qwen-Image-Edit | [diffusion](diffusion/qwen-image-edit/) | Image-to-image editing (DiT) |

## Update Notes

[Update notes](notes/) cover recent features, important RFCs and design discussions,
notable PRs, performance implications, and follow-up items. They are published when
useful rather than on a daily or weekly schedule; each note records its own coverage
period. Use [the template](notes/TEMPLATE.md) for new entries.

## Metrics

| Model Type | Primary Metrics                    |
|------------|------------------------------------|
| Omni       | TTFT, TTFP, TPOT, RTF, E2EL        |
| Diffusion  | E2E latency, throughput            |
| All        | GPU memory, hardware efficiency    |

## Contributing

Step-by-step how-tos live in [CONTRIBUTING.md](CONTRIBUTING.md):
[add a new blog post](CONTRIBUTING.md#how-to-add-a-new-blog-post),
[add a new cookbook release](CONTRIBUTING.md#how-to-add-a-new-cookbook-release),
[add a new model](CONTRIBUTING.md#how-to-add-a-new-model).
Agent skills in [.cursor/skills/](.cursor/skills/) cover the same workflows.

## Resources

- [Blog](https://hsliuustc0106.github.io/vllm-omni-cookbook/) — PR Analysis deep dives
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) — source repository
- [vLLM-Omni Docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/)
