# vLLM-Omni Blog

**Deep dives into important [vLLM-Omni](https://github.com/vllm-project/vllm-omni) PRs, features, and new model support** — what changed, how it works, and the measured impact.

Read it live at **[hsliuustc0106.github.io/vllm-omni-cookbook](https://hsliuustc0106.github.io/vllm-omni-cookbook/)**.

Whenever an important PR, feature, or model is finalized, it gets a **PR Analysis** post explaining it for users: the problem it solves, the design, the key code changes, and the benchmark evidence. Posts have no fixed cadence — they publish when something important lands. The blog is a self-contained Jekyll site in [`blog/`](blog/) (same stack as [vllm.ai/blog](https://vllm.ai/blog)), redeployed to GitHub Pages automatically on every push that touches `blog/`. New posts start from [`blog/TEMPLATE.md`](blog/TEMPLATE.md).

Local preview (needs Ruby ≥ 3, e.g. `brew install ruby`):
`cd blog && bundle install && bundle exec jekyll serve`

Behind the posts, this repo maintains the **vLLM-Omni performance cookbook** — the authoritative numbers the blog cites. For each tracked model it records **what improved, by how much, and why** at every **even** stable release, measured and linked to the PRs behind each change.

**Cookbook cadence:** updated only on **even** vLLM-Omni minor releases — v0.14.0, v0.16.0, v0.18.0, v0.20.0, v0.22.0, … Odd minors (v0.19, v0.21, …) are skipped. Deltas always compare to the **previous even** release (e.g. v0.22 vs v0.20).

## How it is organized

| Layer | File | Role |
|-------|------|------|
| **Blog posts** | `blog/_posts/` | One deep-dive "PR Analysis" post per important PR/feature — the user-facing narrative |
| **Blog site** | [`blog/`](blog/) | Self-contained Jekyll site (config, template, figures) deployed to GitHub Pages |
| **Per model** | `{category}/{model}/index.md` | Cookbook ledger — one `## vX.Y.Z` section per **even** release (metrics, delta vs prior even release, optimization notes) |
| **Per release** | `SUMMARY.md` | Cross-model cookbook snapshot for each **even** release only |
| **Update notes** | [`notes/`](notes/) | Chronological feature, RFC, and PR updates with no fixed cadence |

vLLM-Omni owns deployment recipes and benchmark harnesses; this repo publishes the
blog, the performance history, and the related update notes.

## Models Tracked

| Model       | Category                          | Type                       |
|-------------|-----------------------------------|----------------------------|
| Qwen3-Omni  | [omni](omni/qwen3-omni/)          | Omni-modal / any-to-any    |
| VoxCPM2     | [omni](omni/voxcpm2/)             | Text-to-speech / voice clone |
| WAN2.2      | [diffusion](diffusion/wan2.2/)    | DiT image/video generation |
| Qwen-Image  | [diffusion](diffusion/qwen-image/) | Text-to-image (DiT)       |
| Qwen-Image-Edit | [diffusion](diffusion/qwen-image-edit/) | Image-to-image editing (DiT) |

## Latest Cookbook Release: v0.22.0

See [SUMMARY.md](SUMMARY.md) for the cross-model overview (even releases: v0.14.0 → v0.22.0).

**WAN2.2** — [index](diffusion/wan2.2/index.md) · [Zhihu draft](diffusion/wan2.2/wan22-i2v-performance-zhihu.md).

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

## How to Add a New Blog Post

1. Copy [`blog/TEMPLATE.md`](blog/TEMPLATE.md) → `blog/_posts/YYYY-MM-DD-<slug>.md`
2. Write for users: TL;DR → Background → What the PR does → Key changes → Measured impact → How to use it → Limitations → References
3. Cite numbers only from the cookbook ledgers or upstream perf JSON, with absolute GitHub URLs
4. Figures go in `blog/assets/figures/<slug>/`; preview locally with `bundle exec jekyll serve`
5. Merging to `main` auto-deploys the blog

## How to Add a New Cookbook Release

Update the cookbook only when vLLM-Omni ships an **even** minor release (`v0.24.0`, …). Skip odd minors.

For that release, append the improvement summary for **each tracked model**:

1. Add `## vX.Y.Z (YYYY-MM-DD)` to the model's `index.md` — metrics, **delta from the previous even release**, optimization notes (PR links)
2. Update retro comparison tables if new tag columns were measured
3. Add figures under `assets/` when helpful
4. Update `SUMMARY.md` with the cross-model headline row for this release

## How to Add a New Model

```bash
mkdir -p <category>/<model-name>/assets
touch <category>/<model-name>/assets/.gitkeep
```

Then write `<category>/<model-name>/index.md` following the format of existing models.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and agent skills in [.cursor/skills/](.cursor/skills/) for step-by-step workflows (write a blog post, add model, add release, write narrative).

## Resources

- [Blog](https://hsliuustc0106.github.io/vllm-omni-cookbook/) — PR Analysis deep dives
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) — source repository
- [vLLM-Omni Docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/)
