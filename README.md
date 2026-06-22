# vLLM-Omni Performance Cookbook

**Release-by-release performance summary** for models running on [vLLM-Omni](https://github.com/vllm-project/vllm-omni). For each tracked model, the cookbook records **what improved, by how much, and why** at every **even** stable release — measured numbers plus links to the PRs and optimizations behind them.

**Update cadence:** the cookbook is updated only on **even** vLLM-Omni minor releases — v0.14.0, v0.16.0, v0.18.0, v0.20.0, v0.22.0, … Odd minors (v0.19, v0.21, …) are skipped. Deltas always compare to the **previous even** release (e.g. v0.22 vs v0.20).

## How it is organized

| Layer | File | Role |
|-------|------|------|
| **Per model** | `{category}/{model}/index.md` | Full improvement history — one `## vX.Y.Z` section per **even** release (metrics, delta vs prior even release, optimization notes) |
| **Per release** | `SUMMARY.md` | Cross-model snapshot for each **even** release only |

vLLM-Omni owns deployment recipes and benchmark harnesses; this repo is the **published summary** of performance evolution only.

## Models Tracked

| Model       | Category                          | Type                       |
|-------------|-----------------------------------|----------------------------|
| Qwen3-Omni  | [omni](omni/qwen3-omni/)          | Omni-modal / any-to-any    |
| VoxCPM2     | [omni](omni/voxcpm2/)             | Text-to-speech / voice clone |
| WAN2.2      | [diffusion](diffusion/wan2.2/)    | DiT image/video generation |
| Qwen-Image  | [diffusion](diffusion/qwen-image/) | Text-to-image (DiT)       |
| Qwen-Image-Edit | [diffusion](diffusion/qwen-image-edit/) | Image-to-image editing (DiT) |

## Latest Cookbook Release: v0.20.0

See [SUMMARY.md](SUMMARY.md) for the cross-model overview (even releases: v0.14.0 → v0.22.0). Next cookbook update: **v0.22.0**.

**WAN2.2** — [index](diffusion/wan2.2/index.md) · [Zhihu draft](diffusion/wan2.2/wan22-i2v-performance-zhihu.md).

## Metrics

| Model Type | Primary Metrics                    |
|------------|------------------------------------|
| Omni       | TTFT, TTFP, TPOT, RTF, E2EL        |
| Diffusion  | E2E latency, throughput            |
| All        | GPU memory, hardware efficiency    |

## How to Add a New Release

Update the cookbook only when vLLM-Omni ships an **even** minor release (`v0.22.0`, `v0.24.0`, …). Skip odd minors.

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

See [CONTRIBUTING.md](CONTRIBUTING.md) and agent skills in [.cursor/skills/](.cursor/skills/) for step-by-step workflows (add model, add release, write narrative).

## Resources

- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) — source repository
- [vLLM-Omni Docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/)
