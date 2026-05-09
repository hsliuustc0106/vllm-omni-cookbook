# vLLM-Omni Performance Cookbook — Guidance for Claude Code

This is a performance tracking repository — not a how-to cookbook. Each model folder contains a performance ledger (`index.md`) that records measured performance and optimization deltas across vLLM-Omni stable releases.

## Repository Structure

```
├── README.md              # Overview and model listing
├── SUMMARY.md             # Cross-model release summary
├── omni/                  # Omni-modal models
│   └── qwen3-omni/
│       ├── index.md        # Perf ledger across releases
│       └── assets/         # Charts and figures
├── tts/                   # TTS models
│   └── qwen3-tts/
│       ├── index.md
│       └── assets/
└── diffusion/             # Diffusion models
    └── wan2.2/
        ├── index.md
        └── assets/
```

## Adding Performance Data for a New Release

1. In each model's `index.md`, add a new `## vX.Y.Z (YYYY-MM-DD)` section
2. Include: performance table with deltas, optimization notes (PRs/issues/docs links), figures
3. Update `SUMMARY.md` with the cross-model release table

## Adding a New Model

```bash
mkdir -p <category>/<model-name>/assets
touch <category>/<model-name>/assets/.gitkeep
```

Write `<category>/<model-name>/index.md` following the same format as existing models.

## Resources

- vLLM-Omni: https://github.com/vllm-project/vllm-omni
- vLLM-Omni Docs: https://docs.vllm.ai/projects/vllm-omni/en/latest/
