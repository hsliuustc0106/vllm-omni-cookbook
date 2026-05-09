# vLLM-Omni Performance Cookbook

Performance evolution tracking for omni-modal, TTS, and diffusion models running on [vLLM-Omni](https://github.com/vllm-project/vllm-omni). Each stable release records measured performance and optimization deltas.

Versioning mirrors vLLM-Omni stable releases.

## Models Tracked

| Model       | Category                          | Type                       |
|-------------|-----------------------------------|----------------------------|
| Qwen3-Omni  | [omni](omni/qwen3-omni/)          | Omni-modal / any-to-any    |
| Qwen3-TTS   | [tts](tts/qwen3-tts/)             | Text-to-speech             |
| WAN2.2      | [diffusion](diffusion/wan2.2/)    | DiT image/video generation |

## Latest Release: v0.20.0 (Baseline)

See [SUMMARY.md](SUMMARY.md) for the cross-model release overview.

## Metrics

| Model Type | Primary Metrics                    |
|------------|------------------------------------|
| Omni       | TTFT, TTFP, TPOT, RTF              |
| TTS        | TTFP, RTF                          |
| Diffusion  | E2E Latency                        |
| All        | Throughput, GPU memory, HW efficiency |

## How to Add a New Release

1. Create a new `## vX.Y.Z (YYYY-MM-DD)` section in each model's `index.md`
2. Add a performance table with measured values and delta from the previous release
3. Add optimization notes with links to relevant PRs/issues/docs
4. Add figures to the model's `assets/` directory
5. Update `SUMMARY.md` with the cross-model release table

## How to Add a New Model

```bash
mkdir -p <category>/<model-name>/assets
touch <category>/<model-name>/assets/.gitkeep
```

Then write `<category>/<model-name>/index.md` following the format of existing models.

## Resources

- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) — source repository
- [vLLM-Omni Docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/)
