# vLLM-Omni Performance Summary

Cross-model performance deltas across vLLM-Omni stable releases. For per-model details, see the model index files.

## v0.20.0 (2026-05-01) — Baseline

| Model      | Category   | Key Metric              | Value   | Delta | Notes |
|------------|------------|-------------------------|---------|-------|-------|
| Qwen3-Omni | omni       | —                       | —       | —     | To be measured |
| Qwen3-TTS  | tts        | —                       | —       | —     | To be measured |
| WAN2.2     | diffusion  | I2V E2E latency (832×480, 4 steps, 1×H100) | **26.0 s** | — | Single-GPU baseline |
| WAN2.2     | diffusion  | I2V E2E latency (same, USP2+HSDP+VAE-pp2) | **21.6 s** | **−17%** | vs single-GPU |
| WAN2.2     | diffusion  | T2V mean latency (480p, CFG2 USP2 HSDP) | **21.68 s** | — | A100, VAE-pp=4 |

### Highlights

- **WAN2.2** is the first model with populated cookbook metrics. See [diffusion/wan2.2/index.md](diffusion/wan2.2/index.md) for full PR history, GPU vs NPU guidance, and T2V dashboard numbers.
- Major themes through v0.20.0: parallelism (TP/SP/CFG/HSDP/VAE-pp/PP), fused kernels (GPU + Ascend mindie-sd), serving IPC reduction ([#1715](https://github.com/vllm-project/vllm-omni/pull/1715)), and NPU MXFP8 quantization ([#3140](https://github.com/vllm-project/vllm-omni/pull/3140)).
- Qwen3-Omni and Qwen3-TTS baselines remain to be measured.
