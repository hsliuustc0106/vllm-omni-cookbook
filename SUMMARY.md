# vLLM-Omni Performance Summary

Cross-model performance deltas across vLLM-Omni stable releases. For per-model details, see the model index files.

---

## v0.21.0 (upcoming)

| Model      | Category   | Key Metric | Value | Delta from v0.20.0 |
|------------|------------|------------|-------|---------------------|
| Qwen3-Omni | omni       | —          | —     | —                   |
| Qwen3-TTS  | tts        | TTFP default_voice (c=1, 2×L20X) | **47 ms** | **−21%** vs v0.20 L20X |
| Qwen3-TTS  | tts        | RTF default_voice (c=1, 2×L20X) | **0.145** | ~flat vs v0.20 L20X |
| Qwen3-TTS  | tts        | TTFP default_voice (c=8, std deploy, 2×L20X) | **81 ms** | **−62%** vs v0.20 L20X |
| Qwen3-TTS  | tts        | Throughput default_voice (c=8, std, 2×L20X) | **31.4 audio-s/s** | **+59%** vs v0.20 L20X |
| Qwen3-TTS  | tts        | TTFP default_voice (c=64, hiconc deploy, 2×L20X) | **351 ms** | **−96%** vs main std L20X |
| Qwen3-TTS  | tts        | RTF default_voice (c=64, hiconc deploy, 2×L20X) | **0.996** | **−35%** vs main std L20X |
| Qwen3-TTS  | tts        | Throughput default_voice (c=64, hiconc, 2×L20X) | **60.8 audio-s/s** | **+66%** vs main std L20X |
| WAN2.2     | diffusion  | I2V E2E latency | — | — (to be measured) |
| Qwen-Image | diffusion  | T2I E2E (1536² USP2, 4×L20X retro) | — | — (to be measured) |

### Highlights

- **Qwen3-TTS:** Track **TTFP**, **RTF**, **throughput**. [#3732](https://github.com/vllm-project/vllm-omni/pull/3732) helps at c=1/c=8/c=64 (isolated A/B: TTFP −8–14% @ c=1, −12% @ c=8, RTF −7% @ c=64); [#3662](https://github.com/vllm-project/vllm-omni/pull/3662) hiconc still required for c=16/64 TTFP cliff. See [tts/qwen3-tts/index.md](tts/qwen3-tts/index.md).
- **WAN2.2:** Pipeline parallel ([#2322](https://github.com/vllm-project/vllm-omni/pull/2322)), NPU MXFP8 quantization ([#3140](https://github.com/vllm-project/vllm-omni/pull/3140)). See [diffusion/wan2.2/index.md](diffusion/wan2.2/index.md).

---

## v0.20.0 (2026-05-07)

| Model      | Category   | Key Metric | Value | Delta from v0.18.0 |
|------------|------------|------------|-------|---------------------|
| Qwen3-Omni | omni       | TTFP (c=1, 2500/900, 2×L20X retro) | **1325 ms** | v0.18 blocked on L20X |
| Qwen3-Omni | omni       | RTF (c=1, 2500/900, 2×L20X retro) | **0.175** | v0.18 blocked on L20X |
| Qwen3-Omni | omni       | E2EL (c=1, 2500/900, 2×L20X retro) | **20.1 s** | v0.18 blocked on L20X |
| Qwen3-TTS  | tts        | TTFP default_voice (c=1, 2×L20X retro) | **59 ms** | first L20X retro |
| Qwen3-TTS  | tts        | RTF default_voice (c=1, 2×L20X retro) | **0.145** | first L20X retro |
| Qwen3-TTS  | tts        | TTFP default_voice (c=8, 2×L20X retro) | **214 ms** | first L20X retro |
| Qwen3-TTS  | tts        | Throughput default_voice (c=8, 2×L20X retro) | **19.8 audio-s/s** | first L20X retro |
| WAN2.2     | diffusion  | I2V E2E (832×480, 4 steps, 2×H100 CI) | **26.0 s** | first measured |
| WAN2.2     | diffusion  | I2V E2E (USP2+HSDP+VAE-pp2, H100 CI) | **21.6 s** | first measured |
| WAN2.2     | diffusion  | I2V E2E (832×480, 4×H200 retro) | **22.17 s** | **−5.9%** vs v0.18 H200 |
| WAN2.2     | diffusion  | I2V E2E (USP2, 480p, 4×H200 retro) | **16.43 s** | **−18.9%** vs v0.18 H200 |
| WAN2.2     | diffusion  | T2V mean (480p, VAE-pp=4) | **21.68 s** | first measured |
| Qwen-Image | diffusion  | T2I E2E (512², 2×H100 CI) | **3.50 s** | first measured |
| Qwen-Image | diffusion  | T2I E2E (1536², 2×H100 CI) | **27.0 s** | first measured |
| Qwen-Image | diffusion  | T2I E2E (1536² USP2, 2×H100 CI) | **9.1 s** | first measured |
| Qwen-Image | diffusion  | T2I E2E (1536² USP2, 4×L20X retro) | **8.42 s** | **+3.2%** vs v0.18 L20X |
| Qwen-Image | diffusion  | T2I E2E (1536² single, 4×L20X retro) | **24.48 s** | **+2.2%** vs v0.18 L20X |

### Highlights

- **Qwen3-TTS:** First L20X retro (v0.20 / main) on CustomVoice; universal TTS benchmark ([#2835](https://github.com/vllm-project/vllm-omni/pull/2835)). See [index](tts/qwen3-tts/index.md).
- **Qwen-Image:** First retro T2I comparison (v0.18 / v0.20) on 4× L20X; roughly at parity (~2–9% delta). See [index](diffusion/qwen-image/index.md).
- **WAN2.2:** First cookbook release with GPU perf baselines; fused DiT kernels (GPU + NPU), pipeline refactor, nightly I2V CI ([#3063](https://github.com/vllm-project/vllm-omni/pull/3063)).
- **v0.18.0 → v0.20.0 (4× H200):** [−5.9% / −18.9% / −15.5%](diffusion/wan2.2/index.md#h200-retro-comparison) on standardized I2V workloads. [Blog post](diffusion/wan2.2/wan22-i2v-performance.html).

---

## v0.18.0 (2026-03-28)

| Model      | Category   | Key Metric | Value | Delta from v0.16.0 |
|------------|------------|------------|-------|---------------------|
| WAN2.2     | diffusion  | Standardized I2V JSON (`832×480`, 4 steps) | — | JSON not shipped until [#3063](https://github.com/vllm-project/vllm-omni/pull/3063) (after v0.18.0) |
| WAN2.2     | diffusion  | I2V E2E (832×480, 4×H200 retro) | **23.56 s** | **−24.8%** vs v0.16 H200 single |
| WAN2.2     | diffusion  | I2V E2E (USP2, 480p, 4×H200 retro) | **20.26 s** | **−8.7%** vs v0.16 H200 USP2 † |
| WAN2.2     | diffusion  | Online I2V e2e ([#1715](https://github.com/vllm-project/vllm-omni/pull/1715), other workload) | 31.0 s | **−17.5%** vs 37.5 s (IPC fix) |
| WAN2.2     | diffusion  | 14B weight load | faster | ([#1504](https://github.com/vllm-project/vllm-omni/pull/1504)) |

† v0.16 retro USP2 lacked VAE patch-parallel CLI; see [WAN2.2 index](diffusion/wan2.2/index.md#h200-retro-comparison).

### Highlights

- **WAN2.2:** Serving path optimizations (IPC, startup), cache-dit TI2V fix, L4 test coverage.

---

## v0.16.0 (2026-02-28)

| Model      | Category   | Key Metric | Value | Delta from v0.14.0 |
|------------|------------|------------|-------|---------------------|
| WAN2.2     | diffusion  | I2V E2E (832×480, 4×H200 retro) | **31.33 s** | first retro baseline |
| WAN2.2     | diffusion  | I2V E2E (USP2, 480p, 4×H200 retro) | **22.20 s** | — |
| WAN2.2     | diffusion  | I2V E2E (USP2, 720p, 4×H200 retro) | **133.94 s** | — |

### Highlights

- **WAN2.2:** `/v1/videos` online serving ([#1073](https://github.com/vllm-project/vllm-omni/pull/1073)), tensor parallelism ([#964](https://github.com/vllm-project/vllm-omni/pull/964)). H200 retro: [index](diffusion/wan2.2/index.md#h200-retro-comparison) · [blog](diffusion/wan2.2/wan22-i2v-performance.html).

---

## v0.14.0 (2026-01-31) — WAN2.2 initial stable

| Model  | Category  | Key Metric | Value | Delta |
|--------|-----------|------------|-------|-------|
| WAN2.2 | diffusion | —          | —     | Initial T2V / I2V / TI2V in stable |

### Highlights

- **WAN2.2:** First stable pipelines ([#202](https://github.com/vllm-project/vllm-omni/pull/202), [#329](https://github.com/vllm-project/vllm-omni/pull/329)); SP, cache-dit, CFG parallel, conditional MoE loading.
