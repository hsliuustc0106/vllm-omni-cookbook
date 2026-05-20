# vLLM-Omni Performance Summary

Cross-model performance deltas across vLLM-Omni stable releases. For per-model details, see the model index files.

---

## v0.21.0 (upcoming)

| Model      | Category   | Key Metric | Value | Delta from v0.20.0 |
|------------|------------|------------|-------|---------------------|
| Qwen3-Omni | omni       | —          | —     | —                   |
| Qwen3-TTS  | tts        | —          | —     | —                   |
| WAN2.2     | diffusion  | I2V E2E latency | — | — (to be measured) |

### Highlights

- **WAN2.2:** Pipeline parallel ([#2322](https://github.com/vllm-project/vllm-omni/pull/2322)), NPU MXFP8 quantization ([#3140](https://github.com/vllm-project/vllm-omni/pull/3140)). See [diffusion/wan2.2/index.md](diffusion/wan2.2/index.md).

---

## v0.20.0 (2026-05-07)

| Model      | Category   | Key Metric | Value | Delta from v0.18.0 |
|------------|------------|------------|-------|---------------------|
| Qwen3-Omni | omni       | —          | —     | —                   |
| Qwen3-TTS  | tts        | —          | —     | —                   |
| WAN2.2     | diffusion  | I2V E2E (832×480, 4 steps, 1×H100) | **26.0 s** | first measured |
| WAN2.2     | diffusion  | I2V E2E (USP2+HSDP+VAE-pp2) | **21.6 s** | first measured |
| WAN2.2     | diffusion  | T2V mean (480p, VAE-pp=4) | **21.68 s** | first measured |

### Highlights

- **WAN2.2:** First cookbook release with GPU perf baselines; fused DiT kernels (GPU + NPU), pipeline refactor, nightly I2V CI ([#3063](https://github.com/vllm-project/vllm-omni/pull/3063)).
- **v0.18.0 → v0.20.0:** Kernel fusion, NPU operator path, preprocess/VAE optimizations, S2V ([#2751](https://github.com/vllm-project/vllm-omni/pull/2751)).

---

## v0.18.0 (2026-03-28)

| Model      | Category   | Key Metric | Value | Delta from v0.16.0 |
|------------|------------|------------|-------|---------------------|
| WAN2.2     | diffusion  | Standardized I2V JSON (`832×480`, 4 steps) | — | JSON not shipped until [#3063](https://github.com/vllm-project/vllm-omni/pull/3063) (after v0.18.0) |
| WAN2.2     | diffusion  | Online I2V e2e ([#1715](https://github.com/vllm-project/vllm-omni/pull/1715), other workload) | 31.0 s | **−17.5%** vs 37.5 s (IPC fix) |
| WAN2.2     | diffusion  | 14B weight load | faster | ([#1504](https://github.com/vllm-project/vllm-omni/pull/1504)) |

### Highlights

- **WAN2.2:** Serving path optimizations (IPC, startup), cache-dit TI2V fix, L4 test coverage.

---

## v0.16.0 (2026-02-28)

| Model      | Category   | Key Metric | Value | Delta from v0.14.0 |
|------------|------------|------------|-------|---------------------|
| WAN2.2     | diffusion  | —          | —     | Online API + TP shipped |

### Highlights

- **WAN2.2:** `/v1/videos` online serving ([#1073](https://github.com/vllm-project/vllm-omni/pull/1073)), tensor parallelism ([#964](https://github.com/vllm-project/vllm-omni/pull/964)).

---

## v0.14.0 (2026-01-31) — WAN2.2 initial stable

| Model  | Category  | Key Metric | Value | Delta |
|--------|-----------|------------|-------|-------|
| WAN2.2 | diffusion | —          | —     | Initial T2V / I2V / TI2V in stable |

### Highlights

- **WAN2.2:** First stable pipelines ([#202](https://github.com/vllm-project/vllm-omni/pull/202), [#329](https://github.com/vllm-project/vllm-omni/pull/329)); SP, cache-dit, CFG parallel, conditional MoE loading.
