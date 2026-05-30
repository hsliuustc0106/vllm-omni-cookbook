# vLLM-Omni Performance Summary

Cross-model headline metrics **per even stable release** (v0.14, v0.16, v0.18, v0.20, v0.22, …). Odd minors are not cookbook releases. Each row is a snapshot at that even release; full history lives in each model's `index.md`.

---

## v0.22.0 (upcoming)

| Model      | Category   | Key Metric | Value | Delta from v0.20.0 |
|------------|------------|------------|-------|---------------------|
| Qwen3-Omni | omni       | —          | —     | —                   |
| WAN2.2     | diffusion  | I2V E2E latency | — | — (to be measured) |
| Qwen-Image | diffusion  | T2I E2E (1536² USP2, 4×H200 retro) | — | — (to be measured) |

### Highlights

- **WAN2.2:** Pipeline parallel ([#2322](https://github.com/vllm-project/vllm-omni/pull/2322)), NPU MXFP8 quantization ([#3140](https://github.com/vllm-project/vllm-omni/pull/3140)). See [diffusion/wan2.2/index.md](diffusion/wan2.2/index.md).

---

## v0.20.0 (2026-05-07)

| Model      | Category   | Key Metric | Value | Delta from v0.18.0 |
|------------|------------|------------|-------|---------------------|
| Qwen3-Omni | omni       | TTFP (c=1, 2500/900 async-chunk, 2×H200) | **1325 ms** (v0.20) | v0.18 **736 ms** on same box; main 1417 ms |
| Qwen3-Omni | omni       | RTF (c=1, 2500/900, 2×H200) | **0.175** (v0.20) | v0.18 0.157; main 0.206 |
| Qwen3-Omni | omni       | E2EL (c=1, 2500/900, 2×H200) | **20.1 s** (v0.20) | v0.18 37.7 s; main 33.2 s (output length varies) |
| WAN2.2     | diffusion  | I2V E2E (832×480, 4 steps, 2×H100 CI) | **26.0 s** | first measured |
| WAN2.2     | diffusion  | I2V E2E (USP2+HSDP+VAE-pp2, H100 CI) | **21.6 s** | first measured |
| WAN2.2     | diffusion  | I2V E2E (832×480, 4×H200 retro) | **22.17 s** | **−5.9%** vs v0.18 H200 |
| WAN2.2     | diffusion  | I2V E2E (USP2, 480p, 4×H200 retro) | **16.43 s** | **−18.9%** vs v0.18 H200 |
| WAN2.2     | diffusion  | T2V mean (480p, VAE-pp=4) | **21.68 s** | first measured |
| Qwen-Image | diffusion  | T2I E2E (512², 2×H100 CI) | **3.50 s** | first measured |
| Qwen-Image | diffusion  | T2I E2E (1536², 2×H100 CI) | **27.0 s** | first measured |
| Qwen-Image | diffusion  | T2I E2E (1536² USP2, 2×H100 CI) | **9.1 s** | first measured |
| Qwen-Image | diffusion  | T2I E2E (1536² USP2, 4×H200 retro) | **8.42 s** | **+3.2%** vs v0.18 H200 |
| Qwen-Image | diffusion  | T2I E2E (1536² single, 4×H200 retro) | **24.48 s** | **+2.2%** vs v0.18 H200 |

### Highlights

- **Qwen-Image:** First retro T2I comparison (v0.18 / v0.20) on 4× H200; roughly at parity (~2–9% delta). See [index](diffusion/qwen-image/index.md).
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
