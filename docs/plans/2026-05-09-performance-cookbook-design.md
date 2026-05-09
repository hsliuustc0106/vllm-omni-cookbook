# Performance Cookbook Refactor — Design

**Date:** 2026-05-09
**Context:** Refactor vLLM-Omni cookbook from topic-based how-to recipes into a per-model performance ledger that tracks optimization deltas across stable releases.

## Motivation

The original cookbook was organized by topic (inference, deployment, performance, etc.) with most sections empty. vLLM-Omni already owns per-model recipes in its own repo. The cookbook's unique value is tracking performance evolution: as vLLM-Omni releases stable versions, the cookbook records what improved, by how much, and why — for each model.

The files also serve as raw material for reposting to Xiaohongshu, Zhihu, and other platforms.

## Structure

```
vllm-omni-cookbook/
├── README.md              # what this is, versioning convention
├── SUMMARY.md             # cross-model delta summary per release
├── omni/
│   └── qwen3-omni/
│       ├── index.md        # perf ledger across releases
│       └── assets/         # figures, charts
├── tts/
│   └── qwen3-tts/
│       ├── index.md
│       └── assets/
└── diffusion/
    └── wan2.2/
        ├── index.md
        └── assets/
```

- **Category folders** (`omni`, `tts`, `diffusion`) group model types.
- **Model folders** contain a performance ledger (`index.md`) and assets for figures.
- **Flat within models** — releases are inline sections, not separate files.
- **Cookbook versions mirror vLLM-Omni stable releases** (v0.20.0, v0.22.0, etc.).
- **No how-to recipes** — vLLM-Omni repo owns those; the cookbook links to them.

## Metrics by Model Type

| Model Type   | Metrics                                                |
|-------------|--------------------------------------------------------|
| Omni         | TTFT, TTFP, TPOT, RTF                                  |
| TTS          | TTFP, RTF                                              |
| Diffusion    | E2E Latency                                            |
| All          | Throughput, memory usage, hardware efficiency          |

## Model File Format (index.md)

Each model's `index.md` contains:

1. **Model header** — name, category, link to vLLM-Omni recipe
2. **Hardware baseline** — fixed reference hardware for comparable deltas
3. **Per-release section** (newest first):

```
## v0.22.0 (2026-06-15)

### Performance

| Metric   | Value | Delta from v0.20.0 |
|----------|-------|---------------------|
| TTFT     | 85ms  | -12%                |
| ...      |       |                     |

### Optimization Notes
- Links to relevant PRs, issues, and docs that explain the improvement
- Enabled cuda graph capture for decoder (PR #1234)
- Fixed KV cache fragmentation (Issue #1200)

### Figures
![latency chart](assets/v0.22.0-latency.png)
```

## SUMMARY.md Format

Cross-model release overview — one table per release:

```
## v0.22.0 (2026-06-15)

| Model      | Category   | Key Metric     | Value   | Delta     |
|------------|------------|----------------|---------|-----------|
| Qwen3-Omni | omni       | TTFT           | 85ms    | -12%      |
| Qwen3-Omni | omni       | Throughput     | 24 t/s  | +8%       |
| Qwen3-TTS  | tts        | RTF            | 0.12    | -15%      |
| WAN2.2     | diffusion  | E2E Latency    | 2.1s    | -20%      |

### Highlights
- WAN2.2: 20% faster diffusion via batched DiT inference (PR #1300)
```

## Existing Content

The current `00-quickstart/` recipes and empty category folders (`01-inference/` through `07-troubleshooting/`) are removed. The `templates/` directory is either removed or repurposed. The `topics/index.md` is replaced by `SUMMARY.md`.
