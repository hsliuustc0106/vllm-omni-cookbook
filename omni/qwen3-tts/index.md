# Qwen3-TTS

**Models:** `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
**Recipes:** [Qwen3-TTS](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen3-TTS.md) · [Deploy YAML](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/deploy)
**Spec:** [`test_tts.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_tts.json) · **Runner:** [`run_benchmark.py`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/scripts/run_benchmark.py)
**Raw JSONs:** [`data/`](data/)

vllm-omni `main` (`40b29591`, vllm 0.22.0) vs `v0.21.0rc2` (`5f9aee19`, vllm 0.21.0) — both with `transformers==5.8.1` — measured on two boxes 2026-06-03.

Cells: `median_audio_rtf` (audio s / wall s, **<1 = realtime**) / `median_audio_ttfp_ms` / `audio_throughput`. Δ tput = main vs v0.21.0rc2.

---

## Hardware

| Box | GPU | Mem-BW | Used for | Date |
|-----|-----|-------:|---------|------|
| **H20-3e** (h20-server-1) | NVIDIA H20 96GB HBM3 | ~3.35 TB/s | high-c throughput target | 2026-06-03 |
| **L20X** (h200-hsliu) | NVIDIA L20X 144GB GDDR | ~1.1 TB/s | cheaper streaming target | 2026-06-03/04 |

Both runs: 1 GPU for Stage 0 (Talker), 1 GPU for Stage 1 (Code2Wav), single replica.

---

## Base · `seed-tts`

### H20-3e

| c | n | v0.21.0rc2 | main (v0.22) | Δ tput |
|---:|---:|---|---|---:|
| 1 | 20 | 0.153 / 131 / 6.54 | 0.165 / 132 / 6.14 | −6.1% |
| 4 | 200 | 0.182 / 114 / 21.76 | 0.172 / 95 / 23.02 | **+5.8%** |
| 8 | 80 | 0.304 / 266 / 24.15 | 0.317 / 280 / 23.46 | −2.9% |
| 16 | 128 | 0.443 / 784 / 35.38 | 0.387 / 679 / 39.13 | **+10.6%** |
| 64 | 128 | 1.494 / 5832 / 35.99 | 1.325 / 4887 / 41.35 | **+14.9%** |

### L20X

| c | n | v0.21.0rc2 | main (v0.22) | Δ tput |
|---:|---:|---|---|---:|
| 1 | 20 | 1.582 / 1597 / 0.62 | 1.996 / 1847 / 0.51 | −19.0% |
| 8 | 80 | 4.067 / 3192 / 1.83 | 4.392 / 3494 / 1.72 | −5.6% |
| 16 | 128 | 6.105 / 11394 / 2.52 | 5.089 / 9044 / 2.96 | **+17.4%** |
| 64 | 128 | 20.893 / 84451 / 2.57 | 16.813 / 66002 / 3.23 | **+26.0%** |

Base c=4 quality cell skipped by the pytest marker on L20X main.

---

## CustomVoice · `seed-tts-text`

### H20-3e

| c | n | v0.21.0rc2 | main (v0.22) | Δ tput |
|---:|---:|---|---|---:|
| 1 | 20 | 0.134 / 44 / 7.48 | 0.141 / 47 / 7.08 | −5.3% |
| 4 | 200 | 0.149 / 56 / 26.54 | 0.155 / 59 / 25.61 | −3.5% |
| 8 | 80 | 0.189 / 71 / 36.31 | 0.198 / 81 / 34.49 | −5.0% |
| 16 | 128 | 0.326 / 710 / 48.06 | 0.332 / 735 / 47.21 | −1.8% |
| 64 | 128 | 1.209 / 5776 / 48.58 | 1.178 / 6196 / 47.47 | −2.3% |

### L20X

| c | n | v0.21.0rc2 | main (v0.22) | Δ tput |
|---:|---:|---|---|---:|
| 1 | 20 | 1.314 / 595 / 0.77 | 1.622 / 686 / 0.61 | −20.3% |
| 4 / 2.0 † | 100/200 | _2.0/100_ † | 1.920 / 798 / 2.07 | — |
| 8 | 80 | 2.261 / 1107 / 3.09 | 2.352 / 1005 / 2.88 | −7.0% |
| 16 | 128 | 4.139 / 9400 / 3.83 | 4.226 / 9649 / 3.66 | −4.5% |
| 64 | 128 | 14.828 / 75758 / 3.89 | 15.728 / 80110 / 3.56 | −8.4% |

---

## CustomVoice · `seed-tts-design`

### H20-3e

| c | n | v0.21.0rc2 | main (v0.22) | Δ tput |
|---:|---:|---|---|---:|
| 1 | 20 | 0.126 / 43 / 7.88 | 0.136 / 47 / 7.34 | −6.9% |
| 8 | 80 | 0.185 / 72 / 41.76 | 0.194 / 74 / 38.20 | −8.5% |
| 16 | 128 | 0.314 / 581 / 48.32 | 0.333 / 599 / 46.58 | −3.6% |
| 64 | 128 | 1.190 / 4768 / 48.54 | 1.196 / 5052 / 46.06 | −5.1% |

### L20X

| c | n | v0.21.0rc2 | main (v0.22) | Δ tput |
|---:|---:|---|---|---:|
| 1 | 20 | 1.332 / 593 / 0.75 | 1.758 / 702 / 0.56 | −25.0% |
| 8 | 80 | 2.249 / 1184 / 3.41 | 2.429 / 1009 / 3.16 | −7.3% |
| 16 | 128 | 4.589 / 8846 / 3.41 | 4.341 / 8196 / 3.55 | **+4.1%** |
| 64 | 128 | 16.410 / 67251 / 3.53 | 16.103 / 69200 / 3.53 | +0.1% |

† v0.21.0rc2's quality phase was `c=2.0, n=100`; main moved it to `c=4, n=200`. The two cells are not directly comparable — see `data/` for raw JSONs.

---

## What the deltas say

- **c=1 (latency):** main regresses 5–25% on RTF/TTFP on both boxes. vllm 0.22's heavier scheduler + chunked prefill costs more on Qwen3-TTS's short Stage 0 steps. Single-stream is the next thing to fix.
- **c=16 / c=64 (throughput):** main wins on Base seed-tts everywhere (+10% to +26%). vllm 0.22's batching better saturates the talker decode loop on long Base prompts. CustomVoice stays neutral (−2% to −8%) — the 65k-token reference-audio prefill dominates the per-step budget.
- **H20-3e vs L20X:** L20X is ~12× slower at c=64 RTF (16.8 vs 1.33 s/audio-s on Base main). Stage 1 (Code2Wav) is memory-bandwidth-bound; H20-3e's ~3.4 TB/s HBM3 vs L20X's ~1.1 TB/s GDDR explains the gap. L20X numbers are best read as a floor for cheaper streaming targets, not an H20 substitute.
- **CustomVoice c=1 < Base c=1** on both boxes — voice clone amortizes speaker conditioning once on the reference prefill; Base re-runs speaker selection per request.
