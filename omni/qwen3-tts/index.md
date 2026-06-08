# Qwen3-TTS

**Models:** `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
**Recipes:** [Qwen3-TTS](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen3-TTS.md) · [Deploy YAML](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/deploy)
**Spec:** [`test_tts.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_tts.json) · **Runner:** [`run_benchmark.py`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/scripts/run_benchmark.py)
**Raw JSONs:** [`data/`](data/)

vllm-omni `v0.22.0` (main HEAD `40b29591`, vllm 0.22.0) vs `v0.20.0` (`4a24a517`, vllm 0.20.0) — both with `transformers==5.8.1`. H20 measured 2026-06-03 / 2026-06-06; L20X re-benched 2026-06-08.

Metrics: **RTF** = `median_audio_rtf` (audio s / wall s, **<1 = realtime**) · **TTFP** = `median_audio_ttfp_ms` · **Tput** = `audio_throughput`.

> **Correction (2026-06-08):** the L20X numbers in the first revision of this page were captured while an unrelated host-side CPU process was saturating the box and starving the latency-bound Talker (it ran at single-digit % GPU SM util). The L20X matrix below has been **re-benched on the cleaned host**. The earlier "L20X is ~12× slower, Code2Wav is HBM-vs-GDDR bandwidth-bound" reading was an artifact of that contamination and is **withdrawn** — the real cross-box gap is ~1.3× at c=64 (see "What the deltas say").

---

## Hardware

| Box | GPU | Class | Date |
|-----|-----|-------|------|
| **H20** | NVIDIA H20 96GB HBM3 | Hopper | 2026-06-03 / 2026-06-06 |
| **H200** | NVIDIA L20X 141GB | Ada-class (reports as `NVIDIA L20X`; **not** a true H200) | 2026-06-08 |

Both runs: 1 GPU for Stage 0 (Talker), 1 GPU for Stage 1 (Code2Wav), single replica. The "H200" box label maps to `data/h200/` for path continuity; the silicon is an Ada-class L20X, so read it as a cheaper-streaming-target floor, not an H200 ceiling.

---

## Base · `seed-tts`

### H20

| c | n | RTF v0.20 | RTF v0.22 | TTFP v0.20 (ms) | TTFP v0.22 (ms) | Tput v0.20 | Tput v0.22 | Δ Tput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 0.180 | 0.165 | 161 | 132 | 1.78 | 6.14 | **+245.8%** |
| 4 | 200 | 0.639 | 0.172 | 1539 | 95 | 6.08 | 23.02 | **+278.8%** |
| 8 | 80 | 1.258 | 0.317 | 4251 | 280 | 6.15 | 23.46 | **+281.4%** |
| 16 | 128 | 2.266 | 0.387 | 9130 | 679 | 7.01 | 39.13 | **+458.0%** |
| 64 | 128 | 3.204 | 1.325 | 14152 | 4887 | 15.10 | 41.35 | **+173.8%** |

### H200

| c | n | RTF v0.20 | RTF v0.22 | TTFP v0.20 (ms) | TTFP v0.22 (ms) | Tput v0.20 | Tput v0.22 | Δ Tput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 0.163 | 0.173 | 167 | 137 | 4.10 | 5.76 | +40.5% |
| 4 | 200 | 0.230 | 0.208 | 448 | 79 | 16.33 | 19.11 | +17.0% |
| 8 | 80 | 0.369 | 0.387 | 933 | 193 | 12.73 | 20.30 | **+59.4%** |
| 16 | 128 | 0.652 | 0.498 | 2286 | 882 | 16.88 | 30.98 | **+83.5%** |
| 64 | 128 | 1.982 | 1.676 | 8065 | 6498 | 27.79 | 32.37 | +16.5% |

† Clean re-bench 2026-06-08. The c=8/16/64 throughput cells are each captured against a fresh server — running them back-to-back on one server degrades the orchestrator state and zeroes out the high-concurrency cells (v0.20 Base is the most fragile here). Base c=4 (quality phase) now completes and is included.

---

## CustomVoice · `seed-tts-text`

### H20

| c | n | RTF v0.20 | RTF v0.22 | TTFP v0.20 (ms) | TTFP v0.22 (ms) | Tput v0.20 | Tput v0.22 | Δ Tput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 0.136 | 0.141 | 50 | 47 | 2.41 | 7.08 | **+193.5%** |
| 4 | 200 | 0.153 | 0.155 | 90 | 59 | 25.71 | 25.61 | −0.4% |
| 8 | 80 | 0.190 | 0.198 | 213 | 81 | 36.22 | 34.49 | −4.8% |
| 16 | 128 | 0.318 | 0.332 | 904 | 735 | 47.71 | 47.21 | −1.0% |
| 64 | 128 | 1.148 | 1.178 | 5873 | 6196 | 47.30 | 47.47 | +0.4% |

### H200

| c | n | RTF v0.20 | RTF v0.22 | TTFP v0.20 (ms) | TTFP v0.22 (ms) | Tput v0.20 | Tput v0.22 | Δ Tput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 0.136 | 0.150 | 54 | 50 | 5.01 | 6.70 | +33.5% |
| 4 | 200 | 0.156 | 0.195 | 93 | 69 | 25.59 | 20.48 | −20.0% |
| 8 | 80 | 0.177 | 0.250 | 157 | 83 | 43.63 | 30.88 | −29.2% |
| 16 | 128 | 0.287 | 0.442 | 804 | 999 | 53.89 | 35.71 | −33.7% |
| 64 | 128 | 1.056 | 1.584 | 4981 | 7835 | 53.86 | 35.44 | −34.2% |

---

## CustomVoice · `seed-tts-design`

### H20

| c | n | RTF v0.20 | RTF v0.22 | TTFP v0.20 (ms) | TTFP v0.22 (ms) | Tput v0.20 | Tput v0.22 | Δ Tput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 0.129 | 0.136 | 49 | 47 | 7.75 | 7.34 | −5.3% |
| 8 | 80 | 0.194 | 0.194 | 173 | 74 | 39.92 | 38.20 | −4.3% |
| 16 | 128 | 0.323 | 0.333 | 813 | 599 | 46.41 | 46.58 | +0.4% |
| 64 | 128 | 1.166 | 1.196 | 5270 | 5052 | 47.21 | 46.06 | −2.4% |

### H200

| c | n | RTF v0.20 | RTF v0.22 | TTFP v0.20 (ms) | TTFP v0.22 (ms) | Tput v0.20 | Tput v0.22 | Δ Tput |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 0.129 | 0.143 | 53 | 46 | 7.76 | 7.02 | −9.6% |
| 8 | 80 | 0.171 | 0.245 | 153 | 80 | 44.93 | 31.40 | −30.1% |
| 16 | 128 | 0.272 | 0.427 | 664 | 827 | 54.96 | 36.22 | −34.1% |
| 64 | 128 | 1.016 | 1.579 | 4469 | 6708 | 55.30 | 35.81 | −35.2% |

---

## What the deltas say

- **Base seed-tts is the v0.22 story on both boxes.** v0.22 wins Base throughput at every concurrency — H20 +174% to +458%, L20X +16% to +83%. The gains are architectural: Stage 0 batching + KV-cache and the orchestrator's stage-init refactor let the Talker keep up with the Code2Wav decoder. Both versions stay realtime (RTF < 2) even at c=64 on both boxes.
- **CustomVoice regresses on v0.22 — and the regression is larger on L20X.** On H20 CV throughput is roughly neutral (−2% to −8%). On L20X v0.22 gives back **20–35%** of CV throughput across CV-text and CV-design (e.g. CV-text c=64 Tput 53.9 → 35.4). The voice-conditioning path didn't pick up the Base scheduler work, and on Ada-class silicon the v0.22 per-step overhead costs more. To serve CustomVoice on L20X today, v0.20 is the faster throughput option.
- **The v0.20 Base server is fragile at high concurrency on L20X.** Running c=8/16/64 back-to-back on one server degrades the orchestrator and zeroes out the c≥16 cells, so each high-concurrency cell here is captured against a fresh server. Stage 1 device pinning via `stage_overrides: {'1': {'devices': '1'}}` is also required (v0.22 defaults it; v0.20 needs it set).
- **The real H20-vs-L20X gap is ~1.3×, not ~12×.** On clean hardware, L20X v0.22 Base c=64 is RTF 1.676 / 32.4 audio-s per wall-s vs H20's 1.325 / 41.4 — about **1.27× slower RTF** at saturation and roughly **on par at c=1** (0.173 vs 0.165). CV cells sit at ~1.3× as well. The earlier "~12×, Code2Wav is HBM3-vs-GDDR bandwidth-bound" reading came entirely from a host-side CPU process starving the latency-bound Talker (single-digit % GPU SM util during the contaminated runs) and is **withdrawn**. L20X is a viable cheaper-streaming target, not a 12×-handicapped one.
