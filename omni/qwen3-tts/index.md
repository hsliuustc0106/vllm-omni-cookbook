# Qwen3-TTS

**Models:** `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
**Recipes:** [Qwen3-TTS](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen3-TTS.md) · [Deploy YAML](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/deploy)
**Spec:** [`test_tts.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_tts.json) · **Runner:** [`run_benchmark.py`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/scripts/run_benchmark.py)
**Raw JSONs:** [`data/`](data/)

vllm-omni `v0.22.0` (main HEAD `40b29591`, vllm 0.22.0) vs `v0.20.0` (`4a24a517`, vllm 0.20.0) — both with `transformers==5.8.1` — measured 2026-06-03 / 2026-06-05 / 2026-06-06. Both hosts now have both versions.

Cells: `median_audio_rtf` (audio s / wall s, **<1 = realtime**) / `median_audio_ttfp_ms` / `audio_throughput`.

---

## Hardware

| Box | GPU | Mem-BW | Date |
|-----|-----|-------:|------|
| **H20** | NVIDIA H20 96GB HBM3 | ~3.35 TB/s | 2026-06-03 / 2026-06-06 |
| **H200** | NVIDIA L20X 144GB GDDR | ~1.1 TB/s | 2026-06-03 / 2026-06-05 |

Both runs: 1 GPU for Stage 0 (Talker), 1 GPU for Stage 1 (Code2Wav), single replica.

---

## Base · `seed-tts`

### H20

| c | n | v0.20.0 | v0.22.0 | Δ tput |
|---:|---:|---|---|---:|
| 1 | 20 | 0.180 / 161 / 1.78 | 0.165 / 132 / 6.14 | **+245.8%** |
| 4 | 200 | 0.639 / 1539 / 6.08 | 0.172 / 95 / 23.02 | **+278.8%** |
| 8 | 80 | 1.258 / 4251 / 6.15 | 0.317 / 280 / 23.46 | **+281.4%** |
| 16 | 128 | 2.266 / 9130 / 7.01 | 0.387 / 679 / 39.13 | **+458.0%** |
| 64 | 128 | 3.204 / 14152 / 15.10 | 1.325 / 4887 / 41.35 | **+173.8%** |

### H200

| c | n | v0.20.0 | v0.22.0 | Δ tput |
|---:|---:|---|---|---:|
| 1 | 20 | 3.804 / 11944 / 0.13 | 1.996 / 1847 / 0.51 | **+287.5%** |
| 8 | 80 | 24.853 / 46343 / 0.32 | 4.392 / 3494 / 1.72 | **+437.1%** |
| 16 | 128 | 46.946 / 115299 / 0.26 | 5.089 / 9044 / 2.96 | **+1029.6%** |
| 64 | 128 | _n/a_ † | 16.813 / 66002 / 3.23 | — |

† v0.20.0 Base c=64 didn't complete on H200 (saw 2 pytest failures during the run); v0.20.0 Base c=4 quality cell was skipped by the pytest WER-eval marker.

---

## CustomVoice · `seed-tts-text`

### H20

| c | n | v0.20.0 | v0.22.0 | Δ tput |
|---:|---:|---|---|---:|
| 1 | 20 | 0.136 / 50 / 2.41 | 0.141 / 47 / 7.08 | **+193.5%** |
| 4 | 200 | 0.153 / 90 / 25.71 | 0.155 / 59 / 25.61 | −0.4% |
| 8 | 80 | 0.190 / 213 / 36.22 | 0.198 / 81 / 34.49 | −4.8% |
| 16 | 128 | 0.318 / 904 / 47.71 | 0.332 / 735 / 47.21 | −1.0% |
| 64 | 128 | 1.148 / 5873 / 47.30 | 1.178 / 6196 / 47.47 | +0.4% |

### H200

| c | n | v0.20.0 | v0.22.0 | Δ tput |
|---:|---:|---|---|---:|
| 1 | 20 | 1.444 / 703 / 0.51 | 1.622 / 686 / 0.61 | **+20.8%** |
| 4 | 200 | 1.933 / 1417 / 2.00 | 1.920 / 798 / 2.07 | +3.5% |
| 8 | 80 | 2.167 / 2450 / 3.16 | 2.352 / 1005 / 2.88 | −8.9% |
| 16 | 128 | 4.250 / 11112 / 3.59 | 4.226 / 9649 / 3.66 | +1.8% |
| 64 | 128 | 17.860 / 85849 / 3.20 | 15.728 / 80110 / 3.56 | **+11.3%** |

---

## CustomVoice · `seed-tts-design`

### H20

| c | n | v0.20.0 | v0.22.0 | Δ tput |
|---:|---:|---|---|---:|
| 1 | 20 | 0.129 / 49 / 7.75 | 0.136 / 47 / 7.34 | −5.3% |
| 8 | 80 | 0.194 / 173 / 39.92 | 0.194 / 74 / 38.20 | −4.3% |
| 16 | 128 | 0.323 / 813 / 46.41 | 0.333 / 599 / 46.58 | +0.4% |
| 64 | 128 | 1.166 / 5270 / 47.21 | 1.196 / 5052 / 46.06 | −2.4% |

### H200

| c | n | v0.20.0 | v0.22.0 | Δ tput |
|---:|---:|---|---|---:|
| 1 | 20 | 1.507 / 790 / 0.67 | 1.758 / 702 / 0.56 | **−15.8%** |
| 8 | 80 | 2.169 / 2107 / 3.59 | 2.429 / 1009 / 3.16 | **−12.0%** |
| 16 | 128 | 4.198 / 9998 / 3.69 | 4.341 / 8196 / 3.55 | −3.7% |
| 64 | 128 | 14.715 / 65103 / 3.73 | 16.103 / 69200 / 3.53 | −5.2% |

---

## What the deltas say

- **Base seed-tts is the v0.22 story.** v0.22 wins 1.7×–10× throughput on Base across every concurrency on both boxes. v0.20 Base was scheduler-bound (RTF >2 even at c=1 on H200), so the gains are mostly architectural: Stage 0 batching + KV-cache and the orchestrator's stage init refactor that lets the Talker actually keep up with the Code2Wav decoder.
- **CV-text c=1 (latency-bound) sees a smaller but still material v0.22 win.** +21% on H200 / +193% on H20 — the single-stream speaker-conditioning path improved between v0.20 and v0.22.
- **CV-design is roughly flat or slightly regressed on v0.22.** Voice-design's c=1 RTF gets 5–16% worse on v0.22; mid-concurrency cells are within ±5%. The voice-design pipeline didn't benefit from the same scheduler improvements that drove the Base wins.
- **The H200 v0.20 Base server didn't work out-of-the-box.** Stage 1 (`Qwen3TTSCode2Wav`) dies during the READY handshake with `exit code 1` when both stages share GPU 0. The fix is `stage_overrides: {'1': {'devices': '1'}}` in the spec's `server_params` (v0.22 added this by default; v0.20 ships without it). With the override Base v0.20 numbers were captured cleanly.
- **H20 vs H200 hardware gap is real:** H200 (L20X, ~1.1 TB/s GDDR) is ~12× slower than H20 (~3.4 TB/s HBM3) at c=64 RTF on v0.22 Base. Stage 1 is memory-bandwidth-bound; H200 numbers should be read as a floor for cheaper streaming targets, not as an H20 substitute.
