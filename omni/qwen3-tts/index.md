# Qwen3-TTS

**Models:** `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
**Recipes:** [Qwen3-TTS](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen3-TTS.md) · [Deploy YAML](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/deploy)
**Spec:** [`test_tts.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_tts.json) · **Runner:** [`run_benchmark.py`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/scripts/run_benchmark.py)
**Raw JSONs:** [`data/`](data/)

vllm-omni `v0.22.0` (main HEAD `40b29591`, vllm 0.22.0) vs `v0.20.0` (`4a24a517`, vllm 0.20.0) — both with `transformers==5.8.1` — measured 2026-06-03 / 2026-06-05 / 2026-06-06.

Metrics: **RTF** = `median_audio_rtf` (audio s / wall s, **<1 = realtime**) · **TTFP** = `median_audio_ttfp_ms` · **Tput** = `audio_throughput`.

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
| 1 | 20 | 3.804 | 1.996 | 11944 | 1847 | 0.13 | 0.51 | **+287.5%** |
| 8 | 80 | 24.853 | 4.392 | 46343 | 3494 | 0.32 | 1.72 | **+437.1%** |
| 16 | 128 | 46.946 | 5.089 | 115299 | 9044 | 0.26 | 2.96 | **+1029.6%** |
| 64 | 128 | 176.121 | 16.813 | 698693 | 66002 | 0.32 | 3.23 | **+916.4%** |

† Base c=4 quality was skipped on both versions (pytest WER-eval marker on v0.22, framework skip on v0.20). v0.20.0 c=64 originally crashed the server when run after c=8/c=16 (accumulated state in the orchestrator); re-running c=64 against a fresh server completes cleanly — the numbers shown are from that isolated re-run.

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
| 1 | 20 | 1.444 | 1.622 | 703 | 686 | 0.51 | 0.61 | **+20.8%** |
| 4 | 200 | 1.933 | 1.920 | 1417 | 798 | 2.00 | 2.07 | +3.5% |
| 8 | 80 | 2.167 | 2.352 | 2450 | 1005 | 3.16 | 2.88 | −8.9% |
| 16 | 128 | 4.250 | 4.226 | 11112 | 9649 | 3.59 | 3.66 | +1.8% |
| 64 | 128 | 17.860 | 15.728 | 85849 | 80110 | 3.20 | 3.56 | **+11.3%** |

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
| 1 | 20 | 1.507 | 1.758 | 790 | 702 | 0.67 | 0.56 | **−15.8%** |
| 8 | 80 | 2.169 | 2.429 | 2107 | 1009 | 3.59 | 3.16 | **−12.0%** |
| 16 | 128 | 4.198 | 4.341 | 9998 | 8196 | 3.69 | 3.55 | −3.7% |
| 64 | 128 | 14.715 | 16.103 | 65103 | 69200 | 3.73 | 3.53 | −5.2% |

---

## What the deltas say

- **Base seed-tts is the v0.22 story.** v0.22 wins 1.7×–10× throughput on Base across every concurrency on both boxes. v0.20 Base was scheduler-bound (RTF >2 even at c=1 on H200), so the gains are mostly architectural: Stage 0 batching + KV-cache and the orchestrator's stage init refactor that lets the Talker actually keep up with the Code2Wav decoder.
- **CV-text c=1 (latency-bound) sees a smaller but still material v0.22 win.** +21% on H200 / +193% on H20 — the single-stream speaker-conditioning path improved between v0.20 and v0.22.
- **CV-design is roughly flat or slightly regressed on v0.22.** Voice-design's c=1 RTF gets 5–16% worse on v0.22; mid-concurrency cells are within ±5%. The voice-design pipeline didn't benefit from the same scheduler improvements that drove the Base wins.
- **The H200 v0.20 Base server didn't work out-of-the-box.** Stage 1 (`Qwen3TTSCode2Wav`) dies during the READY handshake with `exit code 1` when both stages share GPU 0. The fix is `stage_overrides: {'1': {'devices': '1'}}` in the spec's `server_params` (v0.22 added this by default; v0.20 ships without it). With the override Base v0.20 numbers were captured cleanly.
- **H20 vs H200 hardware gap is real:** H200 (L20X, ~1.1 TB/s GDDR) is ~12× slower than H20 (~3.4 TB/s HBM3) at c=64 RTF on v0.22 Base. Stage 1 is memory-bandwidth-bound; H200 numbers should be read as a floor for cheaper streaming targets, not as an H20 substitute.
