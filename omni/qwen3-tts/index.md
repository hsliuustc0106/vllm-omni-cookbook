# Qwen3-TTS

**Models:** `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
**Recipes:** [Qwen3-TTS](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen3-TTS.md) · [Deploy YAML](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/deploy)
**Spec:** [`test_tts.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_tts.json) · **Runner:** [`run_benchmark.py`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/scripts/run_benchmark.py)
**Raw JSONs:** [`data/`](data/)

vllm-omni `v0.22.0` (main HEAD `40b29591`, vllm 0.22.0) vs `v0.20.0` (`4a24a517`, vllm 0.20.0) — both with `transformers==5.8.1` — measured 2026-06-03 / 2026-06-05. H200 has both versions; H20 has v0.22.0 only.

Cells: `median_audio_rtf` (audio s / wall s, **<1 = realtime**) / `median_audio_ttfp_ms` / `audio_throughput`.

---

## Hardware

| Box | GPU | Mem-BW | Date |
|-----|-----|-------:|------|
| **H20** | NVIDIA H20 96GB HBM3 | ~3.35 TB/s | 2026-06-03 |
| **H200** | NVIDIA L20X 144GB GDDR | ~1.1 TB/s | 2026-06-03/05 |

Both runs: 1 GPU for Stage 0 (Talker), 1 GPU for Stage 1 (Code2Wav), single replica.

---

## Base · `seed-tts`

### H20

| c | n | v0.22.0 |
|---:|---:|---|
| 1 | 20 | 0.165 / 132 / 6.14 |
| 4 | 200 | 0.172 / 95 / 23.02 |
| 8 | 80 | 0.317 / 280 / 23.46 |
| 16 | 128 | 0.387 / 679 / 39.13 |
| 64 | 128 | 1.325 / 4887 / 41.35 |

### H200

| c | n | v0.20.0 | v0.22.0 | Δ tput |
|---:|---:|---|---|---:|
| 1 | 20 | _n/a_ † | 1.996 / 1847 / 0.51 | — |
| 8 | 80 | _n/a_ † | 4.392 / 3494 / 1.72 | — |
| 16 | 128 | _n/a_ † | 5.089 / 9044 / 2.96 | — |
| 64 | 128 | _n/a_ † | 16.813 / 66002 / 3.23 | — |

† v0.20.0 Base server is unstable on H200 — Stage 1 (`Qwen3TTSCode2Wav`) dies during the READY handshake with `exit code 1`. CV server works fine. Base c=4 quality cell skipped by the pytest marker on H200 v0.22.0.

---

## CustomVoice · `seed-tts-text`

### H20

| c | n | v0.22.0 |
|---:|---:|---|
| 1 | 20 | 0.141 / 47 / 7.08 |
| 4 | 200 | 0.155 / 59 / 25.61 |
| 8 | 80 | 0.198 / 81 / 34.49 |
| 16 | 128 | 0.332 / 735 / 47.21 |
| 64 | 128 | 1.178 / 6196 / 47.47 |

### H200

| c | n | v0.20.0 | v0.22.0 | Δ tput |
|---:|---:|---|---|---:|
| 1 | 20 | 1.444 / 703 / 0.51 | 1.622 / 686 / 0.61 | **+19.6%** |
| 4 | 200 | 1.933 / 1417 / 2.00 | 1.920 / 798 / 2.07 | +3.5% |
| 8 | 80 | 2.167 / 2450 / 3.16 | 2.352 / 1005 / 2.88 | −8.9% |
| 16 | 128 | 4.250 / 11112 / 3.59 | 4.226 / 9649 / 3.66 | +1.9% |
| 64 | 128 | 17.860 / 85849 / 3.20 | 15.728 / 80110 / 3.56 | **+11.2%** |

---

## CustomVoice · `seed-tts-design`

### H20

| c | n | v0.22.0 |
|---:|---:|---|
| 1 | 20 | 0.136 / 47 / 7.34 |
| 8 | 80 | 0.194 / 74 / 38.20 |
| 16 | 128 | 0.333 / 599 / 46.58 |
| 64 | 128 | 1.196 / 5052 / 46.06 |

### H200

| c | n | v0.20.0 | v0.22.0 | Δ tput |
|---:|---:|---|---|---:|
| 1 | 20 | 1.507 / 790 / 0.67 | 1.758 / 702 / 0.56 | **−16.4%** |
| 8 | 80 | 2.169 / 2107 / 3.59 | 2.429 / 1009 / 3.16 | **−12.0%** |
| 16 | 128 | 4.198 / 9998 / 3.69 | 4.341 / 8196 / 3.55 | −3.8% |
| 64 | 128 | 14.715 / 65103 / 3.73 | 16.103 / 69200 / 3.53 | −5.4% |

---

## What the deltas say

- **v0.20 → v0.22 is a mixed picture, not strictly faster.** On H200 CV-text c=64 v0.22.0 wins on TTFP/RTF (15.7 / 80110 ms vs 17.9 / 85849 ms) and tput (+11%). But on CV-design c=64 v0.20.0 actually *beats* v0.22.0 on RTF (14.72 vs 16.10) while tput is a wash. Worth checking individual cells before assuming "newer = faster".
- **CV-text c=1 (latency):** v0.22.0 regresses ~12% on RTF (1.62 vs 1.44) and trades 17 ms of TTFP. vllm 0.22's heavier scheduler + chunked prefill costs more on Qwen3-TTS's short Stage 0 steps. Single-stream is the next thing to fix.
- **H20 vs H200:** H200 is ~12× slower at c=64 RTF (16.8 vs 1.33 s/audio-s on Base v0.22.0). Stage 1 (Code2Wav) is memory-bandwidth-bound; H20's ~3.4 TB/s HBM3 vs H200's ~1.1 TB/s GDDR explains the gap. H200 numbers are best read as a floor for cheaper streaming targets, not an H20 substitute.
- **CustomVoice c=1 < Base c=1** on both boxes — voice clone amortizes speaker conditioning once on the reference prefill; Base re-runs speaker selection per request.
- **v0.20 Base server is broken on H200.** Stage 1 dies during the READY handshake with `exit code 1`; the bench never reaches its cells. Fixed by v0.22.0 — likely the same broader Stage init refactor that landed `--stage-init-timeout` / `--init-timeout` support in `run_benchmark.py`.
