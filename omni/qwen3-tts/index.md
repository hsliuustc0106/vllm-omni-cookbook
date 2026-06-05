# Qwen3-TTS

**Models:** `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
**Recipes:** [Qwen3-TTS](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen3-TTS.md) · [Deploy YAML](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/deploy)
**Spec:** [`test_tts.json`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/tests/test_tts.json) · **Runner:** [`run_benchmark.py`](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/perf/scripts/run_benchmark.py)
**Raw JSONs:** [`data/`](data/)

vllm-omni `main` (`40b29591`, vllm 0.22.0) vs `v0.21.0rc2` (`5f9aee19`, vllm 0.21.0) vs `v0.20.0` (`4a24a517`, vllm 0.20.0) — all with `transformers==5.8.1` — measured on two boxes 2026-06-03 / 2026-06-04 / 2026-06-05. H200 has the full three-version comparison; H20 has main vs v0.21.0rc2 only.

Cells: `median_audio_rtf` (audio s / wall s, **<1 = realtime**) / `median_audio_ttfp_ms` / `audio_throughput`.

---

## Hardware

| Box | GPU | Mem-BW | Date |
|-----|-----|-------:|------|
| **H20** | NVIDIA H20 96GB HBM3 | ~3.35 TB/s | 2026-06-03 |
| **H200** | NVIDIA L20X 144GB GDDR | ~1.1 TB/s | 2026-06-03/04 |

Both runs: 1 GPU for Stage 0 (Talker), 1 GPU for Stage 1 (Code2Wav), single replica.

---

## Base · `seed-tts`

### H20

| c | n | v0.21.0rc2 | main (v0.22) | Δ tput |
|---:|---:|---|---|---:|
| 1 | 20 | 0.153 / 131 / 6.54 | 0.165 / 132 / 6.14 | −6.1% |
| 4 | 200 | 0.182 / 114 / 21.76 | 0.172 / 95 / 23.02 | **+5.8%** |
| 8 | 80 | 0.304 / 266 / 24.15 | 0.317 / 280 / 23.46 | −2.9% |
| 16 | 128 | 0.443 / 784 / 35.38 | 0.387 / 679 / 39.13 | **+10.6%** |
| 64 | 128 | 1.494 / 5832 / 35.99 | 1.325 / 4887 / 41.35 | **+14.9%** |

### H200

| c | n | v0.20.0 | v0.21.0rc2 | main (v0.22) |
|---:|---:|---|---|---|
| 1 | 20 | _n/a_ † | 1.582 / 1597 / 0.62 | 1.996 / 1847 / 0.51 |
| 8 | 80 | _n/a_ † | 4.067 / 3192 / 1.83 | 4.392 / 3494 / 1.72 |
| 16 | 128 | _n/a_ † | 6.105 / 11394 / 2.52 | 5.089 / 9044 / 2.96 |
| 64 | 128 | _n/a_ † | 20.893 / 84451 / 2.57 | 16.813 / 66002 / 3.23 |

† v0.20.0 Base server is unstable on H200 — Stage 1 (`Qwen3TTSCode2Wav`) dies during the READY handshake with `exit code 1`. CV server works fine. Base c=4 quality cell skipped by the pytest marker on H200 main.

---

## CustomVoice · `seed-tts-text`

### H20

| c | n | v0.21.0rc2 | main (v0.22) | Δ tput |
|---:|---:|---|---|---:|
| 1 | 20 | 0.134 / 44 / 7.48 | 0.141 / 47 / 7.08 | −5.3% |
| 4 | 200 | 0.149 / 56 / 26.54 | 0.155 / 59 / 25.61 | −3.5% |
| 8 | 80 | 0.189 / 71 / 36.31 | 0.198 / 81 / 34.49 | −5.0% |
| 16 | 128 | 0.326 / 710 / 48.06 | 0.332 / 735 / 47.21 | −1.8% |
| 64 | 128 | 1.209 / 5776 / 48.58 | 1.178 / 6196 / 47.47 | −2.3% |

### H200

| c | n | v0.20.0 | v0.21.0rc2 | main (v0.22) |
|---:|---:|---|---|---|
| 1 | 20 | 1.444 / 703 / 0.51 | 1.314 / 595 / 0.77 | 1.622 / 686 / 0.61 |
| 4 | 200 | 1.933 / 1417 / 2.00 | _n/a_ | 1.920 / 798 / 2.07 |
| 8 | 80 | 2.167 / 2450 / 3.16 | 2.261 / 1107 / 3.09 | 2.352 / 1005 / 2.88 |
| 16 | 128 | 4.250 / 11112 / 3.59 | 4.139 / 9400 / 3.83 | 4.226 / 9649 / 3.66 |
| 64 | 128 | 17.860 / 85849 / 3.20 | 14.828 / 75758 / 3.89 | 15.728 / 80110 / 3.56 |

---

## CustomVoice · `seed-tts-design`

### H20

| c | n | v0.21.0rc2 | main (v0.22) | Δ tput |
|---:|---:|---|---|---:|
| 1 | 20 | 0.126 / 43 / 7.88 | 0.136 / 47 / 7.34 | −6.9% |
| 8 | 80 | 0.185 / 72 / 41.76 | 0.194 / 74 / 38.20 | −8.5% |
| 16 | 128 | 0.314 / 581 / 48.32 | 0.333 / 599 / 46.58 | −3.6% |
| 64 | 128 | 1.190 / 4768 / 48.54 | 1.196 / 5052 / 46.06 | −5.1% |

### H200

| c | n | v0.20.0 | v0.21.0rc2 | main (v0.22) |
|---:|---:|---|---|---|
| 1 | 20 | 1.507 / 790 / 0.67 | 1.332 / 593 / 0.75 | 1.758 / 702 / 0.56 |
| 8 | 80 | 2.169 / 2107 / 3.59 | 2.249 / 1184 / 3.41 | 2.429 / 1009 / 3.16 |
| 16 | 128 | 4.198 / 9998 / 3.69 | 4.589 / 8846 / 3.41 | 4.341 / 8196 / 3.55 |
| 64 | 128 | 14.715 / 65103 / 3.73 | 16.410 / 67251 / 3.53 | 16.103 / 69200 / 3.53 |

CV-design quality cells (c=4) aren't part of the v0.20–main `test_tts.json` matrix. v0.21.0rc2's c=2.0/n=100 cells are the stress phase and are omitted here for the same reason quality cells were dropped earlier — they don't appear in the other two versions.

---

## What the deltas say

- **v0.21.0rc2 is close to main.** On H200 CV-text the two versions differ by 1–8% on every metric across c=1/8/16/64; on Base seed-tts the gap is wider (main wins +17% c=16 / +26% c=64 throughput, regresses 5–19% at c=1). v0.21.0rc2 is the right baseline for "what does upgrading buy you" questions.
- **v0.20 → v0.21.0rc2 is the bigger story.** On H200 CV-text c=64: v0.20 17.86 RTF vs v0.21.0rc2 14.83 (−17% RTF) but only 3.20 → 3.89 throughput. On CV-design c=64 v0.20 actually wins (14.72 vs 16.41). The throughput dimension is where v0.21+ pays off; raw RTF is a mixed picture. Worth checking individual cells before assuming "newer = faster".
- **c=1 (latency):** main regresses 5–25% on RTF/TTFP on both boxes vs v0.21.0rc2. vllm 0.22's heavier scheduler + chunked prefill costs more on Qwen3-TTS's short Stage 0 steps. Single-stream is the next thing to fix.
- **c=16 / c=64 (throughput):** main wins on Base seed-tts everywhere (+10% to +26% vs v0.21.0rc2). vllm 0.22's batching better saturates the talker decode loop on long Base prompts. CustomVoice stays neutral (−2% to −8%) — the 65k-token reference-audio prefill dominates the per-step budget.
- **H20 vs H200:** H200 is ~12× slower at c=64 RTF (16.8 vs 1.33 s/audio-s on Base main). Stage 1 (Code2Wav) is memory-bandwidth-bound; H20's ~3.4 TB/s HBM3 vs H200's ~1.1 TB/s GDDR explains the gap. H200 numbers are best read as a floor for cheaper streaming targets, not an H20 substitute.
- **CustomVoice c=1 < Base c=1** on both boxes — voice clone amortizes speaker conditioning once on the reference prefill; Base re-runs speaker selection per request.
- **v0.20 Base server is broken on H200.** Stage 1 dies during the READY handshake with `exit code 1`; the bench never reaches its cells. v0.21.0rc2 fixed this — likely the same broader Stage init refactor that landed `--stage-init-timeout`/`--init-timeout` support in `run_benchmark.py`.
