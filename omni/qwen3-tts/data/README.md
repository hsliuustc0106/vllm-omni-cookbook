# Qwen3-TTS perf raw data — H20 2026-06-03 / 2026-06-06, L20X 2026-06-08

Raw JSON outputs from `tests/dfx/perf/scripts/run_benchmark.py --test-config-file tests/dfx/perf/tests/test_tts.json`. Captured outside vllm-omni CI so we can re-render the cookbook page without re-running benches.

> **L20X re-bench (2026-06-08):** the first revision's `h200/` (L20X) JSONs were captured while an unrelated host-side CPU process saturated the box and starved the latency-bound Talker (single-digit % GPU SM util). They have been fully re-benched on the cleaned host; the `h200/{v020,v022}` JSONs here are the clean set. The bench-client patches below are required for `--dataset-name seed-tts*` to parse — without them every request fails (`completed == 0`) and the harness writes zero-filled JSONs.

## Cells per (host, version)

Per `omni_server` (Base = `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, CustomVoice = `Qwen3-TTS-12Hz-1.7B-CustomVoice`):
- **latency** — c=1, n=20
- **throughput** — c=8/16/64, n=80/128/128
- **quality** (UTMOS + WER) — c=4, n=200, three `seed-tts*` partitions
- `stress` phases disabled (saved CI from hours of long-tail probes)

Expected per (host, version) — 14 cells when the full matrix runs:
- Base seed-tts: c=1, 4, 8, 16, 64
- CustomVoice seed-tts-text: c=1, 4, 8, 16, 64
- CustomVoice seed-tts-design: c=1, 8, 16, 64

## Versions

| Slot | vllm-omni SHA | vllm | transformers |
|---|---|---|---|
| `v022` | main HEAD `40b29591` (Add Cosmos3 sound generation, #4073) | 0.22.0 | 5.8.1 |
| `v020` | `v0.20.0` tag → `4a24a517` | 0.20.0 | 5.8.1 |

## Hosts

| Host class | dir | notes |
|---|---|---|
| H20 | `h20/` | NVIDIA H20 96GB HBM3 · `VLLM_USE_FLASHINFER_SAMPLER=0` (no nvcc) |
| H200 | `h200/` | NVIDIA L20X 144GB GDDR · `VLLM_USE_FLASHINFER_SAMPLER=0`, `--stage-init-timeout 1800`, `--init-timeout 2400` |

## Local patches applied (not for upstream — workarounds for this bench session only)

1. **`vllm_omni/entrypoints/cli/benchmark/__init__.py`** — added `from . import serve` so `OmniBenchmarkServingSubcommand` registers when `--omni` delegates. Without it, `OmniBenchmarkSubcommandBase.__subclasses__()` returns empty and `--dataset-name` choices never gets extended.
2. **`vllm_omni/entrypoints/cli/benchmark/serve.py`** — after the existing `dataset_name` choices extension on `parser._actions`, also walk `parser._shadow._actions` and apply the same extension. `TrackingArgumentParser` builds a shadow parser used at parse-time; without extending the shadow, argparse rejects `seed-tts` at parse despite the real action having extended choices.
3. **(h200 only)** `tests/dfx/perf/scripts/run_benchmark.py:76` — `--stage-init-timeout` 600 → 1800, `--init-timeout` 900 → 2400. Stage 0's torch.compile cold-cache pass takes ~5 min on H200.
4. **(v020 only)** `tests/dfx/perf/tests/test_tts.json` — Base server entries had no `dataset_path` field; added `linyueqian/seed-tts-eval-subset` to make the bench client happy. Also expanded `max_concurrency=[8]` throughput entries to `[8,16,64]` to match the v0.22.0 matrix.

## Methodology

- Sequential per (host, version) — never two vllm-omni servers competing for the same GPUs.
- Stage 1 pinned to local device `1` per `test_tts.json` `stage_overrides` (v0.22.0); v0.20.0 puts both stages on the same GPU.
- Each `result_*.json` is from one bench run; CI defaults (`--num-warmups 2`, `--save-result`).
- On L20X, the throughput phase (c=8/16/64 on one server) degrades the orchestrator and zeroes out the c≥16 cells — v0.20 Base is the most fragile. The clean re-bench captures each high-concurrency cell against a fresh server, so `h200/{v020,v022}` now hold the full 14-cell matrix (Base c=1/4/8/16/64 + CV-text c=1/4/8/16/64 + CV-design c=1/8/16/64).

## Headline numbers (H20 · Base seed-tts · v0.22.0)

`median_audio_rtf` / `median_audio_ttfp_ms` / `audio_throughput`:

| c | v0.22.0 |
|---:|---|
| 1 | 0.165 / 132 / 6.14 |
| 4 | 0.172 / 95 / 23.02 |
| 8 | 0.317 / 280 / 23.46 |
| 16 | 0.387 / 679 / 39.13 |
| 64 | 1.325 / 4887 / 41.35 |

See `index.md` for the full v0.20 vs v0.22 narrative across both hosts.
