# Qwen3-TTS perf raw data — 2026-06-03

Raw JSON outputs from `tests/dfx/perf/scripts/run_benchmark.py --test-config-file tests/dfx/perf/tests/test_tts.json`. Captured outside vllm-omni CI so we can re-render the cookbook page without re-running benches.

## Cells per (host, version)

Per `omni_server` (Base = `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, CustomVoice = `Qwen3-TTS-12Hz-1.7B-CustomVoice`):
- **latency** — c=1, n=20
- **throughput** — c=8/16/64, n=80/128/128
- **quality** (UTMOS + WER) — c=4, n=200, three `seed-tts*` partitions
- `stress` phases disabled (saved CI from hours of long-tail probes)

Total 14 result JSONs per (host, version):
- Base seed-tts: c=1, 4, 8, 16, 64
- CustomVoice seed-tts-text: c=1, 4, 8, 16, 64
- CustomVoice seed-tts-design: c=1, 8, 16, 64

## Versions

| Slot | vllm-omni SHA | vllm | transformers |
|---|---|---|---|
| `main` | `40b29591` (Add Cosmos3 sound generation, #4073) | 0.22.0 | 5.8.1 |
| `v021rc2` | `v0.21.0rc2` tag → `5f9aee19` | 0.21.0 | 5.8.1 (downgraded from 5.9.0 to dodge `create_causal_mask(input_embeds=)` drop) |

## Hosts

| Host class | dir | GPUs used | notes |
|---|---|---|---|
| H20 (Alibaba Cloud) | `h20-server-1/` | physical idx `4,5` (idx `0-3` busy with other tenants at run time) | `VLLM_USE_FLASHINFER_SAMPLER=0` (no nvcc) |
| H200 (silicon: L20X) | `h200-hsliu/` | physical idx `0,1` (idx `2,3` busy with `tencent/HunyuanImage-3.0` at run time) | `VLLM_USE_FLASHINFER_SAMPLER=0` (no nvcc), `--stage-init-timeout 1800`, `--init-timeout 2400` |

## Local patches applied (not for upstream — workarounds for this bench session only)

1. **`vllm_omni/entrypoints/cli/benchmark/__init__.py`** — added `from . import serve` so `OmniBenchmarkServingSubcommand` registers when `--omni` delegates. Without it, `OmniBenchmarkSubcommandBase.__subclasses__()` returns empty and `--dataset-name` choices never gets extended.
2. **`vllm_omni/entrypoints/cli/benchmark/serve.py`** — after the existing `dataset_name` choices extension on `parser._actions`, also walk `parser._shadow._actions` and apply the same extension. `TrackingArgumentParser` builds a shadow parser used at parse-time; without extending the shadow, argparse rejects `seed-tts` at parse despite the real action having extended choices.
3. **(v021rc2 only)** `tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` line 568 — `"input_embeds"` → `"inputs_embeds"`. main has a runtime `if "input_embeds" in sig.parameters` guard.
4. **(h200 only)** `tests/helpers/runtime.py` — default `stage_init_timeout` 600 → 1800.

## Methodology

- Sequential per (host, version) — never two vllm-omni servers competing for the same GPUs.
- Stage 1 pinned to local device `1` per `test_tts.json` `stage_overrides`.
- Each `result_*.json` is from one bench run; CI defaults (`--num-warmups 2`, `--save-result`).
- `result_json_count` per dir = 14 (h20) once all enabled phases land.

## Headline numbers (h20-server-1 Base seed-tts)

`median_audio_rtf` / `median_audio_ttfp_ms` / `audio_throughput` :

| c | v0.21.0rc2 | main (`40b29591`) | Δ tput |
|---:|---|---|---:|
| 1 | 0.153 / 131 ms / 6.54 | 0.165 / 132 ms / 6.14 | −6.1% |
| 4 | 0.182 / 114 ms / 21.76 | 0.172 / 95 ms / 23.02 | +5.8% |
| 8 | 0.304 / 266 ms / 24.15 | 0.317 / 280 ms / 23.46 | −2.9% |
| 16 | 0.443 / 784 ms / 35.38 | 0.387 / 679 ms / 39.13 | **+10.6%** |
| 64 | 1.494 / 5832 ms / 35.99 | 1.325 / 4887 ms / 41.35 | **+14.9%** |

CustomVoice cells regress by 2–8% on main (known tradeoff per maintainer).

See `index.md` for the full narrative.
