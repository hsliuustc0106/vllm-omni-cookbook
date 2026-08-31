# Host Weight Runtime TP validation evidence

This directory preserves the compact evidence used by the Host Weight Runtime
blog post. It covers MiniMax-H3 BF16 no-AllGather distributed layerwise
offload (DLO) at vLLM-Omni source `497c537c6f70e44f376b491bf7b50395cf2cba5d`.

## Environment

| Field | Value |
|---|---|
| Date | 2026-08-26 |
| vLLM | 0.27.0 |
| vLLM-Omni source | `497c537c6f70e44f376b491bf7b50395cf2cba5d` |
| Installed vLLM-Omni distribution metadata | `0.27.0rc2.dev60+ge2721dc97`; imports were forced to the source checkout above |
| PyTorch / CUDA | 2.13.0+cu129 / CUDA 12.9 |
| Model | MiniMax-H3 FL2VA snapshot `42ed227ee7df40d41602854ae760620d6eb651fe` |
| Hardware | 4× NVIDIA H200 in one NUMA-0 NV18 domain |
| Internal telemetry label | `NVIDIA L20X`; this cluster label denotes the H200 SKU used by the cookbook |
| Workload | T2VA, 1344×768, 8.7 s, seed 1101, three sigma points / two denoiser evaluations |
| DLO placement | no-AllGather, TP2, text-encoder TP2, VAE tile PP2, 20 resident DiT blocks |

All CUDA work used `gpu run`. The benchmark fixed model, prompt, seed,
attention backend, execution mode, topology, store, and NUMA placement. The
registered and staged pair blocks each excluded one warmup, measured three
concurrent Engine-A/Engine-B waves, and profiled one request per engine after
the memory observer had quiesced.

## Primary results

| Metric | Existing path | HWR staged | HWR registered |
|---|---:|---:|---:|
| Pair process-tree PSS | 388.06 GiB | 254.08 GiB | 245.98 GiB |
| Pair `Private_Dirty` | 387.17 GiB | 191.38 GiB | 183.25 GiB |
| Warm second-engine startup | 176.18 s mean, n=3 | — | 150.45 s mean, n=3 |
| Concurrent pair wall | capacity probe only | 34.61 ± 4.50 s | 28.86 ± 0.04 s |
| Profile CPU `aten::copy_` | not completed | 33.91 s | 14.10 s |
| Profile H2D payload | not completed | 171.943 GiB | 171.943 GiB |
| Profile NVLink Tx / Rx | not completed | 263.058 / 263.058 GiB | 263.058 / 263.058 GiB |
| Peak HBM per GPU | — | 30,286 MiB | 30,286 MiB |

The two HWR engines mapped exactly two artifacts: seven identical payload
inodes for matching TP-rank-0 workers and seven for matching TP-rank-1 workers,
with no cross-rank aliasing. All 16 steady measured/profile video and audio
outputs matched exactly.

## Capacity-control boundary

The existing-loader pair is capacity evidence rather than a transport latency
control. It reached 388.06 GiB PSS and completed one 803.91-second wave. The
next wave hit the default 600-second async-output watchdog while the parent
engines spent sustained kernel time after GPU work. A shortened retry reached
913.23/921.00 GiB (99.16%) cgroup memory before it was stopped at the user's
request. No n=3 existing-path latency or output-parity claim is made.

## Scope boundary

The optional TP2 `dlo_resident_layers=0` scaling profile was omitted after GPUs
were reallocated and the user requested faster completion. The evidence proves
the recipe-placement transport and sharing behavior; it does not claim a TP2
per-layer scaling curve or 50-step production quality.

## Files

- `tp1-report.md` / `tp1-summary.json` — TP1 startup, sharing, transfer, and
  anti-result evidence.
- `tp2-report.md` / `tp2-summary.json` — final TP2 report and machine-readable
  primary matrix.
- `profile-*.json` — per-engine profile aggregates derived from the retained
  torch traces. Raw traces were approximately 866 MiB per TP rank and are not
  copied into the cookbook.
- `commands.md` — controller commands and actual completion boundary.
- `tp2-test-plan.md` — frozen hypotheses, controls, success gates, and stop
  conditions.
- `manifest.json` — environment plus SHA-256 checksums for this compact
  evidence package.

Profiler durations are summed rank-engine work and may overlap; they are not
wall time. RSS counts shared mappings once per process, so the host-memory
comparison uses aggregate proportional set size (PSS).
