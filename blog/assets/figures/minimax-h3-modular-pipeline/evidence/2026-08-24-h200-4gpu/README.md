# MiniMax-H3 four-GPU E2E decomposition evidence

This directory preserves the reproducible evidence behind Blog 1's measured
startup and complete-response decomposition.

## Scope

- vLLM-Omni source: `072bfc02dd74cb0eb5c2f2a914e5dbbddba43b65`
- model snapshot: `42ed227ee7df40d41602854ae760620d6eb651fe`
- task: FL2VA-only T2VA
- hardware: 4×H200 on the contributor cluster; raw `nvidia-smi` telemetry uses
  the internal `NVIDIA L20X` label retained in `manifest.json`
- topology: Ulysses 4 / Ring 1 / DiT TP1, text-encoder TP4, VAE
  patch-parallel 4 in tile mode
- precision/runtime: BF16, CUDNN attention, regional `torch.compile`, no
  quantization, LoRA/Turbo, caching, step execution, or offload
- canonical request: 1344×768, requested 5.0 s, 24 FPS, 124 aligned frames,
  50 sigma points / 49 denoiser evaluations, seed 1101
- diagnostic request: identical except four sigma points / three evaluations;
  non-canonical and not Turbo

The profiled condition ran one cold canonical request, three warm canonical
requests, one diagnostic warmup, and three warm diagnostic requests. The
unprofiled control ran one cold and three nominally warm canonical requests.

## Results used by the post

| Measurement | Result |
|---|---:|
| Profiled startup, process start → first `/health` 200 | 149.880 s |
| Profiled canonical cold request | 114.686 s client total |
| Profiled canonical warm client total, mean ± sample SD (n=3) | 85.845 ± 0.161 s |
| Prompt/text encoder, warm mean | 0.056 s |
| Denoise, warm mean | 80.616 s |
| Video+audio VAE decode, warm mean | 1.859 s |
| Engine/IPC/output residual, warm mean | 1.956 s |
| CPU MP4 encode+mux, warm mean | 1.326 s |
| HTTP residual, warm mean | 0.033 s |
| Four-point diagnostic warm client total, mean ± sample SD (n=3) | 10.375 ± 0.057 s |
| Peak sampled device memory in the profiled condition | 100,872 MiB |

All 12 requests returned HTTP 200 and passed `ffprobe` checks for H.264 video,
1344×768, 124 frames at 24 FPS, and stereo 32 kHz AAC. The three canonical
warm containers share one SHA-256; the three diagnostic warm containers share
another. Every profiled request's direct spans plus signed residuals close to
client-observed complete-body time, and all four GPUs returned to their
pre-run memory baseline.

## Required caveat

The unprofiled controls were nonstationary: 86.247–104.380 s client total
(mean 94.792 s, sample SD 9.111 s). Comparing medians produces an apparent
−8.48% profiler "overhead," which has the wrong sign and exceeds the protocol's
5% disclosure threshold. The post therefore treats 85.845 s as the **profiled
condition only** and does not claim that profiling speeds the model or that
the profiler's observer effect has been quantified. Raw per-phase utilization
shows lower utilization during the two slow unprofiled samples.

The four-point diagnostic is a schedule-sensitivity control, not a quality
comparison or Turbo result. Its latency must not be described as a speedup over
the canonical 50-point workload.

## Files

- `manifest.json` — frozen software, hardware, topology, workload, and sampling
  contract
- `commands.json` — exact profiled and unprofiled server commands
- `harness.py` — exact external harness run through `gpu run`
- `results.json` — per-request spans, signed residuals, media validation,
  decoded-stream hashes, aggregate statistics, and pass/fail checks
- `profiled/` and `unprofiled/` — raw server logs, health polls, response
  headers, GPU/host samples, and launch manifests

CSV and server-log line endings/trailing terminal padding are normalized to LF
in the committed copies so repository diff checks remain clean; field values,
timestamps, and log text are otherwise unchanged.

Generated MP4 bodies are omitted to keep the repository small. Their container
SHA-256 values, decoded video/audio hashes, and full `ffprobe` output are
preserved in `results.json`.
