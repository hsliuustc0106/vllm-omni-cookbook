# TP2 HWR two-independent-engine validation

Frozen main: `497c537c6f70e44f376b491bf7b50395cf2cba5d`.

## Decision

The TP2 use case is validated. Two independent DP1xTP2 engines shared exactly two rank-specific final-layout artifacts, with seven identical payload inodes per matching TP rank and zero cross-rank aliasing. Registered HWR reduced pair PSS by 142.08 GiB (36.6%) versus the existing path and by 8.10 GiB versus HWR staging. It also accelerated a warm second TP2 engine by 25.73 s (14.6%). At recipe placement, registration removed 19.82 s (58.4%) of summed CPU copy work while H2D bytes, GPU-copy time, compute, and NVLink traffic stayed effectively unchanged. The n=3 pair-latency mean improved by 16.6%, but its conservative interval crosses zero because the first staged wave was an outlier, so the latency claim remains directional.

## Topology and workload

- Engine A: DP1xTP2 on GPUs 0,1; Engine B: DP1xTP2 on GPUs 2,3.
- Four NVIDIA L20X GPUs, all NUMA 0 and mutually connected by NV18.
- MiniMax-H3 FL2VA BF16; no DLO AllGather; text-encoder TP2; VAE tile PP2; CUDNN attention; eager mode.
- Recipe-shaped T2VA: 1344x768, 8.7 seconds, fixed recipe prompt/seed 1101, three sigma points (two denoiser evaluations), `dlo_resident_layers=20`.
- One excluded warmup and three concurrent measured waves per HWR mode. The memory/NUMA observer quiesced before timing.

## Artifact and startup

The producer cohort published exactly two artifacts totaling 61.73 GiB: one for TP rank 0 and one for TP rank 1.

| Metric | HWR disabled | HWR registered | Delta |
| --- | ---: | ---: | ---: |
| Second-engine process-to-ready, n=3 | 176.178 +/- 5.837 s | 150.447 +/- 0.811 s | -25.732 s (-14.6%) |
| Conservative 95% interval |  |  | [-40.372, -11.091] s |
| Cold preferred producer init |  | 233.140 s | preparation |
| First warm registered process-to-ready |  | 151.445 s | live Engine A |

Engine B's registered mean was essentially the same as the first warm Engine A; the acceleration comes from the already-published artifacts, not from Engine A being live.

## Concurrent recipe latency

| Mode | Pair wall samples | Mean +/- std | Combined throughput |
| --- | --- | ---: | ---: |
| HWR staged | 39.812, 32.053, 31.969 s | 34.611 +/- 4.504 s | 0.0584 req/s |
| HWR registered | 28.819, 28.871, 28.889 s | 28.859 +/- 0.036 s | 0.0693 req/s |
| Existing path, capacity probe | 803.909 s (n=1) | not a transport statistic | 0.0025 req/s |

Registered - staged: -5.752 s (-16.6%), conservative interval [-16.942, +5.438] s.

## Host memory after warmup

| Mode | Aggregate PSS | Private_Dirty | Pss_File | Pss_Shmem |
| --- | ---: | ---: | ---: | ---: |
| Existing path | 388.06 GiB | 387.17 GiB | 0.92 GiB | 366.10 GiB |
| HWR staged | 254.08 GiB | 191.38 GiB | 62.73 GiB | 170.10 GiB |
| HWR registered | 245.98 GiB | 183.25 GiB | 62.75 GiB | 162.10 GiB |
| Registered - existing | -142.08 GiB | -203.92 GiB | +61.84 GiB | -204.00 GiB |
| Registered - staged | -8.10 GiB | -8.13 GiB | +0.02 GiB | -8.00 GiB |

The shortened capacity retry independently reached 913.23/921.00 GiB (99.16%) cgroup memory, including 525.26 GiB shmem.

## Rank-matched sharing

- Registered TP0: {'engine_a_count': 7, 'engine_b_count': 7, 'identical': True, 'shared_count': 7}.
- Registered TP1: {'engine_a_count': 7, 'engine_b_count': 7, 'identical': True, 'shared_count': 7}.
- Staged TP0: {'engine_a_count': 7, 'engine_b_count': 7, 'identical': True, 'shared_count': 7}.
- Staged TP1: {'engine_a_count': 7, 'engine_b_count': 7, 'identical': True, 'shared_count': 7}.
- Cross-rank shared inode count: registered=0, staged=0.

## Registered versus staged profile

Values are means of two sequential per-engine profile requests. Durations are summed rank-engine work and can overlap.

| Metric | HWR staged | HWR registered | Delta |
| --- | ---: | ---: | ---: |
| CPU `aten::copy_` | 33.914 s | 14.096 s | -19.818 s |
| H2D payload | 171.943 GiB | 171.943 GiB | +0.000000 GiB |
| H2D device time | 3.356 s | 3.366 s | +0.010 s |
| H2D operations | 4361 | 5921 | +1560 |
| Compute kernels | 39.539 s | 39.636 s | +0.097 s |
| NCCL kernels | 1.252 s | 1.192 s | -0.059 s |
| NVLink Tx / Rx | 263.058 / 263.058 GiB | 263.058 / 263.058 GiB | +0.000002 / +0.000001 GiB |

Registration changes host transport, not communication volume: H2D bytes and NVLink counters are unchanged. The extra 1,560 H2D operations are tensor-level fragmentation in the registered path, while device-copy time remains flat.

## Correctness and lifecycle

- Steady measured/profile outputs: 16; unique complete video/audio digests: 1; exact registered/staged match: True.
- Output contract: video [209, 768, 1344, 3] float32; audio [1, 2, 278400] float32.
- Excluded warmups match each other: True; warmup matches steady output: False.
- Every successful engine used `dlo_use_allgather=False`. Ordinary TP/text-encoder collectives remain and explain the unchanged NVLink traffic; they are not DLO weight reconstruction.
- Pair return codes zero, no active children, monitor/NUMA/quiesce gates passed, and Engine B survived Engine A teardown: True.
- Task GPU processes after cleanup: [].

## Capacity-control outcome

The existing-path pair is not part of the transport latency statistic. It used 388.06 GiB PSS, completed one 803.91-second wave, and then hit the default 600-second async-output watchdog on the next wave while parent engines spent sustained kernel time after GPU work. This is a capacity-pressure result caused by the extra private host weights, not evidence of slower GPU kernels.

## Scope boundary

The optional TP2 `resident_layers=0` scaling profile was not run after GPUs were reallocated and the user requested faster completion. Therefore this report establishes the recipe-placement transport benefit but does not claim a measured TP2 per-layer scaling curve.

## Caveats

- Three waves give a directional latency comparison; the conservative interval crosses zero because the first staged wave was an outlier.
- The existing path is represented by a memory snapshot and one capacity-pressure wave, not a completed n=3 throughput matrix or output-hash parity row.
- The excluded first request produced a different audio hash than steady requests while video stayed identical; this happened consistently in both HWR modes. Steady outputs are the parity gate.
- Profile traces were approximately 866 MiB per TP rank and took minutes to export; export time is outside request timing.
- CUDA registration is not reflected reliably by smaps `Locked`; registration logs, inode mappings, PSS, and successful unregistration are the evidence.
- Two denoiser evaluations bound experiment time; this is not a 50-step quality benchmark.

## Artifacts

- `tp2_summary.json`
- `runs/tp2/experiment_result.json`
- `runs/tp2/artifact_inventory.json`
- `runs/tp2/startup-balanced/startup_result.json`
- `runs/tp2/pair-registered-r20/` and its monitor directory
- `runs/tp2/pair-staged-r20/` and its monitor directory
- `runs/tp2/pair-disabled-r20/` preserved capacity attempt
- Per-engine traces and `case_analysis.json` files under each successful pair.
