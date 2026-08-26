# TP2 HWR validation plan

## Decision topology

Use two independent engines, not one DP2 engine:

```text
Engine A (port 8000): DP1 x TP2, physical GPUs 0,1
Engine B (port 8001): DP1 x TP2, physical GPUs 2,3
```

All four NVIDIA L20X GPUs are in NUMA domain 0 and every pair is connected by
NV18, so `(0,1)` and `(2,3)` are equivalent fixed TP pairs. No router is part
of the experiment; the harness controls the two engines directly.

For a matching TP2 layout, the expected HWR backing is **two artifacts**, not
one artifact and not four:

```text
artifact(tp_rank=0) <- Engine A rank 0, Engine B rank 0
artifact(tp_rank=1) <- Engine A rank 1, Engine B rank 1
```

DP/engine identity is excluded from the semantic key. TP coordinate remains in
the key because the two ranks own different final-layout shards.

## Hypotheses

1. Two independent TP2 engines can acquire the same two rank-matched HWR
   artifacts without rebuilding or privately materializing DiT weights.
2. Rank-matched mmap sharing reduces aggregate host PSS relative to the existing
   HWR-disabled loader path while preserving output and per-GPU HBM usage.
3. Registered mmap removes recurrent mmap-to-pinned-staging CPU copies. It does
   not reduce H2D bytes or add a DLO collective, and it should reduce or avoid
   staging-induced TP-rank skew.
4. A live Engine A can accelerate or at least avoid duplicating work for Engine
   B startup, but this is a measured question rather than an assumed benefit.

## Frozen controls

- Repository SHA: `497c537c6f70e44f376b491bf7b50395cf2cba5d`
- Model: MiniMax-H3 `FL2VA`, BF16
- Parallelism per engine: DP1 x TP2, no DLO AllGather
- Text encoder TP2; VAE tile patch-parallel size 2
- Attention/execution: CUDNN attention, eager
- Primary recipe-shaped request: T2VA, 1344x768, 8.7 seconds, seed 1101,
  fixed recipe prompt, and three sigma points (two actual denoiser evaluations)
- Primary placement: `dlo_resident_layers=20`, matching the two-GPU recipe
- Same NUMA affinity, model path, store path, environment, and readiness test
- Preparation, process-to-ready, warmup, and measured request time reported
  separately; no cache dropping

A secondary `dlo_resident_layers=0` transport diagnostic exercises every DiT
block on every evaluation. It must not be presented as the recipe latency.

## Phases

### 0. Preflight and one feasibility run

1. Verify the frozen source, local venv/model/store filesystems, GPU ownership,
   topology, NUMA affinity, and absence of foreign GPU processes.
2. Start one **TP2 producer cohort** against an empty store in `preferred` mode.
3. Require exactly two READY artifacts, one per TP coordinate.
4. Run one short registered TP2 request before the repeated matrix. Stop on
   startup failure, OOM/Xid, corrupt output, missing artifact, or leaked worker.

Cold publication is preparation and is not mixed into warm startup results.

### 1. Rank-matched sharing and second-engine startup

1. Start registered Engine A on GPUs 0,1 and keep it alive.
2. On GPUs 2,3, launch Engine B three times in registered mode and three times
   with HWR disabled, in balanced order:
   `registered, disabled, disabled, registered, registered, disabled`.
3. Use the same Engine A background for every Engine B startup sample.
4. For the first registered pair, inspect every worker process and prove:
   - both ranks report `LOCAL_HIT`;
   - no producer or ordinary DiT materialization runs;
   - A-rank-0 and B-rank-0 map identical device/inode payloads;
   - A-rank-1 and B-rank-1 map identical device/inode payloads;
   - rank 0 and rank 1 do not alias each other's artifacts.
5. Stagger A and B requests by 1.5 seconds to prove there is no cross-engine
   lockstep requirement. Afterward stop A and make B serve another request to
   prove lease/lifetime independence.

This phase reports three Engine B process-to-ready samples per mode with mean,
standard deviation, and a conservative confidence interval. It does not infer
that an Engine A launch must improve Engine B startup unless the measurements
show it.

### 2. Pair-level host-memory comparison

Capture stable, aggregate process-tree memory after a warm request for:

| Case | HWR backing | DLO transport | Purpose |
| --- | --- | --- | --- |
| Existing path | disabled | selected existing loader path | Compatibility and memory baseline |
| HWR staged | required local hit | bounded two-slot pinned staging | Isolate final-layout sharing |
| HWR registered | required local hit | registered mmap direct H2D | Target path |

For each pair, sample all orchestrators and descendants and report aggregate
PSS, RSS, `Private_Dirty`, `Shared_Clean`, `Pss_File`, and locked/pinned memory.
Also report artifact bytes and page-cache state. PSS, not summed RSS, is the
primary host-memory metric.

### 3. Request and transport A/B

The primary A/B is **HWR staged versus HWR registered**. Run one excluded
warmup and three measured concurrent A+B waves for each mode at the
recipe-shaped workload. Profile one additional wave per mode.

Measure:

- per-engine request latency, paired-wave throughput, mean and uncertainty;
- CPU `aten::copy_`/staging time;
- H2D copy count, payload, and device-copy time per TP rank;
- NCCL kernel residence and NVLink Tx/Rx per request;
- compute-kernel time;
- per-GPU peak HBM;
- exact decoded video/audio hashes.

Registered mmap is successful if it removes recurrent CPU staging without
changing output, H2D payload, compute, or HBM materially. A latency improvement
is claimed only if the repeated measurements support it. Lower NCCL residence
with unchanged NVLink payload is evidence of reduced rank skew, not a new DLO
collective.

### 4. Resident-layer scaling diagnostic

Repeat one profiled staged/registered wave with `dlo_resident_layers=0`. Compare
it with the recipe `resident_layers=20` profile. The expected avoided CPU work
scales with:

```text
streamed DiT layers x actual denoiser evaluations
```

It is therefore a per-offloaded-layer invocation effect that accumulates per
denoise step; it is not a one-time per-request optimization.

## Success gates

- Exactly two validated TP2 artifacts exist and both engines share them by TP
  coordinate.
- Every warm required worker is a local hit with zero producer calls and zero
  ordinary DiT materialization.
- All measured outputs are byte-identical across modes and engines.
- No-AllGather adds no HWR/DLO collective; H2D payload is unchanged between
  staged and registered paths.
- Aggregate HWR-pair PSS is lower than the HWR-disabled pair, with the physical
  sharing supported by inode/mapping evidence.
- Registration/unregistration succeeds on all four workers; Engine B survives
  Engine A teardown; all processes and GPU allocations are released.
- Any claimed startup or request-latency gain is based on at least three
  controlled observations and reported with uncertainty.

## Stop conditions

Stop before repetitions on an output mismatch, incomplete TP-coordinate
coverage, unexpected producer call on a required hit, foreign process on a
reserved GPU, OOM/Xid, worker crash, failed cleanup, or inability to keep the
fixed topology and NUMA controls.
