# TP1 HWR independent-engine validation

Frozen main: `497c537c6f70e44f376b491bf7b50395cf2cba5d`.

## Decision

At this frozen main revision, TP1 HWR is functionally correct and shares one 61.73 GiB final-layout artifact between independent engines, but it is not a startup accelerator against the existing TP1 checkpoint-mmap path. Warm required HWR process-to-ready time was +96.96 s (+90.9%) slower. For this maximally streamed 256x256 transport stress case, registered HWR observed 49.7% lower measured request latency and 7.94 GiB lower pair PSS while preserving bytes, output, compute time, and HBM. TP1 already shares raw checkpoint pages, so the unique memory saving here is mainly the removed bounded staging allocation; the TP2 matrix is the decisive final-layout-sharing test.

## Workload

- Two independent DP1xTP1 no-AllGather MiniMax-H3 FL2VA engines on separate GPUs.
- Fresh preferred-mode publication, then required-mode warm consumers.
- Fixed 256x256, four-second requests; startup/sharing smoke rather than production-quality throughput.
- Engine A and the first registered Engine B were released 1.5 seconds apart with different sigma-point counts.

## Startup

| Metric | Checkpoint mmap | Warm HWR registered | Delta |
| --- | ---: | ---: | ---: |
| Second-engine process-to-ready, n=3 | 106.659 +/- 6.250 s | 203.623 +/- 7.298 s | +96.964 s (+90.9%) |
| Conservative 95% interval |  |  | [+73.094, +120.834] s |
| Cold preferred publication |  | 327.167 s | preparation only |
| First warm Engine A process-to-ready |  | 217.886 s | kept alive |
| Warm HWR staged process-to-ready |  | 189.798 s | registration disabled by budget |

`process-to-ready` includes Python/import overhead. The corresponding AsyncOmni engine-init means were 96.430 s for checkpoint mmap and 193.274 s for registered HWR.

## Warm request latency

Each fresh engine ran one excluded warmup followed by one measured request. The six second-engine starts were balanced as registered/checkpoint/checkpoint/registered/registered/checkpoint.

| Metric | Checkpoint mmap | HWR registered | Delta |
| --- | ---: | ---: | ---: |
| Measured request, n=3 | 8.554 +/- 1.172 s | 4.306 +/- 0.279 s | -4.248 s (-49.7%) |
| Conservative 95% interval |  |  | [-7.241, -1.256] s |
| HWR staged diagnostic, n=1 |  | 4.926 s | attribution only |

## Live sharing and host memory

| Snapshot | PSS | Private_Dirty | Pss_File | Locked (smaps) |
| --- | ---: | ---: | ---: | ---: |
| Two warm registered HWR engines | 195.57 GiB | 133.01 GiB | 62.57 GiB | 0.00 GiB |
| Two checkpoint-mmap engines | 203.51 GiB | 140.98 GiB | 62.54 GiB | 0.00 GiB |
| Registered - checkpoint | -7.94 GiB | -7.97 GiB | +0.03 GiB | +0.00 GiB |

- Ready HWR artifacts: 1.
- Shared HWR payload inodes across Engine A/B: 13.
- The HWR-disabled pair selected `checkpoint_mmap`, realized 535 tensors as mmap views, and allocated two bounded pinned staging buffers per engine.
- Direct checkpoint inode capture is unavailable because the snapshot symlinks resolve to extensionless Hugging Face blob paths; aggregate `Pss_File` of 62.54 GiB is consistent with one shared checkpoint working set.

## Transfer diagnostic

| Metric | Checkpoint mmap | HWR staged | HWR registered |
| --- | ---: | ---: | ---: |
| CPU aten::copy_ | 3.657 s | 2.836 s | 0.933 s |
| H2D payload | 185.335 GiB | 185.335 GiB | 185.335 GiB |
| H2D device time | 3.746 s | 4.266 s | 3.614 s |
| H2D operations | 1877 | 1877 | 2837 |
| Compute kernels | 0.554 s | 0.553 s | 0.554 s |

Registered HWR removed 1.903 s (67.1%) of profiled CPU copy work versus HWR staging. It retained the same H2D payload but emitted 960 additional H2D operations; compute time differed by less than 0.4 ms.

## Correctness and lifecycle

- All registered/checkpoint/staged comparison outputs match: True.
- Maximum engine-reported HBM across measured engines: 15088 MiB.
- All summaries report no active children: True.
- Output contract: video [107, 256, 256, 3] float32; audio [1, 2, 142400] float32.
- Every registered consumer skipped ordinary DiT materialization, consumed one HWR lease, unregistered all ranges, and released the lease. The staged consumer released its lease; every checkpoint consumer released its mmap handles.
- Monitor: return code 0, timed_out=False, 1009 samples.

## Caveats

- TP1 already supports direct checkpoint mmap, so this is the least favorable case for a unique HWR startup advantage.
- Three startup samples per mode support only a directional claim; the conservative interval is reported.
- The TP1 monitor continuously sampled process-tree smaps during requests. The large latency delta is consistent with the profile, but the confidence interval covers run variation, not systematic observer overhead; TP2 quiesces monitoring before timed waves.
- CUDA-registered page accounting may not appear in smaps Locked/VmLck; inode sharing and aggregate PSS are reported separately.
- The torch profiler emitted the same non-fatal CUPTI external-callback thread warning once in each profiled mode; all traces were produced and all commands returned zero.
- The baseline checkpoint inode counter is not evidence of zero sharing: its path filter missed extensionless Hugging Face blob targets, as described above.
- Cold publication, engine startup, request execution, and shutdown are measured separately.

## Artifacts

- `tp1_summary.json`
- `runs/tp1/experiment_result.json`
- `runs/tp1/events.jsonl`
- `runs/tp1-monitor/samples.jsonl`
- Per-engine summaries, logs, profiles, and case analyses under `runs/tp1/`.
