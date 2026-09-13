---
layout: post
title: "Understanding PR #6591 — Host Weight Runtime: one TP2 layout, many independent engines"
date: 2026-08-26 12:00:00 +0800
author: hsliuustc0106
summary: >-
  Host Weight Runtime lets independent TP2 DLO engines share two final-layout
  host artifacts. On 4×H200 it cut pair PSS 388.06→245.98 GiB and removed
  58.4% of profiled CPU copy work.
tags: [MiniMax-H3, H200, DLO]
category: PR Analysis
feature: offloader
lang: en
pair: /zh/2026-08-26-understanding-pr-6591-host-weight-runtime/
usage:
  - label: "Populate · preferred"
    blurb: "first matching TP2 cohort"
    title: "MiniMax-H3 · create both TP-coordinate artifacts"
    code: |
      export MODEL=/path/to/MiniMax-H3/FL2VA
      export HWR_ROOT=/var/cache/vllm-omni/hwr

      CUDA_VISIBLE_DEVICES=0,1 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      vllm serve "${MODEL}" --omni \
        --host 0.0.0.0 --port 8000 --trust-remote-code \
        --task-type fl2va \
        --num-gpus 2 --tensor-parallel-size 2 \
        --usp 1 --ring 1 \
        --text-encoder-tp-size 2 \
        --vae-patch-parallel-size 2 \
        --vae-parallel-mode tile --vae-use-tiling \
        --enable-distributed-layerwise-offload \
        --dlo-no-use-allgather --dlo-resident-layers 20 \
        --host-weight-runtime-mode preferred \
        --host-weight-runtime-root "${HWR_ROOT}" \
        --dlo-host-registration-limit-gib 80 \
        --enforce-eager \
        --diffusion-attention-backend CUDNN_ATTN
    note: >-
      Wait for a healthy startup. Publication happens during startup, so no
      inference request is required. A strict prewarm workflow then shuts this
      cohort down cleanly before switching to required mode.
  - label: "Share · second engine"
    blurb: "same root, independent requests"
    title: "MiniMax-H3 · Engine B consumes the same TP0/TP1 artifacts"
    code: |
      export MODEL=/path/to/MiniMax-H3/FL2VA
      export HWR_ROOT=/var/cache/vllm-omni/hwr

      CUDA_VISIBLE_DEVICES=2,3 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      vllm serve "${MODEL}" --omni \
        --host 0.0.0.0 --port 8001 --trust-remote-code \
        --task-type fl2va \
        --num-gpus 2 --tensor-parallel-size 2 \
        --usp 1 --ring 1 \
        --text-encoder-tp-size 2 \
        --vae-patch-parallel-size 2 \
        --vae-parallel-mode tile --vae-use-tiling \
        --enable-distributed-layerwise-offload \
        --dlo-no-use-allgather --dlo-resident-layers 20 \
        --host-weight-runtime-mode preferred \
        --host-weight-runtime-root "${HWR_ROOT}" \
        --dlo-host-registration-limit-gib 80 \
        --enforce-eager \
        --diffusion-attention-backend CUDNN_ATTN
    note: >-
      The engines share files and OS page-cache pages, not a process group.
      Route requests externally if desired; HWR does not provide a router.
  - label: "Enforce · required"
    blurb: "fail instead of rebuilding"
    title: "MiniMax-H3 · require a pre-populated exact identity"
    code: |
      # Use the same model revision, TP/SP layout, root, and serving flags as
      # the preferred producer cohort; change only the policy.
      vllm serve /path/to/MiniMax-H3/FL2VA --omni \
        --num-gpus 2 --tensor-parallel-size 2 \
        --text-encoder-tp-size 2 \
        --vae-patch-parallel-size 2 \
        --vae-parallel-mode tile --vae-use-tiling \
        --enable-distributed-layerwise-offload \
        --dlo-no-use-allgather --dlo-resident-layers 20 \
        --host-weight-runtime-mode required \
        --host-weight-runtime-root /var/cache/vllm-omni/hwr \
        --dlo-host-registration-limit-gib 80
    note: >-
      required is consume-only. An empty, corrupt, or semantically mismatched
      store fails startup instead of running the canonical loader.
decisions:
  - when: "several independent TP2 engines share one node"
    pick: "no-AllGather + HWR preferred"
    why: "Matching TP coordinates map the same final-layout artifacts without joining one cross-engine scheduling or failure domain."
  - when: "one synchronized DP×TP job stays in a fast P2P domain"
    pick: "consider DLO AllGather"
    why: "Ranks already execute one collective sequence, so sharded persistent host weights may be simpler than a final-layout cache."
  - when: "TP1 checkpoint mmap already works"
    pick: "start with HWR disabled"
    why: "The TP1 control started 96.96 s faster; HWR is not automatically a better checkpoint loader."
  - when: "the rollout must prove every cache hit"
    pick: "populate preferred, then use required"
    why: "required never bootstraps an empty store and exposes stale or incompatible identities at startup."
  - when: "registration is unsupported or over budget"
    pick: "keep HWR and accept bounded staging"
    why: "File-backed sharing remains valid; only the recurrent mmap-to-pinned CPU copy returns."
  - when: "online quantization, HSDP, or dynamic LoRA changes weights"
    pick: "use the current canonical path"
    why: "The shipped consumer is exact final-layout BF16 for eligible MiniMax-H3 no-AllGather DLO, not a generic transformed-weight cache."
---

## TL;DR {#tldr}

**Host Weight Runtime (HWR) lets independent engines reuse one exact set of
runtime-ready CPU weights—like two kitchens drawing ingredients from the same
sealed pantry instead of stocking two private storerooms.** For two independent
MiniMax-H3 `DP1×TP2` engines, the pantry contains two artifacts, one per tensor-
parallel (TP) coordinate; it does not couple their request schedules.

On four H200 GPUs, registered HWR reduced aggregate pair proportional set size
(PSS) from **388.06 to 245.98 GiB**, made the second TP2 engine ready **25.73 s
faster**, and removed **58.4%** of profiled CPU copy work compared with HWR's
bounded-staging fallback. H2D payload, HBM, compute, and NVLink traffic stayed
effectively unchanged.

| Decision metric | Control | Registered HWR | Result |
|---|---:|---:|---:|
| Pair PSS | 388.06 GiB existing path | 245.98 GiB | **−142.08 GiB (−36.6%)** |
| Warm second-engine startup, n=3 | 176.18 s HWR-disabled | 150.45 s | **−25.73 s (−14.6%)** |
| Pair wall, n=3 | 34.61 s HWR staged | 28.86 s | −16.6% mean; directional |
| Profile CPU `aten::copy_` | 33.91 s staged | 14.10 s | **−19.82 s (−58.4%)** |
| H2D / NVLink / peak HBM | reference | unchanged | transport bytes did not move |

> [!IMPORTANT]
> HWR is opt-in and workload-specific. TP1 direct checkpoint mmap was the
> better startup path in the companion test, and the TP2 latency confidence
> interval crosses zero because the first staged wave was an outlier. The
> strong claims are exact sharing, startup, PSS, CPU-copy attribution, and
> output parity—not universal request acceleration.

## Background {#background}

**The visible problem is host memory growing with every independent engine—like
renting a new warehouse for each delivery van even when every warehouse holds
the same boxes.** Device memory may fit because DLO streams blocks through two
rotating HBM buffers, while host memory still fails because every worker keeps
private runtime weights.

[PR #6213](https://github.com/vllm-project/vllm-omni/pull/6213) fixed the first
version of this problem for compatible TP1 checkpoints. The loader can retain
read-only checkpoint `mmap` views, so workers share physical file pages through
the operating system's page cache. That is deliberately a **checkpoint-layout**
path: model-specific transforms can be deferred until each block is packed.

TP2 changes the problem. The ordinary loader slices and transforms tensors into
rank-local final layouts. TP rank 0 and TP rank 1 no longer own identical bytes,
and neither final layout is necessarily a direct view of the raw checkpoint.
Before HWR, two independent TP2 engines therefore looked like this:

```text
Engine A: private TP0 runtime weights + private TP1 runtime weights
Engine B: private TP0 runtime weights + private TP1 runtime weights
```

Why not simply put the two engines in a DLO AllGather group? Because AllGather
requires its members to issue the same weight collective in the same logical
order. Two engines accepting unrelated requests may be on different denoising
steps, blocks, or idle periods. NVLink makes a collective fast **after every
participant issues the matching call**; it does not make independent schedules
compatible. Replica routing remains the job of vLLM Router, Dynamo, or an
operator—not the offloader.

The [HWR RFC #6414](https://github.com/vllm-project/vllm-omni/issues/6414)
therefore separates three questions:

1. Which exact runtime representation did the loader request?
2. Where can that representation be obtained safely?
3. How should DLO move it to a device?

## What Host Weight Runtime is {#mental-model}

**HWR is a versioned library and warehouse contract, not a GPU copy engine—like
a catalog that guarantees the right edition while a separate courier chooses
the truck.** The loader defines semantic identity; the store publishes and
validates immutable files; a lease owns process-local mappings; and DLO owns
registration, staging, H2D streams, and teardown.

![Two independent TP2 engines map one HWR artifact per matching TP coordinate and keep independent request schedules]({{ site.baseurl }}/assets/figures/host-weight-runtime/fig1-architecture.svg)

The central contracts landed in
[PR #6419](https://github.com/vllm-project/vllm-omni/pull/6419):

- `HostWeightStore` owns exact lookup, one-builder coordination, hashing,
  validation, atomic publication, quarantine, and file lifecycle.
- `HostWeightLease` owns stable tensor views, mapped ranges, file descriptors,
  and the shared artifact lock for one consumer process.
- `WeightProducer` creates one declared final representation through a store-
  scoped writer.
- `WeightRestorer` validates a mutation-free plan and commits model rebinding
  exactly once.

Artifact identity includes immutable source fingerprint, component ownership,
representation, mixed-precision policy, producer/restorer ABI, and relevant
parallel layout. DP rank is excluded when bytes are replicated. TP size and TP
rank are included when they change tensor bytes; SP layout is guarded when it
changes semantics. Registration policy and device IDs are excluded because
they move the same representation.

For the measured TP2 deployment, HWR published exactly two 30.86-GiB artifacts:

```text
artifact(tp_rank=0) ← Engine A rank 0 + Engine B rank 0
artifact(tp_rank=1) ← Engine A rank 1 + Engine B rank 1
```

Every process has its own virtual mapping and lease. The shared resource is the
kernel's physical file-backed pages—not one Python tensor object or one CUDA
registration shared across processes.

## What landed across the PR series {#key-changes}

**The feature arrived as several small contracts instead of one giant cache
patch—like certifying the warehouse, catalog, assembly line, and delivery truck
separately.** This split keeps storage semantics reusable and prevents CUDA or
DLO from becoming the owner of model identity.

| PR | Responsibility | What it deliberately does not own |
|---|---|---|
| [#6419](https://github.com/vllm-project/vllm-omni/pull/6419) | Neutral runtime, store, lease, exact identity, atomic local filesystem lifecycle | Diffusion, BF16 policy, CUDA, H2D |
| [#6427](https://github.com/vllm-project/vllm-omni/pull/6427) | Explicit post-load publication for producers that need a finalized canonical model | Restoring or mutating the model serving the current cold startup |
| [#6445](https://github.com/vllm-project/vllm-omni/pull/6445) | MiniMax-H3 final-layout BF16 producer/restorer and TP/SP semantic identity | Loader activation and transport |
| [#6486](https://github.com/vllm-project/vllm-omni/pull/6486) | `disabled`/`preferred`/`required`, warm restore, ordinary-DiT skip, transactional lease handoff, bounded staging | CUDA registration |
| [#6591](https://github.com/vllm-project/vllm-omni/pull/6591) | Registered shared mmap → rotating HBM buffers, rollback, ordered teardown, writer/fsync/source-digest optimizations | AllGather, checkpoint-mmap registration, request orchestration |

The loader gates HWR before doing identity or filesystem work when DLO is
disabled, HWR is disabled, or AllGather is enabled:

```python
# host_weight_loader.py — zero-interaction precedence
if mode is RuntimeMode.DISABLED or not dist_offload or use_allgather:
    return None
```

See the pinned
[`host_weight_loader.py`](https://github.com/vllm-project/vllm-omni/blob/497c537c6f70e44f376b491bf7b50395cf2cba5d/vllm_omni/diffusion/model_loader/host_weight_loader.py#L103-L151)
implementation. An eligible warm hit plans and commits exact restoration; a
`preferred` miss runs the canonical loader and publishes for the next startup;
a `required` miss fails.

The transport decision comes later:

```python
# distributed_layerwise_backend.py — same lease, two safe transfer outcomes
self._using_registered_mmap = self._try_register_hwr_mmap(source_tensors)
hook.registered_mmap = self._using_registered_mmap
```

The complete registration and unregister-before-lease-close path is in
[`distributed_layerwise_backend.py`](https://github.com/vllm-project/vllm-omni/blob/497c537c6f70e44f376b491bf7b50395cf2cba5d/vllm_omni/diffusion/offloader/distributed_layerwise_backend.py#L1039-L1150).

## How the data moves {#dataflow}

**Registered mmap removes one repeated host-side relay—like loading a truck
directly from the shared dock instead of moving every pallet through a private
staging room first.** The truck still carries the same bytes to the same HBM
buffer, and the GPU kernel still reads HBM rather than host memory.

The bounded fallback is:

```text
shared read-only final-layout mmap
  → CPU copy / pack into two private pinned staging slots
  → H2D into two rotating HBM block buffers
  → GPU kernel
```

The registered path is:

```text
shared read-only final-layout mmap registered with CUDA
  → H2D into the same two rotating HBM block buffers
  → GPU kernel
```

This is **not** zero-copy GPU execution. Registration page-locks the complete
mapped range for the worker lifetime and lets asynchronous H2D use those pages
directly. It does not eliminate H2D payload, replace the HBM buffers, or remove
ordinary TP collectives.

Registration is all-or-nothing. Unsupported read-only registration, a positive
budget smaller than the full page-aligned mapping, or a safely rolled-back
registration error selects bounded staging. If rollback or unregistration
cannot safely release platform ownership, startup or teardown fails instead of
closing a mapping still owned by CUDA.

## Measured setup {#setup}

**The experiment keeps every meaningful knob fixed—like comparing two delivery
routes with the same cargo, drivers, roads, and destination.** The isolated
variables were host backing and whether the mapped HWR pages were registered.

| Field | Frozen value |
|---|---|
| vLLM / vLLM-Omni | vLLM 0.27.0; source `497c537c`; installed distribution metadata `0.27.0rc2.dev60+ge2721dc97` |
| Model | MiniMax-H3 FL2VA snapshot `42ed227e` |
| Hardware | 4×H200, one NUMA-0 domain, NV18 between every selected pair |
| Internal GPU label | `NVIDIA L20X`; this cluster label denotes the cookbook's H200 SKU |
| Engines | A on GPUs 0–1, B on GPUs 2–3; each `DP1×TP2` |
| DLO | no-AllGather, 20 resident main-DiT blocks, two rotating device buffers |
| Other parallelism | text encoder TP2; VAE tile patch parallel 2; SP1; PP1; CFG1 |
| Workload | T2VA, 1344×768, 8.7 s, seed 1101, three sigma points / two denoiser evaluations |
| Timing | one excluded warmup; three concurrent pair waves; observer quiesced before timing |

The complete compact evidence package—reports, summaries, profile aggregates,
commands, plan, and SHA manifest—is
[`2026-08-26-h200-tp-validation`](https://github.com/hsliuustc0106/vllm-omni-cookbook/tree/main/blog/assets/figures/host-weight-runtime/evidence/2026-08-26-h200-tp-validation).
Raw torch traces were about 866 MiB per TP rank and are intentionally not copied
into the cookbook.

## Measured impact {#measured-impact}

**HWR's clearest win is fitting and starting the second engine without changing
GPU work—like sharing the warehouse while every van still drives its original
route.** Registered HWR retained one physical final-layout working set, removed
private staging slots, and preserved exact output.

![TP2 Host Weight Runtime results: host PSS, second-engine startup, pair latency, and profile attribution]({{ site.baseurl }}/assets/figures/host-weight-runtime/fig2-tp2-results.svg)

### Rank-matched sharing and memory

| Mode | Pair PSS | `Private_Dirty` | `Pss_File` | `Pss_Shmem` |
|---|---:|---:|---:|---:|
| Existing host tensors | 388.06 GiB | 387.17 GiB | 0.92 GiB | 366.10 GiB |
| HWR + bounded staging | 254.08 GiB | 191.38 GiB | 62.73 GiB | 170.10 GiB |
| HWR + registered mmap | **245.98 GiB** | **183.25 GiB** | 62.75 GiB | **162.10 GiB** |

Both HWR modes mapped seven identical payload inodes between the two TP-rank-0
workers and seven between the two TP-rank-1 workers. Cross-rank shared inode
count was zero. Registered mmap saved another 8.10 GiB PSS versus staging by
omitting four workers' private two-slot staging allocation.

### Warm startup

| Second-engine process-to-ready | Samples | Mean ± std | Registered delta |
|---|---|---:|---:|
| HWR disabled | 172.18, 182.88, 173.48 s | 176.18 ± 5.84 s | — |
| HWR registered | 149.56, 150.64, 151.14 s | **150.45 ± 0.81 s** | **−25.73 s (−14.6%)** |

The conservative 95% interval for registered minus disabled was
`[−40.37, −11.09] s`. The first warm registered Engine A took 151.45 s, nearly
the same as Engine B's 150.45 s mean: the artifact accelerates both consumers;
merely keeping Engine A alive does not make Engine B faster.

### Concurrent request waves

| HWR transport | Pair-wall samples | Mean ± std | Combined throughput |
|---|---|---:|---:|
| Bounded staging | 39.812, 32.053, 31.969 s | 34.611 ± 4.504 s | 0.0584 req/s |
| Registered mmap | 28.819, 28.871, 28.889 s | **28.859 ± 0.036 s** | **0.0693 req/s** |

The registered mean was 16.6% lower, but the deliberately conservative interval
was `[−16.94, +5.44] s`; it crosses zero because the first staged wave was an
outlier. Treat request latency as directional. The profile attribution is much
more stable:

| Per TP2 engine profile | HWR staged | HWR registered | Difference |
|---|---:|---:|---:|
| CPU `aten::copy_` rank-work | 33.914 s | **14.096 s** | **−19.818 s (−58.4%)** |
| H2D payload | 171.943 GiB | 171.943 GiB | unchanged |
| H2D device time | 3.356 s | 3.366 s | +0.010 s |
| H2D operations | 4,361 | 5,921 | +1,560 tensor-level copies |
| Compute-kernel rank-work | 39.539 s | 39.636 s | +0.097 s |
| NCCL-kernel rank-work | 1.252 s | 1.192 s | −0.059 s |
| NVLink Tx / Rx | 263.058 / 263.058 GiB | 263.058 / 263.058 GiB | unchanged |
| Peak HBM per GPU | 30,286 MiB | 30,286 MiB | unchanged |

Registration removes recurrent host copying; it does not reduce model bytes.
The 1,560 additional operations are tensor-level direct H2D fragmentation, yet
aggregate device-copy time stayed flat.

### Correctness and capacity boundary

All 16 steady registered/staged measured and profiled outputs had one exact
video/audio digest: video `[209, 768, 1344, 3]` and stereo audio
`[1, 2, 278400]`, both `float32`. The excluded first warmups matched each other
across modes but had a different audio hash from steady requests; video stayed
identical. Steady output is therefore the parity gate.

The existing-path pair is **capacity evidence, not a latency control**. It
reached 388.06 GiB PSS, completed one 803.91-s pair wave, and then hit the
default 600-s async-output watchdog on the next wave while parent processes
spent sustained kernel time after GPU work. A shortened retry reached
913.23/921.00 GiB—99.16% of the container memory limit. No n=3 existing-path
latency or output-parity claim is made.

## The TP1 anti-result {#tp1}

**A shared artifact is not automatically the fastest path—like repackaging a
book for a library even when two readers can already open the publisher's file
directly.** Compatible TP1 MiniMax-H3 already has PR #6213's checkpoint-mmap
path, so final-layout HWR adds identity, validation, and restoration work to a
case that already shares raw pages.

| TP1 metric | Checkpoint mmap | Registered HWR | Result |
|---|---:|---:|---:|
| Second-engine process-to-ready, n=3 | **106.66 s** | 203.62 s | HWR **+96.96 s (+90.9%)** |
| Two-engine PSS | 203.51 GiB | **195.57 GiB** | HWR −7.94 GiB |
| Shared payload | raw checkpoint pages | one 61.73-GiB final artifact | both share file pages |

Registered transport still removed recurrent CPU staging in the TP1 transport
stress test, but HWR was not a startup optimization. For TP1, start with HWR
disabled unless the exact final-layout representation provides another concrete
benefit.

## Pros and costs {#tradeoffs}

**HWR exchanges repeated private work for one durable shared representation—like
buying a communal freezer that saves groceries but needs floor space, power,
and inventory discipline.** The right choice depends on whether those fixed
costs amortize across enough equivalent workers and startups.

| Pros | Costs and risks |
|---|---|
| One immutable page-cache working set for equivalent workers | Cold canonical load plus artifact writing and validation |
| Runtime-ready TP-coordinate artifacts, not raw checkpoint guesses | Roughly one final-layout model copy on local disk per identity |
| Faster warm TP2 startup in the measured topology | A second, versioned model reconstruction path to validate and maintain |
| Registered mmap removes private staging and recurrent CPU copies | Full mapped range is registered/page-locked for each worker lifetime |
| Exact identity prevents TP/SP/revision/ABI aliasing | Layout, model revision, or producer ABI changes create a new identity |
| `preferred` retains canonical fallback for typed recoverable failures | `required` intentionally fails on a miss or invalid artifact |
| H2D, HBM, and TP communication remain unchanged | No automatic eviction or cross-node coherence in V1 |

The store must be on a verified node-local filesystem. Remote filesystems are
rejected because their page-cache and advisory-lock semantics do not satisfy
the local contract. `tmpfs` is legal but consumes host memory and may consume
swap, undermining the disk-backed memory-sharing story.

## How to use it {#how-to-use}

**Enable HWR in two steps: populate the pantry with the exact topology, then let
matching workers acquire leases from the same node-local root.** Use `preferred`
for normal deployment and `required` only after the store is known to contain
the exact identity.

{% include usage-cookbook.html modes=page.usage %}

The three modes have intentionally different availability behavior:

| Mode | Exact local hit | Miss / recoverable store problem |
|---|---|---|
| `disabled` | HWR is not constructed or probed | existing checkpoint-mmap or ordinary-loader path |
| `preferred` | restore exact final-layout lease | canonical load; serve current model; attempt publication for future starts |
| `required` | restore exact final-layout lease | fail startup |

Use the same immutable model revision, dtype, TP/SP layout, and root for producer
and consumers. DP rank and DP size do not enter replicated artifact identity,
so equivalent deployment replicas can share. TP2 population needs one matching
TP2 cohort so both TP-coordinate artifacts are created. Repeat population for
each node or storage domain; V1 is node-local.

## Registration policy and verification {#registration-policy}

**Registration is a fast loading dock with a capacity permit—if the permit is
missing or too small, the warehouse remains shared but traffic uses the staging
room.** Eligible warm HWR hits automatically attempt registration under the
existing pinned-memory policy.

`--dlo-host-registration-limit-gib` is a **per-worker ceiling over the complete
page-aligned mapping**. Zero means no additional ceiling; it does not disable
registration. The measured TP2 workers each mapped 30.86 GiB, and the benchmark
used an 80-GiB ceiling. A positive limit smaller than the complete mapping
selects bounded staging before any partial registration.

Verify the warm registered path in logs:

```text
DLO host-weight plan active (rank-local, host_weight_runtime):
  skipping ordinary materialization for ['transformer.']
DLO consuming final-layout Host Weight Runtime lease ...
Registered 30.86 GiB of HWR mmap in 7 range(s) for direct H2D ...
...
Unregistered HWR mmap ranges
Released Host Weight Runtime lease ...
```

If logs say `using bounded host staging`, sharing still works; only the direct
transport optimization is absent. On shutdown, pending H2D work drains,
registration is released, model/hook references are removed, and the lease is
closed last.

Measure aggregate process-tree **PSS**, not summed RSS. RSS charges the same
physical file page to every mapping process; PSS divides that page among them.

## How to choose {#how-to-choose}

**Choose by scheduling domain first and storage second—like deciding whether
trains share one timetable before choosing a warehouse.** Independent engines
need shared files without shared collectives; synchronized ranks can often use
AllGather directly.

{% include decision-cards.html items=page.decisions %}

Fast NVLink or NVSwitch does not erase this distinction. In the measured H200
topology, staged and registered HWR both generated 263.058 GiB Tx and Rx per
profiled engine because ordinary TP communication was unchanged. HWR saved host
work while preserving the engine's existing TP group.

## Operational checklist {#operations}

**Operate HWR like a local artifact store, not a temporary Python cache—give it
a stable address, enough disk, and a lifecycle plan.** A healthy deployment
should make identity, capacity, and fallback visible before serving traffic.

1. Put `host_weight_runtime_root` on persistent node-local disk visible to all
   workers in the storage domain; do not use NFS, Ceph, or a process-private
   temporary directory.
2. Budget artifact disk separately from host PSS. MiniMax-H3 TP2 produced two
   filesets totaling 61.73 GiB; a different revision or layout creates another
   identity.
3. Populate with the exact serving TP/SP cohort. `required` cannot bootstrap an
   empty store.
4. Budget registration per worker. Full registration trades private staging
   for long-lived page locking; a smaller ceiling safely falls back.
5. Watch cgroup memory, not only host-wide `free`. The existing-path capacity
   probe reached 99.16% of its container limit even though the host had more
   physical RAM.
6. Re-run `preferred` after changing model revision, producer/restorer ABI,
   dtype, or semantic parallel layout.
7. Stop consumers before explicit store cleanup. V1 has capacity checks and
   quarantine but no automatic eviction policy or public prewarm/cleanup CLI.

A router may place requests across Engine A and Engine B, but it is not required
for host-page sharing. Conversely, pointing two engines at the same root does
not create routing, health coordination, or one shared failure domain.

## Limitations and follow-ups {#limitations}

**The shipped path is a narrow, exact bridge rather than a universal cache—like
opening one certified lane before paving every route.** Unsupported semantics
fail or fall back instead of silently reusing the wrong bytes.

- The active model consumer is final-layout BF16-with-preserved-FP32 for
  eligible MiniMax-H3 no-AllGather DLO.
- HWR final-layout lookup is not selected for DLO AllGather, DLO-disabled, or
  HWR-disabled configurations.
- Online quantization needs a representation-specific producer for generated
  scales and physical layout. Dynamic LoRA remains an overlay; static merged
  adapters require a distinct identity and explicit support.
- HSDP and non-default load formats are ineligible in the current consumer.
- The store is node-local. Remote providers, cross-node coordination,
  automatic eviction, and enforceable producer cancellation are future work.
- The optional TP2 `dlo_resident_layers=0` scaling profile was not completed
  after GPUs were reallocated. This post proves recipe-placement sharing and
  transport behavior but makes no measured TP2 per-layer scaling curve.
- The existing-path capacity attempt did not produce an n=3 latency or output-
  parity row; registered/staged steady parity is the local correctness claim.

The roadmap in [RFC #6414](https://github.com/vllm-project/vllm-omni/issues/6414)
tracks FP8 producers and possible future AllGather consumers. Those should
extend the exact store/lease contracts rather than create another cache format.

## References {#references}

**This list is the article's receipt: every architecture and performance claim
maps back to a merged PR, current guide, or preserved benchmark artifact.** Use
the pinned links when auditing code and the current guide when operating a
deployment.

- [PR #6213 — Loader-owned checkpoint host-weight plans for TP1 DLO](https://github.com/vllm-project/vllm-omni/pull/6213)
- [PR #6419 — Host Weight Runtime foundation](https://github.com/vllm-project/vllm-omni/pull/6419)
- [PR #6427 — Explicit post-load HWR publication](https://github.com/vllm-project/vllm-omni/pull/6427)
- [PR #6445 — Final-layout BF16 HWR artifacts](https://github.com/vllm-project/vllm-omni/pull/6445)
- [PR #6486 — no-AllGather DLO HWR consumer](https://github.com/vllm-project/vllm-omni/pull/6486)
- [PR #6591 — Registered HWR mmap for direct DLO H2D](https://github.com/vllm-project/vllm-omni/pull/6591)
- [RFC #6414 — reusable runtime-ready host artifacts](https://github.com/vllm-project/vllm-omni/issues/6414)
- [Current DLO user guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md)
- [Current HWR module design](https://github.com/vllm-project/vllm-omni/blob/main/docs/design/module/host_weight_runtime.md)
- [PR #6213 predecessor article]({{ site.baseurl }}/2026-08-16-pr-6213-loader-owned-host-weight-plans/)
- [2026-08-26 H200 TP validation evidence](https://github.com/hsliuustc0106/vllm-omni-cookbook/tree/main/blog/assets/figures/host-weight-runtime/evidence/2026-08-26-h200-tp-validation)
