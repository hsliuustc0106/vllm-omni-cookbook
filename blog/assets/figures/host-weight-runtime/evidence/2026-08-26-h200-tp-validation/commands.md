# HWR TP validation commands

Frozen source: `/home/hsliu/tmp/vllm-omni-hwr-tp-validation` at
`497c537c6f70e44f376b491bf7b50395cf2cba5d`.

## TP1

```bash
gpu run --gpu-ids 0,1 --wait 60m --timeout 75m \
  --note hwr-tp1-detailed-validation-497c537c -- \
  numactl --cpunodebind=0 --membind=0 \
  env \
    PYTHONPATH=/home/hsliu/tmp/vllm-omni-hwr-tp-validation \
    HF_MODULES_CACHE=/tmp/hf-modules/hwr-tp-validation \
    TMPDIR=/tmp/hwr-tp-validation-497c537c/tmp \
    VLLM_CACHE_ROOT=/tmp/hwr-tp-validation-497c537c/vllm-cache \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
  /tmp/venvs/vllm-omni-pr3-8b1bbdfb/bin/python \
    /tmp/hwr-tp-validation-497c537c/tools/monitor_run.py \
    --output-dir /tmp/hwr-tp-validation-497c537c/runs/tp1-monitor \
    --interval 0.5 --timeout 5000 --gpu-indices 0,1 -- \
  /tmp/venvs/vllm-omni-pr3-8b1bbdfb/bin/python \
    /tmp/hwr-tp-validation-497c537c/tools/run_tp1_validation.py \
    --physical-gpus 0,1 \
    --store /tmp/hwr-tp-validation-497c537c/store-tp1 \
    --source-root /home/hsliu/tmp/vllm-omni-hwr-tp-validation \
    --timeout 4200
```

After successful execution:

```bash
/tmp/venvs/vllm-omni-pr3-8b1bbdfb/bin/python \
  /tmp/hwr-tp-validation-497c537c/tools/analyze_tp1.py
```

Expected report: `/tmp/hwr-tp-validation-497c537c/TP1_REPORT.md`.

## TP2

The master controller reserves all four GPUs once, performs cold publication,
the required-hit feasibility check, balanced second-engine startup samples,
and five monitored pair blocks. Each pair monitor takes its final memory/NUMA
snapshot after warmup and then quiesces before measured waves.

```bash
gpu run --gpu-ids 0,1,2,3 --timeout 5h \
  --note hwr-tp2-detailed-validation-497c537c -- \
  numactl --cpunodebind=0 --membind=0 \
  env \
    TMPDIR=/tmp/hwr-tp-validation-497c537c/tmp/tp2 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/hwr-tp-validation-497c537c/torchinductor-tp2 \
    VLLM_CACHE_ROOT=/tmp/hwr-tp-validation-497c537c/vllm-cache-tp2 \
    HF_MODULES_CACHE=/tmp/hwr-tp-validation-497c537c/hf-modules-cache-tp2 \
  /tmp/venvs/vllm-omni-pr3-8b1bbdfb/bin/python \
    /tmp/hwr-tp-validation-497c537c/tools/run_tp2_validation.py \
    --root /tmp/hwr-tp-validation-497c537c \
    --store /tmp/hwr-tp-validation-497c537c/store-tp2 \
    --source-root /home/hsliu/tmp/vllm-omni-hwr-tp-validation \
    --model /workspace/model/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA \
    --python /tmp/venvs/vllm-omni-pr3-8b1bbdfb/bin/python \
    --physical-gpus 0,1,2,3 \
    --expected-sha 497c537c6f70e44f376b491bf7b50395cf2cba5d \
    --phase-timeout 7200 \
    --total-timeout 16800
```

After successful execution:

```bash
/tmp/venvs/vllm-omni-pr3-8b1bbdfb/bin/python \
  /tmp/hwr-tp-validation-497c537c/tools/analyze_tp2_primary.py
```

Expected report: `/tmp/hwr-tp-validation-497c537c/TP2_REPORT.md`.

### Actual completion boundary

The controller completed prewarm, feasibility, balanced startup, registered
pair, and staged pair phases. The existing-path pair produced its memory/NUMA
snapshot and one 803.91-second measured wave, then its next wave hit the
default 600-second async-output watchdog. A retry with only that watchdog
raised to 2400 seconds was stopped after the user requested faster completion;
its partial cgroup-memory samples are preserved.

The optional TP2 `resident_layers=0` profile was omitted after GPUs were
reallocated. `analyze_tp2_primary.py` therefore makes no TP2 per-layer scaling
claim and treats the existing-path row as capacity evidence rather than an n=3
transport comparison.
