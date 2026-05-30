# Cookbook templates

## index.md skeleton

```markdown
# {Model Name}

**Category:** {Omni|TTS|Diffusion} ({short description})
**Model:** `{org}/{hf-id}`
**Recipe:** [{name}](https://github.com/vllm-project/vllm-omni/blob/main/recipes/...)
**Retro harness:** [benchmark_results/{slug}_retro/](https://github.com/vllm-project/vllm-omni/blob/main/benchmark_results/{slug}_retro/README.md)

Optional:
**Zhihu draft (中文):** [{model}-performance-zhihu.md]({model}-performance-zhihu.md)

---

## Key metrics

| Metric | Definition | Lower / higher? | When it matters |
|--------|------------|-----------------|-----------------|
| ... | ... | ... | ... |

---

## Performance tracks

| Track | Hardware | Source |
|-------|----------|--------|
| **CI** | 2× H100 | [`test_....json`](upstream link) |
| **Retro** | N× {SKU} | [Table below](#retro-comparison) |

---

## Standardized perf test (CI)

(table + reproduction command)

---

## {SKU} retro comparison

Measured **YYYY-MM-DD** on **N× {GPU}**. Metric: **`latency_mean`** / **TTFP** / ...

| Config | Workload | v0.18.0 | v0.20.0 | Δ ... |
|--------|----------|---------|---------|-------|

Footnotes: GPU layout, missing flags, prompt/warmup policy, API backend.

---

## Optimization summary

| Phase | Release | Focus | Notable PRs |
|-------|---------|-------|-------------|

---

## Serve commands

(minimal — match perf JSON)
```

## SUMMARY.md release block

```markdown
## vX.Y.Z (YYYY-MM-DD)

| Model | Category | Key Metric | Value | Delta from vPREV |
|-------|----------|------------|-------|-------------------|
| {Model} | {cat} | {metric + config hint} | **{value}** | **{±N%}** |

### Highlights

- **{Model}:** {one sentence} — [index]({path})
```

## Pull request

```markdown
## Description

Perf ledger update for vX.Y.Z — {model(s)}.

## Type of Change

- [x] Documentation improvement (perf ledger)
- [ ] New model ledger
- [ ] Narrative (Zhihu draft)

## Models updated

- `diffusion/wan2.2/index.md`: v0.22 retro table + optimization notes
- `SUMMARY.md`: v0.22 release section

## Evidence

| Field | Value |
|-------|-------|
| vllm-omni baseline SHA | `abc1234` (v0.20.0 tag) |
| vllm-omni candidate SHA | `def5678` (v0.22.0 tag) |
| vLLM / vLLM-Omni | 0.22.0 / 0.22.0 |
| GPUs | 4× H200, `CUDA_VISIBLE_DEVICES=0,1,2,3` |
| Upstream JSON | `tests/dfx/perf/tests/test_....json` |

## Checklist

- [ ] Numbers from completed retro or CI (not estimates)
- [ ] Non-comparable rows footnoted
- [ ] `SUMMARY.md` updated
- [ ] Links to recipe + perf JSON + PRs work
- [ ] No upstream perf JSON copied into cookbook repo
```

## Comparison table footnote template

```markdown
† v0.16: no `vae-patch-parallel-size=2`; GPUs `0,1,3`.
‡ v0.18/v0.20: full USP2 stack; GPUs `0,1,2,3`.
§ CI uses `num-prompts=10`; retro uses `num-prompts=3`, `warmup-requests=0`.
```
