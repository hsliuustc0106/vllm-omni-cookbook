# Profile for optimization notes (Claude Code stub)

Profiling is **optional** — only needed when explaining *why* a delta happened in narrative or optimization tables.

## Diffusion

Enable pipeline profiler in serve args (see perf JSON):

```bash
vllm serve ... --omni --enable-diffusion-pipeline-profiler
```

Use stage breakdown (text encode → VAE → DiT → decode) in:

- `index.md` **Optimization summary** table
- Optimization notes (`cookbook-write-narrative` skill)

## TTS / Omni

Use upstream bench outputs and design docs; stage separation (talker / code2wav / thinker) maps to PR themes.

## Rules

- Separate CI H100 thresholds from local retro numbers
- Tie claims to profiler logs or PR descriptions — do not speculate
- Benchmark workflow: vllm-omni `benchmark_results/` READMEs and model `index.md` reproduce sections; upstream skills `diffusion-perf-cookbook` / `tts-perf-cookbook` if available
