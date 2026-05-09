# Performance Cookbook Refactor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the topic-based cookbook structure with a per-model performance ledger organized by category (omni/tts/diffusion) that tracks optimization deltas across vLLM-Omni stable releases.

**Architecture:** Three category folders, each containing one model folder with an `index.md` perf ledger and `assets/` directory. `SUMMARY.md` at root for cross-model release overview. All old topic-based directories and files removed.

**Tech Stack:** Markdown files only, no code.

---

### Task 1: Remove old topic-based structure

**Files:**
- Remove: `00-quickstart/` (entire directory)
- Remove: `01-inference/` (entire directory)
- Remove: `02-deployment/` (entire directory)
- Remove: `03-multimodal/` (entire directory)
- Remove: `04-hardware/` (entire directory)
- Remove: `05-best-practices/` (entire directory)
- Remove: `06-performance/` (entire directory)
- Remove: `07-troubleshooting/` (entire directory)
- Remove: `templates/` (entire directory)
- Remove: `topics/` (entire directory)

**Step 1: Delete old directories**

```bash
rm -rf 00-quickstart 01-inference 02-deployment 03-multimodal 04-hardware 05-best-practices 06-performance 07-troubleshooting templates topics
```

**Step 2: Commit**

```bash
git add -A
git commit -m "refactor: remove topic-based structure"
```

---

### Task 2: Create new category/model directory layout

**Files:**
- Create: `omni/qwen3-omni/assets/.gitkeep`
- Create: `tts/qwen3-tts/assets/.gitkeep`
- Create: `diffusion/wan2.2/assets/.gitkeep`

**Step 1: Create directories**

```bash
mkdir -p omni/qwen3-omni/assets tts/qwen3-tts/assets diffusion/wan2.2/assets
```

**Step 2: Add .gitkeep to preserve empty asset dirs**

```bash
touch omni/qwen3-omni/assets/.gitkeep tts/qwen3-tts/assets/.gitkeep diffusion/wan2.2/assets/.gitkeep
```

**Step 3: Commit**

```bash
git add omni/ tts/ diffusion/
git commit -m "feat: add category/model directory layout"
```

---

### Task 3: Write qwen3-omni perf ledger

**Files:**
- Create: `omni/qwen3-omni/index.md`

**Step 1: Write the file**

```markdown
# Qwen3-Omni

**Category:** Omni (omni-modal / any-to-any)
**Recipe:** [Qwen3-Omni on vLLM-Omni](https://github.com/vllm-project/vllm-omni/tree/main/recipes/qwen3-omni)

Omni-modal model supporting text, image, and audio inputs with multimodal outputs.

## Hardware Baseline

All benchmarks run on **1× NVIDIA H100 80GB**.

## v0.20.0 (2026-05-01) — Baseline

First stable release tracked in this cookbook.

### Performance

| Metric           | Value   | Notes        |
|------------------|---------|--------------|
| TTFT (Time to First Token)  | — | To be measured |
| TTFP (Time to First Packet) | — | To be measured |
| TPOT (Time per Output Token)| — | To be measured |
| RTF (Real-Time Factor)      | — | To be measured |
| Throughput       | —       | To be measured |
| GPU Memory       | —       | To be measured |

### Optimization Notes

_Initial baseline — no deltas to report._

### Figures

_No figures for baseline release._
```

**Step 2: Commit**

```bash
git add omni/qwen3-omni/index.md
git commit -m "feat: add qwen3-omni perf ledger (v0.20.0 baseline)"
```

---

### Task 4: Write qwen3-tts perf ledger

**Files:**
- Create: `tts/qwen3-tts/index.md`

**Step 1: Write the file**

```markdown
# Qwen3-TTS

**Category:** TTS (Text-to-Speech)
**Recipe:** [Qwen3-TTS on vLLM-Omni](https://github.com/vllm-project/vllm-omni/tree/main/recipes/qwen3-tts)

Text-to-speech model for audio generation.

## Hardware Baseline

All benchmarks run on **1× NVIDIA H100 80GB**.

## v0.20.0 (2026-05-01) — Baseline

First stable release tracked in this cookbook.

### Performance

| Metric           | Value   | Notes        |
|------------------|---------|--------------|
| TTFP (Time to First Packet) | — | To be measured |
| RTF (Real-Time Factor)      | — | To be measured |
| Throughput       | —       | To be measured |
| GPU Memory       | —       | To be measured |

### Optimization Notes

_Initial baseline — no deltas to report._

### Figures

_No figures for baseline release._
```

**Step 2: Commit**

```bash
git add tts/qwen3-tts/index.md
git commit -m "feat: add qwen3-tts perf ledger (v0.20.0 baseline)"
```

---

### Task 5: Write wan2.2 perf ledger

**Files:**
- Create: `diffusion/wan2.2/index.md`

**Step 1: Write the file**

```markdown
# WAN2.2

**Category:** Diffusion (Image/Video Generation)
**Recipe:** [WAN2.2 on vLLM-Omni](https://github.com/vllm-project/vllm-omni/tree/main/recipes/wan2.2)

Diffusion Transformer (DiT) model for image and video generation.

## Hardware Baseline

All benchmarks run on **1× NVIDIA H100 80GB**.

## v0.20.0 (2026-05-01) — Baseline

First stable release tracked in this cookbook.

### Performance

| Metric           | Value   | Notes        |
|------------------|---------|--------------|
| E2E Latency     | —       | To be measured |
| Throughput       | —       | To be measured |
| GPU Memory       | —       | To be measured |

### Optimization Notes

_Initial baseline — no deltas to report._

### Figures

_No figures for baseline release._
```

**Step 2: Commit**

```bash
git add diffusion/wan2.2/index.md
git commit -m "feat: add wan2.2 perf ledger (v0.20.0 baseline)"
```

---

### Task 6: Write SUMMARY.md

**Files:**
- Create: `SUMMARY.md`

**Step 1: Write the file**

```markdown
# vLLM-Omni Performance Summary

Cross-model performance deltas across vLLM-Omni stable releases. For per-model details, see the model index files.

## v0.20.0 (2026-05-01) — Baseline

| Model      | Category   | Key Metric     | Value   | Delta |
|------------|------------|----------------|---------|-------|
| Qwen3-Omni | omni       | —              | —       | —     |
| Qwen3-TTS  | tts        | —              | —       | —     |
| WAN2.2     | diffusion  | —              | —       | —     |

### Highlights

Initial baseline — performance data to be populated.
```

**Step 2: Commit**

```bash
git add SUMMARY.md
git commit -m "feat: add cross-model performance summary"
```

---

### Task 7: Update README.md

**Files:**
- Modify: `README.md` (full rewrite)

**Step 1: Rewrite README.md**

New content:

```markdown
# vLLM-Omni Performance Cookbook

Performance evolution tracking for omni-modal, TTS, and diffusion models running on vLLM-Omni. Each stable release records measured performance and optimization deltas.

Versioning mirrors [vLLM-Omni](https://github.com/vllm-project/vllm-omni) stable releases.

## Models Tracked

| Model       | Category   | Type              |
|------------|------------|-------------------|
| Qwen3-Omni | [omni](omni/qwen3-omni/)    | Omni-modal / any-to-any |
| Qwen3-TTS  | [tts](tts/qwen3-tts/)      | Text-to-speech     |
| WAN2.2     | [diffusion](diffusion/wan2.2/) | DiT image/video generation |

## Latest Release: v0.20.0 (Baseline)

See [SUMMARY.md](SUMMARY.md) for the cross-model release overview.

## Metrics

| Model Type | Primary Metrics                    |
|-----------|------------------------------------|
| Omni      | TTFT, TTFP, TPOT, RTF             |
| TTS       | TTFP, RTF                         |
| Diffusion | E2E Latency                       |
| All       | Throughput, GPU memory, HW efficiency |

## How to Add a New Release

1. Create a new `## vX.Y.Z (YYYY-MM-DD)` section in each model's `index.md`
2. Add a performance table with measured values and delta from the previous release
3. Add optimization notes with links to relevant PRs/issues/docs
4. Add figures to the model's `assets/` directory
5. Update `SUMMARY.md` with the cross-model release table

## How to Add a New Model

```bash
mkdir -p <category>/<model-name>/assets
touch <category>/<model-name>/assets/.gitkeep
```

Then write `<category>/<model-name>/index.md` following the format of existing models.

## Resources

- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) — source repository
- [vLLM-Omni Docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/)
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for performance cookbook"
```

---

### Task 8: Clean up outdated config files

**Files:**
- Modify: `CLAUDE.md` (update to reflect new structure)

**Step 1: Update CLAUDE.md**

Remove old "Adding New Recipes", "Recipe Format", "Code Style" sections. Replace with:

```markdown
# vLLM-Omni Performance Cookbook - Guidance for Claude Code

This is a performance tracking repository — not a how-to cookbook. Each model folder contains a performance ledger (`index.md`) that records measured performance and optimization deltas across vLLM-Omni stable releases.

## Repository Structure

```
├── README.md              # Overview and model listing
├── SUMMARY.md             # Cross-model release summary
├── omni/                  # Omni-modal models
│   └── qwen3-omni/
│       ├── index.md        # Perf ledger across releases
│       └── assets/         # Charts and figures
├── tts/                   # TTS models
│   └── qwen3-tts/
│       ├── index.md
│       └── assets/
└── diffusion/             # Diffusion models
    └── wan2.2/
        ├── index.md
        └── assets/
```

## Adding Performance Data for a New Release

1. In each model's `index.md`, add a new `## vX.Y.Z (YYYY-MM-DD)` section
2. Include: performance table with deltas, optimization notes (PRs/issues/docs links), figures
3. Update `SUMMARY.md` with the cross-model release table

## Adding a New Model

```bash
mkdir -p <category>/<model-name>/assets
touch <category>/<model-name>/assets/.gitkeep
```

Write `<category>/<model-name>/index.md` following the same format as existing models.

## Resources

- vLLM-Omni: https://github.com/vllm-project/vllm-omni
- vLLM-Omni Docs: https://docs.vllm.ai/projects/vllm-omni/en/latest/
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for performance cookbook"
```
