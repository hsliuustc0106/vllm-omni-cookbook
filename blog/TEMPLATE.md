# Blog Post Template — PR Analysis

Copy this file to `blog/_posts/YYYY-MM-DD-<short-slug>.md` (the filename date is
the publish date) and replace every placeholder. Sections marked *(omit if n/a)*
can be deleted. The dummy post
`blog/_posts/2026-08-16-pr-analysis-template.md` shows the template rendered.

Rules:

- **Audience:** vLLM-Omni users — explain *how* it works and *why* it helps, not
  just what changed.
- **Numbers:** only metrics that exist in the cookbook
  (`{category}/{model}/index.md`, `SUMMARY.md`) or upstream perf JSON. Never
  invent or estimate metrics.
- **Links to the cookbook must be absolute GitHub URLs** — the blog deploys as a
  standalone site, so relative cookbook paths would 404.
- `summary` front matter ≤ 240 characters (SEO description + index teaser).
- One post per PR or feature. Tags: the model plus the technical area
  (e.g. `qwen3-omni`, `scheduler`).
- Figures go in `blog/assets/figures/<slug>/`.

---

````markdown
---
layout: post
title: "Understanding PR #NNNN — <feature name>"
date: YYYY-MM-DD 12:00:00 +0800
author: <GitHub handle>
summary: >-
  <One or two sentences, 240 characters or fewer — the SEO description and the
  teaser shown on the blog index.>
tags: [<model, e.g. qwen3-omni>, <area, e.g. scheduler>]
category: PR Analysis
math: true   # omit when the post has no math
---

> [!NOTE]
> Delete this admonition and every placeholder before publishing.

## TL;DR

Two or three sentences in bold: what the PR does and the headline measured
effect. Follow with the key numbers:

| Metric | Before | After | Δ | Setup |
|--------|--------|-------|---|-------|
|        |        |       |   |       |

## Background

The problem before the PR: first the user-visible symptom (what a user would
observe), then the technical cause (where the time went, what was broken, what
was unsupported). Say which workloads and models were hit hardest.

Use admonitions sparingly — they render as GitHub-style callouts:

> [!TIP]
> NOTE / TIP / IMPORTANT / WARNING / CAUTION are supported.

## What the PR does

The idea in one paragraph, then the moving parts. Prefer a diagram over five
paragraphs of prose — Markdown passes raw HTML through, so inline SVG or styled
`<div>`s work anywhere in the post:

<div style="border:1px solid #d0d7de; border-radius:8px; padding:12px; font-family:monospace; text-align:center;">
  request → <b>new fast path</b> → scheduler → worker → response
</div>

With `math: true` in the front matter, MathJax supports `$inline math$` and:

$$
\text{RTF} = \frac{\text{audio out duration}}{\text{compute time}}
$$

## Key changes

Walk through the diff file by file, with short excerpts and links to the files
on GitHub:

```python
# vllm_omni/<path>.py — link the file, excerpt only the lines that matter
```

## Measured impact

Numbers and their evidence. Follow the cookbook evidence rules: GPU SKU and
count, `vllm` + `vllm_omni` versions, commit SHAs, and footnotes when rows are
not apples-to-apples.

| Metric | Before | After | Δ | Evidence |
|--------|--------|-------|---|----------|
|        |        |       |   |          |

Embed cookbook charts with `site.baseurl` so links survive the project-pages
prefix:

![chart caption]({{ site.baseurl }}/assets/figures/<slug>/fig1.png)

## How to use it

The flags or commands a user needs, minimal working example. Link the upstream
recipe if one exists.

## Limitations & follow-ups

Known caveats, unsupported cases, open issues, planned next steps — with links.

## References

- [PR #NNNN — <title>](https://github.com/vllm-project/vllm-omni/pull/NNNN)
- Related issues, RFCs, cookbook sections (absolute GitHub URLs), upstream
  benchmark JSON.
````
