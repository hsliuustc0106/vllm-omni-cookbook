---
layout: post
title: "[TEMPLATE] Understanding PR #0 — Example Feature"
date: 2026-08-16 12:00:00 +0800
author: your-handle
summary: >-
  Placeholder post that validates the blog pipeline and demonstrates every
  authoring feature of the template. Delete it when the first real post lands.
tags: [example, template]
category: PR Analysis
math: true
---

> [!WARNING]
> This is a **placeholder** post. It exists only to validate the Jekyll build
> and to demonstrate the authoring features (admonitions, HTML diagrams, math,
> tables, figures). Replace it with the first real PR analysis, or delete it.

## TL;DR

**This post is the rendered form of
[TEMPLATE.md](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/blog/TEMPLATE.md) —
copy it, keep the structure, replace the content.** A real TL;DR states what the
PR does and the headline measured effect, followed by the key numbers:

| Metric | Before | After | Δ | Setup |
|--------|--------|-------|---|-------|
| example latency | 100 ms | 50 ms | −50% | 2×H200, c=1 |

*(All numbers in this placeholder are fake — never invent metrics in a real post.)*

## Background

Explain the user-visible symptom first, then the technical cause, then which
workloads were hit hardest.

> [!TIP]
> `NOTE`, `TIP`, `IMPORTANT`, `WARNING`, and `CAUTION` admonitions render as
> GitHub-style callouts via the `jekyll-gfm-admonitions` plugin.

## What the PR does

One paragraph with the idea, then the moving parts. Raw HTML passes through
Markdown untouched — inline SVG diagrams work anywhere in a post:

<div style="text-align:center;">
<svg width="420" height="90" viewBox="0 0 420 90" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Example flow: request to response">
  <rect x="5"  y="30" width="90" height="30" rx="6" fill="#f6f8fa" stroke="#d0d7de"/>
  <text x="50" y="49" font-size="12" text-anchor="middle" fill="#1f2328">request</text>
  <rect x="165" y="30" width="90" height="30" rx="6" fill="#ddf4e4" stroke="#1a7f37"/>
  <text x="210" y="49" font-size="12" text-anchor="middle" fill="#1f2328">fast path</text>
  <rect x="325" y="30" width="90" height="30" rx="6" fill="#f6f8fa" stroke="#d0d7de"/>
  <text x="370" y="49" font-size="12" text-anchor="middle" fill="#1f2328">response</text>
  <line x1="95"  y1="45" x2="165" y2="45" stroke="#d0d7de" stroke-width="2"/>
  <line x1="255" y1="45" x2="325" y2="45" stroke="#d0d7de" stroke-width="2"/>
</svg>
</div>

With `math: true` in the front matter, MathJax supports `$inline math$` and
display math:

$$
\text{RTF} = \frac{\text{audio out duration}}{\text{compute time}}
$$

## Key changes

Walk through the diff file by file with short excerpts and GitHub file links:

```python
# vllm_omni/example.py — link the file, excerpt only the lines that matter
def fast_path(request): ...
```

## Measured impact

Cite the cookbook with **absolute GitHub URLs** (the blog deploys standalone —
relative cookbook paths 404). Follow the cookbook evidence rules: GPU SKU and
count, `vllm` + `vllm_omni` versions, commit SHAs.

| Metric | Before | After | Δ | Evidence |
|--------|--------|-------|---|----------|
| — | — | — | — | [{category}/{model}/index.md](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/README.md) |

Embed charts from `blog/assets/figures/<slug>/` with `site.baseurl`:

```markdown
![chart caption]({{ site.baseurl }}/assets/figures/<slug>/fig1.png)
```

## How to use it

The flags or commands a user needs, plus a minimal working example.

## Limitations & follow-ups

Known caveats, unsupported cases, open issues, planned next steps — with links.

## References

- [TEMPLATE.md — the canonical post template](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/blog/TEMPLATE.md)
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
