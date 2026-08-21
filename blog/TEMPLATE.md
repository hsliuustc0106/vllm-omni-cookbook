# Blog Post Template — PR Analysis

Copy this file to `blog/_posts/YYYY-MM-DD-<short-slug>.md` (the filename date is
the publish date) and replace every placeholder. Sections marked *(omit if n/a)*
can be deleted. The rendered-template dummy post was removed once the first
real post landed; copy this file and replace placeholders directly.

Rules:

- **Audience:** vLLM-Omni users — explain *how* it works and *why* it helps, not
  just what changed.
- **Numbers:** only metrics that exist in the cookbook
  (`{category}/{model}/index.md`, `SUMMARY.md`) or upstream perf JSON. Never
  invent or estimate metrics.
- **Links to the cookbook must be absolute GitHub URLs** — the blog deploys as a
  standalone site, so relative cookbook paths would 404.
- `summary` front matter ≤ 240 characters (SEO description + index teaser).
- One post per PR or feature. Tags: the model plus hardware/workload
  identifiers in their canonical brand casing (e.g. `qwen3-omni`,
  `MiniMax-H3`, `H200`, `Blackwell`). Never use a `site.features` slug as a
  tag — feature classification belongs to the `feature:` front matter only,
  or the card shows the same label twice.
- Optional card front matter: `image:` (cover shown on the home-page card) and
  `featured: true` (pins the post as the featured card; newest is featured when
  unset).
- `feature: <slug>` — one canonical feature area for the home sidebar filter;
  slugs are curated in `blog/_config.yml` (`site.features`) and match
  vLLM-Omni's `docs/design/feature/` page names (e.g. `quantization`,
  `offloader`, `async_chunk`, `disaggregated_inference`).
- Figures go in `blog/assets/figures/<slug>/`.
- The post layout is automatic: a left table of contents with scroll-spy,
  numbered section kickers, a reading-progress bar, and Copy buttons on every
  code block. Authors never hand-write a TOC or section numbers.

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
feature: <site.features slug, e.g. quantization>   # home sidebar filter
math: true   # omit when the post has no math
# image: /assets/figures/<slug>/cover.png   # optional home-card cover
# featured: true                            # optional: pin as featured card
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

The flags or commands a user needs, as a tabbed cookbook: one tab per mode
(topology, stage, or interface), each with a copy-ready command and a one-line
caveat. Define the modes in `usage:` front matter and render them with the
include; link the upstream recipe if one exists. Two or more modes justify the
tabs — a single command can stay a plain code block.

```yaml
usage:
  - label: "Serve · DP2 no-AG"     # tab title (short)
    blurb: "the measured config"   # optional tab subtitle
    title: "MiniMax-H3 · DP2"      # optional panel heading (defaults to label)
    code: |                        # multi-line command; YAML literal block
      vllm serve /path/to/model --omni \
        --data-parallel-size 2
    note: >-                       # optional one-line caveat under the code
      Loader falls back silently when the preflight fails.
```

```markdown
## How to use it
One sentence of setup, then:
{% include usage-cookbook.html modes=page.usage %}
```

## How to choose

"If you are in situation X, pick Y" cards — distilled from the post's own
limitations and the upstream docs' guidance; never introduce claims the post
doesn't already make. Define them in `decisions:` front matter and render them
with the include. Three to seven cards works best.

```yaml
decisions:
  - when: "host RAM tight"          # situation kicker
    pick: "DP + no-AllGather"       # the recommendation
    why: "Workers share checkpoint pages (`--dlo-no-use-allgather`)."  # markdown
```

```markdown
## How to choose
{% include decision-cards.html items=page.decisions %}
```

## Limitations & follow-ups

Known caveats, unsupported cases, open issues, planned next steps — with links.

## References

- [PR #NNNN — <title>](https://github.com/vllm-project/vllm-omni/pull/NNNN)
- Related issues, RFCs, cookbook sections (absolute GitHub URLs), upstream
  benchmark JSON.
````
