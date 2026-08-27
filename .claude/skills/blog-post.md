# Write a PR-analysis blog post from a PR or issue

Repo-scoped skill: given a vLLM-Omni PR number (and optionally an issue number),
produce a published blog post. Invoke like: *create a blog post for PR 4054*.

## Inputs

- **PR number** (required) — from `vllm-project/vllm-omni`. Skip to Step 2.
- **Issue number only** (RFC / design discussion, no merged PR yet) — write a
  *design explainer* instead: same structure, but say explicitly in the TL;DR
  that it is a proposal, not shipped, and mark status in every impact claim.

## Step 1 — Gather (read-only `gh`)

```bash
gh pr view <N> --repo vllm-project/vllm-omni \
  --json number,title,author,body,mergedAt,files,labels,reviews,comments
gh pr diff <N> --repo vllm-project/vllm-omni | head -400   # key files first
# issue context (given or auto-linked):
gh pr view <N> --repo vllm-project/vllm-omni --json closingIssuesReferences
gh issue view <I> --repo vllm-project/vllm-omni --json title,body,comments
# cookbook numbers + existing diagrams:
grep -rn "<N>" omni/ diffusion/ SUMMARY.md
ls docs/visualizations/ | grep "<N>"
```

## Step 2 — Map to the post skeleton

Copy `blog/TEMPLATE.md` → `blog/_posts/$(date +%F)-pr<N>-<short-slug>.md`, then:

| Source | Post section |
|--------|--------------|
| Issue body / PR problem statement | **Background** (symptom first, then cause) |
| PR description + review discussion | **What the PR does** (surface design decisions that reviewers debated) |
| `gh pr diff` (biggest files) | **Key changes** (short excerpts + GitHub file links) |
| Cookbook grep hits (`index.md`, `SUMMARY.md`) | **Measured impact** (cite with absolute GitHub URLs) |
| PR body usage example / recipes | **How to use it** (`usage:` front matter → tabbed cookbook) |
| Limitations, upstream "best for" guidance | **How to choose** (`decisions:` front matter → cards) |
| Open follow-up issues, review TODOs | **Limitations & follow-ups** |

Title: `Understanding PR #N — <feature>`. Tags: `[<model>, <area>]`. `summary`
front matter ≤ 240 chars. `feature:` = one slug from `site.features` in
`blog/_config.yml` (sidebar filter; slugs match vLLM-Omni
`docs/design/feature/` names, e.g. `quantization`, `offloader`,
`async_chunk`, `disaggregated_inference`). **Never invent metrics** — if the
cookbook has no numbers for it, write the design story and say performance is
not yet measured.

Write for three reader tiers at once (plain-first, eli5-style):

| Tier | Reader | They care about |
|------|--------|-----------------|
| Operator | runs serving / batch jobs | which flags, what it costs or saves, how to choose |
| Practitioner | wants the mechanism | how it works, trade-offs, failure modes |
| Expert | vLLM-Omni developer | kernel/collective detail, edge cases |

- The first paragraph after every `##` heading is the plain-language version:
  one sentence of what the section establishes + one everyday analogy, before
  any jargon. Expert detail follows later in the section.
- Define every technical term at first use; the TL;DR must read clean with no
  post context.
- Analogies must be honest — state where they break when it matters.

Every post ships in **both English and Chinese**: the canonical
`YYYY-MM-DD-<slug>.md` plus `YYYY-MM-DD-<slug>.zh.md` (same `_posts/` dir)
with `lang: zh`, mutual `pair` URLs, and a hardcoded
`permalink: /zh/<en-url>/`. The Chinese edition is a plain-first *rewrite*
(Chinese body; tables, commands, numbers verbatim; English terms in
parentheses at first use; same English tags; h2 ids matching the English
anchors, e.g. `## 背景 {#background}`). The summary lint enforces the pairing
in CI. Full schema: `blog/TEMPLATE.md`.

## Step 3 — Figures

- If `docs/visualizations/pr-<N>-*` exists: copy the best assets into
  `blog/assets/figures/<slug>/` and embed with
  `![caption]({{ site.baseurl }}/assets/figures/<slug>/fig1.png)`.
- Otherwise a small inline SVG (raw HTML passes through Markdown) beats prose.

## Step 4 — Verify and publish

```bash
cd blog && bundle exec jekyll serve   # or: JEKYLL_NO_BUNDLER_REQUIRE=1 jekyll build
```

Checklist before the PR: template placeholders removed; every metric cited;
cookbook links absolute; `summary` ≤ 240 chars; placeholders like the
`[TEMPLATE]` dummy post deleted if this is the first real post. Push a branch,
open the PR — merging to `main` auto-deploys to GitHub Pages.

## Step 5 — Xiaohongshu note (same PR)

Every post PR also ships its 小红书 note set: `blog/xhs/notes/<post-slug>.mjs`
(note config) plus the rendered `blog/xhs/out/<post-slug>/` (numbered PNG
cards + `note.txt`), generated and committed before the PR opens:

```bash
cd blog/xhs && node generate.mjs <post-slug>   # fails on limit violations
```

- Reuse the post's own figures as card bodies, in post order; SVGs are fine
  (headless Chrome rasterizes them). Captions in the config are Chinese —
  lift them from the `.zh.md` alt text, or translate when EN-only. Note copy
  is always Chinese regardless of post language.
- Calibration = 小红书读者 persona: layman reader, number-forward hook, one
  analogy per concept, no unexplained jargon — and carry the honesty caveats
  into the note body (author-reported / upstream-doc numbers, draft status).
- Hard limits (the generator fails the build): `note.title` ≤ 20 chars,
  body + hashtags ≤ 1000. Cover uses the post's `feature:` color from
  `blog/_config.yml` (darken it for `colorDeep`/`colorDarkest`). Closing
  card + CTA point to the bio link and `GitHub 搜 vllm-omni-cookbook`;
  never put QR codes in cards.
- Publishing is out of scope for the PR — notes are posted manually later,
  ≤ 1/day. Full authoring + publish checklist: `blog/xhs/README.md`.

Full long-form skill (Cursor): `.cursor/skills/cookbook-blog-post/SKILL.md`
