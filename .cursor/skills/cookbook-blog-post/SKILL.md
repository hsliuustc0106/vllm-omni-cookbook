---
name: cookbook-blog-post
description: >-
  Write a PR-analysis blog post for the vllm-omni-cookbook Jekyll blog (blog/).
  Use when an important vllm-omni PR, feature, or model support is finalized and
  needs a deep-dive HTML post explaining how it works and why it helps. Given a
  PR or issue number, gathers context via gh and drafts the post end to end.
---

# Write a PR-Analysis Blog Post

Trigger: an important vLLM-Omni PR / feature / model support is **finalized**
(merged, or shipped in a release). The cookbook records the numbers; the blog
post explains the *how* and *why* for users.

## Workflow

1. Copy `blog/TEMPLATE.md` → `blog/_posts/YYYY-MM-DD-<short-slug>.md`.
   The filename date is the publish date.
2. Gather context — see **Starting from a PR or issue** below.
3. Write for users, not for reviewers: **TL;DR → Background (symptom, then
   cause) → What the PR does (design + diagram) → Key changes (file-by-file
   excerpts) → Measured impact → How to use it → Limitations → References.**
4. Metrics may **only** come from the cookbook (`{category}/{model}/index.md`,
   `SUMMARY.md`) or upstream perf JSON. Never invent or estimate numbers.
5. Link the cookbook with **absolute GitHub URLs** — the blog deploys standalone
   (`blog/` is its own Jekyll site), so relative cookbook paths 404.
6. Figures → `blog/assets/figures/<slug>/`; embed with
   `{{ site.baseurl }}/assets/figures/<slug>/fig1.png`. Raw HTML (inline SVG,
   styled divs) passes through Markdown anywhere in the post.
7. Front matter: `summary` ≤ 240 chars (SEO + index teaser), `category: PR
   Analysis`, tags = model + technical area, `feature:` = one slug from
   `site.features` in `blog/_config.yml` (home sidebar filter; slugs match
   vLLM-Omni `docs/design/feature/` page names), `math: true` only if using
   `$…$`.
8. Preview locally:
   `cd blog && bundle install && bundle exec jekyll serve`
   → http://127.0.0.1:4000/vllm-omni-cookbook/
9. Delete every template admonition and placeholder before publishing.

## Starting from a PR or issue

Inputs: a PR number from `vllm-project/vllm-omni` (normal case), or an issue
number only (RFC / design discussion, nothing merged yet → write a *design
explainer*: same skeleton, TL;DR states it is a proposal, every impact claim
marked with status).

Gather everything with read-only `gh` before writing:

```bash
gh pr view <N> --repo vllm-project/vllm-omni \
  --json number,title,author,body,mergedAt,files,labels,reviews,comments
gh pr diff <N> --repo vllm-project/vllm-omni | head -400   # key files first
gh pr view <N> --repo vllm-project/vllm-omni --json closingIssuesReferences
gh issue view <I> --repo vllm-project/vllm-omni --json title,body,comments
grep -rn "<N>" omni/ diffusion/ SUMMARY.md        # cookbook evidence hits
ls docs/visualizations/ | grep "<N>"              # existing design diagrams
```

Map gathered material onto the skeleton:

| Source | Post section |
|--------|--------------|
| Issue body / PR problem statement | **Background** (symptom first, then cause) |
| PR description + review discussion | **What the PR does** (surface debated design decisions) |
| `gh pr diff` (biggest files) | **Key changes** (short excerpts + GitHub file links) |
| Cookbook grep hits (`index.md`, `SUMMARY.md`) | **Measured impact** (absolute GitHub URLs) |
| PR usage example / upstream recipe | **How to use it** |
| Open follow-up issues, review TODOs | **Limitations & follow-ups** |

If `docs/visualizations/pr-<N>-*` diagrams exist (e.g. pr-6094 KV control
plane, pr-6206 VAE groups), copy the best ones into
`blog/assets/figures/<slug>/` instead of drawing new ones. If the cookbook has
no numbers for the PR, write the design story and state that performance is
not yet measured — do not estimate.

## Publishing

Push to `main` with any `blog/**` change auto-deploys via
`.github/workflows/blog-pages.yml` to GitHub Pages. No release cadence — posts
publish whenever something important is finalized.

## Post checklist

- [ ] Filename `YYYY-MM-DD-<slug>.md`; title format "Understanding PR #NNNN — <feature>"
- [ ] TL;DR has the headline numbers; every metric cites its evidence
- [ ] Evidence rules: GPU SKU/count, vllm + vllm_omni versions, commit SHAs, footnotes for non-comparable rows
- [ ] Cookbook links are absolute GitHub URLs
- [ ] `summary` ≤ 240 chars; tags include the model name
- [ ] No full duplication of the model ledger — summarize and link it
- [ ] Template placeholders and instructional admonitions removed
