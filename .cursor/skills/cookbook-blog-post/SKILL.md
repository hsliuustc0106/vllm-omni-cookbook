---
name: cookbook-blog-post
description: >-
  Write a PR-analysis blog post for the vllm-omni-cookbook Jekyll blog (blog/).
  Use when an important vllm-omni PR, feature, or model support is finalized and
  needs a deep-dive HTML post explaining how it works and why it helps.
---

# Write a PR-Analysis Blog Post

Trigger: an important vLLM-Omni PR / feature / model support is **finalized**
(merged, or shipped in a release). The cookbook records the numbers; the blog
post explains the *how* and *why* for users.

## Workflow

1. Copy `blog/TEMPLATE.md` → `blog/_posts/YYYY-MM-DD-<short-slug>.md`.
   The filename date is the publish date.
2. Gather from the upstream PR: title, author, description, linked issue/RFC,
   key files changed, before/after behavior, review discussion worth surfacing.
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
   Analysis`, tags = model + technical area, `math: true` only if using `$…$`.
8. Preview locally:
   `cd blog && bundle install && bundle exec jekyll serve`
   → http://127.0.0.1:4000/vllm-omni-cookbook/
9. Delete every template admonition and placeholder before publishing.

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
