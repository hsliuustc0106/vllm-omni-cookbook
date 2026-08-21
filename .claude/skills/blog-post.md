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

Full long-form skill (Cursor): `.cursor/skills/cookbook-blog-post/SKILL.md`
