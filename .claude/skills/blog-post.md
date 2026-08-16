# Write PR-analysis blog post (Claude Code stub)

Load the full skill: `.cursor/skills/cookbook-blog-post/SKILL.md`

Quick path (when an important vllm-omni PR/feature/model is finalized):

1. Copy `blog/TEMPLATE.md` → `blog/_posts/YYYY-MM-DD-<slug>.md`
2. Structure: TL;DR → Background → What the PR does → Key changes → Measured impact → How to use → Limitations → References
3. Metrics only from cookbook ledgers or upstream perf JSON; link cookbook with absolute GitHub URLs
4. Figures: `blog/assets/figures/<slug>/`; raw HTML diagrams OK anywhere
5. Preview: `cd blog && bundle exec jekyll serve` → http://127.0.0.1:4000/vllm-omni-cookbook/

Dummy example: `blog/_posts/2026-08-16-pr-analysis-template.md`
