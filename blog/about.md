---
layout: page
title: About
---

This blog publishes deep dives into important
[vLLM-Omni](https://github.com/vllm-project/vllm-omni) PRs, features, and new
model support, written for users who want to understand *how* an optimization
works and *why* it helps — not just the headline number.

It is the companion of the
[vLLM-Omni Performance Cookbook](https://github.com/hsliuustc0106/vllm-omni-cookbook):

- The **cookbook** is the authoritative, release-by-release performance ledger
  (per-model `index.md` timelines and `SUMMARY.md`).
- The **blog** is the narrative layer: one post per important PR, explaining the
  design and the measured impact, always citing the cookbook for numbers.

Posts are plain Markdown (with embedded HTML diagrams where they help),
published whenever something important is finalized — no fixed cadence. The
site is built with Jekyll and deployed to GitHub Pages automatically on every
push that touches `blog/`.

To contribute a post, see
[CONTRIBUTING.md](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/CONTRIBUTING.md)
and the post template in
[`blog/TEMPLATE.md`](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/blog/TEMPLATE.md).
