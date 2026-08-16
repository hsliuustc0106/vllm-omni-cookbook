# Design Visualizations

Self-contained HTML pages that visualize the architecture of notable vLLM-Omni
changes. Each page is a single file — open it directly in any browser; the only
network dependency is Google Fonts (IBM Plex), and the page renders fine without it.

| Page | Change | Reviewed at | Generated |
|------|--------|-------------|-----------|
| [PR #6213 — Loader-owned host-weight plans for DLO](pr-6213-dlo-host-weight-plans.html) | [vllm-project/vllm-omni#6213](https://github.com/vllm-project/vllm-omni/pull/6213) (Phase A of RFC [#6195](https://github.com/vllm-project/vllm-omni/issues/6195)) | head `5ce549bc` | 2026-08-15 |

## PR #6213 contents

1. **Ownership flow** — before (two comment-synced gates) vs. after (the loader as
   single owner of checkpoint semantics, one-shot `HostWeightPlan` transfer).
2. **Preflight ladder** — the fail-closed checks that produce a plan or fall back
   to the ordinary loader before any mutation.
3. **Storage × transfer matrix** — host storage (loader) and transfer protocol
   (backend) as orthogonal decisions; all four combinations.
4. **Rank-local runtime pipeline** — mmap views → two bounded pinned staging slots
   → double-buffered device slots, with the lifecycle order.
5. **Measured node memory** — the two-worker L20X run: node PSS 283.56 → 150.08 GiB
   (−47.1%) with ordinary-loader fallback, from `smaps_rollup`.

Measured numbers come from the PR's own validation section; diagrams were verified
against the code at the head SHA above during review.
