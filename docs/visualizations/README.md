# Design Visualizations

Review artifacts that visualize the architecture of notable vLLM-Omni changes.
Each HTML page is a single self-contained file — open it directly in any browser;
the only network dependency is Google Fonts (IBM Plex), and pages render fine
without it. SVG sources are editable; the matching PNG is a rendered snapshot.

| Artifact | Change | Added | Notes |
|----------|--------|-------|-------|
| [pr-6094-kv-design.svg](pr-6094-kv-design.svg) · [PNG](pr-6094-kv-control-plane.png) | [vllm-project/vllm-omni#6094](https://github.com/vllm-project/vllm-omni/pull/6094) — diffusion KV cache control plane | 2026-08-16 | editable SVG source + render |
| [pr-6206-vae-groups-design.html](pr-6206-vae-groups-design.html) · [PNG](pr-6206-vae-groups-design.png) | [vllm-project/vllm-omni#6206](https://github.com/vllm-project/vllm-omni/pull/6206) — H3 independent VAE process groups | 2026-08-16 | editable HTML source + render |
| [pr-6213-dlo-host-weight-plans.html](pr-6213-dlo-host-weight-plans.html) | [vllm-project/vllm-omni#6213](https://github.com/vllm-project/vllm-omni/pull/6213) — loader-owned host-weight plans for DLO | 2026-08-16 | analyzed in the [blog post](https://github.com/hsliuustc0106/vllm-omni-cookbook/blob/main/blog/_posts/2026-08-16-pr-6213-loader-owned-host-weight-plans.md); a served copy lives under `blog/assets/figures/pr-6213-dlo-host-weight-plans/` |

## PR #6213 — loader-owned host-weight plans for DLO

Reviewed at PR head `5ce549bc` (merged as `9d2bb23ff6`). Contents:

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

Measured numbers come from the PR's own validation section; diagrams were
verified against the code at the head SHA above during review.
