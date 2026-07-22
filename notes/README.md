# vLLM-Omni Update Notes

This directory is a chronological collection of Chinese, Zhihu-style technical
articles about vLLM-Omni developments worth following between cookbook releases:
feature updates, RFCs and design discussions, notable PRs, and performance work.

Notes have **no fixed cadence**. Create one whenever there is a useful set of updates
to capture. Use a chronological, descriptive filename
(`YYYY-MM-DD-<topic>.md`) and state the reviewed period in the article metadata.

## Publishing a Note

1. Choose one coherent theme; group multiple updates only when they tell one story.
2. Copy [`TEMPLATE.md`](TEMPLATE.md) to `YYYY-MM-DD-<topic>.md`.
3. Start from the problem and explain the architecture, tradeoffs, and impact in
   Chinese prose. Use tables only when they make a comparison clearer.
4. Prefer short code evidence and one-line flows that remain readable on mobile;
   render complex architecture diagrams as images before publishing.
5. Give each major section one bold takeaway for readers who skim.
6. Separate confirmed facts, open proposals, inferences, and measured results.
7. Link every RFC, PR, benchmark, or discussion used as evidence.
8. Add the new note to the list below, newest first.

## Notes

- [2026-07-22](2026-07-22-multi-stage-kv-cache-transfer.md) — vLLM 的 P/D 解耦已经很省了，为什么多模态的
  KV 还在反复搬？
