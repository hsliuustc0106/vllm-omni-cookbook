# vLLM-Omni Update Notes

This directory is a chronological collection of Chinese, Zhihu-style technical
articles about vLLM-Omni developments worth following between cookbook releases:
feature updates, RFCs and design discussions, notable PRs, and performance work.

Notes have **no fixed cadence**. Create one whenever there is a useful set of updates
to capture. Name it by publication date (`YYYY-MM-DD.md`) and state the reviewed
period in the article metadata.

## Publishing a Note

1. Choose one coherent theme; group multiple updates only when they tell one story.
2. Copy [`TEMPLATE.md`](TEMPLATE.md) to `YYYY-MM-DD.md`.
3. Start from the problem and explain the architecture, tradeoffs, and impact in
   Chinese prose. Use tables only when they make a comparison clearer.
4. Separate confirmed facts, open proposals, inferences, and measured results.
5. Link every RFC, PR, benchmark, or discussion used as evidence.
6. Add the new note to the list below, newest first.

## Notes

- [2026-07-22](2026-07-22.md) — KV Cache 为什么还在反复全量搬运？vLLM-Omni
  多阶段传输 RFC 解读。
