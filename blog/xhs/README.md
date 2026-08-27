# Xiaohongshu (小红书) notes

Turns each blog post into a ready-to-upload XHS 图文笔记: the post's own
figures composited onto 3:4 canvases, plus two synthetic cards (hook cover +
link end-card), and `note.txt` copy validated against XHS limits
(title ≤ 20 chars, body + hashtags ≤ 1000).

```bash
cd blog/xhs
node generate.mjs pr-6162-svdquant        # any notes/<name>.mjs
open out/pr-6162-svdquant/                # PNGs + note.txt
```

Headless Chrome renders each card's HTML at exactly 1242×1656. Set `CHROME_BIN`
if Chrome isn't at the default macOS path. No npm dependencies.

## Anatomy of a note (`notes/<name>.mjs`)

- `brand` — series names, GitHub repo handle, Pages URL (no `https://`, fits
  the URL box).
- `color` / `colorDeep` / `colorDarkest` — take these from the post's
  `feature:` entry in `blog/_config.yml` so cards match the blog's taxonomy.
- `cards` — ordered. Types: `cover` (hook poster), `figure` (blog figure +
  zh caption; SVGs work too, Chrome rasterizes them), `stats` (big numbers),
  `code` (trimmed snippet from the post's `usage:` block), `end` (link card).
- `note` — `{title, body, tags}`. The generator prints char counts and
  **fails the build** if limits are exceeded. Emoji count as 1 char except
  where XHS counts differently in-app — keep a few chars of headroom.

## Copy calibration

Write for a reader with zero vLLM background: number-forward hooks, one
analogy per concept, no unexplained jargon, and keep the honesty caveats
(draft-PR status, author-reported numbers) — the blog's credibility is the
product. `note.body` is assembled top-to-bottom; keep it under ~900 chars to
leave hashtag headroom.

## Publishing checklist (manual, by design)

XHS has no official personal-account API and unofficial auto-posters risk
限流 — uploading by hand takes ~30 seconds and is the safe route.

1. **One-time**: put `https://hsliuustc0106.github.io/vllm-omni-cookbook` in
   the account's 个人简介.
2. Paste `note.txt` as title + body.
3. Upload the PNGs **in numbered order** — the cover is image 1.
4. After posting, add a **pinned comment** repeating the Pages URL.
5. Do **not** add QR codes to cards — XHS's image moderation flags them as
   off-platform diversion; plain text URLs in bio/cards are tolerated.

## Backfill status

Posts with figures, ready to convert: `pr-6162-svdquant` (pilot, done),
`pr5491`, `diffusion-sequence-parallelism`, `online-quantization-fp8`,
`minimax-h3-modular-pipeline` (SVG-only figures — supported). Posts without
figures (`pr-6476`, `pr-4820`, `pr-6213`) need figures drawn first — see
`blog/assets/figures/pr5491/pr-5491-ar-diffusion-architecture.html` for the
house pattern.
