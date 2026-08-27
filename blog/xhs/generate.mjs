#!/usr/bin/env node
// Xiaohongshu note generator for the vLLM-Omni blog.
//
//   node generate.mjs <note-name>
//
// Reads blog/xhs/notes/<name>.mjs, screenshots each card's HTML at
// 1242x1656 with headless Chrome, and writes:
//
//   out/<note-name>/01-cover.png … NN-end.png
//   out/<note-name>/note.txt     (title + body + hashtags, ready to paste)
//
// Fails loudly if the title exceeds XHS's 20-char limit or the body
// (hashtags included) exceeds 1000 chars. No npm dependencies.

import { execFileSync } from 'node:child_process';
import { mkdirSync, writeFileSync, rmSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BLOG = path.resolve(HERE, '..');
const ROOT = path.resolve(BLOG, '..');

const CHROME_CANDIDATES = [
  process.env.CHROME_BIN,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
].filter(Boolean);

function charLen(s) {
  return Array.from(s).length;
}

function fail(msg) {
  console.error(`✗ ${msg}`);
  process.exit(1);
}

const name = process.argv[2];
if (!name) fail('usage: node generate.mjs <note-name>   (notes/<name>.mjs)');

const cfgPath = path.join(HERE, 'notes', `${name}.mjs`);
if (!existsSync(cfgPath)) fail(`no such note config: ${cfgPath}`);
const { default: buildCfg } = await import(cfgPath);
const cfg = typeof buildCfg === 'function' ? buildCfg(ROOT) : buildCfg;

// --- validate copy against xiaohongshu limits ------------------------------
const titleLen = charLen(cfg.note.title);
console.log(`title (${titleLen}/20): ${cfg.note.title}`);
if (titleLen > 20) fail('xiaohongshu titles are capped at 20 characters');

const body = cfg.note.body.endsWith('\n') ? cfg.note.body : `${cfg.note.body}\n`;
const tagLine = Array.from(cfg.note.tags, (t) => `#${t}`).join(' ');
const fullBody = body + '\n' + tagLine;
const bodyLen = charLen(fullBody);
console.log(`body  (${bodyLen}/1000 chars incl. tags)`);
if (bodyLen > 1000) fail('body + hashtags exceed the 1000-char cap — trim copy');

// --- sanity-check referenced figures exist ---------------------------------
for (const card of cfg.cards) {
  if (card.type === 'figure' && !existsSync(card.imgAbs)) {
    fail(`figure not found: ${card.imgAbs}`);
  }
}

// --- render ------------------------------------------------------------------
const chromeBin = CHROME_CANDIDATES.find((p) => existsSync(p));
if (!chromeBin) fail('headless Chrome not found; set CHROME_BIN');

const outDir = path.join(HERE, 'out', name);
rmSync(outDir, { recursive: true, force: true });
mkdirSync(path.join(outDir, '.work'), { recursive: true });

const renderers = {
  cover: (c) => import('./lib/template.mjs').then((t) => t.coverHtml(cfg, c)),
  figure: (c, i, n) => import('./lib/template.mjs').then((t) => t.figureHtml(cfg, c, i, n)),
  stats: (c, i, n) => import('./lib/template.mjs').then((t) => t.statsHtml(cfg, c, i, n)),
  code: (c, i, n) => import('./lib/template.mjs').then((t) => t.codeHtml(cfg, c, i, n)),
  end: (c, i, n) => import('./lib/template.mjs').then((t) => t.endHtml(cfg, c, i, n)),
};

let idx = 0;
for (const card of cfg.cards) {
  idx += 1;
  const render = renderers[card.type];
  if (!render) fail(`unknown card type "${card.type}" in ${name}`);
  const html = await render(card, idx, cfg.cards.length);
  const htmlPath = path.join(outDir, '.work', `card-${String(idx).padStart(2, '0')}.html`);
  writeFileSync(htmlPath, html);
  const pngPath = path.join(outDir, `${String(idx).padStart(2, '0')}-${card.type}.png`);
  try {
    execFileSync(
      chromeBin,
      [
        '--headless=new',
        '--disable-gpu',
        '--hide-scrollbars',
        '--force-device-scale-factor=1',
        '--window-size=1242,1656',
        '--default-background-color=00000000',
        '--virtual-time-budget=3000',
        `--screenshot=${pngPath}`,
        `file://${htmlPath}`,
      ],
      { stdio: ['ignore', 'ignore', 'pipe'] },
    );
    console.log(`✓ ${path.relative(ROOT, pngPath)} [${card.type}] ${card.title ?? ''}`);
  } catch (err) {
    fail(`chrome screenshot failed on card ${idx}: ${err.stderr?.toString().slice(0, 400)}`);
  }
}

writeFileSync(path.join(outDir, 'note.txt'), `${cfg.note.title}\n\n${fullBody}`);
console.log(`\n${cfg.cards.length} cards + note.txt → ${path.relative(ROOT, outDir)}/`);
