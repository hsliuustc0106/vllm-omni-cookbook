// Card templates for xiaohongshu notes. One HTML page per image,
// screenshotted at exactly 1242x1656 (XHS 3:4) by generate.mjs.
//
// Design language mirrors the blog: feature colors come from _config.yml,
// figures are the blog's own assets placed full-width on an off-white canvas.

const PAGE_W = 1242;
const PAGE_H = 1656;

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

const BASE_CSS = `
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body {
    width: ${PAGE_W}px; height: ${PAGE_H}px; overflow: hidden;
    background: #faf9fc; color: #201a33;
    font-family: -apple-system, "PingFang SC", "Noto Sans SC", "Microsoft YaHei", sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .mono { font-family: "SF Mono", Menlo, Consolas, monospace; }
  .page {
    width: 100%; height: 100%;
    padding: 72px 76px 40px;
    display: flex; flex-direction: column;
  }
  .kicker {
    display: flex; align-items: center; gap: 18px;
    font-size: 26px; font-weight: 600; letter-spacing: 5px;
    color: __COLOR__;
  }
  .kicker::after { content: ""; flex: 1; height: 1px; background: #ded7ec; }
  h2.card-title {
    margin-top: 26px;
    font-size: 60px; line-height: 1.3; font-weight: 800;
    letter-spacing: 1px; color: #171128;
  }
  .card-sub { margin-top: 14px; font-size: 30px; color: #6b6480; line-height: 1.55; }

  .figzone {
    flex: 1; display: flex; flex-direction: column;
    justify-content: center; align-items: center;
    min-height: 0; margin-top: 10px;
  }
  .figzone img { max-width: 1090px; max-height: 880px; object-fit: contain; }

  .caption {
    margin: 10px 0 30px;
    padding: 26px 32px;
    background: #ffffff;
    border-left: 7px solid __COLOR__;
    border-radius: 0 18px 18px 0;
    box-shadow: 0 2px 14px rgba(60, 30, 120, 0.07);
    font-size: 29px; line-height: 1.72; color: #46405c;
  }
  .caption b { color: __COLOR__; }

  .statrow { display: flex; align-items: center; gap: 36px; padding: 62px 8px; }
  .statrow + .statrow { border-top: 1px solid #e8e2f2; }
  .statval {
    min-width: 430px; text-align: right;
    font-family: "SF Mono", Menlo, monospace;
    font-size: 74px; font-weight: 800; letter-spacing: -2px;
    color: __COLOR__;
  }
  .statdesc { font-size: 33px; line-height: 1.5; color: #38314e; }
  .statdesc .sub { display: block; font-size: 26px; color: #8a83a0; margin-top: 8px; }
  .footnote { margin-top: auto; margin-bottom: 30px; font-size: 25px; color: #8a83a0; line-height: 1.65; }

  .step { margin-top: 38px; display: flex; align-items: center; gap: 18px; }
  .stepchip {
    background: __COLOR__; color: #fff; border-radius: 999px;
    padding: 9px 26px; font-size: 27px; font-weight: 700;
  }
  .steplabel { font-size: 29px; color: #6b6480; }
  .codebox {
    margin-top: 18px; background: #17112b; border-radius: 20px; padding: 30px 34px;
  }
  .codebox pre {
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 23px; line-height: 1.72; color: #d9cdfa;
    white-space: pre-wrap; word-break: break-all;
  }
  .codenote { margin-top: 16px; font-size: 26px; color: #8a83a0; line-height: 1.55; }

  .footer {
    height: 92px; margin-top: auto; flex-shrink: 0;
    border-top: 1px solid #e8e2f2;
    display: flex; align-items: center; justify-content: space-between;
    font-size: 24px; color: #9a93ad;
  }
  .footer b { color: __COLOR__; font-weight: 700; }
`;

function themed(css, cfg) {
  return css.split('__COLOR__').join(cfg.color);
}

function html(cfg, body, extraCss = '') {
  return `<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<style>
${themed(BASE_CSS, cfg)}
${extraCss}
</style></head><body>${body}</body></html>`;
}

function footerStrip(cfg, i, n, dark = false) {
  const style = dark
    ? 'style="color:rgba(255,255,255,.7);border-color:rgba(255,255,255,.25);"'
    : '';
  return `<div class="footer" ${style}>
    <span>${esc(cfg.brand.series)} · <b>${esc(cfg.brand.seriesZh)}</b></span>
    <span class="mono">GitHub · ${esc(cfg.brand.github)}　${i} / ${n}</span>
  </div>`;
}

export function coverHtml(cfg, card) {
  const chips = card.chips
    .map((c) => `<div class="chip"><span class="cv">${esc(c.v)}</span><span class="cl">${esc(c.label)}</span></div>`)
    .join('<div class="chsep"></div>');
  const n = cfg.cards.length;
  const css = `
  * { margin:0; padding:0; box-sizing:border-box; }
  html,body { width:${PAGE_W}px; height:${PAGE_H}px; overflow:hidden;
    font-family:-apple-system,"PingFang SC","Noto Sans SC","Microsoft YaHei",sans-serif;
    -webkit-font-smoothing:antialiased; }
  .page {
    width:${PAGE_W}px; height:${PAGE_H}px;
    padding:90px 96px 64px;
    display:flex; flex-direction:column;
    background:linear-gradient(168deg, ${cfg.color} 0%, ${cfg.colorDeep} 58%, ${cfg.colorDarkest} 100%);
    color:#fff;
  }
  .cover-kicker {
    display:inline-flex; align-self:flex-start;
    border:1px solid rgba(255,255,255,.45); background:rgba(255,255,255,.13);
    border-radius:999px; padding:14px 34px;
    font-size:27px; font-weight:600; letter-spacing:5px;
  }
  h1.cover-h1 { margin-top:54px; font-size:96px; line-height:1.34; font-weight:800; letter-spacing:2px; }
  h1.cover-h1 em { font-style:normal; color:#ffe08a;
    text-decoration:underline; text-decoration-thickness:8px; text-underline-offset:18px;
    text-decoration-color:rgba(255,224,138,.5); }
  .cover-sub { margin-top:44px; font-size:41px; line-height:1.62; color:rgba(255,255,255,.94); }
  .cover-sub b { color:#ffe08a; }
  .chips { margin-top:auto; display:flex; align-items:stretch; gap:34px; padding-bottom:46px; }
  .chip { display:flex; flex-direction:column; gap:12px; }
  .chsep { width:1px; background:rgba(255,255,255,.35); }
  .chip .cv { font-family:"SF Mono",Menlo,monospace; font-size:56px; font-weight:800; letter-spacing:-1px; }
  .chip .cl { font-size:26px; color:rgba(255,255,255,.85); }
  .footer b { color:#ffe08a; }`;
  const body = `
<div class="page">
  <span class="cover-kicker">${esc(card.kicker || cfg.brand.kicker)}</span>
  <h1 class="cover-h1">${card.lines
    .map((l) => (typeof l === 'string' ? esc(l) : `<em>${esc(l.text)}</em>`))
    .join('<br>')}</h1>
  <p class="cover-sub">${card.sub}</p>
  <div class="chips">${chips}</div>
${footerStrip(cfg, 1, n, true)}
</div>`;
  return html(cfg, body, css);
}

export function figureHtml(cfg, card, i, n) {
  const body = `
<div class="page">
  <div class="kicker">${esc(cfg.brand.seriesZh)}</div>
  <h2 class="card-title">${esc(card.title)}</h2>
  ${card.sub ? `<p class="card-sub">${esc(card.sub)}</p>` : ''}
  <div class="figzone"><img src="file://${encodeURI(card.imgAbs)}"></div>
  <div class="caption">${card.tag ? `<b>${esc(card.tag)}</b>　` : ''}${esc(card.caption)}</div>
${footerStrip(cfg, i, n)}
</div>`;
  return html(cfg, body);
}

export function statsHtml(cfg, card, i, n) {
  const rows = card.stats
    .map(
      (s) => `<div class="statrow">
      <div class="statval">${esc(s.v)}</div>
      <div class="statdesc">${esc(s.desc)}${s.sub ? `<span class="sub">${esc(s.sub)}</span>` : ''}</div>
    </div>`,
    )
    .join('\n');
  const body = `
<div class="page" style="padding-right:76px;">
  <div class="kicker">${esc(cfg.brand.seriesZh)}</div>
  <h2 class="card-title">${esc(card.title)}</h2>
  ${card.sub ? `<p class="card-sub">${esc(card.sub)}</p>` : ''}
  <div style="margin-top:36px;">${rows}</div>
  ${card.footnote ? `<p class="footnote">${esc(card.footnote)}</p>` : ''}
${footerStrip(cfg, i, n)}
</div>`;
  return html(cfg, body);
}

export function codeHtml(cfg, card, i, n) {
  const steps = card.steps
    .map(
      (s) => `<div class="step"><span class="stepchip">${esc(s.chip)}</span><span class="steplabel">${esc(s.label)}</span></div>
  <div class="codebox"><pre>${s.code}</pre></div>
  ${s.note ? `<p class="codenote">💡 ${esc(s.note)}</p>` : ''}`,
    )
    .join('\n');
  const body = `
<div class="page">
  <div class="kicker">${esc(cfg.brand.seriesZh)}</div>
  <h2 class="card-title">${esc(card.title)}</h2>
  ${steps}
${footerStrip(cfg, i, n)}
</div>`;
  return html(cfg, body);
}

export function endHtml(cfg, card, i, n) {
  const paths = card.paths
    .map(
      (p, idx) => `<div class="pathrow"><span class="pathnum mono">${idx + 1}</span>
      <div><div class="pathtitle">${esc(p.title)}</div><div class="pathdesc">${esc(p.desc)}</div></div></div>`,
    )
    .join('\n');
  return `<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
html,body { width:${PAGE_W}px; height:${PAGE_H}px; overflow:hidden; background:${cfg.colorDarkest};
  font-family:-apple-system,"PingFang SC","Noto Sans SC","Microsoft YaHei",sans-serif;
  -webkit-font-smoothing:antialiased; color:#fff; }
.wrap { width:100%; height:100%; padding:110px 96px 70px; display:flex; flex-direction:column; }
.pin { font-size:92px; }
h2.endtitle { margin-top:30px; font-size:82px; font-weight:800; line-height:1.3; }
.endsub { margin-top:24px; font-size:33px; line-height:1.7; color:rgba(255,255,255,.85); }
.urlbox {
  margin-top:52px; background:rgba(255,255,255,.10);
  border:1px dashed rgba(255,255,255,.55); border-radius:22px;
  padding:34px 38px; text-align:center;
  font-family:"SF Mono",Menlo,monospace; font-size:29px; line-height:1.65;
  word-break:break-all; color:#ffe08a;
}
.paths { margin-top:64px; display:flex; flex-direction:column; gap:36px; }
.pathrow { display:flex; gap:26px; align-items:flex-start; }
.pathnum { flex-shrink:0; width:54px; height:54px; border-radius:50%;
  background:#ffe08a; color:#3b0764; font-weight:800; font-size:30px;
  display:flex; align-items:center; justify-content:center; }
.pathtitle { font-size:37px; font-weight:700; }
.pathdesc { margin-top:8px; font-size:28px; color:rgba(255,255,255,.8); line-height:1.55; }
.disclaim { margin-top:auto; font-size:24px; line-height:1.7; color:rgba(255,255,255,.6); }
.footline { margin-top:28px; padding-top:24px; border-top:1px solid rgba(255,255,255,.25);
  display:flex; justify-content:space-between; font-size:24px; color:rgba(255,255,255,.7); }
</style></head><body>
<div class="wrap">
  <div class="pin">📌</div>
  <h2 class="endtitle">${esc(card.title)}</h2>
  <p class="endsub">${esc(card.sub)}</p>
  <div class="urlbox">${esc(cfg.brand.site)}</div>
  <div class="paths">${paths}</div>
  <p class="disclaim">${esc(card.disclaimer)}</p>
  <div class="footline"><span>${esc(cfg.brand.series)} · ${esc(cfg.brand.seriesZh)}</span><span class="mono">${i} / ${n}</span></div>
</div>
</body></html>`;
}
