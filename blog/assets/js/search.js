/* Client-side post search: ⌘J / Ctrl+J opens a modal that filters
   search.json (title, summary, tags) with arrow-key navigation. */
(function () {
  "use strict";

  var INDEX_URL = window.VOMNI_SEARCH_URL;
  var MAX_RESULTS = 12;
  var index = null;      // cached posts
  var entries = [];      // current results
  var active = -1;
  var overlay, input, list, statusEl;

  function fetchIndex(cb) {
    if (index) return cb();
    var req = new XMLHttpRequest();
    req.open("GET", INDEX_URL, true);
    req.onload = function () {
      try { index = JSON.parse(req.responseText); } catch (e) { index = []; }
      cb();
    };
    req.onerror = function () { index = []; cb(); };
    req.send();
  }

  function build() {
    overlay = document.createElement("div");
    overlay.className = "search-overlay";
    overlay.hidden = true;
    overlay.innerHTML =
      '<div class="search-modal" role="dialog" aria-modal="true" aria-label="Search posts">' +
      '  <div class="search-box">' +
      '    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">' +
      '      <circle cx="11" cy="11" r="7"></circle>' +
      '      <line x1="21" y1="21" x2="16.65" y2="16.65"></line>' +
      '    </svg>' +
      '    <input type="search" placeholder="Search posts…" aria-label="Search posts" autocomplete="off" spellcheck="false">' +
      '    <kbd>esc</kbd>' +
      '  </div>' +
      '  <ul class="search-results" role="listbox"></ul>' +
      '  <div class="search-status" role="status" aria-live="polite"></div>' +
      '</div>';
    document.body.appendChild(overlay);

    var modal = overlay.querySelector(".search-modal");
    modal.addEventListener("click", function (e) { e.stopPropagation(); });
    overlay.addEventListener("click", close);

    input = overlay.querySelector("input");
    list = overlay.querySelector(".search-results");
    statusEl = overlay.querySelector(".search-status");

    input.addEventListener("input", function () { render(input.value); });
    input.addEventListener("keydown", onKeydown);
  }

  function open() {
    if (!overlay) build();
    overlay.hidden = false;
    document.documentElement.classList.add("search-open");
    input.value = "";
    render("");
    setTimeout(function () { input.focus(); }, 0);
  }

  function close() {
    if (!overlay) return;
    overlay.hidden = true;
    document.documentElement.classList.remove("search-open");
  }

  function isOpen() { return overlay && !overlay.hidden; }

  function onKeydown(e) {
    if (e.key === "Escape") { close(); return; }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!entries.length) return;
      active = e.key === "ArrowDown"
        ? (active + 1) % entries.length
        : (active - 1 + entries.length) % entries.length;
      paintActive();
    } else if (e.key === "Enter") {
      if (entries[active]) window.location.href = entries[active].url;
    }
  }

  function score(post, q) {
    var title = post.title.toLowerCase();
    var tags = post.tags.join(" ").toLowerCase();
    if (title.indexOf(q) !== -1) return 3;
    if (tags.indexOf(q) !== -1) return 2;
    if (post.summary.toLowerCase().indexOf(q) !== -1) return 1;
    return 0;
  }

  function render(query) {
    var q = query.trim().toLowerCase();
    entries = [];
    if (q && index) {
      entries = index
        .map(function (p) { return { p: p, s: score(p, q) }; })
        .filter(function (x) { return x.s > 0; })
        .sort(function (a, b) { return b.s - a.s; })
        .map(function (x) { return x.p; })
        .slice(0, MAX_RESULTS);
    }
    active = entries.length ? 0 : -1;

    list.innerHTML = entries.map(function (p, i) {
      var tags = p.tags.map(function (t) { return "#" + t; }).join(" ");
      return '<li role="option" id="search-result-' + i + '">' +
        '  <a href="' + p.url + '">' +
        '    <span class="search-result-title">' + escapeHtml(p.title) + "</span>" +
        '    <span class="search-result-meta">' + escapeHtml(p.date) + (tags ? " · " + escapeHtml(tags) : "") + "</span>" +
        "  </a></li>";
    }).join("");

    statusEl.textContent = !q ? ""
      : entries.length ? entries.length + " result" + (entries.length > 1 ? "s" : "")
      : "No results found.";
    paintActive();
  }

  function paintActive() {
    var items = list.querySelectorAll("li");
    items.forEach(function (li, i) {
      li.classList.toggle("active", i === active);
      if (i === active) li.setAttribute("aria-selected", "true");
      else li.removeAttribute("aria-selected");
    });
    var el = list.querySelector("li.active");
    if (el && el.scrollIntoView) el.scrollIntoView({ block: "nearest" });
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-search-toggle]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        fetchIndex(function () { isOpen() ? close() : open(); });
      });
    });
  });

  document.addEventListener("keydown", function (e) {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "j") {
      e.preventDefault();
      fetchIndex(function () { isOpen() ? close() : open(); });
    }
  });
})();
