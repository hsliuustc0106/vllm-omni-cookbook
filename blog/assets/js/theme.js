/* Dark-mode toggle: persisted choice overrides the OS-following `auto` skin.
   `data-theme` on <html> is applied by an inline script in <head> before paint;
   here we only handle clicks and icon state. */
(function () {
  "use strict";

  var root = document.documentElement;
  var media = window.matchMedia("(prefers-color-scheme: dark)");

  function effectiveTheme() {
    var t = root.getAttribute("data-theme");
    if (t === "light" || t === "dark") return t;
    return media.matches ? "dark" : "light";
  }

  function updateIcons() {
    var dark = effectiveTheme() === "dark";
    root.setAttribute("data-theme-effective", dark ? "dark" : "light");
  }

  function toggle() {
    var next = effectiveTheme() === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) { /* ignore */ }
    updateIcons();
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-theme-toggle]").forEach(function (btn) {
      btn.addEventListener("click", toggle);
    });
    updateIcons();
  });

  media.addEventListener("change", updateIcons);
})();
