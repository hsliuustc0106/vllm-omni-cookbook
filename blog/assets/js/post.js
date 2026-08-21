/* Post-page enhancements: auto TOC with scroll-spy, reading progress bar,
   usage-cookbook tabs, and copy buttons for code blocks. Everything here is
   progressive — the page stays readable with JavaScript disabled. */
(function () {
  "use strict";

  /* -- clipboard ------------------------------------------------------------ */

  function copyText(text, button) {
    var done = function () {
      button.textContent = "Copied";
      window.setTimeout(function () { button.textContent = "Copy"; }, 1400);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text, done); });
    } else {
      fallbackCopy(text, done);
    }
  }

  function fallbackCopy(text, done) {
    var area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    try { document.execCommand("copy"); } catch (e) { /* no clipboard at all */ }
    area.remove();
    done();
  }

  function initCodeCopy() {
    document.querySelectorAll(".post-content pre").forEach(function (pre) {
      if (pre.closest(".usage-panel")) return; /* panels carry their own button */
      var text = pre.innerText;
      var button = document.createElement("button");
      button.type = "button";
      button.className = "code-copy copy-btn";
      button.textContent = "Copy";
      button.addEventListener("click", function () { copyText(text, button); });
      pre.appendChild(button);
    });
    document.querySelectorAll(".usage-copy").forEach(function (button) {
      var panel = button.closest(".usage-panel");
      var code = panel && panel.querySelector("pre");
      if (!code) return;
      button.addEventListener("click", function () { copyText(code.innerText, button); });
    });
  }

  /* -- TOC + scroll-spy ------------------------------------------------------ */

  function initToc() {
    var nav = document.querySelector(".post-toc");
    var list = nav && nav.querySelector(".post-toc-list");
    if (!nav || !list) return;
    var headings = Array.prototype.slice.call(
      document.querySelectorAll(".post-content > h2[id]")
    );
    if (headings.length === 0) return;

    headings.forEach(function (h) {
      var li = document.createElement("li");
      var a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.textContent;
      li.appendChild(a);
      list.appendChild(li);
    });
    nav.hidden = false;

    if (!("IntersectionObserver" in window)) return;
    var links = Array.prototype.slice.call(list.querySelectorAll("a"));
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        links.forEach(function (link) {
          link.classList.toggle("active", link.getAttribute("href") === "#" + entry.target.id);
        });
      });
    }, { rootMargin: "-15% 0px -75% 0px" });
    headings.forEach(function (h) { observer.observe(h); });
  }

  /* -- usage cookbook tabs ---------------------------------------------------- */

  function initUsageTabs() {
    document.querySelectorAll(".usage-cookbook").forEach(function (root) {
      var tabs = Array.prototype.slice.call(root.querySelectorAll(".usage-tab"));
      tabs.forEach(function (tab) {
        tab.addEventListener("click", function () {
          var target = tab.getAttribute("aria-controls");
          tabs.forEach(function (t) {
            t.setAttribute("aria-selected", String(t === tab));
          });
          root.querySelectorAll(".usage-panel").forEach(function (panel) {
            panel.hidden = panel.id !== target;
          });
        });
      });
    });
  }

  /* -- reading progress ------------------------------------------------------- */

  function initProgress() {
    var bar = document.createElement("div");
    bar.className = "reading-progress";
    bar.setAttribute("aria-hidden", "true");
    var fill = document.createElement("span");
    bar.appendChild(fill);
    document.body.appendChild(bar);
    var update = function () {
      var max = document.documentElement.scrollHeight - window.innerHeight;
      fill.style.width = (max > 0 ? (window.scrollY / max) * 100 : 0) + "%";
    };
    document.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    update();
  }

  initCodeCopy();
  initToc();
  initUsageTabs();
  initProgress();
})();
