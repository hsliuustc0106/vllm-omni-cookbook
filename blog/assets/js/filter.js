/* Home sidebar filters: feature (data-feature) and language (data-lang).
   A card is visible when it matches BOTH selections; the feature choice
   syncs to the URL hash (#feature-<slug>) so links are shareable.

   Language is also the site's layout language: selecting 中文 flips the whole
   page (hero, sidebar, header, card chrome) to its Chinese edition via the
   data-i18n / data-i18n-zh attributes, while "all"/English render the English
   edition. English is the default. */
(function () {
  "use strict";

  var feature = "all";
  var lang = "en";   // default language filter: English

  function isZh() {
    return document.documentElement.lang === "zh";
  }

  /* Swap every [data-i18n] element between its served text (English) and its
     data-i18n-zh override, and show the hero edition matching the language. */
  function applyLang() {
    var zh = isZh();
    document.querySelectorAll("[data-i18n]").forEach(function (el) {
      if (!el.hasAttribute("data-i18n-en")) {
        el.setAttribute("data-i18n-en", el.innerHTML);
      }
      var zhHtml = el.getAttribute("data-i18n-zh");
      el.innerHTML = (zh && zhHtml !== null) ? zhHtml : el.getAttribute("data-i18n-en");
    });
    document.querySelectorAll(".hero[lang]").forEach(function (hero) {
      hero.hidden = (hero.getAttribute("lang") !== document.documentElement.lang);
    });
  }

  function apply() {
    var any = false;
    document.querySelectorAll(".post-card").forEach(function (card) {
      var match = (feature === "all" || card.getAttribute("data-feature") === feature) &&
                  (lang === "all" || card.getAttribute("data-lang") === lang);
      card.hidden = !match;
      if (match) any = true;
    });
    document.querySelectorAll("[data-feature-filter]").forEach(function (btn) {
      btn.classList.toggle("active", btn.getAttribute("data-feature-filter") === feature);
    });
    document.querySelectorAll("[data-lang-filter]").forEach(function (btn) {
      btn.classList.toggle("active", btn.getAttribute("data-lang-filter") === lang);
    });
    var list = document.getElementById("post-list");
    var empty = document.getElementById("filter-empty");
    if (!any && (feature !== "all" || lang !== "all")) {
      if (!empty && list) {
        empty = document.createElement("p");
        empty.id = "filter-empty";
        empty.className = "filter-empty";
        list.after(empty);
      }
      if (empty) empty.textContent = "No posts match this filter yet.";
    } else if (empty) {
      empty.remove();
    }
    applyLang();
  }

  function slugFromHash() {
    var m = location.hash.match(/^#feature-([a-z0-9_]+)$/);
    return m ? m[1] : "all";
  }

  function setLang(next) {
    lang = next;
    document.documentElement.lang = (next === "zh") ? "zh" : "en";
    apply();
  }

  document.addEventListener("DOMContentLoaded", function () {
    var fButtons = document.querySelectorAll("[data-feature-filter]");
    var lButtons = document.querySelectorAll("[data-lang-filter]");
    if (!fButtons.length && !lButtons.length) {
      // No home sidebar (post/tag/about pages): still apply the layout
      // language so chrome matches the current page's language.
      applyLang();
      return;
    }

    fButtons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        feature = btn.getAttribute("data-feature-filter");
        history.replaceState(null, "", feature === "all" ? "#" : "#feature-" + feature);
        apply();
      });
    });
    lButtons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        setLang(btn.getAttribute("data-lang-filter"));
      });
    });

    feature = slugFromHash();
    setLang(lang);   // apply the default (English) filter + layout language
  });

  window.addEventListener("hashchange", function () {
    feature = slugFromHash();
    apply();
  });
})();
