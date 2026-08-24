/* Home sidebar filters: feature (data-feature) and language (data-lang).
   A card is visible when it matches BOTH selections; the feature choice
   syncs to the URL hash (#feature-<slug>) so links are shareable. */
(function () {
  "use strict";

  var feature = "all";
  var lang = "all";

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
  }

  function slugFromHash() {
    var m = location.hash.match(/^#feature-([a-z0-9_]+)$/);
    return m ? m[1] : "all";
  }

  document.addEventListener("DOMContentLoaded", function () {
    var fButtons = document.querySelectorAll("[data-feature-filter]");
    var lButtons = document.querySelectorAll("[data-lang-filter]");
    if (!fButtons.length && !lButtons.length) return;

    fButtons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        feature = btn.getAttribute("data-feature-filter");
        history.replaceState(null, "", feature === "all" ? "#" : "#feature-" + feature);
        apply();
      });
    });
    lButtons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        lang = btn.getAttribute("data-lang-filter");
        apply();
      });
    });

    feature = slugFromHash();
    apply();
  });

  window.addEventListener("hashchange", function () {
    feature = slugFromHash();
    apply();
  });
})();
