/* Home feature-sidebar filter: shows only cards whose data-feature matches,
   syncs the choice to the URL hash (#feature-<slug>) so filters are shareable. */
(function () {
  "use strict";

  function apply(slug) {
    var any = false;
    document.querySelectorAll(".post-card").forEach(function (card) {
      var match = slug === "all" || card.getAttribute("data-feature") === slug;
      card.hidden = !match;
      if (match) any = true;
    });
    document.querySelectorAll("[data-feature-filter]").forEach(function (btn) {
      btn.classList.toggle("active", btn.getAttribute("data-feature-filter") === slug);
    });
    var list = document.getElementById("post-list");
    var empty = document.getElementById("filter-empty");
    if (!any && slug !== "all") {
      if (!empty && list) {
        empty = document.createElement("p");
        empty.id = "filter-empty";
        empty.className = "filter-empty";
        list.after(empty);
      }
      if (empty) empty.textContent = "No posts under this feature yet.";
    } else if (empty) {
      empty.remove();
    }
  }

  function slugFromHash() {
    var m = location.hash.match(/^#feature-([a-z0-9_]+)$/);
    return m ? m[1] : "all";
  }

  document.addEventListener("DOMContentLoaded", function () {
    var buttons = document.querySelectorAll("[data-feature-filter]");
    if (!buttons.length) return;

    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        var slug = btn.getAttribute("data-feature-filter");
        history.replaceState(null, "", slug === "all" ? "#": "#feature-" + slug);
        apply(slug);
      });
    });

    apply(slugFromHash());
  });

  window.addEventListener("hashchange", function () { apply(slugFromHash()); });
})();
