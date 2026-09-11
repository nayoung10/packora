(function () {
  "use strict";

  const configuredUrl = String(window.PACKORA_APP_URL || "").trim();
  const appLinks = document.querySelectorAll("[data-packora-app-link]");
  if (!configuredUrl) return;

  const appUrl = new URL(configuredUrl, window.location.href).href;
  appLinks.forEach((link) => {
    link.href = appUrl;
    link.classList.remove("is-unavailable");
    link.classList.add("is-available");
    link.removeAttribute("aria-disabled");
    const label = link.querySelector(".app-availability-label");
    if (label) label.remove();
  });
})();
