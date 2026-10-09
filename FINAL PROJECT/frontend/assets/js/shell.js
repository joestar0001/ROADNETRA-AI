/* RoadNetra AI app shell: sidebar + top bar shared by every screen in /screens.
 *
 * Usage in a page:
 *   <body data-page="cameras" data-title="Camera Grid" data-crumb="Operations">
 *     <main class="rn-app-main"> ...page content... </main>
 *     <script src="../assets/js/shell.js"></script>
 * The script injects the sidebar before <main> and the top bar as the first child of <main>.
 * Exposes window.RN.toast(title, body, kind) where kind is "info" | "ok" | "warn" | "danger".
 */
(function () {
  const D = window.RN_DATA;  // load assets/js/demo-data.js before this file for live counts
  const p1Count = D ? String(D.INCIDENTS.filter(i => i.sev === "P1").length) : "";
  const camCount = D ? String(D.CAMERAS.length) : "";
  const NAV = [
    { group: "Operations", items: [
      { id: "command-center", label: "Command Center", icon: "space_dashboard", href: "command-center.html", count: p1Count, tone: "rn-p1" },
      { id: "cameras", label: "Camera Grid", icon: "videocam", href: "cameras.html", count: camCount, tone: "rn-p4" },
      { id: "incident-detail", label: "Incident Detail", icon: "emergency", href: "incident-detail.html" },
    ]},
    { group: "Intelligence", items: [
      { id: "analytics", label: "Analytics", icon: "monitoring", href: "analytics.html" },
    ]},
    { group: "Field", items: [
      { id: "field-officer", label: "Field Officer App", icon: "smartphone", href: "field-officer.html" },
    ]},
    { group: "System", items: [
      { id: "settings", label: "Settings & Rules", icon: "tune", href: "settings.html" },
    ]},
  ];

  const page = document.body.dataset.page || "";
  const title = document.body.dataset.title || "";
  const crumb = document.body.dataset.crumb || "";

  const navHtml = NAV.map(g => `
    <div class="px-3 pt-5 pb-2 rn-label">${g.group}</div>
    <div class="px-3 space-y-1">
      ${g.items.map(it => `
        <a class="rn-nav-link" href="${it.href}" ${it.id === page ? 'aria-current="page"' : ""}>
          <span class="material-symbols-rounded ${it.id === page ? "icon-fill" : ""}">${it.icon}</span>
          <span>${it.label}</span>
          ${it.count ? `<span class="rn-nav-count ${it.tone}">${it.count}</span>` : ""}
        </a>`).join("")}
    </div>`).join("");

  const sidebar = document.createElement("aside");
  sidebar.id = "rn-sidebar";
  sidebar.className = "rn-sidebar fixed inset-y-0 left-0 z-[700] flex flex-col bg-panel border-r border-line -translate-x-full lg:translate-x-0 transition-transform duration-300";
  sidebar.innerHTML = `
    <a href="../index.html" class="flex items-center gap-3 h-16 px-5 border-b border-line shrink-0">
      <img src="../assets/img/logo-mark.svg" alt="" class="w-9 h-9">
      <div class="leading-tight">
        <div class="font-display font-bold text-[15px] tracking-tight">RoadNetra <span class="text-cyan">AI</span></div>
        <div class="text-[11px] text-dim font-medium">Highway Command</div>
      </div>
    </a>
    <nav class="flex-1 overflow-y-auto pb-4" aria-label="Main">${navHtml}</nav>
    <div class="p-3 border-t border-line space-y-2 shrink-0">
      <div class="rn-card-flat p-3">
        <div class="flex items-center justify-between">
          <span class="rn-label">Model B · Accident</span>
          <span class="rn-badge rn-ok"><span class="rn-dot rn-dot-pulse"></span>Online</span>
        </div>
        <div class="mt-2 text-[12.5px] text-muted">YOLOv8s · trained on 1,815 images</div>
        <div class="mt-2 grid grid-cols-2 gap-2 text-center">
          <div class="rounded-lg bg-canvas/60 border border-line py-1.5"><div class="rn-mono text-[13px] font-semibold text-ok-soft">83.6%</div><div class="text-[10.5px] text-dim">recall</div></div>
          <div class="rounded-lg bg-canvas/60 border border-line py-1.5"><div class="rn-mono text-[13px] font-semibold text-brand-soft">99.5%</div><div class="text-[10.5px] text-dim">precision</div></div>
        </div>
      </div>
      <div class="flex gap-2">
        <a href="../index.html" class="rn-btn rn-btn-sm flex-1"><span class="material-symbols-rounded text-[18px]">public</span>Site</a>
        <a href="login.html" class="rn-btn rn-btn-sm flex-1"><span class="material-symbols-rounded text-[18px]">logout</span>Sign out</a>
      </div>
    </div>`;

  const scrim = document.createElement("div");
  scrim.className = "fixed inset-0 z-[690] bg-canvas/70 backdrop-blur-sm hidden lg:hidden";
  scrim.id = "rn-scrim";

  const main = document.querySelector("main.rn-app-main") || document.querySelector("main");
  document.body.insertBefore(scrim, document.body.firstChild);
  document.body.insertBefore(sidebar, document.body.firstChild);

  const topbar = document.createElement("header");
  topbar.className = "sticky top-0 z-[600] h-16 flex items-center gap-3 px-4 sm:px-6 rn-glass border-x-0 border-t-0";
  topbar.innerHTML = `
    <button class="rn-icon-btn lg:hidden" id="rn-menu" aria-label="Open menu" aria-controls="rn-sidebar" aria-expanded="false">
      <span class="material-symbols-rounded">menu</span>
    </button>
    <div class="min-w-0">
      <div class="text-[11.5px] text-dim font-medium truncate">${crumb}</div>
      <h1 class="font-display font-bold text-[17px] leading-tight tracking-tight truncate">${title}</h1>
    </div>
    <div class="flex-1"></div>
    <label class="relative hidden md:block w-[240px] xl:w-[300px]">
      <span class="material-symbols-rounded absolute left-3 top-1/2 -translate-y-1/2 text-dim text-[19px]">search</span>
      <input class="rn-input !h-[38px] pl-10 text-[13.5px]" placeholder="Search incidents, cameras, KM…" aria-label="Search">
    </label>
    <span class="hidden xl:inline-flex rn-badge rn-p2" title="Dashboard values are sample data for the hackathon demo">
      <span class="material-symbols-rounded text-[14px]">science</span>Demo data
    </span>
    <div class="hidden sm:flex items-center gap-2 h-[38px] px-3 rounded-[10px] border border-line bg-card/60">
      <span class="rn-dot rn-dot-pulse text-ok"></span>
      <span class="rn-mono text-[12.5px] text-muted" id="rn-clock">--:--:--</span>
      <span class="text-[11px] text-dim">IST</span>
    </div>
    <button class="rn-icon-btn relative" id="rn-bell" aria-label="Notifications">
      <span class="material-symbols-rounded">notifications</span>
      <span class="absolute top-2 right-2 w-2 h-2 rounded-full bg-danger ring-2 ring-panel"></span>
    </button>
    <div class="hidden sm:flex items-center gap-2.5 pl-1">
      <div class="w-9 h-9 rounded-full grid place-items-center font-display font-bold text-[13px] text-white bg-gradient-to-br from-brand to-cyan">CR</div>
      <div class="leading-tight hidden xl:block">
        <div class="text-[13px] font-semibold">Control Room</div>
        <div class="text-[11px] text-dim">Duty operator · Desk 04</div>
      </div>
    </div>`;
  main.insertBefore(topbar, main.firstChild);
  main.classList.add("rn-app-main");

  // mobile drawer
  const menuBtn = document.getElementById("rn-menu");
  const setOpen = open => {
    sidebar.classList.toggle("-translate-x-full", !open);
    scrim.classList.toggle("hidden", !open);
    menuBtn.setAttribute("aria-expanded", String(open));
  };
  menuBtn.addEventListener("click", () => setOpen(sidebar.classList.contains("-translate-x-full")));
  scrim.addEventListener("click", () => setOpen(false));
  document.addEventListener("keydown", e => { if (e.key === "Escape") setOpen(false); });

  // live IST clock
  const clock = document.getElementById("rn-clock");
  const tick = () => {
    clock.textContent = new Date().toLocaleTimeString("en-GB", { timeZone: "Asia/Kolkata", hour12: false });
  };
  tick(); setInterval(tick, 1000);

  // toasts
  const wrap = document.createElement("div");
  wrap.className = "rn-toast-wrap";
  wrap.setAttribute("aria-live", "polite");
  document.body.appendChild(wrap);
  const ICON = { info: "info", ok: "check_circle", warn: "warning", danger: "emergency_home" };
  const TONE = { info: "text-brand-soft", ok: "text-ok-soft", warn: "text-warn-soft", danger: "text-danger-soft" };
  function toast(t, body = "", kind = "info", ms = 5000) {
    const el = document.createElement("div");
    el.className = `rn-toast is-${kind}`;
    el.innerHTML = `
      <span class="material-symbols-rounded icon-fill ${TONE[kind]}">${ICON[kind]}</span>
      <div class="min-w-0 flex-1"><div class="text-[13.5px] font-semibold">${t}</div>${body ? `<div class="text-[12.5px] text-muted mt-0.5">${body}</div>` : ""}</div>
      <button class="text-dim hover:text-ink" aria-label="Dismiss"><span class="material-symbols-rounded text-[18px]">close</span></button>`;
    el.querySelector("button").onclick = () => el.remove();
    wrap.appendChild(el);
    if (ms) setTimeout(() => el.remove(), ms);
    return el;
  }
  document.getElementById("rn-bell").addEventListener("click", () =>
    toast("P1 · Multi-vehicle collision", "NH-48 KM 142.6 · confirmed over 3 frames · Ambulance AMB-108 en route", "danger"));

  // Dark basemap for Leaflet maps (Esri, no API key needed). Usage: RN.darkTiles(map)
  function darkTiles(map) {
    const base = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/";
    L.tileLayer(base + "World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}", {
      maxZoom: 16, attribution: "Tiles &copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors",
    }).addTo(map);
    L.tileLayer(base + "World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}", { maxZoom: 16, opacity: 0.85 }).addTo(map);
  }

  window.RN = Object.assign(window.RN || {}, { toast, darkTiles });
})();
