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
  const D = window.RN_DATA;  // /api/rn-data.js (live) or assets/js/demo-data.js (offline fallback), loaded before this file
  const LIVE = !!(D && D.LIVE);
  const INC = (D && D.INCIDENTS) || [];
  const isOpen = i => String(i.status || "").toLowerCase() !== "resolved";
  const p1Count = D ? String(INC.filter(i => i.sev === "P1" && (!LIVE || isOpen(i))).length) : "";
  const camCount = D ? String((D.CAMERAS || []).length) : "";
  const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
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
          ${it.count ? `<span class="rn-nav-count ${it.tone}" data-nav-count="${it.id}">${it.count}</span>` : ""}
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
    ${LIVE ? `<span class="hidden xl:inline-flex rn-badge rn-ok" id="rn-data-badge" title="Synced with the RoadNetra backend">
      <span class="rn-dot rn-dot-pulse"></span>Live data
    </span>` : `<span class="hidden xl:inline-flex rn-badge rn-p2" id="rn-data-badge" title="Dashboard values are sample data for the hackathon demo">
      <span class="material-symbols-rounded text-[14px]">science</span>Demo data
    </span>`}
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
  // toast(title, body, kind, ms, opts) · opts.href makes the toast body clickable and adds a "View" link
  function toast(t, body = "", kind = "info", ms = 5000, opts = {}) {
    const el = document.createElement("div");
    const href = opts && opts.href;
    el.className = `rn-toast is-${kind}`;
    el.innerHTML = `
      <span class="material-symbols-rounded icon-fill ${TONE[kind] || TONE.info}">${ICON[kind] || ICON.info}</span>
      <div class="min-w-0 flex-1 ${href ? "cursor-pointer" : ""}" data-toast-body><div class="text-[13.5px] font-semibold">${t}</div>${body ? `<div class="text-[12.5px] text-muted mt-0.5">${body}</div>` : ""}${href ? `<a href="${esc(href)}" class="inline-flex items-center gap-0.5 mt-1 text-[12.5px] font-semibold text-brand-soft hover:underline">${esc(opts.linkText || "View")}<span class="material-symbols-rounded text-[16px]">arrow_forward</span></a>` : ""}</div>
      <button class="text-dim hover:text-ink" aria-label="Dismiss"><span class="material-symbols-rounded text-[18px]">close</span></button>`;
    el.querySelector('button[aria-label="Dismiss"]').onclick = () => el.remove();
    if (href) el.querySelector("[data-toast-body]").addEventListener("click", e => { if (!e.target.closest("a")) location.href = href; });
    wrap.appendChild(el);
    if (ms) setTimeout(() => el.remove(), ms);
    return el;
  }
  const detailHref = id => "incident-detail.html?id=" + encodeURIComponent(id);
  const pct = c => (c == null || isNaN(c) ? "—" : Math.round(+c * 100) + "%");

  // bell: newest open P1/P2 incident from RN_DATA (live incidents are sorted newest first)
  document.getElementById("rn-bell").addEventListener("click", () => {
    const list = INC.filter(i => i.sev === "P1" || i.sev === "P2");
    const i = list.find(isOpen) || list[0];
    if (!i) { toast("No P1 / P2 incidents", LIVE ? "Nothing urgent from the RoadNetra backend right now" : "Nothing urgent in the demo data", "ok"); return; }
    const where = [i.road && i.road !== "—" ? i.road + (i.km && i.km !== "—" ? " KM " + i.km : "") : "", i.place].filter(Boolean).join(" · ");
    toast(`${esc(i.sev)} · ${esc(i.title)}`, `${esc(where)} · conf ${pct(i.conf)} · ${esc(i.status)}`,
      i.sev === "P1" ? "danger" : "warn", 8000, { href: detailHref(i.id) });
  });

  // Dark basemap for Leaflet maps (Esri, no API key needed). Usage: RN.darkTiles(map)
  function darkTiles(map) {
    if (window.RNMap && typeof window.RNMap.addTiles === "function") return window.RNMap.addTiles(map);
    const base = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/";
    L.tileLayer(base + "World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}", {
      maxZoom: 16, attribution: "Tiles &copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors",
    }).addTo(map);
    L.tileLayer(base + "World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}", { maxZoom: 16, opacity: 0.85 }).addTo(map);
  }

  window.RN = Object.assign(window.RN || {}, { toast, darkTiles });

  // ---------- live sync with the RoadNetra backend (only when RN_DATA came from /api/rn-data.js) ----------
  const NEW_KEY = "rn-new-incident-toasts";
  try {  // re-show the "New incident" toasts once after a soft reload
    const pending = JSON.parse(sessionStorage.getItem(NEW_KEY) || "[]");
    sessionStorage.removeItem(NEW_KEY);
    pending.slice(0, 3).forEach(p => toast(p.t, p.b, p.kind, 9000, { href: detailHref(p.id) }));
  } catch (e) { /* storage unavailable */ }
  if (!LIVE || !window.fetch) return;

  const NO_RELOAD = ["incident-detail", "field-officer", "settings"].includes(page);
  const POLL_MS = (window.RN_CONFIG && window.RN_CONFIG.stream && +window.RN_CONFIG.stream.incidentPollMs) || 3000;
  const KIND_LABEL = { severe_accident: "Severe accident", pothole: "Pothole", damaged_sign: "Damaged traffic sign",
    damaged_divider: "Damaged divider", faded_zebra_crossing: "Faded zebra crossing" };
  const sigOf = list => list.map(i => i.id + ":" + i.status).sort().join("|");
  const known = new Set(INC.map(i => i.id));
  let signature = sigOf(INC);
  let reloadPending = false, staleToast = null, lastInteraction = 0;
  ["pointerdown", "keydown", "scroll", "wheel", "touchstart"].forEach(ev =>
    window.addEventListener(ev, () => { lastInteraction = Date.now(); }, { passive: true, capture: true }));

  // a drawer / modal is open or a form field has focus: do not reload under the operator
  const busy = () => {
    const a = document.activeElement;
    if (a && a.matches && a.matches("input, textarea, select, [contenteditable=''], [contenteditable='true']")) return true;
    return [...document.querySelectorAll(".rn-drawer, .rn-overlay, [role=dialog], [aria-modal=true]")]
      .some(el => !el.classList.contains("hidden") && el.offsetParent !== null);
  };
  function tryReload() {
    if (!reloadPending || NO_RELOAD) return;
    if (Date.now() - lastInteraction < 15000 || busy()) return;
    location.reload();
  }
  setInterval(tryReload, 2000);

  function setP1(n) {
    const el = document.querySelector('[data-nav-count="command-center"]');
    if (el) el.textContent = String(n);
  }
  async function poll() {
    if (document.hidden) return;
    let list;
    try {
      const r = await fetch("/api/incidents", { cache: "no-store" });
      if (!r.ok) return;
      list = await r.json();
    } catch (e) { return; }
    if (!Array.isArray(list)) return;
    setP1(list.filter(i => i.priority === "P1" && isOpen(i)).length);
    const fresh = list.filter(i => i.id && !known.has(i.id));
    const notes = fresh.slice(0, 3).map(i => ({
      id: i.id, kind: i.priority === "P1" ? "danger" : "warn",
      t: esc(`New ${i.priority || "P3"} · ${KIND_LABEL[i.type] || i.type || "Incident"}`),
      b: esc(`${i.location_name || i.place || "Unknown location"} · ${i.confidence_pct != null ? Math.round(i.confidence_pct) : Math.round((i.confidence || 0) * 100)}% · ${i.assigned_agency || "Routing"}`),
    }));
    notes.forEach(n => toast(n.t, n.b, n.kind, 9000, { href: detailHref(n.id) }));
    fresh.forEach(i => known.add(i.id));
    const sig = sigOf(list);
    if (sig === signature) return;
    signature = sig;
    if (NO_RELOAD) return;
    if (notes.length) {
      try {
        const prev = JSON.parse(sessionStorage.getItem(NEW_KEY) || "[]");
        sessionStorage.setItem(NEW_KEY, JSON.stringify(notes.concat(prev).slice(0, 3)));
      } catch (e) { /* storage unavailable */ }
    }
    reloadPending = true;
    if (!staleToast || !document.body.contains(staleToast)) {
      staleToast = toast("Incident data updated", "This view refreshes once you have been idle for 15 s.", "info", 0,
        { href: location.pathname + location.search, linkText: "Refresh now" });
    }
    tryReload();
  }
  setInterval(poll, Math.max(1000, POLL_MS));
})();
