/* RoadNetra portal runtime (PWD / Hospital / Police).
   One poll of /api/operations per interval returns the portal's incidents plus every figure computed
   from them on the server. Heavy redraws (lists, maps) run only when the incident store changed. */
(function () {
  const POLL_MS = (window.RN_CONFIG && RN_CONFIG.stream && RN_CONFIG.stream.incidentPollMs) || 3000;

  const css = document.createElement('style');
  css.textContent = `
    .ic.fresh{box-shadow:0 0 0 2px var(--red),0 6px 24px rgba(255,59,48,.25);animation:rnPulse 1.6s ease-in-out 3}
    .ic.done{opacity:.6}
    @keyframes rnPulse{50%{box-shadow:0 0 0 5px rgba(255,59,48,.35)}}
    .rn-live{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--mut)}
    .rn-live i{width:8px;height:8px;border-radius:50%;background:var(--grn);display:inline-block}
    .rn-live.off i{background:var(--red)}
    .rn-pin{animation:rnPin 1.2s ease-in-out infinite}
    @keyframes rnPin{50%{stroke-width:8;stroke-opacity:.35}}`;
  document.head.appendChild(css);

  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const inr = n => '₹ ' + Math.round(n || 0).toLocaleString('en-IN');
  const pct = v => (v == null ? '—' : v.toFixed(1) + '%');
  const clock = s => { s = Math.max(0, Math.round(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };
  const time = iso => { try { return new Date(iso).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' }); } catch (e) { return ''; } };
  const ago = iso => {
    const s = (Date.now() - new Date(iso).getTime()) / 1000;
    if (!isFinite(s)) return '';
    if (s < 60) return 'just now';
    if (s < 3600) return Math.floor(s / 60) + ' min ago';
    if (s < 86400) return Math.floor(s / 3600) + ' h ' + Math.floor(s % 3600 / 60) + ' m ago';
    return Math.floor(s / 86400) + ' d ago';
  };
  const isFresh = i => i.source !== 'seed' && (Date.now() - new Date(i.detected_at).getTime()) < 90000;

  function beep() {
    try {
      const ctx = beep.ctx || (beep.ctx = new (window.AudioContext || window.webkitAudioContext)());
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.frequency.value = 880; g.gain.value = 0.05; o.connect(g); g.connect(ctx.destination);
      o.start(); o.stop(ctx.currentTime + 0.18);
    } catch (e) { /* audio blocked until the user interacts with the page */ }
  }

  /* render(data, {changed, fresh}) is called every poll; `changed` is true when incidents / live state moved. */
  function start(portal, render) {
    let lastKey = null, known = null, busy = false, badge = document.getElementById('rnLive');
    async function tick() {
      if (busy) return;
      busy = true;
      try {
        const r = await fetch('/api/operations?portal=' + portal, { cache: 'no-store' });
        if (!r.ok) throw new Error(r.status);
        const d = await r.json();
        const fresh = known ? d.incidents.filter(i => !known.has(i.id)) : [];
        known = new Set(d.incidents.map(i => i.id));
        // lists also refresh every 30 s so countdowns / "x min ago" stay current
        const key = [d.version, d.live.green_wave_active, d.live.vms_text, Math.floor(Date.now() / 30000)].join('|');
        const changed = key !== lastKey;
        lastKey = key;
        render(d, { changed, fresh });
        if (fresh.length) beep();
        if (badge) { badge.classList.remove('off'); badge.lastChild.textContent = ' Live · synced ' + time(d.time); }
      } catch (e) {
        if (badge) { badge.classList.add('off'); badge.lastChild.textContent = ' Offline · retrying'; }
      } finally { busy = false; }
    }
    tick();
    setInterval(tick, Math.max(1000, POLL_MS));
    return tick;
  }

  /* Leaflet layer that keeps one marker per id and updates it in place (no full rebuild per poll). */
  function layer(map) {
    const group = L.layerGroup().addTo(map), byId = new Map();
    let fitted = false;
    return {
      sync(points, focusFresh) {
        const seen = new Set();
        points.forEach(p => {
          if (p.lat == null || p.lng == null) return;
          seen.add(p.id);
          let m = byId.get(p.id);
          const style = { radius: p.radius || 8, color: p.stroke || '#fff', weight: 2.5, fillColor: p.color, fillOpacity: p.opacity ?? 0.9, className: p.pulse ? 'rn-pin' : '' };
          if (!m) {
            m = p.line ? L.polyline(p.line, { color: p.color, weight: 4, dashArray: '8 8', opacity: 0.85 }) : L.circleMarker([p.lat, p.lng], style);
            m.addTo(group);
            byId.set(p.id, m);
          } else if (!p.line) {
            m.setLatLng([p.lat, p.lng]); m.setStyle(style); m.setRadius(style.radius);
            if (m._path) m._path.classList.toggle('rn-pin', !!p.pulse);
          } else { m.setLatLngs(p.line); m.setStyle({ color: p.color }); }
          if (p.popup) { if (m.getPopup()) m.setPopupContent(p.popup); else m.bindPopup(p.popup); }
          if (p.tip) { if (m.getTooltip()) m.setTooltipContent(p.tip); else m.bindTooltip(p.tip); }
        });
        byId.forEach((m, id) => { if (!seen.has(id)) { group.removeLayer(m); byId.delete(id); } });
        const pts = points.filter(p => !p.line && p.lat != null).map(p => [p.lat, p.lng]);
        if (!fitted && pts.length && map.getContainer().offsetWidth > 0) {
          map.fitBounds(pts, { padding: [40, 40], maxZoom: 13 });
          fitted = true;
        }
        if (focusFresh && focusFresh.lat != null && map.getContainer().offsetWidth > 0) {
          map.flyTo([focusFresh.lat, focusFresh.lng], 14, { duration: 0.8 });
          const m = byId.get(focusFresh.id); if (m && m.openPopup) setTimeout(() => m.openPopup(), 850);
        }
      },
      focus(id, zoom) { const m = byId.get(id); if (m && m.getLatLng) { map.setView(m.getLatLng(), zoom || 14); m.openPopup && m.openPopup(); } },
      refit() { fitted = false; },
    };
  }

  window.RNPortal = { start, layer, esc, inr, pct, clock, time, ago, isFresh, beep };
})();
