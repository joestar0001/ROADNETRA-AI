/* RoadNetra AI · shared live-map tiles for every page.
 * Reads window.RN_CONFIG.map (served from frontend/.env at /config.js) and adds the right tile layer:
 *   esri | osm | carto (no key)  ·  maptiler | mapbox | google (MAP_API_KEY)
 * Usage:  <script src="/config.js"></script><script src="/static/rn-map.js"></script>
 *         RNMap.addTiles(leafletMap)            // provider + style from frontend/.env
 *         RNMap.addTiles(leafletMap, "light")   // override style for one map
 */
(function () {
  const CFG = (window.RN_CONFIG && window.RN_CONFIG.map) || { provider: "esri", apiKey: "", style: "dark", center: [28.3589, 76.9387], zoom: 11 };
  const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/";
  const ESRI_ATTR = "Tiles &copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors";
  const OSM_ATTR = "&copy; OpenStreetMap contributors";
  const GOOGLE_DARK = [
    { elementType: "geometry", stylers: [{ color: "#1d2433" }] },
    { elementType: "labels.text.fill", stylers: [{ color: "#8b95a7" }] },
    { elementType: "labels.text.stroke", stylers: [{ color: "#111827" }] },
    { featureType: "road", elementType: "geometry", stylers: [{ color: "#2c3546" }] },
    { featureType: "road.highway", elementType: "geometry", stylers: [{ color: "#3b4a63" }] },
    { featureType: "water", elementType: "geometry", stylers: [{ color: "#0e1626" }] },
    { featureType: "poi", stylers: [{ visibility: "off" }] },
  ];

  function esri(map, style) {
    const layers = [];
    if (style === "satellite") {
      layers.push(L.tileLayer(ESRI + "World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19, attribution: ESRI_ATTR }));
    } else if (style === "streets") {
      layers.push(L.tileLayer(ESRI + "World_Street_Map/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19, attribution: ESRI_ATTR }));
    } else {
      const base = style === "light" ? "Canvas/World_Light_Gray_" : "Canvas/World_Dark_Gray_";
      layers.push(L.tileLayer(ESRI + base + "Base/MapServer/tile/{z}/{y}/{x}", { maxZoom: 16, attribution: ESRI_ATTR }));
      layers.push(L.tileLayer(ESRI + base + "Reference/MapServer/tile/{z}/{y}/{x}", { maxZoom: 16, opacity: 0.85 }));
    }
    layers.forEach(l => l.addTo(map));
    return L.layerGroup(layers);
  }

  let googleReady = null;
  function loadScript(src) {
    return new Promise((ok, fail) => {
      const s = document.createElement("script");
      s.src = src; s.async = true; s.onload = ok; s.onerror = () => fail(new Error("Failed to load " + src));
      document.head.appendChild(s);
    });
  }
  function ensureGoogle(key) {
    if (!googleReady) {
      googleReady = (window.google && window.google.maps ? Promise.resolve()
        : loadScript("https://maps.googleapis.com/maps/api/js?key=" + encodeURIComponent(key) + "&v=weekly"))
        .then(() => (L.gridLayer && L.gridLayer.googleMutant) ? null
          : loadScript("https://unpkg.com/leaflet.gridlayer.googlemutant@0.14.1/dist/Leaflet.GoogleMutant.js"));
    }
    return googleReady;
  }

  function addTiles(map, styleOverride) {
    const provider = CFG.provider || "esri";
    const style = styleOverride || CFG.style || "dark";
    const key = CFG.apiKey || "";
    if (provider === "osm") {
      return L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: OSM_ATTR }).addTo(map);
    }
    if (provider === "carto") {
      const v = style === "light" ? "light_all" : style === "streets" ? "voyager" : "dark_all";
      return L.tileLayer(`https://{s}.basemaps.cartocdn.com/${v}/{z}/{x}/{y}{r}.png`, {
        maxZoom: 20, subdomains: "abcd", attribution: OSM_ATTR + " &copy; CARTO" }).addTo(map);
    }
    if (provider === "maptiler" && key) {
      const v = { dark: "streets-v2-dark", light: "streets-v2-light", streets: "streets-v2", satellite: "hybrid" }[style] || "streets-v2-dark";
      const ext = style === "satellite" ? "jpg" : "png";
      return L.tileLayer(`https://api.maptiler.com/maps/${v}/{z}/{x}/{y}.${ext}?key=${encodeURIComponent(key)}`, {
        tileSize: 512, zoomOffset: -1, maxZoom: 20, crossOrigin: true,
        attribution: '&copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a> ' + OSM_ATTR }).addTo(map);
    }
    if (provider === "mapbox" && key) {
      const v = { dark: "dark-v11", light: "light-v11", streets: "streets-v12", satellite: "satellite-streets-v12" }[style] || "dark-v11";
      return L.tileLayer(`https://api.mapbox.com/styles/v1/mapbox/${v}/tiles/512/{z}/{x}/{y}@2x?access_token=${encodeURIComponent(key)}`, {
        tileSize: 512, zoomOffset: -1, maxZoom: 20, attribution: "&copy; Mapbox " + OSM_ATTR }).addTo(map);
    }
    if (provider === "google" && key) {
      const fallback = esri(map, style);
      ensureGoogle(key).then(() => {
        const type = style === "satellite" ? "hybrid" : "roadmap";
        L.gridLayer.googleMutant({ type, maxZoom: 21, styles: style === "dark" ? GOOGLE_DARK : [] }).addTo(map);
        map.removeLayer(fallback);
      }).catch(e => console.warn("[RoadNetra map] Google Maps unavailable, using Esri tiles:", e));
      return fallback;
    }
    if (provider !== "esri" && !key) console.warn(`[RoadNetra map] MAP_PROVIDER=${provider} needs MAP_API_KEY in frontend/.env; using Esri tiles.`);
    return esri(map, style);
  }

  const PROVIDER_LABEL = { esri: "Esri (free)", osm: "OpenStreetMap", carto: "CARTO", maptiler: "MapTiler", mapbox: "Mapbox", google: "Google Maps" };
  function status() {
    const p = CFG.provider || "esri";
    const needsKey = ["maptiler", "mapbox", "google"].includes(p);
    return { provider: p, label: PROVIDER_LABEL[p] || p, keyLoaded: !!CFG.apiKey, ready: !needsKey || !!CFG.apiKey };
  }

  window.RNMap = { addTiles, status, center: CFG.center, zoom: CFG.zoom, config: CFG };
})();
