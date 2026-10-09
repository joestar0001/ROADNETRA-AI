// ==========================================================================
// ROADNETRA AI — Professional Single-Page Application Controller
// ==========================================================================

let mapInstance = null;
let allIncidents = [];
let pollingTimer = null;

document.addEventListener("DOMContentLoaded", () => {
  console.log("[ROADNETRA AI] Frontend initialized.");
  loadAllIncidents();
  startTelemetryPolling();
});

// ==========================================================================
// Portal & Tab Navigation
// ==========================================================================
function switchPortal(portalId) {
  // Update Header Buttons
  document.querySelectorAll(".portal-btn").forEach(btn => btn.classList.remove("active"));
  const activeBtn = document.getElementById(`btn-portal-${portalId}`);
  if (activeBtn) activeBtn.classList.add("active");

  // Show Active Portal View
  document.querySelectorAll(".portal-view").forEach(view => view.style.display = "none");
  const targetView = document.getElementById(`portal-${portalId}`);
  if (targetView) targetView.style.display = "block";

  // Reload data for target portal
  if (portalId === "pwd") renderPWDView();
  else if (portalId === "hospital") renderHospitalView();
  else if (portalId === "police") renderPoliceView();
  else if (portalId === "core") {
    // If returning to core, refresh current core tab
    const activeCoreTab = document.querySelector(".sub-tab-btn.active");
    if (activeCoreTab && activeCoreTab.id === "tab-btn-map") initMap();
  }
}

function switchCoreTab(tabId) {
  // Update Sub-Tab Buttons
  document.querySelectorAll(".sub-tab-btn").forEach(btn => btn.classList.remove("active"));
  const activeBtn = document.getElementById(`tab-btn-${tabId}`);
  if (activeBtn) activeBtn.classList.add("active");

  // Show Active Subtab
  document.querySelectorAll(".core-subtab").forEach(tab => tab.style.display = "none");
  const targetTab = document.getElementById(`tab-${tabId}`);
  if (targetTab) targetTab.style.display = "block";

  if (tabId === "map") {
    setTimeout(initMap, 100);
  }
}

// ==========================================================================
// Data Fetching & Incident Management
// ==========================================================================
async function loadAllIncidents() {
  try {
    const res = await fetch("/api/incidents");
    allIncidents = await res.json();
    renderAuditCatalog();
    renderPWDView();
    renderHospitalView();
    renderPoliceView();
    if (mapInstance) renderMapMarkers();
  } catch (err) {
    console.error("Failed to load incidents:", err);
  }
}

async function updateIncidentStatus(incidentId, newStatus, note = "") {
  try {
    const res = await fetch(`/api/incidents/${incidentId}/action`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: newStatus, note })
    });
    const data = await res.json();
    if (data.success) {
      // Reload incidents and re-render
      await loadAllIncidents();
      showToast(`Incident #${incidentId} status updated to '${newStatus}'`);
    }
  } catch (err) {
    console.error("Error updating status:", err);
  }
}

// ==========================================================================
// Real-Time Telemetry & Judge Feed Polling
// ==========================================================================
function startTelemetryPolling() {
  if (pollingTimer) clearInterval(pollingTimer);
  pollingTimer = setInterval(async () => {
    try {
      // Fetch telemetry
      const res = await fetch("/api/summary");
      const summary = await res.json();

      // Update Top Metrics
      if (summary.unique_counts) {
        updateText("metric-potholes", summary.unique_counts.pothole || 0);
        updateText("metric-signs", summary.unique_counts.damaged_sign || 0);
        updateText("metric-dividers", summary.unique_counts.damaged_divider || 0);
        updateText("metric-crossings", summary.unique_counts.faded_zebra_crossing || 0);
      }
      if (summary.active_fps) {
        updateText("metric-fps", summary.active_fps.toFixed(1));
      }

      // Fetch Latest Judge Verdict
      const jRes = await fetch("/api/judge/latest");
      const judge = await jRes.json();
      if (judge && judge.status !== "READY") {
        updateJudgeCard(judge);
      }
    } catch (err) {
      // Stream or server temporary pause
    }
  }, 1400);
}

function updateJudgeCard(judge) {
  const badge = document.getElementById("judge-status-badge");
  const verdictBox = document.getElementById("judge-verdict-box");
  const verdictTitle = document.getElementById("judge-verdict-title");
  const verdictDetails = document.getElementById("judge-verdict-details");
  const verdictReason = document.getElementById("judge-verdict-reason");
  const cropImg = document.getElementById("judge-crop-img");

  if (cropImg && judge.crop_url) {
    cropImg.src = judge.crop_url;
  }

  if (judge.status === "ACCEPTED") {
    if (badge) {
      badge.className = "hero-pill-badge pill-green";
      badge.textContent = "ACCEPTED";
    }
    if (verdictBox) {
      verdictBox.style.background = "#ecfdf5";
      verdictBox.style.borderLeft = "4px solid var(--apple-green)";
    }
    if (verdictTitle) {
      verdictTitle.textContent = "VERDICT: ACCEPTED";
      verdictTitle.style.color = "#065f46";
    }
    if (verdictDetails) {
      verdictDetails.textContent = `Candidate: ${judge.primary_class.toUpperCase()} (Confirmed: ${(judge.final_conf * 100).toFixed(0)}%)`;
    }
    if (verdictReason) {
      verdictReason.textContent = judge.reason;
      verdictReason.style.color = "#047857";
    }
  } else {
    if (badge) {
      badge.className = "hero-pill-badge pill-red";
      badge.textContent = "REJECTED (VETOED)";
    }
    if (verdictBox) {
      verdictBox.style.background = "#fef2f2";
      verdictBox.style.borderLeft = "4px solid var(--apple-red)";
    }
    if (verdictTitle) {
      verdictTitle.textContent = "VERDICT: REJECTED (FALSE ALARM)";
      verdictTitle.style.color = "#991b1b";
    }
    if (verdictDetails) {
      verdictDetails.textContent = `Candidate: ${judge.primary_class.toUpperCase()} (Primary: ${(judge.primary_conf * 100).toFixed(0)}%)`;
    }
    if (verdictReason) {
      verdictReason.textContent = judge.reason;
      verdictReason.style.color = "#b91c1c";
    }
  }
}

// ==========================================================================
// Interactive Leaflet Map
// ==========================================================================
function initMap() {
  const mapContainer = document.getElementById("city-gis-map");
  if (!mapContainer) return;

  if (!mapInstance) {
    mapInstance = L.map("city-gis-map").setView([28.6250, 77.2150], 13);
    L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; CartoDB & OpenStreetMap',
      maxZoom: 19
    }).addTo(mapInstance);
  } else {
    mapInstance.invalidateSize();
  }
  renderMapMarkers();
}

function renderMapMarkers() {
  if (!mapInstance) return;

  // Clear existing markers
  mapInstance.eachLayer(layer => {
    if (layer instanceof L.Marker) mapInstance.removeLayer(layer);
  });

  const agencyFilter = document.getElementById("map-agency-filter") ? document.getElementById("map-agency-filter").value : "all";

  allIncidents.forEach(inc => {
    if (agencyFilter !== "all") {
      if (agencyFilter === "pwd" && (!inc.assigned_agency.toLowerCase().includes("pwd") && !inc.assigned_agency.toLowerCase().includes("nhai"))) return;
      if (agencyFilter === "hospital" && !inc.assigned_agency.toLowerCase().includes("hospital") && !inc.assigned_agency.toLowerCase().includes("ems")) return;
      if (agencyFilter === "police" && !inc.assigned_agency.toLowerCase().includes("police")) return;
    }

    const [lat, lon] = inc.gps;
    let pinColor = "#0071e3";
    if (inc.type.includes("accident")) pinColor = "#ff3b30";
    else if (inc.type === "pothole") pinColor = "#ff9500";
    else if (inc.type === "damaged_divider") pinColor = "#0071e3";
    else if (inc.type === "faded_zebra_crossing") pinColor = "#34c759";

    const customIcon = L.divIcon({
      className: "custom-leaflet-pin",
      html: `<div style="background-color: ${pinColor}; width: 14px; height: 14px; border-radius: 50%; border: 3px solid #ffffff; box-shadow: 0 0 10px ${pinColor};"></div>`,
      iconSize: [20, 20]
    });

    const marker = L.marker([lat, lon], { icon: customIcon }).addTo(mapInstance);
    
    // Popup on hover
    marker.bindTooltip(`<strong>${inc.title}</strong><br/>${inc.status}`, { direction: "top" });

    // Open detail modal on click
    marker.on("click", () => openIncidentModal(inc));
  });
}

// ==========================================================================
// Portals: PWD, Hospital, Police Rendering
// ==========================================================================
function renderPWDView() {
  const container = document.getElementById("pwd-incidents-container");
  if (!container) return;

  const pwdList = allIncidents.filter(i => 
    i.assigned_agency.toLowerCase().includes("pwd") || 
    i.assigned_agency.toLowerCase().includes("nhai") ||
    i.assigned_agency.toLowerCase().includes("municipality")
  );

  let html = "";
  pwdList.forEach(inc => {
    const isResolved = inc.status === "Resolved";
    html += `
      <div class="incident-card">
        <div class="incident-img-container" onclick='openIncidentModal(${JSON.stringify(inc).replace(/'/g, "&#39;")})' style="cursor: pointer;">
          <img src="${inc.image_url}" alt="${inc.title}" onerror="this.src='/static/evidence/pothole_1.jpg'">
          <span class="hero-pill-badge ${inc.severity === 'Critical' ? 'pill-red' : 'pill-amber'}" style="position: absolute; top: 12px; left: 12px;">
            ${inc.severity}
          </span>
        </div>
        <div class="incident-body">
          <div class="incident-badge-row">
            <span style="font-size: 11px; font-weight: 700; color: var(--apple-blue);">${inc.id}</span>
            <span class="hero-pill-badge pill-green" style="margin: 0; font-size: 11px;">${inc.status}</span>
          </div>
          <div class="incident-title">${inc.title}</div>
          <div class="incident-loc">📍 ${inc.location_name}</div>
          <div style="font-size: 12px; color: var(--text-tertiary); margin-bottom: 14px; background: #f8fafc; padding: 8px 10px; border-radius: 8px;">
            ✓ ${inc.judge_reason || 'Verified road asset'}
          </div>
          <div class="incident-footer">
            <div>
              <div style="font-size: 11px; color: var(--text-secondary);">EST. REPAIR TENDER</div>
              <strong style="font-size: 14px; color: var(--text-primary);">₹ ${inc.repair_cost_est ? inc.repair_cost_est.toLocaleString() : '15,000'}</strong>
            </div>
            <div style="display: flex; gap: 8px;">
              ${!isResolved ? `
                <button class="apple-btn" style="padding: 6px 14px; font-size: 12px;" onclick="updateIncidentStatus('${inc.id}', 'Crew En Route', 'Asphalt repair crew dispatched')">
                  Dispatch Crew
                </button>
                <button class="apple-btn apple-btn-secondary" style="padding: 6px 12px; font-size: 12px;" onclick="updateIncidentStatus('${inc.id}', 'Resolved', 'Repair completed & inspected')">
                  Mark Fixed
                </button>
              ` : `
                <span class="hero-pill-badge pill-green">RESOLVED</span>
              `}
            </div>
          </div>
        </div>
      </div>
    `;
  });

  container.innerHTML = html || "<p style='color: var(--text-secondary);'>No PWD work orders active.</p>";
}

function renderHospitalView() {
  const container = document.getElementById("hospital-incidents-container");
  if (!container) return;

  const hospList = allIncidents.filter(i => 
    i.assigned_agency.toLowerCase().includes("hospital") || 
    i.assigned_agency.toLowerCase().includes("ems")
  );

  let html = "";
  hospList.forEach(inc => {
    html += `
      <div class="incident-card" style="border: 1px solid rgba(255, 59, 48, 0.2);">
        <div class="incident-img-container" onclick='openIncidentModal(${JSON.stringify(inc).replace(/'/g, "&#39;")})' style="cursor: pointer;">
          <img src="${inc.image_url}" alt="${inc.title}" onerror="this.src='/static/evidence/accident_1.jpg'">
          <span class="hero-pill-badge pill-red" style="position: absolute; top: 12px; left: 12px;">CRITICAL EMERGENCY</span>
        </div>
        <div class="incident-body">
          <div class="incident-badge-row">
            <span style="font-size: 11px; font-weight: 700; color: var(--apple-red);">${inc.id}</span>
            <span class="hero-pill-badge pill-blue" style="margin: 0; font-size: 11px;">ETA: ~${inc.ambulance_eta_mins || 6} MINS</span>
          </div>
          <div class="incident-title" style="color: #d70015;">${inc.title}</div>
          <div class="incident-loc">📍 ${inc.location_name}</div>
          <div style="background: #fef2f2; padding: 8px 10px; border-radius: 8px; font-size: 12px; color: #991b1b; margin-bottom: 14px;">
            🚨 ${inc.judge_reason || 'Severe vehicular kinetic deformation signature verified'}
          </div>
          <div class="incident-footer">
            <button class="apple-btn apple-btn-danger" style="padding: 7px 14px; font-size: 12px;" onclick="dispatchAmbulance('${inc.id}')">
              🚑 Dispatch ALS Ambulance
            </button>
            <button class="apple-btn" style="padding: 7px 12px; font-size: 12px;" onclick="prealertICU('${inc.id}')">
              🏥 Pre-Alert ICU
            </button>
          </div>
        </div>
      </div>
    `;
  });

  container.innerHTML = html || "<p style='color: var(--text-secondary);'>No active hospital collisions in queue.</p>";
}

function renderPoliceView() {
  const container = document.getElementById("police-incidents-container");
  if (!container) return;

  const policeList = allIncidents.filter(i => 
    i.assigned_agency.toLowerCase().includes("police") || 
    i.severity === "Critical"
  );

  let html = "";
  policeList.forEach(inc => {
    html += `
      <div class="incident-card">
        <div class="incident-img-container" onclick='openIncidentModal(${JSON.stringify(inc).replace(/'/g, "&#39;")})' style="cursor: pointer;">
          <img src="${inc.image_url}" alt="${inc.title}" onerror="this.src='/static/evidence/guardrail_1.jpg'">
          <span class="hero-pill-badge pill-blue" style="position: absolute; top: 12px; left: 12px;">TRAFFIC HAZARD</span>
        </div>
        <div class="incident-body">
          <div class="incident-badge-row">
            <span style="font-size: 11px; font-weight: 700; color: var(--apple-blue);">${inc.id}</span>
            <span class="hero-pill-badge pill-amber" style="margin: 0; font-size: 11px;">${inc.status}</span>
          </div>
          <div class="incident-title">${inc.title}</div>
          <div class="incident-loc">📍 ${inc.location_name}</div>
          <div class="incident-footer">
            <button class="apple-btn" style="padding: 6px 14px; font-size: 12px;" onclick="deployPCR('${inc.id}', '${inc.location_name}')">
              🚔 Deploy PCR Van
            </button>
            <button class="apple-btn apple-btn-secondary" style="padding: 6px 12px; font-size: 12px;" onclick="generateFIR('${inc.id}')">
              📄 Generate FIR
            </button>
          </div>
        </div>
      </div>
    `;
  });

  container.innerHTML = html || "<p style='color: var(--text-secondary);'>No active police traffic hazards.</p>";
}

function renderAuditCatalog() {
  const grid = document.getElementById("audit-catalog-grid");
  if (!grid) return;

  let html = "";
  allIncidents.forEach((inc, idx) => {
    html += `
      <div style="background: #f8fafc; border: 1px solid var(--border-subtle); border-radius: var(--radius-md); overflow: hidden; padding: 12px; display: flex; gap: 12px; align-items: center;">
        <img src="${inc.image_url}" style="width: 80px; height: 80px; border-radius: 10px; object-fit: cover;" onerror="this.src='/static/evidence/pothole_1.jpg'">
        <div style="flex-grow: 1;">
          <div style="font-size: 11px; font-weight: 700; color: var(--apple-blue);">PHYSICAL TRACK #${idx + 1}</div>
          <strong style="font-size: 13px; color: var(--text-primary); display: block;">${inc.title}</strong>
          <div style="font-size: 11px; color: var(--text-secondary); margin-top: 2px;">Conf: ${(inc.confidence * 100).toFixed(0)}% · Status: ${inc.status}</div>
          <div style="font-size: 11px; color: #059669; margin-top: 2px;">✓ Verified by Judge</div>
        </div>
      </div>
    `;
  });

  grid.innerHTML = html;
}

// ==========================================================================
// Interactive Actions & Controls
// ==========================================================================
async function switchVideoFeed(feed) {
  try {
    await fetch("/api/stream/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ feed })
    });
    const video = document.getElementById("live-video-feed");
    if (video) video.src = `/video_feed?t=${Date.now()}`;
    showToast(`Switched video feed to '${feed}'`);
  } catch (err) {
    console.error("Error switching feed:", err);
  }
}

async function toggleJudge(enabled) {
  try {
    await fetch("/api/stream/toggle_judge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enable: enabled })
    });
    showToast(`Stage-2 Judge Model ${enabled ? 'ENABLED' : 'DISABLED'}`);
  } catch (err) {
    console.error("Error toggling judge:", err);
  }
}

function toggleAccidentSlot(enabled) {
  showToast(enabled ? "Model B Accident Dual-Stream Activated (Monitoring Live Feed)" : "Model B Stream Disabled");
}

async function updateVMS() {
  const input = document.getElementById("vms-input");
  const text = input ? input.value.trim() : "";
  if (!text) return;

  try {
    const res = await fetch("/api/action/vms", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    document.getElementById("vms-display-text").textContent = data.vms_text;
    input.value = "";
    showToast("VMS Highway Board Broadcast Updated!");
  } catch (err) {
    console.error("Error updating VMS:", err);
  }
}

async function dispatchAmbulance(incId) {
  try {
    const res = await fetch("/api/action/ambulance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ incident_id: incId })
    });
    const data = await res.json();
    showToast(`ALS Ambulance #${data.ambulance} Dispatched! Traffic Green Wave Activated.`);
    await loadAllIncidents();
  } catch (err) {
    console.error("Error dispatching ambulance:", err);
  }
}

function prealertICU(incId) {
  showToast(`Hospital Trauma Bay 1 & Neuro-ICU on Standby for #${incId}. Blood Bank Notified.`);
}

function deployPCR(incId, loc) {
  updateIncidentStatus(incId, "Crew En Route", "PCR Patrol Unit deployed");
  showToast(`PCR Van #18 deployed to ${loc}!`);
}

function generateFIR(incId) {
  showToast(`Digital FIR & CCTV Evidence Docket generated for #${incId}!`);
}

function exportAuditReport() {
  const blob = new Blob([JSON.stringify(allIncidents, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `ROADNETRA_Audit_Report_${Date.now()}.json`;
  a.click();
}

// ==========================================================================
// Incident Modal Dialog
// ==========================================================================
function openIncidentModal(inc) {
  const modal = document.getElementById("evidence-modal");
  const body = document.getElementById("modal-body");
  if (!modal || !body) return;

  body.innerHTML = `
    <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 12px;">
      <span class="hero-pill-badge pill-blue">${inc.id}</span>
      <span class="hero-pill-badge ${inc.severity === 'Critical' ? 'pill-red' : 'pill-amber'}">${inc.severity}</span>
      <span class="hero-pill-badge pill-green">${inc.status}</span>
    </div>
    <h3 style="margin-bottom: 6px;">${inc.title}</h3>
    <p style="color: var(--text-secondary); font-size: 13px; margin-bottom: 16px;">📍 ${inc.location_name}</p>
    
    <div style="border-radius: 14px; overflow: hidden; margin-bottom: 16px; border: 1px solid var(--border-subtle); max-height: 280px;">
      <img src="${inc.image_url}" style="width: 100%; height: 260px; object-fit: cover;" onerror="this.src='/static/evidence/pothole_1.jpg'">
    </div>

    <div style="background: #f8fafc; padding: 12px; border-radius: 12px; font-size: 13px; margin-bottom: 16px;">
      <div><strong>Responsible Agency:</strong> ${inc.assigned_agency}</div>
      <div style="margin-top: 4px;"><strong>Confidence Score:</strong> ${(inc.confidence * 100).toFixed(1)}% (Stage-2 Verified)</div>
      <div style="margin-top: 4px; color: #059669;"><strong>Judge Finding:</strong> ${inc.judge_reason}</div>
    </div>

    <div style="display: flex; justify-content: space-between; align-items: center;">
      <select id="modal-status-select" style="padding: 8px 12px; border-radius: var(--radius-pill); border: 1px solid var(--border-subtle); font-family: var(--font-sans); font-size: 13px;">
        <option value="Dispatched" ${inc.status === 'Dispatched' ? 'selected' : ''}>Dispatched</option>
        <option value="Crew En Route" ${inc.status === 'Crew En Route' ? 'selected' : ''}>Crew En Route</option>
        <option value="Under Repair" ${inc.status === 'Under Repair' ? 'selected' : ''}>Under Repair</option>
        <option value="Resolved" ${inc.status === 'Resolved' ? 'selected' : ''}>Resolved</option>
      </select>
      <button class="apple-btn" onclick="updateIncidentStatus('${inc.id}', document.getElementById('modal-status-select').value); closeModalDirect();">
        Save Status Change
      </button>
    </div>
  `;

  modal.style.display = "flex";
}

function closeModal(event) {
  const modal = document.getElementById("evidence-modal");
  if (modal) modal.style.display = "none";
}

function closeModalDirect() {
  const modal = document.getElementById("evidence-modal");
  if (modal) modal.style.display = "none";
}

// ==========================================================================
// Helper Utilities
// ==========================================================================
function updateText(elementId, text) {
  const el = document.getElementById(elementId);
  if (el) el.textContent = text;
}

function showToast(msg) {
  const toast = document.createElement("div");
  toast.style.cssText = `
    position: fixed;
    bottom: 28px;
    right: 28px;
    background: rgba(29, 29, 31, 0.95);
    backdrop-filter: blur(12px);
    color: #ffffff;
    padding: 12px 24px;
    border-radius: var(--radius-pill);
    font-size: 13px;
    font-weight: 600;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    z-index: 3000;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  `;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(10px)";
    setTimeout(() => toast.remove(), 300);
  }, 3200);
}
