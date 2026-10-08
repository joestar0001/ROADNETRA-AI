/* Shared demo data for every RoadNetra screen. Incidents, cameras and units are SAMPLE data for the
 * hackathon demo. MODEL holds the real test-set results of the trained accident model (Model B). */
window.RN_DATA = (function () {
  const IMG = "../assets/img/";

  // Real results of models/accident_best.pt on the 273 held-out test images (see training/colab_model_b.py).
  const MODEL = {
    name: "Model B · Severe Accident",
    arch: "YOLOv8s",
    imgsz: 640,
    epochs: 100,
    bestEpoch: 90,
    bestConf: 0.47,
    pipelineConf: 0.55,
    temporalFrames: 3,
    dataset: { total: 1815, train: 1270, val: 272, test: 273, accident: 1585, negative: 230 },
    test: { mAP50: 0.830, mAP5095: 0.502, precision: 0.995, recall: 0.836, specificity: 0.966, accuracy: 0.850, f1: 0.909,
            falseAlarm: 0.034, TP: 204, FP: 1, FN: 40, TN: 28 },
    latency: { t4GpuMs: 2.5, laptopCpuMs: 189 },
  };

  const SEVERITY = {
    P1: { label: "P1 Critical", cls: "rn-p1", color: "#EF4444" },
    P2: { label: "P2 High", cls: "rn-p2", color: "#F59E0B" },
    P3: { label: "P3 Medium", cls: "rn-p3", color: "#0EA5E9" },
    P4: { label: "P4 Low", cls: "rn-p4", color: "#94A3B8" },
  };

  // type: "accident" (Model B, 30 FPS stream) | "pothole" / "damage" (Model A, sampled 1 FPS stream)
  const INCIDENTS = [
    { id: "RN-2026-004821", sev: "P1", type: "accident", title: "Multi-vehicle collision", road: "NH-48", km: "142.6",
      place: "Manesar Bypass, northbound", lat: 28.3589, lng: 76.9387, conf: 0.90, frames: 7, cam: "CAM-084",
      authority: "112 Emergency · Highway Police", unit: "AMB-108", eta: "2m 40s", status: "Dispatched",
      detected: "08:42:11", ago: "4 min", img: IMG + "model-junction-crash.jpg" },
    { id: "RN-2026-004826", sev: "P1", type: "accident", title: "Vehicle rollover", road: "NH-48", km: "151.2",
      place: "Bilaspur Chowk, southbound", lat: 28.2925, lng: 76.8786, conf: 0.89, frames: 5, cam: "CAM-091",
      authority: "112 Emergency · Highway Police", unit: "PCR-12", eta: "5m 10s", status: "Responding",
      detected: "08:37:52", ago: "9 min", img: IMG + "model-day-rollover.jpg" },
    { id: "RN-2026-004829", sev: "P1", type: "accident", title: "Overturned car at junction", road: "Dwarka Expwy", km: "18.4",
      place: "Sector 99 junction", lat: 28.4895, lng: 76.9890, conf: 0.91, frames: 4, cam: "CAM-112",
      authority: "112 Emergency · Traffic Police", unit: "Awaiting", eta: "—", status: "New",
      detected: "08:45:03", ago: "1 min", img: IMG + "model-night-rollover.jpg" },
    { id: "RN-2026-004810", sev: "P2", type: "pothole", title: "Large pothole cluster", road: "NH-48", km: "128.0",
      place: "Kherki Daula toll, fast lane", lat: 28.3996, lng: 76.9958, conf: 0.78, frames: 3, cam: "CAM-071",
      authority: "NHAI Maintenance", unit: "PIU Gurugram", eta: "24 h SLA", status: "Work order issued",
      detected: "07:58:40", ago: "48 min", img: IMG + "incident-detail-3.jpg" },
    { id: "RN-2026-004803", sev: "P2", type: "pothole", title: "Deep pothole, lane 1", road: "Sohna Road", km: "6.2",
      place: "Vatika Chowk", lat: 28.4041, lng: 77.0436, conf: 0.74, frames: 3, cam: "CAM-130",
      authority: "Municipal PWD", unit: "PWD Gang-4", eta: "24 h SLA", status: "Assigned",
      detected: "07:31:15", ago: "1 h 15 m", img: IMG + "field-officer-3.jpg" },
    { id: "RN-2026-004798", sev: "P3", type: "damage", title: "Surface cracking", road: "NH-48", km: "134.7",
      place: "IMT Manesar slip road", lat: 28.3810, lng: 76.9640, conf: 0.69, frames: 3, cam: "CAM-078",
      authority: "Municipal PWD", unit: "Queued", eta: "72 h SLA", status: "Logged",
      detected: "06:55:02", ago: "1 h 51 m", img: IMG + "field-officer-2.jpg" },
    { id: "RN-2026-004791", sev: "P3", type: "pothole", title: "Small pothole", road: "NH-48", km: "160.4",
      place: "Dharuhera crossing", lat: 28.2066, lng: 76.7984, conf: 0.66, frames: 3, cam: "CAM-097",
      authority: "Municipal PWD", unit: "Queued", eta: "72 h SLA", status: "Logged",
      detected: "06:12:48", ago: "2 h 34 m", img: IMG + "incident-detail-3.jpg" },
  ];

  // One clean road frame per camera, in camSpots order. Accident cameras use the real Model B output frames.
  const camImgs = ["model-traffic-clear.jpg", "field-officer-2.jpg", "cameras-3.jpg", "model-junction-crash.jpg",
                   "cameras-4.jpg", "model-day-rollover.jpg", "index-1.jpg", "cameras-3.jpg",
                   "model-night-rollover.jpg", "cameras-2.jpg", "cameras-4.jpg", "incident-detail-3.jpg"];
  const camSpots = [
    ["CAM-064", "NH-48", "118.4", "Rajiv Chowk flyover", 28.4405, 77.0235], ["CAM-071", "NH-48", "128.0", "Kherki Daula toll", 28.3996, 76.9958],
    ["CAM-078", "NH-48", "134.7", "IMT Manesar slip", 28.3810, 76.9640], ["CAM-084", "NH-48", "142.6", "Manesar Bypass", 28.3589, 76.9387],
    ["CAM-088", "NH-48", "146.1", "Panchgaon chowk", 28.3337, 76.9163], ["CAM-091", "NH-48", "151.2", "Bilaspur Chowk", 28.2925, 76.8786],
    ["CAM-097", "NH-48", "160.4", "Dharuhera crossing", 28.2066, 76.7984], ["CAM-104", "NH-48", "171.8", "Kapriwas", 28.1525, 76.7322],
    ["CAM-112", "Dwarka Expwy", "18.4", "Sector 99 junction", 28.4895, 76.9890], ["CAM-118", "Dwarka Expwy", "11.2", "Sector 84", 28.4380, 76.9850],
    ["CAM-123", "SPR", "4.0", "Southern Peripheral Rd", 28.4078, 77.0587], ["CAM-130", "Sohna Road", "6.2", "Vatika Chowk", 28.4041, 77.0436],
  ];
  const CAMERAS = camSpots.map(([id, road, km, name, lat, lng], i) => ({
    id, road, km, name, lat, lng,
    status: id === "CAM-104" || id === "CAM-123" ? "offline" : "online",
    fps: id === "CAM-104" || id === "CAM-123" ? 0 : [30, 30, 25, 30][i % 4],
    res: ["1920×1080", "2560×1440", "1920×1080"][i % 3],
    img: IMG + camImgs[i % camImgs.length],
    incident: INCIDENTS.find(x => x.cam === id)?.id || null,
  }));

  return { MODEL, SEVERITY, INCIDENTS, CAMERAS, IMG };
})();
