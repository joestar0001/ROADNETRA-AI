# 🛣️ RoadNetra AI — Autonomous Road Safety & Multi-Agency Dispatch Platform

> **IEEE Hackathon 2026 — Problem Statement 03.1**  
> **Team ESPADA** · *Apple Human Interface System · Two-Stage Cascaded Edge Vision · Automated Multi-Agency ICCC*

---

## 📌 Executive Summary

**RoadNetra AI** is a state-of-the-art intelligent road infrastructure monitoring and emergency dispatch platform. Operating on live CCTV and dashcam streams, RoadNetra bridges the gap between high-recall edge computer vision and municipal response coordination.

Traditional road defect detectors suffer from crippling false-alarm rates (mistaking moving vehicle bodies, headlights, shadows, and painted stripes for road defects) and lack automated, deduped dispatch pipelines. RoadNetra solves this using a **Two-Stage Cascaded Vision Architecture**:
1. **Stage-1 High-Recall Edge Detector:** Custom YOLOv8s fine-tuned on 7,639 infrastructure hazard instances across four classes (`pothole`, `damaged_sign`, `damaged_divider`, `faded_zebra_crossing`).
2. **Stage-2 Semantic & Geometric Judge Engine:** Context-aware semantic verifier and geometric prior network that inspects candidate crops, vetoing false alarms before dispatch.
3. **Spatial-Temporal Deduplication Tracker:** IoU and centroid tracker ensuring stationary road hazards are counted and dispatched exactly **once**, eliminating duplicate work orders.
4. **Automated Multi-Agency Dispatch & SLA Tracking:** Real-time dispatching to Municipal PWD & NHAI, 108 Emergency Medical Services, and Traffic Police Control Rooms.

---

## 🏛️ System Architecture

Three models run behind one Flask backend. The browser never waits for AI: the video canvas renders at full frame rate and draws the latest cached boxes from each model.

```
 CCTV / uploaded video / photo ──► Browser canvas (render loop, never blocks)
        │
        ├── every frame, latest-frame-wins ─► POST /api/detect/accident ─► Model B · Severe Accident (YOLOv8s @ 640)
        │                                                                 confirmed over 3 frames
        └── every 1 s, non-blocking ────────► POST /api/detect/infra    ─► Model A · Road Infrastructure (YOLOv8s @ 960)
                                                                          pothole · damaged_sign · damaged_divider · faded_zebra_crossing
                                                                          confirmed over 2 sampled frames
                       both ─► Stage-2 Judge (COCO YOLOv8n): vehicle / paint / geometry vetoes
                            ─► per-stream tracker (multi-frame verification, zero duplicates)
                            ─► Smart Incident Engine: ID · GPS · timestamp · confidence · severity (P1–P4)
                               · evidence frame + crop · SLA · duplicate merging
                            ─► Automated routing: Municipality/State PWD · NHAI · Traffic Police · 108 Emergency
                            ─► Command center, City Map, PWD / Hospital / Police portals, design-system frontend
```

| Model | File | Classes | Accept rule |
|---|---|---|---|
| **Model A** · Road Infrastructure | `models/pothole_best.pt` | pothole, damaged_sign, damaged_divider, faded_zebra_crossing | per-class threshold + judge vetoes + 2 frames |
| **Model B** · Severe Accident | `models/accident_best_colab.pt` | severe_accident | conf ≥ 0.55, or ≥ 0.40 with a vehicle/person in context + 3 frames |
| **Stage-2 Judge** | `models/yolov8n.pt` | 80 COCO classes | vetoes cars/people/paint mistaken for defects |

Model B on its 273-image held-out test set: mAP@50 0.830, precision 0.995, recall 0.836. With the accept rule above, recall is 82.8% with 2/29 false alarms per image (before multi-frame verification).

Routing by incident type and the camera's road type (`data/cameras.json`):

| Incident | City road | State road | National highway |
|---|---|---|---|
| Pothole | Municipality & PWD | State PWD | NHAI & PWD |
| Sign / zebra / divider | + Traffic Police | + Traffic Police | + Traffic Police |
| Severe accident (P1) | 108 Emergency & Hospital · Traffic Police | same | same + NHAI |

---

## 🖥️ Multi-Agency Portals

RoadNetra AI provides tailored dashboards adhering to the **Apple Human Interface Guidelines (San Francisco typography, glassmorphism, responsive telemetry)**:

| Portal | URL | Description |
|---|---|---|
| **Primary Command Center** | `http://localhost:8080/` | Full AI vision studio, interactive drag-and-drop feed, live Judge Engine cards, and audit evidence gallery |
| **Municipal PWD & NHAI** | `http://localhost:8080/pwd` | Work order ticket management, crew dispatch roster (Crew A/B/C), SLA timer, and resolution verification |
| **108 Hospital Trauma Desk** | `http://localhost:8080/hospital` | Critical accident dispatch, ICU/ER trauma bed availability tracker, and ALS ambulance coordination |
| **Traffic Police Control Room** | `http://localhost:8080/police` | Green corridor signal preemption, lane blocking, variable message sign (VMS) broadcasts |
| **Design-System Frontend** | `http://localhost:8080/frontend/` | Landing page, login, live command center, CCTV grid, analytics, incident detail, field officer and settings screens |

---

## 🚀 Quickstart & Installation

### Prerequisites
- Python 3.9+ (3.10–3.13 tested on Windows CPU)
- Optional: NVIDIA GPU (CUDA) for real 30 FPS accident detection; `ffmpeg` on PATH to auto-convert uploaded videos the browser cannot play

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Configure (`.env` files)
```bash
cp .env.example .env                       # backend: models, thresholds, default camera
cp frontend/.env.example frontend/.env     # frontend: live map provider + API key
```
Paste your live map key into **`frontend/.env`**:
```ini
MAP_PROVIDER=auto        # auto-detects: AIza… = Google, pk.… = Mapbox, anything else = MapTiler
MAP_API_KEY=your-key-here
MAP_STYLE=dark           # dark | light | streets | satellite
```
With no key, every map uses free Esri tiles. Both `.env` files are git-ignored; restart the server after editing them.

### 3. Run
```bash
python server.py        # Windows / macOS / Linux
```
Open `http://localhost:8080` → **Live Lab**:
- **Run demo road**, **Try accident photo**, **Upload video / photo** (or drag and drop), or **Laptop camera**
- Pick the **camera / location** (registered camera, your GPS, or custom coordinates) so every incident is geo-tagged correctly
- Each confirmed detection shows its incident ID, confidence, severity, location + GPS, camera, time, SLA, routed authorities and evidence

---

## 🔌 API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/stream/start` | `{mode: video\|image\|live, source_name, location:{camera_id} or {lat,lng,place,road_type}}` → `stream_id` |
| POST | `/api/detect/accident` | Model B on one frame `{frame: dataURL, stream_id, video_time}` |
| POST | `/api/detect/infra` | Model A on one frame (same body) |
| POST | `/api/detect` | All models on one image (`mode: image`) |
| GET | `/api/stream/<id>/summary` | Confirmed hazards and counts for a scan |
| GET | `/api/incidents?agency=pwd\|hospital\|police&type=…` | Incident list (shared by every dashboard) |
| GET | `/api/incidents/<id>` | One incident |
| POST | `/api/incidents/<id>/action` | `{authority: pwd\|nhai\|hosp\|pol}` advances that authority, or `{status, authority?}` |
| GET | `/api/health` · `/api/summary` · `/api/cameras` · `/config.js` | Model status/latency, KPIs, camera registry, public frontend config |

---

## 📊 Model Training & Benchmarks

The primary hazard detector was trained on a curated infrastructure dataset under varied environmental conditions (daylight, night glare, monsoon rain, motion blur):

| Hazard Class | Annotations | Precision | Recall | mAP@50 | Assigned Agency |
|---|---|---|---|---|---|
| **Damaged Traffic Signs** | 1,240 | 93.4% | 91.2% | **96.1%** | Municipal PWD |
| **Asphalt Potholes** | 3,118 | 88.6% | 85.4% | **91.8%** | Municipal PWD & NHAI |
| **Faded Zebra Crossings** | 1,894 | 86.2% | 83.1% | **88.4%** | Municipal PWD & Traffic Police |
| **Damaged Concrete Dividers** | 1,387 | 84.1% | 79.8% | **85.2%** | Traffic Police & NHAI |
| **Overall Infrastructure Pipeline** | **7,639** | **88.1%** | **84.9%** | **90.4%** | **Unified ICCC** |

---

## 📁 Repository Structure

```text
├── RoadNetra AI.html                 # Command center: Live Lab, City Map, dashboards
├── RoadNetra_PWD.html                # Municipal PWD & NHAI portal
├── RoadNetra_Hospital.html           # 108 Emergency hospital desk
├── RoadNetra_Police.html             # Traffic Police control room
├── server.py                         # Flask REST API, pages, live config
├── pipelines.py                      # Model hub (3 models) + dual-stream detection service
├── judge.py                          # Stage-2 semantic & geometric judge
├── tracker.py                        # Multi-frame verification / deduplication tracker
├── dispatcher.py                     # Smart incident engine: severity, routing, SLA, evidence, merging
├── config.py                         # Loads .env and frontend/.env
├── .env.example                      # Backend settings template
├── requirements.txt
├── models/
│   ├── pothole_best.pt               # Model A · road infrastructure
│   ├── accident_best_colab.pt        # Model B · severe accident
│   ├── yolov8n.pt                    # Stage-2 judge (COCO)
│   └── yolov8s.pt                    # Base architecture weights
├── data/
│   ├── cameras.json                  # Camera registry (id, GPS, road, road_type)
│   ├── incidents_cache.json          # Incident store (created at runtime)
│   └── test_videos/                  # Sample clips
├── frontend/                         # Design-system UI (served at /frontend/), frontend/.env.example
└── static/                           # rn-map.js, evidence images, samples, uploads
```

---

## 👥 Team ESPADA

Developed with ❤️ for the **IEEE Hackathon 2026** by Team ESPADA.
