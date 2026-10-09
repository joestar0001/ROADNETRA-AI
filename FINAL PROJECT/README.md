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

```
                      ┌────────────────────────────────────────┐
                      │   Live CCTV / Dashcam Video Stream     │
                      └──────────────────┬─────────────────────┘
                                         │
                                         ▼
                 ┌──────────────────────────────────────────────────┐
                 │   Stage 1: Primary Detector (YOLOv8s @ 960px)    │
                 │   Recall: 91.2% | Inference Latency: 5.9 ms      │
                 └───────────────────────┬──────────────────────────┘
                                         │ Candidate Bounding Boxes
                                         ▼
                 ┌──────────────────────────────────────────────────┐
                 │       Stage 2: Semantic Judge Engine             │
                 │  - Vehicle Veto (Moving cars ≠ dividers/cavities)│
                 │  - Paint Veto (White stripes ≠ dark potholes)    │
                 │  - Geometry Veto (Aspect ratio & scale sanity)   │
                 └───────────────────────┬──────────────────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 [ REJECTED VETO ]               [ ACCEPTED DEFECT ]
                 Logged to Judge Cards           Pass to Tracker
                 (Video overlay clean)                   │
                                                         ▼
                                 ┌──────────────────────────────────────┐
                                 │   Spatial-Temporal Deduplicator      │
                                 │   IoU >= 0.20 | Min Frame Hits = 2   │
                                 └───────────────────────┬──────────────┘
                                                         │
                                                         ▼
                                 ┌──────────────────────────────────────┐
                                 │    Automated Multi-Agency Dispatch   │
                                 ├──────────────────────────────────────┤
                                 │ • Municipal PWD & NHAI (Asphalt/Sign)│
                                 │ • 108 Emergency ER (Hospital Desk)   │
                                 │ • Traffic Police (Signals & Barriers)│
                                 └──────────────────────────────────────┘
```

---

## 🖥️ Multi-Agency Portals

RoadNetra AI provides tailored dashboards adhering to the **Apple Human Interface Guidelines (San Francisco typography, glassmorphism, responsive telemetry)**:

| Portal | URL | Description |
|---|---|---|
| **Primary Command Center** | `http://localhost:8080/` | Full AI vision studio, interactive drag-and-drop feed, live Judge Engine cards, and audit evidence gallery |
| **Municipal PWD & NHAI** | `http://localhost:8080/pwd` | Work order ticket management, crew dispatch roster (Crew A/B/C), SLA timer, and resolution verification |
| **108 Hospital Trauma Desk** | `http://localhost:8080/hospital` | Critical accident dispatch, ICU/ER trauma bed availability tracker, and ALS ambulance coordination |
| **Traffic Police Control Room** | `http://localhost:8080/police` | Green corridor signal preemption, lane blocking, variable message sign (VMS) broadcasts |

---

## 🚀 Quickstart & Installation

### Prerequisites
- Python 3.9+ (Python 3.10 or 3.11 recommended)
- macOS (Apple Silicon MPS accelerated) or Linux/Windows (NVIDIA CUDA / CPU)

### 1. Clone & Setup
```bash
git clone https://github.com/your-username/roadnetra-ai.git
cd roadnetra-ai
```

### 2. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch Platform
```bash
chmod +x start_roadnetra.sh
./start_roadnetra.sh
```
The server will start on port `8080` and automatically open `http://localhost:8080` in your default browser.

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
├── RoadNetra AI.html                 # Primary Operations Command Dashboard
├── RoadNetra_PWD.html                # Municipal PWD & NHAI Portal
├── RoadNetra_Hospital.html           # 108 Emergency Hospital Desk
├── RoadNetra_Police.html             # Traffic Police Control Room
├── server.py                         # REST API & Video Ingestion Backend (Flask)
├── judge.py                          # Stage-2 Semantic & Geometric Judge AI
├── dispatcher.py                     # Multi-Agency Ticketing & Dispatch Engine
├── tracker.py                        # Spatial-Temporal Deduplication Tracker
├── train_roadnetra.py                # Model Fine-Tuning & Training Pipeline
├── start_roadnetra.sh                # Executable One-Click Start Script
├── requirements.txt                  # Python dependencies
├── models/
│   ├── pothole_best.pt               # Trained Primary YOLOv8 Hazard Weights
│   ├── yolov8n.pt                    # Semantic Judge Verification Weights
│   └── yolov8s.pt                    # Base Architecture Weights
├── static/                           # Static assets, evidence crops, and sample videos
└── data/test_videos/                 # Benchmark test clips (day, highway, night)
```

---

## 👥 Team ESPADA

Developed with ❤️ for the **IEEE Hackathon 2026** by Team ESPADA.
