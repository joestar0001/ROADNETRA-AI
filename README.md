# ROADNETRA AI

**Autonomous Real-Time Edge Intelligence for Municipal & Highway CCTV**[cite: 1]

**Event:** IEEE Hackathon 2026[cite: 1]  
**Track:** Gen AIML | **Problem ID:** 03.1 (Vision in the Wild)[cite: 1]  
**Team:** ESPADA (Lead: Bhawesh Gautam)[cite: 1]  

## Project Mission
ROADNETRA AI is an autonomous, real-time edge intelligence platform designed to transform passive municipal and highway CCTV cameras into proactive guardians[cite: 1]. It detects road hazards (potholes, structural damage) and severe traffic accidents, eliminates false alarms through multi-frame temporal smoothing, computes hazard severity scores, and dispatches geotagged alerts and work orders to authorities (NHAI, Municipal PWD, Emergency Services)[cite: 1].

## Technology Stack
*   **Core Vision Models:** Ultralytics YOLOv8s/YOLOv8n (Pretrained on COCO; lightweight for real-time edge inference)[cite: 3].
*   **Video Ingestion & CV:** OpenCV (cv2), NumPy, Pillow[cite: 3].
*   **Runtime Acceleration:** PyTorch 2.x (CUDA / Apple MPS / CPU)[cite: 3].
*   **Web Dashboard & UI:** Streamlit + Leaflet / Folium[cite: 3].
*   **Incident Storage:** SQLite (Local) / PostgreSQL + PostGIS[cite: 3].
*   **Automated Reporting:** ReportLab (Python PDF generation for work orders)[cite: 3].

## Architecture
The system utilizes a Dual-Stream Decoupled Vision Engine to maintain high framerates and prevent dataset annotation interference[cite: 1]. 
*   **Fast Stream (30 FPS):** Evaluates every frame for life-safety emergencies like vehicle rollovers or collisions[cite: 1]. 
*   **Sampled Stream (1-2 FPS):** Runs in a background worker thread to detect infrastructure hazards like potholes and surface cracks without dropping the main video framerate[cite: 1].

## Project Structure
The repository is organized according to the following directory architecture[cite: 3]:

```text
roadnetra-ai/
├── models/                     # Trained model checkpoints
│   ├── pothole_best.pt         # Model A: Fine-tuned Pothole weights
│   └── accident_best.pt        # Model B: Fine-tuned Severe Accident weights
├── engine/                     # Core Computer Vision & Processing
│   ├── __init__.py
│   ├── video_stream.py         # Asynchronous multi-threaded video capture
│   ├── dual_dispatcher.py      # Dual-stream frame scheduler
│   └── temporal_filter.py      # Multi-frame false alarm suppression
├── core/                       # Business Logic & Intelligence
│   ├── __init__.py
│   ├── severity_engine.py      # Computes LOW / MED / HIGH / CRITICAL rating
│   ├── authority_router.py     # Dispatches to PWD, NHAI, or Emergency
│   └── report_generator.py     # PDF work order generator 
├── dashboard/                  # Interactive User Interface
│   └── app.py                  # Streamlit web app
├── data/                       # Datasets & sample test videos
│   ├── test_videos/            # Real CCTV footage clips
│   └── incidents.db            # SQLite incident database
├── requirements.txt            # Python dependencies
└── README.md                   # Setup and launch instructions
