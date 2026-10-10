---
title: RoadNetra AI
emoji: 🛣️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: CCTV road-hazard and accident detection with live portals
---

# RoadNetra AI

AI-powered CCTV road-hazard and accident detection (IEEE Hackathon 2026, Team ESPADA).

- `/` command center and Live Lab
- `/pwd`, `/hospital`, `/police` authority portals
- `/frontend/` design-system dashboard

Runs three YOLOv8 models on CPU: Model A (road infrastructure), Model B (severe accident) and the COCO Stage-2 Judge.
Incidents are kept for as long as the Space is running; a restart starts again from the sample incidents.

Source: https://github.com/tanmayai23/ROADNETRA-AI
