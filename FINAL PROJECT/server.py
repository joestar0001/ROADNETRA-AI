import os
import cv2
import time
import json
import base64
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from ultralytics import YOLO
from judge import RoadNetraJudge
from tracker import DefectTracker
from dispatcher import MultiAgencyDispatcher

app = Flask(__name__, static_folder="static", static_url_path="/static")

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "pothole_best.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "pothole_best.pt")

print("[ROADNETRA] Loading YOLOv8 Primary Defect Model from:", MODEL_PATH)
primary_model = YOLO(MODEL_PATH)
judge_engine = RoadNetraJudge()
dispatcher = MultiAgencyDispatcher()

# Global State for Live Video Streaming & Judge Feed
stream_state = {
    "source_type": "highway",
    "video_path": os.path.join(BASE_DIR, "data/test_videos/sample_highway_drive.mp4"),
    "enable_judge": True,
    "enable_accident_slot": False,
    "infer_res": 640,
    "thresholds": {
        "pothole": 0.25,
        "damaged_sign": 0.25,
        "damaged_divider": 0.20,
        "faded_zebra_crossing": 0.20
    },
    "vms_text": "CAUTION: ROAD HAZARD AHEAD - NH-44 CORRIDOR - REDUCE SPEED TO 30 KM/H",
    "green_wave_active": False,
    "active_fps": 30.0
}

@app.after_request
def add_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# -------------------------------------------------------------
# Web & Static Routes
# -------------------------------------------------------------
@app.route("/")
def index():
    # Primary dashboard HTML file provided by user
    html_path = os.path.join(BASE_DIR, "RoadNetra AI.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return send_from_directory(app.static_folder, "index.html")

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route("/api/upload_video", methods=["POST"])
def api_upload_video():
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400
    
    upload_dir = os.path.join(app.static_folder, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
    filename = f"uploaded_{int(time.time())}{ext}"
    save_path = os.path.join(upload_dir, filename)
    file.save(save_path)
    
    video_url = f"/static/uploads/{filename}"
    return jsonify({"success": True, "video_url": video_url, "filename": file.filename})

# Authority dashboard routes
@app.route("/pwd")
def pwd_dashboard():
    p = os.path.join(BASE_DIR, "RoadNetra_PWD.html")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return send_from_directory(app.static_folder, "pwd.html")

@app.route("/hospital")
def hospital_dashboard():
    p = os.path.join(BASE_DIR, "RoadNetra_Hospital.html")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return send_from_directory(app.static_folder, "hospital.html")

@app.route("/police")
def police_dashboard():
    p = os.path.join(BASE_DIR, "RoadNetra_Police.html")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return send_from_directory(app.static_folder, "police.html")

# RoadNetra AI design-system frontend (landing, command center, cameras, analytics, etc.)
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

@app.route("/frontend/")
@app.route("/frontend/<path:filename>")
def serve_frontend(filename="index.html"):
    return send_from_directory(FRONTEND_DIR, filename)

# Tracking and Deduplication State
tracker = DefectTracker(iou_thresh=0.20, max_dist=100.0, max_age=25, min_hits=2)
dispatched_track_ids = set()
global_frame_idx = 0

@app.route("/api/tracker/reset", methods=["POST"])
def api_tracker_reset():
    global tracker, dispatched_track_ids, global_frame_idx
    tracker = DefectTracker(iou_thresh=0.20, max_dist=100.0, max_age=25, min_hits=2)
    dispatched_track_ids = set()
    global_frame_idx = 0
    return jsonify({"success": True, "message": "Tracker reset for new stream"})

# -------------------------------------------------------------
# Real-Time AI Detection Endpoint (Frame Ingestion)
# -------------------------------------------------------------
@app.route("/api/detect", methods=["POST"])
def api_detect():
    global global_frame_idx, tracker, dispatched_track_ids
    data = request.json or {}
    frame_b64 = data.get("frame")
    if not frame_b64:
        return jsonify({"candidates": [], "unique_counts": tracker.get_unique_counts(), "all_unique_defects": []})

    try:
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"candidates": [], "unique_counts": tracker.get_unique_counts(), "all_unique_defects": []})

        global_frame_idx += 1
        h_img, w_img = frame.shape[:2]

        # Primary Proposal (Model A) at High Resolution (960px)
        res = primary_model.predict(frame, conf=0.15, imgsz=960, verbose=False)[0]
        accepted_for_tracking = []
        candidates_out = []

        class_thresholds = {
            "pothole": 0.28,
            "damaged_sign": 0.25,
            "damaged_divider": 0.20,
            "faded_zebra_crossing": 0.20,
            "severe_accident": 0.25
        }

        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                cls_id = int(b.cls[0].item())
                cls_name = primary_model.names.get(cls_id, "pothole")
                conf = float(b.conf[0].item())
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]

                # Filter proposals below class sensitivity threshold
                req_thresh = class_thresholds.get(cls_name, 0.25)
                if conf < req_thresh:
                    continue

                # Stage-2 Judge Evaluation
                verdict = judge_engine.evaluate_candidate(frame, (x1, y1, x2, y2), cls_name, conf)
                is_accepted = (verdict["status"] == "ACCEPTED")

                # Extract crop
                crop_bgr = frame[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
                crop_b64 = ""
                if crop_bgr.size > 0:
                    _, crop_buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    crop_b64 = "data:image/jpeg;base64," + base64.b64encode(crop_buf).decode("utf-8")

                # Authority Assignment
                if cls_name in ["pothole", "faded_zebra_crossing", "damaged_sign"]:
                    authority = "Municipal PWD & NHAI"
                elif cls_name == "damaged_divider":
                    authority = "Municipal PWD & Traffic Police"
                elif "accident" in cls_name:
                    authority = "108 Emergency & Traffic Police"
                else:
                    authority = "Municipal PWD"

                if is_accepted:
                    accepted_for_tracking.append({
                        'box': [x1, y1, x2, y2],
                        'class_name': cls_name,
                        'conf': verdict['final_conf'],
                        'crop_rgb': crop_b64,
                        'verdict_reason': verdict['reason'],
                        'authority': authority
                    })
                else:
                    # Rejected candidate (e.g. car veto)
                    candidates_out.append({
                        "id": f"rej-{cls_id}-{global_frame_idx}",
                        "kind": cls_name,
                        "x": max(0.0, x1 / w_img),
                        "y": max(0.0, y1 / h_img),
                        "w": min(1.0, (x2 - x1) / w_img),
                        "h": min(1.0, (y2 - y1) / h_img),
                        "conf": round(conf * 100, 1),
                        "ok": False,
                        "why": verdict["reason"],
                        "authority": "Suppressed (No Dispatch)",
                        "crop": crop_b64,
                        "is_confirmed": False,
                        "is_new_unique": False
                    })

        # Spatial-Temporal Deduplication Tracker
        tracked_results = tracker.update(accepted_for_tracking, global_frame_idx)

        for track_id, is_confirmed, det in tracked_results:
            x1, y1, x2, y2 = det['box']
            cls_name = det['class_name']
            authority = det.get('authority', 'Municipal PWD & NHAI')

            is_new_unique = False
            if is_confirmed and track_id not in dispatched_track_ids:
                dispatched_track_ids.add(track_id)
                is_new_unique = True
                # Log to dispatcher cache
                dispatcher.add_live_detection(
                    defect_type=cls_name,
                    conf=det['conf'],
                    crop_rgb=det.get('crop_rgb'),
                    location_name=f"Camera Feed (Zone Sector 14, Track #{track_id})",
                    judge_reason=det.get('verdict_reason', "Verified by Stage-2 Judge")
                )

            candidates_out.append({
                "id": f"{track_id}",
                "kind": cls_name,
                "x": max(0.0, x1 / w_img),
                "y": max(0.0, y1 / h_img),
                "w": min(1.0, (x2 - x1) / w_img),
                "h": min(1.0, (y2 - y1) / h_img),
                "conf": round(det['conf'] * 100, 1),
                "ok": True,
                "why": det.get('verdict_reason', 'Verified road defect'),
                "authority": authority,
                "crop": det.get('crop_rgb', ''),
                "is_confirmed": is_confirmed,
                "is_new_unique": is_new_unique
            })

        # Form unique confirmed defects list for the gallery
        all_unique = []
        for def_obj in tracker.get_all_confirmed_defects():
            all_unique.append({
                "track_id": def_obj.track_id,
                "kind": def_obj.class_name,
                "conf": round(def_obj.highest_conf * 100, 1),
                "hits": def_obj.hits,
                "why": def_obj.verdict_reason,
                "crop": def_obj.best_crop if isinstance(def_obj.best_crop, str) else "",
                "authority": "Municipal PWD & NHAI" if def_obj.class_name in ["pothole", "damaged_sign", "faded_zebra_crossing"] else "Municipal PWD & Traffic Police"
            })

        return jsonify({
            "candidates": candidates_out,
            "unique_counts": tracker.get_unique_counts(),
            "all_unique_defects": all_unique
        })
    except Exception as e:
        print("[API DETECT ERROR]", e)
        return jsonify({"candidates": [], "unique_counts": tracker.get_unique_counts(), "all_unique_defects": []})

# -------------------------------------------------------------
# REST APIs for Multi-Agency Dashboards
# -------------------------------------------------------------
@app.route("/api/incidents", methods=["GET"])
def api_incidents():
    agency = request.args.get("agency")
    status = request.args.get("status")
    incidents = dispatcher.get_all_incidents()
    if agency and agency != "all":
        incidents = [i for i in incidents if agency.lower() in i.get("assigned_agency", "").lower()]
    if status and status != "all":
        incidents = [i for i in incidents if i.get("status", "").lower() == status.lower()]
    return jsonify(incidents)

@app.route("/api/incidents/<incident_id>/action", methods=["POST"])
def api_incident_action(incident_id):
    data = request.json or {}
    new_status = data.get("status", "Under Repair")
    note = data.get("note", "Action executed via Command Portal")
    success = dispatcher.update_incident_status(incident_id, new_status, note)
    return jsonify({"success": success, "id": incident_id, "status": new_status})

@app.route("/api/incidents/create", methods=["POST"])
def api_incident_create():
    data = request.json or {}
    inc = dispatcher.add_live_detection(
        defect_type=data.get("type", "pothole"),
        conf=data.get("conf", 0.85),
        location_name=data.get("location_name", "Live Camera Feed #04"),
        judge_reason=data.get("judge_reason", "Verified by Stage-2 Judge Model")
    )
    return jsonify({"success": True, "incident": inc})

@app.route("/api/summary")
def api_summary():
    metrics = dispatcher.get_summary_metrics()
    metrics["active_fps"] = stream_state["active_fps"]
    metrics["vms_text"] = stream_state["vms_text"]
    metrics["green_wave_active"] = stream_state["green_wave_active"]
    return jsonify(metrics)

@app.route("/api/action/vms", methods=["POST"])
def api_action_vms():
    data = request.json or {}
    text = data.get("text", stream_state["vms_text"])
    stream_state["vms_text"] = text.upper()
    return jsonify({"success": True, "vms_text": stream_state["vms_text"]})

@app.route("/api/action/ambulance", methods=["POST"])
def api_action_ambulance():
    data = request.json or {}
    inc_id = data.get("incident_id")
    if inc_id:
        dispatcher.update_incident_status(inc_id, "Crew En Route", "ALS Ambulance Dispatched with Siren")
    stream_state["green_wave_active"] = True
    return jsonify({"success": True, "ambulance": "ALS-108-04", "eta_mins": 5, "green_wave": True})

@app.route("/api/action/green_wave", methods=["POST"])
def api_action_green_wave():
    data = request.json or {}
    active = data.get("active", not stream_state["green_wave_active"])
    stream_state["green_wave_active"] = active
    return jsonify({"success": True, "green_wave_active": active})

# -------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\n=======================================================")
    print(f"🚀 ROADNETRA AI Unified Platform Server Running!")
    print(f"🌐 Access at: http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
