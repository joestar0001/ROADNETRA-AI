import os
import json
import time
import shutil
import subprocess
from datetime import datetime

from flask import Flask, Response, jsonify, request, send_from_directory

import config
import operations
from dispatcher import MultiAgencyDispatcher, AUTHORITY_NAMES, AUTHORITY_STEPS
from pipelines import ModelHub, DetectionService

BASE_DIR = config.BASE_DIR
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB uploads

dispatcher = MultiAgencyDispatcher(seed_demo=config.SEED_DEMO_INCIDENTS,
                                   merge_window_infra_min=config.MERGE_WINDOW_INFRA_MIN,
                                   merge_window_accident_min=config.MERGE_WINDOW_ACCIDENT_MIN)
hub = ModelHub()
service = DetectionService(hub, dispatcher)

# Shared operator state (VMS board, green wave)
stream_state = {
    "vms_text": "CAUTION: ROAD HAZARD AHEAD - REDUCE SPEED TO 30 KM/H",
    "green_wave_active": False,
}


@app.after_request
def add_cache_headers(response):
    if request.path.startswith("/api/") or request.path in ("/config.js",) or response.mimetype == "text/html":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    return response


def _html(name):
    return send_from_directory(BASE_DIR, name, mimetype="text/html")


# -------------------------------------------------------------
# Pages
# -------------------------------------------------------------
@app.route("/")
def index():
    return _html("RoadNetra AI.html")


@app.route("/pwd")
def pwd_dashboard():
    return _html("RoadNetra_PWD.html")


@app.route("/hospital")
def hospital_dashboard():
    return _html("RoadNetra_Hospital.html")


@app.route("/police")
def police_dashboard():
    return _html("RoadNetra_Police.html")


@app.route("/frontend")
def frontend_redirect():
    return Response(status=302, headers={"Location": "/frontend/"})


@app.route("/frontend/")
@app.route("/frontend/<path:filename>")
def serve_frontend(filename="index.html"):
    if os.path.basename(filename).startswith(".env"):
        return Response(status=404)
    return send_from_directory(FRONTEND_DIR, filename)


# -------------------------------------------------------------
# Config shared with every page (live map provider + key from frontend/.env)
# -------------------------------------------------------------
def _public_config():
    cfg = config.frontend_config()
    cfg["cameras"] = dispatcher.cameras
    cfg["authorities"] = AUTHORITY_NAMES
    cfg["authoritySteps"] = AUTHORITY_STEPS
    cfg["modelB"] = MODEL_B_CARD
    return cfg


@app.route("/api/config")
def api_config():
    return jsonify(_public_config())


@app.route("/config.js")
def config_js():
    body = "window.RN_CONFIG = " + json.dumps(_public_config()) + ";"
    return Response(body, mimetype="application/javascript")


@app.route("/api/health")
def api_health():
    return jsonify({"ok": True, "time": datetime.now().isoformat(timespec="seconds"),
                    "incidents": len(dispatcher.incidents), **hub.info()})


@app.route("/api/cameras")
def api_cameras():
    return jsonify(dispatcher.cameras)


# -------------------------------------------------------------
# Detection: Model B (accident, every frame) and Model A (infrastructure, ~1 FPS)
# -------------------------------------------------------------
@app.route("/api/stream/start", methods=["POST"])
def api_stream_start():
    data = request.json or {}
    s = service.start_session(data.get("mode", "video"), data.get("source_name"), data.get("location"))
    return jsonify({"stream_id": s.id, "mode": s.mode, "location": s.location})


@app.route("/api/stream/<stream_id>/summary")
def api_stream_summary(stream_id):
    out = service.summary(stream_id)
    return (jsonify(out), 200) if out else (jsonify({"error": "unknown stream"}), 404)


@app.route("/api/detect/accident", methods=["POST"])
def api_detect_accident():
    return jsonify(service.detect("accident", request.json or {}))


@app.route("/api/detect/infra", methods=["POST"])
def api_detect_infra():
    return jsonify(service.detect("infra", request.json or {}))


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """Runs all models on one frame (used for single images and older clients)."""
    data = request.json or {}
    data.setdefault("mode", "image")
    out = service.detect("both", data)
    out["candidates"] = out["detections"]  # legacy key
    return jsonify(out)


@app.route("/api/tracker/reset", methods=["POST"])
def api_tracker_reset():
    return jsonify({"success": True, "message": "Streams are isolated; start a new one with /api/stream/start"})


@app.route("/api/upload_video", methods=["POST"])
def api_upload_video():
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400
    upload_dir = os.path.join(app.static_folder, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    stamp = int(time.time())
    ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
    raw_path = os.path.join(upload_dir, f"uploaded_{stamp}{ext}")
    file.save(raw_path)
    filename = os.path.basename(raw_path)
    # Browsers only decode H.264/VP8/VP9/AV1. Transcode anything else with ffmpeg when it is installed.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        out_path = os.path.join(upload_dir, f"uploaded_{stamp}_h264.mp4")
        try:
            subprocess.run([ffmpeg, "-v", "error", "-y", "-i", raw_path, "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "23", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                            "-movflags", "+faststart", "-an", out_path], check=True, timeout=900)
            os.remove(raw_path)
            filename = os.path.basename(out_path)
        except Exception as e:
            print("[UPLOAD] ffmpeg transcode failed, serving original:", e)
    return jsonify({"success": True, "video_url": f"/static/uploads/{filename}", "filename": file.filename,
                    "transcoded": filename.endswith("_h264.mp4")})


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(FRONTEND_DIR, "assets", "img"), "logo-mark.svg", mimetype="image/svg+xml")


# -------------------------------------------------------------
# Incidents (shared by the main dashboard, the design-system frontend and the 3 portals)
# -------------------------------------------------------------
@app.route("/api/incidents", methods=["GET"])
def api_incidents():
    agency = request.args.get("agency")
    status = request.args.get("status")
    itype = request.args.get("type")
    incidents = dispatcher.get_all_incidents()
    if agency and agency != "all":
        key = {"pwd": "pwd", "nhai": "nhai", "hospital": "hospital", "hosp": "hospital", "police": "police", "pol": "police"}.get(agency.lower(), agency.lower())
        incidents = [i for i in incidents if key in i.get("assigned_agency", "").lower()]
    if status and status != "all":
        incidents = [i for i in incidents if i.get("status", "").lower() == status.lower()]
    if itype and itype != "all":
        incidents = [i for i in incidents if i.get("type") == itype or i.get("category") == itype]
    return jsonify(incidents)


@app.route("/api/incidents/<incident_id>", methods=["GET"])
def api_incident_get(incident_id):
    inc = dispatcher.get_incident(incident_id)
    return (jsonify(inc), 200) if inc else (jsonify({"error": "not found"}), 404)


@app.route("/api/incidents/<incident_id>/action", methods=["POST"])
def api_incident_action(incident_id):
    data = request.json or {}
    authorities = [a.strip() for a in str(data.get("authority") or "").split(",") if a.strip()]
    if authorities and not data.get("status"):
        # Advance every listed authority that this incident is routed to (e.g. "pwd,nhai")
        inc = None
        for a in authorities:
            inc = dispatcher.advance_authority(incident_id, a, data.get("step"), data.get("note", "")) or inc
        return jsonify({"success": inc is not None, "id": incident_id, "incident": inc,
                        "status": inc["status"] if inc else None})
    new_status = data.get("status", "Under Repair")
    note = data.get("note", "Action executed via Command Portal")
    success = dispatcher.update_incident_status(incident_id, new_status, note, authorities or None)
    inc = dispatcher.get_incident(incident_id)
    return jsonify({"success": success, "id": incident_id, "status": inc["status"] if inc else new_status, "incident": inc})


@app.route("/api/incidents/create", methods=["POST"])
def api_incident_create():
    data = request.json or {}
    inc = dispatcher.add_live_detection(
        defect_type=data.get("type", "pothole"),
        conf=float(data.get("conf", 0.85)),
        location_name=data.get("location_name"),
        gps=data.get("gps"),
        judge_reason=data.get("judge_reason", "Manually reported by operator"),
        camera_id=data.get("camera_id"),
    )
    return jsonify({"success": True, "incident": inc})


@app.route("/api/summary")
def api_summary():
    metrics = dispatcher.get_summary_metrics()
    metrics["vms_text"] = stream_state["vms_text"]
    metrics["green_wave_active"] = stream_state["green_wave_active"]
    meters = hub.info()["models"]
    metrics["active_fps"] = meters["accident"]["fps"]
    metrics["models"] = {k: {"avg_ms": v.get("avg_ms"), "fps": v.get("fps")} for k, v in meters.items() if k != "judge"}
    return jsonify(metrics)


@app.route("/api/action/vms", methods=["POST"])
def api_action_vms():
    data = request.json or {}
    stream_state["vms_text"] = data.get("text", stream_state["vms_text"]).upper()
    return jsonify({"success": True, "vms_text": stream_state["vms_text"]})


@app.route("/api/action/ambulance", methods=["POST"])
def api_action_ambulance():
    data = request.json or {}
    inc = dispatcher.get_incident(data.get("incident_id") or "")
    if not inc or "hosp" not in inc.get("authority_status", {}):
        return jsonify({"success": False, "error": "Unknown accident incident"}), 404
    stream_state["green_wave_active"] = True
    with dispatcher.lock:
        if inc.get("ambulance_unit") and inc["authority_status"]["hosp"] >= 2:
            return jsonify({"success": True, "ambulance": inc["ambulance_unit"], "eta_mins": inc.get("ambulance_eta_mins"),
                            "green_wave": True, "already": True})
        view = operations.build("hospital", dispatcher.get_all_incidents(), dispatcher.cameras, stream_state)
        free = next((u["id"] for u in view["units"] if not u["incident_id"]), None)
        if not free:
            return jsonify({"success": False, "error": "No ambulance available"}), 409
        km, eta = operations.ambulance_eta(inc, True)
        inc["ambulance_unit"], inc["ambulance_eta_mins"], inc["ambulance_distance_km"] = free, eta, km
        dispatcher.advance_authority(inc["id"], "hosp", 2, f"{free} dispatched with siren · {km} km · ETA {eta} min")
    return jsonify({"success": True, "ambulance": free, "eta_mins": eta, "distance_km": km, "green_wave": True})


@app.route("/api/action/green_wave", methods=["POST"])
def api_action_green_wave():
    data = request.json or {}
    stream_state["green_wave_active"] = bool(data.get("active", not stream_state["green_wave_active"]))
    return jsonify({"success": True, "green_wave_active": stream_state["green_wave_active"]})


@app.route("/api/operations")
def api_operations():
    """One poll per portal: its incidents plus every operational figure computed from them."""
    portal = (request.args.get("portal") or "").lower()
    portal = {"hosp": "hospital", "pol": "police", "nhai": "pwd"}.get(portal, portal)
    if portal not in ("pwd", "hospital", "police"):
        return jsonify({"error": "portal must be pwd, hospital or police"}), 400
    with dispatcher.lock:
        view = operations.build(portal, dispatcher.get_all_incidents(), dispatcher.cameras, stream_state)
        version = dispatcher.version
    return jsonify({"portal": portal, "version": version, "time": datetime.now().isoformat(timespec="seconds"),
                    "live": dict(stream_state), **view})


# -------------------------------------------------------------
# Live data for the design-system frontend (/frontend/), same schema as its demo-data.js
# -------------------------------------------------------------
MODEL_B_CARD = {
    "name": "Model B · Severe Accident", "arch": "YOLOv8s", "imgsz": 640, "epochs": 100, "bestEpoch": 90,
    "bestConf": 0.47, "pipelineConf": config.ACCIDENT_CONF, "temporalFrames": config.ACCIDENT_CONFIRM_FRAMES,
    "dataset": {"total": 1815, "train": 1270, "val": 272, "test": 273, "accident": 1585, "negative": 230},
    "test": {"mAP50": 0.830, "mAP5095": 0.502, "precision": 0.995, "recall": 0.836, "specificity": 0.966,
             "accuracy": 0.850, "f1": 0.909, "falseAlarm": 0.034, "TP": 204, "FP": 1, "FN": 40, "TN": 28},
    "latency": {"t4GpuMs": 2.5, "laptopCpuMs": 189},
}
SEVERITY_CARD = {
    "P1": {"label": "P1 Critical", "cls": "rn-p1", "color": "#EF4444"},
    "P2": {"label": "P2 High", "cls": "rn-p2", "color": "#F59E0B"},
    "P3": {"label": "P3 Medium", "cls": "rn-p3", "color": "#0EA5E9"},
    "P4": {"label": "P4 Low", "cls": "rn-p4", "color": "#94A3B8"},
}


def _ago(iso):
    try:
        secs = int((datetime.now() - datetime.fromisoformat(iso)).total_seconds())
    except Exception:
        return ""
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60} min"
    if secs < 86400:
        return f"{secs // 3600} h {secs % 3600 // 60} m"
    return f"{secs // 86400} d"


def _frontend_incident(i):
    status_unit = {"hosp": "AMB-108", "pol": "PCR-12", "nhai": "NHAI PIU", "pwd": "PWD Gang-4"}
    first = (i.get("authorities") or ["pwd"])[0]
    acting = i.get("authority_status", {}).get(first, 0) >= 2
    eta = (f"{i.get('ambulance_eta_mins')} min" if i.get("ambulance_eta_mins") else
           ("Awaiting" if i["type"] == "severe_accident" else f"{i.get('sla_hours', 72)} h SLA"))
    try:
        detected = datetime.fromisoformat(i["detected_at"]).strftime("%H:%M:%S")
    except Exception:
        detected = i.get("timestamp", "")
    return {
        "id": i["id"], "sev": i.get("priority", "P3"), "type": i.get("category", "damage"),
        "kind": i["type"], "title": i.get("title", "").replace("[Sample] ", ""),
        "road": i.get("road") or "—", "km": i.get("km") or "—", "place": i.get("place") or i.get("location_name", ""),
        "lat": i.get("lat") or i["gps"][0], "lng": i.get("lng") or i["gps"][1],
        "conf": i.get("confidence", 0), "frames": i.get("frames_confirmed", 1), "cam": i.get("camera_id", "—"),
        "authority": i.get("assigned_agency", ""), "unit": status_unit.get(first, "Queued") if acting else "Awaiting",
        "eta": eta, "status": i.get("status", "Dispatched"), "detected": detected, "ago": _ago(i.get("detected_at", "")),
        "img": i.get("image_url"), "crop": i.get("crop_url"), "why": i.get("judge_reason", ""),
        "history": i.get("action_history", []), "sample": i.get("source") == "seed",
        "severity": i.get("severity"), "score": i.get("severity_score"), "slaDue": i.get("sla_due"),
    }


@app.route("/api/rn-data.js")
def api_rn_data_js():
    incidents = [_frontend_incident(i) for i in dispatcher.get_all_incidents()]
    open_by_cam = {}
    for inc in incidents:
        if inc["status"] != "Resolved":
            open_by_cam.setdefault(inc["cam"], inc["id"])
    cameras = [{**c, "img": f"../assets/img/{['model-traffic-clear.jpg', 'field-officer-2.jpg', 'cameras-3.jpg', 'cameras-4.jpg', 'index-1.jpg', 'cameras-2.jpg'][n % 6]}",
                "res": c.get("res", "1920x1080").replace("x", "×"), "incident": open_by_cam.get(c["id"])}
               for n, c in enumerate(dispatcher.cameras)]
    data = {"MODEL": MODEL_B_CARD, "SEVERITY": SEVERITY_CARD, "INCIDENTS": incidents, "CAMERAS": cameras,
            "IMG": "../assets/img/", "LIVE": True, "CONFIG": _public_config()}
    return Response("window.RN_DATA = " + json.dumps(data) + ";", mimetype="application/javascript")


# -------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------
if __name__ == "__main__":
    print("\n=======================================================")
    print("ROADNETRA AI Unified Platform Server Running!")
    print(f"Command center : http://localhost:{config.PORT}/")
    print(f"Design frontend: http://localhost:{config.PORT}/frontend/")
    print(f"Portals        : /pwd  /hospital  /police")
    print("=======================================================\n")
    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)
