"""
ROADNETRA AI detection pipelines.

  CCTV / uploaded video ──► browser canvas (never blocks)
        ├─ every frame (as fast as the model allows) ─► /api/detect/accident ─► Model B  (severe accident)
        └─ every ~1 s, non-blocking ──────────────────► /api/detect/infra    ─► Model A  (pothole, sign, divider, zebra)
  Both ─► Stage-2 Judge (COCO YOLOv8n) ─► multi-frame tracker ─► Smart Incident Engine (dispatcher)

Each model has its own lock, so the accident stream is never queued behind the slower
infrastructure model, and each video stream has its own trackers (multi-frame verification and
duplicate merging per stream).
"""
import base64
import threading
import time
import uuid
from collections import deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config
from judge import RoadNetraJudge
from tracker import DefectTracker
from dispatcher import TYPE_META

INFRA_THRESHOLDS = {
    "pothole": config.POTHOLE_CONF,
    "damaged_sign": config.DAMAGED_SIGN_CONF,
    "damaged_divider": config.DAMAGED_DIVIDER_CONF,
    "faded_zebra_crossing": config.FADED_ZEBRA_CONF,
}


def resolve_device(pref):
    pref = (pref or "auto").lower()
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decode_frame(data_url):
    if not data_url:
        return None
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        buf = np.frombuffer(base64.b64decode(data_url), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def thumb_b64(frame, box, max_side=200):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return ""
    ch, cw = crop.shape[:2]
    s = max_side / max(ch, cw)
    if s < 1:
        crop = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii") if ok else ""


class LatencyMeter:
    def __init__(self, n=30):
        self.samples = deque(maxlen=n)
        self.stamps = deque(maxlen=n)

    def add(self, ms):
        self.samples.append(ms)
        self.stamps.append(time.time())

    def snapshot(self):
        avg = sum(self.samples) / len(self.samples) if self.samples else 0.0
        fps = 0.0
        if len(self.stamps) >= 2 and time.time() - self.stamps[-1] < 3:
            span = self.stamps[-1] - self.stamps[0]
            fps = (len(self.stamps) - 1) / span if span > 0 else 0.0
        return {"avg_ms": round(avg, 1), "fps": round(fps, 2), "runs": len(self.samples)}


class ModelHub:
    """Loads the three models once, warms them up, and serialises access per model."""

    def __init__(self):
        self.device = resolve_device(config.DEVICE)
        if self.device == "cpu":
            torch.set_num_threads(max(1, config.TORCH_THREADS))
        print(f"[ROADNETRA] Device: {self.device} · torch threads: {torch.get_num_threads()}")
        print("[ROADNETRA] Model A (infrastructure):", config.POTHOLE_MODEL_PATH)
        self.infra = YOLO(config.POTHOLE_MODEL_PATH)
        print("[ROADNETRA] Model B (accident):", config.ACCIDENT_MODEL_PATH)
        self.accident = YOLO(config.ACCIDENT_MODEL_PATH)
        print("[ROADNETRA] Stage-2 Judge:", config.JUDGE_MODEL_PATH)
        self.judge = RoadNetraJudge(config.JUDGE_MODEL_PATH, accident_conf=config.ACCIDENT_CONF,
                                    accident_min_conf=config.ACCIDENT_MIN_CONF, device=self.device)
        self.infra_lock = threading.Lock()
        self.accident_lock = threading.Lock()
        self.judge_lock = threading.Lock()
        self.meters = {"accident": LatencyMeter(), "infra": LatencyMeter()}
        self.half = self.device.startswith("cuda")
        self._warmup()

    def _warmup(self):
        t0 = time.time()
        blank = np.zeros((480, 640, 3), np.uint8)
        self.infra.predict(blank, imgsz=config.POTHOLE_IMGSZ, device=self.device, half=self.half, verbose=False)
        self.accident.predict(blank, imgsz=config.ACCIDENT_IMGSZ, device=self.device, half=self.half, verbose=False)
        self.judge.coco_model.predict(blank, imgsz=320, device=self.device, half=self.half, verbose=False)
        print(f"[ROADNETRA] 3 models warmed up in {time.time() - t0:.1f}s")

    def judge_candidate(self, frame, box, cls_name, conf):
        with self.judge_lock:
            return self.judge.evaluate_candidate(frame, box, cls_name, conf)

    def info(self):
        return {
            "device": self.device,
            "models": {
                "accident": {"name": "Model B · Severe Accident", "path": config.ACCIDENT_MODEL_PATH,
                             "classes": list(self.accident.names.values()), "imgsz": config.ACCIDENT_IMGSZ,
                             "accept_conf": config.ACCIDENT_CONF, "min_conf": config.ACCIDENT_MIN_CONF,
                             "confirm_frames": config.ACCIDENT_CONFIRM_FRAMES, **self.meters["accident"].snapshot()},
                "infra": {"name": "Model A · Road Infrastructure", "path": config.POTHOLE_MODEL_PATH,
                          "classes": list(self.infra.names.values()), "imgsz": config.POTHOLE_IMGSZ,
                          "thresholds": INFRA_THRESHOLDS, "confirm_frames": config.INFRA_CONFIRM_FRAMES,
                          **self.meters["infra"].snapshot()},
                "judge": {"name": "Stage-2 Judge · COCO YOLOv8n", "path": config.JUDGE_MODEL_PATH,
                          "classes": len(self.judge.coco_model.names)},
            },
        }


class StreamSession:
    """Per-stream state: location, trackers (multi-frame verification) and counters."""

    def __init__(self, mode, source_name, location):
        self.id = uuid.uuid4().hex[:12]
        self.mode = mode if mode in ("video", "image", "live") else "video"
        self.source_name = source_name or self.mode
        self.location = location
        self.created = self.last_used = time.time()
        single = self.mode == "image"
        self.infra_tracker = DefectTracker(iou_thresh=0.20, max_dist=100.0, max_age=25,
                                           min_hits=1 if single else config.INFRA_CONFIRM_FRAMES)
        self.accident_tracker = DefectTracker(iou_thresh=0.10, max_dist=160.0, max_age=15,
                                              min_hits=1 if single else config.ACCIDENT_CONFIRM_FRAMES)
        self.infra_frames = 0
        self.accident_frames = 0
        self.judge_ok = 0
        self.judge_rejected = 0
        self.lock = threading.Lock()

    @property
    def source(self):
        name = self.source_name if self.mode != "live" else f"live:{self.location.get('camera_id')}"
        return {"kind": self.mode, "name": name}


class DetectionService:
    def __init__(self, hub, dispatcher):
        self.hub = hub
        self.dispatcher = dispatcher
        self.sessions = {}
        self.sessions_lock = threading.Lock()

    # ------------------------------------------------------------- sessions
    def start_session(self, mode="video", source_name=None, location=None):
        loc = self.dispatcher.resolve_location(location or {"camera_id": config.DEFAULT_CAMERA_ID})
        s = StreamSession(mode, source_name, loc)
        with self.sessions_lock:
            self._prune()
            self.sessions[s.id] = s
        return s

    def get_session(self, stream_id, mode="video", source_name=None, location=None):
        with self.sessions_lock:
            s = self.sessions.get(stream_id) if stream_id else None
        if s is None:
            s = self.start_session(mode, source_name, location)
        elif location:
            s.location = self.dispatcher.resolve_location(location)
        s.last_used = time.time()
        return s

    def _prune(self, max_idle=1800, max_sessions=40):
        now = time.time()
        for sid in [k for k, v in self.sessions.items() if now - v.last_used > max_idle]:
            del self.sessions[sid]
        if len(self.sessions) > max_sessions:
            for sid, _ in sorted(self.sessions.items(), key=lambda kv: kv[1].last_used)[:len(self.sessions) - max_sessions]:
                del self.sessions[sid]

    # ------------------------------------------------------------- core
    def _box_norm(self, box, w, h):
        x1, y1, x2, y2 = box
        return {"x": round(max(0.0, x1 / w), 4), "y": round(max(0.0, y1 / h), 4),
                "w": round(min(1.0, (x2 - x1) / w), 4), "h": round(min(1.0, (y2 - y1) / h), 4)}

    def _run(self, kind, session, frame, video_time):
        hub = self.hub
        h, w = frame.shape[:2]
        if kind == "accident":
            model, lock, imgsz, floor = hub.accident, hub.accident_lock, config.ACCIDENT_IMGSZ, config.ACCIDENT_MIN_CONF * 0.85
            tracker = session.accident_tracker
        else:
            model, lock, imgsz, floor = hub.infra, hub.infra_lock, config.POTHOLE_IMGSZ, min(INFRA_THRESHOLDS.values()) * 0.75
            tracker = session.infra_tracker

        t0 = time.time()
        with lock:
            res = model.predict(frame, conf=floor, imgsz=imgsz, device=hub.device, half=hub.half, verbose=False)[0]
        infer_ms = (time.time() - t0) * 1000

        detections, accepted = [], []
        if res.boxes is not None and len(res.boxes):
            for b in res.boxes:
                cls_name = model.names.get(int(b.cls[0].item()), "pothole")
                if kind == "accident":
                    cls_name = "severe_accident"
                conf = float(b.conf[0].item())
                box = [int(v) for v in b.xyxy[0].tolist()]
                if kind == "infra" and conf < INFRA_THRESHOLDS.get(cls_name, 0.25):
                    continue
                verdict = hub.judge_candidate(frame, box, cls_name, conf)
                if verdict["status"] == "ACCEPTED":
                    session.judge_ok += 1
                    accepted.append({"box": box, "class_name": cls_name, "conf": float(verdict["final_conf"]),
                                     "raw_conf": conf, "crop_rgb": None, "verdict_reason": verdict["reason"]})
                else:
                    session.judge_rejected += 1
                    detections.append({
                        "id": f"rej-{kind}-{cls_name}-{round(box[0] / max(1, w), 1)}-{round(box[1] / max(1, h), 1)}",
                        "kind": cls_name, "model": "B" if kind == "accident" else "A",
                        **self._box_norm(box, w, h),
                        "conf": round(conf * 100, 1), "raw_conf": round(conf * 100, 1),
                        "ok": False, "why": verdict["reason"], "authority": "Suppressed (no dispatch)",
                        "is_confirmed": False, "hits": 0, "needed": tracker.min_hits,
                        "crop": thumb_b64(frame, box),
                    })

        new_incidents = []
        with session.lock:
            if kind == "accident":
                session.accident_frames += 1
                frame_idx = session.accident_frames
            else:
                session.infra_frames += 1
                frame_idx = session.infra_frames
            results = tracker.update(accepted, frame_idx)
            for track_id, confirmed, det in results:
                track = tracker.get_track(track_id)
                first_hit = track.hits == 1
                if det["conf"] >= track.highest_conf and track.incident_id is None:
                    track.payload = {"frame": frame, "box": det["box"], "video_time": video_time}
                incident, merged = None, False
                if confirmed and track.incident_id is None:
                    p = track.payload or {"frame": frame, "box": det["box"], "video_time": video_time}
                    incident, merged = self.dispatcher.create_incident(
                        det["class_name"], track.highest_conf, frame=p["frame"], box_px=p["box"],
                        location=session.location, source=session.source, frames_confirmed=track.hits,
                        judge_reason=det.get("verdict_reason", ""), video_time=p.get("video_time"),
                        stream_id=session.id)
                    track.incident_id = incident["id"]
                    track.incident = incident
                    track.payload = {}
                    new_incidents.append(self._incident_brief(incident, merged))
                inc = getattr(track, "incident", None)
                keys, agency = self.dispatcher.route(det["class_name"], session.location["road_type"])
                detections.append({
                    "id": f"{'B' if kind == 'accident' else 'A'}{track_id}", "track_id": track_id,
                    "kind": det["class_name"], "model": "B" if kind == "accident" else "A",
                    **self._box_norm(det["box"], w, h),
                    "conf": round(det["conf"] * 100, 1), "raw_conf": round(det["raw_conf"] * 100, 1),
                    "ok": True, "why": det["verdict_reason"], "authority": agency,
                    "is_confirmed": confirmed, "hits": track.hits, "needed": tracker.min_hits,
                    "incident_id": track.incident_id,
                    "severity": inc["severity"] if inc else None, "priority": inc["priority"] if inc else None,
                    "is_new_incident": incident is not None and not merged, "merged": merged,
                    "crop": thumb_b64(frame, det["box"]) if (first_hit or incident is not None) else "",
                })

        total_ms = (time.time() - t0) * 1000
        hub.meters[kind].add(total_ms)
        return {
            "model": kind, "stream_id": session.id, "frame_index": frame_idx,
            "frame_size": [w, h], "infer_ms": round(infer_ms, 1), "latency_ms": round(total_ms, 1),
            "detections": detections, "new_incidents": new_incidents,
            "location": session.location,
        }

    @staticmethod
    def _incident_brief(inc, merged):
        keys = ("id", "type", "title", "severity", "priority", "confidence_pct", "location_name", "lat", "lng",
                "camera_id", "assigned_agency", "timestamp", "detected_at", "image_url", "crop_url", "judge_reason",
                "frames_confirmed", "sla_hours", "status")
        return {**{k: inc.get(k) for k in keys}, "merged": merged}

    def detect(self, kind, payload):
        frame = decode_frame(payload.get("frame"))
        session = self.get_session(payload.get("stream_id"), payload.get("mode", "video"),
                                   payload.get("source_name"), payload.get("location"))
        if frame is None:
            return {"model": kind, "stream_id": session.id, "detections": [], "new_incidents": [],
                    "error": "No decodable frame", "location": session.location}
        video_time = payload.get("video_time")
        if kind == "both":
            out_b = self._run("accident", session, frame, video_time)
            out_a = self._run("infra", session, frame, video_time)
            return {"model": "both", "stream_id": session.id, "frame_size": out_a["frame_size"],
                    "latency_ms": round(out_a["latency_ms"] + out_b["latency_ms"], 1),
                    "detections": out_b["detections"] + out_a["detections"],
                    "new_incidents": out_b["new_incidents"] + out_a["new_incidents"],
                    "location": session.location}
        return self._run(kind, session, frame, video_time)

    def summary(self, stream_id):
        with self.sessions_lock:
            s = self.sessions.get(stream_id)
        if s is None:
            return None
        defects = []
        for tracker in (s.accident_tracker, s.infra_tracker):
            for t in tracker.get_all_confirmed_defects():
                inc = getattr(t, "incident", None)
                defects.append({
                    "track_id": t.track_id, "kind": t.class_name,
                    "label": TYPE_META.get(t.class_name, {}).get("label", t.class_name),
                    "conf": round(t.highest_conf * 100, 1), "hits": t.hits, "why": t.verdict_reason,
                    "incident_id": t.incident_id,
                    "crop": inc.get("crop_url") if inc else "",
                    "image": inc.get("image_url") if inc else "",
                    "severity": inc.get("severity") if inc else None,
                    "authority": inc.get("assigned_agency") if inc else "",
                    "location_name": inc.get("location_name") if inc else s.location["location_name"],
                })
        counts = s.infra_tracker.get_unique_counts()
        counts["severe_accident"] = s.accident_tracker.get_unique_counts()["severe_accident"]
        return {"stream_id": s.id, "mode": s.mode, "source_name": s.source_name, "location": s.location,
                "unique_counts": counts, "all_unique_defects": defects,
                "frames": {"accident": s.accident_frames, "infra": s.infra_frames},
                "judge": {"approved": s.judge_ok, "rejected": s.judge_rejected}}
