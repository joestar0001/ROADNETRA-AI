import os
import json
import shutil
import threading
from datetime import datetime, timedelta

import cv2

from operations import estimate_repair


# Authority keys shared with every dashboard
AUTHORITY_NAMES = {
    "pwd": "Municipal / State PWD",
    "nhai": "NHAI",
    "hosp": "108 Emergency & Hospital",
    "pol": "Traffic Police",
}
AUTHORITY_STEPS = ["Notified", "Acknowledged", "En route", "Resolved"]

TYPE_META = {
    "pothole": {"category": "pothole", "label": "Pothole", "model": "Model A · Road Infrastructure"},
    "damaged_sign": {"category": "damage", "label": "Damaged Traffic Sign", "model": "Model A · Road Infrastructure"},
    "damaged_divider": {"category": "damage", "label": "Damaged Divider / Crash Barrier", "model": "Model A · Road Infrastructure"},
    "faded_zebra_crossing": {"category": "damage", "label": "Faded Zebra Crossing", "model": "Model A · Road Infrastructure"},
    "severe_accident": {"category": "accident", "label": "Severe Accident", "model": "Model B · Severe Accident"},
}
PRIORITY = {"Critical": "P1", "High": "P2", "Medium": "P3", "Low": "P4"}
SLA_HOURS = {"Critical": 1, "High": 24, "Medium": 72, "Low": 168}
BOX_COLORS = {  # BGR
    "pothole": (48, 59, 255),
    "damaged_sign": (10, 214, 255),
    "damaged_divider": (10, 159, 255),
    "faded_zebra_crossing": (10, 214, 255),
    "severe_accident": (58, 69, 255),
}


def canonical_type(t):
    t = (t or "pothole").lower()
    if "accident" in t:
        return "severe_accident"
    return t if t in TYPE_META else "pothole"


def _iou(a, b):
    """IoU of two normalized [x, y, w, h] boxes."""
    if not a or not b:
        return 0.0
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class MultiAgencyDispatcher:
    """
    Smart Incident Engine for ROADNETRA AI.

    Turns verified detections into incident records (ID, GPS, timestamp, confidence, severity,
    evidence, SLA), merges duplicates, and routes each incident to the right authorities:
      Municipality / State PWD (city & state roads), NHAI (national highways),
      Traffic Police (signals, signage, accidents), 108 Emergency & Hospitals (accidents).
    """

    def __init__(self, data_file=None, evidence_dir=None, cameras_file=None, seed_demo=True,
                 merge_window_infra_min=720, merge_window_accident_min=15):
        base = os.path.dirname(os.path.abspath(__file__))
        self.data_file = data_file or os.path.join(base, "data", "incidents_cache.json")
        self.cameras_file = cameras_file or os.path.join(base, "data", "cameras.json")
        self.evidence_dir = evidence_dir or os.path.join(base, "static", "evidence")
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        os.makedirs(self.evidence_dir, exist_ok=True)
        self.merge_window_infra = timedelta(minutes=merge_window_infra_min)
        self.merge_window_accident = timedelta(minutes=merge_window_accident_min)
        self.lock = threading.RLock()
        self.version = 0  # bumped on every change so dashboards can skip unchanged polls
        self.cameras = self._load_cameras()
        self.incidents = self._load(seed_demo)

    # ------------------------------------------------------------------ storage
    def _load_cameras(self):
        try:
            with open(self.cameras_file, "r", encoding="utf-8") as f:
                return json.load(f).get("cameras", [])
        except Exception as e:
            print("[DISPATCHER] cameras.json not loaded:", e)
            return []

    def _load(self, seed_demo):
        data = []
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    data = json.load(f) or []
            except Exception:
                data = []
        if data and not all("priority" in i for i in data):
            # Older schema from earlier prototypes: keep a copy, start a clean store
            legacy = self.data_file.replace(".json", ".legacy.json")
            if not os.path.exists(legacy):
                shutil.copyfile(self.data_file, legacy)
                print(f"[DISPATCHER] Legacy incident store backed up to {legacy}")
            data = []
        self.incidents = data
        for inc in self.incidents:  # keep stored costs on the current repair model
            self._apply_repair_estimate(inc)
        if not self.incidents and seed_demo:
            self.incidents = self._seed()
            self._save()
        return self.incidents

    def _save(self):
        self.version += 1
        tmp = self.data_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.incidents, f, indent=2)
            os.replace(tmp, self.data_file)
        except Exception as e:
            print("[DISPATCHER SAVE ERR]", e)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _apply_repair_estimate(inc):
        est = estimate_repair(inc)
        if est:
            inc["repair_cost_est"] = est["total"]
            inc["asphalt_tons"] = est["tons"]

    @staticmethod
    def _stamp(inc, authority, old, new, when):
        """Records when each authority reached each step (used for SLA, clearance and golden-hour timing)."""
        times = inc.setdefault("authority_times", {}).setdefault(authority, {})
        for s in range(old + 1, new + 1):
            times.setdefault(str(s), when.isoformat(timespec="seconds"))

    def get_camera(self, camera_id):
        for c in self.cameras:
            if c["id"] == camera_id:
                return c
        return None

    def resolve_location(self, ctx):
        """Normalise the location context sent by a dashboard (camera pick or custom GPS)."""
        ctx = ctx or {}
        cam = self.get_camera(ctx.get("camera_id")) if ctx.get("camera_id") else None
        lat = ctx.get("lat")
        lng = ctx.get("lng")
        loc = {
            "camera_id": ctx.get("camera_id") or (cam["id"] if cam else "CUSTOM"),
            "road": ctx.get("road") or (cam["road"] if cam else ""),
            "km": ctx.get("km") or (cam["km"] if cam else ""),
            "road_type": (ctx.get("road_type") or (cam["road_type"] if cam else "city")).lower(),
            "place": ctx.get("place") or (cam["name"] if cam else "Custom location"),
            "lat": float(lat) if lat not in (None, "") else (cam["lat"] if cam else None),
            "lng": float(lng) if lng not in (None, "") else (cam["lng"] if cam else None),
        }
        if loc["lat"] is None or loc["lng"] is None:
            fallback = self.cameras[0] if self.cameras else {"lat": 28.6139, "lng": 77.2090}
            loc["lat"], loc["lng"] = fallback["lat"], fallback["lng"]
        parts = [loc["place"]]
        if loc["road"]:
            parts.append(loc["road"] + (f" KM {loc['km']}" if loc["km"] else ""))
        loc["location_name"] = ", ".join(p for p in parts if p)
        return loc

    def _next_id(self, when):
        seq = 0
        for inc in self.incidents:
            try:
                seq = max(seq, int(inc["id"].rsplit("-", 1)[1]))
            except (ValueError, IndexError, KeyError):
                pass
        return f"IR-{when:%Y}-{when:%m%d}-{seq + 1:04d}"

    @staticmethod
    def route(defect_type, road_type):
        t = canonical_type(defect_type)
        if t == "severe_accident":
            keys = ["hosp", "pol"] + (["nhai"] if road_type == "national" else [])
            agency = "108 Emergency & Hospital Trauma · Traffic Police" + (" · NHAI" if road_type == "national" else "")
            return keys, agency
        if road_type == "national":
            keys, agency = ["nhai", "pwd"], "NHAI & PWD Highway Division"
        elif road_type == "state":
            keys, agency = ["pwd"], "State PWD"
        else:
            keys, agency = ["pwd"], "Municipality & PWD"
        if t in ("damaged_sign", "faded_zebra_crossing", "damaged_divider"):
            keys = keys + ["pol"]
            agency += " · Traffic Police"
        return keys, agency

    @staticmethod
    def score_severity(defect_type, conf, area_frac, road_type):
        t = canonical_type(defect_type)
        highway = 10 if road_type == "national" else 0
        if t == "severe_accident":
            score = 90 + conf * 10
        elif t == "pothole":
            score = 35 + conf * 30 + min(20.0, area_frac * 800) + highway
        elif t == "damaged_divider":
            score = 55 + conf * 25 + highway
        elif t == "damaged_sign":
            score = 35 + conf * 25 + highway / 2
        else:  # faded zebra crossing
            score = 30 + conf * 25 + (10 if road_type == "city" else 0)
        # Critical is reserved for accidents; infrastructure defects top out at High
        score = round(min(100.0 if t == "severe_accident" else 89.0, score), 1)
        if score >= 90:
            sev = "Critical"
        elif score >= 68:
            sev = "High"
        elif score >= 45:
            sev = "Medium"
        else:
            sev = "Low"
        return sev, score

    def _save_evidence(self, inc_id, frame, box_px, label, color):
        """Writes an annotated full frame and a tight crop. Returns (frame_url, crop_url)."""
        if frame is None:
            return None, None
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in box_px]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            crop_url = None
            if x2 > x1 and y2 > y1:
                pad_x, pad_y = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
                crop = frame[max(0, y1 - pad_y):min(h, y2 + pad_y), max(0, x1 - pad_x):min(w, x2 + pad_x)]
                cv2.imwrite(os.path.join(self.evidence_dir, f"{inc_id}_crop.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
                crop_url = f"/static/evidence/{inc_id}_crop.jpg"
            annotated = frame.copy()
            thick = max(2, int(round(min(w, h) / 220)))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
            scale = max(0.5, min(w, h) / 900)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, thick - 1))
            ty = y1 - 8 if y1 - th - 12 > 0 else y2 + th + 12
            cv2.rectangle(annotated, (x1, ty - th - 8), (x1 + tw + 10, ty + 6), color, -1)
            cv2.putText(annotated, label, (x1 + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), max(1, thick - 1), cv2.LINE_AA)
            cv2.imwrite(os.path.join(self.evidence_dir, f"{inc_id}_frame.jpg"), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return f"/static/evidence/{inc_id}_frame.jpg", crop_url
        except Exception as e:
            print("[DISPATCHER EVIDENCE ERR]", e)
            return None, None

    def _find_duplicate(self, t, loc, source, bbox, video_time, now):
        for inc in self.incidents:
            if inc.get("status") == "Resolved" or inc.get("type") != t:
                continue
            if inc.get("camera_id") != loc["camera_id"] or inc.get("source_name") != source.get("name"):
                continue
            try:
                age = now - datetime.fromisoformat(inc.get("last_seen") or inc["detected_at"])
            except Exception:
                continue
            if t == "severe_accident":
                if age <= self.merge_window_accident:
                    return inc
                continue
            if age > self.merge_window_infra:
                continue
            same_spot = _iou(inc.get("bbox"), bbox) >= 0.2
            if source.get("kind") == "video" and video_time is not None and inc.get("video_time_s") is not None:
                if abs(float(inc["video_time_s"]) - float(video_time)) <= 2.0 and same_spot:
                    return inc
            elif same_spot:
                return inc
        return None

    # ------------------------------------------------------------------ public API
    def create_incident(self, defect_type, conf, frame=None, box_px=None, location=None, source=None,
                        frames_confirmed=1, judge_reason="", video_time=None, stream_id=None):
        """Creates (or merges into) an incident. Returns (incident, merged: bool)."""
        t = canonical_type(defect_type)
        meta = TYPE_META[t]
        loc = location if location and "location_name" in location else self.resolve_location(location)
        source = source or {"kind": "live", "name": f"live:{loc['camera_id']}"}
        now = datetime.now()

        bbox, area_frac = None, 0.0
        if frame is not None and box_px is not None:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = box_px
            bbox = [round(max(0.0, x1 / w), 4), round(max(0.0, y1 / h), 4),
                    round(min(1.0, (x2 - x1) / w), 4), round(min(1.0, (y2 - y1) / h), 4)]
            area_frac = bbox[2] * bbox[3]

        with self.lock:
            dup = self._find_duplicate(t, loc, source, bbox, video_time, now)
            if dup is not None:
                dup["merged_count"] = dup.get("merged_count", 0) + 1
                dup["last_seen"] = now.isoformat(timespec="seconds")
                dup["frames_confirmed"] = max(dup.get("frames_confirmed", 1), frames_confirmed)
                if conf > dup.get("confidence", 0):
                    dup["confidence"] = round(float(conf), 3)
                    dup["confidence_pct"] = round(float(conf) * 100, 1)
                dup["action_history"].append({
                    "time": now.strftime("%I:%M %p"),
                    "action": f"Duplicate detection merged (seen again at {conf*100:.1f}% confidence)"
                })
                self._save()
                return dup, True

            inc_id = self._next_id(now)
            sev, score = self.score_severity(t, conf, area_frac, loc["road_type"])
            keys, agency = self.route(t, loc["road_type"])
            label = f"{meta['label'].upper()} {conf*100:.0f}%"
            frame_url, crop_url = self._save_evidence(inc_id, frame, box_px, label, BOX_COLORS.get(t, (0, 200, 255)))
            fallback_img = {
                "pothole": "/static/evidence/pothole_2.jpg", "damaged_divider": "/static/evidence/guardrail_2.jpg",
                "damaged_sign": "/static/evidence/sign_2.jpg", "faded_zebra_crossing": "/static/evidence/zebra_1.jpg",
                "severe_accident": "/static/evidence/accident_1.jpg",
            }[t]
            now_str = now.strftime("%I:%M %p")
            sla_h = SLA_HOURS[sev]
            src_label = {"video": "uploaded video", "image": "uploaded image", "live": "live camera"}.get(source.get("kind"), "camera")
            incident = {
                "id": inc_id,
                "type": t,
                "category": meta["category"],
                "model": meta["model"],
                "title": f"{meta['label']} · {loc['place']}",
                "severity": sev,
                "priority": PRIORITY[sev],
                "severity_score": score,
                "confidence": round(float(conf), 3),
                "confidence_pct": round(float(conf) * 100, 1),
                "frames_confirmed": int(frames_confirmed),
                "location_name": loc["location_name"],
                "place": loc["place"],
                "road": loc["road"],
                "km": loc["km"],
                "road_type": loc["road_type"],
                "camera_id": loc["camera_id"],
                "gps": [round(loc["lat"], 6), round(loc["lng"], 6)],
                "lat": round(loc["lat"], 6),
                "lng": round(loc["lng"], 6),
                "source": source.get("kind", "live"),
                "source_name": source.get("name"),
                "stream_id": stream_id,
                "video_time_s": round(float(video_time), 2) if video_time is not None else None,
                "bbox": bbox,
                "detected_at": now.isoformat(timespec="seconds"),
                "last_seen": now.isoformat(timespec="seconds"),
                "timestamp": f"Today, {now_str}",
                "assigned_agency": agency,
                "authorities": keys,
                "authority_status": {k: 0 for k in keys},
                "status": "Dispatched",
                "sla_hours": sla_h,
                "sla_due": (now + timedelta(hours=sla_h)).isoformat(timespec="seconds"),
                "verified_by_judge": True,
                "judge_reason": judge_reason or "Verified by Stage-2 Judge",
                "image_url": frame_url or crop_url or fallback_img,
                "crop_url": crop_url or frame_url or fallback_img,
                "merged_count": 0,
                "action_history": [
                    {"time": now_str, "action": f"{meta['model']} detected {meta['label'].lower()} on {src_label} "
                                                f"({conf*100:.1f}% confidence, {frames_confirmed} frame(s))"},
                    {"time": now_str, "action": f"Judge: {judge_reason or 'verified'}"},
                    {"time": now_str, "action": f"{PRIORITY[sev]} {sev} · routed to {agency} · SLA {sla_h} h"},
                ],
            }
            if t == "severe_accident":
                incident["ambulance_eta_mins"] = None
            else:
                self._apply_repair_estimate(incident)

            self.incidents.insert(0, incident)
            self._save()
            return incident, False

    # Backwards-compatible entry point used by older dashboards (POST /api/incidents/create)
    def add_live_detection(self, defect_type, conf, crop_rgb=None, location_name=None, gps=None, judge_reason="",
                           camera_id=None):
        ctx = {"camera_id": camera_id}
        if gps:
            ctx.update({"lat": gps[0], "lng": gps[1]})
        if location_name:
            ctx["place"] = location_name
        inc, _ = self.create_incident(defect_type, float(conf), location=ctx,
                                      source={"kind": "manual", "name": "manual"}, judge_reason=judge_reason)
        return inc

    def get_all_incidents(self):
        return self.incidents

    def get_incident(self, incident_id):
        for inc in self.incidents:
            if inc["id"] == incident_id:
                return inc
        return None

    def get_incidents_by_agency(self, agency_prefix):
        return [inc for inc in self.incidents if agency_prefix.lower() in inc.get("assigned_agency", "").lower()]

    def _overall_status(self, inc):
        steps = list(inc.get("authority_status", {}).values())
        if steps and all(s >= 3 for s in steps):
            return "Resolved"
        if any(s >= 2 for s in steps):
            return "Crew En Route"
        if any(s >= 1 for s in steps):
            return "Acknowledged"
        return "Dispatched"

    def advance_authority(self, incident_id, authority, step=None, note=""):
        with self.lock:
            inc = self.get_incident(incident_id)
            if not inc or authority not in inc.get("authority_status", {}):
                return None
            cur = inc["authority_status"][authority]
            new = min(3, max(0, int(step) if step is not None else cur + 1))
            now = datetime.now()
            self._stamp(inc, authority, cur, new, now)
            inc["authority_status"][authority] = new
            inc["status"] = self._overall_status(inc)
            inc["action_history"].append({
                "time": datetime.now().strftime("%I:%M %p"),
                "action": f"{AUTHORITY_NAMES.get(authority, authority)}: {AUTHORITY_STEPS[new]}" + (f" ({note})" if note else "")
            })
            if inc["status"] == "Resolved":
                inc["resolved_at"] = now.isoformat(timespec="seconds")
            self._save()
            return inc

    def update_incident_status(self, incident_id, new_status, note="", authorities=None):
        """Status update from a portal. `authorities` limits the change to that portal's own keys."""
        with self.lock:
            inc = self.get_incident(incident_id)
            if not inc:
                return False
            low = new_status.lower()
            if "resolv" in low or "verified" in low:
                target = 3
            elif "route" in low or "repair" in low or "crew" in low:
                target = 2
            else:
                target = 1
            steps = inc.setdefault("authority_status", {})
            keys = [k for k in (authorities or steps.keys()) if k in steps] or list(steps.keys())
            now = datetime.now()
            for k in keys:
                self._stamp(inc, k, steps[k], max(steps[k], target), now)
                steps[k] = max(steps[k], target)
            overall = self._overall_status(inc)
            # Keep the portal's wording, but never mark the whole incident resolved while another authority is still acting
            inc["status"] = new_status if (target < 3 or overall == "Resolved") else overall
            if inc["status"] == "Resolved":
                inc["resolved_at"] = datetime.now().isoformat(timespec="seconds")
            who = ", ".join(AUTHORITY_NAMES.get(k, k) for k in keys)
            action_text = f"{who}: status '{new_status}'" + (f" ({note})" if note else "")
            inc["action_history"].append({"time": datetime.now().strftime("%I:%M %p"), "action": action_text})
            self._save()
            return True

    def get_summary_metrics(self):
        inc = self.incidents
        agency = lambda i: i.get("assigned_agency", "").lower()
        now = datetime.now()
        overdue = 0
        for i in inc:
            try:
                if i.get("status") != "Resolved" and datetime.fromisoformat(i["sla_due"]) < now:
                    overdue += 1
            except Exception:
                pass
        return {
            "total_incidents": len(inc),
            "critical_incidents": sum(1 for i in inc if i.get("severity") == "Critical"),
            "active_interventions": sum(1 for i in inc if i.get("status") in ("Acknowledged", "Crew En Route", "Under Repair")),
            "resolved_incidents": sum(1 for i in inc if i.get("status") == "Resolved"),
            "new_incidents": sum(1 for i in inc if i.get("status") == "Dispatched"),
            "overdue_incidents": overdue,
            "accidents": sum(1 for i in inc if i.get("type") == "severe_accident"),
            "potholes": sum(1 for i in inc if i.get("type") == "pothole"),
            "road_damage": sum(1 for i in inc if i.get("category") == "damage"),
            "pwd_tickets": sum(1 for i in inc if "pwd" in agency(i) or "nhai" in agency(i) or "municipality" in agency(i)),
            "hospital_tickets": sum(1 for i in inc if "hospital" in agency(i)),
            "police_tickets": sum(1 for i in inc if "police" in agency(i)),
        }

    # ------------------------------------------------------------------ demo seed
    def _seed(self):
        """Five sample incidents placed on registered cameras so maps and portals are never empty."""
        samples = [
            ("severe_accident", 0.94, "CAM-084", "accident_1.jpg", "Verified: multi-vehicle collision with lane blockage", 25),
            ("pothole", 0.88, "CAM-071", "pothole_1.jpg", "Verified: road surface asphalt cavity & depression", 70),
            ("damaged_divider", 0.84, "CAM-112", "guardrail_1.jpg", "Verified: median guardrail structural deflection", 110),
            ("damaged_sign", 0.91, "CAM-130", "sign_1.jpg", "Verified: vertical signage bent at 45 degrees", 150),
            ("faded_zebra_crossing", 0.79, "CAM-130", "zebra_1.jpg", "Verified: crosswalk paint degradation in high-footfall zone", 200),
        ]
        self.incidents = []
        base_time = datetime.now()
        for t, conf, cam_id, img, why, mins_ago in samples:
            inc, _ = self.create_incident(t, conf, location={"camera_id": cam_id},
                                          source={"kind": "seed", "name": f"seed:{t}"}, judge_reason=why)
            when = base_time - timedelta(minutes=mins_ago)
            inc["detected_at"] = inc["last_seen"] = when.isoformat(timespec="seconds")
            inc["timestamp"] = f"Today, {when:%I:%M %p}"
            inc["sla_due"] = (when + timedelta(hours=inc["sla_hours"])).isoformat(timespec="seconds")
            inc["image_url"] = inc["crop_url"] = f"/static/evidence/{img}"
            inc["title"] = "[Sample] " + inc["title"]
            for h in inc["action_history"]:
                h["time"] = f"{when:%I:%M %p}"
        self.incidents.sort(key=lambda i: i["detected_at"], reverse=True)
        return self.incidents
