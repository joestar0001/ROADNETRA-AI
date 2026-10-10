"""
Operational figures for the PWD / Hospital / Police portals, computed from the live incident store.

Every number a portal shows (budget, asphalt, SLA compliance, ambulance ETAs, golden hour,
patrol deployment, clearance time) is derived here from the detected incidents plus the
authority resources in config.py, so all dashboards agree with each other.
"""
import math
from datetime import datetime

import config

# ---------------- Repair cost model (one model for every view) ----------------
ASPHALT_DENSITY = 2.4        # t / m³ compacted cold mix
POTHOLE_DEPTH_CM = 6.5       # assumed average depth (camera cannot measure depth)
ROAD_VIEW_M2 = 40.0          # road surface visible in a typical CCTV frame, used to scale bbox area to m²
DEFAULT_POTHOLE_M2 = 0.4     # when a detection has no box (seed / manual report)
BITUMEN_L_PER_TON = 38       # RS-1 tack coat
FIXED_REPAIR = {             # (total ₹, material share) per work order
    "damaged_divider": (45000, 0.65),
    "damaged_sign": (8500, 0.70),
    "faded_zebra_crossing": (12000, 0.60),
}
SLA_PENALTY = {"pothole": 10000, "damaged_divider": 25000, "damaged_sign": 5000, "faded_zebra_crossing": 5000}
INFRA_LABELS = {"pothole": "Pothole", "damaged_divider": "Damaged Crash Barrier / Divider",
                "damaged_sign": "Damaged Traffic Sign", "faded_zebra_crossing": "Faded Zebra Crossing"}

# ---------------- Emergency response model ----------------
ROAD_FACTOR = 1.35           # road distance / straight-line distance
AMB_SPEED_KMH = 40
AMB_GREEN_WAVE_KMH = 55
TURNOUT_MIN = 2              # crew turnout before the ambulance rolls
GOLDEN_HOUR_S = 3600
VMS_RANGE_KM = 5.0          # a gantry warns about hazards within this distance


def pothole_area_m2(inc):
    bbox = inc.get("bbox")
    if not bbox:
        return DEFAULT_POTHOLE_M2
    return round(min(2.5, max(0.15, bbox[2] * bbox[3] * ROAD_VIEW_M2)), 2)


def pothole_estimate(area_m2, depth_cm=POTHOLE_DEPTH_CM, count=1):
    """Same formula as the PWD estimator page (rates are sent to the browser)."""
    tons = area_m2 * (depth_cm / 100.0) * ASPHALT_DENSITY * count
    material = tons * config.ASPHALT_RATE_PER_TON
    labour = config.LABOUR_PER_POTHOLE * count
    return {"area_m2": round(area_m2, 2), "tons": round(tons, 3), "bitumen_l": round(tons * BITUMEN_L_PER_TON, 1),
            "roller_h": round(max(2.0, tons * 1.6), 1) if count else 0.0,
            "material": round(material), "labour": round(labour), "total": round(material + labour)}


def estimate_repair(inc):
    t = inc.get("type")
    if t == "pothole":
        return pothole_estimate(pothole_area_m2(inc))
    if t in FIXED_REPAIR:
        total, share = FIXED_REPAIR[t]
        return {"area_m2": 0.0, "tons": 0.0, "bitumen_l": 0.0, "roller_h": 0.0,
                "material": round(total * share), "labour": round(total * (1 - share)), "total": total}
    return None


def haversine_km(a, b):
    lat1, lng1, lat2, lng2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def ambulance_eta(inc, green_wave):
    """Minutes from dispatch to scene: turnout + road distance at corridor speed."""
    km = haversine_km((config.HOSPITAL_LAT, config.HOSPITAL_LNG), (inc["lat"], inc["lng"])) * ROAD_FACTOR
    speed = AMB_GREEN_WAVE_KMH if green_wave else AMB_SPEED_KMH
    return round(km, 1), TURNOUT_MIN + math.ceil(km / speed * 60)


# ---------------- helpers ----------------
def _ts(iso):
    try:
        return datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return None


def step_of(inc, keys):
    """Progress of one portal's own authorities (0 Notified … 3 Resolved), None if not routed to it."""
    st = inc.get("authority_status", {})
    vals = [st[k] for k in keys if k in st]
    return min(vals) if vals else None


def step_time(inc, keys, step):
    """When the portal's authorities reached `step` (latest of them), falling back to resolved_at."""
    times = inc.get("authority_times", {})
    found = [_ts(times.get(k, {}).get(str(step))) for k in keys if k in inc.get("authority_status", {})]
    found = [t for t in found if t]
    if found:
        return max(found)
    return _ts(inc.get("resolved_at")) if step >= 3 else None


def _mins(a, b):
    return round((b - a).total_seconds() / 60.0, 1)


def _green_wave_saving(accidents):
    """Average minutes a pre-empted corridor saves per ambulance run to these accidents."""
    if not accidents:
        return None
    saved = [ambulance_eta(a, False)[1] - ambulance_eta(a, True)[1] for a in accidents]
    return round(sum(saved) / len(saved), 1)


def _lite(inc):
    return {k: v for k, v in inc.items() if k != "action_history"}


# ---------------- portals ----------------
PWD_KEYS = ("pwd", "nhai")


def pwd_view(incidents, now):
    rows, sla = [], {t: {"type": t, "label": INFRA_LABELS[t], "total": 0, "on_time": 0, "overdue": 0, "late": 0}
                     for t in INFRA_LABELS}
    budget = {"open_total": 0, "open_material": 0, "open_labour": 0, "spent": 0, "tons_open": 0.0, "bitumen_open": 0.0}
    open_potholes = []
    for inc in incidents:
        step = step_of(inc, PWD_KEYS)
        if step is None or inc.get("type") not in INFRA_LABELS:
            continue
        est = estimate_repair(inc)
        due = _ts(inc.get("sla_due"))
        done = step >= 3
        done_at = step_time(inc, PWD_KEYS, 3) if done else None
        overdue = bool(due and not done and now > due)
        late = bool(due and done and done_at and done_at > due)
        row = sla[inc["type"]]
        row["total"] += 1
        row["overdue"] += overdue
        row["late"] += late
        row["on_time"] += not (overdue or late)
        if done:
            budget["spent"] += est["total"]
        else:
            budget["open_total"] += est["total"]
            budget["open_material"] += est["material"]
            budget["open_labour"] += est["labour"]
            budget["tons_open"] += est["tons"]
            budget["bitumen_open"] += est["bitumen_l"]
            if inc["type"] == "pothole":
                open_potholes.append(est["area_m2"])
        rows.append({**_lite(inc), "ops": {
            "step": step, "repair": est, "overdue": overdue,
            "sla_left_h": round((due - now).total_seconds() / 3600, 1) if due and not done else None,
            "resolved_at": done_at.isoformat(timespec="seconds") if done_at else None}})

    total = sum(r["total"] for r in sla.values())
    on_time = sum(r["on_time"] for r in sla.values())
    for r in sla.values():
        r["compliance"] = round(100.0 * r["on_time"] / r["total"], 1) if r["total"] else None
        r["penalty"] = (r["overdue"] + r["late"]) * SLA_PENALTY[r["type"]]
    budget["tons_open"] = round(budget["tons_open"], 2)
    budget["bitumen_open"] = round(budget["bitumen_open"], 1)

    open_rows = [r for r in rows if r["ops"]["step"] < 3]
    open_rows.sort(key=lambda r: (-r.get("severity_score", 0), r.get("detected_at", "")))
    crews = []
    for crew_id, name, kit, types in (
            ("PV-01", "Patch Van Alpha", "Bitumen dispenser · cold-mix patching", {"pothole"}),
            ("RU-03", "Heavy Roller Unit Bravo", "Double-drum roller · compaction", {"pothole"}),
            ("SR-02", "Signage & Guardrail Rig", "Post driver · W-beam barrier · road marking",
             {"damaged_sign", "damaged_divider", "faded_zebra_crossing"})):
        job = next((r for r in open_rows if r["type"] in types and r["id"] not in {c["incident_id"] for c in crews}), None)
        crews.append({"id": crew_id, "name": name, "kit": kit, "incident_id": job["id"] if job else None,
                      "location": job["location_name"] if job else "Central PWD depot",
                      "step": job["ops"]["step"] if job else None,
                      "status": ("ON SITE · REPAIRING" if job["ops"]["step"] >= 2 else "ASSIGNED · AWAITING ACK"
                                 if job["ops"]["step"] == 0 else "MOBILISING") if job else "STANDBY IN DEPOT",
                      "queue": sum(1 for r in open_rows if r["type"] in types)})

    return {
        "incidents": rows,
        "kpi": {"open": len(open_rows), "resolved": len(rows) - len(open_rows),
                "sla_rate": round(100.0 * on_time / total, 1) if total else None,
                "overdue": sum(r["overdue"] for r in sla.values()),
                "penalty": sum(r["penalty"] for r in sla.values())},
        "budget": budget,
        "sla": list(sla.values()),
        "estimator": {"count": len(open_potholes),
                      "area_m2": round(sum(open_potholes) / len(open_potholes), 2) if open_potholes else DEFAULT_POTHOLE_M2,
                      "depth_cm": POTHOLE_DEPTH_CM,
                      "rates": {"density": ASPHALT_DENSITY, "asphalt_per_ton": config.ASPHALT_RATE_PER_TON,
                                "labour_per_pothole": config.LABOUR_PER_POTHOLE, "bitumen_l_per_ton": BITUMEN_L_PER_TON}},
        "crews": crews,
    }


def hospital_view(incidents, now, green_wave):
    rows, units_busy = [], {}
    for inc in incidents:
        step = step_of(inc, ("hosp",))
        if step is None:
            continue
        km, eta_normal = ambulance_eta(inc, False)
        _, eta_gw = ambulance_eta(inc, True)
        detected = _ts(inc.get("detected_at")) or now
        dispatched = step_time(inc, ("hosp",), 2) if step >= 2 else None
        received = step_time(inc, ("hosp",), 3) if step >= 3 else None
        eta_total = inc.get("ambulance_eta_mins") or (eta_gw if green_wave else eta_normal)
        if received:
            eta_left = 0
        elif dispatched:
            eta_left = max(0, math.ceil(eta_total - _mins(dispatched, now)))
        else:
            eta_left = eta_gw if green_wave else eta_normal
        unit = inc.get("ambulance_unit")
        if unit and dispatched and not received:
            units_busy[unit] = inc["id"]
        end = received or now
        rows.append({**_lite(inc), "ops": {
            "step": step, "distance_km": km, "eta_normal": eta_normal, "eta_green_wave": eta_gw, "eta_left": eta_left,
            "unit": unit, "dispatched_at": dispatched.isoformat(timespec="seconds") if dispatched else None,
            "received_at": received.isoformat(timespec="seconds") if received else None,
            "dispatch_after_s": int((dispatched - detected).total_seconds()) if dispatched else None,
            "golden_left_s": max(0, GOLDEN_HOUR_S - int((end - detected).total_seconds())),
            "golden_met": (end - detected).total_seconds() <= GOLDEN_HOUR_S if received else None}})

    active = [r for r in rows if r["ops"]["step"] < 3]
    done = [r for r in rows if r["ops"]["step"] >= 3]
    units = []
    for n in range(1, config.AMBULANCE_FLEET + 1):
        uid = f"ALS-108-{n:02d}"
        inc_id = units_busy.get(uid)
        r = next((x for x in active if x["id"] == inc_id), None)
        units.append({"id": uid, "incident_id": inc_id, "status": "DISPATCHED · SIREN ON" if r else "STANDBY",
                      "location": r["location_name"] if r else config.HOSPITAL_NAME,
                      "eta_left": r["ops"]["eta_left"] if r else None})
    urgent = min(active, key=lambda r: r["ops"]["golden_left_s"]) if active else None
    incoming = len(active)
    met = [r for r in done if r["ops"]["golden_met"]]
    return {
        "incidents": rows,
        "hospital": {"name": config.HOSPITAL_NAME, "lat": config.HOSPITAL_LAT, "lng": config.HOSPITAL_LNG},
        "kpi": {"active": incoming, "received": len(done),
                "ambulances_free": sum(1 for u in units if not u["incident_id"]), "fleet": config.AMBULANCE_FLEET,
                "avg_eta": round(sum(r["ops"]["eta_left"] for r in active) / incoming, 1) if incoming else None,
                "golden_rate": round(100.0 * len(met) / len(done), 1) if done else None,
                "green_wave_saving": _green_wave_saving(rows)},
        "units": units,
        "golden": {"incident_id": urgent["id"], "location": urgent["location_name"], "detected_at": urgent["detected_at"],
                   "seconds_left": urgent["ops"]["golden_left_s"], "dispatched_at": urgent["ops"]["dispatched_at"],
                   "dispatch_after_s": urgent["ops"]["dispatch_after_s"], "eta_left": urgent["ops"]["eta_left"],
                   "unit": urgent["ops"]["unit"]} if urgent else None,
        "beds": {"bays_total": config.TRAUMA_BAYS, "bays_free": max(0, config.TRAUMA_BAYS - incoming), "incoming": incoming,
                 "icu_prealerts": sum(1 for r in active if r.get("severity") == "Critical"),
                 "blood_units_held": 4 * incoming},
    }


def police_view(incidents, now, cameras, vms_text):
    rows = []
    for inc in incidents:
        step = step_of(inc, ("pol",))
        if step is None:
            continue
        detected = _ts(inc.get("detected_at")) or now
        cleared = step_time(inc, ("pol",), 3) if step >= 3 else None
        rows.append({**_lite(inc), "ops": {
            "step": step, "cleared_at": cleared.isoformat(timespec="seconds") if cleared else None,
            "clearance_min": _mins(detected, cleared) if cleared else None,
            "open_min": None if cleared else _mins(detected, now)}})
    active = [r for r in rows if r["ops"]["step"] < 3]
    cleared = [r["ops"]["clearance_min"] for r in rows if r["ops"]["clearance_min"] is not None]
    deployed = [r for r in active if r["ops"]["step"] >= 2]
    patrols = [{"id": f"PCR-{n + 1:02d}", "incident_id": r["id"], "location": r["location_name"], "status": "AT INCIDENT"}
               for n, r in enumerate(deployed[:config.PCR_FLEET])]

    # VMS gantries sit at registered cameras; a gantry broadcasts when an open hazard is within VMS_RANGE_KM
    boards = []
    for n, c in enumerate(cameras[:config.VMS_BOARDS]):
        near = [r for r in active if r.get("lat") is not None and haversine_km((c["lat"], c["lng"]), (r["lat"], r["lng"])) <= VMS_RANGE_KM]
        boards.append({"id": f"VMS-{n + 1:02d}", "name": f"{c['road']} KM {c['km']} · {c['name']}", "lat": c["lat"], "lng": c["lng"],
                       "road": c["road"], "active": bool(near), "hazards": [r["id"] for r in near],
                       "text": vms_text if near else "DRIVE SAFE · FOLLOW LANE DISCIPLINE"})

    newest = sorted(active, key=lambda r: r.get("detected_at", ""), reverse=True)
    acc = next((r for r in newest if r["type"] == "severe_accident"), None)
    haz = next((r for r in newest if r["type"] != "severe_accident"), None)
    where = lambda r: (f"{r['road']} KM {r['km']}" if r.get("road") else r.get("place", "")).upper()
    presets = {
        "accident": f"ACCIDENT AHEAD - {where(acc)} - SLOW DOWN TO 20 KM/H - KEEP LEFT" if acc else
                    "ACCIDENT AHEAD - SLOW DOWN TO 20 KM/H - KEEP LEFT",
        "hazard": f"{INFRA_LABELS.get(haz['type'], 'ROAD HAZARD').upper()} AHEAD - {where(haz)} - REDUCE SPEED TO 30 KM/H" if haz else
                  "ROAD HAZARD AHEAD - REDUCE SPEED TO 30 KM/H",
        "clear": "CORRIDOR CLEAR - DRIVE SAFE - FOLLOW LANE DISCIPLINE",
    }
    return {
        "incidents": rows,
        "kpi": {"active": len(active), "accidents": sum(1 for r in active if r["type"] == "severe_accident"),
                "pcr_deployed": len(patrols), "pcr_fleet": config.PCR_FLEET,
                "vms_active": sum(1 for b in boards if b["active"]), "vms_total": len(boards),
                "avg_clearance_min": round(sum(cleared) / len(cleared), 1) if cleared else None,
                "green_wave_saving": _green_wave_saving([r for r in rows if r["type"] == "severe_accident"])},
        "patrols": patrols,
        "boards": boards,
        "vms_presets": presets,
    }


def build(portal, incidents, cameras, live):
    now = datetime.now()
    if portal == "pwd":
        return pwd_view(incidents, now)
    if portal == "hospital":
        return hospital_view(incidents, now, live.get("green_wave_active", False))
    if portal == "police":
        return police_view(incidents, now, cameras, live.get("vms_text", ""))
    raise ValueError(portal)
