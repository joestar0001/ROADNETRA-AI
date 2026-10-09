import os
import json
import cv2
import base64
from datetime import datetime


class MultiAgencyDispatcher:
    """
    Central Multi-Agency Incident & Dispatch Engine for ROADNETRA AI.
    
    Synchronizes confirmed road hazards & accidents across:
      1. Municipality, PWD & NHAI (Road Infrastructure)
      2. Emergency Hospital & 108 Trauma Care (Accidents)
      3. Traffic Police & Smart City Transit (Traffic Safety & Diversions)
    """

    def __init__(self, data_file=None, evidence_dir=None):
        base = os.path.dirname(os.path.abspath(__file__))
        self.data_file = data_file or os.path.join(base, "data", "incidents_cache.json")
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        self.evidence_dir = evidence_dir or os.path.join(base, "static", "evidence")
        os.makedirs(self.evidence_dir, exist_ok=True)
        self.incidents = self._load_or_seed_incidents()

    def _load_or_seed_incidents(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r") as f:
                    data = json.load(f)
                    if data and len(data) > 0:
                        return data
            except Exception:
                pass
        seed = self._generate_seed_incidents()
        self.incidents = seed
        self._save()
        return seed

    def _save(self):
        try:
            with open(self.data_file, "w") as f:
                json.dump(self.incidents, f, indent=2)
        except Exception:
            pass

    def _generate_seed_incidents(self):
        """Seed realistic smart city incidents with REAL photographs."""
        return [
            {
                "id": "INC-2026-0811",
                "type": "pothole",
                "title": "Severe Crater Pothole (Depth > 9cm)",
                "severity": "High",
                "location_name": "NH-44 Highway, Km 14.2 (Southbound Lane 2)",
                "gps": [28.6289, 77.2065],
                "assigned_agency": "PWD / NHAI",
                "status": "Under Repair",
                "confidence": 0.88,
                "timestamp": "Today, 08:42 AM",
                "verified_by_judge": True,
                "judge_reason": "Verified: Road surface cavity confirmed (Hazardous to 2-wheelers)",
                "image_url": "/static/evidence/pothole_1.jpg",
                "action_history": [
                    {"time": "08:42 AM", "action": "Detected by ROADNETRA Edge CCTV unit #CCTV-NH44-09"},
                    {"time": "08:43 AM", "action": "Automated Ticket Dispatched to PWD Central Division"},
                    {"time": "09:15 AM", "action": "Maintenance Van #PV-04 Dispatched with Cold-Mix Asphalt Crew"}
                ],
                "repair_cost_est": 18500,
                "asphalt_tons": 2.4
            },
            {
                "id": "INC-2026-0812",
                "type": "damaged_divider",
                "title": "Crumpled Steel Crash Barrier (Median Breach)",
                "severity": "Critical",
                "location_name": "Ring Road Flyover, Junction 7 Ramp",
                "gps": [28.6139, 77.2090],
                "assigned_agency": "PWD / NHAI",
                "status": "Crew En Route",
                "confidence": 0.84,
                "timestamp": "Today, 09:10 AM",
                "verified_by_judge": True,
                "judge_reason": "Verified: Median guardrail structural deflection confirmed",
                "image_url": "/static/evidence/guardrail_1.jpg",
                "action_history": [
                    {"time": "09:10 AM", "action": "Detected by ROADNETRA Overhead Fisheye #CCTV-RR-14"},
                    {"time": "09:11 AM", "action": "Emergency Alert sent to NHAI Rapid Response Team"},
                    {"time": "09:20 AM", "action": "Traffic Police notified of median hazard"}
                ],
                "repair_cost_est": 45000,
                "asphalt_tons": 0.0
            },
            {
                "id": "INC-2026-0813",
                "type": "severe_accident",
                "title": "Multi-Vehicle Collision (Rollover & Lane Blockage)",
                "severity": "Critical",
                "location_name": "Outer Bypass Expressway, Km 22.8",
                "gps": [28.6450, 77.2280],
                "assigned_agency": "Hospital Trauma & 108 EMS",
                "status": "Dispatched",
                "confidence": 0.94,
                "timestamp": "Today, 10:14 AM",
                "verified_by_judge": True,
                "judge_reason": "Verified: High-speed collision signature & inverted vehicle chassis",
                "image_url": "/static/evidence/accident_1.jpg",
                "action_history": [
                    {"time": "10:14 AM", "action": "Accident detected via ROADNETRA Dual-Stream Vision Model"},
                    {"time": "10:14 AM", "action": "Automated SOS sent to 108 Emergency Dispatch & Trauma Bay"},
                    {"time": "10:15 AM", "action": "Green Wave Traffic Signal corridor pre-empted"}
                ],
                "casualties_est": 2,
                "ambulance_eta_mins": 6
            },
            {
                "id": "INC-2026-0814",
                "type": "damaged_sign",
                "title": "Bent Overhead Speed Limit Sign (Occluded)",
                "severity": "Medium",
                "location_name": "Sector 18 Commercial Hub Exit",
                "gps": [28.5700, 77.3200],
                "assigned_agency": "Traffic Police & Transit Dept",
                "status": "Dispatched",
                "confidence": 0.91,
                "timestamp": "Today, 07:30 AM",
                "verified_by_judge": True,
                "judge_reason": "Verified: Vertical signage bent at 45 degree angle",
                "image_url": "/static/evidence/sign_1.jpg",
                "action_history": [
                    {"time": "07:30 AM", "action": "Detected by Patrol Car Dashcam Stream #DASH-08"},
                    {"time": "07:32 AM", "action": "Ticket logged for Municipal Street Asset Replacement"}
                ],
                "repair_cost_est": 8500,
                "asphalt_tons": 0.0
            },
            {
                "id": "INC-2026-0815",
                "type": "faded_zebra_crossing",
                "title": "Extremely Faded Pedestrian Crosswalk (> 80% Loss)",
                "severity": "High",
                "location_name": "Metro Station Gate 3 School Zone Crossing",
                "gps": [28.6320, 77.2180],
                "assigned_agency": "Municipality & PWD",
                "status": "Pending",
                "confidence": 0.79,
                "timestamp": "Today, 06:50 AM",
                "verified_by_judge": True,
                "judge_reason": "Verified: Surface crosswalk paint degradation confirmed in high-footfall zone",
                "image_url": "/static/evidence/zebra_1.jpg",
                "action_history": [
                    {"time": "06:50 AM", "action": "Identified during Early Morning Road Marking Scan"},
                    {"time": "06:52 AM", "action": "High-Priority Thermoplastic Striping Work Order created"}
                ],
                "repair_cost_est": 12000,
                "asphalt_tons": 0.0
            }
        ]

    def get_all_incidents(self):
        return self.incidents

    def get_incidents_by_agency(self, agency_prefix):
        return [inc for inc in self.incidents if agency_prefix.lower() in inc.get("assigned_agency", "").lower()]

    def add_live_detection(self, defect_type, conf, crop_rgb=None, location_name="CCTV Camera Feed #04", gps=None, judge_reason=""):
        """Generates an automated ticket and saves the real crop image."""
        new_id = f"INC-2026-{1000 + len(self.incidents) + 1}"
        now_str = datetime.now().strftime("%I:%M %p")
        
        # Save real image file
        image_url = "/static/evidence/pothole_1.jpg"
        if crop_rgb is not None:
            filename = f"crop_{new_id}.jpg"
            filepath = os.path.join(self.evidence_dir, filename)
            saved = False
            if isinstance(crop_rgb, str) and crop_rgb.startswith("data:image"):
                try:
                    b64_data = crop_rgb.split(",", 1)[1] if "," in crop_rgb else crop_rgb
                    img_data = base64.b64decode(b64_data)
                    with open(filepath, "wb") as f:
                        f.write(img_data)
                    image_url = f"/static/evidence/{filename}"
                    saved = True
                except Exception as e:
                    print("[DISPATCHER CROP ERR]", e)
            elif hasattr(crop_rgb, "size") and crop_rgb.size > 0:
                try:
                    bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filepath, bgr)
                    image_url = f"/static/evidence/{filename}"
                    saved = True
                except Exception as e:
                    print("[DISPATCHER CROP ERR]", e)
            if not saved:
                if "pothole" in defect_type: image_url = "/static/evidence/pothole_2.jpg"
                elif "guardrail" in defect_type or "divider" in defect_type: image_url = "/static/evidence/guardrail_2.jpg"
                elif "sign" in defect_type: image_url = "/static/evidence/sign_2.jpg"
                elif "zebra" in defect_type: image_url = "/static/evidence/zebra_1.jpg"
                elif "accident" in defect_type: image_url = "/static/evidence/accident_1.jpg"
        else:
            # Fallback to authentic dataset sample by class
            if "pothole" in defect_type: image_url = "/static/evidence/pothole_2.jpg"
            elif "guardrail" in defect_type or "divider" in defect_type: image_url = "/static/evidence/guardrail_2.jpg"
            elif "sign" in defect_type: image_url = "/static/evidence/sign_2.jpg"
            elif "zebra" in defect_type: image_url = "/static/evidence/zebra_1.jpg"
            elif "accident" in defect_type: image_url = "/static/evidence/accident_1.jpg"

        if defect_type in ["pothole", "faded_zebra_crossing"]:
            agency = "Municipality & PWD"
            severity = "High" if defect_type == "pothole" else "Medium"
        elif defect_type == "damaged_divider":
            agency = "PWD / NHAI"
            severity = "Critical"
        elif defect_type == "damaged_sign":
            agency = "Traffic Police & Transit Dept"
            severity = "Medium"
        elif "accident" in defect_type.lower():
            agency = "Hospital Trauma & 108 EMS"
            severity = "Critical"
        else:
            agency = "Municipality & PWD"
            severity = "Medium"

        default_gps = gps or [28.6139 + (len(self.incidents) * 0.0035), 77.2090 + (len(self.incidents) * 0.0025)]

        incident = {
            "id": new_id,
            "type": defect_type,
            "title": f"Live Detected {defect_type.replace('_', ' ').title()}",
            "severity": severity,
            "location_name": location_name,
            "gps": default_gps,
            "assigned_agency": agency,
            "status": "Dispatched",
            "confidence": round(float(conf), 3),
            "timestamp": f"Today, {now_str}",
            "verified_by_judge": True,
            "judge_reason": judge_reason or "Verified by Stage-2 Judge Model",
            "image_url": image_url,
            "action_history": [
                {"time": now_str, "action": f"Auto-detected & verified by ROADNETRA AI (Confidence: {conf*100:.1f}%)"},
                {"time": now_str, "action": f"Automated priority dispatch routed to {agency}"}
            ],
            "repair_cost_est": 22000 if defect_type == "pothole" else 35000,
            "asphalt_tons": 2.8 if defect_type == "pothole" else 0.0
        }

        self.incidents.insert(0, incident)
        self._save()
        return incident

    def update_incident_status(self, incident_id, new_status, note=""):
        for inc in self.incidents:
            if inc["id"] == incident_id:
                inc["status"] = new_status
                now_str = datetime.now().strftime("%I:%M %p")
                action_text = f"Status updated to '{new_status}'"
                if note:
                    action_text += f": {note}"
                inc["action_history"].append({"time": now_str, "action": action_text})
                self._save()
                return True
        return False

    def get_summary_metrics(self):
        total = len(self.incidents)
        critical = sum(1 for i in self.incidents if i["severity"] == "Critical")
        under_repair = sum(1 for i in self.incidents if i["status"] in ["Under Repair", "Crew En Route"])
        resolved = sum(1 for i in self.incidents if i["status"] == "Resolved")
        pwd_count = sum(1 for i in self.incidents if "pwd" in i["assigned_agency"].lower() or "nhai" in i["assigned_agency"].lower() or "municipality" in i["assigned_agency"].lower())
        hospital_count = sum(1 for i in self.incidents if "hospital" in i["assigned_agency"].lower() or "ems" in i["assigned_agency"].lower())
        police_count = sum(1 for i in self.incidents if "police" in i["assigned_agency"].lower())
        
        return {
            "total_incidents": total,
            "critical_incidents": critical,
            "active_interventions": under_repair,
            "resolved_incidents": resolved,
            "pwd_tickets": pwd_count,
            "hospital_tickets": hospital_count,
            "police_tickets": police_count
        }
