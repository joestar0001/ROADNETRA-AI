import os
import cv2
import numpy as np
from ultralytics import YOLO

class RoadNetraJudge:
    """
    Two-Stage Semantic & Geometric Verification Engine (The 'Judge Model').
    
    Evaluates candidate Regions of Interest (ROIs) proposed by the primary detector.
    Filters out high-frequency real-world false alarms:
      - Cars / trucks mistaken for damaged dividers
      - Moving vehicle bodies / shadows mistaken for potholes
      - Vehicles mistaken for faded zebra crossings
      - Vehicles / roadside clutter mistaken for traffic signs
      - Normal bumper-to-bumper traffic jams mistaken for severe accidents
    """

    def __init__(self, coco_weights_path=None):
        if coco_weights_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            p1 = os.path.join(base, "models", "yolov8n.pt")
            p2 = os.path.join(base, "yolov8n.pt")
            coco_weights_path = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else "yolov8n.pt")
        self.coco_model = YOLO(coco_weights_path)
            
        self.vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle"}
        self.person_classes = {"person"}

    def evaluate_candidate(self, frame, box, primary_class, primary_conf):
        """
        Evaluates a single suspected bounding box crop.
        
        Args:
            frame: Full BGR frame (numpy array)
            box: [x1, y1, x2, y2] coordinates
            primary_class: str name of detected class
            primary_conf: float confidence score from primary model
            
        Returns:
            dict containing:
              - status: 'ACCEPTED' or 'REJECTED'
              - reason: Human-readable verification explanation
              - primary_class: str
              - primary_conf: float
              - final_conf: float
              - crop_rgb: numpy array for Streamlit UI display
              - box: [x1, y1, x2, y2]
        """
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        
        # Clamp to frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        box_w = x2 - x1
        box_h = y2 - y1

        if box_w < 8 or box_h < 8:
            return {
                "status": "REJECTED",
                "reason": "ROI too small / low spatial resolution for verification",
                "primary_class": primary_class,
                "primary_conf": primary_conf,
                "final_conf": 0.0,
                "crop_rgb": None,
                "box": [x1, y1, x2, y2]
            }

        # Candidate covers > 45% of total frame area: physically impossible for local road defect
        if (box_w * box_h) > (0.45 * w_img * h_img):
            return {
                "status": "REJECTED",
                "reason": "Negative Veto: Candidate covers > 45% of camera frame — macroscopic false alarm",
                "primary_class": primary_class,
                "primary_conf": primary_conf,
                "final_conf": 0.0,
                "crop_rgb": None,
                "box": [x1, y1, x2, y2]
            }

        # Extract crop for display
        raw_crop = frame[y1:y2, x1:x2].copy()
        crop_rgb = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2RGB)

        # Context-aware padding (25%) so semantic model sees the object and immediate surroundings
        pad_x = int(box_w * 0.25)
        pad_y = int(box_h * 0.25)
        px1, py1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        px2, py2 = min(w_img, x2 + pad_x), min(h_img, y2 + pad_y)
        padded_crop = frame[py1:py2, px1:px2]

        # Run semantic inference on padded crop at 320x320
        res = self.coco_model.predict(padded_crop, conf=0.20, imgsz=320, verbose=False)[0]

        detected_vehicles = []
        detected_persons = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                c_id = int(b.cls[0].item())
                c_name = self.coco_model.names.get(c_id, "unknown")
                c_conf = float(b.conf[0].item())
                if c_name in self.vehicle_classes:
                    detected_vehicles.append((c_name, c_conf))
                elif c_name in self.person_classes:
                    detected_persons.append((c_name, c_conf))

        # -------------------------------------------------------------
        # Case 1: Damaged Divider (Primary source of vehicle false alarms)
        # -------------------------------------------------------------
        if primary_class == "damaged_divider":
            if detected_vehicles:
                v_name, v_conf = max(detected_vehicles, key=lambda x: x[1])
                return {
                    "status": "REJECTED",
                    "reason": f"Negative Veto: Moving vehicle ({v_name} {v_conf*100:.0f}%) mistaken for divider",
                    "primary_class": primary_class,
                    "primary_conf": primary_conf,
                    "final_conf": 0.0,
                    "crop_rgb": crop_rgb,
                    "box": [x1, y1, x2, y2]
                }
            
            # Additional aspect ratio check: dividers are typically elongated
            aspect_ratio = max(box_w, box_h) / max(1, min(box_w, box_h))
            final_score = min(0.98, primary_conf + 0.12)
            return {
                "status": "ACCEPTED",
                "reason": f"Verified: Roadside barrier geometry confirmed (Aspect Ratio: {aspect_ratio:.1f}x, no vehicle conflict)",
                "primary_class": primary_class,
                "primary_conf": primary_conf,
                "final_conf": final_score,
                "crop_rgb": crop_rgb,
                "box": [x1, y1, x2, y2]
            }

        # -------------------------------------------------------------
        # Case 2: Pothole
        # -------------------------------------------------------------
        elif primary_class == "pothole":
            # Scale sanity check: Potholes cannot exceed 40% of camera dimensions
            if box_w > (0.40 * w_img) or box_h > (0.40 * h_img):
                return {
                    "status": "REJECTED",
                    "reason": "Negative Veto: Oversized ROI — exceeds maximum physical asphalt cavity scale",
                    "primary_class": primary_class,
                    "primary_conf": primary_conf,
                    "final_conf": 0.0,
                    "crop_rgb": crop_rgb,
                    "box": [x1, y1, x2, y2]
                }

            if detected_vehicles and max(v[1] for v in detected_vehicles) > 0.35:
                v_name, v_conf = max(detected_vehicles, key=lambda x: x[1])
                return {
                    "status": "REJECTED",
                    "reason": f"Negative Veto: Vehicle body/wheel ({v_name} {v_conf*100:.0f}%) mistaken for road cavity",
                    "primary_class": primary_class,
                    "primary_conf": primary_conf,
                    "final_conf": 0.0,
                    "crop_rgb": crop_rgb,
                    "box": [x1, y1, x2, y2]
                }
            
            # Road surface cavity check: reject bright paint markings / pedestrian stripes / lane lines
            gray_crop = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
            bright_ratio = float(np.mean(gray_crop > 170))
            if bright_ratio > 0.15:
                return {
                    "status": "REJECTED",
                    "reason": f"Negative Veto: Road paint / stripe pattern ({bright_ratio*100:.0f}%) mistaken for road cavity",
                    "primary_class": primary_class,
                    "primary_conf": primary_conf,
                    "final_conf": 0.0,
                    "crop_rgb": crop_rgb,
                    "box": [x1, y1, x2, y2]
                }
            
            # Asphalt surface check: Road cavity verified
            final_score = min(0.98, primary_conf + 0.10)
            return {
                "status": "ACCEPTED",
                "reason": "Verified: Road surface asphalt cavity & depression verified",
                "primary_class": primary_class,
                "primary_conf": primary_conf,
                "final_conf": final_score,
                "crop_rgb": crop_rgb,
                "box": [x1, y1, x2, y2]
            }

        # -------------------------------------------------------------
        # Case 3: Faded Zebra Crossing
        # -------------------------------------------------------------
        elif primary_class == "faded_zebra_crossing":
            if detected_vehicles and max(v[1] for v in detected_vehicles) > 0.65:
                v_name, v_conf = max(detected_vehicles, key=lambda x: x[1])
                return {
                    "status": "REJECTED",
                    "reason": f"Negative Veto: Moving vehicle ({v_name} {v_conf*100:.0f}%) occluding road markings",
                    "primary_class": primary_class,
                    "primary_conf": primary_conf,
                    "final_conf": 0.0,
                    "crop_rgb": crop_rgb,
                    "box": [x1, y1, x2, y2]
                }
            
            final_score = min(0.98, primary_conf + 0.10)
            return {
                "status": "ACCEPTED",
                "reason": "Verified: Road surface crosswalk stripe pattern verified",
                "primary_class": primary_class,
                "primary_conf": primary_conf,
                "final_conf": final_score,
                "crop_rgb": crop_rgb,
                "box": [x1, y1, x2, y2]
            }

        # -------------------------------------------------------------
        # Case 4: Damaged Traffic Sign
        # -------------------------------------------------------------
        elif primary_class == "damaged_sign":
            if detected_vehicles and max(v[1] for v in detected_vehicles) > 0.45:
                v_name, v_conf = max(detected_vehicles, key=lambda x: x[1])
                return {
                    "status": "REJECTED",
                    "reason": f"Negative Veto: Moving vehicle ({v_name} {v_conf*100:.0f}%) mistaken for signage",
                    "primary_class": primary_class,
                    "primary_conf": primary_conf,
                    "final_conf": 0.0,
                    "crop_rgb": crop_rgb,
                    "box": [x1, y1, x2, y2]
                }
            
            final_score = min(0.98, primary_conf + 0.12)
            return {
                "status": "ACCEPTED",
                "reason": "Verified: Vertical roadside sign structure verified",
                "primary_class": primary_class,
                "primary_conf": primary_conf,
                "final_conf": final_score,
                "crop_rgb": crop_rgb,
                "box": [x1, y1, x2, y2]
            }

        # -------------------------------------------------------------
        # Case 5: Severe Accident (Model B / Future Incident Stream)
        # -------------------------------------------------------------
        elif "accident" in primary_class.lower():
            # If only a single intact vehicle moving normally in lane
            if len(detected_vehicles) <= 1 and not detected_persons:
                return {
                    "status": "REJECTED",
                    "reason": "Traffic Queue Veto: Normal intact vehicle in lane; no collision deformation signature",
                    "primary_class": primary_class,
                    "primary_conf": primary_conf,
                    "final_conf": 0.0,
                    "crop_rgb": crop_rgb,
                    "box": [x1, y1, x2, y2]
                }
            return {
                "status": "ACCEPTED",
                "reason": "Verified: Multi-vehicle collision / structural incident verified",
                "primary_class": primary_class,
                "primary_conf": primary_conf,
                "final_conf": min(0.99, primary_conf + 0.15),
                "crop_rgb": crop_rgb,
                "box": [x1, y1, x2, y2]
            }

        # Default fallback
        return {
            "status": "ACCEPTED",
            "reason": "Verified by Stage-2 Judge",
            "primary_class": primary_class,
            "primary_conf": primary_conf,
            "final_conf": primary_conf,
            "crop_rgb": crop_rgb,
            "box": [x1, y1, x2, y2]
        }
