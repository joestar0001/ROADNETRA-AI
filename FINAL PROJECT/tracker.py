import math
import numpy as np

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def get_centroid(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

def centroid_distance(boxA, boxB):
    c1 = get_centroid(boxA)
    c2 = get_centroid(boxB)
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

class TrackedDefect:
    def __init__(self, track_id, class_name, initial_box, initial_conf, initial_crop, frame_idx, verdict_reason="", min_hits=2, window=None):
        self.min_hits = max(1, int(min_hits))
        # Optional sliding window: confirm only if min_hits hits fall within the last `window` processed frames
        self.window = window
        self.hit_frames = [frame_idx]
        self.track_id = track_id
        self.class_name = class_name
        self.last_box = list(initial_box)
        self.best_box = list(initial_box)
        self.highest_conf = float(initial_conf)
        self.best_crop = initial_crop
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.hits = 1
        self.time_since_update = 0
        self.is_confirmed = self._enough_hits(frame_idx)
        self.verdict_reason = verdict_reason
        self.payload = {}          # extra data kept from the best detection (e.g. full frame for evidence)
        self.incident_id = None    # set once the confirmed track has been dispatched

    def _enough_hits(self, frame_idx):
        if not self.window:
            return self.hits >= self.min_hits
        recent = [f for f in self.hit_frames if f > frame_idx - self.window]
        return len(recent) >= self.min_hits

    def update(self, box, conf, crop, frame_idx, verdict_reason=""):
        self.hit_frames = self.hit_frames[-15:] + [frame_idx]
        self.last_box = list(box)
        self.last_frame = frame_idx
        self.hits += 1
        self.time_since_update = 0
        if not self.is_confirmed and self._enough_hits(frame_idx):
            self.is_confirmed = True
        if conf > self.highest_conf:
            self.highest_conf = float(conf)
            self.best_box = list(box)
            if crop is not None:
                self.best_crop = crop
        if verdict_reason:
            self.verdict_reason = verdict_reason

class DefectTracker:
    """
    Spatial-Temporal Deduplication & Tracking Engine.
    
    Prevents stationary objects (potholes, signs, zebra crossings, dividers)
    from being counted repeatedly across consecutive video frames.
    """
    def __init__(self, iou_thresh=0.20, max_dist=100.0, max_age=25, min_hits=2, window=None):
        self.iou_thresh = iou_thresh
        self.max_dist = max_dist
        self.max_age = max_age
        self.min_hits = min_hits
        self.window = window
        self.tracks = []
        self.confirmed_defects = {}  # track_id -> TrackedDefect
        self.next_id = 1

    def update(self, detections, frame_idx):
        """
        detections: list of dicts:
          [
            {
               'box': [x1, y1, x2, y2],
               'class_name': str,
               'conf': float,
               'crop_rgb': numpy array or None,
               'verdict_reason': str
            }, ...
          ]
        Returns:
          list of tuples: (track_id, is_confirmed, detection_dict)
        """
        for t in self.tracks:
            t.time_since_update += 1

        matched_track_indices = set()
        matched_det_indices = set()
        matched_results = []

        # Greedy matching by IoU & Centroid Distance for same class
        for d_idx, det in enumerate(detections):
            best_t_idx = -1
            best_score = -1.0

            for t_idx, track in enumerate(self.tracks):
                if t_idx in matched_track_indices:
                    continue
                if track.class_name != det['class_name']:
                    continue

                iou = compute_iou(track.last_box, det['box'])
                dist = centroid_distance(track.last_box, det['box'])
                
                # Check spatial continuity
                diag = math.sqrt((det['box'][2] - det['box'][0])**2 + (det['box'][3] - det['box'][1])**2)
                allowed_dist = max(self.max_dist, diag * 0.75)

                if iou >= self.iou_thresh or dist <= allowed_dist:
                    score = iou + max(0.0, 1.0 - (dist / allowed_dist))
                    if score > best_score:
                        best_score = score
                        best_t_idx = t_idx

            if best_t_idx != -1:
                track = self.tracks[best_t_idx]
                track.update(
                    det['box'],
                    det['conf'],
                    det.get('crop_rgb'),
                    frame_idx,
                    det.get('verdict_reason', "")
                )
                matched_track_indices.add(best_t_idx)
                matched_det_indices.add(d_idx)
                matched_results.append((track.track_id, track.is_confirmed, det))
                
                if track.is_confirmed:
                    self.confirmed_defects[track.track_id] = track

        # Unmatched detections become new tracks
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_indices:
                new_track = TrackedDefect(
                    track_id=self.next_id,
                    class_name=det['class_name'],
                    initial_box=det['box'],
                    initial_conf=det['conf'],
                    initial_crop=det.get('crop_rgb'),
                    frame_idx=frame_idx,
                    verdict_reason=det.get('verdict_reason', ""),
                    min_hits=self.min_hits,
                    window=self.window
                )
                self.next_id += 1
                self.tracks.append(new_track)
                if new_track.is_confirmed:
                    self.confirmed_defects[new_track.track_id] = new_track
                matched_results.append((new_track.track_id, new_track.is_confirmed, det))

        # Prune inactive tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return matched_results

    def get_unique_counts(self):
        """Returns de-duplicated count dictionary of confirmed defects."""
        counts = {
            "pothole": 0,
            "damaged_sign": 0,
            "damaged_divider": 0,
            "faded_zebra_crossing": 0,
            "severe_accident": 0
        }
        for track in self.confirmed_defects.values():
            if track.class_name in counts:
                counts[track.class_name] += 1
        return counts

    def get_track(self, track_id):
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return self.confirmed_defects.get(track_id)

    def get_all_confirmed_defects(self):
        """Returns list of confirmed defects sorted by first detection frame."""
        return sorted(self.confirmed_defects.values(), key=lambda t: t.first_frame)
