"""
ROADNETRA AI configuration.

Settings come from two plain .env files (no extra dependency needed):
  FINAL PROJECT/.env           -> backend (models, thresholds, default camera, server)
  FINAL PROJECT/frontend/.env  -> frontend (live map provider + API key, map centre)
Real environment variables override both files. Copy the matching .env.example to start.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ENV = os.path.join(BASE_DIR, ".env")
FRONTEND_ENV = os.path.join(BASE_DIR, "frontend", ".env")


def _parse_env_file(path):
    values = {}
    if not os.path.exists(path):
        return values
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            values[key] = value
    return values


class Settings:
    def __init__(self):
        self.backend = _parse_env_file(BACKEND_ENV)
        self.frontend = _parse_env_file(FRONTEND_ENV)

    def _get(self, source, key, default):
        if key in os.environ and os.environ[key] != "":
            return os.environ[key]
        value = source.get(key, "")
        return value if value != "" else default

    def str(self, key, default="", frontend=False):
        return str(self._get(self.frontend if frontend else self.backend, key, default))

    def float(self, key, default, frontend=False):
        try:
            return float(self._get(self.frontend if frontend else self.backend, key, default))
        except (TypeError, ValueError):
            return float(default)

    def int(self, key, default, frontend=False):
        return int(self.float(key, default, frontend))

    def bool(self, key, default, frontend=False):
        value = str(self._get(self.frontend if frontend else self.backend, key, default)).lower()
        return value in ("1", "true", "yes", "on")

    def path(self, key, default):
        value = self.str(key, default)
        return value if os.path.isabs(value) else os.path.join(BASE_DIR, value)


settings = Settings()

# ---------------- Server ----------------
PORT = settings.int("PORT", 8080)
HOST = settings.str("HOST", "0.0.0.0")

# ---------------- Models ----------------
POTHOLE_MODEL_PATH = settings.path("POTHOLE_MODEL_PATH", "models/pothole_best.pt")
ACCIDENT_MODEL_PATH = settings.path("ACCIDENT_MODEL_PATH", "../models/accident_best.pt")
JUDGE_MODEL_PATH = settings.path("JUDGE_MODEL_PATH", "models/yolov8n.pt")
DEVICE = settings.str("DEVICE", "auto")                  # auto | cpu | cuda | cuda:0 | mps
TORCH_THREADS = settings.int("TORCH_THREADS", max(2, (os.cpu_count() or 6) // 3))

POTHOLE_IMGSZ = settings.int("POTHOLE_IMGSZ", 960)
ACCIDENT_IMGSZ = settings.int("ACCIDENT_IMGSZ", 640)

# Model A (road infrastructure) per-class thresholds
POTHOLE_CONF = settings.float("POTHOLE_CONF", 0.28)
DAMAGED_SIGN_CONF = settings.float("DAMAGED_SIGN_CONF", 0.25)
DAMAGED_DIVIDER_CONF = settings.float("DAMAGED_DIVIDER_CONF", 0.20)
FADED_ZEBRA_CONF = settings.float("FADED_ZEBRA_CONF", 0.20)
INFRA_CONFIRM_FRAMES = settings.int("INFRA_CONFIRM_FRAMES", 2)

# Model B (severe accident). Tuned on the 273-image held-out test set:
# accept >= ACCIDENT_CONF, or >= ACCIDENT_MIN_CONF when the judge sees a vehicle in context.
ACCIDENT_CONF = settings.float("ACCIDENT_CONF", 0.55)
ACCIDENT_MIN_CONF = settings.float("ACCIDENT_MIN_CONF", 0.40)
ACCIDENT_CONFIRM_FRAMES = settings.int("ACCIDENT_CONFIRM_FRAMES", 3)
ACCIDENT_CONFIRM_WINDOW = settings.int("ACCIDENT_CONFIRM_WINDOW", 5)

# Duplicate merging windows (minutes)
MERGE_WINDOW_INFRA_MIN = settings.float("MERGE_WINDOW_INFRA_MIN", 720)
MERGE_WINDOW_ACCIDENT_MIN = settings.float("MERGE_WINDOW_ACCIDENT_MIN", 15)

# ---------------- Default camera / location ----------------
DEFAULT_CAMERA_ID = settings.str("DEFAULT_CAMERA_ID", "CAM-084")
SEED_DEMO_INCIDENTS = settings.bool("SEED_DEMO_INCIDENTS", True)

# ---------------- Authority resources (portal operations are computed from incidents + these) ----------------
HOSPITAL_NAME = settings.str("HOSPITAL_NAME", "District Trauma Centre")
HOSPITAL_LAT = settings.float("HOSPITAL_LAT", 28.4211)   # receiving trauma centre (ambulance ETAs start here)
HOSPITAL_LNG = settings.float("HOSPITAL_LNG", 77.0130)
AMBULANCE_FLEET = settings.int("AMBULANCE_FLEET", 6)
TRAUMA_BAYS = settings.int("TRAUMA_BAYS", 4)
PCR_FLEET = settings.int("PCR_FLEET", 14)
VMS_BOARDS = settings.int("VMS_BOARDS", 8)
ASPHALT_RATE_PER_TON = settings.float("ASPHALT_RATE_PER_TON", 4800)
LABOUR_PER_POTHOLE = settings.float("LABOUR_PER_POTHOLE", 900)

# ---------------- Frontend (frontend/.env) ----------------
MAP_PROVIDER = settings.str("MAP_PROVIDER", "auto", frontend=True).lower()
MAP_API_KEY = settings.str("MAP_API_KEY", "", frontend=True)
MAP_STYLE = settings.str("MAP_STYLE", "dark", frontend=True)
MAP_DEFAULT_LAT = settings.float("MAP_DEFAULT_LAT", 28.3589, frontend=True)
MAP_DEFAULT_LNG = settings.float("MAP_DEFAULT_LNG", 76.9387, frontend=True)
MAP_DEFAULT_ZOOM = settings.int("MAP_DEFAULT_ZOOM", 11, frontend=True)
ACCIDENT_STREAM_WIDTH = settings.int("ACCIDENT_STREAM_WIDTH", 640, frontend=True)
INFRA_STREAM_WIDTH = settings.int("INFRA_STREAM_WIDTH", 1280, frontend=True)
INFRA_INTERVAL_MS = settings.int("INFRA_INTERVAL_MS", 1000, frontend=True)
INCIDENT_POLL_MS = settings.int("INCIDENT_POLL_MS", 3000, frontend=True)


def frontend_config():
    """Public settings safe to send to the browser (map keys are browser keys by design)."""
    provider = MAP_PROVIDER
    if provider == "auto":
        if not MAP_API_KEY:
            provider = "esri"
        elif MAP_API_KEY.startswith("AIza"):
            provider = "google"
        elif MAP_API_KEY.startswith(("pk.", "sk.")):
            provider = "mapbox"
        else:
            provider = "maptiler"
    return {
        "map": {
            "provider": provider,
            "apiKey": MAP_API_KEY,
            "style": MAP_STYLE,
            "center": [MAP_DEFAULT_LAT, MAP_DEFAULT_LNG],
            "zoom": MAP_DEFAULT_ZOOM,
        },
        "stream": {
            "accidentWidth": ACCIDENT_STREAM_WIDTH,
            "infraWidth": INFRA_STREAM_WIDTH,
            "infraIntervalMs": INFRA_INTERVAL_MS,
            "incidentPollMs": INCIDENT_POLL_MS,
        },
        "defaultCameraId": DEFAULT_CAMERA_ID,
    }
