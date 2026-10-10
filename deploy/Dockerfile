# RoadNetra AI · Hugging Face Docker Space (CPU)
# Built from the folder staged by deploy/publish_space.py (app files + the 3 model weights).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=7860 \
    HOST=0.0.0.0 \
    DEVICE=cpu \
    TORCH_THREADS=2 \
    POTHOLE_MODEL_PATH=models/pothole_best.pt \
    ACCIDENT_MODEL_PATH=models/accident_best.pt \
    JUDGE_MODEL_PATH=models/yolov8n.pt

# libgl/glib for OpenCV, ffmpeg so uploaded phone videos are transcoded to browser-playable H.264
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch wheels (~5x smaller than the default CUDA build)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt gunicorn

COPY . .
# Spaces run the container as uid 1000; the incident store, evidence and uploads must be writable
RUN useradd -m -u 1000 user && mkdir -p data static/evidence static/uploads && chown -R user:user /app
USER user

EXPOSE 7860
# One process keeps the incident store, stream sessions and portals consistent; threads serve concurrent requests
CMD ["gunicorn", "server:app", "--workers", "1", "--threads", "8", "--timeout", "300", "--bind", "0.0.0.0:7860"]
