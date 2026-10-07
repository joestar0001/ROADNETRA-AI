"""Verifies models/accident_best.pt against the Model B contract on this machine.

Usage:
    python training/benchmark_local.py                     # contract + latency
    python training/benchmark_local.py --video clip.mp4    # also preview detections on a CCTV clip
"""
import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    raise SystemExit('ultralytics is not installed: run  pip install "ultralytics>=8.1.0"  (in Colab: !pip install -q "ultralytics>=8.1.0")')

# __file__ is undefined when this is pasted into a notebook cell; fall back to the working directory
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
TARGET_MS = {"cuda": 25, "mps": 25, "cpu": 75}


def devices():
    yield "cpu"
    if torch.cuda.is_available():
        yield "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        yield "mps"


def bench(model, device, n=50):
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    for _ in range(5):
        model.predict(frame, imgsz=640, device=device, verbose=False)
    times = []
    for _ in range(n):
        start = time.perf_counter()
        model.predict(frame, imgsz=640, device=device, verbose=False)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times), sorted(times)[int(0.95 * len(times)) - 1]


def preview(model, video):
    import cv2

    cap = cv2.VideoCapture(video)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, imgsz=640, conf=0.55, verbose=False)[0]
        cv2.imshow("Model B preview (q to quit)", result.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "models" / "accident_best.pt"))
    parser.add_argument("--video")
    args, _ = parser.parse_known_args()  # tolerate Jupyter's own -f argument

    model = YOLO(args.weights)
    assert model.names == {0: "severe_accident"}, f"class map mismatch: {model.names}"
    imgsz = model.ckpt.get("train_args", {}).get("imgsz")
    assert imgsz == 640, f"trained at imgsz={imgsz}, contract requires 640"
    print(f"Contract OK: {model.names}, imgsz {imgsz}")

    for device in devices():
        median, p95 = bench(model, device)
        verdict = "PASS" if median <= TARGET_MS[device] else "FAIL"
        print(f"{device:>4}: median {median:6.1f} ms | p95 {p95:6.1f} ms | ~{1000 / median:4.1f} FPS -> {verdict} (<= {TARGET_MS[device]} ms)")

    if args.video:
        preview(model, args.video)


if __name__ == "__main__":
    main()
