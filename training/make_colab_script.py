"""Regenerates colab_model_b.py from build_accident_dataset.py + train_eval_local.py. Run after editing either:  python training/make_colab_script.py"""
import ast, pathlib, re

T = pathlib.Path(__file__).resolve().parent
builder = (T / "build_accident_dataset.py").read_text(encoding="utf-8")
evaluator = (T / "train_eval_local.py").read_text(encoding="utf-8")


def body(src, start_marker, end_marker):
    a = src.index(start_marker)
    b = src.index(end_marker, a) if end_marker else len(src)
    return src[a:b].rstrip() + "\n"


builder_part = body(builder, "KAGGLE_URL =", 'if __name__ == "__main__":')
eval_part = body(evaluator, "def log(msg):", "def main():")
eval_part = eval_part.replace(
    'def latency(model, imgsz):', 'def latency_cpu(model, imgsz):')

HEADER = '''"""RoadNetra AI - Model B (severe_accident) - complete Colab training script.

Builds the dataset from public Kaggle sources, cleans and de-duplicates it, makes a leak-free
70/15/15 split, trains YOLOv8 with 'Vision in the Wild' augmentation, then reports best epoch,
best confidence threshold, precision / recall / F1 / mAP, image-level accuracy, confusion matrix
and latency, and exports accident_best.pt.

HOW TO RUN IN GOOGLE COLAB  (Runtime > Change runtime type > T4 GPU)
---------------------------------------------------------------------
Cell 1:
    from google.colab import drive; drive.mount('/content/drive')      # keeps results if Colab disconnects
    !pip -q install "ultralytics>=8.1.0" albumentations
Cell 2 (either way works):
    a) paste this whole file into the cell and run it, or
    b) upload it via the Files panel, then:  !python /content/colab_model_b.py
Optional flags (option b only, e.g.  !python /content/colab_model_b.py --model yolov8n.pt):
    --model yolov8n.pt            smaller / faster model (default yolov8s.pt)
    --epochs 100 --patience 20    training length and early-stopping patience
    --prebuilt-zip /content/drive/MyDrive/accident_dataset_v3.zip   skip the Kaggle download + build
    --resume                      continue an interrupted run from last.pt
    --eval-only                   re-run every metric on an existing best.pt (no training)
Outputs: /content/accident_best.pt, /content/roadnetra_model_b_results.zip and, with Drive
mounted, everything under /content/drive/MyDrive/roadnetra_accident/.
"""
import argparse
import ctypes
import glob
import json
import pickle
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from ultralytics import YOLO

PIPELINE_CONF = 0.55


# =====================================================================================
# 1. DATASET BUILDER  (download -> filter classes -> clean -> de-duplicate -> split)
# =====================================================================================
'''

MIDDLE = '''

# =====================================================================================
# 2. TRAINING-SET AUGMENTATION + EVALUATION
# =====================================================================================
'''

MAIN = '''

def latency_gpu(model, imgsz):
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    for _ in range(10):
        model.predict(frame, imgsz=imgsz, device=0, verbose=False)
    ts = []
    for _ in range(200):
        s = time.perf_counter(); model.predict(frame, imgsz=imgsz, device=0, verbose=False)
        ts.append((time.perf_counter() - s) * 1000)
    med = float(np.median(ts))
    return dict(gpu=torch.cuda.get_device_name(0), gpu_median_ms=round(med, 1),
                gpu_p95_ms=round(float(np.percentile(ts, 95)), 1), gpu_fps=round(1000 / med, 1),
                meets_25ms_contract=med <= 25)


def prepare_dataset(a):
    data_dir = Path(a.data_dir)
    if a.prebuilt_zip and Path(a.prebuilt_zip).exists() and not (data_dir / "data.yaml").exists():
        log(f"Unpacking pre-built dataset {a.prebuilt_zip}")
        with zipfile.ZipFile(a.prebuilt_zip) as z:
            top = z.namelist()[0].split("/")[0]
            z.extractall(data_dir.parent)
        if (data_dir.parent / top) != data_dir:
            shutil.move(str(data_dir.parent / top), str(data_dir))
    elif not (data_dir / "data.yaml").exists():
        log("Building dataset from Kaggle (download ~1.9 GB, then ~10-15 min of processing)")
        build(data_dir, Path(a.raw_dir), a.seed)
    else:
        log(f"Using existing dataset {data_dir}")
    cfg = yaml.safe_load((data_dir / "data.yaml").read_text())
    cfg["path"] = str(data_dir.resolve())
    (data_dir / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    assert cfg["names"] == {0: "severe_accident"}, cfg["names"]
    return data_dir / "data.yaml"


def print_report(s):
    d, c, td, il = s["epochs"], s["confidence"], s["test_detection"], s["test_image_level"]
    b, p = il["at_best_conf"], il["at_pipeline_conf"]
    lines = [
        "", "=" * 72, " ROADNETRA MODEL B - FINAL REPORT (held-out TEST split)", "=" * 72,
        f" Model                : {s['model']} @ {s['imgsz']} px",
        f" Epochs run / BEST    : {d['epochs_run']} / {d['best_epoch']}  (lowest val loss at epoch {d['lowest_val_loss_epoch']})",
        f" BEST CONFIDENCE      : {c['best_conf']}   (pipeline uses {PIPELINE_CONF})",
        "-" * 72, " Detection (boxes)",
        f"   mAP50 {td['mAP50']:.3f} | mAP50-95 {td['mAP50-95']:.3f} | P {td['precision@best_conf']:.3f} | R {td['recall@best_conf']:.3f} | F1 {td['F1@best_conf']:.3f}",
        "-" * 72, f" Accident / no-accident per image ({il['test_images']} test images: {il['accident_images']} accident, {il['negative_images']} no-accident)",
        f"   {'threshold':>9} {'accuracy':>9} {'precision':>9} {'recall':>7} {'specif.':>8} {'F1':>6} {'false-alarm':>11}",
    ]
    for m in (b, p):
        lines.append(f"   {m['threshold']:>9} {m['accuracy']:>9.3f} {m['precision']:>9.3f} {m['recall']:>7.3f} "
                     f"{m['specificity']:>8.3f} {m['f1']:>6.3f} {m['false_alarm_rate']:>11.3f}")
    lines.append(f"   confusion @best: TP {b['TP']}  FP {b['FP']}  FN {b['FN']}  TN {b['TN']}")
    lat = s["latency"]
    lines += ["-" * 72, f" Latency (720p frame -> {s['imgsz']} px)"]
    if "gpu_median_ms" in lat:
        lines.append(f"   GPU {lat['gpu']}: {lat['gpu_median_ms']} ms ({lat['gpu_fps']} FPS) -> {'PASS' if lat['meets_25ms_contract'] else 'FAIL'} (<=25 ms)")
    lines.append(f"   CPU (this machine): {lat['cpu_median_ms']} ms ({lat['cpu_fps']} FPS) -> {'PASS' if lat['meets_75ms_contract'] else 'FAIL'} (<=75 ms)")
    lines.append("=" * 72)
    print("\\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser(description="RoadNetra Model B: build dataset, train, evaluate, export")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--weather", type=float, default=0.25, help="share of train positives given a rain/fog/blur/glare copy")
    ap.add_argument("--data-dir", default="/content/accident_dataset")
    ap.add_argument("--raw-dir", default="/content/raw_kaggle")
    ap.add_argument("--prebuilt-zip", default="/content/drive/MyDrive/accident_dataset_v3.zip")
    default_project = "/content/drive/MyDrive/roadnetra_accident" if Path("/content/drive/MyDrive").is_dir() else "/content/roadnetra_accident"
    ap.add_argument("--project", default=default_project)
    ap.add_argument("--name", default="model_b_yolov8s")
    ap.add_argument("--out", default="/content")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="skip training; evaluate the existing best.pt")
    ap.add_argument("--fraction", type=float, default=1.0, help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=0)
    # parse_known_args: when this file is pasted into a notebook cell, Jupyter adds its own "-f <kernel.json>"
    a, unknown = ap.parse_known_args()
    unknown = [u for u in unknown if not u.endswith(".json") and u != "-f"]
    if unknown:
        sys.exit(f"Unknown option(s): {unknown}  (see --help)")

    log(f"torch {torch.__version__} | CUDA {torch.cuda.is_available()} | device {a.device}")
    if a.device != "cpu" and not torch.cuda.is_available():
        sys.exit("No GPU found: Runtime > Change runtime type > T4 GPU (or pass --device cpu)")
    if not str(a.project).startswith("/content/drive"):
        log("WARNING: Google Drive not mounted - results are lost if Colab disconnects")

    data_yaml = prepare_dataset(a)
    data_dir = data_yaml.parent
    run_dir = Path(a.project) / a.name

    if a.eval_only:
        log(f"Evaluation only: {run_dir / 'weights' / 'best.pt'}")
    elif a.resume:
        log(f"Resuming {run_dir / 'weights' / 'last.pt'}")
        YOLO(str(run_dir / "weights" / "last.pt")).train(resume=True)
    else:
        add_weather_copies(data_dir, a.weather, a.seed)
        log(f"Training {a.model} on {a.device}: up to {a.epochs} epochs, patience {a.patience}, imgsz {a.imgsz}")
        YOLO(a.model).train(
            data=str(data_yaml), epochs=a.epochs, patience=a.patience, imgsz=a.imgsz, batch=a.batch,
            device=a.device, workers=a.workers, seed=a.seed, fraction=a.fraction,
            cos_lr=True, close_mosaic=10,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=10.0, translate=0.1, scale=0.5,
            perspective=0.0005, fliplr=0.5, mosaic=1.0, mixup=0.15,
            project=a.project, name=a.name, exist_ok=True,
        )
    log("Training finished - evaluating best.pt")

    best = YOLO(str(run_dir / "weights" / "best.pt"))
    assert best.names == {0: "severe_accident"}, best.names
    trained_from = (best.ckpt or {}).get("train_args", {}).get("model", a.model)
    summary = {"model": Path(str(trained_from)).name, "imgsz": a.imgsz, "device": a.device}
    summary["epochs"] = epoch_analysis(run_dir)
    best_conf, summary["confidence"] = confidence_sweep(best, data_yaml, a.imgsz, run_dir)
    summary["test_detection"] = test_detection(best, data_yaml, a.imgsz, best_conf, run_dir)
    summary["test_image_level"] = test_image_level(best, data_dir, a.imgsz, best_conf, run_dir)
    summary["latency"] = latency_cpu(best, a.imgsz)
    if a.device != "cpu":
        summary["latency"].update(latency_gpu(best, a.imgsz))
    report = data_dir / "dataset_report.json"
    if report.exists():
        summary["dataset"] = json.loads(report.read_text())

    (run_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2))
    out = Path(a.out)
    shutil.copy(run_dir / "weights" / "best.pt", out / "accident_best.pt")
    shutil.make_archive(str(out / "roadnetra_model_b_results"), "zip", run_dir)
    print_report(summary)
    log(f"Saved {out / 'accident_best.pt'}  (copy to RoadNetra models/)")
    log(f"Saved {out / 'roadnetra_model_b_results.zip'}  (charts, confusion matrices, summary_metrics.json)")
    log("ALL DONE")


if __name__ == "__main__":
    main()
'''

script = HEADER + builder_part + MIDDLE + eval_part + MAIN
ast.parse(script)
# every name used at module level must be defined once
defs = re.findall(r"^def (\w+)", script, re.M)
dupes = {d for d in defs if defs.count(d) > 1}
assert not dupes, dupes
out = T / "colab_model_b.py"
out.write_text(script, encoding="utf-8")
print("wrote", out, len(script.splitlines()), "lines; functions:", len(defs))
