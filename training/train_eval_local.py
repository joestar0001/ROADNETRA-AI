"""Local (CPU/GPU) version of model_b_kaggle_train_eval.ipynb.

Runs: weather copies -> training with early stopping -> best epoch -> best confidence ->
held-out test metrics (detection + image-level) -> mistakes grid -> latency -> summary JSON,
then copies best.pt to models/accident_best.pt. All charts are saved as PNGs in the run folder.

Usage:
    python training/train_eval_local.py --data data/accident_dataset/data.yaml --model yolov8n.pt --epochs 40
    python training/train_eval_local.py ... --resume      # continue an interrupted run
    python training/train_eval_local.py ... --eval-only   # re-run all metrics on an existing best.pt
"""
import argparse
import ctypes
import glob
import json
import random
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_CONF = 0.55


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def keep_awake():
    """Stop Windows sleeping while training (released automatically when the process exits)."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def label_of(img):
    img = Path(img)
    return img.parents[2] / "labels" / img.parent.name / (img.stem + ".txt")


def add_weather_copies(data_dir, fraction, seed):
    import albumentations as A

    train = data_dir / "images" / "train"
    if any(train.glob("wx_*")):
        log("Weather copies already present, skipping")
        return
    aug = A.OneOf([
        A.RandomRain(p=1), A.RandomFog(p=1), A.MotionBlur(blur_limit=(3, 9), p=1), A.GaussianBlur(p=1),
        A.RandomSunFlare(p=1),
        A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.2), contrast_limit=(-0.4, 0.1), p=1),
    ], p=1)
    pos = sorted(p for p in train.glob("*.jpg") if "_pos_" in p.name)
    picks = random.Random(seed).sample(pos, int(len(pos) * fraction))
    for p in picks:
        out = aug(image=cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))["image"]
        q = p.with_name("wx_" + p.name)
        cv2.imwrite(str(q), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        shutil.copy(label_of(p), label_of(q))
    log(f"Weather copies added to train: {len(picks)} (train now {len(list(train.glob('*.jpg')))} images)")


def epoch_analysis(run_dir):
    res = pd.read_csv(run_dir / "results.csv")
    res.columns = res.columns.str.strip()
    res["fitness"] = 0.1 * res["metrics/mAP50(B)"] + 0.9 * res["metrics/mAP50-95(B)"]
    res["val/total_loss"] = res[["val/box_loss", "val/cls_loss", "val/dfl_loss"]].sum(axis=1)
    res["train/total_loss"] = res[["train/box_loss", "train/cls_loss", "train/dfl_loss"]].sum(axis=1)
    off = 1 if res.epoch.min() == 0 else 0
    best = res.loc[res.fitness.idxmax()]
    min_val = res.loc[res["val/total_loss"].idxmin()]

    fig, ax = plt.subplots(1, 3, figsize=(20, 4.5))
    ax[0].plot(res.epoch + off, res["train/total_loss"], label="train")
    ax[0].plot(res.epoch + off, res["val/total_loss"], label="val")
    ax[0].set_title("Total loss (box + cls + dfl)"); ax[0].legend()
    for k in ("metrics/precision(B)", "metrics/recall(B)"):
        ax[1].plot(res.epoch + off, res[k], label=k.split("/")[1])
    ax[1].set_title("Precision / Recall per epoch"); ax[1].legend()
    ax[2].plot(res.epoch + off, res["metrics/mAP50(B)"], label="mAP50")
    ax[2].plot(res.epoch + off, res["metrics/mAP50-95(B)"], label="mAP50-95")
    ax[2].set_title(f"mAP per epoch (best = {int(best.epoch) + off})"); ax[2].legend()
    for a in ax:
        a.axvline(best.epoch + off, ls="--", c="grey"); a.set_xlabel("epoch")
    fig.savefig(run_dir / "rn_epoch_curves.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    return dict(epochs_run=len(res), best_epoch=int(best.epoch) + off,
                best_epoch_val_mAP50=round(float(best["metrics/mAP50(B)"]), 4),
                best_epoch_val_mAP50_95=round(float(best["metrics/mAP50-95(B)"]), 4),
                lowest_val_loss_epoch=int(min_val.epoch) + off)


def confidence_sweep(model, data_yaml, imgsz, run_dir):
    val = model.val(data=str(data_yaml), split="val", imgsz=imgsz, plots=True, verbose=False,
                    project=str(run_dir), name="val_best", exist_ok=True)
    px, p_c, r_c, f1_c = val.box.px, val.box.p_curve[0], val.box.r_curve[0], val.box.f1_curve[0]
    i_best = int(np.argmax(f1_c))
    i_pipe = int(np.argmin(abs(px - PIPELINE_CONF)))
    fig = plt.figure(figsize=(8, 4.5))
    plt.plot(px, p_c, label="Precision"); plt.plot(px, r_c, label="Recall"); plt.plot(px, f1_c, label="F1", lw=2.5)
    plt.axvline(px[i_best], ls="--", c="k", label=f"best F1 @ {px[i_best]:.2f}")
    plt.axvline(PIPELINE_CONF, ls=":", c="r", label=f"pipeline {PIPELINE_CONF}")
    plt.xlabel("confidence threshold"); plt.legend(); plt.title("Validation: metric vs confidence")
    fig.savefig(run_dir / "rn_confidence_sweep.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    return float(px[i_best]), dict(
        best_conf=round(float(px[i_best]), 3),
        val_at_best_conf=dict(precision=round(float(p_c[i_best]), 4), recall=round(float(r_c[i_best]), 4), f1=round(float(f1_c[i_best]), 4)),
        val_at_pipeline_conf=dict(precision=round(float(p_c[i_pipe]), 4), recall=round(float(r_c[i_pipe]), 4), f1=round(float(f1_c[i_pipe]), 4)),
        val_mAP50=round(float(val.box.map50), 4), val_mAP50_95=round(float(val.box.map), 4))


def test_detection(model, data_yaml, imgsz, best_conf, run_dir):
    t = model.val(data=str(data_yaml), split="test", imgsz=imgsz, plots=True, verbose=False,
                  project=str(run_dir), name="test_best", exist_ok=True)
    i = int(np.argmin(abs(t.box.px - best_conf)))
    return {"mAP50": round(float(t.box.map50), 4), "mAP50-95": round(float(t.box.map), 4),
            "precision@best_conf": round(float(t.box.p_curve[0][i]), 4),
            "recall@best_conf": round(float(t.box.r_curve[0][i]), 4),
            "F1@best_conf": round(float(t.box.f1_curve[0][i]), 4),
            "speed_ms_per_image": {k: round(v, 2) for k, v in t.speed.items()}}


def test_image_level(model, data_dir, imgsz, best_conf, run_dir):
    imgs = sorted(glob.glob(str(data_dir / "images" / "test" / "*")))
    y_true = [int(bool(open(label_of(p)).read().strip())) for p in imgs]
    scores = [float(r.boxes.conf.max()) if len(r.boxes) else 0.0
              for r in model.predict(imgs, conf=0.001, imgsz=imgsz, verbose=False, stream=True)]

    def metrics(th):
        y_pred = [int(s >= th) for s in scores]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
        return dict(threshold=round(th, 3), accuracy=round((tp + tn) / len(y_true), 4),
                    precision=round(tp / max(1, tp + fp), 4), recall=round(tp / max(1, tp + fn), 4),
                    specificity=round(tn / max(1, tn + fp), 4), f1=round(2 * tp / max(1, 2 * tp + fp + fn), 4),
                    false_alarm_rate=round(fp / max(1, fp + tn), 4), TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn)), y_pred

    at_best, pred = metrics(best_conf)
    at_pipe, _ = metrics(PIPELINE_CONF)
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, pred, labels=[0, 1]), display_labels=["no accident", "severe accident"])
    disp.plot(cmap="Blues"); plt.title(f"Test confusion matrix @ conf {best_conf:.2f}")
    plt.savefig(run_dir / "rn_test_confusion_matrix.png", dpi=110, bbox_inches="tight"); plt.close("all")

    wrong = [(p, t, s) for p, t, s in zip(imgs, y_true, scores) if int(s >= best_conf) != t][:8]
    if wrong:
        fig, axes = plt.subplots(2, 4, figsize=(20, 9))
        for a, (p, t, s) in zip(axes.flat, wrong):
            a.imshow(cv2.cvtColor(model.predict(p, conf=best_conf, imgsz=imgsz, verbose=False)[0].plot(), cv2.COLOR_BGR2RGB))
            a.set_title(f'{"MISSED crash" if t else "FALSE ALARM"}  max conf {s:.2f}', fontsize=10)
        for a in axes.flat:
            a.axis("off")
        fig.tight_layout(); fig.savefig(run_dir / "rn_test_mistakes.png", dpi=90); plt.close(fig)
    return dict(test_images=len(y_true), accident_images=sum(y_true), negative_images=len(y_true) - sum(y_true),
                at_best_conf=at_best, at_pipeline_conf=at_pipe)


def latency(model, imgsz):
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    for _ in range(5):
        model.predict(frame, imgsz=imgsz, device="cpu", verbose=False)
    ts = []
    for _ in range(50):
        s = time.perf_counter(); model.predict(frame, imgsz=imgsz, device="cpu", verbose=False)
        ts.append((time.perf_counter() - s) * 1000)
    med = float(np.median(ts))
    return dict(cpu_median_ms=round(med, 1), cpu_p95_ms=round(float(np.percentile(ts, 95)), 1),
                cpu_fps=round(1000 / med, 1), meets_75ms_contract=med <= 75)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "accident_dataset" / "data.yaml"))
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--weather", type=float, default=0.25)
    ap.add_argument("--project", default=str(ROOT / "runs"))
    ap.add_argument("--name", default="accident_yolov8n_local")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="skip training; evaluate the existing best.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    keep_awake()
    data_yaml = Path(a.data).resolve()
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
            device=a.device, workers=a.workers, seed=a.seed, cos_lr=True, close_mosaic=10,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=10.0, translate=0.1, scale=0.5,
            perspective=0.0005, fliplr=0.5, mosaic=1.0, mixup=0.15,
            project=a.project, name=a.name, exist_ok=True,
        )
    log("Training finished - evaluating")

    best = YOLO(str(run_dir / "weights" / "best.pt"))
    assert best.names == {0: "severe_accident"}, best.names
    summary = {"model": a.model, "imgsz": a.imgsz, "device": a.device}
    summary["epochs"] = epoch_analysis(run_dir); log(f"Epochs: {summary['epochs']}")
    best_conf, summary["confidence"] = confidence_sweep(best, data_yaml, a.imgsz, run_dir); log(f"Confidence: {summary['confidence']}")
    summary["test_detection"] = test_detection(best, data_yaml, a.imgsz, best_conf, run_dir); log(f"Test detection: {summary['test_detection']}")
    summary["test_image_level"] = test_image_level(best, data_dir, a.imgsz, best_conf, run_dir); log(f"Test image-level: {summary['test_image_level']}")
    summary["latency"] = latency(best, a.imgsz); log(f"Latency: {summary['latency']}")
    report = data_dir / "dataset_report.json"
    if report.exists():
        summary["dataset"] = json.loads(report.read_text())["splits"]

    (run_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2))
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(run_dir / "weights" / "best.pt", ROOT / "models" / "accident_best.pt")
    log(f"Saved {ROOT / 'models' / 'accident_best.pt'} and {run_dir / 'summary_metrics.json'}")
    print(json.dumps(summary, indent=2))
    log("ALL DONE")


if __name__ == "__main__":
    main()
