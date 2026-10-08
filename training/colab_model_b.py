"""RoadNetra AI - Model B (severe_accident) - complete Colab training script.

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
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/{}"

# pos: classes that count as severe_accident (remapped to 0)
# conflict: lesser-damage classes; an image containing them is skipped, since dropping
#           their boxes would teach the model that a damaged car is background
# negative: classes that mark intact vehicles; images with only these become empty-label negatives
SOURCES = {
    "amedeo": dict(
        kaggle="amedeograndi/accidents-detection-dataset",
        names={0: "moderate", 1: "severe"},
        pos={1}, conflict={0}, negative=set(),
        quota_pos=1100, quota_neg=0,
    ),
    "marsl": dict(
        kaggle="marslanarshad/car-accidents-and-deformation-datasetannotated",
        names={0: "no_accident", 1: "minor", 2: "moderate", 3: "severe", 4: "totaled_fire_flipped"},
        # its no_accident frames include pre/post-crash video frames, so they are not used as negatives
        pos={3, 4}, conflict={1, 2}, negative=set(),
        quota_pos=450, quota_neg=0,
    ),
    "mehwish": dict(
        kaggle="mehwishtahir722/accident-and-nonaccident-dataset-for-yolo",
        names={0: "accident", 1: "non_accident"},
        # simulated CCTV; the only clean negatives found, so pos is matched to stop
        # "synthetic look = no accident" becoming a shortcut
        pos={0}, conflict=set(), negative={1},
        quota_pos=200, quota_neg=210,
    ),
    "sg": dict(
        # real Singapore LTA traffic cameras, hand-labelled by density; heavy traffic / jams are the
        # key false-alarm case ("stopped cars are not a crash"). Image-level labels only -> all negatives.
        kaggle="rahat52/traffic-density-singapore",
        names={}, pos=set(), conflict=set(), negative=set(),
        negative_folders={"Medium", "High", "Traffic Jam"},
        camera_sim=0.85,   # fixed cameras: group each camera's shots so a camera never spans two splits
        quota_pos=0, quota_neg=150,
    ),
}

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}
MAX_SIDE = 640
MIN_BOX_AREA = 0.0004      # boxes under 0.04 % of the frame are annotation noise
GROUP_BITS = 14            # images within this many of 256 hash bits (any flip/rotation, chained) share a split
COPY_BITS = 8              # within a group, an image this close to one already kept is a copy and is dropped
SCENE_SIM = 0.95          # thumbnail cosine similarity at/above which two images are the same scene


def download(slug, dst):
    if dst.exists() and zipfile.is_zipfile(dst):
        return dst
    print(f"Downloading {slug} ...")
    req = urllib.request.Request(KAGGLE_URL.format(slug), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    return dst


def fit_max_side(bgr):
    h, w = bgr.shape[:2]
    s = MAX_SIDE / max(h, w)
    return cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr


def photo_bits(gray):
    """(8, 32) uint8: 256-bit average hash of a 16x16 thumbnail for each of the 8 flips/rotations."""
    g = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    variants = []
    for k in range(4):
        r = np.rot90(g, k)
        for v in (r, r[:, ::-1]):
            variants.append(np.packbits(v > v.mean()))
    return np.stack(variants)


def photo_key(gray):
    """Exact flip/rotation/greyscale-invariant key (min over the 8 variants)."""
    return min(v.tobytes() for v in photo_bits(gray))


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def hamming_min(bits_a, bits_b):
    """Smallest bit distance between image A's canonical hash and any flip/rotation of image B."""
    return int(_POPCOUNT[bits_a[0] ^ bits_b].sum(axis=1).min())


def hash_groups(bits, max_bits):
    """Union-find over images whose hashes differ by <= max_bits under any flip/rotation.
    Catches copies that were also brightened, noised or greyscaled before saving."""
    n = len(bits)
    uf = UnionFind(n)
    canon = bits[:, 0, :]                                   # (n, 32)
    for i in range(n - 1):
        d = _POPCOUNT[canon[i + 1:, None, :] ^ bits[i][None, :, :]].sum(axis=2).min(axis=1)
        for j in np.nonzero(d <= max_bits)[0]:
            uf.union(i, i + 1 + j)
    return [uf.find(i) for i in range(n)]


def scene_vector(gray):
    """Unit-norm 32x32 thumbnail; cosine similarity >= SCENE_SIM means the same scene / adjacent frame."""
    v = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def scene_groups(vectors, bits=None, camera=None, block=2048):
    """Union-find over pairs whose thumbnails are near-identical (blocked matrix product),
    plus pairs within GROUP_BITS hash bits when hashes are given. Chaining is intended:
    it only enlarges groups, and a group never spans two splits."""
    n = len(vectors)
    uf = UnionFind(n)
    if bits is not None:
        for i, g in enumerate(hash_groups(bits, GROUP_BITS)):
            uf.union(i, g)
    for s in range(0, n, block):
        sim = vectors[s:s + block] @ vectors.T
        for i, j in zip(*np.nonzero(sim >= SCENE_SIM)):
            if s + i < j:
                uf.union(s + i, j)
    # looser threshold among images of the same fixed-camera source (camera: list of (source, sim) or None)
    if camera is not None:
        for src in {c for c in camera if c}:
            idx = np.array([i for i, c in enumerate(camera) if c == src])
            sim = vectors[idx] @ vectors[idx].T
            for a, b in zip(*np.nonzero(sim >= src[1])):
                if a < b:
                    uf.union(int(idx[a]), int(idx[b]))
    return [uf.find(i) for i in range(n)]


def parse_labels(text, names):
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
        if len(vals) > 4:  # polygon -> bbox
            xs, ys = vals[0::2], vals[1::2]
            vals = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
        rows.append((cls, *vals))
    return rows


def clean_box(x, y, w, h):
    x0, y0 = max(0.0, x - w / 2), max(0.0, y - h / 2)
    x1, y1 = min(1.0, x + w / 2), min(1.0, y + h / 2)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0 or w * h < MIN_BOX_AREA:
        return None
    return ((x0 + x1) / 2, (y0 + y1) / 2, w, h)


def scan_source(tag, cfg, zpath, report):
    """Returns candidate records (cached next to the zip, keyed on zip size/mtime and class config)."""
    st = Path(zpath).stat()
    key = (st.st_size, st.st_mtime, MIN_BOX_AREA, MAX_SIDE, "hash-after-resize",
           repr({k: cfg.get(k) for k in ("names", "pos", "conflict", "negative", "negative_folders")}))
    cache = Path(zpath).with_suffix(".scan.pkl")
    if cache.exists():
        cached = pickle.loads(cache.read_bytes())
        if cached["key"] == key:
            report["sources"][tag] = cached["stats"]
            print(f"  {tag}: {cached['stats']} (cached scan)")
            for r in cached["records"]:
                r["zip"] = str(Path(zpath).resolve())  # the cache may have been written from another folder
            return cached["records"]
    records, stats = _scan(tag, cfg, zpath)
    cache.write_bytes(pickle.dumps({"key": key, "stats": stats, "records": records}))
    report["sources"][tag] = stats
    print(f"  {tag}: {stats}")
    return records


def _scan(tag, cfg, zpath):
    z = zipfile.ZipFile(zpath)
    members = set(z.namelist())
    stats = Counter()
    records = []
    for img in sorted(m for m in members if m.lower().endswith(IMG_EXT)):
        if cfg.get("negative_folders") is not None:
            if img.split("/")[-2] not in cfg["negative_folders"]:
                stats["skipped (folder not used)"] += 1
                continue
            record = _record(tag, zpath, z, img, "neg", [], stats)
            if record:
                records.append(record)
            continue
        stem = img.rsplit(".", 1)[0]
        cands = [stem.replace("/images/", "/labels/") + ".txt", stem.replace("images/", "label/") + ".txt",
                 stem.replace("images/", "labels/") + ".txt", stem + ".txt"]
        lbl = next((c for c in cands if c in members), None)
        if lbl is None:
            stats["no label file"] += 1
            continue
        rows = parse_labels(z.read(lbl).decode("utf-8", "replace"), cfg["names"])
        classes = {r[0] for r in rows}
        if classes & cfg["pos"] and not classes & cfg["conflict"]:
            boxes = [b for b in (clean_box(*r[1:5]) for r in rows if r[0] in cfg["pos"]) if b]
            if not boxes:
                stats["positive but all boxes invalid"] += 1
                continue
            kind = "pos"
        elif classes and classes <= cfg["negative"]:
            boxes, kind = [], "neg"
        else:
            stats["skipped (lesser damage / unusable)"] += 1
            continue
        record = _record(tag, zpath, z, img, kind, boxes, stats)
        if record:
            records.append(record)
    return records, dict(stats)


def _record(tag, zpath, z, img, kind, boxes, stats):
    bgr = cv2.imdecode(np.frombuffer(z.read(img), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or min(bgr.shape[:2]) < 64:
        stats["corrupt / too small"] += 1
        return None
    bgr = fit_max_side(bgr)  # hash the image exactly as it will be saved
    b, g, r = cv2.split(bgr.astype(np.int16))
    colour = float(np.mean(np.abs(b - g)) + np.mean(np.abs(g - r)))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    stats[kind] += 1
    return dict(tag=tag, zip=str(Path(zpath).resolve()), member=img, kind=kind, boxes=boxes,
                bits=photo_bits(gray), vec=scene_vector(gray), colour=colour)


def final_copy_sweep(out, counts):
    """Re-hash the saved JPEGs and drop any image within COPY_BITS of one already kept.
    JPEG re-encoding can nudge a borderline pair under the threshold; this makes the guarantee exact."""
    files = sorted((out / "images").glob("*/*.jpg"))
    bits = [photo_bits(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)) for f in files]
    kept, removed = [], 0
    for i, f in enumerate(files):
        if any(hamming_min(bits[i], bits[k]) <= COPY_BITS for k in kept):
            split, kind = f.parent.name, ("neg" if "_neg_" in f.name else "pos")
            lbl = out / "labels" / split / (f.stem + ".txt")
            counts[(split, "boxes")] -= len([l for l in lbl.read_text().splitlines() if l.strip()])
            counts[(split, kind)] -= 1
            f.unlink(); lbl.unlink()
            removed += 1
        else:
            kept.append(i)
    print(f"  post-save copy sweep removed {removed} images")
    return removed


def build(out, raw, seed=0):
    rng = random.Random(seed)
    raw.mkdir(parents=True, exist_ok=True)
    report = {"sources": {}, "dedupe": {}, "final": {}}

    records = []
    for tag, cfg in SOURCES.items():
        zpath = download(cfg["kaggle"], raw / (cfg["kaggle"].replace("/", "_") + ".zip"))
        records += scan_source(tag, cfg, zpath, report)

    print(f"De-duplicating {len(records)} candidates ...")
    # 1. exact copies (incl. flips / rotations / greyscale): keep the most colourful one
    by_key = defaultdict(list)
    for i, r in enumerate(records):
        by_key[min(v.tobytes() for v in r["bits"])].append(i)
    keep, conflicts = [], 0
    for idxs in by_key.values():
        if len({records[i]["kind"] for i in idxs}) > 1:  # same photo labelled accident and no-accident
            conflicts += 1
            continue
        keep.append(max(idxs, key=lambda i: records[i]["colour"]))
    records = [records[i] for i in keep]
    exact_unique = len(records)

    # 2. leakage groups: near copies and same-scene frames, chained; a group never spans two splits
    camera = [(r["tag"], SOURCES[r["tag"]]["camera_sim"]) if SOURCES[r["tag"]].get("camera_sim") else None for r in records]
    scene = scene_groups(np.stack([r["vec"] for r in records]), np.stack([r["bits"] for r in records]), camera)

    # 3. inside each group drop noisy / brightened copies of an image already kept (no chaining)
    members = defaultdict(list)
    for i, g in enumerate(scene):
        members[g].append(i)
    final, copies = [], 0
    for idxs in members.values():
        kept = []
        for i in sorted(idxs, key=lambda i: -records[i]["colour"]):
            near = [k for k in kept if hamming_min(records[i]["bits"], records[k]["bits"]) <= COPY_BITS]
            if not near:
                kept.append(i)
                continue
            copies += 1
            if any(records[k]["kind"] != records[i]["kind"] for k in near):
                conflicts += 1
                for k in near:
                    records[k]["conflict"] = True
        final += [k for k in kept if not records[k].get("conflict")]
    for i in final:
        records[i]["scene"] = scene[i]
    records = [records[i] for i in final]

    group_sizes = Counter(r["scene"] for r in records).values()
    report["dedupe"] = dict(candidates=sum(len(v) for v in by_key.values()), exact_unique=exact_unique,
                            near_copies_dropped=copies, conflicting_labels_dropped=conflicts, kept=len(records),
                            split_groups=len(group_sizes), largest_split_group=max(group_sizes))
    print(f"  {report['dedupe']}")

    # quota sampling, per source and kind
    chosen = []
    for tag, cfg in SOURCES.items():
        for kind, quota in (("pos", cfg["quota_pos"]), ("neg", cfg["quota_neg"])):
            pool = [r for r in records if r["tag"] == tag and r["kind"] == kind]
            rng.shuffle(pool)
            chosen += pool[:quota]
            report["final"][f"{tag}/{kind}"] = dict(available=len(pool), used=min(quota, len(pool)))

    # split by scene group so near-identical frames never cross splits
    scenes = defaultdict(list)
    for r in chosen:
        scenes[r["scene"]].append(r)
    groups = list(scenes.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)  # place big groups first, then fill greedily
    target = {s: SPLIT_RATIO[s] * len(chosen) for s in SPLIT_RATIO}
    filled = Counter()
    for g in groups:
        split = max(SPLIT_RATIO, key=lambda s: (target[s] - filled[s]) / target[s])
        filled[split] += len(g)
        for r in g:
            r["split"] = split

    if out.exists():
        shutil.rmtree(out)
    zips = {}
    counts = Counter()
    for i, r in enumerate(chosen):
        z = zips.setdefault(r["zip"], zipfile.ZipFile(r["zip"]))
        bgr = fit_max_side(cv2.imdecode(np.frombuffer(z.read(r["member"]), np.uint8), cv2.IMREAD_COLOR))
        name = f"{r['tag']}_{r['kind']}_{i:05d}"
        (out / "images" / r["split"]).mkdir(parents=True, exist_ok=True)
        (out / "labels" / r["split"]).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "images" / r["split"] / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "labels" / r["split"] / f"{name}.txt").write_text(
            "".join(f"0 {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n" for x, y, bw, bh in r["boxes"]))
        counts[(r["split"], r["kind"])] += 1
        counts[(r["split"], "boxes")] += len(r["boxes"])

    swept = final_copy_sweep(out, counts)
    report["dedupe"]["copies_removed_after_saving"] = swept

    cfg = {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
           "names": {0: "severe_accident"}}
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    report["splits"] = {s: dict(accident_images=counts[(s, "pos")], negative_images=counts[(s, "neg")],
                                boxes=counts[(s, "boxes")]) for s in SPLIT_RATIO}
    report["total_images"] = sum(v["accident_images"] + v["negative_images"] for v in report["splits"].values())
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2))

    print("\nSplit summary:")
    for s, v in report["splits"].items():
        print(f"  {s:5s}: {v['accident_images']:5d} accident + {v['negative_images']:4d} negative images, {v['boxes']:5d} boxes")
    print(f"  total: {report['total_images']} images  ->  {out / 'data.yaml'}")
    return out / "data.yaml"


# =====================================================================================
# 2. TRAINING-SET AUGMENTATION + EVALUATION
# =====================================================================================
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


def latency_cpu(model, imgsz):
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
    print("\n".join(lines), flush=True)


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
