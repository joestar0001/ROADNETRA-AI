"""Builds the Model B (severe_accident) dataset from public Kaggle sources.

Pipeline:
  1. Download source zips from Kaggle (public, no login needed).
  2. Keep images whose boxes are severe accidents (rollover / crushed / fire / pile-up);
     images with only intact vehicles become negatives with empty labels.
  3. Remove corrupt images, clip boxes to the frame, drop slivers.
  4. Collapse flipped / rotated / greyscale copies of the same photo to one image.
  5. Group near-identical frames (video sequences) so a scene never spans two splits.
  6. Sample to the target size, split 70/15/15 by group, resize to <=640 px, write YOLO layout.

Usage:
    python training/build_accident_dataset.py --out data/accident_dataset
"""
import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

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
}

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}
MAX_SIDE = 640
MIN_BOX_AREA = 0.0004      # boxes under 0.04 % of the frame are annotation noise
SCENE_SIM = 0.95           # thumbnail cosine similarity at/above which two images are the same scene


def download(slug, dst):
    if dst.exists() and zipfile.is_zipfile(dst):
        return dst
    print(f"Downloading {slug} ...")
    req = urllib.request.Request(KAGGLE_URL.format(slug), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    return dst


def photo_key(gray):
    """Flip/rotation/greyscale-invariant key: min over the 8 dihedral variants of a 16x16 average hash.
    Augmented copies of one photo share a key; distinct photos almost never do."""
    g = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    variants = []
    for k in range(4):
        r = np.rot90(g, k)
        for v in (r, r[:, ::-1]):
            variants.append(np.packbits(v > v.mean()).tobytes())
    return min(variants)


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


def scene_groups(vectors, block=2048):
    """Union-find over pairs whose thumbnails are near-identical (blocked matrix product)."""
    n = len(vectors)
    uf = UnionFind(n)
    for s in range(0, n, block):
        sim = vectors[s:s + block] @ vectors.T
        for i, j in zip(*np.nonzero(sim >= SCENE_SIM)):
            if s + i < j:
                uf.union(s + i, j)
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
    """Returns candidate records: dict(tag, member, kind, boxes, gray, colour)."""
    z = zipfile.ZipFile(zpath)
    members = set(z.namelist())
    stats = Counter()
    records = []
    for img in sorted(m for m in members if m.lower().endswith(IMG_EXT)):
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
        data = np.frombuffer(z.read(img), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None or min(bgr.shape[:2]) < 64:
            stats["corrupt / too small"] += 1
            continue
        b, g, r = cv2.split(bgr.astype(np.int16))
        colour = float(np.mean(np.abs(b - g)) + np.mean(np.abs(g - r)))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        records.append(dict(tag=tag, zip=str(zpath), member=img, kind=kind, boxes=boxes,
                            key=photo_key(gray), vec=scene_vector(gray), colour=colour))
        stats[kind] += 1
    report["sources"][tag] = dict(stats)
    print(f"  {tag}: {dict(stats)}")
    return records


def build(out, raw, seed=0):
    rng = random.Random(seed)
    raw.mkdir(parents=True, exist_ok=True)
    report = {"sources": {}, "dedupe": {}, "final": {}}

    records = []
    for tag, cfg in SOURCES.items():
        zpath = download(cfg["kaggle"], raw / (cfg["kaggle"].replace("/", "_") + ".zip"))
        records += scan_source(tag, cfg, zpath, report)

    print(f"De-duplicating {len(records)} candidates ...")
    by_photo = defaultdict(list)
    for i, r in enumerate(records):
        by_photo[r["key"]].append(i)
    keep = []
    conflicts = 0
    for idxs in by_photo.values():
        kinds = {records[i]["kind"] for i in idxs}
        if len(kinds) > 1:  # same photo labelled accident in one place, no-accident in another
            conflicts += 1
            continue
        # prefer the most colourful copy (drops greyscale augmentations)
        keep.append(max(idxs, key=lambda i: records[i]["colour"]))
    report["dedupe"] = dict(candidates=len(records), unique_photos=len(by_photo),
                            conflicting_labels_dropped=conflicts, kept=len(keep))
    print(f"  {report['dedupe']}")

    records = [records[i] for i in keep]
    scene = scene_groups(np.stack([r["vec"] for r in records]))
    for r, s in zip(records, scene):
        r["scene"] = s
    sizes = Counter(Counter(scene).values())
    report["dedupe"]["scene_groups"] = len(set(scene))
    report["dedupe"]["largest_scene_group"] = max(sizes)
    print(f"  scene groups: {len(set(scene))}, largest {max(sizes)} images")

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
        bgr = cv2.imdecode(np.frombuffer(z.read(r["member"]), np.uint8), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        s = MAX_SIDE / max(h, w)
        if s < 1:
            bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        name = f"{r['tag']}_{r['kind']}_{i:05d}"
        (out / "images" / r["split"]).mkdir(parents=True, exist_ok=True)
        (out / "labels" / r["split"]).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "images" / r["split"] / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "labels" / r["split"] / f"{name}.txt").write_text(
            "".join(f"0 {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n" for x, y, bw, bh in r["boxes"]))
        counts[(r["split"], r["kind"])] += 1
        counts[(r["split"], "boxes")] += len(r["boxes"])

    cfg = {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
           "names": {0: "severe_accident"}}
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    report["splits"] = {s: dict(accident_images=counts[(s, "pos")], negative_images=counts[(s, "neg")],
                                boxes=counts[(s, "boxes")]) for s in SPLIT_RATIO}
    report["total_images"] = len(chosen)
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2))

    print("\nSplit summary:")
    for s, v in report["splits"].items():
        print(f"  {s:5s}: {v['accident_images']:5d} accident + {v['negative_images']:4d} negative images, {v['boxes']:5d} boxes")
    print(f"  total: {len(chosen)} images  ->  {out / 'data.yaml'}")
    return out / "data.yaml"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/accident_dataset", type=Path)
    ap.add_argument("--raw", default="data/raw_kaggle", type=Path)
    ap.add_argument("--seed", default=0, type=int)
    a = ap.parse_args()
    build(a.out, a.raw, a.seed)
