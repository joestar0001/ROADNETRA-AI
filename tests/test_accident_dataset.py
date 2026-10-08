"""Integrity tests for the Model B dataset and its builder.

    pytest tests/test_accident_dataset.py -v
Dataset checks are skipped if data/accident_dataset has not been built (override with ACCIDENT_DATA=<dir>).
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import build_accident_dataset as bad  # noqa: E402

DATA = Path(os.environ.get("ACCIDENT_DATA", ROOT / "data" / "accident_dataset"))
SPLITS = ("train", "val", "test")
needs_data = pytest.mark.skipif(not (DATA / "data.yaml").exists(), reason="dataset not built")


# ---------- builder unit tests ----------

def test_polygon_label_becomes_bbox():
    rows = bad.parse_labels("1 0.2 0.2 0.4 0.2 0.4 0.4 0.2 0.4", {})
    assert rows == [(1, pytest.approx(0.3), pytest.approx(0.3), pytest.approx(0.2), pytest.approx(0.2))]


def test_clean_box_clips_and_drops_slivers():
    x, y, w, h = bad.clean_box(0.95, 0.5, 0.2, 0.2)          # sticks out on the right
    assert x + w / 2 == pytest.approx(1.0) and w == pytest.approx(0.15)
    assert bad.clean_box(0.5, 0.5, 0.01, 0.01) is None       # 0.01 % of frame
    assert bad.clean_box(0.5, 0.5, 0.0, 0.3) is None


def test_photo_key_matches_flips_rotations_and_greyscale():
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8), (15, 15), 0)
    gray = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    key = bad.photo_key(gray(img))
    for variant in (img[:, ::-1], img[::-1], np.rot90(img), np.rot90(img, 2)):
        assert bad.photo_key(gray(np.ascontiguousarray(variant))) == key
    other = cv2.GaussianBlur(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8), (15, 15), 0)
    assert bad.photo_key(gray(other)) != key


def test_scene_groups_join_near_frames_only():
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur(rng.integers(0, 255, (100, 100), dtype=np.uint8), (9, 9), 0)
    near = np.clip(base.astype(int) + rng.integers(-3, 4, base.shape), 0, 255).astype(np.uint8)
    far = cv2.GaussianBlur(rng.integers(0, 255, (100, 100), dtype=np.uint8), (9, 9), 0)
    g = bad.scene_groups(np.stack([bad.scene_vector(x) for x in (base, near, far)]))
    assert g[0] == g[1] and g[0] != g[2]


# ---------- built dataset checks ----------

def _items():
    out = []
    for s in SPLITS:
        for img in sorted((DATA / "images" / s).glob("*.jpg")):
            out.append((s, img, DATA / "labels" / s / (img.stem + ".txt")))
    return out


@needs_data
def test_data_yaml_contract():
    cfg = yaml.safe_load((DATA / "data.yaml").read_text())
    assert cfg["names"] == {0: "severe_accident"}
    for s in SPLITS:
        assert (DATA / cfg[s if s != "val" else "val"]).is_dir()


@needs_data
def test_every_image_has_label_and_vice_versa():
    for s in SPLITS:
        imgs = {p.stem for p in (DATA / "images" / s).glob("*.jpg")}
        lbls = {p.stem for p in (DATA / "labels" / s).glob("*.txt")}
        assert imgs == lbls, f"{s}: {len(imgs ^ lbls)} orphan files"


@needs_data
def test_labels_are_valid_yolo_class0():
    for s, img, lbl in _items():
        for line in lbl.read_text().splitlines():
            parts = line.split()
            assert len(parts) == 5, f"{lbl}: {line}"
            c, x, y, w, h = int(parts[0]), *map(float, parts[1:])
            assert c == 0, f"{lbl}: class {c}"
            assert 0 < w <= 1 and 0 < h <= 1, f"{lbl}: {line}"
            assert x - w / 2 >= -1e-5 and x + w / 2 <= 1 + 1e-5 and y - h / 2 >= -1e-5 and y + h / 2 <= 1 + 1e-5, f"{lbl}: {line}"


@needs_data
def test_positives_have_boxes_and_negatives_are_empty():
    for s, img, lbl in _items():
        n = len([l for l in lbl.read_text().splitlines() if l.strip()])
        if "_neg_" in img.name:
            assert n == 0, img
        elif "_pos_" in img.name:
            assert n >= 1, img


@needs_data
def test_images_decode_and_are_at_most_640():
    for s, img, _ in _items():
        if img.name.startswith("wx_"):
            continue
        a = cv2.imread(str(img))
        assert a is not None, img
        assert max(a.shape[:2]) <= 640, (img, a.shape)


@needs_data
def test_no_photo_or_scene_leaks_across_splits():
    """Copies of one photo (even noised / greyscaled / flipped) and same-camera shots stay in one split."""
    items = [(s, img) for s, img, _ in _items() if not img.name.startswith("wx_")]
    bits = np.stack([bad.photo_bits(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for _, p in items])
    groups = defaultdict(set)
    for (s, _), g in zip(items, bad.hash_groups(bits, bad.GROUP_BITS)):
        groups[g].add(s)
    crossing = [g for g, splits in groups.items() if len(splits) > 1]
    assert not crossing, f"{len(crossing)} photo/scene groups span more than one split"


@needs_data
def test_no_duplicate_photos_kept():
    items = [img for _, img, _ in _items() if not img.name.startswith("wx_")]
    bits = np.stack([bad.photo_bits(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in items])
    g = bad.hash_groups(bits, bad.COPY_BITS)
    assert len(set(g)) == len(items), f"{len(items) - len(set(g))} duplicate photos remain"


@needs_data
def test_no_near_identical_frames_across_splits():
    items = [(s, img) for s, img, _ in _items() if not img.name.startswith("wx_")]
    vecs = np.stack([bad.scene_vector(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for _, p in items])
    split = np.array([s for s, _ in items])
    leaks = []
    for i in range(0, len(items), 1024):
        sim = vecs[i:i + 1024] @ vecs.T
        for a, b in zip(*np.nonzero(sim >= bad.SCENE_SIM)):
            if split[i + a] != split[b]:
                leaks.append((items[i + a][1].name, items[b][1].name))
    assert not leaks, f"{len(leaks)} near-duplicate pairs cross splits, e.g. {leaks[:3]}"


@needs_data
def test_counts_match_report_and_target_size():
    rep = json.loads((DATA / "dataset_report.json").read_text())
    for s in SPLITS:
        pos = len([p for p in (DATA / "images" / s).glob("*_pos_*.jpg") if not p.name.startswith("wx_")])
        neg = len(list((DATA / "images" / s).glob("*_neg_*.jpg")))
        assert pos == rep["splits"][s]["accident_images"] and neg == rep["splits"][s]["negative_images"]
    assert 1500 <= rep["total_images"] <= 2000
    assert sum(rep["splits"][s]["negative_images"] for s in SPLITS) >= 200   # spec: 200+ negatives
    total = rep["total_images"]
    for s, share in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        n = rep["splits"][s]["accident_images"] + rep["splits"][s]["negative_images"]
        assert abs(n / total - share) < 0.03, (s, n / total)


@needs_data
def test_weather_copies_only_in_train():
    for s in ("val", "test"):
        assert not list((DATA / "images" / s).glob("wx_*")), f"augmented copies leaked into {s}"
