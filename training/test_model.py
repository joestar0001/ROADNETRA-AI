"""Try the accident model on images, a folder, a video or a webcam and see what it detects.

Run from the ROADNETRA-AI folder:
    .venv\\Scripts\\python training\\test_model.py                          # 5 test crashes + 5 normal-traffic images
    .venv\\Scripts\\python training\\test_model.py --n 20                   # more test images
    .venv\\Scripts\\python training\\test_model.py --source my_photo.jpg    # one image
    .venv\\Scripts\\python training\\test_model.py --source my_folder       # every image in a folder
    .venv\\Scripts\\python training\\test_model.py --source clip.mp4 --show # a video, with a live window (q to quit)
    .venv\\Scripts\\python training\\test_model.py --source 0 --show       # webcam
Annotated results are saved under runs/test_model/.
"""
import argparse
import glob
import random
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def test_dataset_images(model, n, conf):
    test_dir = ROOT / "data" / "accident_dataset" / "images" / "test"
    imgs = sorted(glob.glob(str(test_dir / "*.jpg")))
    if not imgs:
        raise SystemExit(f"No test images in {test_dir}; pass --source <image/folder/video> instead")
    rng = random.Random(0)
    crash = rng.sample([p for p in imgs if "_pos_" in Path(p).name], n)
    normal = rng.sample([p for p in imgs if "_neg_" in Path(p).name], n)

    results = model.predict(crash + normal, conf=conf, save=True, project=str(ROOT / "runs" / "test_model"),
                            name="dataset_sample", exist_ok=True, verbose=False)
    caught = false_alarms = 0
    print(f"\n{'image':32s} {'truth':10s} {'model says':22s} result")
    for path, r in zip(crash + normal, results):
        is_crash = "_pos_" in Path(path).name
        found = len(r.boxes) > 0
        top = f" (conf {float(r.boxes.conf.max()):.2f})" if found else ""
        ok = found == is_crash
        caught += is_crash and found
        false_alarms += (not is_crash) and found
        print(f"{Path(path).name:32s} {'CRASH' if is_crash else 'normal':10s} "
              f"{('ACCIDENT' + top) if found else 'nothing':22s} {'OK' if ok else 'WRONG'}")
    print(f"\nCrashes detected: {caught}/{n}   False alarms on normal traffic: {false_alarms}/{n}")
    print(f"Annotated images: {results[0].save_dir}")


def test_source(model, source, conf, show):
    is_video = source.isdigit() or source.lower().endswith(VIDEO_EXT)
    frames = with_accident = 0
    for r in model.predict(source if not source.isdigit() else int(source), conf=conf, save=True, show=show,
                           stream=True, project=str(ROOT / "runs" / "test_model"), name="source",
                           exist_ok=True, verbose=False):
        frames += 1
        if len(r.boxes):
            with_accident += 1
            if not is_video:
                print(f"{Path(r.path).name}: ACCIDENT (conf {float(r.boxes.conf.max()):.2f})")
        elif not is_video:
            print(f"{Path(r.path).name}: nothing")
    if is_video:
        print(f"Frames with an accident box: {with_accident}/{frames}")
    print(f"Annotated output: {ROOT / 'runs' / 'test_model' / 'source'}")


def main():
    ap = argparse.ArgumentParser(description="Test the RoadNetra severe-accident model")
    ap.add_argument("--weights", default=str(ROOT / "models" / "accident_best.pt"))
    ap.add_argument("--source", help="image, folder, video file, or webcam index (e.g. 0)")
    ap.add_argument("--conf", type=float, default=0.4, help="confidence threshold (use BEST CONFIDENCE from the report)")
    ap.add_argument("--n", type=int, default=5, help="test images per class when no --source is given")
    ap.add_argument("--show", action="store_true", help="open a live window (videos / webcam)")
    a = ap.parse_args()

    if not Path(a.weights).exists():
        raise SystemExit(f"Model not found: {a.weights}\nDownload accident_best.pt from Colab into the models folder.")
    model = YOLO(a.weights)
    print(f"Loaded {a.weights}  classes={model.names}  conf={a.conf}")
    if model.names != {0: "severe_accident"}:
        raise SystemExit("This is not the RoadNetra accident model (wrong classes)")

    if a.source:
        test_source(model, a.source, a.conf, a.show)
    else:
        test_dataset_images(model, a.n, a.conf)


if __name__ == "__main__":
    main()
