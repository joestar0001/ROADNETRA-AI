"""Stages a self-contained build folder for the RoadNetra container.

    python deploy/stage_app.py <out_dir> [--hf]

Copies FINAL PROJECT/ without secrets (.env), generated incidents/evidence, uploads or
training-only files, adds the three model weights under models/ and the Dockerfile.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "FINAL PROJECT"
DEPLOY = Path(__file__).resolve().parent

SKIP_NAMES = {".env", ".vercel", ".vercelignore", "vercel.json", "__pycache__", ".gitignore", "uploads",
              "train_roadnetra.py", "yolov8s.pt", "start_roadnetra.sh"}
SKIP_SUFFIXES = {".pdf", ".ipynb", ".pyc"}
MODELS = ("pothole_best.pt", "accident_best.pt", "yolov8n.pt")


def skip(path: Path) -> bool:
    parts = path.relative_to(APP).parts
    if any(part in SKIP_NAMES for part in parts) or path.suffix in SKIP_SUFFIXES:
        return True
    if parts[0] == "data" and path.name.startswith("incidents_cache"):
        return True
    return parts[:2] == ("static", "evidence") and path.name.startswith("IR-")  # generated evidence


def stage(out: Path, hf_space=False):
    out.mkdir(parents=True, exist_ok=True)
    for src in APP.rglob("*"):
        if src.is_dir() or skip(src):
            continue
        dst = out / src.relative_to(APP)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    shutil.copy2(ROOT / "models" / "accident_best.pt", out / "models" / "accident_best.pt")
    shutil.copy2(DEPLOY / "Dockerfile", out / "Dockerfile")
    if hf_space:  # Space card with the Docker SDK settings
        shutil.copy2(DEPLOY / "huggingface" / "README.md", out / "README.md")
    for model in MODELS:
        assert (out / "models" / model).exists(), f"missing model {model}"
    return sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--hf", action="store_true", help="add the Hugging Face Space README")
    args = ap.parse_args()
    files = stage(args.out, args.hf)
    print(f"Staged {len(files)} files in {args.out}")
