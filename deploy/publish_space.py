"""Publishes the RoadNetra app to a Hugging Face Docker Space.

    huggingface-cli login            # once
    python deploy/publish_space.py --space <user>/roadnetra-ai

Stages FINAL PROJECT/ (without secrets, generated incidents or training-only files) plus the
three model weights, then uploads it. Map settings can be set as Space variables
(MAP_PROVIDER, MAP_API_KEY, ...): real environment variables override the .env files.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "FINAL PROJECT"
HERE = Path(__file__).resolve().parent / "huggingface"

SKIP_NAMES = {".env", ".vercel", ".vercelignore", "vercel.json", "__pycache__", ".gitignore", "uploads",
              "train_roadnetra.py", "yolov8s.pt", "start_roadnetra.sh"}
SKIP_SUFFIXES = {".pdf", ".ipynb", ".pyc"}


def skip(path: Path) -> bool:
    parts = path.relative_to(APP).parts
    if any(part in SKIP_NAMES for part in parts) or path.suffix in SKIP_SUFFIXES:
        return True
    if parts[0] == "data" and path.name.startswith("incidents_cache"):
        return True
    return parts[:2] == ("static", "evidence") and path.name.startswith("IR-")  # generated evidence


def stage(out: Path):
    for src in APP.rglob("*"):
        if src.is_dir() or skip(src):
            continue
        dst = out / src.relative_to(APP)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    shutil.copy2(ROOT / "models" / "accident_best.pt", out / "models" / "accident_best.pt")
    shutil.copy2(HERE / "Dockerfile", out / "Dockerfile")
    shutil.copy2(HERE / "README.md", out / "README.md")
    for model in ("pothole_best.pt", "accident_best.pt", "yolov8n.pt"):
        assert (out / "models" / model).exists(), f"missing model {model}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="<user>/<space-name>")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        stage(out)
        files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
        print(f"Uploading {len(files)} files to spaces/{args.space}")
        api.upload_folder(folder_path=str(out), repo_id=args.space, repo_type="space",
                          commit_message="Deploy RoadNetra AI")
    host = args.space.replace("/", "-").replace("_", "-").lower()
    print(f"Space: https://huggingface.co/spaces/{args.space}\nApp:   https://{host}.hf.space")


if __name__ == "__main__":
    main()
