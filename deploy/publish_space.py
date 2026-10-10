"""Publishes RoadNetra to a Hugging Face Docker Space (needs a PRO account for Docker Spaces).

    huggingface-cli login            # once
    python deploy/publish_space.py --space <user>/roadnetra-ai

Map settings can be set as Space variables (MAP_PROVIDER, MAP_API_KEY, ...):
real environment variables override the .env files.
"""
import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from stage_app import stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="<user>/<space-name>")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp), hf_space=True)
        print(f"Uploading {len(files)} files to spaces/{args.space}")
        api.upload_folder(folder_path=tmp, repo_id=args.space, repo_type="space", commit_message="Deploy RoadNetra AI")
    host = args.space.replace("/", "-").replace("_", "-").lower()
    print(f"Space: https://huggingface.co/spaces/{args.space}\nApp:   https://{host}.hf.space")


if __name__ == "__main__":
    main()
