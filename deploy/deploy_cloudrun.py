"""Builds and deploys RoadNetra to Google Cloud Run, then points Vercel at it.

    gcloud auth login                # once
    python deploy/deploy_cloudrun.py --project <gcp-project-id> [--region us-central1]

Cloud Build builds deploy/Dockerfile from a staged folder (no local Docker needed).
The service runs as a single instance (max-instances 1) so every portal shares one incident
store. It scales to zero when idle; the next request cold-starts (~30 s) with the sample incidents.
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from stage_app import ROOT, stage

GCLOUD = shutil.which("gcloud") or shutil.which("gcloud.cmd") or "gcloud"
APIS = ["run.googleapis.com", "cloudbuild.googleapis.com", "artifactregistry.googleapis.com"]


def gcloud(*args, capture=False):
    cmd = [GCLOUD, *args, "--quiet"]
    print("$ gcloud " + " ".join(args))
    res = subprocess.run(cmd, check=True, text=True, capture_output=capture)
    return res.stdout.strip() if capture else None


def point_vercel(url):
    """FINAL PROJECT/vercel.json proxies every path on the Vercel domain to the Cloud Run service."""
    cfg = {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "framework": None,
        "buildCommand": "",
        "installCommand": "",
        "outputDirectory": ".",
        "rewrites": [{"source": "/(.*)", "destination": f"{url}/$1"}],
    }
    path = ROOT / "FINAL PROJECT" / "vercel.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {path.relative_to(ROOT)} -> {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--service", default="roadnetra-ai")
    ap.add_argument("--no-vercel", action="store_true", help="do not rewrite FINAL PROJECT/vercel.json")
    args = ap.parse_args()

    gcloud("services", "enable", *APIS, "--project", args.project)
    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp))
        print(f"Staged {len(files)} files; building on Cloud Build…")
        gcloud("run", "deploy", args.service, "--source", tmp, "--project", args.project, "--region", args.region,
               "--allow-unauthenticated", "--memory", "2Gi", "--cpu", "2", "--concurrency", "20",
               "--min-instances", "0", "--max-instances", "1", "--timeout", "300",
               "--cpu-boost", "--set-env-vars", "TORCH_THREADS=2")
    url = gcloud("run", "services", "describe", args.service, "--project", args.project, "--region", args.region,
                 "--format", "value(status.url)", capture=True)
    print(f"Cloud Run: {url}")
    if not args.no_vercel:
        point_vercel(url)


if __name__ == "__main__":
    main()
