#!/usr/bin/env python3
"""
deploy_to_hf.py — Publish EyeSpot to a Hugging Face Space.

Usage (from the project root, with the venv active):
    python deploy/deploy_to_hf.py
    python deploy/deploy_to_hf.py --space myuser/eyespot --private

The Space is created if it does not exist, then the app is uploaded. Files are
staged in a temporary directory first so the Space gets exactly what it needs
and nothing else — no .venv, no outputs, no cameras.json.

Two files are swapped in for the hosted environment:
  - deploy/requirements-hf.txt -> requirements.txt   (headless OpenCV, CPU torch)
  - deploy/README_hf.md        -> README.md          (the Space card + its config)
  - deploy/Dockerfile          -> Dockerfile         (Hugging Face dropped the
    native Streamlit SDK, so Streamlit apps are deployed as Docker Spaces)

Authentication uses your stored Hugging Face token (`huggingface-cli login`).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"

# Everything the hosted app needs. Anything not listed here is not uploaded.
FILES = [
    "main.py",
    "config.py",
    "test_smoke.py",
    "cameras.example.json",
    "sample.jpg",
    "REPORT.md",
]
DIRS = ["detector", "utils", "ui"]


def stage(target: Path) -> None:
    """Copy the app into `target`, swapping in the hosted-environment files."""
    for name in FILES:
        source = ROOT / name
        if source.exists():
            shutil.copy2(source, target / name)
        else:
            print(f"  ! skipping missing file: {name}")

    for name in DIRS:
        shutil.copytree(
            ROOT / name,
            target / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    # Hosted variants replace their local counterparts
    shutil.copy2(DEPLOY / "README_hf.md", target / "README.md")
    shutil.copy2(DEPLOY / "requirements-hf.txt", target / "requirements.txt")
    shutil.copy2(DEPLOY / "Dockerfile", target / "Dockerfile")


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy EyeSpot to a Hugging Face Space.")
    parser.add_argument("--space", default=None, help="Space id, e.g. myuser/eyespot (default: <you>/eyespot)")
    parser.add_argument("--private", action="store_true", help="Create the Space as private")
    parser.add_argument("--dry-run", action="store_true", help="Stage the files and list them, but do not upload")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, whoami
    except ImportError:
        print("huggingface_hub is not installed. Run:  pip install huggingface_hub")
        return 1

    api = HfApi()
    try:
        user = whoami()["name"]
    except Exception as exc:
        print(f"Not logged in to Hugging Face ({exc}).\nRun:  huggingface-cli login")
        return 1

    space_id = args.space or f"{user}/eyespot"
    print(f"User  : {user}")
    print(f"Space : {space_id}")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        stage(staging)

        staged = sorted(p.relative_to(staging) for p in staging.rglob("*") if p.is_file())
        total_mb = sum((staging / p).stat().st_size for p in staged) / 1e6
        print(f"\nStaged {len(staged)} files ({total_mb:.1f} MB):")
        for p in staged:
            print(f"  {p}")

        if args.dry_run:
            print("\nDry run — nothing uploaded.")
            return 0

        print("\nCreating the Space if it does not already exist...")
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
        )

        print("Uploading...")
        api.upload_folder(
            repo_id=space_id,
            repo_type="space",
            folder_path=str(staging),
            commit_message="Deploy EyeSpot",
        )

    print(f"\nDone: https://huggingface.co/spaces/{space_id}")
    print("The first build takes a few minutes while the dependencies install.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
