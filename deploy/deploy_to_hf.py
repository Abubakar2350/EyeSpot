#!/usr/bin/env python3
"""
deploy_to_hf.py — Publish EyeSpot to a Hugging Face Space.

Two deployment modes:

  --mode bundled          Upload the application files to the Space itself.
                          Simple, but a public Space would then show the source
                          in its Files tab, so pair this with --private.

  --mode private-source   Upload only a Dockerfile. The application is cloned at
                          build time from the private GitHub repository using a
                          Space secret. This is what allows a PUBLIC demo whose
                          source stays private — visitors get the running app,
                          not the code.

Usage (from the project root, with the venv active):
    python deploy/deploy_to_hf.py --mode bundled --private
    python deploy/deploy_to_hf.py --mode private-source --ssh-key ~/.ssh/eyespot_deploy_key

Hugging Face dropped its native Streamlit SDK, so either way the Space is a
Docker Space serving Streamlit on port 7860.

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


def stage_private_source(target: Path) -> None:
    """
    Stage a Space that holds nothing but a Dockerfile and its card.

    The application never reaches the Space repository, so a public Space
    exposes no source code.
    """
    shutil.copy2(DEPLOY / "README_hf.md", target / "README.md")
    shutil.copy2(DEPLOY / "Dockerfile.private-source", target / "Dockerfile")


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
    parser.add_argument(
        "--mode", choices=["bundled", "private-source"], default="bundled",
        help="bundled = upload the app; private-source = upload only a Dockerfile that clones it",
    )
    parser.add_argument(
        "--ssh-key", default=None, metavar="PATH",
        help=(
            "Path to a GitHub deploy key (read-only) for the private repo; its contents "
            "are stored as the Space secret SSH_DEPLOY_KEY (private-source mode)"
        ),
    )
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

    if args.mode == "private-source" and not args.ssh_key:
        print(
            "private-source mode needs --ssh-key: the Space clones the code at build\n"
            "time, and without the deploy key that clone fails."
        )
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        if args.mode == "private-source":
            stage_private_source(staging)
        else:
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

        if args.ssh_key:
            print("Storing the SSH_DEPLOY_KEY Space secret...")
            api.add_space_secret(
                repo_id=space_id,
                key="SSH_DEPLOY_KEY",
                value=Path(args.ssh_key).read_text(),
                description="Read-only GitHub deploy key used to clone the private source at build time",
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
