"""
cameras.py — Camera profiles: save a camera under a name, then reuse the name.

The problem: typing the full URL every time
(`rtsp://admin:pass@192.168.1.10:554/stream1`) is tedious, and the password
ends up in your shell history.

The fix: save it once in `cameras.json`, then just pass `--camera gate`.

> **Important:** `cameras.json` is **not** committed to git (it is listed in
> .gitignore) because it contains CCTV passwords. The repository ships
> `cameras.example.json` as a template — copy it on a new machine and fill in
> your own values.

This module also builds URLs for you, so there is no need to remember each
brand's RTSP format:

    build_url("tplink", "192.168.1.10", user="admin", password="pass@123")
    -> "rtsp://admin:pass%40123@192.168.1.10:554/stream2"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import quote

import config

logger = logging.getLogger(__name__)

# Camera type — used for display and documentation only; it does not affect behaviour
KINDS = ("mobile", "cctv", "usb", "other")


# ---------------------------------------------------------------------------
# Brand templates — "what does this brand's RTSP URL look like"
# ---------------------------------------------------------------------------

# main = full resolution (accurate but heavy), sub = lower resolution
# (noticeably faster for detection — prefer this on CPU)
CAMERA_BRANDS: dict[str, dict] = {
    "hikvision": {
        "label": "Hikvision",
        "port": 554,
        "main": "/Streaming/Channels/101",
        "sub": "/Streaming/Channels/102",
    },
    "dahua": {
        "label": "Dahua",
        "port": 554,
        "main": "/cam/realmonitor?channel=1&subtype=0",
        "sub": "/cam/realmonitor?channel=1&subtype=1",
    },
    "cpplus": {  # CP Plus runs the same firmware as Dahua
        "label": "CP Plus",
        "port": 554,
        "main": "/cam/realmonitor?channel=1&subtype=0",
        "sub": "/cam/realmonitor?channel=1&subtype=1",
    },
    "tplink": {  # Tapo C100 / C200 / C210
        "label": "TP-Link Tapo (C100 / C200 / C210)",
        "port": 554,
        "main": "/stream1",
        "sub": "/stream2",
    },
    "reolink": {
        "label": "Reolink",
        "port": 554,
        "main": "/h264Preview_01_main",
        "sub": "/h264Preview_01_sub",
    },
    "onvif": {  # generic — covers most inexpensive IP cameras
        "label": "ONVIF (generic — most inexpensive IP cameras)",
        "port": 554,
        "main": "/onvif1",
        "sub": "/onvif2",
    },
}

# Apps that turn a phone into an IP camera. These stream over HTTP rather than
# RTSP, which is why they are kept separate.
MOBILE_APPS: dict[str, dict] = {
    "ip-webcam": {
        "label": "IP Webcam — Android (most common, free)",
        "scheme": "http",
        "port": 8080,
        "path": "/video",
    },
    "droidcam": {
        "label": "DroidCam — Android and iPhone",
        "scheme": "http",
        "port": 4747,
        "path": "/video",
    },
    "iriun": {
        "label": "Iriun Webcam — Android and iPhone",
        "scheme": "http",
        "port": 8080,
        "path": "/video",
    },
}


def describe(key: str) -> str:
    """A human-readable name for use in dropdowns."""
    spec = MOBILE_APPS.get(key) or CAMERA_BRANDS.get(key) or {}
    return spec.get("label", key)


# ---------------------------------------------------------------------------
# Building and masking URLs
# ---------------------------------------------------------------------------

def mask_credentials(source: str | int) -> str:
    """
    Hide the password before printing: rtsp://user:pass@host -> rtsp://user:***@host

    Use this everywhere a URL is written to logs or the screen.
    """
    text = str(source)
    if "@" not in text or "://" not in text:
        return text
    scheme, rest = text.split("://", 1)
    creds, host = rest.split("@", 1)
    if ":" in creds:
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return text


def build_url(
    brand: str,
    ip: str,
    user: str | None = None,
    password: str | None = None,
    port: int | None = None,
    quality: str = "sub",
) -> str:
    """
    Build a complete stream URL from a brand name.

    Args:
        brand: any key from `CAMERA_BRANDS` or `MOBILE_APPS` (e.g. "tplink", "ip-webcam")
        ip: the camera's IP address, e.g. "192.168.1.10"
        user/password: CCTV login (phone apps usually need neither)
        port: override the brand's default port
        quality: "sub" (lower resolution, faster — the default) or "main" (full)

    Special characters in the password (@ : / ?) are URL-encoded automatically,
    otherwise the URL would be malformed.
    """
    key = brand.strip().lower()

    if key in MOBILE_APPS:
        spec = MOBILE_APPS[key]
        auth = _auth_part(user, password)
        return f"{spec['scheme']}://{auth}{ip}:{port or spec['port']}{spec['path']}"

    if key not in CAMERA_BRANDS:
        known = ", ".join(sorted({*CAMERA_BRANDS, *MOBILE_APPS}))
        raise ValueError(f"Unknown brand '{brand}'. Available: {known}")

    spec = CAMERA_BRANDS[key]
    if quality not in ("main", "sub"):
        raise ValueError("quality must be either 'main' or 'sub'")

    auth = _auth_part(user, password)
    return f"rtsp://{auth}{ip}:{port or spec['port']}{spec[quality]}"


def _auth_part(user: str | None, password: str | None) -> str:
    """The "user:pass@" portion of a URL — empty when no credentials are given."""
    if not user:
        return ""
    if not password:
        return f"{quote(user, safe='')}@"
    return f"{quote(user, safe='')}:{quote(password, safe='')}@"


# ---------------------------------------------------------------------------
# Saved profiles (cameras.json)
# ---------------------------------------------------------------------------

@dataclass
class CameraProfile:
    """A saved camera — its name, its source, and a little context."""

    name: str
    url: str                 # "0" (USB webcam) / "http://..." / "rtsp://..."
    kind: str = "other"      # mobile | cctv | usb | other
    note: str = ""

    @property
    def safe_url(self) -> str:
        """The URL with the password masked, safe to print."""
        return mask_credentials(self.url)


class CameraStoreError(RuntimeError):
    """Raised when cameras.json cannot be read or written (bad JSON, permissions...)."""


def _store_path(path: Path | str | None = None) -> Path:
    return Path(path) if path else config.CAMERAS_FILE


def load_cameras(path: Path | str | None = None) -> dict[str, CameraProfile]:
    """
    Read every saved camera. A missing file yields an empty dict — that is not
    an error, it is simply the normal state on a first run.
    """
    store = _store_path(path)
    if not store.is_file():
        return {}

    try:
        raw = json.loads(store.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CameraStoreError(
            f"{store} is not valid JSON (line {exc.lineno}): {exc.msg}\n"
            "A comma or bracket was probably lost while editing. You can also "
            "delete the file and copy cameras.example.json again."
        ) from exc

    if not isinstance(raw, dict):
        raise CameraStoreError(f"{store} must contain a JSON object at the top level.")

    cameras: dict[str, CameraProfile] = {}
    for name, entry in raw.items():
        # Shorthand form {"phone": "http://..."} is accepted alongside the full object
        if isinstance(entry, str):
            entry = {"url": entry}
        if not isinstance(entry, dict) or not entry.get("url"):
            logger.warning("Skipping camera '%s' — no 'url' field.", name)
            continue
        cameras[name] = CameraProfile(
            name=name,
            url=str(entry["url"]),
            kind=str(entry.get("kind", "other")),
            note=str(entry.get("note", "")),
        )
    return cameras


def get_camera(name: str, path: Path | str | None = None) -> CameraProfile | None:
    """Look up one camera by name (case-insensitive). Returns None if not found."""
    cameras = load_cameras(path)
    if name in cameras:
        return cameras[name]
    lowered = {k.lower(): v for k, v in cameras.items()}
    return lowered.get(name.strip().lower())


def save_camera(
    name: str,
    url: str,
    kind: str = "other",
    note: str = "",
    path: Path | str | None = None,
) -> CameraProfile:
    """
    Save or update a camera and write cameras.json.
    Reusing a name replaces the existing entry.
    """
    store = _store_path(path)
    cameras = load_cameras(store)

    profile = CameraProfile(name=name, url=str(url), kind=kind, note=note)
    cameras[name] = profile

    payload = {
        cam.name: {k: v for k, v in asdict(cam).items() if k != "name" and v != ""}
        for cam in cameras.values()
    }
    try:
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise CameraStoreError(f"Could not write {store}: {exc}") from exc

    logger.info("Camera '%s' saved in %s", name, store)
    return profile


def delete_camera(name: str, path: Path | str | None = None) -> bool:
    """Remove a camera. Returns True if it existed and was deleted."""
    store = _store_path(path)
    cameras = load_cameras(store)
    if name not in cameras:
        return False

    del cameras[name]
    payload = {
        cam.name: {k: v for k, v in asdict(cam).items() if k != "name" and v != ""}
        for cam in cameras.values()
    }
    store.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return True


def format_camera_table(cameras: dict[str, CameraProfile]) -> str:
    """A clean table for `--list-cameras`, with passwords masked."""
    if not cameras:
        return (
            "No cameras saved yet.\n\n"
            "To add one:\n"
            "  python main.py --source http://192.168.1.5:8080/video --save-camera phone\n"
            "Or copy cameras.example.json to cameras.json and edit it."
        )

    width = max(len(name) for name in cameras)
    lines = [f"{len(cameras)} saved camera(s):", ""]
    for cam in cameras.values():
        line = f"  {cam.name:<{width}}  [{cam.kind}]  {cam.safe_url}"
        if cam.note:
            line += f"   — {cam.note}"
        lines.append(line)
    lines.append("")
    lines.append("Use: python main.py --camera <name>")
    return "\n".join(lines)
