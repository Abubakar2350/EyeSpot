"""
cameras.py — Camera profiles: camera ko ek naam se save karo, phir wohi naam use karo.

Masla: har baar poora URL type karna (`rtsp://admin:pass@192.168.1.10:554/stream1`)
takleef deh hai, aur password shell history mein chala jaata hai.

Hal: ek baar `cameras.json` mein save kar do, phir sirf `--camera gate` likhein.

> **Ahem:** `cameras.json` git mein **nahi** jaati (.gitignore mein hai) kyunke
> us mein CCTV ke passwords hote hain. Repo mein `cameras.example.json` template
> hai — doosre PC par usay copy kar ke apni values daal lein.

Yahan URL banane ke helpers bhi hain taake brand ka RTSP format yaad rakhne ki
zaroorat na pade:

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

# Camera ki qism — sirf display/documentation ke liye, behaviour par asar nahi
KINDS = ("mobile", "cctv", "usb", "other")


# ---------------------------------------------------------------------------
# Brand templates — "is brand ka RTSP URL kaisa hota hai"
# ---------------------------------------------------------------------------

# main = full resolution (accurate lekin bhaari), sub = chhoti resolution
# (detection ke liye kaafi tez — CPU par isi ko tarjeeh dein)
CAMERA_BRANDS: dict[str, dict] = {
    "hikvision": {
        "port": 554,
        "main": "/Streaming/Channels/101",
        "sub": "/Streaming/Channels/102",
    },
    "dahua": {
        "port": 554,
        "main": "/cam/realmonitor?channel=1&subtype=0",
        "sub": "/cam/realmonitor?channel=1&subtype=1",
    },
    "cpplus": {  # CP Plus Dahua ka hi firmware use karta hai
        "port": 554,
        "main": "/cam/realmonitor?channel=1&subtype=0",
        "sub": "/cam/realmonitor?channel=1&subtype=1",
    },
    "tplink": {  # Tapo C100 / C200 / C210
        "port": 554,
        "main": "/stream1",
        "sub": "/stream2",
    },
    "reolink": {
        "port": 554,
        "main": "/h264Preview_01_main",
        "sub": "/h264Preview_01_sub",
    },
    "onvif": {  # generic — zyada tar sasti IP cameras
        "port": 554,
        "main": "/onvif1",
        "sub": "/onvif2",
    },
}

# Mobile phone ko IP camera banane wali apps — ye HTTP par stream deti hain
# (RTSP nahi), isliye inhe alag rakha hai.
MOBILE_APPS: dict[str, dict] = {
    "ip-webcam": {  # Android — "IP Webcam" (sabse aam, free)
        "scheme": "http",
        "port": 8080,
        "path": "/video",
    },
    "droidcam": {  # Android + iPhone — "DroidCam"
        "scheme": "http",
        "port": 4747,
        "path": "/video",
    },
    "iriun": {  # Android + iPhone — "Iriun Webcam"
        "scheme": "http",
        "port": 8080,
        "path": "/video",
    },
}


# ---------------------------------------------------------------------------
# URL banana aur chhupana
# ---------------------------------------------------------------------------

def mask_credentials(source: str | int) -> str:
    """
    Logs/screen par password na dikhe: rtsp://user:pass@host -> rtsp://user:***@host

    Har us jagah use karein jahan URL print ho raha ho.
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
    Brand ke naam se poora stream URL bana do.

    Args:
        brand: `CAMERA_BRANDS` ya `MOBILE_APPS` ka koi naam (e.g. "tplink", "ip-webcam")
        ip: camera ka IP, e.g. "192.168.1.10"
        user/password: CCTV ka login (mobile apps par aam tor par zaroorat nahi)
        port: default port override karna ho to
        quality: "sub" (chhoti resolution, tez — default) ya "main" (full)

    Password mein special characters (@ : / ?) khud URL-encode ho jaate hain,
    warna URL toot jaata hai.
    """
    key = brand.strip().lower()

    if key in MOBILE_APPS:
        spec = MOBILE_APPS[key]
        auth = _auth_part(user, password)
        return f"{spec['scheme']}://{auth}{ip}:{port or spec['port']}{spec['path']}"

    if key not in CAMERA_BRANDS:
        known = ", ".join(sorted({*CAMERA_BRANDS, *MOBILE_APPS}))
        raise ValueError(f"Brand '{brand}' maloom nahi. Available: {known}")

    spec = CAMERA_BRANDS[key]
    if quality not in ("main", "sub"):
        raise ValueError("quality sirf 'main' ya 'sub' ho sakti hai")

    auth = _auth_part(user, password)
    return f"rtsp://{auth}{ip}:{port or spec['port']}{spec[quality]}"


def _auth_part(user: str | None, password: str | None) -> str:
    """URL ka "user:pass@" hissa — dono khali hon to kuch nahi."""
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
    """Ek saved camera — naam, uska source, aur thoda context."""

    name: str
    url: str                 # "0" (USB webcam) / "http://..." / "rtsp://..."
    kind: str = "other"      # mobile | cctv | usb | other
    note: str = ""

    @property
    def safe_url(self) -> str:
        """Print karne ke liye — password chhupa hua."""
        return mask_credentials(self.url)


class CameraStoreError(RuntimeError):
    """cameras.json parh/likh na sakein to (corrupt JSON, permissions, etc.)."""


def _store_path(path: Path | str | None = None) -> Path:
    return Path(path) if path else config.CAMERAS_FILE


def load_cameras(path: Path | str | None = None) -> dict[str, CameraProfile]:
    """
    Saari saved cameras parho. File na ho to khali dict (ye error nahi hai —
    pehli baar chalane par normal hai).
    """
    store = _store_path(path)
    if not store.is_file():
        return {}

    try:
        raw = json.loads(store.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CameraStoreError(
            f"{store} valid JSON nahi hai (line {exc.lineno}): {exc.msg}\n"
            "File edit karte waqt comma ya bracket reh gaya hoga. "
            "Ya usay delete kar ke cameras.example.json dobara copy kar lein."
        ) from exc

    if not isinstance(raw, dict):
        raise CameraStoreError(f"{store} mein top level par object (dict) hona chahiye.")

    cameras: dict[str, CameraProfile] = {}
    for name, entry in raw.items():
        # Shortcut: {"phone": "http://..."} bhi chalta hai, poora object bhi
        if isinstance(entry, str):
            entry = {"url": entry}
        if not isinstance(entry, dict) or not entry.get("url"):
            logger.warning("Camera '%s' skip ki gayi — 'url' nahi mila.", name)
            continue
        cameras[name] = CameraProfile(
            name=name,
            url=str(entry["url"]),
            kind=str(entry.get("kind", "other")),
            note=str(entry.get("note", "")),
        )
    return cameras


def get_camera(name: str, path: Path | str | None = None) -> CameraProfile | None:
    """Naam se ek camera nikalo (case-insensitive). Na mile to None."""
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
    Ek camera save/update karo aur cameras.json likh do.
    Wohi naam dobara dein to purani entry replace ho jaati hai.
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
        raise CameraStoreError(f"{store} likhi nahi ja saki: {exc}") from exc

    logger.info("Camera '%s' saved in %s", name, store)
    return profile


def delete_camera(name: str, path: Path | str | None = None) -> bool:
    """Ek camera hata do. Mili aur hat gayi to True."""
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
    """`--list-cameras` ke liye ek saaf table (passwords chhupe hue)."""
    if not cameras:
        return (
            "Koi camera save nahi hai.\n\n"
            "Add karne ke liye:\n"
            "  python main.py --source http://192.168.1.5:8080/video --save-camera phone\n"
            "Ya cameras.example.json ko cameras.json bana kar edit kar lein."
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
