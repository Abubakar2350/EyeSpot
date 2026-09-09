"""
model_loader.py — YOLO model load karne wala code.

Yahan sirf ek kaam hota hai: weights load karna aur device (CPU/GPU) decide karna.
Detection logic detect.py mein hai.
"""

from __future__ import annotations

import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Model load na ho paaye to ye raise hoti hai (saaf error message ke sath)."""


def resolve_device(device: str | None = None) -> str:
    """
    Konsa device use karna hai ye decide karo.

    Agar user ne explicitly diya hai to wohi, warna auto-detect:
    CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.
    """
    if device:
        return device

    try:
        import torch
    except ImportError:  # torch ke bagair ultralytics chal hi nahi sakta
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_weights_path(model_name: str) -> str:
    """
    Weights ka path nikalo.

    - Agar user ne poora path diya hai (aur file mojood hai) to wohi use karo.
    - Agar models/ folder mein pehle se downloaded copy hai to wo use karo.
    - Warna sirf naam return karo — Ultralytics khud download kar lega.
    """
    candidate = Path(model_name)
    if candidate.is_file():
        return str(candidate)

    cached = config.MODELS_DIR / candidate.name
    if cached.is_file():
        return str(cached)

    return model_name


def load_model(model_name: str | None = None, device: str | None = None):
    """
    YOLO model load kar ke return karo.

    Args:
        model_name: e.g. "yolov8n.pt" ya koi custom .pt file ka path.
        device: "cpu" / "cuda" / "mps" / None (auto).

    Returns:
        (model, device) ka tuple.

    Raises:
        ModelLoadError: agar ultralytics installed na ho ya weights load na hon.
    """
    model_name = model_name or config.MODEL_NAME
    device = resolve_device(device or config.DEVICE)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelLoadError(
            "Ultralytics installed nahi hai.\n"
            "Install karein:  pip install -r requirements.txt"
        ) from exc

    weights = _resolve_weights_path(model_name)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model '%s' on device '%s'...", weights, device)
    try:
        model = YOLO(weights)
        model.to(device)
    except Exception as exc:  # network fail, corrupt weights, wrong path, etc.
        raise ModelLoadError(
            f"Model '{weights}' load nahi ho saka: {exc}\n"
            "Pehli baar run kar rahe hain to internet connection check karein "
            "(weights download hoti hain), ya --model se sahi path dein."
        ) from exc

    logger.info("Model ready. Classes: %d", len(model.names))
    return model, device


def expand_class_groups(names: list[str]) -> list[str]:
    """
    Group ke naam ko us ki classes mein khol do.

        ["animals"]        -> ["bird", "cat", "dog", ...]
        ["person", "pets"] -> ["person", "cat", "dog", "bird"]

    Jo naam group nahi hai wo waise ka waisa rehta hai. Duplicates hat jaate
    hain lekin tarteeb wohi rehti hai jo user ne di.
    """
    expanded: list[str] = []
    for name in names:
        key = name.strip().lower()
        expanded.extend(config.CLASS_GROUPS.get(key, [name]))

    seen: set[str] = set()
    unique: list[str] = []
    for name in expanded:
        if name.lower() not in seen:
            seen.add(name.lower())
            unique.append(name)
    return unique


def class_name_to_id(model, names: list[str]) -> list[int]:
    """
    Class ke naam (e.g. ["person", "car"]) ko YOLO ke class IDs mein badlo.

    Group ke naam (e.g. "animals") pehle apni classes mein khul jaate hain.
    Unknown naam par ValueError raise hoti hai taake user ko turant pata chale
    ke usne typo kiya hai.
    """
    if not names:
        return []

    names = expand_class_groups(names)

    # model.names => {0: "person", 1: "bicycle", ...}
    lookup = {label.lower(): idx for idx, label in model.names.items()}

    ids: list[int] = []
    unknown: list[str] = []
    for name in names:
        key = name.strip().lower()
        if key in lookup:
            ids.append(lookup[key])
        else:
            unknown.append(name)

    if unknown:
        groups = ", ".join(sorted(config.CLASS_GROUPS))
        available = ", ".join(sorted(lookup))
        raise ValueError(
            f"Ye class name(s) model mein mojood nahi: {unknown}\n"
            f"Groups: {groups}\n"
            f"Available classes: {available}"
        )

    return ids
