"""
model_loader.py — Loads the YOLO model.

This module does exactly two things: load the weights and decide which device
to run on. The detection logic lives in detect.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised, with an actionable message, when the model cannot be loaded."""


def resolve_device(device: str | None = None) -> str:
    """
    Decide which device to run inference on.

    An explicit choice always wins; otherwise auto-detect in this order:
    CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.
    """
    if device:
        return device

    try:
        import torch
    except ImportError:  # ultralytics cannot run without torch anyway
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_weights_path(model_name: str) -> str:
    """
    Work out where the weights live.

    - If the user gave a full path and the file exists, use it.
    - If a previously downloaded copy sits in models/, use that.
    - Otherwise return the bare name and let Ultralytics download it.
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
    Load and return the YOLO model.

    Args:
        model_name: e.g. "yolov8n.pt", or a path to a custom .pt file.
        device: "cpu" / "cuda" / "mps" / None (auto).

    Returns:
        A (model, device) tuple.

    Raises:
        ModelLoadError: if ultralytics is missing or the weights fail to load.
    """
    model_name = model_name or config.MODEL_NAME
    device = resolve_device(device or config.DEVICE)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelLoadError(
            "Ultralytics is not installed.\n"
            "Install it with:  pip install -r requirements.txt"
        ) from exc

    weights = _resolve_weights_path(model_name)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model '%s' on device '%s'...", weights, device)
    try:
        model = YOLO(weights)
        model.to(device)
    except Exception as exc:  # network failure, corrupt weights, bad path, etc.
        raise ModelLoadError(
            f"Could not load model '{weights}': {exc}\n"
            "On a first run the weights are downloaded, so check your internet "
            "connection — or pass a valid path with --model."
        ) from exc

    logger.info("Model ready. Classes: %d", len(model.names))
    return model, device


def expand_class_groups(names: list[str]) -> list[str]:
    """
    Expand group names into the classes they stand for.

        ["animals"]        -> ["bird", "cat", "dog", ...]
        ["person", "pets"] -> ["person", "cat", "dog", "bird"]

    Names that are not groups pass through unchanged. Duplicates are removed
    while the user's original ordering is preserved.
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
    Convert class names (e.g. ["person", "car"]) into YOLO class IDs.

    Group names (e.g. "animals") are expanded first. Unknown names raise a
    ValueError so that typos surface immediately.
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
            f"Unknown class name(s): {unknown}\n"
            f"Groups: {groups}\n"
            f"Available classes: {available}"
        )

    return ids
