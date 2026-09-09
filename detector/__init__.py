"""Detection package — model loading and frame processing logic."""

from .detect import Detection, Detector
from .model_loader import load_model, resolve_device

__all__ = ["Detection", "Detector", "load_model", "resolve_device"]
