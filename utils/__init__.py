"""Utility package — FPS counting, camera profiles, output saving and the on-screen overlay."""

from .cameras import (
    CameraProfile,
    build_url,
    get_camera,
    load_cameras,
    mask_credentials,
    save_camera,
)
from .fps_counter import FPSCounter
from .overlay import draw_overlay
from .save_output import DetectionLogger, VideoWriter, save_screenshot

__all__ = [
    "CameraProfile",
    "build_url",
    "get_camera",
    "load_cameras",
    "mask_credentials",
    "save_camera",
    "FPSCounter",
    "draw_overlay",
    "DetectionLogger",
    "VideoWriter",
    "save_screenshot",
]
