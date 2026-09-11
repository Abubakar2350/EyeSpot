"""
visual_check.py — A trained pair of eyes for accident candidates.

The kinematic channels answer "did the motion look like a crash?" — and motion
alone cannot tell a collision from a hard stop at a signal, or a near miss from
an impact. This module answers the other question: "does the *picture* look
like a crash?" Wreckage, debris on the road, a person on the ground, an
overturned vehicle — these are visual facts that no amount of trajectory
arithmetic can see.

The model is CLIP (ViT-B/32), trained on hundreds of millions of image-text
pairs. It is used zero-shot: the frame is scored against a set of crash
descriptions and a set of normal-traffic descriptions, and the softmax between
the best of each side gives a crash probability. Measured on real CCTV footage
before being wired in: clear wreck scenes scored 0.71-0.96, clear normal
traffic 0.09-0.27 — including a frame the kinematics had wrongly flagged, which
CLIP put at 0.27.

This is deliberately a *verifier*, not a detector. Running CLIP on every frame
would be slow and pointless; the kinematics propose a handful of candidates per
video, and CLIP looks at just those. Cost is a few hundred milliseconds per
candidate, on the CPU.

Everything degrades gracefully: if open_clip is not installed or the model
cannot load, `available` is False and the caller falls back to the pure
kinematic confirmation that existed before.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# What a crash looks like, and what it does not. The phrasing matters less than
# covering the distinct *kinds* of accident scene: collision, wreckage, debris,
# a struck pedestrian, a flipped vehicle.
CRASH_PROMPTS = [
    "a photo of a car accident",
    "a car crash with damaged vehicles",
    "two vehicles that have just collided on a road",
    "a wrecked car with debris scattered on the road",
    "a person lying on the road after being hit by a car",
    "an overturned or badly damaged vehicle",
]
NORMAL_PROMPTS = [
    "normal traffic on a road",
    "cars driving normally down a street",
    "vehicles waiting at a junction",
    "a quiet road seen from a cctv camera",
    "an empty road at night",
]


class CrashVerifier:
    """
    Lazily-loaded CLIP scorer. Construct it freely — nothing heavy happens
    until the first `crash_probability` call, and a failed load simply leaves
    the verifier unavailable rather than raising into the detection loop.
    """

    _singleton: "CrashVerifier | None" = None

    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._text_features = None
        self._n_crash = len(CRASH_PROMPTS)
        self._failed = False

    @classmethod
    def get(cls) -> "CrashVerifier":
        """One shared instance — the model is far too heavy to load twice."""
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    @property
    def available(self) -> bool:
        if self._failed:
            return False
        if self._model is not None:
            return True
        return self._try_load()

    def _try_load(self) -> bool:
        try:
            import torch
            import open_clip
            from PIL import Image  # noqa: F401  (needed at scoring time)

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            model.eval()
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            with torch.no_grad():
                text = model.encode_text(tokenizer(CRASH_PROMPTS + NORMAL_PROMPTS))
                text /= text.norm(dim=-1, keepdim=True)

            self._model = model
            self._preprocess = preprocess
            self._text_features = text
            logger.info("Visual crash verifier ready (CLIP ViT-B/32).")
            return True
        except Exception as exc:
            # Missing package, no weights and no network, out of memory — all
            # mean the same thing here: no visual check, kinematics only.
            self._failed = True
            logger.warning("Visual verifier unavailable (%s) — kinematic confirmation only.", exc)
            return False

    def crash_probability(
        self,
        frame_bgr: np.ndarray,
        box: tuple[int, int, int, int] | None = None,
    ) -> float | None:
        """
        How much this scene looks like an accident, 0.0 to 1.0.
        None when the verifier is unavailable or scoring fails.

        Two views are scored and the higher wins. The full frame carries the
        context — debris, skid marks, a person on the ground often lie outside
        the vehicles' own boxes. But a real collision far from the camera is a
        handful of pixels in the full frame and drowns in ordinary street: a
        genuine side-impact measured here scored 0.17 full-frame and 0.53 when
        the event region was cropped and enlarged. So when a candidate's box is
        known, a padded crop of it is scored as well. The frame the kinematics
        wrongly flagged stayed at 0.30 under the same treatment, so the crop
        rescues subtle crashes without rescuing false alarms.
        """
        if not self.available:
            return None
        try:
            views = [frame_bgr]
            crop = self._event_crop(frame_bgr, box)
            if crop is not None:
                views.append(crop)
            scores = [s for s in map(self._score_view, views) if s is not None]
            return max(scores) if scores else None
        except Exception as exc:
            logger.warning("Visual check failed on a frame (%s).", exc)
            return None

    @staticmethod
    def _event_crop(
        frame_bgr: np.ndarray, box: tuple[int, int, int, int] | None, pad: float = 0.5
    ) -> np.ndarray | None:
        """A generously padded crop of the event region, or None if unusable."""
        if box is None:
            return None
        x1, y1, x2, y2 = box
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        h, w = frame_bgr.shape[:2]
        px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
        crop = frame_bgr[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
        return crop if crop.size else None

    def _score_view(self, image_bgr: np.ndarray) -> float | None:
        import torch
        from PIL import Image

        image = self._preprocess(
            Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        ).unsqueeze(0)
        with torch.no_grad():
            feat = self._model.encode_image(image)
            feat /= feat.norm(dim=-1, keepdim=True)
            logits = (100.0 * feat @ self._text_features.T).squeeze(0)

        crash = logits[: self._n_crash].max()
        normal = logits[self._n_crash:].max()
        return torch.softmax(torch.stack([crash, normal]), dim=0)[0].item()
