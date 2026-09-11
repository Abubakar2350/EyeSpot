"""
accident.py — Road accident detection from tracked objects.

Detection tells you what is in a frame. Tracking gives each object a persistent
identity. This module is the layer above both: it watches how those identities
*move* over time and decides when the motion stops looking like traffic and
starts looking like a crash.

Nothing here runs a neural network of its own. It consumes the `Detection`
objects that `Detector.track()` already produces, so it costs almost nothing on
top of tracking — the work is arithmetic on positions.

## The pipeline

    Detection -> Tracking -> Kinematics -> Signals -> Fusion -> Event

## The signals

Six independent channels, each scored 0.0 to 1.0, then fused:

| Signal | What it looks for |
|---|---|
| `impact` | Two road users whose boxes overlap sharply while closing on each other |
| `hard_brake` | Deceleration far beyond normal traffic behaviour |
| `sudden_stop` | A moving object arriving at a standstill within a fraction of a second |
| `swerve` | Heading changing faster than steering can explain |
| `ttc` | Time-to-collision for a converging pair falling below a threshold |
| `person_down` | A person's bounding box turning from tall to wide, and staying wide |

No single channel fires an event. That is deliberate: each one alone has an
innocent explanation — a car braking at a light, a pedestrian crouching, two
vehicles overlapping only because one is behind the other. Requiring several to
agree is what separates an accident from ordinary traffic.

## Scale normalisation

Speeds are measured in **object lengths per second**, not pixels per second.
A pixel means nothing on its own: the same car crossing the same road produces
wildly different pixel speeds depending on how far away the camera is and what
resolution it records at. Dividing by the object's own diagonal cancels most of
that out, so one set of thresholds works across a dashcam, a CCTV pole and a
phone held at the roadside.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

import config
from .detect import Detection

# COCO classes that behave like vehicles on a road
VEHICLE_CLASSES = frozenset({"car", "motorcycle", "bus", "truck", "bicycle"})

# Road users who get hurt — used to weight an impact more heavily
VULNERABLE_CLASSES = frozenset({"person", "bicycle", "motorcycle"})

# Everything this module pays attention to
ROAD_USER_CLASSES = VEHICLE_CLASSES | {"person"}

# Event HUD colours (BGR)
_ALERT = (60, 60, 240)       # red
_ALERT_DIM = (90, 90, 200)
_TEXT = (255, 255, 255)


# ---------------------------------------------------------------------------
# Per-track kinematics
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    """
    Motion history for one tracked object.

    Samples are stored with real timestamps rather than frame numbers, so the
    derived speeds stay correct whether the source runs at 5 FPS or 60.
    """

    track_id: int
    label: str
    # (timestamp, centre_x, centre_y, diagonal, width, height)
    samples: deque = field(default_factory=lambda: deque(maxlen=30))
    # (timestamp, acceleration) — kept so an impact can be correlated with a
    # jolt that happened a frame or two either side of it, rather than only in
    # the exact frame the boxes overlapped
    accel_history: deque = field(default_factory=lambda: deque(maxlen=30))
    last_seen: float = 0.0

    def update(self, det: Detection, now: float) -> None:
        cx, cy = det.center
        diagonal = math.hypot(det.width, det.height) or 1.0
        self.samples.append((now, float(cx), float(cy), diagonal, det.width, det.height))
        self.last_seen = now
        self.label = det.label
        self.accel_history.append((now, self.acceleration()))

    def peak_deceleration(self, window: float = 0.6) -> float:
        """
        The sharpest deceleration seen in the last `window` seconds, as a
        positive number. Zero if the object only ever sped up.

        An impact and the deceleration it causes rarely land in the same frame —
        boxes may overlap slightly before the tracker registers the speed change,
        or just after. Looking back over a short window catches both.
        """
        if not self.accel_history:
            return 0.0
        now = self.accel_history[-1][0]
        recent = [a for t, a in self.accel_history if now - t <= window]
        return max(0.0, -min(recent)) if recent else 0.0

    # -- derived quantities -------------------------------------------------

    @property
    def scale(self) -> float:
        """The object's own diagonal, used to convert pixels into body lengths."""
        return self.samples[-1][3] if self.samples else 1.0

    def velocity(self, window: float = 0.25) -> tuple[float, float]:
        """
        Velocity in pixels per second, averaged over the last `window` seconds.

        A short window would track noise in the box edges; a long one would
        smooth a real impact away. A quarter of a second is a reasonable
        compromise for traffic.
        """
        if len(self.samples) < 2:
            return 0.0, 0.0

        latest = self.samples[-1]
        earliest = self.samples[0]
        for sample in reversed(self.samples):
            if latest[0] - sample[0] >= window:
                earliest = sample
                break

        dt = latest[0] - earliest[0]
        if dt <= 0:
            return 0.0, 0.0
        return (latest[1] - earliest[1]) / dt, (latest[2] - earliest[2]) / dt

    def speed(self, window: float = 0.25) -> float:
        """Speed in object lengths per second — scale independent."""
        vx, vy = self.velocity(window)
        return math.hypot(vx, vy) / max(self.scale, 1.0)

    def acceleration(self, window: float = 0.25) -> float:
        """
        Change in speed per second, in object lengths. Negative means slowing.

        Computed by comparing the most recent window against the one before it,
        which is why at least two windows of history are needed.
        """
        if len(self.samples) < 4:
            return 0.0

        now = self.samples[-1][0]
        recent = [s for s in self.samples if now - s[0] <= window]
        older = [s for s in self.samples if window < now - s[0] <= window * 2]
        if len(recent) < 2 or len(older) < 2:
            return 0.0

        def segment_speed(seg) -> float:
            dt = seg[-1][0] - seg[0][0]
            if dt <= 0:
                return 0.0
            dist = math.hypot(seg[-1][1] - seg[0][1], seg[-1][2] - seg[0][2])
            return dist / dt / max(self.scale, 1.0)

        dt = max(now - older[0][0], 1e-3)
        return (segment_speed(recent) - segment_speed(older)) / dt

    def heading(self, window: float = 0.25) -> float | None:
        """Direction of travel in radians, or None while effectively stationary."""
        vx, vy = self.velocity(window)
        if math.hypot(vx, vy) < max(self.scale, 1.0) * 0.15:
            return None
        return math.atan2(vy, vx)

    def yaw_rate(self, window: float = 0.25) -> float:
        """
        How fast the direction of travel is changing, in radians per second.

        A vehicle taking a normal corner turns slowly; a vehicle knocked
        sideways changes heading almost instantly.
        """
        if len(self.samples) < 6:
            return 0.0

        now = self.samples[-1][0]
        mid = [s for s in self.samples if now - s[0] <= window]
        old = [s for s in self.samples if window < now - s[0] <= window * 2]
        if len(mid) < 2 or len(old) < 2:
            return 0.0

        def direction(seg) -> float | None:
            dx, dy = seg[-1][1] - seg[0][1], seg[-1][2] - seg[0][2]
            if math.hypot(dx, dy) < max(self.scale, 1.0) * 0.1:
                return None
            return math.atan2(dy, dx)

        a, b = direction(old), direction(mid)
        if a is None or b is None:
            return 0.0

        delta = abs(math.atan2(math.sin(b - a), math.cos(b - a)))  # wrap to ±π
        dt = max(now - old[0][0], 1e-3)
        return delta / dt

    def aspect_ratio(self) -> float:
        """Width ÷ height of the latest box. A standing person is well below 1."""
        if not self.samples:
            return 0.0
        _, _, _, _, w, h = self.samples[-1]
        return w / h if h else 0.0


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class AccidentEvent:
    """One triggered accident, with the evidence that produced it."""

    frame: int
    timestamp: datetime
    score: float
    signals: dict[str, float]
    track_ids: list[int]
    labels: list[str]
    box: tuple[int, int, int, int]

    @property
    def summary(self) -> str:
        """The signals that actually contributed, strongest first."""
        firing = sorted(
            ((name, value) for name, value in self.signals.items() if value > 0.05),
            key=lambda kv: -kv[1],
        )
        return ", ".join(f"{name} {value:.2f}" for name, value in firing) or "—"


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------

class AccidentDetector:
    """
    Watches tracked road users and raises an event when their motion stops
    resembling traffic.

    Usage:
        accidents = AccidentDetector()
        event = accidents.update(detections, frame_index)   # once per frame
        accidents.draw(frame)
    """

    def __init__(
        self,
        threshold: float | None = None,
        persistence: int | None = None,
        cooldown: float | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.threshold = threshold if threshold is not None else config.ACCIDENT_THRESHOLD
        self.persistence = persistence if persistence is not None else config.ACCIDENT_PERSISTENCE
        self.cooldown = cooldown if cooldown is not None else config.ACCIDENT_COOLDOWN
        self.weights = dict(weights or config.ACCIDENT_WEIGHTS)

        self.tracks: dict[int, TrackState] = {}
        self.events: list[AccidentEvent] = []

        self._above_threshold = 0
        # Negative infinity, not zero: the clock starts near zero on a fresh
        # clip, so a zero here would silence the cooldown's whole length at the
        # very start — exactly where a dashcam accident usually is.
        self._last_event_time = float("-inf")
        self._last_score = 0.0
        self._last_signals: dict[str, float] = {}
        self._person_down_since: dict[int, float] = {}
        self._banner_until = 0.0

    # -- main entry point ---------------------------------------------------

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        now: float | None = None,
    ) -> AccidentEvent | None:
        """
        Feed one frame's detections. Returns an AccidentEvent when one fires,
        otherwise None.

        `now` accepts an explicit monotonic timestamp so that tests can drive
        the clock deterministically instead of sleeping.
        """
        if now is None:
            import time

            now = time.perf_counter()

        self._prune(now)

        road_users = [
            d for d in detections
            if d.track_id is not None and d.label in ROAD_USER_CLASSES
        ]
        for det in road_users:
            state = self.tracks.get(det.track_id)
            if state is None:
                state = TrackState(track_id=det.track_id, label=det.label)
                self.tracks[det.track_id] = state
            state.update(det, now)

        signals, involved = self._score(road_users, now)
        score = sum(signals.get(name, 0.0) * weight for name, weight in self.weights.items())
        score = min(score, 1.0)

        self._last_score = score
        self._last_signals = signals

        # Persistence: a score has to hold for several frames. A single frame of
        # nonsense — one bad box, one missed detection — should never fire.
        if score >= self.threshold:
            self._above_threshold += 1
        else:
            self._above_threshold = 0

        if self._above_threshold < self.persistence:
            return None

        # Cooldown: one crash produces many frames of chaos. Without this, a
        # single accident would raise dozens of events.
        if now - self._last_event_time < self.cooldown:
            return None

        event = self._build_event(frame_index, score, signals, involved)
        self._last_event_time = now
        self._above_threshold = 0
        self._banner_until = now + 3.0
        self.events.append(event)
        return event

    # -- signal channels ----------------------------------------------------

    def _score(
        self, detections: list[Detection], now: float
    ) -> tuple[dict[str, float], list[Detection]]:
        """Evaluate every channel and note which objects are implicated."""
        signals = {
            "impact": 0.0,
            "hard_brake": 0.0,
            "sudden_stop": 0.0,
            "swerve": 0.0,
            "ttc": 0.0,
            "person_down": 0.0,
        }
        involved: dict[int, Detection] = {}

        by_id = {d.track_id: d for d in detections}

        # --- single-object channels ---
        for det in detections:
            state = self.tracks.get(det.track_id)
            if state is None or len(state.samples) < 4:
                continue

            accel = state.acceleration()
            if accel < -config.ACCIDENT_BRAKE_DECEL:
                strength = _ramp(-accel, config.ACCIDENT_BRAKE_DECEL, config.ACCIDENT_BRAKE_DECEL * 3)
                if strength > signals["hard_brake"]:
                    signals["hard_brake"] = strength
                involved[det.track_id] = det

            if self._is_sudden_stop(state):
                signals["sudden_stop"] = 1.0
                involved[det.track_id] = det

            yaw = state.yaw_rate()
            if yaw > config.ACCIDENT_YAW_RATE:
                strength = _ramp(yaw, config.ACCIDENT_YAW_RATE, config.ACCIDENT_YAW_RATE * 3)
                if strength > signals["swerve"]:
                    signals["swerve"] = strength
                involved[det.track_id] = det

            if det.label == "person":
                strength = self._person_down_score(state, now)
                if strength > signals["person_down"]:
                    signals["person_down"] = strength
                if strength > 0:
                    involved[det.track_id] = det

        # --- pairwise channels ---
        for i, a in enumerate(detections):
            for b in detections[i + 1:]:
                state_a = self.tracks.get(a.track_id)
                state_b = self.tracks.get(b.track_id)
                if state_a is None or state_b is None:
                    continue
                if len(state_a.samples) < 3 or len(state_b.samples) < 3:
                    continue

                overlap = _iou(a.box, b.box)
                if overlap > config.ACCIDENT_IMPACT_IOU:
                    # Overlap alone is perspective, not a crash — one vehicle
                    # passing behind another looks identical. Require that the
                    # pair was closing, and that one of them took a jolt within
                    # the last fraction of a second.
                    approached = _was_closing(state_a, state_b)
                    jolt = max(state_a.peak_deceleration(), state_b.peak_deceleration())
                    if approached and jolt > config.ACCIDENT_BRAKE_DECEL * 0.5:
                        # Those two corroborating facts are what make this an
                        # impact rather than perspective, so meeting them earns
                        # most of the channel's strength outright. How much the
                        # boxes overlap only modulates the rest: colliding
                        # vehicles graze edge to edge, and a large overlap is
                        # the exception, not the signature.
                        strength = 0.6 + 0.4 * _ramp(overlap, config.ACCIDENT_IMPACT_IOU, 0.40)
                        # A pedestrian or rider being hit is the worse outcome,
                        # and the one worth being more sensitive about.
                        if a.label in VULNERABLE_CLASSES or b.label in VULNERABLE_CLASSES:
                            strength = min(1.0, strength * 1.3)
                        if strength > signals["impact"]:
                            signals["impact"] = strength
                        involved[a.track_id] = a
                        involved[b.track_id] = b

                ttc = _time_to_collision(state_a, state_b)
                if ttc is not None and ttc < config.ACCIDENT_TTC_SECONDS:
                    strength = _ramp(config.ACCIDENT_TTC_SECONDS - ttc, 0.0, config.ACCIDENT_TTC_SECONDS)
                    if strength > signals["ttc"]:
                        signals["ttc"] = strength
                    involved[a.track_id] = a
                    involved[b.track_id] = b

        # Keep only implicated objects that still exist this frame
        involved_dets = [by_id[tid] for tid in involved if tid in by_id]
        return signals, involved_dets

    def _is_sudden_stop(self, state: TrackState) -> bool:
        """Was this object moving a moment ago and is now effectively still?"""
        if len(state.samples) < 6:
            return False
        recent = state.speed(window=0.2)
        earlier = state.speed(window=0.6)
        return earlier > config.ACCIDENT_MOVING_SPEED and recent < earlier * 0.15

    def _person_down_score(self, state: TrackState, now: float) -> float:
        """
        A person's box is normally much taller than wide. When it turns wide and
        stays wide, they are probably on the ground.

        The dwell requirement matters: someone bending to tie a shoelace
        produces the same shape for a moment. Someone who has been knocked down
        stays there.
        """
        wide = state.aspect_ratio() > config.ACCIDENT_PERSON_DOWN_RATIO
        if not wide:
            self._person_down_since.pop(state.track_id, None)
            return 0.0

        since = self._person_down_since.setdefault(state.track_id, now)
        held = now - since
        if held < config.ACCIDENT_PERSON_DOWN_SECONDS:
            return 0.0
        return _ramp(held, config.ACCIDENT_PERSON_DOWN_SECONDS, config.ACCIDENT_PERSON_DOWN_SECONDS * 3)

    # -- housekeeping -------------------------------------------------------

    def _prune(self, now: float, max_age: float = 3.0) -> None:
        """Forget tracks that have not been seen for a while."""
        stale = [tid for tid, s in self.tracks.items() if now - s.last_seen > max_age]
        for tid in stale:
            del self.tracks[tid]
            self._person_down_since.pop(tid, None)

    def _build_event(
        self,
        frame_index: int,
        score: float,
        signals: dict[str, float],
        involved: list[Detection],
    ) -> AccidentEvent:
        if involved:
            x1 = min(d.box[0] for d in involved)
            y1 = min(d.box[1] for d in involved)
            x2 = max(d.box[2] for d in involved)
            y2 = max(d.box[3] for d in involved)
            box = (x1, y1, x2, y2)
        else:
            box = (0, 0, 0, 0)

        return AccidentEvent(
            frame=frame_index,
            timestamp=datetime.now(),
            score=score,
            signals=dict(signals),
            track_ids=[d.track_id for d in involved if d.track_id is not None],
            labels=sorted({d.label for d in involved}),
            box=box,
        )

    # -- reporting ----------------------------------------------------------

    @property
    def score(self) -> float:
        """The most recent fused score, for a live risk meter."""
        return self._last_score

    @property
    def signals(self) -> dict[str, float]:
        return dict(self._last_signals)

    def status_line(self) -> str:
        """A compact line for the overlay panel."""
        if self.events:
            return f"Accident risk: {self._last_score:.2f}  ({len(self.events)} event(s))"
        return f"Accident risk: {self._last_score:.2f}"

    def draw(self, frame: np.ndarray, now: float | None = None) -> np.ndarray:
        """
        Draw the risk meter, and a banner over the scene while an event is fresh.
        """
        if now is None:
            import time

            now = time.perf_counter()

        h, w = frame.shape[:2]

        # Risk meter, bottom-right, always visible while enabled
        meter_w, meter_h = 160, 10
        x, y = w - meter_w - 16, h - 46
        cv2.rectangle(frame, (x, y), (x + meter_w, y + meter_h), (60, 60, 60), -1)
        filled = int(meter_w * min(self._last_score, 1.0))
        colour = _ALERT if self._last_score >= self.threshold else (80, 200, 120)
        if filled:
            cv2.rectangle(frame, (x, y), (x + filled, y + meter_h), colour, -1)
        cv2.putText(
            frame, f"RISK {self._last_score:.2f}", (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TEXT, 1, cv2.LINE_AA,
        )

        if now > self._banner_until or not self.events:
            return frame

        event = self.events[-1]

        # Outline the region where it happened
        x1, y1, x2, y2 = event.box
        if x2 > x1 and y2 > y1:
            cv2.rectangle(frame, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), _ALERT, 3)

        # Banner across the top
        text = f"ACCIDENT DETECTED  {event.score:.2f}"
        detail = event.summary
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)

        cv2.rectangle(frame, (0, 0), (w, th + 34), (25, 25, 25), -1)
        cv2.rectangle(frame, (0, 0), (w, 4), _ALERT, -1)
        cv2.putText(frame, text, (14, th + 8), font, 0.8, _ALERT, 2, cv2.LINE_AA)
        cv2.putText(frame, detail, (14, th + 27), font, 0.45, _ALERT_DIM, 1, cv2.LINE_AA)
        return frame


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _ramp(value: float, low: float, high: float) -> float:
    """Map `value` onto 0..1 across the range low..high, clamped at both ends."""
    if high <= low:
        return 1.0 if value >= high else 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over union of two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    if intersection == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _closing_speed(a: TrackState, b: TrackState) -> float:
    """
    How fast two objects are approaching each other, in pixels per second.
    Negative means they are separating.
    """
    if not a.samples or not b.samples:
        return 0.0

    ax, ay = a.samples[-1][1], a.samples[-1][2]
    bx, by = b.samples[-1][1], b.samples[-1][2]
    rx, ry = bx - ax, by - ay
    distance = math.hypot(rx, ry)
    if distance < 1e-6:
        return 0.0

    avx, avy = a.velocity()
    bvx, bvy = b.velocity()
    # Component of relative velocity along the line joining them
    return -((bvx - avx) * rx + (bvy - avy) * ry) / distance


def _was_closing(a: TrackState, b: TrackState, window: float = 0.8) -> bool:
    """
    Were these two objects approaching each other over the last `window` seconds?

    Instantaneous closing speed is the wrong test immediately after an impact,
    because both objects have usually stopped by then and their relative
    velocity is near zero. Comparing the distance between them now against the
    distance a moment ago answers the question that actually matters: did they
    come together, or were they always apart?
    """
    if len(a.samples) < 3 or len(b.samples) < 3:
        return False

    now = a.samples[-1][0]

    def distance_at(target: float) -> float | None:
        sa = min(a.samples, key=lambda s: abs(s[0] - target))
        sb = min(b.samples, key=lambda s: abs(s[0] - target))
        if abs(sa[0] - target) > window or abs(sb[0] - target) > window:
            return None
        return math.hypot(sb[1] - sa[1], sb[2] - sa[2])

    then = distance_at(now - window)
    current = distance_at(now)
    if then is None or current is None:
        return False
    return current < then


def _time_to_collision(a: TrackState, b: TrackState) -> float | None:
    """
    Seconds until two objects meet, assuming both keep their current velocity.

    Returns None when they are not closing, or are already touching — in the
    latter case the impact channel is the right one to speak, not this.
    """
    closing = _closing_speed(a, b)
    if closing <= 0:
        return None

    ax, ay = a.samples[-1][1], a.samples[-1][2]
    bx, by = b.samples[-1][1], b.samples[-1][2]
    distance = math.hypot(bx - ax, by - ay)

    # Subtract the objects' own extents: they collide when the gap between their
    # edges closes, not when their centres coincide.
    gap = distance - (a.scale + b.scale) / 2.0
    if gap <= 0:
        return None

    return gap / closing
