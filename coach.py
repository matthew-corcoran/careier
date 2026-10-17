#!/usr/bin/env python3
"""Live REBA-style coach: webcam pose in, injury-risk score out."""

from __future__ import annotations

import argparse
import subprocess
import time
import urllib.request
from collections import deque
from dataclasses import replace
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
from ultralytics import YOLO

from reba import (
    SQUAT_CLOSE,
    STOOP_TWIST,
    RebaAngles,
    RebaResult,
    score_reba,
)
from hub_bridge import HubBridge

POSE_CONNECTIONS = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (0, 11),
    (0, 12),
)

LM = {
    "nose": 0,
    "l_ear": 7,
    "r_ear": 8,
    "l_sh": 11,
    "r_sh": 12,
    "l_el": 13,
    "r_el": 14,
    "l_wr": 15,
    "r_wr": 16,
    "l_hip": 23,
    "r_hip": 24,
    "l_kn": 25,
    "r_kn": 26,
    "l_an": 27,
    "r_an": 28,
}

VIS_MIN = 0.35
EMA = 0.35
ACTIVITY_JUMP_DEG = 22.0
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
YOLO_MODEL_PATH = ROOT / "yolo26s-pose.pt"

INK = (32, 32, 30)
PAPER = (236, 236, 234)
CARD = (252, 252, 250)
MUTED = (130, 128, 122)
HAIRLINE = (214, 214, 210)
BLUE = (255, 107, 47)
GREEN = (96, 168, 52)
AMBER = (32, 148, 214)
RED = (68, 68, 214)
WHITE = (255, 255, 255)
BLACK = INK
WINDOW_TITLE = "careier."
STAGE_W, STAGE_H = 1280, 760


def _xyz(lm) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64)


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = _unit(u), _unit(v)
    dot = float(np.clip(np.dot(nu, nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return angle_between(a - b, c - b)


def band_color(band: str) -> tuple[int, int, int]:
    if band in ("NEGLIGIBLE", "LOW"):
        return GREEN
    if band == "MEDIUM":
        return AMBER
    return RED


class AngleFilter:
    """Hold last good angles when a landmark drops out, then EMA-smooth."""

    def __init__(self) -> None:
        self._raw: RebaAngles | None = None
        self._smooth: RebaAngles | None = None
        self._prev_trunk = 0.0
        self._trunk_hist: deque[tuple[float, float]] = deque(maxlen=12)

    def update(self, incoming: RebaAngles | None, vis_ok: bool, now: float) -> RebaAngles | None:
        if incoming is not None and vis_ok:
            self._raw = incoming
        if self._raw is None:
            return None
        if self._smooth is None:
            self._smooth = self._raw
        else:
            self._smooth = _ema(self._smooth, self._raw, EMA)
        self._trunk_hist.append((now, self._smooth.trunk_flex_deg))
        return self._smooth

    def reset(self) -> None:
        self._raw = None
        self._smooth = None
        self._prev_trunk = 0.0
        self._trunk_hist.clear()

    def activity(self, now: float) -> int:
        if len(self._trunk_hist) < 4:
            return 0
        window = [(t, a) for t, a in self._trunk_hist if now - t <= 0.45]
        if len(window) < 2:
            return 0
        span = max(a for _, a in window) - min(a for _, a in window)
        return 1 if span >= ACTIVITY_JUMP_DEG else 0


def _ema(prev: RebaAngles, new: RebaAngles, a: float) -> RebaAngles:
    def mix(p: float, n: float) -> float:
        return p * (1.0 - a) + n * a

    return RebaAngles(
        trunk_flex_deg=mix(prev.trunk_flex_deg, new.trunk_flex_deg),
        trunk_twist_deg=mix(prev.trunk_twist_deg, new.trunk_twist_deg),
        trunk_side_deg=mix(prev.trunk_side_deg, new.trunk_side_deg),
        neck_flex_deg=mix(prev.neck_flex_deg, new.neck_flex_deg),
        neck_twist=new.neck_twist,
        knee_flex_deg=mix(prev.knee_flex_deg, new.knee_flex_deg),
        upper_arm_deg=mix(prev.upper_arm_deg, new.upper_arm_deg),
        lower_arm_deg=mix(prev.lower_arm_deg, new.lower_arm_deg),
        arm_abducted=new.arm_abducted,
        unstable_base=new.unstable_base,
    )


def _lm_vis(lm) -> float:
    return float(getattr(lm, "visibility", 1.0) or 0.0)


def _vis(landmarks, *names: str) -> float:
    return min(_lm_vis(landmarks[LM[n]]) for n in names)


def ensure_pose_model() -> Path:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1000:
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def open_landmarker() -> PoseLandmarker:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_pose_model())),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    return PoseLandmarker.create_from_options(options)


class RemappedLandmark:
    """Image landmark remapped from a crop into full-frame coordinates."""

    def __init__(self, landmark, x0: float, width: float) -> None:
        self.x = x0 + landmark.x * width
        self.y = landmark.y
        self.z = landmark.z * width
        self.visibility = _lm_vis(landmark)
        self.presence = float(getattr(landmark, "presence", 1.0) or 0.0)


def detect_in_roi(
    frame: np.ndarray,
    landmarker: PoseLandmarker,
    x0: float,
    x1: float,
    expected_side: str,
) -> dict | None:
    """Detect one person in a dedicated overlapping horizontal ROI."""
    height, width = frame.shape[:2]
    px0 = max(0, min(width - 1, int(x0 * width)))
    px1 = max(px0 + 1, min(width, int(x1 * width)))
    crop = frame[:, px0:px1]
    rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    output = landmarker.detect(image)
    if not output.pose_landmarks:
        return None

    span = x1 - x0
    image_lms = [RemappedLandmark(lm, x0, span) for lm in output.pose_landmarks[0]]
    world_lms = (
        output.pose_world_landmarks[0]
        if output.pose_world_landmarks
        else output.pose_landmarks[0]
    )
    person = pose_features(image_lms, world_lms)

    # Reject the other person when they dominate the ROI overlap.
    if expected_side == "left" and person["hip_x"] > 0.62:
        return None
    if expected_side == "right" and person["hip_x"] < 0.38:
        return None
    return person


def angles_from_pose(landmarks) -> tuple[RebaAngles | None, bool]:
    # World landmarks often omit visibility; caller gates on image landmarks.

    l_sh, r_sh = _xyz(landmarks[LM["l_sh"]]), _xyz(landmarks[LM["r_sh"]])
    l_hip, r_hip = _xyz(landmarks[LM["l_hip"]]), _xyz(landmarks[LM["r_hip"]])
    nose = _xyz(landmarks[LM["nose"]])
    mid_sh, mid_hip = _mid(l_sh, r_sh), _mid(l_hip, r_hip)
    trunk = mid_sh - mid_hip
    vertical = np.array([0.0, -1.0, 0.0])

    trunk_flex = angle_between(trunk, vertical)
    # Image landmarks have no depth, so this "side" angle is the same lean
    # already counted as trunk flexion. Leave it at 0 instead of double scoring.
    trunk_side = 0.0

    sh_line = np.array([r_sh[0] - l_sh[0], r_sh[1] - l_sh[1]])
    hip_line = np.array([r_hip[0] - l_hip[0], r_hip[1] - l_hip[1]])
    trunk_twist = angle_between(
        np.array([sh_line[0], sh_line[1], 0.0]),
        np.array([hip_line[0], hip_line[1], 0.0]),
    )

    neck_vec = nose - mid_sh
    neck_vs_up = angle_between(neck_vec, vertical)
    neck_flex = neck_vs_up - trunk_flex * 0.35
    neck_twist = trunk_twist >= 25

    knees: list[float] = []
    for side in ("l", "r"):
        if _vis(landmarks, f"{side}_hip", f"{side}_kn", f"{side}_an") >= VIS_MIN:
            hip = _xyz(landmarks[LM[f"{side}_hip"]])
            kn = _xyz(landmarks[LM[f"{side}_kn"]])
            an = _xyz(landmarks[LM[f"{side}_an"]])
            knees.append(180.0 - joint_angle(hip, kn, an))
    knee_flex = max(knees) if knees else 10.0
    # One knee down and one up is a half-kneel, not an unstable base.
    unstable = False

    arm_elevs: list[float] = []
    elbow_angs: list[float] = []
    abducted = False
    for side in ("l", "r"):
        if _vis(landmarks, f"{side}_sh", f"{side}_el") < VIS_MIN:
            continue
        sh = _xyz(landmarks[LM[f"{side}_sh"]])
        el = _xyz(landmarks[LM[f"{side}_el"]])
        upper = el - sh
        arm_elevs.append(angle_between(upper, -trunk))
        lateral = abs(upper[0])
        if lateral > 0.82 * (np.linalg.norm(upper) + 1e-8):
            abducted = True
        if _vis(landmarks, f"{side}_wr") >= VIS_MIN:
            wr = _xyz(landmarks[LM[f"{side}_wr"]])
            elbow_angs.append(joint_angle(sh, el, wr))

    upper_arm = max(arm_elevs) if arm_elevs else 20.0
    lower_arm = min(elbow_angs) if elbow_angs else 90.0

    return (
        RebaAngles(
            trunk_flex_deg=trunk_flex,
            trunk_twist_deg=trunk_twist,
            trunk_side_deg=trunk_side,
            neck_flex_deg=neck_flex,
            neck_twist=neck_twist,
            knee_flex_deg=knee_flex,
            upper_arm_deg=upper_arm,
            lower_arm_deg=lower_arm,
            arm_abducted=abducted,
            unstable_base=unstable,
        ),
        True,
    )


CARER_COLOR = BLUE
PATIENT_COLOR = (42, 42, 40)


def _pt(landmarks, name: str, w: int, h: int) -> tuple[int, int]:
    lm = landmarks[LM[name]]
    return int(lm.x * w), int(lm.y * h)


def hip_center(landmarks) -> tuple[float, float]:
    l_hip, r_hip = landmarks[LM["l_hip"]], landmarks[LM["r_hip"]]
    return ((l_hip.x + r_hip.x) * 0.5, (l_hip.y + r_hip.y) * 0.5)


def patient_neck_angle(landmarks, aspect: float) -> float | None:
    """Angle between the patient's torso axis and head axis in image space."""
    def xy(name: str) -> np.ndarray:
        lm = landmarks[LM[name]]
        return np.array([lm.x * aspect, lm.y], dtype=np.float64)

    def visible_mean(names: tuple[str, ...], threshold: float) -> np.ndarray | None:
        points = [
            xy(name)
            for name in names
            if _lm_vis(landmarks[LM[name]]) >= threshold
        ]
        return np.mean(points, axis=0) if points else None

    shoulder = visible_mean(("l_sh", "r_sh"), 0.15)
    hip = visible_mean(("l_hip", "r_hip"), 0.15)
    if shoulder is None or hip is None:
        return None
    visible_ears = [
        xy(name)
        for name in ("l_ear", "r_ear")
        if _lm_vis(landmarks[LM[name]]) >= 0.12
    ]
    if visible_ears:
        head = sum(visible_ears) / len(visible_ears)
    elif _lm_vis(landmarks[LM["nose"]]) >= 0.15:
        head = xy("nose")
    else:
        return None

    torso_axis = shoulder - hip
    head_axis = head - shoulder
    if np.linalg.norm(torso_axis) < 1e-5 or np.linalg.norm(head_axis) < 1e-5:
        return None
    return angle_between(
        np.array([torso_axis[0], torso_axis[1], 0.0]),
        np.array([head_axis[0], head_axis[1], 0.0]),
    )


def patient_head_ok(neck_angle: float | None) -> bool | None:
    if neck_angle is None:
        return None
    return neck_angle <= 55.0


def neck_band(neck_angle: float | None) -> str | None:
    if neck_angle is None:
        return None
    if neck_angle <= 35.0:
        return "safe"
    if neck_angle <= 55.0:
        return "caution"
    return "danger"


def pose_features(image_lms, world_lms) -> dict:
    hx, hy = hip_center(image_lms)
    angles, _ = angles_from_pose(world_lms)
    trunk = angles.trunk_flex_deg if angles else 45.0
    return {
        "hip_x": hx,
        "hip_y": hy,
        "trunk_flex": trunk,
        "patient_score": trunk + hy * 55.0,
        "angles": angles,
        "image": image_lms,
        "world": world_lms,
    }


class PointLandmark:
    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        confidence: float = 0.0,
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = 0.0
        self.visibility = float(confidence)
        self.presence = float(confidence)


# COCO-17 keypoints emitted by YOLO -> the MediaPipe-style slots used below.
YOLO_TO_LM = {
    0: LM["nose"],
    3: LM["l_ear"],
    4: LM["r_ear"],
    5: LM["l_sh"],
    6: LM["r_sh"],
    7: LM["l_el"],
    8: LM["r_el"],
    9: LM["l_wr"],
    10: LM["r_wr"],
    11: LM["l_hip"],
    12: LM["r_hip"],
    13: LM["l_kn"],
    14: LM["r_kn"],
    15: LM["l_an"],
    16: LM["r_an"],
}


def load_yolo_pose() -> YOLO:
    model = YOLO(str(YOLO_MODEL_PATH if YOLO_MODEL_PATH.exists() else "yolo26s-pose.pt"))
    # Warm up once so the first live frame does not freeze the UI.
    model.predict(
        np.zeros((384, 640, 3), dtype=np.uint8),
        imgsz=640,
        conf=0.25,
        device="cpu",
        verbose=False,
    )
    return model


def yolo_people(frame: np.ndarray, model: YOLO) -> list[dict]:
    """One full-frame inference that returns every detected person."""
    h, w = frame.shape[:2]
    output = model.track(
        frame,
        imgsz=640,
        conf=0.22,
        iou=0.55,
        max_det=4,
        device="cpu",
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )[0]
    if output.keypoints is None or output.boxes is None:
        return []

    xy = output.keypoints.xy.cpu().numpy()
    conf_tensor = output.keypoints.conf
    conf = (
        conf_tensor.cpu().numpy()
        if conf_tensor is not None
        else np.ones(xy.shape[:2], dtype=np.float32)
    )
    boxes = output.boxes.xyxy.cpu().numpy()
    box_conf = output.boxes.conf.cpu().numpy()
    track_ids = (
        output.boxes.id.cpu().numpy().astype(int)
        if output.boxes.id is not None
        else np.full(len(boxes), -1, dtype=int)
    )
    people: list[dict] = []
    for points, scores, box, detection_conf, track_id in zip(
        xy,
        conf,
        boxes,
        box_conf,
        track_ids,
    ):
        landmarks = [PointLandmark() for _ in range(33)]
        for yolo_idx, target_idx in YOLO_TO_LM.items():
            px, py = points[yolo_idx]
            landmarks[target_idx] = PointLandmark(px / w, py / h, scores[yolo_idx])
        person = pose_features(landmarks, landmarks)
        person["box"] = tuple(float(value) for value in box)
        person["detection_conf"] = float(detection_conf)
        person["track_id"] = int(track_id)
        person["center_x"] = float((box[0] + box[2]) * 0.5 / w)
        # Box center survives when hip keypoints are partly occluded.
        if _vis(landmarks, "l_hip", "r_hip") < 0.2:
            person["hip_x"] = person["center_x"]
        people.append(person)
    return people


class PoseSmoother:
    """EMA keypoints independently for each role; hold briefly through dropouts."""

    def __init__(self, alpha: float = 0.45, hold_frames: int = 5) -> None:
        self.alpha = alpha
        self.hold_frames = hold_frames
        self.previous: dict[str, dict] = {}
        self.missing: dict[str, int] = {}

    def update(self, role: str, person: dict | None) -> dict | None:
        if person is None:
            self.missing[role] = self.missing.get(role, 0) + 1
            if self.missing[role] <= self.hold_frames:
                return self.previous.get(role)
            self.previous.pop(role, None)
            return None

        self.missing[role] = 0
        old = self.previous.get(role)
        if old is not None:
            for current_lm, old_lm in zip(person["image"], old["image"]):
                if current_lm.visibility >= 0.15 and old_lm.visibility >= 0.15:
                    current_lm.x = old_lm.x * (1.0 - self.alpha) + current_lm.x * self.alpha
                    current_lm.y = old_lm.y * (1.0 - self.alpha) + current_lm.y * self.alpha
        self.previous[role] = person
        return person

    def reset(self) -> None:
        self.previous.clear()
        self.missing.clear()


class FeedbackSmoother:
    """Rolling medians plus hysteresis to prevent red/green flicker."""

    def __init__(self) -> None:
        self.neck_values: deque[float] = deque(maxlen=7)
        self.trunk_values: deque[float] = deque(maxlen=7)
        self.neck_state: str | None = None
        self.neck_missing = 0
        self.trunk_missing = 0

    def update(
        self,
        neck_deg: float | None,
        trunk_deg: float | None,
    ) -> tuple[float | None, str | None, float | None]:
        if neck_deg is not None:
            self.neck_values.append(neck_deg)
            self.neck_missing = 0
        else:
            self.neck_missing += 1
            if self.neck_missing > 8:
                self.neck_values.clear()
                self.neck_state = None
        if trunk_deg is not None:
            self.trunk_values.append(trunk_deg)
            self.trunk_missing = 0
        else:
            self.trunk_missing += 1
            if self.trunk_missing > 8:
                self.trunk_values.clear()
        neck = (
            float(np.median(self.neck_values))
            if self.neck_values
            else None
        )
        trunk = (
            float(np.median(self.trunk_values))
            if self.trunk_values
            else None
        )
        if neck is None:
            return None, self.neck_state, trunk

        if self.neck_state is None:
            self.neck_state = neck_band(neck)
        elif self.neck_state == "safe":
            if neck > 58.0:
                self.neck_state = "danger"
            elif neck > 38.0:
                self.neck_state = "caution"
        elif self.neck_state == "caution":
            if neck < 32.0:
                self.neck_state = "safe"
            elif neck > 58.0:
                self.neck_state = "danger"
        elif self.neck_state == "danger" and neck < 50.0:
            self.neck_state = "safe" if neck < 32.0 else "caution"
        return neck, self.neck_state, trunk

    def reset(self) -> None:
        self.neck_values.clear()
        self.trunk_values.clear()
        self.neck_state = None
        self.neck_missing = 0
        self.trunk_missing = 0


def _floor_score(person: dict) -> tuple[float, float]:
    """Higher means more like the person on the floor: lower hips, wider body."""
    box = person.get("box") or (0.0, 0.0, 1.0, 1.0)
    width = max(1.0, float(box[2]) - float(box[0]))
    height = max(1.0, float(box[3]) - float(box[1]))
    return (float(person.get("hip_y", 0.5)), width / height)


class RoleTracker:
    """Assign detections to stable roles.

    Live camera keeps the demo layout: carer starts on the left.
    A floor-transfer video uses body height instead, because the patient
    is the person lying or sitting lower in the frame.
    """

    def __init__(self, mode: str = "sides") -> None:
        self.mode = mode
        self.carer_on_left = True
        self.carer_x: float | None = None
        self.patient_x: float | None = None
        self.carer_y: float | None = None
        self.patient_y: float | None = None
        self.carer_id: int | None = None
        self.patient_id: int | None = None

    def _remember(self, carer: dict | None, patient: dict | None) -> None:
        if carer is not None:
            self.carer_x = carer["center_x"]
            self.carer_y = float(carer.get("hip_y", 0.5))
            self.carer_id = carer.get("track_id")
        if patient is not None:
            self.patient_x = patient["center_x"]
            self.patient_y = float(patient.get("hip_y", 0.5))
            self.patient_id = patient.get("track_id")

    def _match_remaining(self, remaining: list[dict], role: str) -> dict:
        if self.mode == "floor":
            target = self.patient_y if role == "patient" else self.carer_y
            if target is None:
                target = 0.7 if role == "patient" else 0.4
            return min(remaining, key=lambda person: abs(float(person.get("hip_y", 0.5)) - target))
        target = self.patient_x if role == "patient" else self.carer_x
        if target is None:
            target = 0.75 if role == "patient" else 0.25
        return min(remaining, key=lambda person: abs(person["center_x"] - target))

    def split(self, people: list[dict]) -> tuple[dict | None, dict | None]:
        if not people:
            return None, None
        by_id = {
            person["track_id"]: person
            for person in people
            if person.get("track_id", -1) >= 0
        }
        tracked_carer = by_id.get(self.carer_id) if self.carer_id is not None else None
        tracked_patient = by_id.get(self.patient_id) if self.patient_id is not None else None
        if tracked_carer is not None or tracked_patient is not None:
            used_ids = {
                person["track_id"]
                for person in (tracked_carer, tracked_patient)
                if person is not None
            }
            remaining = [
                person for person in people if person.get("track_id") not in used_ids
            ]
            if tracked_carer is None and remaining:
                tracked_carer = self._match_remaining(remaining, "carer")
                remaining.remove(tracked_carer)
            if tracked_patient is None and remaining:
                tracked_patient = self._match_remaining(remaining, "patient")
            self._remember(tracked_carer, tracked_patient)
            return tracked_carer, tracked_patient
        if len(people) == 1:
            person = people[0]
            # Do not lock a role from one body during a floor transfer.
            # The lying patient is often the only clear detection at first.
            if self.mode == "floor" and self.carer_id is None and self.patient_id is None:
                hip_y, aspect = _floor_score(person)
                if hip_y >= 0.55 or aspect >= 0.72:
                    return None, person
                return person, None
            x = person["center_x"]
            if self.patient_x is not None and self.carer_x is not None:
                if abs(x - self.patient_x) < abs(x - self.carer_x):
                    self._remember(None, person)
                    return None, person
            self._remember(person, None)
            return person, None

        candidates = sorted(
            people,
            key=lambda person: person["detection_conf"],
            reverse=True,
        )[:3]
        if self.carer_x is None or self.patient_x is None:
            if self.mode == "floor":
                ordered = sorted(candidates, key=_floor_score, reverse=True)
                patient, carer = ordered[0], ordered[1]
            else:
                ordered = sorted(candidates, key=lambda person: person["center_x"])
                carer, patient = (
                    (ordered[0], ordered[-1])
                    if self.carer_on_left
                    else (ordered[-1], ordered[0])
                )
        else:
            assignments = [
                (
                    abs(carer["center_x"] - self.carer_x)
                    + abs(patient["center_x"] - self.patient_x),
                    carer,
                    patient,
                )
                for carer in candidates
                for patient in candidates
                if carer is not patient
            ]
            _, carer, patient = min(assignments, key=lambda item: item[0])

        self._remember(carer, patient)
        return carer, patient

    def swap(self) -> None:
        self.carer_on_left = not self.carer_on_left
        self.carer_x, self.patient_x = self.patient_x, self.carer_x
        self.carer_y, self.patient_y = self.patient_y, self.carer_y
        self.carer_id, self.patient_id = self.patient_id, self.carer_id

    def reset(self) -> None:
        self.carer_x = None
        self.patient_x = None
        self.carer_y = None
        self.patient_y = None
        self.carer_id = None
        self.patient_id = None

    def recalibrate(self) -> None:
        """Drop the locked identities and mark the lower person as the patient."""
        self.mode = "floor"
        self.reset()


_FONT_PATHS = {
    "sans": "/System/Library/Fonts/Supplemental/Arial.ttf",
    "bold": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "black": "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "italic": "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
}
_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    key = (kind, size)
    font = _FONT_CACHE.get(key)
    if font is None:
        font = ImageFont.truetype(_FONT_PATHS[kind], size)
        _FONT_CACHE[key] = font
    return font


def _text_size(text: str, size: int, kind: str) -> tuple[int, int, int, int]:
    left, top, right, bottom = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
        (0, 0), text, font=_font(kind, size)
    )
    return left, top, right - left, bottom - top


def _blit_text(
    frame: np.ndarray,
    text: str,
    xy: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
    kind: str = "sans",
) -> None:
    left, top, width, height = _text_size(text, size, kind)
    plate = Image.new("RGBA", (width + 2, height + 2), (0, 0, 0, 0))
    ImageDraw.Draw(plate).text(
        (1 - left, 1 - top),
        text,
        font=_font(kind, size),
        fill=(color[2], color[1], color[0], 255),
    )
    arr = np.asarray(plate)
    alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
    src = arr[:, :, :3][:, :, ::-1]
    x, y = xy
    h, w = arr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1 = min(frame.shape[1], x + w)
    y1 = min(frame.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sx0, sy0 = x0 - x, y0 - y
    crop = frame[y0:y1, x0:x1]
    patch = src[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
    mask = alpha[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
    frame[y0:y1, x0:x1] = (crop * (1.0 - mask) + patch * mask).astype(np.uint8)


def _round(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    radius: int,
    color: tuple[int, int, int] | int,
) -> None:
    x, y, w, h = int(x), int(y), int(w), int(h)
    radius = int(max(1, min(radius, w // 2, h // 2)))
    cv2.rectangle(img, (x + radius, y), (x + w - radius, y + h), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y + radius), (x + w, y + h - radius), color, -1, cv2.LINE_AA)
    cv2.circle(img, (x + radius, y + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x + w - radius, y + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x + radius, y + h - radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x + w - radius, y + h - radius), radius, color, -1, cv2.LINE_AA)


def _pill(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    bg: tuple[int, int, int],
    fg: tuple[int, int, int],
    size: int = 18,
    kind: str = "bold",
    pad_x: int = 16,
    pad_y: int = 8,
) -> tuple[int, int]:
    _left, _top, width, height = _text_size(text, size, kind)
    box_w = width + pad_x * 2
    box_h = height + pad_y * 2
    if bg == WHITE:
        _round(frame, x, y, box_w, box_h, box_h // 2, HAIRLINE)
        _round(frame, x + 2, y + 2, box_w - 4, box_h - 4, max(1, (box_h - 4) // 2), bg)
    else:
        _round(frame, x, y, box_w, box_h, box_h // 2, bg)
    _blit_text(frame, text, (x + pad_x, y + pad_y), size, fg, kind)
    return box_w, box_h


def draw_skeleton(
    frame: np.ndarray,
    landmarks,
    w: int,
    h: int,
    color: tuple[int, int, int] = CARER_COLOR,
    label: str = "",
) -> None:
    pts = []
    for lm in landmarks:
        pts.append((int(lm.x * w), int(lm.y * h)))
    for a, b in POSE_CONNECTIONS:
        if _lm_vis(landmarks[a]) >= 0.4 and _lm_vis(landmarks[b]) >= 0.4:
            cv2.line(frame, pts[a], pts[b], WHITE, 7, cv2.LINE_AA)
            cv2.line(frame, pts[a], pts[b], color, 3, cv2.LINE_AA)
    for idx in LM.values():
        if _lm_vis(landmarks[idx]) >= 0.4:
            cv2.circle(frame, pts[idx], 7, WHITE, -1, cv2.LINE_AA)
            cv2.circle(frame, pts[idx], 4, color, -1, cv2.LINE_AA)
    if label:
        pretty = {"CARER": "Carer", "PATIENT": "Patient"}.get(label, label.title())
        x, y = _pt(landmarks, "nose", w, h)
        _pill(frame, pretty, x - 36, max(8, y - 46), WHITE, color, size=16, pad_x=12, pad_y=6)


def draw_human_setup(frame: np.ndarray, carer_on_left: bool) -> None:
    """Show the two independent detector zones used in HUMAN mode."""
    h, w = frame.shape[:2]
    left_label = "CARER ZONE" if carer_on_left else "PATIENT ZONE"
    right_label = "PATIENT ZONE" if carer_on_left else "CARER ZONE"
    left_color = CARER_COLOR if carer_on_left else PATIENT_COLOR
    right_color = PATIENT_COLOR if carer_on_left else CARER_COLOR
    cv2.line(frame, (int(w * 0.36), 185), (int(w * 0.36), h - 45), left_color, 2)
    cv2.line(frame, (int(w * 0.64), 185), (int(w * 0.64), h - 45), right_color, 2)
    _put(frame, left_label, (20, h - 58), 0.55, left_color, 1)
    _put(frame, right_label, (int(w * 0.67), h - 58), 0.55, right_color, 1)


def draw_neck_measurement(
    frame: np.ndarray,
    landmarks,
    neck_deg: float | None,
    status: str | None = None,
) -> None:
    if neck_deg is None:
        return
    h, w = frame.shape[:2]

    def mean_point(names: tuple[str, ...], threshold: float) -> tuple[int, int] | None:
        points = [
            _pt(landmarks, name, w, h)
            for name in names
            if _lm_vis(landmarks[LM[name]]) >= threshold
        ]
        if not points:
            return None
        return (
            sum(point[0] for point in points) // len(points),
            sum(point[1] for point in points) // len(points),
        )

    shoulder = mean_point(("l_sh", "r_sh"), 0.15)
    hip = mean_point(("l_hip", "r_hip"), 0.15)
    if shoulder is None or hip is None:
        return
    visible_ears = [
        _pt(landmarks, name, w, h)
        for name in ("l_ear", "r_ear")
        if _lm_vis(landmarks[LM[name]]) >= 0.12
    ]
    head = (
        (
            sum(point[0] for point in visible_ears) // len(visible_ears),
            sum(point[1] for point in visible_ears) // len(visible_ears),
        )
        if visible_ears
        else _pt(landmarks, "nose", w, h)
    )
    status = status or neck_band(neck_deg)
    color = GREEN if status == "safe" else AMBER if status == "caution" else RED
    cv2.line(frame, hip, shoulder, WHITE, 8, cv2.LINE_AA)
    cv2.line(frame, shoulder, head, WHITE, 8, cv2.LINE_AA)
    cv2.line(frame, hip, shoulder, color, 3, cv2.LINE_AA)
    cv2.line(frame, shoulder, head, color, 3, cv2.LINE_AA)
    _pill(frame, f"{neck_deg:.0f}°", head[0] + 10, head[1] - 14, WHITE, color, size=15, pad_x=10, pad_y=5)


def draw_trunk_measurement(
    frame: np.ndarray,
    landmarks,
    trunk_deg: float | None,
) -> None:
    if trunk_deg is None:
        return
    h, w = frame.shape[:2]
    shoulders = [
        _pt(landmarks, name, w, h)
        for name in ("l_sh", "r_sh")
        if _lm_vis(landmarks[LM[name]]) >= 0.15
    ]
    hips = [
        _pt(landmarks, name, w, h)
        for name in ("l_hip", "r_hip")
        if _lm_vis(landmarks[LM[name]]) >= 0.15
    ]
    if not shoulders or not hips:
        return
    shoulder = (
        sum(point[0] for point in shoulders) // 2,
        sum(point[1] for point in shoulders) // 2,
    )
    hip = (
        sum(point[0] for point in hips) // 2,
        sum(point[1] for point in hips) // 2,
    )
    color = GREEN if trunk_deg <= 20.0 else AMBER if trunk_deg <= 60.0 else RED
    cv2.line(frame, hip, shoulder, WHITE, 9, cv2.LINE_AA)
    cv2.line(frame, hip, shoulder, color, 3, cv2.LINE_AA)
    _pill(frame, f"Trunk {trunk_deg:.0f}°", hip[0] + 8, hip[1] - 12, WHITE, INK, size=15, pad_x=10, pad_y=5)


def _put(frame, text, org, scale, color, thick=2) -> None:
    size = max(12, int(32 * scale))
    _blit_text(frame, text, org, size, color, "bold" if thick >= 2 else "sans")


def lift_mode(people_count: int) -> str:
    if people_count >= 2:
        return "human"
    if people_count == 1:
        return "box"
    return "wait"


def _status_copy(
    mode: str,
    neck_status: str | None,
    people_count: int,
) -> tuple[str, str, tuple[int, int, int], tuple[int, int, int]]:
    """Short label, detail line, pill background, pill text. Matches the site pills."""
    if mode == "box":
        return "Box lift", "No neck to protect", WHITE, INK
    if mode == "human" and neck_status == "danger":
        return "Head drop", "A box cannot fail this", INK, WHITE
    if mode == "human" and neck_status == "safe":
        return "Head up", "Neck stayed with the torso", BLUE, WHITE
    if mode == "human" and neck_status == "caution":
        return "Check head", "Support the head more", WHITE, INK
    if mode == "human" and people_count < 2:
        return "Not visible", "Keep both people in frame", WHITE, INK
    if mode == "human":
        return "Finding patient", "Expose the head and shoulders", WHITE, INK
    return "Watching", "Press 1 for a box, 2 for a person", WHITE, INK


def _row(frame: np.ndarray, label: str, value: str, x: int, y: int, width: int) -> None:
    _blit_text(frame, label, (x, y), 18, MUTED, "sans")
    _left, _top, text_w, _height = _text_size(value, 18, "bold")
    _blit_text(frame, value, (x + width - text_w, y), 18, INK, "bold")


def present(
    video: np.ndarray,
    result: RebaResult | None,
    peak: int,
    subtitle: str = "",
    patient_head_ok_flag: bool | None = None,
    patient_neck_deg: float | None = None,
    patient_neck_status: str | None = None,
    carer_trunk_deg: float | None = None,
    people_count: int = 0,
    demo_mode: str | None = None,
) -> np.ndarray:
    """Light Careier stage: rounded video on the left, score card on the right."""
    canvas = np.full((STAGE_H, STAGE_W, 3), PAPER, np.uint8)
    margin, gap, card_w = 22, 18, 430
    well_x = well_y = margin
    well_w = STAGE_W - margin * 2 - gap - card_w
    well_h = STAGE_H - margin * 2
    _round(canvas, well_x, well_y, well_w, well_h, 28, (226, 226, 224))

    vh, vw = video.shape[:2]
    scale = min((well_w - 20) / vw, (well_h - 20) / vh)
    nw = max(1, int(vw * scale))
    nh = max(1, int(vh * scale))
    resized = cv2.resize(video, (nw, nh), interpolation=cv2.INTER_AREA)
    ox = well_x + (well_w - nw) // 2
    oy = well_y + (well_h - nh) // 2
    mask = np.zeros((nh, nw), np.uint8)
    _round(mask, 0, 0, nw, nh, 22, 255)
    alpha = mask[:, :, None].astype(np.float32) / 255.0
    roi = canvas[oy : oy + nh, ox : ox + nw]
    canvas[oy : oy + nh, ox : ox + nw] = (resized * alpha + roi * (1.0 - alpha)).astype(np.uint8)

    mode = demo_mode or lift_mode(people_count)
    neck_status = patient_neck_status
    if neck_status is None and patient_head_ok_flag is not None:
        neck_status = "safe" if patient_head_ok_flag else "danger"
    status, detail, pill_bg, pill_fg = _status_copy(mode, neck_status, people_count)

    _pill(canvas, "careier.", ox + 16, oy + 16, WHITE, INK, size=18, kind="bold", pad_x=14, pad_y=7)
    _pill(canvas, status, ox + 16, oy + 58, pill_bg, pill_fg, size=16, kind="bold", pad_x=14, pad_y=7)

    card_x = well_x + well_w + gap
    _round(canvas, card_x, well_y, card_w, well_h, 28, CARD)
    x = card_x + 32
    y = well_y + 36
    inner = card_w - 64
    _blit_text(canvas, "careier.", (x, y), 40, INK, "black")
    y += 58
    sensing = "—   Now sensing   —"
    _left, _top, sensing_w, _sensing_h = _text_size(sensing, 20, "italic")
    _blit_text(canvas, sensing, (x + (inner - sensing_w) // 2, y), 20, MUTED, "italic")
    y += 52
    _blit_text(canvas, "Carer", (x, y), 16, MUTED, "sans")
    y += 28
    score = "–" if result is None else str(result.score)
    _blit_text(canvas, score, (x, y), 78, INK, "black")
    y += 96
    band = result.band.title() if result else "Waiting"
    band_bg, band_fg = (INK, WHITE) if result and result.band in ("HIGH", "VERY HIGH") else (WHITE, INK)
    _pill(canvas, band, x, y, band_bg, band_fg, size=16, pad_x=18, pad_y=8)
    y += 64
    _blit_text(canvas, "Patient", (x, y), 16, MUTED, "sans")
    y += 28
    _pill(canvas, status, x, y, pill_bg, pill_fg, size=16, pad_x=16, pad_y=8)
    y += 46
    _blit_text(canvas, detail, (x, y), 16, MUTED, "sans")
    y += 42
    cv2.line(canvas, (x, y), (x + inner, y), HAIRLINE, 1, cv2.LINE_AA)
    y += 18
    neck_text = "—" if patient_neck_deg is None else f"{patient_neck_deg:.0f}°"
    trunk_text = "—" if carer_trunk_deg is None else f"{carer_trunk_deg:.0f}°"
    _row(canvas, "Neck", neck_text, x, y, inner)
    y += 32
    _row(canvas, "Trunk", trunk_text, x, y, inner)
    y += 32
    _row(canvas, "Peak", str(peak), x, y, inner)
    if subtitle:
        y += 40
        _blit_text(canvas, subtitle, (x, y), 15, MUTED, "sans")
    _blit_text(
        canvas,
        "C  lower is patient     S swap     Q quit",
        (x, well_y + well_h - 58),
        14,
        MUTED,
        "sans",
    )
    _blit_text(canvas, "Camera trial. The vest is the product.", (x, well_y + well_h - 34), 13, MUTED, "sans")
    return canvas


class _FakeLm:
    def __init__(self, x: float, y: float, vis: float = 1.0) -> None:
        self.x = x
        self.y = y
        self.z = 0.0
        self.visibility = vis


def _demo_landmarks(kind: str, t: float) -> list[_FakeLm]:
    """Schematic side-on person so --demo works without a webcam."""
    bounce = 0.01 * np.sin(t * 6.0)
    if kind == "stoop":
        hip = (0.46, 0.62)
        sh = (0.62 + bounce, 0.38)
        nose = (0.70, 0.28)
        l_kn, r_kn = (0.44, 0.80), (0.50, 0.79)
        l_an, r_an = (0.43, 0.95), (0.51, 0.95)
        l_el, r_el = (0.74, 0.50), (0.78, 0.46)
        l_wr, r_wr = (0.80, 0.62), (0.84, 0.58)
    elif kind == "squat":
        hip = (0.50, 0.68)
        sh = (0.52, 0.36 + bounce)
        nose = (0.53, 0.24)
        l_kn, r_kn = (0.42, 0.78), (0.58, 0.78)
        l_an, r_an = (0.40, 0.95), (0.60, 0.95)
        l_el, r_el = (0.50, 0.50), (0.56, 0.50)
        l_wr, r_wr = (0.48, 0.62), (0.54, 0.62)
    else:
        hip = (0.50, 0.58)
        sh = (0.51, 0.32)
        nose = (0.52, 0.20)
        l_kn, r_kn = (0.47, 0.76), (0.54, 0.76)
        l_an, r_an = (0.46, 0.95), (0.55, 0.95)
        l_el, r_el = (0.46, 0.46), (0.57, 0.46)
        l_wr, r_wr = (0.45, 0.60), (0.58, 0.60)

    l_sh = (sh[0] - 0.04, sh[1])
    r_sh = (sh[0] + 0.05, sh[1] - 0.01 if kind == "stoop" else sh[1])
    l_hip = (hip[0] - 0.03, hip[1])
    r_hip = (hip[0] + 0.03, hip[1])

    pts = {i: (0.5, 0.5) for i in range(33)}
    pts[0] = nose
    pts[11], pts[12] = l_sh, r_sh
    pts[13], pts[14] = l_el, r_el
    pts[15], pts[16] = l_wr, r_wr
    pts[23], pts[24] = l_hip, r_hip
    pts[25], pts[26] = l_kn, r_kn
    pts[27], pts[28] = l_an, r_an
    return [_FakeLm(x, y) for x, y in (pts[i] for i in range(33))]


def _demo_patient(kind: str, t: float) -> list[_FakeLm]:
    bounce = 0.008 * np.sin(t * 5.0)
    if kind == "drop":
        hip, sh, nose = (0.80, 0.58), (0.70, 0.46 + bounce), (0.78, 0.64)
        l_kn, r_kn = (0.88, 0.62), (0.90, 0.60)
        l_an, r_an = (0.94, 0.66), (0.95, 0.64)
        l_el, r_el = (0.66, 0.52), (0.64, 0.50)
        l_wr, r_wr = (0.62, 0.58), (0.60, 0.56)
    else:
        hip, sh, nose = (0.80, 0.56), (0.70, 0.44 + bounce), (0.66, 0.32)
        l_kn, r_kn = (0.88, 0.60), (0.90, 0.58)
        l_an, r_an = (0.94, 0.64), (0.95, 0.62)
        l_el, r_el = (0.66, 0.48), (0.64, 0.46)
        l_wr, r_wr = (0.62, 0.54), (0.60, 0.52)
    l_sh, r_sh = (sh[0] - 0.03, sh[1]), (sh[0] + 0.03, sh[1])
    l_hip, r_hip = (hip[0] - 0.03, hip[1]), (hip[0] + 0.03, hip[1])
    pts = {i: (0.75, 0.55) for i in range(33)}
    pts[0] = nose
    pts[11], pts[12] = l_sh, r_sh
    pts[13], pts[14] = l_el, r_el
    pts[15], pts[16] = l_wr, r_wr
    pts[23], pts[24] = l_hip, r_hip
    pts[25], pts[26] = l_kn, r_kn
    pts[27], pts[28] = l_an, r_an
    return [_FakeLm(x, y) for x, y in (pts[i] for i in range(33))]


def demo_phase(elapsed: float) -> tuple[str, RebaAngles, int, str, int, bool | None, str]:
    """Three-beat loop: box, human head-drop, human head-up."""
    cycle = elapsed % 21.0
    if cycle < 6.5:
        return (
            "idle",
            replace(SQUAT_CLOSE, trunk_flex_deg=14, knee_flex_deg=36, upper_arm_deg=28),
            0,
            "TAKE 1  —  box / bag",
            1,
            None,
            "box",
        )
    if cycle < 13.5:
        return "stoop", STOOP_TWIST, 1, "TAKE 2  —  human, head hanging", 2, False, "drop"
    return "squat", SQUAT_CLOSE, 0, "TAKE 3  —  human, head up", 2, True, "up"


def run_demo(record_path: str | None, hub: HubBridge | None = None) -> None:
    hub = hub or HubBridge(None)
    w, h = 900, 720
    output_w, output_h = STAGE_W, STAGE_H
    writer = None
    if record_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(record_path, fourcc, 30.0, (output_w, output_h))
        if not writer.isOpened():
            raise SystemExit(f"Could not open video writer for {record_path}")

    peak = 1
    last_phase = ""
    fps = 30.0
    frame_i = 0
    show_ok = True
    t0 = time.time()
    if writer is None:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_TITLE, STAGE_W, STAGE_H)
    while True:
        elapsed = frame_i / fps if writer is not None else (time.time() - t0)
        kind, angles, activity, subtitle, people_count, head_ok, extra = demo_phase(elapsed)
        if kind != last_phase:
            peak = 1
        last_phase = kind

        result = score_reba(angles, activity=activity)
        peak = max(peak, result.score)

        frame = np.full((h, w, 3), 36, dtype=np.uint8)
        cv2.rectangle(frame, (0, int(h * 0.92)), (w, h), (50, 50, 50), -1)
        draw_skeleton(frame, _demo_landmarks(kind, elapsed), w, h, CARER_COLOR, "CARER")
        if extra == "box":
            cv2.rectangle(frame, (int(w * 0.62), int(h * 0.58)), (int(w * 0.78), int(h * 0.78)), AMBER, 3)
            _put(frame, "BOX", (int(w * 0.64), int(h * 0.56)), 0.8, AMBER, 2)
        elif extra in ("drop", "up"):
            draw_skeleton(frame, _demo_patient(extra, elapsed), w, h, PATIENT_COLOR, "PATIENT")
        frame = present(
            frame,
            result,
            peak,
            subtitle,
            patient_head_ok_flag=head_ok,
            patient_neck_deg=(
                68.0 if extra == "drop" else 14.0 if extra == "up" else None
            ),
            patient_neck_status=(
                "danger" if extra == "drop" else "safe" if extra == "up" else None
            ),
            carer_trunk_deg=angles.trunk_flex_deg,
            people_count=people_count,
            demo_mode="box" if extra == "box" else "human",
        )
        hub.send(
            mode="box" if extra == "box" else "human",
            carer_trunk_deg=angles.trunk_flex_deg,
            reba_score=result.score,
            reba_band=result.band,
            patient_neck_deg=68.0 if extra == "drop" else 14.0 if extra == "up" else None,
            patient_neck_status="danger" if extra == "drop" else "safe" if extra == "up" else None,
            patient_head_ok=head_ok,
            people=people_count,
        )

        if writer is not None:
            writer.write(frame)
            if elapsed >= 21.0:
                break

        if writer is None:
            try:
                cv2.imshow(WINDOW_TITLE, frame)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                show_ok = False
                key = 0
                raise SystemExit("No display. Re-run with --demo --record demo.mp4")
        else:
            key = 0
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            peak = 1
        frame_i += 1

    if writer is not None:
        writer.release()
        print(f"Wrote {record_path}")
    if show_ok:
        cv2.destroyAllWindows()


def macos_camera_names() -> list[str]:
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPCameraDataType"],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    names: list[str] = []
    for line in out.splitlines():
        if line.startswith("    ") and not line.startswith("        ") and line.rstrip().endswith(":"):
            names.append(line.strip().rstrip(":"))
    return names


def discover_cameras(max_index: int = 6) -> list[tuple[int, str]]:
    names = macos_camera_names()
    found: list[tuple[int, str]] = []
    backend = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    misses = 0
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        ok = cap.isOpened()
        cap.release()
        if not ok:
            misses += 1
            if found and misses >= 1:
                break
            continue
        misses = 0
        label = names[i] if i < len(names) else f"Camera {i}"
        found.append((i, label))
    if not found and names:
        return [(i, name) for i, name in enumerate(names)]
    return found


def print_cameras(cameras: list[tuple[int, str]]) -> None:
    if not cameras:
        print("No cameras found.")
        return
    print("Cameras:")
    for index, name in cameras:
        tag = "  ← iPhone / Continuity" if _is_iphone_name(name) else ""
        print(f"  [{index}] {name}{tag}")


def _is_iphone_name(name: str) -> bool:
    lower = name.lower()
    return "iphone" in lower or "continuity" in lower


def pick_iphone(cameras: list[tuple[int, str]]) -> int:
    for index, name in cameras:
        if _is_iphone_name(name):
            print(f"Using {name} (camera {index})")
            return index
    print_cameras(cameras)
    raise SystemExit(
        "No iPhone camera found. Unlock the phone, keep it nearby, "
        "and enable Continuity Camera (same Apple ID). Then run: python coach.py --iphone"
    )


def pick_camera(cameras: list[tuple[int, str]], default: int = 0) -> int:
    print_cameras(cameras)
    if not cameras:
        return default
    if len(cameras) == 1:
        print(f"Using {cameras[0][1]}")
        return cameras[0][0]
    raw = input("Camera number (iPhone is marked): ").strip()
    if raw == "":
        return cameras[0][0]
    try:
        index = int(raw)
    except ValueError:
        raise SystemExit("Camera must be a number, e.g. 1")
    if index not in {i for i, _ in cameras}:
        raise SystemExit(f"Camera {index} is not in the list above.")
    return index


def try_open_camera(camera: int, wait_s: float = 2.5) -> cv2.VideoCapture | None:
    """Open a camera without killing the app. Returns None on failure."""
    backends = [getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY), cv2.CAP_ANY]
    deadline = time.time() + wait_s
    while time.time() < deadline:
        for backend in backends:
            cap = cv2.VideoCapture(camera, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap
            cap.release()
        time.sleep(0.25)
    return None


def open_camera(camera: int) -> cv2.VideoCapture:
    """First open can wait for the macOS permission dialog."""
    cap = try_open_camera(camera, wait_s=12.0)
    if cap is None:
        raise SystemExit(
            f"Could not open camera {camera}. On macOS: System Settings → Privacy & Security → "
            "Camera → enable Terminal (or Cursor). Or run: python coach.py --demo"
        )
    return cap


def switch_camera(
    cap: cv2.VideoCapture | None,
    current: int,
    nxt: int,
) -> tuple[cv2.VideoCapture | None, int, str]:
    """Cycle cameras. Never exits. Falls back to the previous device if the next fails."""
    if cap is not None:
        cap.release()
        time.sleep(0.35)
    new = try_open_camera(nxt, wait_s=2.0)
    if new is not None:
        print(f"Switched to camera {nxt}")
        return new, nxt, ""
    print(f"Camera {nxt} failed; returning to camera {current}")
    old = try_open_camera(current, wait_s=3.0)
    if old is not None:
        return old, current, f"CAMERA {nxt} FAILED  —  stayed on {current}"
    return None, current, "CAMERA LOST  —  press N to retry"


def reset_tracks(model: YOLO) -> None:
    """Drop ByteTrack history so a seek or loop does not glue old identities on."""
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return
    predictor.trackers = []


def fit_display(frame: np.ndarray, max_w: int = 1100, max_h: int = 980) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(max_w / width, max_h / height, 1.0)
    if scale >= 0.99:
        return frame
    return cv2.resize(
        frame,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def annotate_lift(
    frame: np.ndarray,
    pose_model: YOLO,
    roles: RoleTracker,
    smoother: PoseSmoother,
    filt: AngleFilter,
    feedback: FeedbackSmoother,
    demo_mode: str,
    peak: int,
    subtitle: str,
    hub: HubBridge | None = None,
) -> tuple[np.ndarray, int]:
    """Draw both people and the same scores used on the live camera."""
    height, width = frame.shape[:2]
    people = yolo_people(frame, pose_model)
    carer, patient = roles.split(people)
    if demo_mode == "box":
        patient = None
    carer = smoother.update("carer", carer)
    patient = smoother.update("patient", patient)
    people_count = int(carer is not None) + int(patient is not None)
    result = None
    neck_deg = None
    trunk_deg = None
    if carer is not None:
        draw_skeleton(frame, carer["image"], width, height, CARER_COLOR, "CARER")
        incoming = carer["angles"]
        now = time.time()
        angles = filt.update(incoming, incoming is not None, now)
        if angles is not None:
            result = score_reba(angles, activity=filt.activity(now))
            peak = max(peak, result.score)
            trunk_deg = angles.trunk_flex_deg
    if patient is not None:
        draw_skeleton(frame, patient["image"], width, height, PATIENT_COLOR, "PATIENT")
        neck_deg = patient_neck_angle(patient["image"], width / max(height, 1))

    neck_deg, neck_status, trunk_deg = feedback.update(neck_deg, trunk_deg)
    patient_flag = patient_head_ok(neck_deg)
    if patient is not None:
        draw_neck_measurement(frame, patient["image"], neck_deg, neck_status)
    if carer is not None:
        draw_trunk_measurement(frame, carer["image"], trunk_deg)

    frame = present(
        frame,
        result,
        peak,
        subtitle=subtitle,
        patient_head_ok_flag=patient_flag,
        patient_neck_deg=neck_deg,
        patient_neck_status=neck_status,
        carer_trunk_deg=trunk_deg,
        people_count=people_count,
        demo_mode=demo_mode,
    )
    if hub is not None:
        hub.send(
            mode=demo_mode,
            carer_trunk_deg=trunk_deg,
            reba_score=result.score if result is not None else None,
            reba_band=result.band if result is not None else None,
            patient_neck_deg=neck_deg,
            patient_neck_status=neck_status,
            patient_head_ok=patient_flag,
            people=people_count,
        )
    return frame, peak


def run_video(path: str, record_path: str | None, hub: HubBridge | None = None) -> None:
    video_path = Path(path)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    roles = RoleTracker(mode="floor")
    demo_mode = "human"
    fullscreen = False
    paused = False
    filt = AngleFilter()
    smoother = PoseSmoother()
    feedback = FeedbackSmoother()
    peak = 1
    pose_model = load_yolo_pose()
    writer = None
    shown: np.ndarray | None = None
    print(f"Playing {video_path.name}  ({total} frames @ {fps:.0f} fps)")
    print("Patient is the person lower in the frame. S swaps if that is wrong.")
    print("C recalibrate (lower person = patient)    S swap    Q quit")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_TITLE, STAGE_W, STAGE_H)

    def clear_motion() -> None:
        roles.reset()
        smoother.reset()
        feedback.reset()
        filt.reset()
        reset_tracks(pose_model)

    def seek(delta_frames: int) -> None:
        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        target = current + delta_frames
        if total > 0:
            target = target % total
        target = max(0, target)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        clear_motion()

    try:
        while True:
            if not paused or shown is None:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    clear_motion()
                    continue
                index = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                seconds = max(0, index - 1) / fps
                state = "PAUSED" if paused else "PLAYING"
                shown, peak = annotate_lift(
                    frame,
                    pose_model,
                    roles,
                    smoother,
                    filt,
                    feedback,
                    demo_mode,
                    peak,
                    subtitle=(
                        f"{state}  {seconds:0.1f}s / {total / fps:0.1f}s"
                        if total
                        else f"{state}  {seconds:0.1f}s"
                    ),
                    hub=hub,
                )
                if record_path and writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        record_path,
                        fourcc,
                        fps,
                        (shown.shape[1], shown.shape[0]),
                    )
                if writer is not None:
                    writer.write(shown)

            cv2.imshow(WINDOW_TITLE, shown)
            key = cv2.waitKeyEx(1 if not paused else 30)
            if key in (-1, 255):
                continue
            low = key & 0xFF
            if low in (ord("q"), ord("Q")):
                break
            if key == 27 or low == 27:
                fullscreen = False
                cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_TITLE, STAGE_W, STAGE_H)
            if low in (ord("w"), ord("W")):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WINDOW_TITLE,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
                if not fullscreen:
                    cv2.resizeWindow(WINDOW_TITLE, STAGE_W, STAGE_H)
            if low in (ord("r"), ord("R")):
                peak = 1
            if low == ord("1"):
                demo_mode = "box"
                clear_motion()
            if low == ord("2"):
                demo_mode = "human"
                clear_motion()
            if low in (ord("c"), ord("C")):
                roles.recalibrate()
                smoother.reset()
                feedback.reset()
                filt.reset()
                print("Recalibrated: lower person is the patient")
            if low in (ord("s"), ord("S")):
                roles.swap()
                smoother.reset()
                feedback.reset()
                filt.reset()
            if low == ord(" "):
                paused = not paused
            if low == ord("0"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                clear_motion()
                shown = None
            if low in (ord(","), ord("<")) or key in (63234, 65361):
                seek(-int(fps))
                shown = None
            if low in (ord("."), ord(">")) or key in (63235, 65363):
                seek(int(fps))
                shown = None
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Wrote {record_path}")
        cv2.destroyAllWindows()


def run_camera(camera: int, record_path: str | None, hub: HubBridge | None = None) -> None:
    hub = hub or HubBridge(None)
    cameras = discover_cameras()
    indices = [i for i, _ in cameras] or [camera]
    cap = open_camera(camera)
    current = camera
    roles = RoleTracker()
    camera_note = ""
    demo_mode = "human"
    mirror = False
    fullscreen = False
    print(f"Live camera {current}. HUMAN mode. C sets the lower person as the patient. S swaps.")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_TITLE, STAGE_W, STAGE_H)

    writer = None
    filt = AngleFilter()
    smoother = PoseSmoother()
    feedback = FeedbackSmoother()
    peak = 1
    pose_model = load_yolo_pose()

    try:
        while True:
            if cap is None:
                frame = np.full((720, 1280, 3), 24, dtype=np.uint8)
                _put(frame, camera_note or "CAMERA LOST  —  press N to retry", (40, 80), 1.0, RED, 2)
                result = None
                patient_flag = None
                people_count = 0
            else:
                ok, frame = cap.read()
                if not ok:
                    camera_note = "CAMERA LOST  —  press N to retry"
                    cap.release()
                    cap = None
                    continue
                if mirror:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                result = None
                patient_flag = None
                neck_deg = None
                neck_status = None
                trunk_deg = None
                people = yolo_people(frame, pose_model)
                carer, patient = roles.split(people)
                if demo_mode == "box":
                    patient = None
                carer = smoother.update("carer", carer)
                patient = smoother.update("patient", patient)
                people_count = int(carer is not None) + int(patient is not None)
                if carer is not None:
                    draw_skeleton(frame, carer["image"], w, h, CARER_COLOR, "CARER")
                    incoming = carer["angles"]
                    now = time.time()
                    angles = filt.update(incoming, incoming is not None, now)
                    if angles is not None:
                        result = score_reba(angles, activity=filt.activity(now))
                        peak = max(peak, result.score)
                        trunk_deg = angles.trunk_flex_deg
                if patient is not None:
                    draw_skeleton(frame, patient["image"], w, h, PATIENT_COLOR, "PATIENT")
                    neck_deg = patient_neck_angle(patient["image"], w / max(h, 1))

                neck_deg, neck_status, trunk_deg = feedback.update(neck_deg, trunk_deg)
                patient_flag = patient_head_ok(neck_deg)
                if patient is not None:
                    draw_neck_measurement(
                        frame,
                        patient["image"],
                        neck_deg,
                        neck_status,
                    )
                if carer is not None:
                    draw_trunk_measurement(frame, carer["image"], trunk_deg)

                frame = present(
                    frame,
                    result,
                    peak,
                    subtitle=camera_note or (
                        "Carer LEFT · patient RIGHT"
                        if roles.carer_on_left
                        else "Patient LEFT · carer RIGHT"
                    ),
                    patient_head_ok_flag=patient_flag,
                    patient_neck_deg=neck_deg,
                    patient_neck_status=neck_status,
                    carer_trunk_deg=trunk_deg,
                    people_count=people_count,
                    demo_mode=demo_mode,
                )
                hub.send(
                    mode=demo_mode,
                    carer_trunk_deg=trunk_deg,
                    reba_score=result.score if result is not None else None,
                    reba_band=result.band if result is not None else None,
                    patient_neck_deg=neck_deg,
                    patient_neck_status=neck_status,
                    patient_head_ok=patient_flag,
                    people=people_count,
                )

            if record_path and writer is None and cap is not None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(record_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
            if writer is not None:
                writer.write(frame)

            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key == 27:
                fullscreen = False
                cv2.setWindowProperty(
                    WINDOW_TITLE,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL,
                )
                cv2.resizeWindow(WINDOW_TITLE, STAGE_W, STAGE_H)
            if key in (ord("w"), ord("W")):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WINDOW_TITLE,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
                if not fullscreen:
                    cv2.resizeWindow(WINDOW_TITLE, STAGE_W, STAGE_H)
            if key in (ord("r"), ord("R")):
                peak = 1
            if key in (ord("1"),):
                demo_mode = "box"
                camera_note = "BOX mode"
                roles.reset()
                smoother.reset()
                feedback.reset()
            if key in (ord("2"),):
                demo_mode = "human"
                camera_note = "HUMAN mode"
                roles.reset()
                smoother.reset()
                feedback.reset()
            if key in (ord("c"), ord("C")):
                roles.recalibrate()
                smoother.reset()
                feedback.reset()
                filt.reset()
                camera_note = "Recalibrated: lower person is the patient"
            if key in (ord("s"), ord("S")):
                roles.swap()
                smoother.reset()
                feedback.reset()
                camera_note = "Swapped carer / patient"
            if key in (ord("f"), ord("F")):
                mirror = not mirror
            if key in (ord("n"), ord("N")):
                if not indices:
                    indices = [current]
                nxt = indices[(indices.index(current) + 1) % len(indices)] if current in indices else indices[0]
                cap, current, camera_note = switch_camera(cap, current, nxt)
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
            print(f"Wrote {record_path}")
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live REBA-style lift coach")
    parser.add_argument("--demo", action="store_true", help="Play bad-then-good lifts without a webcam")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (0 = laptop, 1+ often iPhone)")
    parser.add_argument("--iphone", action="store_true", help="Use Continuity Camera / iPhone if it is listed")
    parser.add_argument("--list-cameras", action="store_true", help="Print cameras and exit")
    parser.add_argument("--video", metavar="PATH", help="Score a video file instead of a camera")
    parser.add_argument("--record", metavar="PATH", help="Save mp4 (demo writes one 16s loop)")
    parser.add_argument("--hub", default="127.0.0.1:5005", metavar="HOST:PORT", help="Send angles to the Careier hub over UDP")
    parser.add_argument("--no-hub", action="store_true", help="Don't send angles to the hub")
    args = parser.parse_args()
    hub = HubBridge(None if args.no_hub else args.hub)
    if args.demo:
        run_demo(args.record, hub)
        return
    if args.video:
        run_video(args.video, args.record, hub)
        return
    cameras = discover_cameras()
    if args.list_cameras:
        print_cameras(cameras)
        return
    if args.iphone:
        camera = pick_iphone(cameras)
    elif args.camera is not None:
        camera = args.camera
    else:
        camera = pick_camera(cameras)
    run_camera(camera, args.record, hub)


if __name__ == "__main__":
    main()
