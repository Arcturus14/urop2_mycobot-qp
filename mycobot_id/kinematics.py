"""Modified DH kinematics and capsule-based self-collision for myCobot 280 JN."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .joint_limits import JOINT_LIMITS_DEG


# Manufacturer's 280 DH layout and angle offsets, with JN-specific dimensions
# from the official JN URDF. The 1 mm J1-to-J2 axial shift is absorbed into d1.
# Rows are (a[i-1], alpha[i-1], d[i], theta_offset[i]) in metres and radians.
_J1_ORIGIN_SHIFT_M = 0.001
_HALF_PI = math.pi / 2.0
MODIFIED_DH = (
    (0.0, 0.0, 0.15756 - _J1_ORIGIN_SHIFT_M, 0.0),
    (0.0, _HALF_PI, 0.0, -_HALF_PI),
    (-0.1104, 0.0, 0.0, 0.0),
    (-0.0960, 0.0, 0.06462, -_HALF_PI),
    (0.0, _HALF_PI, 0.07318, _HALF_PI),
    (0.0, -_HALF_PI, 0.0456, 0.0),
)

# Conservative intersection of the current JN product page, official JN URDF,
# and the MyCobot 280 API limits.  This avoids accepting a value rejected by one
# of the official interfaces.  Firmware-reported limits may narrow these later.
SPEED_LIMIT = (1, 100)


@dataclass(frozen=True)
class ToolGeometry:
    """Configurable collision/display proxy for the mounted end accessories."""

    include_gripper: bool = False
    include_camera: bool = False
    gripper_mount_m: float = 0.034  # official adaptive-gripper fixed joint
    gripper_length_m: float = 0.095
    gripper_radius_m: float = 0.040
    # STEP/DAE bounding box center after URDF joint + mesh origins, not mesh origin alone.
    camera_offset_m: tuple[float, float, float] = (0.024000001327, 0.000499983175, 0.010004101791)
    camera_size_m: tuple[float, float, float] = (0.082965087891, 0.016, 0.033965091705)
    gripper_size_m: tuple[float, float, float] = (0.094, 0.050, 0.112)
    gripper_center_m: tuple[float, float, float] = (0.0, -0.012, 0.090)
    gripper_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_rpy_deg: tuple[float, float, float] = (89.529964148845, -0.000001247268, 179.999847965620)
    clearance_m: float = 0.005


@dataclass(frozen=True)
class ToolBox:
    name: str
    transform: np.ndarray
    size: np.ndarray

    def corners(self) -> np.ndarray:
        signs = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        return (signs * self.size / 2.0) @ self.transform[:3, :3].T + self.transform[:3, 3]


@dataclass(frozen=True)
class PoseGeometry:
    transforms: tuple[np.ndarray, ...]
    joints: np.ndarray
    gripper: np.ndarray
    camera: np.ndarray
    tool_boxes: tuple[ToolBox, ...] = ()


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    message: str
    sample_index: int | None = None


def _rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def _rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def _rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _origin_matrix(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    transform = np.eye(4)
    roll, pitch, yaw = rpy
    transform[:3, :3] = _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)
    transform[:3, 3] = xyz
    return transform


def _modified_dh_matrix(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Rx(alpha) Tx(a) Rz(theta) Tz(d)."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array((
        (ct, -st, 0.0, a),
        (st * ca, ct * ca, -sa, -d * sa),
        (st * sa, ct * sa, ca, d * ca),
        (0.0, 0.0, 0.0, 1.0),
    ))


def forward_kinematics(
    angles_deg: Sequence[float], tool: ToolGeometry = ToolGeometry()
) -> PoseGeometry:
    if len(angles_deg) != 6:
        raise ValueError("exactly six joint angles are required")
    transform = np.eye(4)
    transforms = [transform.copy()]
    for angle, (a, alpha, d, offset) in zip(angles_deg, MODIFIED_DH):
        transform = transform @ _modified_dh_matrix(a, alpha, d, math.radians(float(angle)) + offset)
        transforms.append(transform.copy())

    # Return the physical J1 frame for link rendering and capsule endpoints.
    transforms[1][2, 3] += _J1_ORIGIN_SHIFT_M
    points = np.array([frame[:3, 3] for frame in transforms])

    flange = transform
    boxes = []
    for enabled, name, center, rpy_deg, size in (
        (tool.include_gripper, "gripper", tool.gripper_center_m, tool.gripper_rpy_deg, tool.gripper_size_m),
        (tool.include_camera, "camera", tool.camera_offset_m, tool.camera_rpy_deg, tool.camera_size_m),
    ):
        if enabled:
            boxes.append(ToolBox(name, flange @ _origin_matrix(center, np.radians(rpy_deg)), np.asarray(size)))
    if tool.include_gripper:
        gripper_start = _point(flange, (0.0, 0.0, tool.gripper_mount_m))
        gripper_end = _point(
            flange, (0.0, 0.0, tool.gripper_mount_m + tool.gripper_length_m)
        )
        gripper = np.vstack((gripper_start, gripper_end))
    else:
        gripper = np.empty((0, 3), dtype=float)
    if tool.include_camera:
        camera_center = _point(flange, tool.camera_offset_m)
        camera_axis = boxes[-1].transform[:3, 0]
        half = tool.camera_size_m[0] / 2.0
        camera = np.vstack(
            (camera_center - camera_axis * half, camera_center + camera_axis * half)
        )
    else:
        camera = np.empty((0, 3), dtype=float)
    return PoseGeometry(
        tuple(transforms), points,
        gripper, camera, tuple(boxes),
    )


def _point(transform: np.ndarray, xyz: Sequence[float]) -> np.ndarray:
    return (transform @ np.array([*xyz, 1.0], dtype=float))[:3]


def validate_angles(angles_deg: Sequence[float]) -> ValidationResult:
    if len(angles_deg) != 6:
        return ValidationResult(False, "관절값은 6개여야 합니다.")
    for index, (value, limits) in enumerate(zip(angles_deg, JOINT_LIMITS_DEG), 1):
        value = float(value)
        if not math.isfinite(value):
            return ValidationResult(False, f"J{index} 값이 유한수가 아닙니다.")
        if not limits[0] <= value <= limits[1]:
            return ValidationResult(
                False, f"J{index}: {value:.1f}°는 제한 {limits[0]:.0f}~{limits[1]:.0f}° 밖입니다."
            )
    return ValidationResult(True, "관절 제한 통과")


def _segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Shortest distance between two finite 3-D segments."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    epsilon = 1e-12
    if a <= epsilon and e <= epsilon:
        return float(np.linalg.norm(p1 - p2))
    if a <= epsilon:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = float(d1 @ r)
        if e <= epsilon:
            t, s = 0.0, min(max(-c / a, 0.0), 1.0)
        else:
            b = float(d1 @ d2)
            denominator = a * e - b * b
            s = min(max((b * f - c * e) / denominator, 0.0), 1.0) if denominator else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


_CAPSULE_RADII_M = (0.040, 0.035, 0.032, 0.030, 0.028, 0.026)


def self_collision_pairs(geometry: PoseGeometry):
    """Use indices 0–5 for links and 6 onward for mounted tool boxes."""
    for box_index, box in enumerate(geometry.tool_boxes, 6):
        # L6 and the camera-side L5 housing share the physical mount.
        for link in range(4 if box.name == "camera" else 5):
            yield link, box_index
    # The two-away links still share the same local joint assembly.
    for first in range(3):
        for second in range(first + 3, 6):
            yield first, second


def self_clearance(geometry: PoseGeometry, tool: ToolGeometry, pair: tuple[int, int]) -> float:
    """Signed clearance beyond the existing capsule or tool safety envelope."""
    first, second = pair
    joints = geometry.joints
    if second >= 6:
        box = geometry.tool_boxes[second - 6]
        minimum = _CAPSULE_RADII_M[first] + tool.clearance_m
        return segment_box_distance(joints[first], joints[first + 1], box) - minimum
    minimum = _CAPSULE_RADII_M[first] + _CAPSULE_RADII_M[second]
    return _segment_distance(joints[first], joints[first + 1], joints[second], joints[second + 1]) - minimum


def self_collision(angles_deg: Sequence[float], tool: ToolGeometry = ToolGeometry()) -> str | None:
    geometry = forward_kinematics(angles_deg, tool)
    for first, second in self_collision_pairs(geometry):
        if self_clearance(geometry, tool, (first, second)) < 0:
            name = geometry.tool_boxes[second - 6].name if second >= 6 else f"L{second + 1}"
            return f"자체 충돌 가능: L{first + 1} ↔ {name}"
    return None


def segment_box_distance(start: np.ndarray, end: np.ndarray, box: ToolBox) -> float:
    """Exact segment-to-OBB distance via piecewise quadratic minimization."""
    rotation = box.transform[:3, :3]
    p = (start - box.transform[:3, 3]) @ rotation
    d = (end - start) @ rotation
    half = box.size / 2.0
    cuts = [0.0, 1.0]
    for axis in range(3):
        if abs(d[axis]) > 1e-12:
            cuts.extend(t for t in ((-half[axis]-p[axis])/d[axis], (half[axis]-p[axis])/d[axis]) if 0 < t < 1)
    cuts = sorted(cuts)
    candidates = list(cuts)
    for lo, hi in zip(cuts, cuts[1:]):
        midpoint = p + d * ((lo + hi) / 2.0)
        active = np.abs(midpoint) > half
        a = d[active]
        b = p[active] - np.sign(midpoint[active]) * half[active]
        denominator = float(a @ a)
        if denominator > 1e-15:
            candidates.append(float(np.clip(-(a @ b) / denominator, lo, hi)))
    return min(float(np.linalg.norm(np.maximum(np.abs(p + t*d) - half, 0.0))) for t in candidates)


def validate_path(
    current_deg: Sequence[float], target_deg: Sequence[float],
    step_deg: float = 2.0, tool: ToolGeometry = ToolGeometry(),
) -> ValidationResult:
    for angles in (current_deg, target_deg):
        result = validate_angles(angles)
        if not result.ok:
            return result
    current = np.asarray(current_deg, dtype=float)
    target = np.asarray(target_deg, dtype=float)
    samples = max(2, int(math.ceil(float(np.max(np.abs(target - current))) / step_deg)) + 1)
    for index, fraction in enumerate(np.linspace(0.0, 1.0, samples)):
        collision = self_collision(current + (target - current) * fraction, tool)
        if collision:
            return ValidationResult(False, f"경로 {fraction * 100:.0f}%에서 {collision}", index)
    return ValidationResult(True, f"관절 보간 경로 검사 통과 ({samples}개 자세)")

