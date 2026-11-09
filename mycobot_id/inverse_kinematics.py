"""Laptop-only IK for the calibrated, fully closed fingertip midpoint."""
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .kinematics import JOINT_LIMITS_DEG, ToolGeometry, forward_kinematics, self_collision


@dataclass(frozen=True)
class IKResult:
    ok: bool
    message: str
    angles_deg: tuple = ()
    position_error_mm: float = float("inf")
    orientation_error_deg: float = float("inf")


def solve_closed_tip(xyz_mm: Sequence[float], yaw_deg: float, calibration: dict,
                     tool: ToolGeometry, seed_deg: Sequence[float]) -> IKResult:
    xyz = np.asarray(xyz_mm, dtype=float)
    seed = np.asarray(seed_deg, dtype=float)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)) or not np.isfinite(yaw_deg):
        return IKResult(False, "좌표와 방향에는 유한한 숫자를 입력하세요.")
    if not calibration.get("enabled"):
        return IKResult(False, "테이블 및 닫힌 손끝 보정 설정이 필요합니다.")
    if xyz[2] < 0:
        return IKResult(False, "테이블 아래(Z<0)는 입력할 수 없습니다.")
    tcp = np.asarray(calibration["closed_tip_midpoint_flange_m"], dtype=float)
    # Explicit convention: flange +Z points down; yaw rotates about table +Z.
    desired_r = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix() @ np.diag([1., -1., -1.])
    target = xyz / 1000.0
    target[2] -= calibration["model_to_table_z_offset_m"]
    limits = np.radians(JOINT_LIMITS_DEG)
    def residual(q):
        t = forward_kinematics(np.degrees(q), tool).transforms[-1]
        p = t[:3,:3] @ tcp + t[:3,3]
        rotation_error = Rotation.from_matrix(desired_r.T @ t[:3,:3]).as_rotvec()
        return np.r_[p - target, rotation_error * .1]
    seed = np.clip(np.radians(seed), limits[:,0], limits[:,1])
    rng = np.random.default_rng(280)
    seeds = [seed,
             np.radians([36.82,-54.84,-66.53,33.48,-1.31,-58.35]),
             np.radians([-55.01,-33.48,-109.86,53.87,1.31,-33.48])]
    seeds += [rng.uniform(limits[:,0], limits[:,1]) for _ in range(9)]
    candidates = []
    for initial in seeds:
        fit = least_squares(residual, np.clip(initial, limits[:,0]+1e-8, limits[:,1]-1e-8),
                            bounds=(limits[:,0],limits[:,1]), max_nfev=180,
                            ftol=1e-9, xtol=1e-9, gtol=1e-9)
        error = residual(fit.x)
        pe = float(np.linalg.norm(error[:3]) * 1000)
        re = float(np.degrees(np.linalg.norm(error[3:]) / .1))
        q = np.degrees(fit.x)
        if pe <= .5 and re <= 1.0 and self_collision(q, tool) is None:
            candidates.append((float(np.linalg.norm(fit.x - seed)), q, pe, re))
    if not candidates:
        return IKResult(False, "입력한 위치·방향에서 충돌 없는 IK 해를 찾지 못했습니다. 좌표 또는 방향을 바꿔주세요.")
    _, q, pe, re = min(candidates, key=lambda item:item[0])
    return IKResult(True, f"IK 성공: 위치 오차 {pe:.3f} mm / 방향 오차 {re:.3f}°", tuple(float(v) for v in q), pe, re)
