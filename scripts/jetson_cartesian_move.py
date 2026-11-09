"""Safely preview or execute one small Cartesian move on a myCobot 280 JN.

The script keeps the currently measured firmware RPY, asks pymycobot for an
inverse-kinematics solution, validates the endpoint, and defaults to a dry run.
It intentionally does not claim to check the continuous Cartesian path or
environment collisions.
"""
from __future__ import annotations

import argparse
import json
import math
import signal
from pathlib import Path
import sys
from typing import Callable, Sequence


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mycobot_id.joint_limits import JOINT_LIMITS_DEG


CONFIRMATION = "I_UNDERSTAND"


def emit(event: str, **values: object) -> None:
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def finite_six(values: object, name: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise RuntimeError("{} must be a six-element list, got {!r}".format(name, values))
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError("{} contains a non-finite value: {!r}".format(name, values))
    return result


def decode_ik_response(values: object, robot: object) -> tuple[list[float], str]:
    """Decode the firmware IK response returned by pymycobot.

    The vendor MyCobot class installed on the JN (pymycobot 4.0.7) converts
    GET_ANGLES protocol integers with ``_int2angle`` but omits the equivalent
    conversion for SOLVE_INV_KINEMATICS.  Its raw IK response is therefore a
    list of centidegree integers.  A future fixed implementation is expected to
    return floats, so only an all-integer response is decoded here.
    """
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise RuntimeError(
            "solve_inv_kinematics response must contain six values, got {!r}".format(values)
        )
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        # Observed from the JN firmware for an unreachable target.  Treat this
        # repeated protocol value as failure, never as six valid 82.41 deg joints.
        if list(values) == [8241] * 6:
            raise RuntimeError(
                "firmware inverse kinematics found no solution "
                "(raw response [8241, 8241, 8241, 8241, 8241, 8241])"
            )
        converter = getattr(robot, "_int2angle", None)
        if not callable(converter):
            raise RuntimeError(
                "integer IK response requires pymycobot's _int2angle converter"
            )
        decoded = [float(converter(value)) for value in values]
        return finite_six(decoded, "decoded solve_inv_kinematics response"), "centidegree_int"
    return finite_six(values, "solve_inv_kinematics response"), "degree_float"


def angle_error_deg(actual: float, target: float) -> float:
    """Smallest signed Euler-angle difference, in degrees."""
    return (actual - target + 180.0) % 360.0 - 180.0


def validate_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("{} must be finite and positive".format(name))


def validate_joint_vector(
    angles: Sequence[float], *, margin_deg: float, label: str
) -> None:
    if not math.isfinite(margin_deg) or margin_deg < 0.0:
        raise ValueError("joint limit margin must be finite and non-negative")
    for index, (angle, (low, high)) in enumerate(zip(angles, JOINT_LIMITS_DEG), 1):
        safe_low = low + margin_deg
        safe_high = high - margin_deg
        if safe_low > safe_high:
            raise ValueError("joint limit margin is too large")
        if not safe_low <= angle <= safe_high:
            raise RuntimeError(
                "{} J{}={:.2f} deg is outside the safe interval {:.2f}..{:.2f} deg".format(
                    label, index, angle, safe_low, safe_high
                )
            )


def make_plan(
    current_coords: Sequence[float],
    current_angles: Sequence[float],
    ik_angles: Sequence[float],
    target_xyz_mm: Sequence[float],
    *,
    max_cartesian_step_mm: float,
    max_joint_delta_deg: Sequence[float],
    joint_limit_margin_deg: float,
) -> dict[str, object]:
    current_coords = finite_six(current_coords, "get_coords response")
    current_angles = finite_six(current_angles, "get_angles response")
    ik_angles = finite_six(ik_angles, "solve_inv_kinematics response")
    target_xyz = [float(value) for value in target_xyz_mm]
    if len(target_xyz) != 3 or not all(math.isfinite(value) for value in target_xyz):
        raise ValueError("target XYZ must contain three finite millimetre values")
    if len(max_joint_delta_deg) != 6:
        raise ValueError("six joint-delta limits are required")
    delta_limits = [float(value) for value in max_joint_delta_deg]
    if not all(math.isfinite(value) and value > 0.0 for value in delta_limits):
        raise ValueError("joint-delta limits must be finite and positive")
    validate_positive_finite(max_cartesian_step_mm, "max Cartesian step")

    validate_joint_vector(
        current_angles, margin_deg=joint_limit_margin_deg, label="current angle"
    )
    validate_joint_vector(
        ik_angles, margin_deg=joint_limit_margin_deg, label="IK angle"
    )

    xyz_delta = [target - current for target, current in zip(target_xyz, current_coords[:3])]
    cartesian_distance = math.sqrt(sum(value * value for value in xyz_delta))
    if cartesian_distance > max_cartesian_step_mm:
        raise RuntimeError(
            "Cartesian displacement {:.2f} mm exceeds {:.2f} mm".format(
                cartesian_distance, max_cartesian_step_mm
            )
        )

    joint_delta = [target - current for target, current in zip(ik_angles, current_angles)]
    violations = [
        "J{} {:.2f}>{:.2f} deg".format(index, abs(delta), limit)
        for index, (delta, limit) in enumerate(zip(joint_delta, delta_limits), 1)
        if abs(delta) > limit
    ]
    if violations:
        raise RuntimeError("joint displacement limit exceeded: " + ", ".join(violations))

    target_coords = target_xyz + current_coords[3:]
    return {
        "current_coords": current_coords,
        "current_angles_deg": current_angles,
        "target_coords": target_coords,
        "ik_angles_deg": ik_angles,
        "cartesian_distance_mm": cartesian_distance,
        "joint_delta_deg": joint_delta,
        "joint_limit_margin_deg": joint_limit_margin_deg,
        "max_joint_delta_deg": delta_limits,
    }


def connect(port: str, baud: int):
    from pymycobot.mycobot import MyCobot

    return MyCobot(port, baud)


def close_robot(robot: object) -> None:
    serial_port = getattr(robot, "_serial_port", None)
    if serial_port is not None:
        serial_port.close()


def install_signal_handlers() -> None:
    def interrupted(signum, _frame):
        raise KeyboardInterrupt("signal {}".format(signum))

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)


def execute(args: argparse.Namespace, robot_factory: Callable[[str, int], object] = connect) -> dict[str, object]:
    robot = robot_factory(args.port, args.baud)
    motion_started = False
    try:
        current_coords = finite_six(robot.get_coords(), "get_coords response")
        current_angles = finite_six(robot.get_angles(), "get_angles response")
        target_coords = [*args.xyz, *current_coords[3:]]
        raw_ik_angles = robot.solve_inv_kinematics(target_coords, current_angles)
        emit(
            "snapshot",
            current_coords=current_coords,
            current_angles_deg=current_angles,
            target_coords=target_coords,
            raw_ik_response=raw_ik_angles,
        )
        ik_angles, ik_response_format = decode_ik_response(raw_ik_angles, robot)
        plan = make_plan(
            current_coords,
            current_angles,
            ik_angles,
            args.xyz,
            max_cartesian_step_mm=args.max_cartesian_step_mm,
            max_joint_delta_deg=args.max_joint_delta_deg,
            joint_limit_margin_deg=args.joint_limit_margin_deg,
        )
        plan["raw_ik_response"] = raw_ik_angles
        plan["ik_response_format"] = ik_response_format
        emit("plan", method=args.method, dry_run=args.confirm_motion != CONFIRMATION, **plan)

        if args.confirm_motion != CONFIRMATION:
            emit(
                "dry_run_complete",
                message="No motion sent. Re-run with --confirm-motion {} after reviewing the plan.".format(
                    CONFIRMATION
                ),
            )
            return {"status": "dry_run", **plan}

        motion_started = True
        if args.method == "angles":
            result = robot.sync_send_angles(plan["ik_angles_deg"], args.speed, args.timeout)
        else:
            result = robot.sync_send_coords(
                plan["target_coords"], args.speed, args.coord_mode, args.timeout
            )
        if result != 1:
            raise RuntimeError("{} returned {!r}".format(
                "sync_send_angles" if args.method == "angles" else "sync_send_coords", result
            ))

        final_angles = finite_six(robot.get_angles(), "final get_angles response")
        final_coords = finite_six(robot.get_coords(), "final get_coords response")
        validate_joint_vector(
            final_angles, margin_deg=args.joint_limit_margin_deg, label="final angle"
        )
        joint_error = [
            actual - target for actual, target in zip(final_angles, plan["ik_angles_deg"])
        ]
        xyz_error = [actual - target for actual, target in zip(final_coords[:3], plan["target_coords"][:3])]
        rpy_error = [
            angle_error_deg(actual, target)
            for actual, target in zip(final_coords[3:], plan["target_coords"][3:])
        ]
        position_error = math.sqrt(sum(value * value for value in xyz_error))
        max_orientation_error = max(abs(value) for value in rpy_error)
        max_joint_error = max(abs(value) for value in joint_error)
        reached = (
            position_error <= args.position_tolerance_mm
            and max_orientation_error <= args.orientation_tolerance_deg
            and (args.method != "angles" or max_joint_error <= args.joint_tolerance_deg)
        )
        result_data = {
            "status": "reached" if reached else "final_tolerance_failed",
            "method": args.method,
            "api_return": result,
            "final_coords": final_coords,
            "final_angles_deg": final_angles,
            "xyz_error_mm": xyz_error,
            "position_error_mm": position_error,
            "rpy_error_deg": rpy_error,
            "max_orientation_error_deg": max_orientation_error,
            "joint_error_from_ik_deg": joint_error,
            "max_joint_error_from_ik_deg": max_joint_error,
        }
        emit("result", **result_data)
        if not reached:
            raise RuntimeError("final pose did not satisfy the configured tolerances")
        return result_data
    except BaseException:
        if motion_started:
            try:
                robot.stop()
                emit("stop_sent", reason="motion exception or interruption")
            except Exception as stop_error:
                emit("stop_failed", error=str(stop_error))
        raise
    finally:
        close_robot(robot)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=1000000)
    parser.add_argument("--xyz", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--method", choices=("angles", "coords"), default="angles")
    parser.add_argument(
        "--coord-mode", type=int, choices=(0, 1), default=0,
        help="sync_send_coords mode: 0=angular, 1=linear (only with --method coords)",
    )
    parser.add_argument("--speed", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-cartesian-step-mm", type=float, default=30.0)
    parser.add_argument(
        "--max-joint-delta-deg", type=float, nargs=6,
        default=[15.0] * 6, metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
    )
    parser.add_argument("--joint-limit-margin-deg", type=float, default=2.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--joint-tolerance-deg", type=float, default=2.0)
    parser.add_argument(
        "--confirm-motion", default="",
        help="motion is disabled unless this is exactly {}".format(CONFIRMATION),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 1 <= args.speed <= 100:
        parser.error("--speed must be in 1..100")
    for value, name in (
        (args.timeout, "--timeout"),
        (args.position_tolerance_mm, "--position-tolerance-mm"),
        (args.orientation_tolerance_deg, "--orientation-tolerance-deg"),
        (args.joint_tolerance_deg, "--joint-tolerance-deg"),
    ):
        try:
            validate_positive_finite(value, name)
        except ValueError as exc:
            parser.error(str(exc))
    if args.method == "angles" and args.coord_mode != 0:
        parser.error("--coord-mode is only meaningful with --method coords")
    install_signal_handlers()
    try:
        result = execute(args)
    except Exception as exc:
        emit("error", type=type(exc).__name__, message=str(exc))
        raise SystemExit(1)
    if result["status"] not in {"dry_run", "reached"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
