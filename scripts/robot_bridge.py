"""One-shot JSON/stdio bridge for laptop-controlled myCobot motion.

The bridge waits for PROCEED before sending the target. An optional J4
correction requires a separate path approval from the laptop.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import select
import time
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mycobot_id.joint_limits import JOINT_LIMITS_DEG

POLL_INTERVAL_S = 0.10
TARGET_TOLERANCE_DEG = 2.0
PATH_TOLERANCE_DEG = 10.0
START_TOLERANCE_DEG = 1.0
HOLD_DRIFT_DEG = 1.0
J4_CORRECTION_MIN_DEG = 0.5
J4_CORRECTION_PROGRESS_DEG = 0.1
J4_CORRECTION_OTHER_DRIFT_DEG = 0.3
from typing import Sequence


def emit(event: str, **values: object) -> None:
    print(json.dumps({"event": event, **values}, separators=(",", ":")), flush=True)


def valid_angles(values: Sequence[float]) -> bool:
    return isinstance(values, (list, tuple)) and len(values) == 6 and all(math.isfinite(float(value)) for value in values)


def read_coords(robot):
    """Read firmware pose once; missing data must never be logged as zero."""
    try:
        values = robot.get_coords()
        if valid_angles(values):
            return [float(value) for value in values], None
        return None, "invalid get_coords response: {!r}".format(values)
    except Exception as exc:
        return None, str(exc)


def validate_grip(value: int, speed: int) -> None:
    if type(value) is not int or not 0 <= value <= 100:
        raise ValueError("gripper opening must be an integer in 0..100")
    if type(speed) is not int or not 1 <= speed <= 100:
        raise ValueError("gripper speed must be an integer in 1..100")


def operate_gripper(robot, value: int, speed: int) -> None:
    validate_grip(value, speed)
    robot.set_gripper_value(value, speed, 1)
    emit("gripper_sent", requested=value, speed=speed)
    time.sleep(0.25)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        state = robot.is_gripper_moving()
        if state == 0:
            actual = robot.get_gripper_value()
            emit("gripper_final", requested=value, actual=actual)
            return
        if state not in (0, 1):
            raise RuntimeError("gripper motion status unavailable")
        time.sleep(0.1)
    raise TimeoutError("gripper completion not confirmed")


def connect(port: str, baud: int):
    from pymycobot.mycobot import MyCobot

    return MyCobot(port, baud)


def close(robot: object) -> None:
    serial_port = getattr(robot, "_serial_port", None)
    if serial_port is not None:
        serial_port.close()


def snapshot(args: argparse.Namespace) -> None:
    robot = connect(args.port, args.baud)
    try:
        angles = robot.get_angles()
        coords = robot.get_coords()
        if not valid_angles(angles):
            raise RuntimeError("invalid get_angles response: {!r}".format(angles))
        emit("snapshot", angles=[float(value) for value in angles], coords=coords)
    finally:
        close(robot)


def sample_angles(robot, samples, started, phase):
    timestamp = round(time.monotonic() - started, 4)
    try:
        values = robot.get_angles()
        angles = [float(value) for value in values] if valid_angles(values) else None
    except Exception:
        angles = None
    if samples is not None:
        samples.append({"t_s": timestamp, "phase": phase, "angles_deg": angles})
    if angles is None:
        raise RuntimeError("joint feedback unavailable ({})".format(phase))
    return angles


def max_joint_error(actual, reference):
    return max(abs(a - b) for a, b in zip(actual, reference))


def path_error(actual, start, target):
    direction = [b - a for a, b in zip(start, target)]
    length_squared = sum(d * d for d in direction)
    if not length_squared:
        return max_joint_error(actual, start)
    fraction = sum((x - a) * d for x, a, d in zip(actual, start, direction)) / length_squared
    fraction = min(1.0, max(0.0, fraction))
    return max_joint_error(actual, [a + fraction * d for a, d in zip(start, direction)])


def supervise_move(robot, start, target, timeout, samples, started, phase_prefix=""):
    deadline = time.monotonic() + timeout
    settle_deadline = None
    hold_start = None
    hold_reference = None
    hold_duration = 5.0 if samples is not None else 1.0
    while time.monotonic() < deadline:
        stop_if_requested()
        moving = robot.is_moving()
        if moving not in (0, 1):
            raise RuntimeError("robot motion status unavailable")
        actual = sample_angles(robot, samples, started, phase_prefix + ("moving" if moving else "holding"))
        deviation = path_error(actual, start, target)
        if deviation > PATH_TOLERANCE_DEG:
            raise RuntimeError("joint path deviation ({:.2f} deg); grasp skipped".format(deviation))
        now = time.monotonic()
        error = max_joint_error(actual, target)
        if hold_start is not None:
            if moving or error > TARGET_TOLERANCE_DEG or max_joint_error(actual, hold_reference) > HOLD_DRIFT_DEG:
                raise RuntimeError("target pose did not hold (max error {:.2f} deg); grasp skipped".format(error))
            if now - hold_start >= hold_duration:
                return actual, error
        elif not moving:
            if settle_deadline is None:
                settle_deadline = now + 2.0
            if error <= TARGET_TOLERANCE_DEG:
                hold_start, hold_reference = now, actual
            elif now >= settle_deadline:
                raise RuntimeError("target not reached (max error {:.2f} deg); grasp skipped".format(error))
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError("motion or position hold timed out; grasp skipped")


def wait_for_final_angles(robot, timeout, started):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stop_if_requested()
        moving = robot.is_moving()
        if moving == 0:
            return sample_angles(robot, None, started, "final")
        if moving != 1:
            raise RuntimeError("robot motion status unavailable")
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError("motion timed out; grasp skipped")


def check_joint_limits(angles):
    for joint, (angle, (low, high)) in enumerate(zip(angles, JOINT_LIMITS_DEG), 1):
        if not low <= angle <= high:
            raise RuntimeError("joint limit exceeded (J{}: {:.2f} deg)".format(joint, angle))


def observe_joint_limits(robot, timeout, samples, started):
    deadline = time.monotonic() + timeout
    stopped_at = None
    while time.monotonic() < deadline:
        stop_if_requested()
        moving = robot.is_moving()
        if moving not in (0, 1):
            raise RuntimeError("robot motion status unavailable")
        actual = sample_angles(robot, samples, started, "moving" if moving else "holding")
        check_joint_limits(actual)
        now = time.monotonic()
        if moving:
            stopped_at = None
        elif samples is None:
            return actual
        elif stopped_at is None:
            stopped_at = now
        elif now - stopped_at >= 5.0:
            return actual
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError("motion timed out; grasp skipped")


def emit_trace(samples, target):
    payload = {"samples": samples, "target_angles_deg": target}
    directory = Path.home() / ".local" / "state" / "urop2-identification" / "motion"
    path = directory / f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid4().hex}.json"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        payload["jetson_file"] = str(path)
    except OSError as exc:
        payload["save_error"] = str(exc)
    emit("motion_trace", **payload)


def stop_if_requested():
    if select.select([sys.stdin], [], [], 0)[0]:
        # EOF (laptop disconnected) and STOP both cancel grasping.
        sys.stdin.readline()
        raise RuntimeError("motion cancelled; grasp skipped")


def correct_j4_once(robot, final, target, args, samples, started):
    residual = target[3] - final[3]
    if abs(residual) <= J4_CORRECTION_MIN_DEG:
        return final, None
    command = list(final)
    command[3] = target[3] + residual
    emit("correction_proposed", start_angles_deg=final, command_angles_deg=command,
         original_target_deg=target, error_before_deg=abs(residual))
    if not select.select([sys.stdin], [], [], 10.0)[0] or sys.stdin.readline().strip() != "CORRECT":
        raise RuntimeError("J4 correction not approved")
    fresh = sample_angles(robot, samples, started, "correction_before")
    if robot.is_moving() != 0 or max_joint_error(fresh, final) > 0.25:
        raise RuntimeError("J4 correction start pose changed")
    stop_if_requested()
    robot.send_angle(4, command[3], args.speed)
    time.sleep(0.25)
    corrected, _ = supervise_move(robot, fresh, command, args.timeout, samples, started, "correction_")
    after = abs(corrected[3] - target[3])
    other_drift = max(abs(b - a) for i, (a, b) in enumerate(zip(final, corrected)) if i != 3)
    full_error_before = max_joint_error(final, target)
    full_error_after = max_joint_error(corrected, target)
    accepted = (
        abs(residual) - after >= J4_CORRECTION_PROGRESS_DEG
        and other_drift <= J4_CORRECTION_OTHER_DRIFT_DEG
        and full_error_after <= full_error_before
    )
    emit("correction_result", error_before_deg=abs(residual), error_after_deg=after,
         command_angles_deg=command, measured_angles_deg=corrected, accepted=accepted,
         other_joint_drift_deg=other_drift, full_error_before_deg=full_error_before,
         full_error_after_deg=full_error_after)
    return corrected, accepted


def transaction(args: argparse.Namespace) -> None:
    target = [float(value) for value in args.target]
    if not valid_angles(target):
        raise ValueError("target must contain six finite angles")
    if not 1 <= args.speed <= 100:
        raise ValueError("speed must be in 1..100")
    robot = connect(args.port, args.baud)
    samples = None
    try:
        current = robot.get_angles()
        if not valid_angles(current):
            raise RuntimeError("invalid get_angles response: {!r}".format(current))
        coords, coords_error = read_coords(robot)
        emit("current", angles=[float(value) for value in current],
             coords=coords, coords_error=coords_error, motion_trace_supported=True,
             j4_correction_supported=True, final_only_supported=True, joint_limits_test_supported=True)
        decision = sys.stdin.readline().strip()
        options = json.loads(decision) if decision.startswith("{") else {"action": decision}
        if options.get("action") != "PROCEED":
            emit("aborted")
            return
        grip = options.get("grip")
        correct_j4 = options.get("correct_j4") is True
        final_only = options.get("final_only") is True
        joint_limits_test = options.get("joint_limits_test") is True
        if joint_limits_test and (final_only or correct_j4 or grip is not None):
            raise RuntimeError("joint limits test requires feedback, no correction, and no grasp")
        if final_only and (correct_j4 or options.get("record_motion") is True):
            raise RuntimeError("final-only mode requires feedback recording and correction to be off")
        if correct_j4 and (grip is not None or args.speed > 10 or
                           any(abs(a - b) > 0.25 for i, (a, b) in enumerate(zip(current, target)) if i != 3)
                           or abs(target[3] - current[3]) > 5.0):
            raise RuntimeError("J4 correction requires a J4-only test up to 5 deg, speed at most 10, and no grasp")
        if grip is not None:
            validate_grip(grip["value"], grip["speed"])
        started = time.monotonic()
        samples = [] if correct_j4 or options.get("record_motion") is True else None
        before = sample_angles(robot, samples, started, "before")
        if joint_limits_test:
            check_joint_limits(target)
            check_joint_limits(before)
        if max_joint_error(before, current) > (0.25 if correct_j4 else START_TOLERANCE_DEG):
            raise RuntimeError("start pose changed; repeat the path check")
        stop_if_requested()
        try:
            if correct_j4:
                robot.send_angle(4, target[3], args.speed)
            else:
                robot.send_angles(target, args.speed)
            time.sleep(0.25)
            if final_only:
                final = wait_for_final_angles(robot, args.timeout, started)
            elif joint_limits_test:
                final = observe_joint_limits(robot, args.timeout, samples, started)
            else:
                final, _ = supervise_move(robot, current, target, args.timeout, samples, started)
            improved = None
            if correct_j4:
                final, improved = correct_j4_once(robot, final, target, args, samples, started)
            stop_if_requested()
        except Exception:
            try:
                robot.stop()
            except Exception as exc:
                print("stop failed: {}".format(exc), file=sys.stderr)
            raise
        coords, coords_error = read_coords(robot)
        emit("final", angles=[float(value) for value in final],
             coords=coords, coords_error=coords_error)
        error = max_joint_error(final, target)
        if error > TARGET_TOLERANCE_DEG and final_only:
            raise RuntimeError("target not reached (max error {:.2f} deg); grasp skipped".format(error))
        if improved is False or (error > TARGET_TOLERANCE_DEG and not joint_limits_test):
            raise RuntimeError("J4 correction failed (original target max error {:.2f} deg)".format(error))
        if grip is not None:
            if select.select([sys.stdin], [], [], 0)[0]:
                raise RuntimeError("motion cancelled; grasp skipped")
            operate_gripper(robot, grip["value"], grip["speed"])
    finally:
        try:
            if samples is not None:
                emit_trace(samples, target)
        finally:
            close(robot)
    emit("done", max_joint_error_deg=error, joint_limits_test=joint_limits_test)


def stop_motion(args: argparse.Namespace) -> None:
    robot = connect(args.port, args.baud)
    try:
        result = robot.stop()
        emit("stopped", result=result)
    finally:
        close(robot)


def gripper(args: argparse.Namespace) -> None:
    validate_grip(args.gripper_value, args.gripper_speed)
    robot = connect(args.port, args.baud)
    try:
        if robot.is_moving() != 0:
            raise RuntimeError("arm must be stationary before manual gripper control")
        operate_gripper(robot, args.gripper_value, args.gripper_speed)
    finally:
        close(robot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "transaction", "stop", "gripper"))
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--target", type=float, nargs=6)
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--gripper-value", type=int, default=100)
    parser.add_argument("--gripper-speed", type=int, default=20)
    args = parser.parse_args()
    try:
        if args.action == "snapshot":
            snapshot(args)
        elif args.action == "stop":
            stop_motion(args)
        elif args.action == "gripper":
            gripper(args)
        else:
            if args.target is None:
                parser.error("transaction requires --target with six values")
            transaction(args)
    except Exception as exc:
        emit("error", message=str(exc), error_type=type(exc).__name__)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
