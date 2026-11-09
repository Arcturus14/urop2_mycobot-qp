from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from mycobot_id.acquisition import sample_row, timed_read, write_csv
from mycobot_id.robot import create_robot
from .common import add_connection_arguments


def load_config(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"output", "joint", "initial_angles_deg", "target_deg", "speed",
                "settle_before_s", "record_after_s", "poll_interval_s"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")
    if len(config["initial_angles_deg"]) != 6:
        raise ValueError("initial_angles_deg must contain six values")
    if not 1 <= int(config["joint"]) <= 6:
        raise ValueError("joint must be between 1 and 6")
    if not 1 <= int(config["speed"]) <= 100:
        raise ValueError("speed must be between 1 and 100")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one closed-loop joint step test")
    add_connection_arguments(parser)
    parser.add_argument("--config", required=True)
    parser.add_argument("--confirm-motion", default="",
                        help="Real backend requires the exact value I_UNDERSTAND")
    args = parser.parse_args()
    if args.backend == "real" and args.confirm_motion != "I_UNDERSTAND":
        raise SystemExit("Real motion blocked. Re-run with --confirm-motion I_UNDERSTAND after safety checks.")

    config = load_config(args.config)
    robot = create_robot(args.backend, args.port, args.baud)
    initial = [float(value) for value in config["initial_angles_deg"]]
    target = list(initial)
    joint = int(config["joint"])
    target[joint - 1] = float(config["target_deg"])
    speed = int(config["speed"])
    run_id = str(uuid.uuid4())
    rows: list[dict[str, object]] = []

    # Establish the requested start pose; allow it to settle before timing the step.
    robot.send_angles(initial, speed)
    time.sleep(float(config["settle_before_s"]))
    origin = time.perf_counter()
    pre_duration = min(1.0, max(0.2, float(config["settle_before_s"]) / 2.0))
    next_sample = origin
    while time.perf_counter() - origin < pre_duration:
        angles, rs, re = timed_read(robot, origin)
        rows.append(sample_row(run_id=run_id, phase="pre_step", target_joint=joint,
                               speed=speed, q_cmd=initial, q_actual=angles,
                               read_start_s=rs, read_end_s=re))
        next_sample += float(config["poll_interval_s"])
        time.sleep(max(0.0, next_sample - time.perf_counter()))

    command_start = time.perf_counter()
    robot.send_angle(joint, target[joint - 1], speed)
    command_end = time.perf_counter()
    command_start_s, command_end_s = command_start - origin, command_end - origin
    end_at = command_end + float(config["record_after_s"])
    next_sample = time.perf_counter()
    while time.perf_counter() < end_at:
        angles, rs, re = timed_read(robot, origin)
        rows.append(sample_row(run_id=run_id, phase="post_step", target_joint=joint,
                               speed=speed, q_cmd=target, q_actual=angles,
                               read_start_s=rs, read_end_s=re,
                               command_start_s=command_start_s, command_end_s=command_end_s))
        next_sample += float(config["poll_interval_s"])
        time.sleep(max(0.0, next_sample - time.perf_counter()))

    output = write_csv(config["output"], rows)
    print(f"Wrote {len(rows)} samples to {output}")
    if config.get("return_to_initial", False):
        robot.send_angles(initial, speed)
        print("Return-to-initial command sent")


if __name__ == "__main__":
    main()
