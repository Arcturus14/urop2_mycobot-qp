"""Small, single-joint, bidirectional identification around the current pose."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import uuid

from mycobot_id.acquisition import CsvRecorder, sample_row, timed_read
from mycobot_id.robot import create_robot
from .common import add_connection_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_connection_arguments(parser)
    parser.add_argument("--confirm-motion", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--joints", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--reference-angles", type=float, nargs=6,
                        help="Optional fixed pose reference; no reposition command is issued")
    args = parser.parse_args()
    if args.confirm_motion != "I_UNDERSTAND":
        parser.error("--confirm-motion I_UNDERSTAND required")
    if args.repeats != 2 or any(j not in range(1, 7) for j in args.joints):
        parser.error("Exactly two repeats and joints 1..6 are supported")
    if len(set(args.joints)) != len(args.joints):
        parser.error("Duplicate joint")
    if args.backend == "real":
        owners = subprocess.run(["fuser", args.port], capture_output=True, text=True)
        if owners.returncode != 1:
            raise SystemExit("Serial device is occupied or ownership check failed: " + owners.stdout + owners.stderr)
    robot = create_robot(args.backend, args.port, args.baud)
    raw = Path(args.output_dir)
    raw.mkdir(parents=True, exist_ok=True)
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    metadata_path = raw / (run + "_identification.json")
    origin = time.perf_counter()
    initial_angles, _, _ = timed_read(robot, origin)
    baseline = list(args.reference_angles) if args.reference_angles else list(initial_angles)
    if not all(math.isfinite(value) for value in baseline):
        parser.error("Reference angles must be finite")
    if any(abs(a - b) > 1.5 for a, b in zip(initial_angles, baseline)):
        raise SystemExit("Actual pose is too far from the fixed experiment reference")
    # Conservative software envelope, separate from the library's wider limits.
    minimum = [-160, -90, -150, -145, -100, -165]
    maximum = [160, 90, 45, 145, 100, 165]
    for joint in args.joints:
        if not minimum[joint - 1] + 3.3 <= baseline[joint - 1] <= maximum[joint - 1] - 3.3:
            raise SystemExit("Insufficient +/-3 deg margin at joint " + str(joint))
    status = {}
    if args.backend == "real":
        status = {"power_on": robot._robot.is_power_on(),
                  "all_servos_enabled": robot._robot.is_all_servo_enable()}
        if status != {"power_on": 1, "all_servos_enabled": 1}:
            raise SystemExit("Robot must already be powered and servo-enabled: " + str(status))
    settings = {"speed": 20, "pre_s": 0.5, "post_s": 2.0, "poll_interval_s": 0.05,
                "target_offsets_deg": [1.5, 0.0, -1.5, 0.0, 3.0, 0.0, -3.0, 0.0],
                "repeats": args.repeats, "joints": args.joints,
                "minimum_deg": minimum, "maximum_deg": maximum,
                "envelope_margin_deg": 0.5, "other_joint_tolerance_deg": 1.5,
                "stationary_tail_s": 0.4, "stationary_peak_to_peak_deg": 0.3,
                "command_quiet_s": 0.05, "max_consecutive_read_failures": 3}
    metadata = {"run_id": run, "backend": args.backend, "host": platform.node(),
                "utc_start": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "port": args.port, "baud": args.baud, "initial_angles_deg": initial_angles,
                "reference_angles_deg": baseline,
                "initial_status": status, "settings": settings, "events": [], "status": "running",
                "read_failures": [],
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "pymycobot": importlib.metadata.version("pymycobot") if args.backend == "real" else None}

    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def interrupted(signum, _frame):
        raise KeyboardInterrupt("Signal " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    commanded = list(baseline)
    held_angles = list(initial_angles)
    save_metadata()
    print("RUN", run, "BASELINE", baseline, "STATUS", status, flush=True)

    def record_window(recorder, seconds, phase, joint, command_start=None, command_end=None):
        end = time.perf_counter() + seconds
        next_sample = time.perf_counter()
        last = None
        history = []
        consecutive_failures = 0
        while time.perf_counter() < end:
            read_attempt = time.perf_counter() - origin
            try:
                angles, rs, re = timed_read(robot, origin)
            except RuntimeError as error:
                consecutive_failures += 1
                metadata["read_failures"].append({"joint": joint, "phase": phase,
                    "read_start_s": read_attempt, "read_end_s": time.perf_counter() - origin,
                    "error": str(error)})
                save_metadata()
                if consecutive_failures >= settings["max_consecutive_read_failures"]:
                    raise
                next_sample = time.perf_counter() + settings["poll_interval_s"]
                time.sleep(settings["poll_interval_s"])
                continue
            consecutive_failures = 0
            recorder.append(sample_row(
                run_id=run, phase=phase, target_joint=joint, speed=settings["speed"],
                q_cmd=commanded, q_actual=angles, read_start_s=rs, read_end_s=re,
                command_start_s=command_start, command_end_s=command_end))
            for index, angle in enumerate(angles):
                if abs(angle - baseline[index]) > 3.0 + settings["envelope_margin_deg"]:
                    raise RuntimeError("Movement envelope exceeded at J" + str(index + 1))
                if index != joint - 1 and abs(angle - held_angles[index]) > settings["other_joint_tolerance_deg"]:
                    raise RuntimeError("Unexpected other-joint motion at J" + str(index + 1))
            last = angles
            history.append((time.perf_counter(), angles))
            next_sample += settings["poll_interval_s"]
            time.sleep(max(0.0, next_sample - time.perf_counter()))
        if len(history) < 3:
            raise RuntimeError("Insufficient valid joint samples")
        tail = [angles[joint - 1] for timestamp, angles in history
                if timestamp >= history[-1][0] - settings["stationary_tail_s"]]
        spread = max(tail) - min(tail)
        return last, spread

    try:
        for joint in args.joints:
            # Earlier commanded joints can retain steady-state error. Compare
            # uncommanded joints with their actual pose at this joint's start.
            held_angles, _, _ = timed_read(robot, origin)
            for repeat in range(args.repeats):
                for step, offset in enumerate(settings["target_offsets_deg"]):
                    filename = f"{run}_J{joint}_r{repeat + 1}_s{step + 1}.csv"
                    target = round(baseline[joint - 1] + offset, 2)
                    previous = commanded[joint - 1]
                    event = {"file": filename, "joint": joint, "repeat": repeat + 1,
                             "step": step + 1, "offset_deg": offset,
                             "from_deg": previous, "to_deg": target,
                             "split": "train" if repeat == 0 else "validation", "status": "running"}
                    metadata["events"].append(event)
                    save_metadata()
                    with CsvRecorder(raw / filename) as recorder:
                        last, spread = record_window(recorder, settings["pre_s"], "pre_step", joint)
                        if spread > settings["stationary_peak_to_peak_deg"]:
                            raise RuntimeError("Joint is not stationary before the step")
                        start = time.perf_counter() - origin
                        robot.send_angle(joint, target, settings["speed"])
                        end = time.perf_counter() - origin
                        commanded[joint - 1] = target
                        event.update(command_start_s=start, command_end_s=end)
                        save_metadata()
                        time.sleep(settings["command_quiet_s"])
                        last, spread = record_window(recorder, settings["post_s"], "post_step", joint, start, end)
                        event["samples"] = recorder.count
                        event["final_angles_deg"] = last
                        event["endpoint_error_deg"] = target - last[joint - 1]
                        event["tail_peak_to_peak_deg"] = spread
                        if spread > settings["stationary_peak_to_peak_deg"]:
                            raise RuntimeError("Joint still moving at the end of recording")
                        # A stable position error/deadband is an identification
                        # result. It must not be rejected as failure to reach.
                    event["status"] = "complete"
                    print(f"J{joint} repeat={repeat + 1} step={step + 1}/8 {previous:.2f}->{target:.2f} final={last[joint - 1]:.2f}", flush=True)
                    save_metadata()
        metadata["status"] = "complete"
    except BaseException as error:
        metadata["status"] = "aborted"
        metadata["error"] = repr(error)
        if metadata["events"] and metadata["events"][-1]["status"] == "running":
            metadata["events"][-1]["status"] = "aborted"
        if args.backend == "real":
            robot._robot.stop()
        # Do not issue an automatic return motion after an error.
        raise
    finally:
        metadata["utc_end"] = datetime.now(timezone.utc).isoformat()
        save_metadata()
        if hasattr(robot, "close"):
            robot.close()
        print("METADATA", metadata_path, "STATUS", metadata["status"], flush=True)


if __name__ == "__main__":
    main()
