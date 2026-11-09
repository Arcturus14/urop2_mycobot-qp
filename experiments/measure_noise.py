from __future__ import annotations

import argparse
import time
import uuid

from mycobot_id.acquisition import CsvRecorder, sample_row, timed_read
from mycobot_id.robot import create_robot
from .common import add_connection_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description="Record stationary joint-angle noise")
    add_connection_arguments(parser)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=float, default=0.02)
    parser.add_argument("--output", default="data/raw/noise.csv")
    args = parser.parse_args()
    if args.duration <= 0 or args.poll_interval <= 0:
        parser.error("--duration and --poll-interval must be > 0")
    robot = create_robot(args.backend, args.port, args.baud)
    origin = time.perf_counter()
    run_id = str(uuid.uuid4())
    next_sample = origin
    with CsvRecorder(args.output) as recorder:
        while time.perf_counter() - origin < args.duration:
            angles, read_start, read_end = timed_read(robot, origin)
            recorder.append(sample_row(
                run_id=run_id, phase="stationary", target_joint=0, speed=0,
                q_cmd=None, q_actual=angles,
                read_start_s=read_start, read_end_s=read_end,
            ))
            next_sample += args.poll_interval
            time.sleep(max(0.0, next_sample - time.perf_counter()))
    print(f"Wrote {recorder.count} samples to {args.output}")


if __name__ == "__main__":
    main()
