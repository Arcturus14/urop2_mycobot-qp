from __future__ import annotations

import argparse
import statistics
import time
import uuid

from mycobot_id.acquisition import CsvRecorder, sample_row, timed_read
from mycobot_id.robot import create_robot
from .common import add_connection_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure get_angles timing and jitter")
    add_connection_arguments(parser)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output", default="data/raw/sampling.csv")
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be >= 2")
    robot = create_robot(args.backend, args.port, args.baud)
    origin = time.perf_counter()
    run_id = str(uuid.uuid4())
    midpoints, durations = [], []
    with CsvRecorder(args.output) as recorder:
        for _ in range(args.samples):
            angles, read_start, read_end = timed_read(robot, origin)
            midpoint = (read_start + read_end) / 2.0
            midpoints.append(midpoint)
            durations.append(read_end - read_start)
            recorder.append(sample_row(
                run_id=run_id, phase="sampling", target_joint=0, speed=0,
                q_cmd=None, q_actual=angles,
                read_start_s=read_start, read_end_s=read_end,
            ))
    intervals = [b - a for a, b in zip(midpoints, midpoints[1:])]
    print(f"samples={args.samples}")
    print(f"read mean={statistics.mean(durations):.6f}s max={max(durations):.6f}s")
    if intervals:
        print(f"interval mean={statistics.mean(intervals):.6f}s stdev={statistics.stdev(intervals) if len(intervals) > 1 else 0:.6f}s")
        print(f"interval min={min(intervals):.6f}s max={max(intervals):.6f}s")
    print(f"Wrote timing samples to {args.output}")


if __name__ == "__main__":
    main()
