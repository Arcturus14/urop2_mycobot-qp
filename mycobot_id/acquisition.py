from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Iterable, Sequence

from .robot import Robot


FIELDNAMES = [
    "run_id", "phase", "target_joint", "speed", "t_s",
    "read_start_s", "read_end_s", "read_duration_s",
    "command_start_s", "command_end_s",
] + [f"q_cmd_j{i}" for i in range(1, 7)] + [f"q_actual_j{i}" for i in range(1, 7)]


def timed_read(robot: Robot, experiment_start: float) -> tuple[list[float], float, float]:
    start = time.perf_counter()
    angles = robot.get_angles()
    end = time.perf_counter()
    if len(angles) != 6:
        raise RuntimeError(f"Expected six joint angles, received {angles!r}")
    if not all(math.isfinite(float(value)) for value in angles):
        raise RuntimeError(f"Non-finite joint angle: {angles!r}")
    return angles, start - experiment_start, end - experiment_start


def sample_row(
    *, run_id: str, phase: str, target_joint: int, speed: int,
    q_cmd: Sequence[float] | None, q_actual: Sequence[float],
    read_start_s: float, read_end_s: float,
    command_start_s: float | None = None, command_end_s: float | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": run_id,
        "phase": phase,
        "target_joint": target_joint,
        "speed": speed,
        "t_s": (read_start_s + read_end_s) / 2.0,
        "read_start_s": read_start_s,
        "read_end_s": read_end_s,
        "read_duration_s": read_end_s - read_start_s,
        "command_start_s": "" if command_start_s is None else command_start_s,
        "command_end_s": "" if command_end_s is None else command_end_s,
    }
    row.update({f"q_cmd_j{i + 1}": "" if q_cmd is None else float(q_cmd[i]) for i in range(6)})
    row.update({f"q_actual_j{i + 1}": float(q_actual[i]) for i in range(6)})
    return row


class CsvRecorder:
    """Exclusive creation and per-sample flush preserve partial measurements."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=FIELDNAMES)
        self.writer.writeheader()
        self.handle.flush()
        self.count = 0

    def append(self, row: dict[str, object]) -> None:
        self.writer.writerow(row)
        self.handle.flush()
        self.count += 1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.handle.close()


def write_csv(path: str | Path, rows: Iterable[dict[str, object]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return output
