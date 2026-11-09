"""Plot per-step FK position error and report the last recorded error."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.csv_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error("CSV contains no position samples")

    time_s = [float(row["sample_s"]) for row in rows]
    error = np.asarray([
        [float(row[f"ee_{axis}_mm"]) - float(row[f"goal_{axis}_mm"]) for axis in "xyz"]
        for row in rows
    ])
    magnitude = np.linalg.norm(error, axis=1)
    last = rows[-1]
    post_stop = last["status"] in {"goal_reached", "goal_not_held"} and last.get("work_ms") == ""
    label = "Post-stop" if post_stop else "Last sample"

    fig, (components, norm) = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                                          constrained_layout=True)
    for axis, values in zip("XYZ", error.T):
        components.plot(time_s, values, marker=".", label=axis)
    components.axhline(0, color="0.7", linewidth=1)
    components.set_ylabel("FK position minus goal (mm)")
    components.legend()
    norm.plot(time_s, magnitude, marker=".")
    tolerance = float(last.get("tolerance_mm") or 3.0)
    norm.axhline(tolerance, linestyle="--", color="0.5", label=f"{tolerance:g} mm tolerance")
    norm.scatter(time_s[-1], magnitude[-1], zorder=3,
                 label=f"{label}: {magnitude[-1]:.3f} mm")
    norm.set(xlabel="Sample time (s)", ylabel="Error norm (mm)")
    norm.legend()
    output = args.output or args.csv_file.with_name(args.csv_file.stem + "_error.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    x, y, z = error[-1]
    print(f"{label} ({last['status']}): XYZ = ({x:.3f}, {y:.3f}, {z:.3f}) mm, "
          f"norm = {magnitude[-1]:.3f} mm")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
