"""Fit a local delayed joint response from one or more recorded QP runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_run(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["work_ms"]]
    if len(rows) < 3:
        raise ValueError(f"Too few control samples in {path}")
    times = np.array([float(row["sample_s"]) for row in rows])
    angles = np.array([[float(row[f"q_actual_j{i}_deg"]) for i in range(1, 7)] for row in rows])
    sent = [index for index, row in enumerate(rows) if row["send_ms"]]
    sent_times = np.array([
        times[index] + (float(rows[index]["work_ms"]) -
                        0.5 * (float(rows[index]["read_ms"]) + float(rows[index]["send_ms"]))) / 1000
        for index in sent
    ])
    commands = np.array([[float(rows[index][f"q_cmd_j{i}_deg"]) for i in range(1, 7)] for index in sent])
    return times, angles, sent_times, commands


def fit(runs):
    durations = np.concatenate([np.diff(run[0]) for run in runs])
    actual = np.vstack([run[1][:-1] for run in runs])
    observed = np.vstack([np.diff(run[1], axis=0) for run in runs])
    midpoints = [(run[0][1:] + run[0][:-1]) / 2 for run in runs]
    rates = np.linspace(2.0, 65.0, 127)
    response = -np.expm1(-durations[:, None] * rates)
    best = np.full(6, np.inf)
    parameters = np.zeros((6, 3))
    for delay in np.linspace(0, 0.20, 41):
        held = []
        for (_, angles, sent_times, commands), midpoint in zip(runs, midpoints):
            indices = np.searchsorted(sent_times, midpoint - delay, side="right")
            held.append(np.vstack((angles[0], commands))[indices])
        error = np.vstack(held) - actual
        for deadband in np.linspace(0, 1.5, 31):
            effective = np.sign(error) * np.maximum(np.abs(error) - deadband, 0)
            residual = effective[:, None, :] * response[:, :, None] - observed[:, None, :]
            mse = np.mean(residual**2, axis=0)
            index = mse.argmin(axis=0)
            score = mse[index, np.arange(6)]
            improved = score < best
            best[improved] = score[improved]
            parameters[improved] = np.column_stack(
                (rates[index], np.full(6, delay), np.full(6, deadband))
            )[improved]
    return {
        f"J{index}": dict(rate_gain_per_s=float(rate), delay_s=float(delay),
                         deadband_deg=float(deadband), fit_rmse_deg=float(np.sqrt(mse)))
        for index, ((rate, delay, deadband), mse) in enumerate(zip(parameters, best), 1)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fit([read_run(log) for log in args.logs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved fitted response to {args.output}")


if __name__ == "__main__":
    main()
