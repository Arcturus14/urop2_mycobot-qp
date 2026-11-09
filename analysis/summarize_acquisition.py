"""Summarize timing and stationary-angle CSVs fetched from the Jetson."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Timing data must contain finite samples")
    return {
        "mean_ms": float(np.mean(values) * 1000),
        "std_ms": float(np.std(values, ddof=1) * 1000) if len(values) > 1 else 0.0,
        "min_ms": float(np.min(values) * 1000),
        "p50_ms": float(np.percentile(values, 50) * 1000),
        "p95_ms": float(np.percentile(values, 95) * 1000),
        "p99_ms": float(np.percentile(values, 99) * 1000),
        "max_ms": float(np.max(values) * 1000),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Downloaded directory containing CSVs")
    parser.add_argument("--output", required=True, help="Directory for JSON and PNG report")
    args = parser.parse_args()
    source = Path(args.directory)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"source": str(source), "datasets": {}}
    csvs = sorted(source.rglob("*.csv"))
    if not csvs:
        parser.error("No CSV files found")
    for path in csvs:
        frame = pd.read_csv(path)
        t = frame["t_s"].to_numpy(float)
        if len(t) < 2 or not (np.diff(t) > 0).all():
            raise ValueError(f"Invalid monotonic timestamps: {path}")
        angles = frame[[f"q_actual_j{i}" for i in range(1, 7)]]
        if not np.isfinite(angles.to_numpy(float)).all():
            raise ValueError(f"Missing or non-finite angles: {path}")
        summary = {
            "samples": len(frame), "span_s": float(t[-1] - t[0]),
            "phase": str(frame["phase"].iloc[0]),
            "sample_interval": distribution(np.diff(t)),
            "read_duration": distribution(frame["read_duration_s"]),
            "joints": {},
        }
        for i in range(1, 7):
            values = angles[f"q_actual_j{i}"]
            summary["joints"][f"J{i}"] = {
                "mean_deg": float(values.mean()),
                "std_deg": float(values.std(ddof=1)),
                "peak_to_peak_deg": float(values.max() - values.min()),
                "first_deg": float(values.iloc[0]), "last_deg": float(values.iloc[-1]),
                "distinct_values": int(values.nunique()),
            }
        report["datasets"][path.name] = summary
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(t[1:], np.diff(t) * 1000, linewidth=0.8)
        axes[0].set_ylabel("Sample interval (ms)")
        axes[1].plot(t, frame["read_duration_s"] * 1000, linewidth=0.8)
        axes[1].set_ylabel("API read duration (ms)")
        for i in range(1, 7):
            values = angles[f"q_actual_j{i}"]
            axes[2].plot(t, values - values.iloc[0], label=f"J{i}", linewidth=0.8)
        axes[2].set(xlabel="Jetson monotonic time from start (s)",
                    ylabel="Angle change (deg)")
        axes[2].legend(ncol=6)
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.suptitle(path.stem)
        fig.tight_layout()
        fig.savefig(output / (path.stem + ".png"), dpi=160)
        plt.close(fig)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved report:", output)


if __name__ == "__main__":
    main()
