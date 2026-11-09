from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize stationary joint-angle noise")
    parser.add_argument("csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    columns = [f"q_actual_j{i}" for i in range(1, 7)]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    summary = pd.DataFrame({
        "mean_deg": frame[columns].mean(),
        "std_deg": frame[columns].std(ddof=1),
        "peak_to_peak_deg": frame[columns].max() - frame[columns].min(),
    })
    summary.index = [f"J{i}" for i in range(1, 7)]
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
