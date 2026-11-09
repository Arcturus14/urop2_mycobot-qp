from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from mycobot_id.model import simulate_second_order
from .common import load_joint_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a delayed second-order closed-loop model")
    parser.add_argument("csv")
    parser.add_argument("--joint", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()
    frame, cmd_col, actual_col = load_joint_data(args.csv, args.joint)
    t = frame["t_s"].to_numpy(float)
    t -= t[0]
    command = frame[cmd_col].to_numpy(float)
    actual = frame[actual_col].to_numpy(float)

    def residual(parameters: np.ndarray) -> np.ndarray:
        prediction = simulate_second_order(t, command, *parameters, y0=actual[0])
        return prediction - actual

    fit = least_squares(residual, x0=[5.0, 0.8, 0.05, 1.0],
                        bounds=([0.05, 0.05, 0.0, 0.5], [50.0, 5.0, 2.0, 1.5]))
    omega_n, zeta, delay_s, gain = map(float, fit.x)
    prediction = simulate_second_order(t, command, omega_n, zeta, delay_s, gain, actual[0])
    rmse = float(np.sqrt(np.mean((prediction - actual) ** 2)))
    ss_res = float(np.sum((actual - prediction) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    result = {
        "source_csv": str(Path(args.csv)), "joint": args.joint,
        "model": "delayed_second_order", "omega_n_rad_s": omega_n,
        "zeta": zeta, "delay_s": delay_s, "dc_gain": gain,
        "rmse_deg": rmse, "r_squared": r_squared,
        "success": bool(fit.success), "optimizer_message": fit.message,
    }
    output = Path(args.output or f"data/processed/{Path(args.csv).stem}_fit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
