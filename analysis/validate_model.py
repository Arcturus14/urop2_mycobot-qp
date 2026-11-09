from __future__ import annotations

import argparse
import json

import numpy as np

from mycobot_id.model import simulate_second_order
from .common import load_joint_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate fitted parameters on a CSV dataset")
    parser.add_argument("csv")
    parser.add_argument("model_json")
    args = parser.parse_args()
    model = json.loads(open(args.model_json, encoding="utf-8").read())
    joint = int(model["joint"])
    frame, cmd_col, actual_col = load_joint_data(args.csv, joint)
    t = frame["t_s"].to_numpy(float)
    t -= t[0]
    command = frame[cmd_col].to_numpy(float)
    actual = frame[actual_col].to_numpy(float)
    prediction = simulate_second_order(
        t, command, float(model["omega_n_rad_s"]), float(model["zeta"]),
        float(model["delay_s"]), float(model["dc_gain"]), actual[0],
    )
    rmse = float(np.sqrt(np.mean((prediction - actual) ** 2)))
    mae = float(np.mean(np.abs(prediction - actual)))
    print(f"validation_csv={args.csv}")
    print(f"joint={joint} samples={len(frame)} RMSE={rmse:.6f}deg MAE={mae:.6f}deg")


if __name__ == "__main__":
    main()
