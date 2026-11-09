from __future__ import annotations

import pandas as pd


def load_joint_data(path: str, joint: int) -> tuple[pd.DataFrame, str, str]:
    if not 1 <= joint <= 6:
        raise ValueError("joint must be between 1 and 6")
    frame = pd.read_csv(path).sort_values("t_s").drop_duplicates("t_s")
    cmd, actual = f"q_cmd_j{joint}", f"q_actual_j{joint}"
    missing = {"t_s", cmd, actual} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    return frame, cmd, actual
