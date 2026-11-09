from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .common import load_joint_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot command and measured joint response")
    parser.add_argument("csv")
    parser.add_argument("--joint", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()
    frame, cmd, actual = load_joint_data(args.csv, args.joint)
    output = Path(args.output or f"reports/figures/{Path(args.csv).stem}_J{args.joint}.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(frame["t_s"], frame[cmd], where="post", label="command", linewidth=1.5)
    ax.plot(frame["t_s"], frame[actual], label="actual", linewidth=1.3)
    ax.set(xlabel="Jetson monotonic time from start (s)", ylabel="Angle (deg)",
           title=f"Joint {args.joint} closed-loop step response")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
