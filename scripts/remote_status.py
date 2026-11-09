"""Read robot status without sending a motion or power command."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess

from pymycobot.mycobot import MyCobot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=1000000)
    args = parser.parse_args()
    owners = subprocess.run(["fuser", args.port], capture_output=True, text=True)
    if owners.returncode != 1:
        raise SystemExit("Serial port in use or ownership check failed")
    robot = MyCobot(args.port, args.baud)
    try:
        print(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
            "angles_deg": robot.get_angles(), "is_moving": robot.is_moving(),
            "power_on": robot.is_power_on(), "all_servos_enabled": robot.is_all_servo_enable()}))
    finally:
        robot._serial_port.close()


if __name__ == "__main__":
    main()
