"""Move once to the laptop GUI's reset target without a path check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUI_INITIAL_TARGET_DEG = (0.0, 0.0, 0.0, 0.0, 90.0, -45.0)
CONFIRMATION = "I_UNDERSTAND_NO_PATH_CHECK"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "jetson.json")
    parser.add_argument("--state", type=Path, default=PROJECT_ROOT / ".jetson" / "state.json")
    parser.add_argument("--speed", type=int, default=20, help="Firmware speed value, 1..100")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--confirm-motion", default="")
    args = parser.parse_args()
    if not 1 <= args.speed <= 100:
        parser.error("--speed must be in 1..100")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    state = json.loads(args.state.read_text(encoding="utf-8"))
    remote_python = config["remote_root"] + "/.venv/bin/python"
    remote_bridge = state["release"] + "/scripts/robot_bridge.py"
    target = [float(value) for value in GUI_INITIAL_TARGET_DEG]

    print("GUI reset target (deg):", target)
    print("Path and collision checks: DISABLED")
    if args.confirm_motion != CONFIRMATION:
        print("No motion sent. Re-run with:")
        print(f"  --confirm-motion {CONFIRMATION}")
        return

    command = [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=yes", config["ssh_host"],
        remote_python, remote_bridge, "transaction",
        "--port", config["port"], "--baud", str(config["baud"]),
        "--target", *(format(value, ".6f") for value in target),
        "--speed", str(args.speed), "--timeout", str(args.timeout),
    ]
    decision = json.dumps({"action": "PROCEED", "final_only": True}) + "\n"
    # Keep stdin open after sending PROCEED.  robot_bridge deliberately treats
    # EOF as a cancellation signal, so subprocess.run(input=...) used to cancel
    # the motion immediately after its initial 250 ms delay.
    process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True)
    assert process.stdin is not None
    process.stdin.write(decision)
    process.stdin.flush()
    try:
        returncode = process.wait()
    except KeyboardInterrupt:
        # Closing the SSH process disconnects the remote session.  The bridge
        # sees stdin EOF and invokes robot.stop() in its exception handler.
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        process.stdin.close()
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
