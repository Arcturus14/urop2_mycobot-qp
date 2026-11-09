"""Print and save myCobot 280 joint angles until Ctrl+C."""
import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between reads")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")

    from pymycobot.mycobot import MyCobot

    directory = Path.home() / "workspace" / "joint_logs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / datetime.now().strftime("joints_%Y%m%d_%H%M%S_%f.csv")
    robot = MyCobot(args.port, args.baud)
    try:
        with path.open("x", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["timestamp", "elapsed_s", *(f"J{i}_deg" for i in range(1, 7)), "error"])
            print(f"CSV: {path} (Ctrl+C to stop)", flush=True)
            start = time.monotonic()
            while True:
                read_started = time.monotonic()
                stamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
                try:
                    angles = robot.get_angles()
                    if not isinstance(angles, (list, tuple)) or len(angles) != 6:
                        raise ValueError(f"Invalid angle response: {angles!r}")
                    values = [float(angle) for angle in angles]
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError(f"Non-finite angle response: {angles!r}")
                    error = ""
                    message = "  ".join(f"J{i}={value:7.2f}°" for i, value in enumerate(values, 1))
                except Exception as exc:
                    values = [""] * 6
                    error = str(exc)
                    message = f"Read error: {error}"
                writer.writerow([stamp, f"{read_started - start:.3f}", *values, error])
                output.flush()
                print(f"{stamp}  {message}", flush=True)
                time.sleep(max(0.0, args.interval - (time.monotonic() - read_started)))
    except KeyboardInterrupt:
        print(f"Stopped. CSV: {path}", flush=True)
    finally:
        robot._serial_port.close()


if __name__ == "__main__":
    main()
