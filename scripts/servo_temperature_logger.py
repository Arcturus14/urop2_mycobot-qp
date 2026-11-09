"""Read and log six myCobot servo temperatures every 15 seconds."""
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
    args = parser.parse_args()

    from pymycobot.mycobot280 import MyCobot280

    directory = Path.home() / "workspace" / "temperature_logs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / datetime.now().strftime("servo_temps_%Y%m%d_%H%M%S_%f.csv")
    robot = MyCobot280(args.port, args.baud)
    try:
        with path.open("x", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["timestamp", "elapsed_s", *(f"J{i}_C" for i in range(1, 7)), "error"])
            print(f"CSV: {path} (Ctrl+C to stop)", flush=True)
            start = time.monotonic()
            next_sample = start
            while True:
                time.sleep(max(0, next_sample - time.monotonic()))
                stamp = datetime.now().astimezone().isoformat(timespec="seconds")
                elapsed = time.monotonic() - start
                try:
                    response = robot.get_servo_temps()
                    if not isinstance(response, (list, tuple)) or len(response) != 6:
                        raise ValueError(f"Invalid temperature response: {response!r}")
                    values = [float(value) for value in response]
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError(f"Non-finite temperature response: {response!r}")
                    error = ""
                    message = "  ".join(f"J{i}={value:g} C" for i, value in enumerate(values, 1))
                except Exception as exc:
                    values = [""] * 6
                    error = str(exc)
                    message = f"Read error: {error}"
                writer.writerow([stamp, f"{elapsed:.3f}", *values, error])
                output.flush()
                print(f"{stamp}  {message}", flush=True)
                next_sample += 15
                now = time.monotonic()
                if next_sample < now:
                    next_sample += 15 * math.ceil((now - next_sample) / 15)
    except KeyboardInterrupt:
        print(f"Stopped. CSV: {path}", flush=True)
    finally:
        robot._serial_port.close()


if __name__ == "__main__":
    main()
