"""Read-only acquisition on Jetson. Never sends a position/power command."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    args = parser.parse_args()
    if args.samples < 2 or not 0 < args.duration <= 300 or args.poll_interval <= 0:
        parser.error("samples >= 2, 0 < duration <= 300, poll-interval > 0 required")
    # Refuse to compete with another process that has the serial device open.
    check = subprocess.run(["fuser", args.port], capture_output=True, text=True)
    if check.returncode == 0:
        raise SystemExit("Serial device is in use: " + check.stdout.strip())
    if check.returncode != 1:
        raise SystemExit("Unable to check serial ownership: " + check.stderr.strip())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = stamp + "_" + uuid.uuid4().hex[:6]
    raw = Path("data/raw")
    raw.mkdir(parents=True, exist_ok=True)
    metadata_path = raw / (run + "_environment.json")
    metadata = {
        "run_id": run, "utc_start": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "platform": platform.platform(),
        "python": sys.version, "pymycobot": importlib.metadata.version("pymycobot"),
        "pyserial": importlib.metadata.version("pyserial"),
        "port": args.port, "baud": args.baud,
        "clock": str(time.get_clock_info("perf_counter")),
        "position_commands_sent": False, "status": "running",
        "stationary_assumption": "No position commands sent; inspect CSV for external motion",
        "files": [],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        commands = [
            ["experiments.measure_sampling", "--samples", str(args.samples),
             "--output", str(raw / (run + "_sampling.csv"))],
            ["experiments.measure_noise", "--duration", str(args.duration),
             "--poll-interval", str(args.poll_interval),
             "--output", str(raw / (run + "_noise.csv"))],
        ]
        for module, *options in commands:
            subprocess.run([sys.executable, "-m", module, "--backend", "real",
                            "--port", args.port, "--baud", str(args.baud), *options],
                           check=True, timeout=args.duration + 90)
            metadata["files"].append(options[-1])
        metadata["status"] = "complete"
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        raise
    finally:
        metadata["utc_end"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
