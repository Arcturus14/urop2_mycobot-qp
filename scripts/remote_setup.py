"""Create an isolated acquisition environment using vendor-installed packages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_dir")
    args = parser.parse_args()
    if sys.version_info < (3, 8):
        raise SystemExit("Acquisition requires Python >= 3.8")
    destination = Path(args.env_dir)
    if not destination.exists():
        venv.EnvBuilder(system_site_packages=True, with_pip=False).create(destination)
    python = destination / "bin" / "python"
    subprocess.run([str(python), "-c",
                    "import sys, pymycobot, serial; from pymycobot.mycobot import MyCobot; "
                    "from mycobot_id.robot import MockRobot; "
                    "print('Python:', sys.version); print('pymycobot:', pymycobot.__file__); "
                    "print('serial:', serial.__file__); print('mock:', MockRobot().get_angles())"],
                   check=True)
    print(json.dumps({"env_python": str(python), "system_site_packages": True}))


if __name__ == "__main__":
    main()
