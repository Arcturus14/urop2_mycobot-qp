from __future__ import annotations

import argparse


def add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("mock", "real"), default="mock")
    parser.add_argument("--port", help="Jetson Nano serial device, normally /dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=1000000)
