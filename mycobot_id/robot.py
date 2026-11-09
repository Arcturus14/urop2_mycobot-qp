from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Protocol, Sequence


class Robot(Protocol):
    def get_angles(self) -> list[float]: ...
    def send_angles(self, angles: Sequence[float], speed: int) -> None: ...
    def send_angle(self, joint: int, angle: float, speed: int) -> None: ...
    def stop(self) -> None: ...


class MyCobotRobot:
    """Thin, lazy adapter around pymycobot.MyCobot."""

    def __init__(self, port: str, baud: int = 1000000,
                 angle_read_timeout_s: float = 0.05) -> None:
        try:
            from pymycobot.common import ProtocolCode
            from pymycobot.mycobot import MyCobot
        except ImportError as exc:
            raise RuntimeError(
                "pymycobot is required for --backend real; install requirements-jetson.txt"
            ) from exc
        self._robot = MyCobot(port, baud)
        # pymycobot 4.0.7 waits up to 0.5 s per read attempt on Linux even
        # though MyCobot's serial timeout defaults to 0.1 s.  Bound only the
        # GET_ANGLES read; other commands retain the library's timing rules.
        original_read = self._robot._read

        def bounded_read(genre, *args, **kwargs):
            if genre == ProtocolCode.GET_ANGLES:
                kwargs.setdefault("timeout", angle_read_timeout_s)
            return original_read(genre, *args, **kwargs)

        self._robot._read = bounded_read

    def get_angles(self) -> list[float]:
        values = self._robot.get_angles()
        if not isinstance(values, (list, tuple)) or len(values) != 6:
            raise RuntimeError(f"Invalid get_angles() response: {values!r}")
        angles = [float(value) for value in values]
        if not all(math.isfinite(value) for value in angles):
            raise RuntimeError(f"Non-finite get_angles() response: {values!r}")
        return angles

    def close(self) -> None:
        self._robot._serial_port.close()

    def send_angles(self, angles: Sequence[float], speed: int) -> None:
        self._robot.send_angles(list(angles), speed)

    def send_angle(self, joint: int, angle: float, speed: int) -> None:
        self._robot.send_angle(joint, angle, speed)

    def stop(self) -> None:
        self._robot.stop()


@dataclass
class MockRobot:
    """Second-order delayed robot used only to test the local pipeline."""

    omega_n: float = 5.0
    zeta: float = 0.8
    delay_s: float = 0.08
    read_latency_s: float = 0.004
    noise_std_deg: float = 0.03

    def __post_init__(self) -> None:
        self._q = [0.0] * 6
        self._dq = [0.0] * 6
        self._target = [0.0] * 6
        self._pending: list[tuple[float, list[float]]] = []
        self._last = time.perf_counter()

    def _integrate(self, dt_total: float) -> None:
        if dt_total <= 0.0:
            return
        steps = max(1, math.ceil(dt_total / 0.002))
        dt = dt_total / steps
        for _ in range(steps):
            for index in range(6):
                ddq = self.omega_n**2 * (self._target[index] - self._q[index])
                ddq -= 2.0 * self.zeta * self.omega_n * self._dq[index]
                self._dq[index] += ddq * dt
                self._q[index] += self._dq[index] * dt

    def _advance(self) -> None:
        now = time.perf_counter()
        cursor = self._last
        # Integrate exactly to each delayed-command boundary, change the target,
        # and only then integrate the remaining interval.
        while self._pending and self._pending[0][0] <= now:
            apply_at, target = self._pending.pop(0)
            self._integrate(max(0.0, apply_at - cursor))
            cursor = max(cursor, apply_at)
            self._target = target
        self._integrate(max(0.0, now - cursor))
        self._last = now

    def get_angles(self) -> list[float]:
        self._advance()
        time.sleep(self.read_latency_s)
        self._advance()
        return [q + random.gauss(0.0, self.noise_std_deg) for q in self._q]

    def send_angles(self, angles: Sequence[float], speed: int) -> None:
        del speed
        self._advance()
        self._pending.append((time.perf_counter() + self.delay_s, list(map(float, angles))))

    def send_angle(self, joint: int, angle: float, speed: int) -> None:
        del speed
        self._advance()
        target = list(self._target)
        target[joint - 1] = float(angle)
        self._pending.append((time.perf_counter() + self.delay_s, target))

    def stop(self) -> None:
        self._advance()
        self._pending.clear()
        self._target = list(self._q)


def create_robot(backend: str, port: str | None, baud: int) -> Robot:
    if backend == "mock":
        return MockRobot()
    if not port:
        raise ValueError("--port is required for --backend real")
    return MyCobotRobot(port, baud)
