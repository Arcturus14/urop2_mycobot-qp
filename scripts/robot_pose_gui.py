"""Single-scene current/target pose GUI for myCobot 280 JN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Sequence

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
# Matplotlib 3.1 on the Jetson does not lazily register the 3-D projection.
from mpl_toolkits import mplot3d as _mplot3d  # noqa: F401
import numpy as np

# Support both ``python -m scripts.robot_pose_gui`` and direct file execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mycobot_id.kinematics import (
    JOINT_LIMITS_DEG, SPEED_LIMIT, ToolGeometry, forward_kinematics, validate_angles,
    validate_path,
)
from mycobot_id.robot import create_robot


class RobotWorker(threading.Thread):
    def __init__(self, backend: str, port: str | None, baud: int) -> None:
        super().__init__(daemon=True)
        self.backend, self.port, self.baud = backend, port, baud
        self.commands: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.events: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.stop_event = threading.Event()

    def run(self) -> None:
        robot = None
        try:
            robot = create_robot(self.backend, self.port, self.baud)
            self.events.put(("status", "로봇 연결됨" if self.backend == "real" else "Mock 연결됨"))
            next_read = 0.0
            while not self.stop_event.is_set():
                try:
                    name, payload = self.commands.get(timeout=0.01)
                    if name == "move":
                        angles, speed = payload
                        robot.send_angles(angles, speed)
                        self.events.put(("status", f"이동 명령 전송: 속도 {speed}"))
                    elif name == "stop" and hasattr(robot, "stop"):
                        robot.stop()
                        self.events.put(("status", "정지 명령 전송"))
                except queue.Empty:
                    pass
                now = time.monotonic()
                if now >= next_read:
                    self.events.put(("angles", robot.get_angles()))
                    next_read = now + 0.10
        except Exception as exc:
            self.events.put(("error", f"로봇 통신 오류: {exc}"))
        finally:
            if robot is not None and hasattr(robot, "close"):
                try:
                    robot.close()
                except Exception:
                    pass


class PoseGUI:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root, self.args = root, args
        self.tool: ToolGeometry = args.tool
        self.current = np.zeros(6)
        self.have_current = False
        self.target_vars = [tk.DoubleVar(value=0.0) for _ in range(6)]
        self.current_vars = [tk.StringVar(value="--") for _ in range(6)]
        self.speed_var = tk.IntVar(value=20)
        self.status_var = tk.StringVar(value="연결 중…")
        self.path_ok = False

        root.title("myCobot 280 JN 자세 확인 및 이동")
        root.geometry("1280x760")
        root.minsize(980, 620)
        self._build()
        self.worker = RobotWorker(args.backend, args.port, args.baud)
        self.worker.start()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self.poll)
        self.preview()

    def _build(self) -> None:
        outer = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True)
        plot_frame, controls = ttk.Frame(outer), ttk.Frame(outer, padding=12)
        outer.add(plot_frame, weight=4)
        outer.add(controls, weight=1)

        self.figure = Figure(figsize=(8, 7), dpi=100)
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame).update()

        ttk.Label(controls, text="관절 목표", font=("TkDefaultFont", 13, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        ttk.Label(controls, text="관절").grid(row=1, column=0)
        ttk.Label(controls, text="현재(°)").grid(row=1, column=1)
        ttk.Label(controls, text="목표(°)").grid(row=1, column=2)
        for i, ((low, high), target, current) in enumerate(
            zip(JOINT_LIMITS_DEG, self.target_vars, self.current_vars), 2
        ):
            ttk.Label(controls, text=f"J{i - 1}").grid(row=i, column=0, pady=3)
            ttk.Label(controls, textvariable=current, width=9, anchor="e").grid(row=i, column=1)
            spin = ttk.Spinbox(
                controls, from_=low, to=high, increment=1.0, textvariable=target, width=9,
                command=self.preview,
            )
            spin.grid(row=i, column=2, padx=(6, 0))
            spin.bind("<KeyRelease>", lambda _event: self.preview())

        row = 8
        ttk.Separator(controls).grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Label(controls, text="속도 (1~100)").grid(row=row + 1, column=0, columnspan=2, sticky="w")
        ttk.Spinbox(
            controls, from_=SPEED_LIMIT[0], to=SPEED_LIMIT[1], textvariable=self.speed_var, width=9
        ).grid(row=row + 1, column=2)
        ttk.Button(controls, text="현재값을 목표로", command=self.copy_current).grid(
            row=row + 2, column=0, columnspan=3, sticky="ew", pady=(12, 4)
        )
        ttk.Button(controls, text="목표 자세 미리보기", command=self.preview).grid(
            row=row + 3, column=0, columnspan=3, sticky="ew", pady=4
        )
        self.move_button = ttk.Button(controls, text="이동", command=self.move, state=tk.DISABLED)
        self.move_button.grid(row=row + 4, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Button(controls, text="정지", command=self.stop).grid(
            row=row + 5, column=0, columnspan=3, sticky="ew", pady=4
        )
        ttk.Label(
            controls, textvariable=self.status_var, wraplength=245, foreground="#333333"
        ).grid(row=row + 6, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        ttk.Label(
            controls,
            text="파랑: 현재 자세\n주황 점선: 목표 자세\n충돌 검사는 캡슐 근사 모델입니다.",
            foreground="#555555",
        ).grid(row=row + 7, column=0, columnspan=3, sticky="w", pady=(14, 0))
        controls.columnconfigure(1, weight=1)

    def target(self) -> list[float] | None:
        try:
            return [float(item.get()) for item in self.target_vars]
        except (ValueError, tk.TclError):
            self.status_var.set("목표 관절값을 숫자로 입력하세요.")
            return None

    def preview(self) -> None:
        target = self.target()
        if target is None:
            self.path_ok = False
            self._update_move_state()
            return
        limits = validate_angles(target)
        if not limits.ok:
            self.path_ok = False
            self.status_var.set(limits.message)
        elif self.have_current:
            path = validate_path(self.current, target, tool=self.tool)
            self.path_ok = path.ok
            self.status_var.set(path.message)
        else:
            self.path_ok = False
            self.status_var.set("현재 관절값을 기다리는 중입니다.")
        self.draw(target)
        self._update_move_state()

    def draw(self, target: Sequence[float]) -> None:
        elevation, azimuth = self.axis.elev, self.axis.azim
        self.axis.clear()
        current_geometry = forward_kinematics(self.current, self.tool)
        target_geometry = forward_kinematics(target, self.tool)
        for geometry, color, style, label, alpha in (
            (current_geometry, "#1769aa", "-", "Current", 1.0),
            (target_geometry, "#ef6c00", "--", "Target", 0.75),
        ):
            p = geometry.joints
            self.axis.plot(p[:, 0], p[:, 1], p[:, 2], style, color=color, lw=5, alpha=alpha, label=label)
            self.axis.scatter(p[:, 0], p[:, 1], p[:, 2], color=color, s=32, alpha=alpha)
            if len(geometry.gripper):
                self.axis.plot(*geometry.gripper.T, color=color, lw=8, alpha=alpha)
            if len(geometry.camera):
                self.axis.plot(*geometry.camera.T, color="#6a1b9a", lw=7, alpha=alpha)
        self.axis.set(xlim=(-0.32, 0.32), ylim=(-0.32, 0.32), zlim=(-0.10, 0.48),
                      xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
        if hasattr(self.axis, "set_box_aspect"):
            self.axis.set_box_aspect((1.0, 1.0, 0.9))
        self.axis.view_init(elev=elevation, azim=azimuth)
        self.axis.legend(loc="upper left")
        self.axis.set_title("Current and target pose")
        self.canvas.draw_idle()

    def _update_move_state(self) -> None:
        enabled = self.path_ok and (self.args.backend == "mock" or self.args.enable_motion)
        self.move_button.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def copy_current(self) -> None:
        if not self.have_current:
            return
        for variable, value in zip(self.target_vars, self.current):
            variable.set(round(float(value), 2))
        self.preview()

    def move(self) -> None:
        target = self.target()
        if target is None or not self.have_current:
            return
        result = validate_path(self.current, target, tool=self.tool)
        if not result.ok:
            self.status_var.set(result.message)
            return
        try:
            speed = int(self.speed_var.get())
        except (ValueError, tk.TclError):
            self.status_var.set("속도는 1~100 정수여야 합니다.")
            return
        if not SPEED_LIMIT[0] <= speed <= SPEED_LIMIT[1]:
            self.status_var.set("속도는 1~100 범위여야 합니다.")
            return
        if self.args.backend == "real" and not messagebox.askyesno(
            "실제 로봇 이동 확인", f"목표 {np.round(target, 1).tolist()}°\n속도 {speed}\n\n이동할까요?"
        ):
            return
        self.worker.commands.put(("move", (target, speed)))

    def stop(self) -> None:
        self.worker.commands.put(("stop", None))

    def poll(self) -> None:
        latest_angles = None
        try:
            while True:
                name, payload = self.worker.events.get_nowait()
                if name == "angles":
                    latest_angles = np.asarray(payload, dtype=float)
                else:
                    self.status_var.set(str(payload))
                    if name == "error":
                        self.path_ok = False
        except queue.Empty:
            pass
        if latest_angles is not None:
            self.current = latest_angles
            self.have_current = True
            for variable, value in zip(self.current_vars, self.current):
                variable.set(f"{value:.2f}")
            self.preview()
        self.root.after(50, self.poll)

    def close(self) -> None:
        self.worker.stop_event.set()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mock", "real"), default="mock")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "robot_pose_gui.json")
    )
    parser.add_argument(
        "--enable-motion", action="store_true",
        help="allow the Move button for the real backend after validation and confirmation",
    )
    args = parser.parse_args()
    if args.backend == "real" and not args.port:
        parser.error("--port is required with --backend real")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    args.tool = ToolGeometry(**config.get("tool", {}))
    root = tk.Tk()
    PoseGUI(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()

