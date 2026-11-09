"""Laptop GUI with a one-shot SSH transaction to a myCobot 280 JN."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Iterable, Iterator, Sequence

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits import mplot3d as _mplot3d  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mycobot_id.kinematics import (
    JOINT_LIMITS_DEG,
    SPEED_LIMIT,
    ToolGeometry,
    forward_kinematics,
    validate_angles,
    validate_path,
)

DEFAULT_TARGET_DEG = (0.0, 0.0, 0.0, 0.0, 90.0, -45.0)


def _bridge_events(lines: Iterable[str]) -> Iterator[dict]:
    for line in lines:
        line = line.strip()
        if not line or line.startswith("Note: This class is no longer maintained since v3.6.0"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Jetson 브리지의 JSON이 아닌 응답: {line[:250]}") from exc
        if not isinstance(event, dict):
            raise RuntimeError(f"Jetson 브리지의 잘못된 응답: {line[:250]}")
        yield event


class RemoteTransaction(threading.Thread):
    """Keep one SSH process open between current-pose check and approval."""

    def __init__(self, args: argparse.Namespace, target: Sequence[float], speed: int) -> None:
        super().__init__(daemon=True)
        self.args = args
        self.target = tuple(float(value) for value in target)
        self.speed = speed
        self.events: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.decision: "queue.Queue[str]" = queue.Queue(maxsize=1)
        self.process: subprocess.Popen[str] | None = None
        self.motion_trace_supported = False
        self.j4_correction_supported = False
        self.final_only_supported = False
        self.joint_limits_test_supported = False

    def _command(self) -> list[str]:
        return [
            "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=yes", self.args.ssh_host,
            self.args.remote_python, self.args.remote_bridge, "transaction",
            "--port", self.args.port, "--baud", str(self.args.baud),
            "--target", *(format(value, ".6f") for value in self.target),
            "--speed", str(self.speed),
        ]

    def run(self) -> None:
        try:
            self.process = subprocess.Popen(
                self._command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1,
            )
            assert self.process.stdout is not None and self.process.stdin is not None
            responses = _bridge_events(self.process.stdout)
            first = next(responses, None)
            if first is None:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(stderr.strip() or "Jetson bridge returned no data")
            if first.get("event") == "error":
                raise RuntimeError(str(first.get("message", "unknown bridge error")))
            if first.get("event") != "current":
                raise RuntimeError("unexpected bridge response: {!r}".format(first))
            self.motion_trace_supported = first.get("motion_trace_supported") is True
            self.j4_correction_supported = first.get("j4_correction_supported") is True
            self.final_only_supported = first.get("final_only_supported") is True
            self.joint_limits_test_supported = first.get("joint_limits_test_supported") is True
            self.events.put(("current", first))
            decision = self.decision.get()
            self.process.stdin.write(decision + "\n")
            self.process.stdin.flush()
            if decision == "ABORT":
                self.process.wait(timeout=5)
                self.events.put(("aborted", "경로 검사 세션 취소"))
                return
            correction_allowed = json.loads(decision).get("correct_j4") is True
            for response in responses:
                event = str(response.get("event"))
                if event == "final":
                    self.events.put(("final", response))
                elif event == "correction_proposed":
                    if not correction_allowed:
                        self.process.stdin.write("ABORT\n")
                        self.process.stdin.flush()
                        raise RuntimeError("unexpected J4 correction proposal")
                    correction_allowed = False
                    self.events.put((event, response))
                    check = validate_path(response["start_angles_deg"], response["command_angles_deg"],
                                          step_deg=0.25, tool=self.args.tool)
                    self.process.stdin.write("CORRECT\n" if check.ok else "ABORT\n")
                    self.process.stdin.flush()
                    if not check.ok:
                        self.events.put(("correction_rejected", check.message))
                elif event in ("gripper_sent", "gripper_final", "motion_trace", "correction_result", "done"):
                    self.events.put((event, response))
                elif event == "error":
                    raise RuntimeError(str(response.get("message", "unknown bridge error")))
            return_code = self.process.wait(timeout=5)
            if return_code:
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(stderr.strip() or f"SSH bridge exited with {return_code}")
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def proceed(self, grip: dict | None = None, record_motion: bool = False,
                correct_j4: bool = False, final_only: bool = False, joint_limits_test: bool = False) -> None:
        if self.decision.empty():
            self.decision.put(json.dumps({"action": "PROCEED", "grip": grip, "record_motion": record_motion,
                                          "correct_j4": correct_j4, "final_only": final_only,
                                          "joint_limits_test": joint_limits_test}))

    def abort(self) -> None:
        if self.decision.empty():
            self.decision.put("ABORT")

    def request_stop(self) -> None:
        if self.process and self.process.stdin and self.process.poll() is None:
            try:
                self.process.stdin.write("STOP\n")
                self.process.stdin.flush()
            except (OSError, ValueError):
                pass


class LaptopPoseGUI:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root, self.args = root, args
        self.tool: ToolGeometry = args.tool
        self.floor_calibration = getattr(args, "floor_calibration", {})
        self.current = np.zeros(6)
        self.have_current = False
        self.transaction: RemoteTransaction | None = None
        self.checked_target: tuple[float, ...] | None = None
        self.target_vars = [tk.DoubleVar(value=value) for value in DEFAULT_TARGET_DEG]
        self.current_vars = [tk.StringVar(value="--") for _ in range(6)]
        self.speed_var = tk.IntVar(value=20)
        self.grip_value = tk.IntVar(value=100)
        self.grip_speed = tk.IntVar(value=20)
        self.grasp_after = tk.BooleanVar(value=False)
        self.record_motion = tk.BooleanVar(value=False)
        self.correct_j4 = tk.BooleanVar(value=False)
        self.final_only = tk.BooleanVar(value=False)
        self.joint_limits_test = tk.BooleanVar(value=False)
        self.grasp_value = tk.IntVar(value=0)
        self.grip_status = tk.StringVar(value="그리퍼: 미확인 (0 닫힘 / 100 열림)")
        self.manual_events = queue.Queue()
        self.manual_busy = False
        self.moving = False
        self.ik_busy = False
        self.ik_events = queue.Queue()
        self.ee_solution = None
        self.ee_input_vars = [tk.StringVar(value=value) for value in ("240", "80", "30", "0")]
        self.ee_status = tk.StringVar(value="테이블 기준 닫힌 손끝 중점 입력 (mm)")
        self.status_var = tk.StringVar(value="목표값 입력 후 '현재 읽기 + 경로 검사'를 누르세요.")
        log_dir = PROJECT_ROOT / "reports" / "gui"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / (datetime.now().strftime("pose_%Y%m%d_%H%M%S_%f") + ".jsonl")

        root.title("myCobot 280 JN 노트북 자세 확인 및 이동")
        root.geometry("1280x800")
        root.minsize(980, 620)
        self._build()
        for variable in self.ee_input_vars:
            variable.trace_add("write", self.ee_changed)
        self.log_message("시작", {"log_file": str(self.log_path)})
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self.poll)
        self.preview()

    def _build(self) -> None:
        outer = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True)
        plot_frame = ttk.Frame(outer)
        control_tabs = ttk.Notebook(outer)
        controls = ttk.Frame(control_tabs, padding=12)
        gripper_controls = ttk.Frame(control_tabs, padding=12)
        control_tabs.add(controls, text="로봇팔")
        control_tabs.add(gripper_controls, text="그리퍼")
        ee_controls = ttk.Frame(control_tabs, padding=10)
        control_tabs.insert(1, ee_controls, text="EE 좌표")
        ttk.Label(ee_controls, text="테이블 기준 / 닫힌 손끝 중점\n방향: 수직 아래 고정", wraplength=300).grid(row=0, column=0, columnspan=2, sticky="w", pady=8)
        for row, (label, variable) in enumerate(zip(("X (mm)", "Y (mm)", "Z (mm)", "회전 방향 Yaw (°)"), self.ee_input_vars), 1):
            ttk.Label(ee_controls, text=label).grid(row=row, column=0, sticky="w", pady=5)
            ttk.Entry(ee_controls, textvariable=variable, width=12).grid(row=row, column=1, padx=8)
        ttk.Button(ee_controls, text="현재 손끝 좌표 가져오기", command=self.copy_current_ee).grid(row=5, column=0, columnspan=2, sticky="ew", pady=4)
        self.ik_button = ttk.Button(ee_controls, text="IK 계산 + 미리보기", command=self.solve_ee)
        self.ik_button.grid(row=6, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(ee_controls, text="현재 읽기 + 경로 검사", command=self.check_ee).grid(row=7, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(ee_controls, text="EE 위치로 이동", command=self.move_ee).grid(row=8, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(ee_controls, text="정지 / 검사 취소", command=self.stop).grid(row=9, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Label(ee_controls, textvariable=self.ee_status, wraplength=310).grid(row=10, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Label(ee_controls, textvariable=self.status_var, wraplength=310).grid(row=11, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Checkbutton(ee_controls, text="이동 관절각 기록 (정지 후 5초 포함)",
                        variable=self.record_motion).grid(row=12, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(ee_controls, text="목표 1회 전송 (종료 각도만 확인)",
                        variable=self.final_only).grid(row=13, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(ee_controls, text="동작 확인 (관절 한계 감시)",
                        variable=self.joint_limits_test).grid(row=14, column=0, columnspan=2, sticky="w")
        workspace_text = "테이블 Z=0에서의 수치 탐색 범위\n범위 파일 없음: IK로 개별 판정"
        workspace_file = PROJECT_ROOT / "reports/floor_workspace.json"
        if workspace_file.exists():
            bounds = json.loads(workspace_file.read_text(encoding="utf-8"))["results"]
            workspace_text = (
                "테이블 Z=0 / 아래 방향 / Yaw 자유\n"
                f"X: {bounds['x_min']['xyz_mm'][0]:.1f} ~ {bounds['x_max']['xyz_mm'][0]:.1f} mm\n"
                f"Y: {bounds['y_min']['xyz_mm'][1]:.1f} ~ {bounds['y_max']['xyz_mm'][1]:.1f} mm\n"
                f"반경: {bounds['radius_min']['radius_mm']:.1f} ~ {bounds['radius_max']['radius_mm']:.1f} mm\n"
                "직사각형 전체가 도달 가능한 것은 아닙니다.\nZ≠0 또는 Yaw 고정 시 범위가 달라집니다.\n"
                "케이스·테이블 충돌은 미검증입니다.\n손끝 목표는 완전 닫힘 기준 추정값입니다.")
        ttk.Label(ee_controls, text=workspace_text, wraplength=310, foreground="#555555").grid(row=15, column=0, columnspan=2, sticky="w", pady=8)
        logs = ttk.Frame(control_tabs, padding=6)
        control_tabs.add(logs, text="좌표값")
        self.pose_values = {}
        for key, title, axes in (
            ("robot", "로봇 좌표 (설정된 기준/TCP)", ("X (mm)", "Y (mm)", "Z (mm)", "Rx (°)", "Ry (°)", "Rz (°)")),
            ("model", "모델 플랜지 (테이블 Z 보정)" if self.floor_calibration.get("enabled") else "모델 플랜지 좌표", ("X (mm)", "Y (mm)", "Z (mm)")),
            ("tip", "손끝 중점 추정 (완전 닫힘 가정)", ("X (mm)", "Y (mm)", "Z (mm)")),
        ):
            frame = ttk.LabelFrame(logs, text=title, padding=8)
            frame.pack(fill=tk.X, pady=(0, 12))
            for col, header in enumerate(("좌표", "이동 전", "이동 후")):
                ttk.Label(frame, text=header).grid(row=0, column=col, padx=8, pady=5)
            self.pose_values[key] = {}
            for phase in ("before", "after"):
                self.pose_values[key][phase] = [tk.StringVar(value="—") for _ in axes]
            for row, axis in enumerate(axes, 1):
                ttk.Label(frame, text=axis).grid(row=row, column=0, sticky="w", pady=4)
                for col, phase in enumerate(("before", "after"), 1):
                    ttk.Label(frame, textvariable=self.pose_values[key][phase][row-1],
                              width=9, anchor="e").grid(row=row, column=col, padx=8, pady=4)
        self.coords_status = tk.StringVar(value="이동 전: 경로 검사 시 갱신")
        ttk.Label(logs, textvariable=self.coords_status, wraplength=300).pack(anchor="w")
        if self.floor_calibration.get("enabled"):
            ttk.Label(logs, text="테이블 Z=0: 단일 접촉 잠정 보정\nXY·기울기는 미보정. 손끝값은 완전 닫힘 가정입니다.",
                      wraplength=300).pack(anchor="w", pady=6)
        outer.add(plot_frame, weight=4)
        outer.add(control_tabs, weight=1)
        self.figure = Figure(figsize=(8, 7), dpi=100)
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame).update()

        ttk.Label(controls, text="관절 목표", font=("TkDefaultFont", 13, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        for column, text in enumerate(("관절", "현재(°)", "목표(°)")):
            ttk.Label(controls, text=text).grid(row=1, column=column)
        for row, ((low, high), target, current) in enumerate(
            zip(JOINT_LIMITS_DEG, self.target_vars, self.current_vars), 2
        ):
            ttk.Label(controls, text=f"J{row - 1}").grid(row=row, column=0, pady=3)
            ttk.Label(controls, textvariable=current, width=9, anchor="e").grid(row=row, column=1)
            spin = ttk.Spinbox(
                controls, from_=low, to=high, increment=1.0, textvariable=target,
                width=9, command=self.target_changed,
            )
            spin.grid(row=row, column=2, padx=(6, 0))
            spin.bind("<KeyRelease>", self.target_changed)

        row = 8
        ttk.Separator(controls).grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Label(controls, text="속도 (1~100)").grid(row=row + 1, column=0, columnspan=2, sticky="w")
        ttk.Spinbox(
            controls, from_=1, to=100, textvariable=self.speed_var, width=9,
            command=self.target_changed,
        ).grid(row=row + 1, column=2)
        presets = ttk.Frame(controls)
        presets.grid(row=row + 2, column=0, columnspan=3, sticky="ew", pady=(12, 4))
        ttk.Button(presets, text="현재값을 목표로", command=self.copy_current).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(presets, text="전체 목표 초기화", command=self.reset_targets).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        ttk.Button(controls, text="목표 자세 미리보기", command=self.preview).grid(
            row=row + 3, column=0, columnspan=3, sticky="ew", pady=4
        )
        self.check_button = ttk.Button(
            controls, text="현재 읽기 + 경로 검사", command=self.check_path
        )
        self.check_button.grid(row=row + 4, column=0, columnspan=3, sticky="ew", pady=4)
        self.move_button = ttk.Button(controls, text="이동", command=self.move, state=tk.DISABLED)
        self.move_button.grid(row=row + 5, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Button(controls, text="긴급 정지", command=self.stop).grid(
            row=row + 6, column=0, columnspan=3, sticky="ew", pady=4
        )
        ttk.Label(controls, textvariable=self.status_var, wraplength=260).grid(
            row=row + 7, column=0, columnspan=3, sticky="ew", pady=(12, 0)
        )
        ttk.Checkbutton(controls, text="이동 관절각 기록 (정지 후 5초 포함)",
                        variable=self.record_motion).grid(row=row + 8, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(controls, text="J4 자동 미세 보정 1회 (시험)",
                        variable=self.correct_j4).grid(row=row + 9, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(controls, text="목표 1회 전송 (종료 각도만 확인)",
                        variable=self.final_only).grid(row=row + 10, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(controls, text="동작 확인 (관절 한계 감시)",
                        variable=self.joint_limits_test).grid(row=row + 11, column=0, columnspan=3, sticky="w")
        ttk.Label(
            controls, wraplength=260, foreground="#555555",
            text="카메라: 공식 CAD 외곽·표준 장착\n그리퍼: 전체 개방 근사 외형\n초록: 그리퍼 / 보라: 카메라\n케이블과 잡은 물체는 검사에서 제외됩니다.",
        ).grid(row=row + 12, column=0, columnspan=3, sticky="w", pady=8)
        grip_frame = ttk.LabelFrame(gripper_controls, text="그리퍼", padding=6)
        grip_frame.pack(fill=tk.X)
        ttk.Label(grip_frame, text="개방값 (0~100)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(grip_frame, from_=0, to=100, width=7, textvariable=self.grip_value).grid(row=0, column=1)
        ttk.Label(grip_frame, text="속도 (1~100)").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(grip_frame, from_=1, to=100, width=7, textvariable=self.grip_speed).grid(row=1, column=1)
        ttk.Button(grip_frame, text="열기", command=lambda: self.manual_grip(100)).grid(row=2, column=0, sticky="ew")
        ttk.Button(grip_frame, text="닫기", command=lambda: self.manual_grip(0)).grid(row=2, column=1, sticky="ew")
        ttk.Button(grip_frame, text="개방값 적용", command=self.manual_grip).grid(row=3, column=0, columnspan=2, sticky="ew")
        ttk.Checkbutton(grip_frame, text="목표 도착 후 잡기", variable=self.grasp_after).grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Label(grip_frame, text="잡기 개방값").grid(row=5, column=0, sticky="w")
        ttk.Spinbox(grip_frame, from_=0, to=100, width=7, textvariable=self.grasp_value).grid(row=5, column=1)
        ttk.Label(grip_frame, textvariable=self.grip_status, wraplength=260).grid(row=6, column=0, columnspan=2)
        controls.columnconfigure(1, weight=1)

    def target(self) -> list[float] | None:
        try:
            return [float(item.get()) for item in self.target_vars]
        except (ValueError, tk.TclError):
            self.status_var.set("목표 관절값을 숫자로 입력하세요.")
            return None

    def speed(self) -> int | None:
        try:
            value = int(self.speed_var.get())
        except (ValueError, tk.TclError):
            self.status_var.set("속도는 1~100 정수여야 합니다.")
            return None
        if not SPEED_LIMIT[0] <= value <= SPEED_LIMIT[1]:
            self.status_var.set("속도는 1~100 범위여야 합니다.")
            return None
        return value

    def target_changed(self, _event: object = None) -> None:
        if self.moving:
            return
        self.ee_solution = None
        target = self.target()
        if self.transaction is not None and target is not None:
            if tuple(target) != self.checked_target:
                self.transaction.abort()
                self.transaction = None
                self.checked_target = None
                self.move_button.configure(state=tk.DISABLED)
                self.check_button.configure(state=tk.NORMAL)
                self.status_var.set("목표가 변경되어 경로 검사를 다시 해야 합니다.")
        self.preview()

    def preview(self) -> None:
        target = self.target()
        if target is None:
            return
        limits = validate_angles(target)
        if not limits.ok:
            self.status_var.set(limits.message)
        self.draw(target)

    def draw(self, target: Sequence[float]) -> None:
        elevation, azimuth = self.axis.elev, self.axis.azim
        self.axis.clear()
        display_shift = np.array([0.0, 0.0, self.floor_calibration.get("model_to_table_z_offset_m", 0.0)
                                  if self.floor_calibration.get("enabled") else 0.0])
        geometries = [(forward_kinematics(target, self.tool), "#ef6c00", "--", "Target", 0.78)]
        if self.have_current:
            geometries.insert(
                0, (forward_kinematics(self.current, self.tool), "#1769aa", "-", "Current", 1.0)
            )
        for geometry, color, style, label, alpha in geometries:
            points = geometry.joints + display_shift
            self.axis.plot(*points.T, style, color=color, lw=5, alpha=alpha, label=label)
            self.axis.scatter(*points.T, color=color, s=32, alpha=alpha)
            for box in geometry.tool_boxes:
                corners = box.corners() + display_shift
                faces = [corners[list(indices)] for indices in (
                    (0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
                    (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5),
                )]
                box_color = "#2e7d32" if box.name == "gripper" else "#7b1fa2"
                self.axis.add_collection3d(Poly3DCollection(
                    faces, facecolors=box_color, edgecolors=color,
                    alpha=0.20 * alpha, linewidths=1.0, linestyles=style,
                ))
                self.axis.text(*(box.transform[:3, 3] + display_shift), f"{label} {box.name}", fontsize=8)
            if self.floor_calibration.get("enabled"):
                tcp = np.r_[self.floor_calibration["closed_tip_midpoint_flange_m"], 1.]
                point = (geometry.transforms[-1] @ tcp)[:3] + display_shift
                self.axis.scatter(*point, color=color, marker="x", s=80)
                self.axis.text(*point, f"{label} TCP", fontsize=8)
        self.axis.set(
            xlim=(-0.48, 0.48), ylim=(-0.48, 0.48), zlim=(-0.16, 0.65),
            xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
        )
        if hasattr(self.axis, "set_box_aspect"):
            self.axis.set_box_aspect((1.0, 1.0, 0.9))
        self.axis.view_init(elev=elevation, azim=azimuth)
        self.axis.legend(loc="upper left")
        self.axis.set_title("myCobot 280 JN - Modified DH")
        if self.floor_calibration.get("enabled"):
            self.axis.set_title("Table Z reference (single-contact calibration)")
            floor_z = 0.0
            self.axis.add_collection3d(Poly3DCollection(
                [[(-.48,-.48,floor_z),(.48,-.48,floor_z),(.48,.48,floor_z),(-.48,.48,floor_z)]],
                facecolors="gray", alpha=.08, edgecolors="gray"))
        self.canvas.draw_idle()

    def check_path(self) -> None:
        if self.moving or self.manual_busy or self.ik_busy:
            return
        if not self.args.enable_motion:
            self.status_var.set("실제 검사를 활성화하려면 --enable-motion으로 실행하세요.")
            return
        target, speed = self.target(), self.speed()
        if target is None or speed is None:
            return
        limits = validate_angles(target)
        if not limits.ok:
            self.status_var.set(limits.message)
            return
        if self.transaction is not None:
            self.transaction.abort()
        self.checked_target = tuple(target)
        self.transaction = RemoteTransaction(self.args, target, speed)
        self.transaction.start()
        self.check_button.configure(state=tk.DISABLED)
        self.move_button.configure(state=tk.DISABLED)
        self.status_var.set("Jetson에서 현재 관절값을 1회 읽는 중…")

    def move(self) -> None:
        if self.moving or self.ik_busy or self.manual_busy:
            return
        if self.transaction is None or self.checked_target is None:
            return
        if self.target() != list(self.checked_target):
            self.target_changed()
            return
        final_only = self.final_only.get()
        joint_limits_test = self.joint_limits_test.get()
        if joint_limits_test and not self.transaction.joint_limits_test_supported:
            self.status_var.set("동작 확인 모드를 지원하는 Jetson 브리지로 업데이트하세요.")
            return
        if joint_limits_test and (final_only or self.correct_j4.get() or self.grasp_after.get()):
            self.status_var.set("동작 확인 모드에서는 목표 1회 전송 옵션, J4 보정, 파지를 꺼 주세요.")
            return
        if final_only and not self.transaction.final_only_supported:
            self.status_var.set("목표 1회 전송을 지원하는 Jetson 브리지로 업데이트하세요.")
            return
        correct_j4 = self.correct_j4.get() and not final_only
        if correct_j4 and not self.transaction.j4_correction_supported:
            self.status_var.set("J4 자동 보정을 지원하는 Jetson 브리지로 업데이트하세요.")
            return
        if correct_j4 and (self.grasp_after.get() or self.transaction.speed > 10
                           or any(abs(a - b) > 0.25 for i, (a, b) in enumerate(zip(self.current, self.checked_target)) if i != 3)
                           or abs(self.current[3] - self.checked_target[3]) > 5.0):
            self.status_var.set("J4만 5° 이내 변경, 속도 10 이하, 파지 끔으로 시험하세요.")
            return
        record_motion = (self.record_motion.get() or correct_j4) and not final_only
        if record_motion and not self.transaction.motion_trace_supported:
            self.status_var.set("관절각 기록을 지원하는 Jetson 브리지로 업데이트하세요.")
            return
        grip = None
        if self.grasp_after.get():
            try:
                grip = self.grip_parameters(self.grasp_value.get())
            except tk.TclError:
                self.grip_status.set("잡기 개방값은 0~100 정수여야 합니다.")
                return
            if grip is None:
                return
        if not messagebox.askyesno(
            "실제 로봇 이동 확인",
            f"목표 {np.round(self.checked_target, 1).tolist()}°\n속도 {self.transaction.speed}\n"
            + (f"도착 후 그리퍼: {grip['value']} (속도 {grip['speed']})\n" if grip else "")
            + ("관절각 기록: 이동 중 및 정지 후 5초\n" if record_motion else "")
            + ("필요 시 경로 검사 후 J4 미세 보정을 한 번 실행합니다.\n" if correct_j4 else "")
            + ("목표 1회 전송: 이동 중 경로 감시, 기록 및 J4 보정 없이 종료 각도만 확인합니다.\n"
               if final_only else "")
            + ("동작 확인: 이동 중 관절 한계 감시, 경로·유지 편차 판정 해제.\n" if joint_limits_test else "")
            + "\n이동할까요?",
        ):
            return
        self.move_button.configure(state=tk.DISABLED)
        self.check_button.configure(state=tk.DISABLED)
        self.status_var.set("이동 중… 종료 각도 확인 대기" if final_only else
                            "이동 중… Jetson 관절 피드백 확인 중")
        self.moving = True
        self.log_message("이동 명령", {"target_angles_deg": list(self.checked_target), "speed": self.transaction.speed,
                                        "grip_after": grip, "record_motion": record_motion, "correct_j4": correct_j4,
                                        "final_only": final_only, "joint_limits_test": joint_limits_test})
        self.transaction.proceed(grip, record_motion, correct_j4, final_only, joint_limits_test)

    def show_motion_trace(self, payload: dict) -> None:
        self.log_message("이동 관절각 기록", payload)
        samples = payload["samples"]
        times = [sample["t_s"] for sample in samples]
        angles = np.array([sample["angles_deg"] if sample["angles_deg"] is not None else [np.nan] * 6
                           for sample in samples])
        holding = next((sample["t_s"] for sample in samples if sample["phase"] == "holding"), None)
        correction = next((sample["t_s"] for sample in samples if sample["phase"] == "correction_before"), None)
        window = tk.Toplevel(self.root)
        window.title("이동 관절각 기록")
        figure = Figure(figsize=(9, 6), dpi=100, constrained_layout=True)
        axes = figure.subplots(3, 2, sharex=True)
        for joint, axis in enumerate(axes.flat):
            axis.plot(times, angles[:, joint], ".-", markersize=3)
            axis.axhline(payload["target_angles_deg"][joint], color="tab:orange", linestyle="--")
            if holding is not None:
                axis.axvline(holding, color="gray", linestyle=":")
            if correction is not None:
                axis.axvline(correction, color="purple", linestyle=":")
            axis.set_ylabel(f"J{joint + 1} (°)")
            axis.grid(alpha=0.25)
        for axis in axes[-1]:
            axis.set_xlabel("Time (s)")
        ttk.Label(window, text="주황: 원래 목표각 / 회색: 첫 정지 / 보라: 보정 직전").pack()
        if payload.get("save_error"):
            ttk.Label(window, text="Jetson 파일 저장 오류: " + payload["save_error"]).pack()
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, window).update()
        canvas.draw()

    def poll(self) -> None:
        try:
            while True:
                request, result = self.ik_events.get_nowait()
                self.ik_busy = False
                self.ik_button.configure(state=tk.NORMAL)
                if request != tuple(v.get() for v in self.ee_input_vars):
                    self.ee_status.set("입력이 변경되어 계산 결과를 취소했습니다.")
                    continue
                self.ee_status.set(result.message)
                if result.ok:
                    self.cancel_pending_check()
                    for variable, value in zip(self.target_vars, result.angles_deg):
                        variable.set(value)
                    self.ee_solution = tuple(result.angles_deg)
                    self.preview()
        except queue.Empty:
            pass
        try:
            while True:
                ok, message = self.manual_events.get_nowait()
                self.manual_busy = False
                self.grip_status.set(message)
        except queue.Empty:
            pass
        transaction = self.transaction
        if transaction is not None:
            try:
                while True:
                    event, payload = transaction.events.get_nowait()
                    if event == "current":
                        self.log_pose("이동 전(경로 검사 시)", payload)
                        self.current = np.asarray(payload["angles"], dtype=float)
                        self.have_current = True
                        self._show_current()
                        result = validate_path(self.current, self.checked_target, tool=self.tool)
                        if result.ok:
                            self.status_var.set(result.message + " — 이동 버튼을 누르세요.")
                            self.move_button.configure(state=tk.NORMAL)
                        else:
                            transaction.abort()
                            self.status_var.set(result.message)
                            self.transaction = None
                            self.checked_target = None
                            self.check_button.configure(state=tk.NORMAL)
                    elif event == "final":
                        self.log_pose("이동 후", payload)
                        final = np.asarray(payload["angles"], dtype=float)
                        error = float(np.max(np.abs(final - np.asarray(self.checked_target))))
                        self.current = final
                        self.have_current = True
                        self._show_current()
                        self.status_var.set(f"최종 관절값 수신. 최대 오차 {error:.2f}°")
                    elif event == "gripper_sent":
                        self.grip_status.set(f"그리퍼 동작 중: 목표 {payload['requested']}")
                    elif event == "gripper_final":
                        self.log_message("그리퍼 결과", payload)
                        self.grip_status.set(f"그리퍼 정지: 목표 {payload['requested']}, 측정 {payload['actual']} (파지 여부 미확인)")
                    elif event == "motion_trace":
                        self.show_motion_trace(payload)
                    elif event in ("correction_proposed", "correction_result"):
                        self.log_message(event, payload)
                        if event == "correction_result" and payload.get("accepted") is False:
                            self.status_var.set("J4 보정 실패: 전체 관절 오차 또는 다른 관절의 편차를 확인하세요.")
                    elif event == "correction_rejected":
                        self.log_message(event, {"message": str(payload)})
                    elif event == "done":
                        label = "동작 확인 종료" if payload.get("joint_limits_test") else "동작 완료"
                        self.status_var.set(f"{label}. 최대 관절 오차 {payload['max_joint_error_deg']:.2f}°")
                        self.moving = False
                        self.transaction = None
                        self.checked_target = None
                        self.check_button.configure(state=tk.NORMAL)
                    elif event == "error":
                        self.log_message("오류", {"message": str(payload)})
                        self.moving = False
                        self.move_button.configure(state=tk.DISABLED)
                        label = "이동 오류" if str(payload).startswith((
                            "target not reached", "target pose did not hold", "joint path deviation",
                            "start pose changed", "motion cancelled", "motion or position hold timed out",
                            "J4 correction",
                            "motion timed out",
                            "joint limit exceeded", "joint limits test requires",
                        )) else "통신 오류"
                        self.status_var.set(label + ": " + str(payload))
                        self.transaction = None
                        self.checked_target = None
                        self.check_button.configure(state=tk.NORMAL)
                    elif event == "aborted":
                        self.check_button.configure(state=tk.NORMAL)
            except queue.Empty:
                pass
        self.root.after(50, self.poll)

    def cancel_pending_check(self):
        if self.transaction is not None and not self.moving:
            self.transaction.abort()
            self.transaction = None
            self.checked_target = None
        self.move_button.configure(state=tk.DISABLED)
        self.check_button.configure(state=tk.NORMAL)

    def ee_changed(self, *_args):
        self.ee_solution = None
        if not self.moving:
            self.cancel_pending_check()
        self.ee_status.set("좌표가 변경되었습니다. IK를 다시 계산하세요.")

    def copy_current_ee(self):
        if not self.have_current or not self.floor_calibration.get("enabled"):
            self.ee_status.set("먼저 현재 읽기 및 경로 검사를 수행하세요.")
            return
        t = forward_kinematics(self.current, self.tool).transforms[-1]
        p = (t @ np.r_[self.floor_calibration["closed_tip_midpoint_flange_m"], 1.])[:3] * 1000
        p[2] += self.floor_calibration["model_to_table_z_offset_m"] * 1000
        yaw = np.degrees(np.arctan2(t[1,0], t[0,0]))
        for variable, value in zip(self.ee_input_vars, (*p, yaw)):
            variable.set(f"{value:.3f}")

    def solve_ee(self):
        if self.moving or self.manual_busy or self.ik_busy:
            return
        try:
            request = tuple(v.get() for v in self.ee_input_vars)
            numbers = tuple(float(v) for v in request)
        except ValueError:
            self.ee_status.set("좌표와 방향을 숫자로 입력하세요.")
            return
        self.cancel_pending_check()
        self.ee_solution = None
        self.ik_busy = True
        self.ik_button.configure(state=tk.DISABLED)
        self.ee_status.set("IK 계산 중…")
        seed = self.current.copy() if self.have_current else np.zeros(6)
        def run():
            try:
                from mycobot_id.inverse_kinematics import solve_closed_tip
                result = solve_closed_tip(numbers[:3], numbers[3], self.floor_calibration, self.tool, seed)
            except Exception as exc:
                from types import SimpleNamespace
                result = SimpleNamespace(ok=False, message=f"IK 오류: {exc}")
            self.ik_events.put((request, result))
        threading.Thread(target=run, daemon=True).start()

    def check_ee(self):
        if self.ee_solution is None or self.target() != list(self.ee_solution):
            self.ee_status.set("먼저 입력한 좌표로 IK 계산을 수행하세요.")
            return
        self.check_path()

    def move_ee(self):
        if self.ee_solution is None or self.checked_target != self.ee_solution:
            self.ee_status.set("IK 계산 후 현재 읽기 및 경로 검사를 통과해야 합니다.")
            return
        self.move()

    def log_message(self, label: str, values: dict) -> None:
        record = {"time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                  "label": label, **values}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        print(line, flush=True)
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            self.status_var.set(f"로그 저장 실패: {exc}")

    def log_pose(self, label: str, payload: dict) -> None:
        geometry = forward_kinematics(payload["angles"], self.tool)
        xyz = (geometry.transforms[-1][:3, 3] * 1000.0).tolist()
        table_xyz = list(xyz)
        tip_xyz = None
        if self.floor_calibration.get("enabled"):
            offset_mm = self.floor_calibration["model_to_table_z_offset_m"] * 1000
            table_xyz[2] += offset_mm
            tcp = np.array([*self.floor_calibration["closed_tip_midpoint_flange_m"], 1.0])
            tip_xyz = (geometry.transforms[-1] @ tcp)[:3] * 1000.0
            tip_xyz[2] += offset_mm
        phase = "after" if label == "이동 후" else "before"
        if phase == "before":
            for groups in self.pose_values.values():
                for variable in groups["after"]:
                    variable.set("—")
        coords = payload.get("coords")
        for index, variable in enumerate(self.pose_values["robot"][phase]):
            variable.set(f"{float(coords[index]):.2f}" if coords is not None else "—")
        for variable, value in zip(self.pose_values["model"][phase], table_xyz):
            variable.set(f"{value:.2f}")
        for index, variable in enumerate(self.pose_values["tip"][phase]):
            value = float(tip_xyz[index]) if tip_xyz is not None else None
            variable.set(f"{0.0 if abs(value) < 0.005 else value:.2f}" if value is not None else "—")
        self.coords_status.set(label + (": 로봇 좌표 수신 실패" if coords is None else ": 좌표 갱신됨"))
        self.log_message(label, {
            "joints_deg": payload["angles"],
            "robot_coords_mm_deg": payload.get("coords"),
            "robot_coords_error": payload.get("coords_error"),
            "model_flange_xyz_mm": [round(float(value), 3) for value in xyz],
            "table_flange_xyz_mm": [round(float(value), 3) for value in table_xyz],
            "table_closed_tip_estimate_xyz_mm": tip_xyz.tolist() if tip_xyz is not None else None,
        })

    def grip_parameters(self, value=None):
        try:
            value = int(self.grip_value.get() if value is None else value)
            speed = int(self.grip_speed.get())
            if not 0 <= value <= 100 or not 1 <= speed <= 100:
                raise ValueError()
            return {"value": value, "speed": speed}
        except (ValueError, tk.TclError):
            self.grip_status.set("개방값 0~100, 속도 1~100 정수를 입력하세요.")
            return None

    def manual_grip(self, value=None):
        if not self.args.enable_motion or self.moving or self.manual_busy:
            self.grip_status.set("그리퍼 제어 불가: 동작 활성화 여부와 진행 상태를 확인하세요.")
            return
        if self.transaction is not None:
            self.grip_status.set("검사 세션 대기 중입니다. 목표값 변경으로 취소 후 제어하세요.")
            return
        grip = self.grip_parameters(value)
        if grip is None:
            return
        self.grip_value.set(grip["value"])
        self.manual_busy = True
        self.grip_status.set("그리퍼 명령 전송 중…")
        def run():
            command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                       "-o", "StrictHostKeyChecking=yes", self.args.ssh_host,
                       self.args.remote_python, self.args.remote_bridge, "gripper",
                       "--port", self.args.port, "--baud", str(self.args.baud),
                       "--gripper-value", str(grip["value"]), "--gripper-speed", str(grip["speed"])]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=35)
                events = list(_bridge_events(result.stdout.splitlines()))
                last = events[-1] if events else {}
                if result.returncode or last.get("event") != "gripper_final":
                    raise RuntimeError(last.get("message") or result.stderr or "그리퍼 응답 없음")
                self.manual_events.put((True, f"그리퍼 정지: 목표 {grip['value']}, 측정 {last['actual']} (파지 여부 미확인)"))
            except Exception as exc:
                self.manual_events.put((False, "그리퍼 오류: " + str(exc)))
        threading.Thread(target=run, daemon=True).start()

    def _show_current(self) -> None:
        for variable, value in zip(self.current_vars, self.current):
            variable.set(f"{value:.2f}")
        target = self.target()
        if target is not None:
            self.draw(target)

    def reset_targets(self) -> None:
        if self.moving or self.manual_busy or self.ik_busy:
            self.status_var.set("진행 중인 동작이 끝난 후 목표를 초기화하세요.")
            return
        self.cancel_pending_check()
        self.ee_solution = None
        for variable, value in zip(self.target_vars, DEFAULT_TARGET_DEG):
            variable.set(value)
        self.preview()
        angles = ", ".join(f"{value:g}°" for value in DEFAULT_TARGET_DEG)
        self.status_var.set(f"목표 관절각을 [{angles}]로 초기화했습니다. 이동하려면 경로 검사를 수행하세요.")

    def copy_current(self) -> None:
        if not self.have_current:
            return
        for variable, value in zip(self.target_vars, self.current):
            variable.set(round(float(value), 2))
        self.target_changed()

    def stop(self) -> None:
        if self.transaction is not None:
            if self.moving:
                self.transaction.request_stop()
                self.status_var.set("정지 요청 전송. 도착 후 잡기 취소 요청됨.")
            else:
                self.transaction.abort()
                self.transaction = None
                self.checked_target = None
                self.move_button.configure(state=tk.DISABLED)
                self.check_button.configure(state=tk.NORMAL)
                self.status_var.set("검사 세션 취소됨.")
            return
        def send_stop() -> None:
            command = [
                "ssh", "-T", "-o", "BatchMode=yes", self.args.ssh_host,
                self.args.remote_python, self.args.remote_bridge, "stop",
                "--port", self.args.port, "--baud", str(self.args.baud),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            message = result.stdout.strip() or result.stderr.strip() or "정지 명령 완료"
            self.root.after(0, self.status_var.set, message)
        threading.Thread(target=send_stop, daemon=True).start()

    def close(self) -> None:
        if self.transaction is not None:
            if self.moving:
                self.transaction.request_stop()
            else:
                self.transaction.abort()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "jetson.json"))
    parser.add_argument("--pose-config", default=str(PROJECT_ROOT / "configs" / "robot_pose_gui.json"))
    parser.add_argument("--ssh-host")
    parser.add_argument("--remote-python")
    parser.add_argument("--remote-bridge")
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int)
    parser.add_argument("--enable-motion", action="store_true")
    args = parser.parse_args()
    connection = json.loads(Path(args.config).read_text(encoding="utf-8"))
    state = json.loads((PROJECT_ROOT / ".jetson" / "state.json").read_text(encoding="utf-8"))
    args.ssh_host = args.ssh_host or connection["ssh_host"]
    args.port = args.port or connection["port"]
    args.baud = args.baud or int(connection["baud"])
    args.remote_python = args.remote_python or connection["remote_root"] + "/.venv/bin/python"
    args.remote_bridge = args.remote_bridge or state["release"] + "/scripts/robot_bridge.py"
    pose_config = json.loads(Path(args.pose_config).read_text(encoding="utf-8"))
    args.tool = ToolGeometry(**pose_config.get("tool", {}))
    args.floor_calibration = pose_config.get("floor_calibration", {})
    root = tk.Tk()
    LaptopPoseGUI(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
