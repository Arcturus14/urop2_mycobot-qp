"""Replay a recorded closed-tip QP simulation and optionally export a video."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mycobot_id.kinematics import ToolGeometry, forward_kinematics
from qp_control.obstacle import HALF_SIDE_M


BOX_FACES = np.array([
    [0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
    [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5],
])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="CSV from simulate_qp --b-profile --trace-dir")
    parser.add_argument("--output", type=Path, help="Export .gif or .mp4 (requires FFmpeg)")
    args = parser.parse_args()
    with args.log.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error("Replay log has no samples")
    if args.output and args.output.suffix.lower() not in {".gif", ".mp4"}:
        parser.error("--output must end in .gif or .mp4")
    if args.output and args.output.suffix.lower() == ".mp4" and not FFMpegWriter.isAvailable():
        parser.error("MP4 export requires FFmpeg on PATH. Use .gif for Pillow export.")

    time_s = np.array([float(row["sample_s"]) for row in rows])
    angles = [[float(row[f"q_actual_j{i}_deg"]) for i in range(1, 7)] for row in rows]
    tip, goal = (np.array([[float(row[f"{name}_{axis}_mm"]) for axis in "xyz"]
                           for row in rows]) for name in ("ee", "goal"))
    error = np.linalg.norm(tip - goal, axis=1)
    tolerance = float(rows[0]["tolerance_mm"])
    dt = round(float(np.median(np.diff(time_s))), 6) if len(rows) > 1 else 0.1
    config_path = Path(__file__).resolve().parents[1] / "configs" / "robot_pose_gui.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tool = ToolGeometry(**config["tool"])
    shift = np.array([0, 0, 1000 * config["floor_calibration"]["model_to_table_z_offset_m"]])
    geometries = [forward_kinematics(q, tool) for q in angles]
    tip += shift
    goal += shift

    fig = plt.figure(figsize=(11, 6))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], bottom=0.16, wspace=0.38)
    arm = fig.add_subplot(grid[0], projection="3d")
    chart = fig.add_subplot(grid[1])
    arm.set(xlim=(-300, 450), ylim=(-375, 375), zlim=(-65, 570),
            xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    arm.set_box_aspect((750, 750, 590))
    arm.set_proj_type("ortho")
    arm.view_init(elev=25, azim=-60)
    arm.add_collection3d(Poly3DCollection(
        [[(-300, -375, 0), (450, -375, 0), (450, 375, 0), (-300, 375, 0)]],
        facecolors="#819098", alpha=0.13,
    ))
    if rows[0].get("obstacle_x_mm"):
        center = np.array([float(rows[0][f"obstacle_{axis}_mm"]) for axis in "xyz"])
        corners = center + 1000 * HALF_SIDE_M * np.array([
            [x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)
        ])
        arm.add_collection3d(Poly3DCollection(
            corners[BOX_FACES], facecolors="#ca462d", edgecolors="#a22d21", alpha=0.35,
        ))
        arm.text(*center, "50 mm cube", color="#a22d21", fontsize=8)
    links, = arm.plot([], [], [], "o-", lw=4, color="#19669a", label="Robot")
    gripper, = arm.plot([], [], [], color="#247849", lw=2)
    path, = arm.plot([], [], [], color="#df8520", label="Closed-tip path")
    tip_dot, = arm.plot([], [], [], "o", color="#df8520", zorder=8, label="Current closed tip")
    target_label = rows[0].get("target_label") or "B reference"
    goal_dot, = arm.plot([], [], [], "*", color="#b32636", ms=13, zorder=9, label="EE target")
    floor_dot, = arm.plot([], [], [], "x", color="#b32636", ms=9, mew=2,
                          zorder=7, label="Target XY on table")
    goal_height, = arm.plot([], [], [], ":", color="#b32636", lw=1.5, zorder=6)
    ee_axes = []
    for index, (name, color) in enumerate(zip("XYZ", ("#b32636", "#247849", "#713eab"))):
        direction = np.eye(3)[index]
        arm.quiver(0, 0, 0, *direction, length=65, color=color, alpha=0.6)
        arm.text(*(75 * direction), f"W +{name}", color=color, fontsize=8)
        line, = arm.plot([], [], [], "o-", markevery=[1], color=color, lw=2.5)
        axis_label = arm.text(0, 0, 0, f"EE +{name}", color=color, fontsize=8)
        ee_axes.append((line, axis_label))
    target_axis, = arm.plot([], [], [], "v--", markevery=[1], color="#087d78", lw=2,
                            label="Target: workspace -Z")
    boxes = []
    for box in geometries[0].tool_boxes:
        color = "#247849" if box.name == "gripper" else "#7956a2"
        artist = Poly3DCollection([], facecolors=color, edgecolors=color, alpha=0.24)
        arm.add_collection3d(artist)
        boxes.append(artist)
    arm.legend(loc="upper left", fontsize=8)

    chart.plot(time_s, error, color="#19669a", lw=2)
    chart.axhline(tolerance, color="#b32636", ls="--", label=f"{tolerance:g} mm threshold")
    cursor = chart.axvline(time_s[0], color="#df8520", alpha=0.8)
    point, = chart.plot([], [], "o", color="#df8520")
    chart.set(xlabel="Simulation time (s)", ylabel="Closed-tip distance to target (mm)",
              title="Position error")
    chart.grid(alpha=0.25)
    chart.legend()
    label = fig.suptitle("")

    def draw(frame):
        geometry = geometries[frame]
        flange = geometry.transforms[-1]
        origin = tip[frame]
        for direction, (line, axis_label) in zip(flange[:3, :3].T, ee_axes):
            endpoint = origin + 55 * direction
            line.set_data_3d(*np.vstack((origin, endpoint)).T)
            axis_label.set_position_3d(endpoint)
        target = goal[frame]
        table_point = target.copy()
        table_point[2] = 0
        target_axis.set_data_3d(*np.vstack((target + [0, 0, 75], target + [0, 0, 15])).T)
        tilt = np.degrees(np.arccos(np.clip(-flange[2, 2], -1.0, 1.0)))
        links.set_data_3d(*(geometry.joints * 1000 + shift).T)
        gripper.set_data_3d(*np.vstack((geometry.joints[-1] * 1000 + shift, tip[frame])).T)
        path.set_data_3d(*tip[:frame + 1].T)
        tip_dot.set_data_3d(*tip[frame:frame + 1].T)
        goal_dot.set_data_3d(*goal[frame:frame + 1].T)
        floor_dot.set_data_3d(*table_point.reshape(1, 3).T)
        goal_height.set_data_3d(*np.vstack((table_point, target)).T)
        arm.set_title(f"EE target ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}) mm\n"
                      f"Table mark ({table_point[0]:.2f}, {table_point[1]:.2f}, 0) mm", fontsize=10)
        for artist, box in zip(boxes, geometry.tool_boxes):
            artist.set_verts((box.corners() * 1000 + shift)[BOX_FACES])
        cursor.set_xdata([time_s[frame], time_s[frame]])
        point.set_data(time_s[frame:frame + 1], error[frame:frame + 1])
        speed = rows[frame].get("ee_speed_mm_s")
        speed_label = f"  |  EE speed {float(speed):.1f} mm/s" if speed else ""
        label.set_text(f"myCobot {target_label}  |  {time_s[frame]:.1f} s  |  "
                       f"error {error[frame]:.1f} mm  |  EE +Z to W -Z: {tilt:.1f}°{speed_label}  |  "
                       f"{rows[frame]['status']}")

    draw(0)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        animation = FuncAnimation(fig, draw, frames=len(rows), cache_frame_data=False)
        writer = PillowWriter(fps=1 / dt) if args.output.suffix.lower() == ".gif" else FFMpegWriter(fps=1 / dt)
        animation.save(args.output, writer=writer, dpi=85)
        print(f"Saved: {args.output}")
    else:
        slider = Slider(fig.add_axes([0.18, 0.06, 0.66, 0.04]), "Frame",
                        0, max(1, len(rows) - 1), valinit=0, valstep=1)
        slider.on_changed(lambda frame: draw(min(int(frame), len(rows) - 1)))
        timer = fig.canvas.new_timer(interval=max(1, round(1000 * dt)))
        timer.add_callback(lambda: slider.set_val((int(slider.val) + 1) % len(rows)))
        playing = True
        top_view = False

        def key_pressed(event):
            nonlocal playing, top_view
            if event.key == " ":
                playing = not playing
                timer.start() if playing else timer.stop()
            elif event.key == "t":
                top_view = not top_view
                if top_view:
                    arm.view_init(elev=90, azim=-90)
                else:
                    arm.view_init(elev=25, azim=-60)
                goal_dot.set_visible(not top_view)
                target_axis.set_visible(not top_view)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("key_press_event", key_pressed)
        fig.text(0.5, 0.015, "Space: play / pause   |   T: top view   |   Drag slider: seek", ha="center")
        timer.start()
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
