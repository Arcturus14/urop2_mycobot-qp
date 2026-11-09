"""Compare deployed QP weights with a local, delayed joint response model."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

from mycobot_id.kinematics import forward_kinematics, self_clearance, self_collision_pairs
from qp_control.obstacle import AVOIDANCE_MARGIN_M, MIN_BODY_CLEARANCE_M, body_clearances, body_spheres, box_distance
from qp_control.run import (
    B_APPROACH_LEAD, B_APPROACH_PERIOD, B_APPROACH_POSTURE, B_APPROACH_POSTURE_WEIGHT,
    B_APPROACH_TOLERANCE_M, B_APPROACH_WEIGHTS, B_CONTACT_DEG, B_HOVER_HEIGHT_M,
    B_HOVER_PERIOD, B_HOVER_POSTURE, B_HOVER_SETTLE_SAMPLES, B_HOVER_SMOOTHNESS,
    B_HOVER_SPEED_M_S, B_HOVER_TERMINAL_GAIN, B_HOVER_TILT_DEG, B_HOVER_TOLERANCE_M,
    BS_COMMAND_LEAD_DEG, BS_MAX_SPEED_DEG_S, BS_MIDPOINT_MM, BS_SELF_COLLISION_WEIGHT,
    QP_ONLY_SELF_COLLISION_WEIGHT,
    BS_TERMINAL_GAIN,
    PositionQP, TABLE_HOVER_BODY_MARGIN_M, TABLE_HOVER_REFERENCE_WEIGHT, TABLE_HOVER_VERTICAL_WEIGHT,
    TABLE_HOVER_WEIGHTS,
    TOLERANCE_M, WAYPOINT_TOLERANCE_M,
    accumulate_command, b_approach_setup, b_path_issue, detour_command, detour_posture,
    ee_position, grasp_axis, hover_reference, table_point, tool_tilt_deg,
)


PERIOD = 0.1
SUBSTEPS = 20


def pose(q, tcp_offset):
    flange = forward_kinematics(np.degrees(q)).transforms[-1]
    return flange[:3, 3], flange[:3, 3] + flange[:3, :3] @ tcp_offset


def audit_clearance(poses, tcp, tool, table_offset, obstacle_center=None):
    min_self = min_table = min_body = math.inf
    for q in poses:
        geometry = forward_kinematics(np.degrees(q), tool)
        min_self = min(min_self, *(self_clearance(geometry, tool, pair)
                                   for pair in self_collision_pairs(geometry)))
        flange = geometry.transforms[-1]
        tip = flange[:3, 3] + flange[:3, :3] @ tcp
        lowest = min(tip[2], *(box.corners()[:, 2].min() for box in geometry.tool_boxes))
        min_table = min(min_table, lowest + table_offset)
        if obstacle_center is not None:
            min_body = min(min_body, float(np.min(body_clearances(*body_spheres(geometry), obstacle_center))))
    return min_self, min_table, min_body


def simulate(start, goal, tip_goal, tcp_offset, rates, delays, deadband, weights, *,
             track_tip=False, period=PERIOD, command_lead_deg=0.0, tolerance_m=TOLERANCE_M,
             posture=None, posture_weight=0.0, control_goal=None, approach_geometry=None,
             trace=None, vertical_weight=0.0, record_blocked=False, duration_s=20.0,
             tilt_tolerance_deg=180.0, target_label="B reference", smoothness_weight=0.0,
             settle_weight=0.0, require_settle=False, obstacle_center=None, obstacle_weight=0.0,
             whole_arm=False, guard_obstacle=False, obstacle_body_margin=AVOIDANCE_MARGIN_M,
             detour=False, terminal_gain=1.0, via=None, terminal_body_margin=AVOIDANCE_MARGIN_M,
             protect_tool_height=False, max_speed_deg_s=10.0, self_collision_weight=0.0,
             qp_only=False, decision_variable="delta_q", max_acceleration_deg_s2=125.0,
             target_axis=None):
    table_offset = approach_geometry[1] if approach_geometry is not None and track_tip else None
    obstacle_tool = approach_geometry[0] if whole_arm else None
    target_axis = grasp_axis() if target_axis is None else np.asarray(target_axis, dtype=float)
    lead = np.radians(command_lead_deg)
    qp = PositionQP(period, position_weight=weights[0],
                    x_position_weight=weights[1], brake_weight=weights[2],
                    tcp_offset=tcp_offset if track_tip else None,
                    posture=posture, posture_weight=posture_weight, vertical_weight=vertical_weight,
                    table_offset=table_offset, smoothness_weight=smoothness_weight,
                    settle_weight=settle_weight, obstacle_center=obstacle_center,
                    obstacle_weight=obstacle_weight, obstacle_tool=obstacle_tool,
                    obstacle_body_margin=obstacle_body_margin, terminal_gain=terminal_gain,
                    terminal_body_margin=terminal_body_margin,
                    table_tool=approach_geometry[0] if protect_tool_height else None,
                    max_speed=np.radians(max_speed_deg_s), self_collision_weight=self_collision_weight,
                    command_lead=lead, decision_variable=decision_variable,
                    max_acceleration=(None if max_acceleration_deg_s2 == 0 else
                                      np.radians(max_acceleration_deg_s2)),
                    target_axis=target_axis)
    target_point = tip_goal if track_tip else goal
    q = start.copy()
    steps = math.floor(duration_s / period + 1e-9)
    commands = np.empty((steps + 1, 6))
    commands[0] = q
    joints = np.arange(6)
    substeps = max(SUBSTEPS, round(period / 0.005))
    dt = period / substeps
    response = -np.expm1(-rates * dt)
    min_tip_z = math.inf
    min_tool_z = math.inf
    min_obstacle_distance = math.inf
    min_body_distance = math.inf
    min_self_clearance = math.inf
    audit_poses = [q.copy()] if qp_only else None
    if whole_arm:
        spheres = body_spheres(forward_kinematics(np.degrees(q), obstacle_tool))
        min_body_distance = float(np.min(body_clearances(*spheres, obstacle_center)))
    peak_command_step = 0.0
    blocked = ""
    previous_position = None
    settled_samples = 0
    first_entry_s = None
    exits = 0
    inside_before = False
    waypoint = detour_posture(q) if detour else None
    waypoint_samples = 0
    state_seconds = 0.0
    qp_seconds = 0.0
    response_seconds = 0.0
    audit_seconds = 0.0
    control_steps = 0
    simulation_started = time.perf_counter()
    for tick in range(steps + 1):
        state_started = time.perf_counter()
        if detour:
            waypoint_samples = waypoint_samples + 1 if np.all(
                np.abs(q - waypoint) <= np.radians(1.25)
            ) else 0
            detour = waypoint_samples < 3
        position, tip = pose(q, tcp_offset)
        if protect_tool_height and not qp_only:
            tool = approach_geometry[0]
            geometry = forward_kinematics(np.degrees(q), tool)
            clearance = min(self_clearance(geometry, tool, pair) for pair in self_collision_pairs(geometry))
            min_self_clearance = min(min_self_clearance, clearance)
        min_tip_z = min(min_tip_z, tip[2])
        if table_offset is not None and tip[2] + table_offset < 0 and not qp_only:
            raise RuntimeError(f"Simulated EE below the table at {tick * period:.3f} s")
        tracked = tip if track_tip else position
        if via is not None and np.linalg.norm(tracked - via) <= WAYPOINT_TOLERANCE_M:
            via = None
        if obstacle_center is not None:
            min_obstacle_distance = min(min_obstacle_distance, box_distance(tracked, obstacle_center)[0])
        error = np.linalg.norm(tracked - target_point)
        inside = error <= tolerance_m and (
            tilt_tolerance_deg == 180.0 or tool_tilt_deg(q, target_axis) <= tilt_tolerance_deg
        )
        speed = (np.linalg.norm(tracked - previous_position) / period
                 if previous_position is not None else None)
        previous_position = tracked
        if inside and first_entry_s is None:
            first_entry_s = tick * period
        exits += int(inside_before and not inside)
        inside_before = inside
        settled_samples = (
            settled_samples + 1 if via is None and inside and speed is not None
            and speed <= B_HOVER_SPEED_M_S else 0
        )
        reached = settled_samples >= B_HOVER_SETTLE_SAMPLES if require_settle else inside
        reached = reached and via is None
        if trace is not None:
            trace.append(dict(
                sample_s=tick * period, tolerance_mm=1000 * tolerance_m,
                ee_error_mm=1000 * error,
                target_label=target_label,
                ee_speed_mm_s=None if speed is None else 1000 * speed,
                status="goal_reached" if reached else
                "duration_elapsed" if tick == steps else "detour_j1" if detour else
                "via" if via is not None else "running",
                **{f"q_actual_j{i}_deg": value for i, value in enumerate(np.degrees(q), 1)},
                **{f"ee_{axis}_mm": 1000 * value for axis, value in zip("xyz", tracked)},
                **{f"goal_{axis}_mm": 1000 * value for axis, value in zip("xyz", target_point)},
            ))
            if obstacle_center is not None:
                center_on_table = obstacle_center + [0, 0, table_offset]
                trace[-1].update({f"obstacle_{axis}_mm": 1000 * value
                                  for axis, value in zip("xyz", center_on_table)})
        if reached or tick == steps:
            break
        state_seconds += time.perf_counter() - state_started
        control_steps += 1
        previous_velocity = (commands[tick] - commands[tick - 1]) / period if tick else None
        qp_started = time.perf_counter()
        if detour:
            target = detour_command(q, commands[tick], waypoint, period)
        else:
            active_goal = via if via is not None else target_point if control_goal is None else control_goal
            target, status = qp.command(
                q, tracked, active_goal,
                previous_velocity=previous_velocity, command_base=commands[tick] if lead else None,
            )
            if target is None:
                raise RuntimeError(f"QP failed at step {tick}: {status}")
            if lead:
                target = accumulate_command(q, commands[tick], target, lead)
        if qp_only:
            segments = max(1, math.ceil(np.max(np.abs(np.degrees(target - q)))))
            audit_poses.extend(np.linspace(q, target, segments + 1))
        elif approach_geometry is not None:
            issue = b_path_issue(q, target, tcp_offset, *approach_geometry,
                                 obstacle_center=obstacle_center if guard_obstacle else None)
            if issue:
                if record_blocked:
                    blocked = issue
                    if trace is not None:
                        trace[-1]["status"] = "path_blocked"
                    break
                raise RuntimeError(f"Simulated path blocked at {tick * period:.1f} s: {issue}")
        qp_seconds += time.perf_counter() - qp_started
        peak_command_step = max(peak_command_step, np.max(np.abs(target - commands[tick])))
        commands[tick + 1] = target
        response_started = time.perf_counter()
        for substep in range(substeps):
            time_s = tick * period + (substep + 0.5) * dt
            indices = np.maximum(0, np.floor((time_s - delays) / period).astype(int) + 1)
            delayed_target = commands[indices, joints]
            delta = delayed_target - q
            q += response * np.sign(delta) * np.maximum(np.abs(delta) - deadband, 0)
            if qp_only and ((substep + 1) % 4 == 0 or substep + 1 == substeps):
                audit_poses.append(q.copy())
            if table_offset is not None:
                if whole_arm or protect_tool_height:
                    geometry = forward_kinematics(np.degrees(q), approach_geometry[0])
                    flange = geometry.transforms[-1]
                    tip_substep = flange[:3, 3] + flange[:3, :3] @ tcp_offset
                    if whole_arm and not qp_only:
                        min_body_distance = min(min_body_distance, float(np.min(
                            body_clearances(*body_spheres(geometry), obstacle_center)
                        )))
                    if protect_tool_height:
                        lowest = min(tip_substep[2], *(box.corners()[:, 2].min()
                                                       for box in geometry.tool_boxes))
                        min_tool_z = min(min_tool_z, lowest + table_offset)
                else:
                    tip_substep = ee_position(q, tcp_offset)
                tip_z = tip_substep[2]
                min_tip_z = min(min_tip_z, tip_z)
                if obstacle_center is not None:
                    min_obstacle_distance = min(
                        min_obstacle_distance, box_distance(tip_substep, obstacle_center)[0]
                    )
                if tip_z + table_offset < 0 and not qp_only:
                    raise RuntimeError(f"Simulated EE below the table during step {tick}")
        response_seconds += time.perf_counter() - response_started
    if qp_only:
        audit_started = time.perf_counter()
        min_self_clearance, min_audited_table, min_audited_body = audit_clearance(
            audit_poses, tcp_offset, *approach_geometry, obstacle_center=obstacle_center,
        )
        audit_seconds = time.perf_counter() - audit_started
        min_tool_z = min(min_tool_z, min_audited_table)
        min_body_distance = min(min_body_distance, min_audited_body)
    command_velocity = np.degrees(np.diff(commands[:tick + 1], axis=0)) / period
    velocity_change = np.diff(command_velocity, axis=0, prepend=np.zeros((1, 6)))
    command_acceleration = velocity_change / period
    speed_saturated = np.any(
        np.abs(command_velocity) >= max_speed_deg_s - 1e-6, axis=1
    ) if tick else np.zeros(0, dtype=bool)
    timing_denominator = max(1, control_steps)
    simulation_seconds = time.perf_counter() - simulation_started
    return {
        "flange_mm": 1000 * np.linalg.norm(position - goal),
        "tip_mm": 1000 * np.linalg.norm(tip - tip_goal),
        "time_s": tick * period,
        "reached": reached,
        "first_entry_s": first_entry_s,
        "tolerance_exits": exits,
        "final_ee_speed_mm_s": None if speed is None else 1000 * speed,
        "min_tip_z_mm": 1000 * min_tip_z,
        "min_tool_table_clearance_mm": 1000 * min_tool_z if math.isfinite(min_tool_z) else None,
        "min_self_clearance_mm": 1000 * min_self_clearance if math.isfinite(min_self_clearance) else None,
        "peak_command_step_deg": np.degrees(peak_command_step),
        "tool_tilt_deg": tool_tilt_deg(q, target_axis),
        "rms_velocity_change_deg_s": np.sqrt(np.mean(velocity_change**2)) if tick else 0.0,
        "rms_command_acceleration_deg_s2": (
            np.sqrt(np.mean(command_acceleration**2)) if tick else 0.0
        ),
        "peak_command_acceleration_deg_s2": (
            np.max(np.abs(command_acceleration)) if tick else 0.0
        ),
        "speed_saturation_fraction": (
            np.mean(speed_saturated) if len(speed_saturated) else 0.0
        ),
        "simulation_state_ms_per_step": 1000 * state_seconds / timing_denominator,
        "simulation_qp_ms_per_step": 1000 * qp_seconds / timing_denominator,
        "simulation_response_ms_per_step": 1000 * response_seconds / timing_denominator,
        "simulation_audit_ms_per_step": 1000 * audit_seconds / timing_denominator,
        "simulation_total_ms_per_step": 1000 * simulation_seconds / timing_denominator,
        "path_blocked": blocked,
        "min_obstacle_distance_mm": None if obstacle_center is None else 1000 * min_obstacle_distance,
        "min_body_distance_mm": 1000 * min_body_distance if whole_arm else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="QP CSV with the physical starting pose")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-tip", action="store_true")
    parser.add_argument("--b-profile", action="store_true", help="Verify the deployed B approach profile")
    hover_mode = parser.add_mutually_exclusive_group()
    hover_mode.add_argument("--b-hover", action="store_true", help="Use a point 60 mm above B for alignment")
    hover_mode.add_argument("--hover-over-mm", type=float, nargs=3, metavar=("X", "Y", "Z"),
                            help="Simulate a closed-tip goal 60 mm above the table-frame point")
    hover_mode.add_argument("--bs-midpoint-hover", action="store_true")
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--grasp-direction", choices=("vertical", "horizontal"), default="vertical")
    parser.add_argument("--grasp-angle-deg", type=float, default=0.0,
                        help="For horizontal grasping, angle from workspace +X toward +Y")
    parser.add_argument("--vertical-weight", type=float, nargs="+",
                        help="B profile weights for alignment of J6 +Z with workspace -Z")
    parser.add_argument("--smoothness-weight", type=float, nargs="+")
    parser.add_argument("--decision-variable", choices=("velocity", "delta_q"), nargs="+",
                        default=["delta_q"],
                        help="QP variable; both forms use the same physical angular-acceleration cost")
    parser.add_argument("--max-acceleration-deg-s2", type=float, nargs="+", default=[500.0],
                        help="Hard command acceleration limit; 0 disables it for baseline comparison")
    parser.add_argument("--max-speed-deg-s", type=float, nargs="+",
                        help="Override the tuned command-speed limit for comparison")
    parser.add_argument("--deadband-factor", type=float, nargs="+", default=[1.0, 1.25],
                        help="Scale the fitted joint deadband for robustness testing")
    parser.add_argument("--settle-weight", type=float, nargs="+",
                        help="Test terminal EE damping and require three slow in-tolerance samples")
    parser.add_argument("--obstacle-center-mm", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="50 mm cube center in the replay grid, with Z=0 at the table")
    parser.add_argument("--via-mm", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--reference-posture-deg", type=float, nargs=6)
    parser.add_argument("--obstacle-weight", type=float, nargs="+", default=[0.0])
    parser.add_argument("--whole-arm", action="store_true", help="Apply and evaluate full arm obstacle clearance")
    parser.add_argument("--guard-obstacle", action="store_true", help="Apply the Jetson command clearance check")
    parser.add_argument("--body-margin-mm", type=float, nargs="+", default=[25.0])
    parser.add_argument("--response-model", type=Path, help="Joint parameters from fit_qp_response")
    parser.add_argument("--detour-j1", action="store_true")
    parser.add_argument("--trace-dir", type=Path, help="Save one replay log per B profile scenario")
    parser.add_argument("--period", type=float, nargs="+")
    parser.add_argument("--terminal-gain", type=float, nargs="+",
                        help="B hover gain for terminal position and alignment tracking")
    parser.add_argument("--qp-only", action="store_true",
                        help="Use the tuned B–S weights for a hover target without a path veto, then audit the path")
    parser.add_argument("--self-collision-weight", type=float, nargs="+",
                        help="Override the B–S simulation's soft self-clearance weight")
    parser.add_argument("--command-lead-deg", type=float, nargs="+", default=[0.0],
                        help="0 uses measured angles; positive values cap accumulated command lead")
    parser.add_argument("--tolerance-mm", type=float, default=1000 * TOLERANCE_M)
    args = parser.parse_args()
    if args.qp_only and not (args.bs_midpoint_hover or args.hover_over_mm is not None):
        parser.error("--qp-only requires --bs-midpoint-hover or --hover-over-mm")
    if args.qp_only and (args.guard_obstacle or (args.obstacle_center_mm is not None and not args.whole_arm)):
        parser.error("--qp-only obstacle testing requires --whole-arm and omits --guard-obstacle")
    tuned_hover = args.bs_midpoint_hover or args.qp_only
    if args.self_collision_weight is not None and not tuned_hover:
        parser.error("--self-collision-weight requires --bs-midpoint-hover or --qp-only")
    default_weight = QP_ONLY_SELF_COLLISION_WEIGHT if args.qp_only else (
        BS_SELF_COLLISION_WEIGHT if args.bs_midpoint_hover else 0.0
    )
    self_collision_weights = args.self_collision_weight or [default_weight]
    if any(not np.isfinite(weight) or weight < 0 for weight in self_collision_weights):
        parser.error("--self-collision-weight values must be finite and nonnegative")
    if args.trace_dir is not None and len(self_collision_weights) != 1:
        parser.error("--trace-dir requires a single self-collision weight")
    if args.bs_midpoint_hover:
        args.hover_over_mm = BS_MIDPOINT_MM
        if args.obstacle_center_mm is not None or args.via_mm is not None:
            parser.error("--bs-midpoint-hover is tuned for an unobstructed workspace")
    custom_hover = args.hover_over_mm is not None
    try:
        target_axis = grasp_axis(args.grasp_direction, args.grasp_angle_deg)
    except ValueError as exc:
        parser.error(str(exc))
    if args.grasp_direction == "horizontal" and not custom_hover:
        parser.error("--grasp-direction horizontal requires a custom hover target")
    if custom_hover and (not np.all(np.isfinite(args.hover_over_mm)) or args.hover_over_mm[2] < 0):
        parser.error("--hover-over-mm requires a finite table-frame point with Z >= 0")
    if args.via_mm is not None and (
        not custom_hover or not np.all(np.isfinite(args.via_mm)) or args.via_mm[2] < 20
    ):
        parser.error("--via-mm requires --hover-over-mm and a finite table-frame point with Z >= 20")
    if args.reference_posture_deg is not None and not (
        custom_hover and args.whole_arm and (args.guard_obstacle or args.qp_only)
    ):
        parser.error("--reference-posture-deg requires a custom hover and whole-arm obstacle checks")
    if custom_hover:
        args.b_profile = True
        args.settle_weight = args.settle_weight or [0.05]
    hover = args.b_hover or custom_hover
    args.vertical_weight = args.vertical_weight or [TABLE_HOVER_VERTICAL_WEIGHT if custom_hover else 0.0]
    args.smoothness_weight = args.smoothness_weight or [B_HOVER_SMOOTHNESS if custom_hover else 0.0]
    default_terminal_gain = BS_TERMINAL_GAIN if tuned_hover else B_HOVER_TERMINAL_GAIN
    args.terminal_gain = args.terminal_gain or [default_terminal_gain if custom_hover else 1.0]
    if args.period is None:
        args.period = [B_HOVER_PERIOD if hover else B_APPROACH_PERIOD if args.b_profile else PERIOD]
    if any(not np.isfinite(value) or value < 1 for value in args.terminal_gain):
        parser.error("--terminal-gain values must be finite and at least 1")
    if args.terminal_gain != [1.0] and not hover:
        parser.error("--terminal-gain requires a hover mode")
    if args.trace_dir is not None and (len(args.period) != 1 or len(args.terminal_gain) != 1):
        parser.error("--trace-dir requires a single period and terminal gain")
    if args.detour_j1 and not (args.whole_arm and args.guard_obstacle and args.b_hover):
        parser.error("--detour-j1 requires --whole-arm, --guard-obstacle, and --b-hover")
    if any(not np.isfinite(margin) or margin < 25 for margin in args.body_margin_mm):
        parser.error("--body-margin-mm values must be finite and at least 25 mm")
    if args.body_margin_mm != [25.0] and not args.whole_arm:
        parser.error("--body-margin-mm requires --whole-arm")
    if args.trace_dir is not None and len(args.body_margin_mm) != 1:
        parser.error("--trace-dir requires a single body margin")
    if args.whole_arm and args.obstacle_center_mm is None:
        parser.error("--whole-arm requires --obstacle-center-mm")
    if args.guard_obstacle and not args.whole_arm:
        parser.error("--guard-obstacle requires --whole-arm")
    if args.obstacle_center_mm is not None and (
        not args.b_profile or not np.all(np.isfinite(args.obstacle_center_mm))
    ):
        parser.error("--obstacle-center-mm requires --b-profile and finite coordinates")
    if any(not np.isfinite(weight) or weight < 0 for weight in args.obstacle_weight):
        parser.error("--obstacle-weight values must be finite and nonnegative")
    if args.obstacle_weight != [0.0] and args.obstacle_center_mm is None:
        parser.error("--obstacle-weight requires --obstacle-center-mm")
    if args.trace_dir is not None and len(args.obstacle_weight) != 1:
        parser.error("--trace-dir requires a single obstacle weight")
    if args.settle_weight is not None:
        if not hover:
            parser.error("--settle-weight requires a hover mode")
        if any(not np.isfinite(weight) or weight < 0 for weight in args.settle_weight):
            parser.error("--settle-weight values must be finite and nonnegative")
        if args.trace_dir is not None and len(args.settle_weight) != 1:
            parser.error("--trace-dir requires a single settle weight")
    if args.b_hover and not args.b_profile:
        parser.error("--b-hover requires --b-profile")
    if not np.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error("--duration-s must be finite and positive")
    if args.trace_dir is not None and not args.b_profile:
        parser.error("--trace-dir requires --b-profile")
    if any(not np.isfinite(weight) or weight < 0 for weight in args.vertical_weight):
        parser.error("--vertical-weight values must be finite and nonnegative")
    if args.vertical_weight != [0.0] and not args.b_profile:
        parser.error("--vertical-weight requires --b-profile")
    if args.trace_dir is not None and len(args.vertical_weight) != 1:
        parser.error("--trace-dir requires a single vertical weight")
    if any(not np.isfinite(weight) or weight < 0 for weight in args.smoothness_weight):
        parser.error("--smoothness-weight values must be finite and nonnegative")
    if args.smoothness_weight != [0.0] and not args.b_profile:
        parser.error("--smoothness-weight requires --b-profile")
    if args.trace_dir is not None and len(args.smoothness_weight) != 1:
        parser.error("--trace-dir requires a single smoothness weight")
    if args.trace_dir is not None and len(args.decision_variable) != 1:
        parser.error("--trace-dir requires a single decision variable")
    if any(not np.isfinite(value) or value < 0 for value in args.max_acceleration_deg_s2):
        parser.error("--max-acceleration-deg-s2 values must be finite and nonnegative")
    if args.trace_dir is not None and len(args.max_acceleration_deg_s2) != 1:
        parser.error("--trace-dir requires a single acceleration limit")
    if args.max_speed_deg_s is not None and any(
        not np.isfinite(value) or value <= 0 for value in args.max_speed_deg_s
    ):
        parser.error("--max-speed-deg-s values must be finite and positive")
    if args.trace_dir is not None and args.max_speed_deg_s is not None and len(args.max_speed_deg_s) != 1:
        parser.error("--trace-dir requires a single speed limit")
    if any(not np.isfinite(value) or value <= 0 for value in args.deadband_factor):
        parser.error("--deadband-factor values must be finite and positive")
    if args.trace_dir is not None and len(args.deadband_factor) != 1:
        parser.error("--trace-dir requires a single deadband factor")
    if any(not np.isfinite(value) or value < 0.05 for value in args.period):
        parser.error("--period values must be finite and at least 0.05 s")
    if any(not np.isfinite(value) or value < 0 for value in args.command_lead_deg):
        parser.error("--command-lead-deg values must be finite and nonnegative")
    if not np.isfinite(args.tolerance_mm) or args.tolerance_mm <= 0:
        parser.error("--tolerance-mm must be finite and positive")
    with args.log.open(newline="", encoding="utf-8") as stream:
        first = next(csv.DictReader(stream))
    start = np.radians([float(first[f"q_actual_j{i}_deg"]) for i in range(1, 7)])
    logged_goal = np.array([float(first[f"goal_{axis}_mm"]) for axis in "xyz"]) / 1000
    if args.response_model is None:
        models = json.loads(Path("reports/identification_20261104/model_comparison.json").read_text())
        parameters = [models[f"J{i}"]["models"]["deadzone_velocity"]["parameters"] for i in range(1, 7)]
    else:
        models = json.loads(args.response_model.read_text(encoding="utf-8"))
        parameters = [models[f"J{i}"] for i in range(1, 7)]
    rates = np.array([p["rate_gain_per_s"] for p in parameters])
    delays = np.array([p["delay_s"] for p in parameters])
    deadband = np.radians([p["deadband_deg"] for p in parameters])
    calibration = json.loads(Path("configs/robot_pose_gui.json").read_text())["floor_calibration"]
    tcp = np.asarray(calibration["closed_tip_midpoint_flange_m"])
    goal, tip_goal = pose(np.radians(B_CONTACT_DEG), tcp)
    if args.b_profile:
        tcp, tip_goal, tool, table_offset = b_approach_setup()
        obstacle_center = None
        if args.obstacle_center_mm is not None:
            obstacle_center = table_point(args.obstacle_center_mm, table_offset)
        posture = B_HOVER_POSTURE if args.b_hover else B_APPROACH_POSTURE
        posture_weight = B_APPROACH_POSTURE_WEIGHT
        tolerance = B_HOVER_TOLERANCE_M if hover else B_APPROACH_TOLERANCE_M
        tilt_tolerance = B_HOVER_TILT_DEG if hover else 180.0
        target_label = "B +60 mm" if args.b_hover else "B reference"
        if args.b_hover:
            tip_goal = tip_goal + [0, 0, B_HOVER_HEIGHT_M]
            goal = ee_position(posture)
        if custom_hover:
            tip_goal = table_point(args.hover_over_mm, table_offset) + [0, 0, B_HOVER_HEIGHT_M]
            if obstacle_center is not None and box_distance(tip_goal, obstacle_center)[0] < MIN_BODY_CLEARANCE_M:
                parser.error("closed-tip goal is inside or too close to the obstacle")
            goal = tip_goal
            posture = None
            posture_weight = 0.0
            if args.reference_posture_deg is not None:
                try:
                    posture = hover_reference(args.reference_posture_deg, tip_goal, tcp, tool,
                                              table_offset, obstacle_center, target_axis)
                except ValueError as exc:
                    parser.error(str(exc))
                posture_weight = TABLE_HOVER_REFERENCE_WEIGHT
            target_label = f"Table point +60 mm ({args.grasp_direction})"
        if args.bs_midpoint_hover:
            target_label = "B–S midpoint +60 mm"
        command_lead_deg = BS_COMMAND_LEAD_DEG if tuned_hover else np.degrees(B_APPROACH_LEAD)
        max_speed_values = args.max_speed_deg_s or [BS_MAX_SPEED_DEG_S if tuned_hover else 10.0]
        rows = []
        scenarios = itertools.product(
            args.period, args.terminal_gain, self_collision_weights,
            args.vertical_weight, args.smoothness_weight, args.settle_weight or [0.0],
            args.obstacle_weight, args.body_margin_mm, args.decision_variable,
            args.max_acceleration_deg_s2, max_speed_values, args.deadband_factor
        )
        for (period, terminal_gain, self_collision_weight, vertical_weight, smoothness_weight,
             settle_weight, obstacle_weight, body_margin, decision_variable,
             max_acceleration_deg_s2, max_speed_deg_s, factor) in scenarios:
            trace = [] if args.trace_dir is not None else None
            result = simulate(
                start, goal, tip_goal, tcp, rates, delays, deadband * factor,
                TABLE_HOVER_WEIGHTS if custom_hover else B_APPROACH_WEIGHTS,
                track_tip=True, period=period, command_lead_deg=command_lead_deg,
                tolerance_m=tolerance, posture=posture,
                posture_weight=posture_weight,
                control_goal=tip_goal if hover else ee_position(posture, tcp),
                approach_geometry=(tool, table_offset), trace=trace,
                vertical_weight=vertical_weight, record_blocked=True, duration_s=args.duration_s,
                tilt_tolerance_deg=tilt_tolerance, target_label=target_label,
                smoothness_weight=smoothness_weight,
                settle_weight=settle_weight, require_settle=args.settle_weight is not None,
                obstacle_center=obstacle_center, obstacle_weight=obstacle_weight, whole_arm=args.whole_arm,
                guard_obstacle=args.guard_obstacle, obstacle_body_margin=body_margin / 1000,
                detour=args.detour_j1, terminal_gain=terminal_gain,
                via=table_point(args.via_mm, table_offset) if args.via_mm is not None else None,
                terminal_body_margin=TABLE_HOVER_BODY_MARGIN_M if custom_hover else AVOIDANCE_MARGIN_M,
                protect_tool_height=custom_hover,
                max_speed_deg_s=max_speed_deg_s, self_collision_weight=self_collision_weight,
                qp_only=args.qp_only,
                decision_variable=decision_variable,
                max_acceleration_deg_s2=max_acceleration_deg_s2,
                target_axis=target_axis,
            )
            result["min_tip_z_mm"] += 1000 * table_offset
            rows.append(dict(period_s=period, terminal_gain=terminal_gain,
                             max_speed_deg_s=max_speed_deg_s, command_lead_deg=command_lead_deg,
                             self_collision_weight=self_collision_weight,
                             qp_only=args.qp_only,
                             decision_variable=decision_variable,
                             max_acceleration_deg_s2=max_acceleration_deg_s2,
                             vertical_weight=vertical_weight, smoothness_weight=smoothness_weight,
                             acceleration_weight=smoothness_weight * period**2,
                             settle_weight=settle_weight, obstacle_weight=obstacle_weight, body_margin_mm=body_margin,
                             obstacle_margin_mm=1000 * AVOIDANCE_MARGIN_M if obstacle_center is not None else None,
                             deadband_factor=factor, **result))
            if trace is not None:
                name = "b_nominal.csv" if factor == 1.0 else "b_deadband_125.csv"
                write_results(args.trace_dir / name, trace)
        write_results(args.output, rows)
        return
    near = np.radians(np.asarray(B_CONTACT_DEG) + [0, 5, 0, 0, 0, 0])
    baseline = (5.0, 2.0, 4e-4)  # Weights used by the historical hardware run.
    print("Historical target, nominal model:",
          simulate(start, logged_goal, tip_goal, tcp, rates, delays, deadband, baseline))

    rows = []
    for period, lead, *weights in itertools.product(args.period, args.command_lead_deg,
                                               (0.2, 1.0, 5.0, 20.0),
                                               (1.0, 2.0, 4.0), (1e-4, 4e-4)):
        for scenario, initial, factor in (
            ("far_nominal", start, 1.0), ("far_larger_deadband", start, 1.25),
            ("near_nominal", near, 1.0),
        ):
            result = simulate(initial, goal, tip_goal, tcp, rates, delays, deadband * factor,
                              weights, track_tip=args.track_tip, period=period,
                              command_lead_deg=lead, tolerance_m=args.tolerance_mm / 1000,
                              duration_s=args.duration_s)
            result["min_tip_z_mm"] += 1000 * calibration["model_to_table_z_offset_m"]
            rows.append(dict(period_s=period, command_lead_deg=lead, position_weight=weights[0], x_weight=weights[1], brake_weight=weights[2],
                             scenario=scenario, **result))
    write_results(args.output, rows)


def write_results(output, rows):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
