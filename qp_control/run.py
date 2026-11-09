"""Run EE tracking on the Jetson using QP joint position commands."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
import math
from pathlib import Path
import signal
import time
import uuid

import numpy as np
import osqp
from scipy import sparse

from mycobot_id.kinematics import (
    JOINT_LIMITS_DEG, ToolGeometry, forward_kinematics, self_clearance, self_collision_pairs,
    validate_path,
)
from mycobot_id.robot import MyCobotRobot
from qp_control.obstacle import (
    AVOIDANCE_MARGIN_M, LINK_FRACTIONS, MIN_BODY_CLEARANCE_M, TUNED_BODY_MARGIN_M, TUNED_OBSTACLE_WEIGHT,
    body_clearances, body_spheres, box_distance, closest_body_spheres,
)


JOINT_MIN, JOINT_MAX = np.radians(JOINT_LIMITS_DEG).T
MAX_SPEED = np.radians(10.0)
MAX_ACCELERATION = np.radians(125.0)
SEND_SPEED = 10
TOLERANCE_M = 0.003
REGULARIZATION = 1e-5
POSITION_WEIGHT = 5.0
X_POSITION_WEIGHT = 1.0
BRAKE_RADIUS_M = 0.015
BRAKE_WEIGHT = 1e-4
POST_STOP_DELAY_S = 0.25
DEADLINE_MISS_LIMIT = 10
ANGLE_READ_FAILURE_LIMIT = 3
B_CONTACT_DEG = [-2.10, -74.97, -20.56, 5.62, -1.31, -53.34]
B_APPROACH_PERIOD = 0.1
B_HOVER_PERIOD = 0.08
B_APPROACH_LEAD = np.radians(2.0)
B_APPROACH_TOLERANCE_M = 0.02
B_APPROACH_WEIGHTS = (1.0, 4.0, 1e-4)
# FK reference found with at least 22.5 mm of modeled tool clearance.
B_APPROACH_POSTURE = np.radians([-40.37574, -113.83064, 0.0, 13.74792, 83.60022, -100.14459])
B_APPROACH_POSTURE_WEIGHT = 0.1
WORKSPACE_NEGATIVE_Z = np.array([0.0, 0.0, -1.0])
B_HOVER_POSTURE = np.radians([-2.2886, -61.2823, -12.5365, -16.1812, 0.0, -106.6733])
B_HOVER_HEIGHT_M = 0.06
B_HOVER_TOLERANCE_M = 0.005
B_HOVER_TILT_DEG = 5.0
B_HOVER_SPEED_M_S = 0.01
B_HOVER_SETTLE_SAMPLES = 3
B_HOVER_TERMINAL_GAIN = 4.0
B_HOVER_SMOOTHNESS = 0.008
B_HOVER_TERMINAL_RADIUS_M = 0.03
B_HOVER_BRAKE_RADIUS_M = 0.01
B_HOVER_TERMINAL_SPEED = np.radians(10.0)
TABLE_HOVER_WEIGHTS = (4.0, 1.0, 1e-4)
TABLE_HOVER_VERTICAL_WEIGHT = 0.2
TABLE_HOVER_BODY_MARGIN_M = 0.009
TABLE_HOVER_REFERENCE_WEIGHT = 0.2
SELF_CLEARANCE_MARGIN_M = 0.030
BS_MIDPOINT_MM = (116.155, -68.905, 0.0)
BS_MAX_SPEED_DEG_S = 25.0
BS_COMMAND_LEAD_DEG = 3.5
BS_TERMINAL_GAIN = 8.0
BS_SELF_COLLISION_WEIGHT = 20.0
FAST_HOVER_ACCELERATION = np.radians(500.0)
QP_ONLY_SELF_COLLISION_WEIGHT = 300.0
WAYPOINT_TOLERANCE_M = 0.035
DETOUR_J1_DEG = -25.0
DETOUR_START_DEG = np.array([-1.0, -0.5, -0.5, -0.9, 89.0, -44.0])
UPPER_TRIANGLE = np.tril_indices(6)[::-1]  # Column-major upper triangle for CSC.
NO_TOOLS = ToolGeometry()


def ee_position(q, tcp_offset=None):
    flange = forward_kinematics(np.degrees(q)).transforms[-1]
    return point_on_flange(flange, tcp_offset)


def point_on_flange(flange, tcp_offset):
    if tcp_offset is None:
        return flange[:3, 3]
    return flange[:3, 3] + flange[:3, :3] @ tcp_offset


def grasp_axis(direction="vertical", angle_deg=0.0):
    if direction == "vertical":
        return WORKSPACE_NEGATIVE_Z.copy()
    if direction != "horizontal":
        raise ValueError("grasp direction must be 'vertical' or 'horizontal'")
    if not math.isfinite(angle_deg):
        raise ValueError("grasp angle must be finite")
    angle = math.radians(angle_deg)
    return np.array([math.cos(angle), math.sin(angle), 0.0])


def tool_tilt_deg(q, target_axis=WORKSPACE_NEGATIVE_Z):
    axis = forward_kinematics(np.degrees(q)).transforms[-1][:3, 2]
    return np.degrees(np.arccos(np.clip(axis @ target_axis, -1.0, 1.0)))


def command_window(q, previous, lead):
    lower = np.minimum(np.maximum(q - lead, JOINT_MIN), previous)
    upper = np.maximum(np.minimum(q + lead, JOINT_MAX), previous)
    return lower, upper


def accumulate_command(q, previous, target, lead):
    lower, upper = command_window(q, previous, lead)
    return np.clip(previous + (target - q), lower, upper)


def detour_posture(q):
    if np.any(np.abs(np.degrees(q) - DETOUR_START_DEG) > 5):
        raise ValueError("J1 detour requires the tested upright starting pose (within 5 deg per joint)")
    target = q.copy()
    target[0] = np.radians(DETOUR_J1_DEG)
    return target


def detour_command(q, previous, waypoint, period):
    target = previous.copy()
    step = np.clip(waypoint[0] - previous[0], -period * MAX_SPEED, period * MAX_SPEED)
    lower = min(q[0] - B_APPROACH_LEAD, previous[0])
    upper = max(q[0] + B_APPROACH_LEAD, previous[0])
    target[0] = np.clip(previous[0] + step, lower, upper)
    return target


def b_approach_setup():
    path = Path(__file__).resolve().parents[1] / "configs" / "robot_pose_gui.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    calibration = config["floor_calibration"]
    if not calibration["enabled"]:
        raise ValueError("B approach requires the closed-tip and table calibration")
    tcp = np.asarray(calibration["closed_tip_midpoint_flange_m"])
    tool = ToolGeometry(**config["tool"])
    table_offset = calibration["model_to_table_z_offset_m"]
    goal = ee_position(np.radians(B_CONTACT_DEG), tcp)
    return tcp, goal, tool, table_offset


def table_point(xyz_mm, table_offset):
    return np.asarray(xyz_mm) / 1000 - [0, 0, table_offset]


def hover_reference(angles_deg, goal, tcp, tool, table_offset, obstacle_center,
                    target_axis=WORKSPACE_NEGATIVE_Z):
    q = np.radians(angles_deg)
    if not np.all(np.isfinite(q)) or np.any(q < JOINT_MIN) or np.any(q > JOINT_MAX):
        raise ValueError("reference posture violates joint limits")
    if (np.linalg.norm(ee_position(q, tcp) - goal) > 0.001
            or tool_tilt_deg(q, target_axis) > 1.0):
        raise ValueError("reference posture must reach the hover goal within 1 mm and 1 degree")
    if issue := b_path_issue(q, q, tcp, tool, table_offset, obstacle_center):
        raise ValueError(f"reference posture blocked: {issue}")
    return q


def b_path_issue(q, target, tcp, tool, table_offset, obstacle_center=None):
    actual, commanded = np.degrees([q, target])
    check = validate_path(actual, commanded, step_deg=1.0, tool=tool)
    if not check.ok:
        return check.message
    steps = max(1, math.ceil(float(np.max(np.abs(commanded - actual)))))
    for angles in np.linspace(actual, commanded, steps + 1):
        geometry = forward_kinematics(angles, tool)
        flange = geometry.transforms[-1]
        tip = flange[:3, 3] + flange[:3, :3] @ tcp
        lowest = min(tip[2], *(box.corners()[:, 2].min() for box in geometry.tool_boxes))
        if lowest + table_offset < 0.02:
            return "Tool path has less than 20 mm modeled table clearance"
        if obstacle_center is not None:
            clearance = np.min(body_clearances(*body_spheres(geometry), obstacle_center))
            if clearance < MIN_BODY_CLEARANCE_M:
                return f"Arm-to-obstacle clearance {1000 * clearance:.1f} mm is below 5 mm"
    return None


class PositionQP:
    def __init__(self, period, *, position_weight=POSITION_WEIGHT,
                 x_position_weight=X_POSITION_WEIGHT, brake_weight=BRAKE_WEIGHT, tcp_offset=None,
                 posture=None, posture_weight=0.0, vertical_weight=0.0,
                 table_offset=None, smoothness_weight=0.0, settle_weight=0.0,
                 obstacle_center=None, obstacle_weight=0.0, obstacle_tool=None,
                 obstacle_body_margin=AVOIDANCE_MARGIN_M, terminal_gain=1.0,
                 terminal_body_margin=AVOIDANCE_MARGIN_M, table_tool=None,
                 max_speed=MAX_SPEED, self_collision_weight=0.0, command_lead=B_APPROACH_LEAD,
                 decision_variable="delta_q", max_acceleration=MAX_ACCELERATION,
                 target_axis=WORKSPACE_NEGATIVE_Z):
        if decision_variable not in {"velocity", "delta_q"}:
            raise ValueError("decision_variable must be 'velocity' or 'delta_q'")
        self.period = period
        self.decision_variable = decision_variable
        # The optimization variable x maps to a one-step joint change as
        # delta_q = motion_scale * x.  The legacy formulation uses x=qdot;
        # the position-command formulation uses x=delta_q directly.
        self.motion_scale = period if decision_variable == "velocity" else 1.0
        self.position_weight_sqrt = np.sqrt(position_weight * np.array([x_position_weight, 1.0, 1.0]))
        self.brake_weight = brake_weight
        self.tcp_offset = tcp_offset
        self.posture = posture
        self.posture_weight = posture_weight
        self.vertical_weight_sqrt = np.sqrt(vertical_weight)
        self.table_offset = table_offset
        self.table_tool = table_tool
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        target_axis = np.asarray(target_axis, dtype=float)
        if target_axis.shape != (3,) or not np.all(np.isfinite(target_axis)):
            raise ValueError("target_axis must contain three finite values")
        norm = np.linalg.norm(target_axis)
        if norm <= 0:
            raise ValueError("target_axis must be nonzero")
        self.target_axis = target_axis / norm
        self.command_lead = command_lead
        self.self_collision_weight_sqrt = np.sqrt(self_collision_weight)
        # The historical smoothness term was
        #   smoothness_weight * ||qdot_k - qdot_(k-1)||^2.
        # Record and implement it explicitly as an angular-acceleration cost
        #   acceleration_weight * ||qddot_k||^2
        # with acceleration_weight = smoothness_weight * period^2.  This keeps
        # every previously tuned simulation exactly comparable.
        self.acceleration_weight = smoothness_weight * period**2
        self.settle_weight_sqrt = np.sqrt(settle_weight)
        self.obstacle_center = obstacle_center
        self.obstacle_weight_sqrt = np.sqrt(obstacle_weight)
        self.obstacle_tool = obstacle_tool
        self.obstacle_body_margin = obstacle_body_margin
        self.terminal_body_margin = terminal_body_margin
        self.terminal_gain = terminal_gain
        self.solver = osqp.OSQP()
        pattern = sparse.csc_matrix(np.triu(np.ones((6, 6))) + np.eye(6))
        constraints = sparse.eye(6, format="csc")
        variable_speed_limit = self.max_speed * period / self.motion_scale
        lower = np.full(6, -variable_speed_limit)
        upper = np.full(6, variable_speed_limit)
        if table_offset is not None:
            height_rows = 1
            if table_tool is not None:
                height_rows += 8 * (int(table_tool.include_gripper) + int(table_tool.include_camera))
            constraints = sparse.vstack((constraints, np.ones((height_rows, 6))), format="csc")
            self.height_indices = np.flatnonzero(constraints.indices >= 6).astype(np.int32)
            lower = np.r_[lower, np.full(height_rows, -np.inf)]
            upper = np.r_[upper, np.full(height_rows, np.inf)]
        self.solver.setup(P=pattern, q=np.zeros(6), A=constraints, l=lower, u=upper,
                          verbose=False, eps_abs=1e-5, eps_rel=1e-5,
                          time_limit=period / 2)

    def command(self, q, position, goal, *, previous_velocity=None, command_base=None):
        geometry = forward_kinematics(np.degrees(q), self.obstacle_tool or self.table_tool or NO_TOOLS)
        # In the modified DH convention each joint rotates about the Z axis
        # of its own frame. Translation along that axis does not affect it.
        frames = np.asarray(geometry.transforms[1:])
        axes = frames[:, :3, 2]
        origins = frames[:, :3, 3]
        step = self.motion_scale * np.cross(axes, position - origins).T
        error = position - goal
        distance_to_goal = np.linalg.norm(error)
        brake = max(0.0, 1.0 - distance_to_goal / BRAKE_RADIUS_M)
        speed_weight = REGULARIZATION + self.brake_weight * brake**2
        terminal = np.clip(
            (B_HOVER_TERMINAL_RADIUS_M - distance_to_goal) / B_HOVER_TERMINAL_RADIUS_M,
            0.0, 1.0,
        )
        terminal = terminal**2 * (3 - 2 * terminal)
        tracking_gain = 1 + terminal * (self.terminal_gain - 1)
        obstacle_speed_scale = 1.0
        posture_weight = self.posture_weight / tracking_gain
        position_weights = self.position_weight_sqrt * np.sqrt(tracking_gain)
        weighted_step = position_weights[:, None] * step
        weighted_error = position_weights * error
        if self.vertical_weight_sqrt:
            axis = axes[-1]
            axis_step = self.motion_scale * np.cross(axes, axis).T
            vertical_weight = self.vertical_weight_sqrt * np.sqrt(tracking_gain)
            weighted_step = np.vstack((weighted_step, vertical_weight * axis_step))
            weighted_error = np.r_[weighted_error, vertical_weight * (axis - self.target_axis)]
        if self.settle_weight_sqrt:
            settle_radius = B_HOVER_BRAKE_RADIUS_M if self.terminal_gain > 1.0 else 0.03
            ramp = max(0.0, 1.0 - distance_to_goal / settle_radius)
            speed_map = (self.settle_weight_sqrt * ramp * np.sqrt(tracking_gain) / self.period) * step
            lead_speed = np.zeros(3)
            if command_base is not None:
                lead_speed = speed_map @ (command_base - q) / self.motion_scale
            weighted_step = np.vstack((weighted_step, speed_map))
            weighted_error = np.r_[weighted_error, lead_speed]
        if self.obstacle_center is not None and self.obstacle_weight_sqrt:
            distance, normal = box_distance(position, self.obstacle_center)
            obstacle_speed_scale = min(
                obstacle_speed_scale,
                np.clip(distance / (AVOIDANCE_MARGIN_M + 0.03), 0.5, 1.0),
            )
            distance_step = normal @ step
            if command_base is not None:
                distance += distance_step @ (command_base - q) / self.motion_scale
            if distance < AVOIDANCE_MARGIN_M:
                weighted_step = np.vstack((weighted_step, self.obstacle_weight_sqrt * distance_step))
                weighted_error = np.r_[weighted_error, self.obstacle_weight_sqrt * (distance - AVOIDANCE_MARGIN_M)]
            if self.obstacle_tool is not None:
                body_margin = np.clip(distance_to_goal / 2, self.terminal_body_margin, self.obstacle_body_margin)
                centers, radii = body_spheres(geometry)
                clearances = body_clearances(centers, radii, self.obstacle_center)
                obstacle_speed_scale = min(
                    obstacle_speed_scale,
                    np.clip(np.min(clearances) / (self.obstacle_body_margin + 0.03), 0.5, 1.0),
                )
                active = [i for i in closest_body_spheres(clearances) if clearances[i] < body_margin + 0.02]
                if active:
                    # Link sphere centers interpolate joint origins. Tool spheres
                    # move rigidly with the flange. Downstream joints do not move
                    # upstream link endpoints.
                    joints = geometry.joints
                    joint_steps = np.cross(axes, joints[:, None, :] - origins)
                    joint_steps[np.arange(6) >= np.arange(7)[:, None]] = 0
                    body_step = self.motion_scale * np.cross(axes, centers[active, None, :] - origins)
                    for row, index in enumerate(active):
                        link, sample = divmod(index, len(LINK_FRACTIONS))
                        if link < 6:
                            fraction = LINK_FRACTIONS[sample, 0]
                            body_step[row] = self.motion_scale * (
                                (1 - fraction) * joint_steps[link] + fraction * joint_steps[link + 1]
                            )
                    for index, jacobian in zip(active, body_step):
                        _, normal = box_distance(centers[index], self.obstacle_center)
                        distance_step = jacobian @ normal
                        distance = clearances[index]
                        if command_base is not None:
                            distance += distance_step @ (command_base - q) / self.motion_scale
                        if distance < body_margin:
                            weighted_step = np.vstack((weighted_step, self.obstacle_weight_sqrt * distance_step))
                            weighted_error = np.r_[weighted_error, self.obstacle_weight_sqrt *
                                                   (distance - body_margin)]
        if self.self_collision_weight_sqrt:
            tool = self.table_tool or self.obstacle_tool or NO_TOOLS
            pairs = list(self_collision_pairs(geometry))
            clearances = np.array([self_clearance(geometry, tool, pair) for pair in pairs])
            # Differentiate only pairs near the soft margin. The nonlinear
            # path guard still checks every pair before sending the command.
            nearby = clearances < SELF_CLEARANCE_MARGIN_M + 0.015
            if np.any(nearby):
                selected = [pair for pair, close in zip(pairs, nearby) if close]
                delta = 1e-4
                poses = [forward_kinematics(np.degrees(q + offset), tool) for offset in delta * np.eye(6)]
                predicted = clearances[nearby]
                clearance_jacobian = (
                    np.array([[self_clearance(pose, tool, pair) for pose in poses] for pair in selected])
                    - predicted[:, None]
                ) / delta
                if command_base is not None:
                    predicted = predicted + clearance_jacobian @ (command_base - q)
                active = predicted < SELF_CLEARANCE_MARGIN_M
                if np.any(active):
                    weight = self.self_collision_weight_sqrt
                    weighted_step = np.vstack((weighted_step, weight * self.motion_scale * clearance_jacobian[active]))
                    weighted_error = np.r_[weighted_error, weight * (predicted[active] - SELF_CLEARANCE_MARGIN_M)]
        velocity_scale = self.motion_scale / self.period
        acceleration_scale = velocity_scale / self.period
        acceleration_quadratic_weight = self.acceleration_weight * acceleration_scale**2
        speed_weight *= velocity_scale**2
        speed_weight += posture_weight * self.motion_scale**2 + acceleration_quadratic_weight
        gram = np.einsum("ki,kj->ij", weighted_step, weighted_step)
        hessian = 2 * (gram + speed_weight * np.eye(6))
        gradient = 2 * np.einsum("ki,k->i", weighted_step, weighted_error)
        if self.posture is not None:
            gradient += 2 * posture_weight * self.motion_scale * (q - self.posture)
        if previous_velocity is not None:
            previous_variable = previous_velocity * self.period / self.motion_scale
            gradient -= 2 * acceleration_quadratic_weight * previous_variable
        terminal_speed = min(self.max_speed, B_HOVER_TERMINAL_SPEED)
        local_speed_limit = (
            (1.0 - terminal) * self.max_speed + terminal * terminal_speed
        ) * obstacle_speed_scale
        variable_speed_limit = local_speed_limit * self.period / self.motion_scale
        lower = np.maximum(-variable_speed_limit, (JOINT_MIN - q) / self.motion_scale)
        upper = np.minimum(variable_speed_limit, (JOINT_MAX - q) / self.motion_scale)
        if self.max_acceleration is not None:
            previous_velocity_for_limit = (
                np.zeros(6) if previous_velocity is None else previous_velocity
            )
            acceleration_step = self.max_acceleration * self.period
            lower = np.maximum(
                lower,
                (previous_velocity_for_limit - acceleration_step)
                * self.period / self.motion_scale,
            )
            upper = np.minimum(
                upper,
                (previous_velocity_for_limit + acceleration_step)
                * self.period / self.motion_scale,
            )
        if command_base is not None and (self.table_tool is not None or (self.terminal_gain > 1.0 and terminal > 0)):
            # Allocate correction to joints whose next command can actually move
            # instead of relying on an increment that will be clipped after solving.
            command_lower, command_upper = command_window(q, command_base, self.command_lead)
            lower = np.maximum(lower, (command_lower - command_base) / self.motion_scale)
            upper = np.minimum(upper, (command_upper - command_base) / self.motion_scale)
        bound_violation = float(np.max(lower - upper))
        if bound_violation > 1e-12:
            return None, f"acceleration_bounds_infeasible:{bound_violation:.3e}"
        bounds = dict(l=lower, u=upper)
        if self.table_offset is not None:
            heights = np.array([position[2]])
            height_steps = step[2:3]
            minimum_height = 0.0
            if self.table_tool is not None:
                corners = np.concatenate([box.corners() for box in geometry.tool_boxes]) if geometry.tool_boxes else np.empty((0, 3))
                corner_steps = self.motion_scale * np.cross(axes, corners[:, None, :] - origins)[:, :, 2]
                heights = np.r_[heights, corners[:, 2]]
                height_steps = np.vstack((height_steps, corner_steps))
                minimum_height = 0.027
            base_height = heights + self.table_offset
            if command_base is not None:
                base_height += height_steps @ (command_base - q) / self.motion_scale
            bounds = dict(l=np.r_[lower, minimum_height - base_height],
                          u=np.r_[upper, np.full(len(heights), np.inf)],
                          Ax=height_steps.ravel(order="F"), Ax_idx=self.height_indices)
        self.solver.update(Px=hessian[UPPER_TRIANGLE], q=gradient, **bounds)
        result = self.solver.solve()
        status = result.info.status
        if result.info.status_val != 1 or not np.all(np.isfinite(result.x)):
            return None, status
        # Remove numerical residuals at active bounds before building the command.
        variable = np.clip(result.x, lower, upper)
        target = np.clip(q + self.motion_scale * variable, JOINT_MIN, JOINT_MAX)
        return target, status


def run(robot, qp, goal, duration, log_path, *, approach_geometry=None, control_goal=None,
        hover=False, detour=False, target_label=None, via=None,
        realtime_path_guard=True):
    period = qp.period
    tolerance = TOLERANCE_M if approach_geometry is None else B_APPROACH_TOLERANCE_M
    previous = None
    previous_velocity = None
    previous_position = None
    previous_sample = None
    settled_samples = 0
    waypoint = None
    waypoint_samples = 0
    if hover:
        tolerance = B_HOVER_TOLERANCE_M
    control_goal = goal if control_goal is None else control_goal
    fieldnames = [
        "sample_s", "period_s", "read_ms", "qp_ms", "guard_ms", "send_ms", "work_ms",
        "deadline_missed", "status", "ee_error_mm", "tolerance_mm", "ee_speed_mm_s", "tool_tilt_deg",
        "target_label", "path_issue", "obstacle_x_mm", "obstacle_y_mm", "obstacle_z_mm",
    ]
    fieldnames += [f"{name}_{axis}_mm" for name in ("ee", "goal") for axis in "xyz"]
    fieldnames += [f"q_{name}_j{i}_deg" for name in ("actual", "cmd") for i in range(1, 7)]
    missed = 0
    read_failures = 0
    if target_label is None:
        target_label = "B +60 mm" if hover else "B reference" if approach_geometry is not None else "Goal"
    obstacle_mm = [None] * 3
    if qp.obstacle_center is not None:
        obstacle_mm = 1000 * (qp.obstacle_center + [0, 0, approach_geometry[1]])
    with log_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fieldnames)
        started = scheduled = time.monotonic()
        while scheduled < started + duration:
            time.sleep(max(0.0, scheduled - time.monotonic()))
            tick = time.monotonic()
            if tick >= started + duration:
                break
            deadline = scheduled + period
            try:
                q = np.radians(robot.get_angles())
            except RuntimeError as exc:
                read_done = finished = time.monotonic()
                read_failures += 1
                overrun = finished >= deadline
                writer.writerow([
                    tick - started, period, (read_done - tick) * 1000,
                    None, None, None, (finished - tick) * 1000, int(overrun),
                    "angle_read_failed", None, tolerance * 1000, None, None,
                    target_label, str(exc), *obstacle_mm, *([None] * 3), *(goal * 1000),
                    *([None] * 12),
                ])
                stream.flush()
                if read_failures >= ANGLE_READ_FAILURE_LIMIT:
                    return "repeated_angle_read_failure"
                scheduled = deadline + max(
                    0, math.ceil((finished - deadline) / period)
                ) * period
                continue
            read_done = time.monotonic()
            read_failures = 0
            if detour and waypoint is None:
                waypoint = detour_posture(q)
                previous = q
            if detour:
                waypoint_samples = waypoint_samples + 1 if np.all(
                    np.abs(q - waypoint) <= np.radians(1.25)
                ) else 0
                detour = waypoint_samples < 3
            position = ee_position(q, qp.tcp_offset)
            if via is not None and np.linalg.norm(position - via) <= WAYPOINT_TOLERANCE_M:
                via = None
            error = np.linalg.norm(position - goal)
            sample = (tick + read_done) / 2
            speed = tilt = None
            reached = error <= tolerance
            if hover:
                tilt = tool_tilt_deg(q, qp.target_axis)
                if previous_sample is not None:
                    interval = sample - previous_sample
                    speed = np.linalg.norm(position - previous_position) / interval
                    if (via is None and reached and tilt <= B_HOVER_TILT_DEG
                            and speed <= B_HOVER_SPEED_M_S and interval <= 1.5 * period):
                        settled_samples += 1
                    else:
                        settled_samples = 0
                reached = settled_samples >= B_HOVER_SETTLE_SAMPLES
                previous_position, previous_sample = position, sample
            target = None
            qp_ms = guard_ms = send_ms = None
            sent = [None] * 6
            issue = None
            if np.any(q < JOINT_MIN) or np.any(q > JOINT_MAX):
                status = "joint_angle_out_of_bounds"
            elif reached and not detour and via is None:
                status = "goal_entered"
            else:
                qp_started = time.monotonic()
                if detour:
                    target = detour_command(q, previous, waypoint, period)
                    status = "detour_j1"
                else:
                    target, status = qp.command(q, position, control_goal if via is None else via,
                                                previous_velocity=previous_velocity,
                                                command_base=previous if hover else None)
                    if target is not None and via is not None:
                        status = "via"
                qp_ms = (time.monotonic() - qp_started) * 1000
                if target is not None and approach_geometry is not None:
                    if previous is None:
                        previous = q
                    if not detour:
                        target = accumulate_command(q, previous, target, qp.command_lead)
                    if realtime_path_guard:
                        guard_started = time.monotonic()
                        issue = b_path_issue(q, target, qp.tcp_offset, *approach_geometry,
                                             obstacle_center=qp.obstacle_center)
                        guard_ms = (time.monotonic() - guard_started) * 1000
                        if issue:
                            print(f"B approach blocked: {issue}", flush=True)
                            target, status = None, "path_blocked"
                if target is not None:
                    send_started = time.monotonic()
                    if send_started < min(deadline, started + duration):
                        previous_for_velocity = q if previous is None else previous
                        sent = np.degrees(target).tolist()
                        robot.send_angles(sent, SEND_SPEED)
                        previous_velocity = (target - previous_for_velocity) / period
                        previous = target
                        send_ms = (time.monotonic() - send_started) * 1000
                    else:
                        status = "command_skipped"
            finished = time.monotonic()
            overrun = finished >= deadline
            writer.writerow([
                sample - started, period, (read_done - tick) * 1000,
                qp_ms, guard_ms, send_ms, (finished - tick) * 1000, int(overrun), status,
                error * 1000, tolerance * 1000, None if speed is None else speed * 1000, tilt,
                target_label, issue, *obstacle_mm, *(position * 1000), *(goal * 1000),
                *np.degrees(q), *sent,
            ])
            stream.flush()
            if status == "goal_entered":
                robot.stop()
                time.sleep(POST_STOP_DELAY_S)
                read_started = time.monotonic()
                final_q = np.radians(robot.get_angles())
                read_done = time.monotonic()
                final_position = ee_position(final_q, qp.tcp_offset)
                final_error = np.linalg.norm(final_position - goal)
                in_bounds = np.all((JOINT_MIN <= final_q) & (final_q <= JOINT_MAX))
                final_sample = (read_started + read_done) / 2
                final_tilt = tool_tilt_deg(final_q, qp.target_axis) if hover else None
                final_speed = np.linalg.norm(final_position - position) / (final_sample - sample) if hover else None
                held = in_bounds and final_error <= tolerance
                if hover:
                    held = (held and final_tilt <= B_HOVER_TILT_DEG and final_speed <= B_HOVER_SPEED_M_S
                            and b_path_issue(final_q, final_q, qp.tcp_offset, *approach_geometry,
                                             obstacle_center=qp.obstacle_center) is None)
                status = "goal_reached" if held else "goal_not_held"
                writer.writerow([
                    final_sample - started, period,
                    (read_done - read_started) * 1000, None, None, None, None, None,
                    status, final_error * 1000, tolerance * 1000,
                    None if final_speed is None else final_speed * 1000, final_tilt,
                    target_label, None, *obstacle_mm, *(final_position * 1000), *(goal * 1000),
                    *np.degrees(final_q), *sent,
                ])
                return status
            if target is None:
                return status
            missed = missed + 1 if overrun else 0
            if missed >= DEADLINE_MISS_LIMIT:
                return "repeated_deadline_miss"
            # Skip expired slots so a late cycle cannot trigger a burst of commands.
            scheduled = deadline + max(0, math.ceil((finished - deadline) / period)) * period
        return "duration_elapsed"


def hover_qp(tcp, table_offset, *, period=B_HOVER_PERIOD, obstacle_center=None, tool=None,
             custom=False, reference=None, bs_midpoint=False,
             target_axis=WORKSPACE_NEGATIVE_Z):
    weight, x_weight, brake_weight = TABLE_HOVER_WEIGHTS if custom else B_APPROACH_WEIGHTS
    return PositionQP(
        period, position_weight=weight, x_position_weight=x_weight,
        brake_weight=brake_weight, tcp_offset=tcp, posture=reference if custom else B_HOVER_POSTURE,
        posture_weight=(TABLE_HOVER_REFERENCE_WEIGHT if reference is not None else 0.0)
        if custom else B_APPROACH_POSTURE_WEIGHT,
        vertical_weight=TABLE_HOVER_VERTICAL_WEIGHT if custom else 0.05,
        table_offset=table_offset, smoothness_weight=B_HOVER_SMOOTHNESS, settle_weight=0.05,
        terminal_gain=BS_TERMINAL_GAIN if bs_midpoint else B_HOVER_TERMINAL_GAIN,
        obstacle_center=obstacle_center,
        obstacle_weight=TUNED_OBSTACLE_WEIGHT if obstacle_center is not None else 0.0,
        obstacle_tool=tool if obstacle_center is not None else None,
        obstacle_body_margin=TUNED_BODY_MARGIN_M,
        terminal_body_margin=TABLE_HOVER_BODY_MARGIN_M if custom else AVOIDANCE_MARGIN_M,
        table_tool=tool if custom else None,
        max_speed=np.radians(BS_MAX_SPEED_DEG_S) if (custom or bs_midpoint) else MAX_SPEED,
        command_lead=np.radians(BS_COMMAND_LEAD_DEG) if (custom or bs_midpoint) else B_APPROACH_LEAD,
        self_collision_weight=(BS_SELF_COLLISION_WEIGHT if bs_midpoint
                               else QP_ONLY_SELF_COLLISION_WEIGHT if custom else 0.0),
        max_acceleration=FAST_HOVER_ACCELERATION if (custom or bs_midpoint) else MAX_ACCELERATION,
        target_axis=target_axis,
    )


def self_test(period):
    from pymycobot.mycobot import MyCobot  # Verify import without opening serial.

    qp = PositionQP(period)
    q = np.zeros(6)
    position = ee_position(q)
    target, status = qp.command(q, position, position)
    if target is None or not np.allclose(target, q, atol=1e-6):
        raise RuntimeError(f"QP zero-error test failed: {status}")
    for q in (np.zeros(6), JOINT_MAX.copy()):
        position = ee_position(q)
        target, status = qp.command(q, position, position + [0.01, -0.01, 0.01])
        if (target is None or np.any(target < JOINT_MIN) or np.any(target > JOINT_MAX)
                or np.any(np.abs(target - q) > period * MAX_SPEED + 1e-12)):
            raise RuntimeError(f"QP self-test failed: {status}")
    tcp, goal, tool, table_offset = b_approach_setup()
    q = np.radians([0, 0, 0, 0, 89, -44])
    weight, x_weight, brake_weight = B_APPROACH_WEIGHTS
    approach = PositionQP(B_APPROACH_PERIOD, position_weight=weight, x_position_weight=x_weight,
                          brake_weight=brake_weight, tcp_offset=tcp, posture=B_APPROACH_POSTURE,
                          posture_weight=B_APPROACH_POSTURE_WEIGHT)
    target, status = approach.command(q, ee_position(q, tcp), ee_position(B_APPROACH_POSTURE, tcp))
    if target is None or (issue := b_path_issue(q, target, tcp, tool, table_offset)):
        raise RuntimeError(f"B approach self-test failed: {status if target is None else issue}")
    hover = hover_qp(tcp, table_offset)
    hover_goal = goal + [0, 0, B_HOVER_HEIGHT_M]
    target, status = hover.command(q, ee_position(q, tcp), hover_goal, command_base=q)
    if target is None or (issue := b_path_issue(q, target, tcp, tool, table_offset)):
        raise RuntimeError(f"B hover self-test failed: {status if target is None else issue}")
    q = B_HOVER_POSTURE
    position = ee_position(q, tcp)
    previous = q + B_APPROACH_LEAD
    target, status = hover.command(q, position, position + [0.0, 0.02, 0.0], command_base=previous)
    if target is None or np.any(target > q + 1e-12):
        raise RuntimeError(f"B hover command lead self-test failed: {status}")
    q = np.radians([0, 0, 0, 0, 89, -44])
    center = ee_position(q, tcp)
    obstacle_qp = hover_qp(tcp, table_offset, obstacle_center=center, tool=tool)
    target, status = obstacle_qp.command(q, center, hover_goal, command_base=q)
    if target is None:
        raise RuntimeError(f"Obstacle QP self-test failed: {status}")
    issue = b_path_issue(q, q, tcp, tool, table_offset, obstacle_center=center)
    if issue is None or "Arm-to-obstacle" not in issue:
        raise RuntimeError("Obstacle guard failed to detect a cube at the closed tip")
    print(f"QP self-test passed (OSQP {osqp.__version__}, adapter {MyCobot.__name__})")


def interrupt(signum, _frame):
    raise SystemExit(f"Interrupted by signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=1000000)
    parser.add_argument("--goal-mm", type=float, nargs=3, metavar=("X", "Y", "Z"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--b-approach", action="store_true")
    mode.add_argument("--b-hover", action="store_true")
    mode.add_argument("--hover-over-mm", type=float, nargs=3, metavar=("X", "Y", "Z"),
                      help="Closed-tip goal 60 mm above this table-frame point")
    mode.add_argument("--bs-midpoint-hover", action="store_true",
                      help="Use the tuned hover profile for the B–S midpoint")
    parser.add_argument("--preview", action="store_true", help="Read the B approach starting pose")
    parser.add_argument("--period", type=float)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Save Jetson process CPU and peak RSS")
    parser.add_argument("--obstacle-center-mm", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--via-mm", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Intermediate closed-tip waypoint in the table frame")
    parser.add_argument("--reference-posture-deg", type=float, nargs=6,
                        help="Collision-checked IK reference at the custom hover goal")
    parser.add_argument("--detour-j1", action="store_true", help="Use the verified J1-first obstacle detour")
    parser.add_argument("--grasp-direction", choices=("vertical", "horizontal"), default="vertical",
                        help="Align EE/J6 +Z with table -Z or a horizontal workspace direction")
    parser.add_argument("--grasp-angle-deg", type=float, default=0.0,
                        help="Horizontal direction angle from workspace +X toward +Y")
    args = parser.parse_args()
    if args.bs_midpoint_hover:
        args.hover_over_mm = BS_MIDPOINT_MM
        if args.obstacle_center_mm is not None or args.via_mm is not None:
            parser.error("--bs-midpoint-hover is tuned for an unobstructed workspace")
    custom_hover = args.hover_over_mm is not None
    hover = args.b_hover or custom_hover
    try:
        target_axis = grasp_axis(args.grasp_direction, args.grasp_angle_deg)
    except ValueError as exc:
        parser.error(str(exc))
    if args.grasp_direction == "horizontal" and not custom_hover:
        parser.error("--grasp-direction horizontal requires --hover-over-mm or --bs-midpoint-hover")
    if custom_hover and (not np.all(np.isfinite(args.hover_over_mm)) or args.hover_over_mm[2] < 0):
        parser.error("--hover-over-mm requires a finite table-frame point with Z >= 0")
    if args.via_mm is not None and (
        not custom_hover or not np.all(np.isfinite(args.via_mm)) or args.via_mm[2] < 20
    ):
        parser.error("--via-mm requires --hover-over-mm and a finite table-frame point with Z >= 20")
    if args.reference_posture_deg is not None and not (custom_hover and args.obstacle_center_mm is not None):
        parser.error("--reference-posture-deg requires --hover-over-mm and --obstacle-center-mm")
    if args.detour_j1 and (not args.b_hover or args.obstacle_center_mm is None):
        parser.error("--detour-j1 requires --b-hover and --obstacle-center-mm")
    if args.obstacle_center_mm is not None and (
        not hover or not np.all(np.isfinite(args.obstacle_center_mm))
    ):
        parser.error("--obstacle-center-mm requires a hover mode and finite workspace coordinates")
    period = args.period
    if period is None:
        period = B_HOVER_PERIOD if hover else B_APPROACH_PERIOD if args.b_approach else 0.05
    if not math.isfinite(period) or period < 0.05:
        parser.error("--period must be finite and at least 0.05 seconds")
    if args.self_test:
        self_test(period)
        return
    if args.profile:
        import resource
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            total_ram_kib = int(meminfo.readline().split()[1])
    b_mode = args.b_approach or hover
    if args.preview and not b_mode:
        parser.error("--preview requires an approach or hover mode")
    if b_mode and args.goal_mm is not None:
        parser.error("B modes use the recorded B target")
    if not args.port or (not b_mode and
                         (args.goal_mm is None or not np.all(np.isfinite(args.goal_mm)))):
        parser.error("motion requires --port and a finite --goal-mm X Y Z")
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")

    approach_geometry = None
    control_goal = None
    target_label = None
    via = None
    if b_mode:
        tcp, goal, tool, table_offset = b_approach_setup()
        approach_geometry = (tool, table_offset)
        weight, x_weight, brake_weight = B_APPROACH_WEIGHTS
        qp = PositionQP(period, position_weight=weight, x_position_weight=x_weight,
                        brake_weight=brake_weight, tcp_offset=tcp, posture=B_APPROACH_POSTURE,
                        posture_weight=B_APPROACH_POSTURE_WEIGHT)
        control_goal = ee_position(B_APPROACH_POSTURE, tcp)
        if hover:
            if custom_hover:
                goal = table_point(args.hover_over_mm, table_offset)
                target_label = f"Table point +60 mm ({args.grasp_direction} {args.grasp_angle_deg:g} deg)"
                if args.bs_midpoint_hover:
                    target_label = f"B–S midpoint +60 mm ({args.grasp_direction} {args.grasp_angle_deg:g} deg)"
                if args.via_mm is not None:
                    via = table_point(args.via_mm, table_offset)
            goal = goal + [0, 0, B_HOVER_HEIGHT_M]
            obstacle_center = None
            if args.obstacle_center_mm is not None:
                obstacle_center = table_point(args.obstacle_center_mm, table_offset)
                if box_distance(goal, obstacle_center)[0] < MIN_BODY_CLEARANCE_M:
                    parser.error("closed-tip goal is inside or too close to the obstacle")
                if via is not None and box_distance(via, obstacle_center)[0] < MIN_BODY_CLEARANCE_M:
                    parser.error("closed-tip waypoint is inside or too close to the obstacle")
            reference = None
            if args.reference_posture_deg is not None:
                try:
                    reference = hover_reference(args.reference_posture_deg, goal, tcp, tool,
                                                table_offset, obstacle_center, target_axis)
                except ValueError as exc:
                    parser.error(str(exc))
            qp = hover_qp(tcp, table_offset, period=period, obstacle_center=obstacle_center, tool=tool,
                          custom=custom_hover, reference=reference, bs_midpoint=args.bs_midpoint_hover,
                          target_axis=target_axis)
            control_goal = goal
    else:
        goal = np.asarray(args.goal_mm) / 1000
        qp = PositionQP(period)

    if args.preview:
        robot = MyCobotRobot(args.port, args.baud)
        try:
            q = np.radians(robot.get_angles())
            current = ee_position(q, tcp)
            print(f"QP control period: {1000 * qp.period:g} ms")
            print(f"Current joint angles (deg): {np.degrees(q).round(2).tolist()}")
            print(f"Current closed-tip error: {1000 * np.linalg.norm(current - goal):.2f} mm")
            if hover:
                print(f"Closed-tip goal in table frame (mm): {(1000 * (goal + [0, 0, table_offset])).round(2).tolist()}")
                print(f"Grasp direction: {args.grasp_direction}, angle: {args.grasp_angle_deg:g} deg")
                print(f"Current EE +Z alignment error: {tool_tilt_deg(q, qp.target_axis):.2f} deg")
                if via is not None:
                    print(f"Closed-tip waypoint in table frame (mm): {args.via_mm}")
            if issue := b_path_issue(q, q, tcp, tool, table_offset, obstacle_center=qp.obstacle_center):
                raise SystemExit(f"Starting pose blocked: {issue}")
            if args.detour_j1:
                try:
                    waypoint = detour_posture(q)
                except ValueError as exc:
                    raise SystemExit(str(exc)) from exc
                if issue := b_path_issue(q, waypoint, tcp, tool, table_offset, obstacle_center=qp.obstacle_center):
                    raise SystemExit(f"J1 detour blocked: {issue}")
                print(f"J1 detour to {DETOUR_J1_DEG:g} deg passed modeled clearance checks.")
            print("Starting pose passed modeled clearance checks.")
        finally:
            robot.close()
        return

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)
    output = Path("data/raw")
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = output / f"{stamp}_{uuid.uuid4().hex[:6]}_qp.csv"
    robot = MyCobotRobot(args.port, args.baud)
    status = None
    wall_start = time.monotonic()
    cpu_start = time.process_time()
    try:
        print(f"QP log: {log}", flush=True)
        status = run(robot, qp, goal, args.duration, log, approach_geometry=approach_geometry,
                     control_goal=control_goal, hover=hover, detour=args.detour_j1,
                     target_label=target_label, via=via,
                     realtime_path_guard=not custom_hover)
    finally:
        try:
            if status not in {"goal_reached", "goal_not_held"}:
                robot.stop()
        finally:
            try:
                robot.close()
            finally:
                if args.profile:
                    wall_s = time.monotonic() - wall_start
                    cpu_s = time.process_time() - cpu_start
                    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    profile = {
                        "csv": log.name,
                        "status": status or "interrupted_or_failed",
                        "wall_s": wall_s,
                        "process_cpu_s": cpu_s,
                        "cpu_percent_one_core": 100 * cpu_s / wall_s,
                        "peak_rss_mib": peak_rss_kib / 1024,
                        "peak_rss_percent_of_ram": 100 * peak_rss_kib / total_ram_kib,
                    }
                    profile_path = log.with_suffix(".profile.json")
                    profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
                    print(f"QP resource profile: {profile_path}", flush=True)
                    print(json.dumps(profile), flush=True)
    print(f"QP finished: {status}")
    if status != "goal_reached":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
