"""Geometry of the axis-aligned 50 mm cube used by the QP simulation."""
from __future__ import annotations

import numpy as np


HALF_SIDE_M = 0.025
AVOIDANCE_MARGIN_M = 0.025
MIN_BODY_CLEARANCE_M = 0.005
TUNED_OBSTACLE_WEIGHT = 400.0
TUNED_BODY_MARGIN_M = 0.055
LINK_RADII_M = np.array([0.040, 0.035, 0.032, 0.030, 0.028, 0.026])
OCTANTS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
LINK_FRACTIONS = np.linspace(0, 1, 5)[:, None]


def box_distance(point, center):
    """Return signed distance to the box surface and an outward gradient."""
    relative = point - center
    excess = np.abs(relative) - HALF_SIDE_M
    outside = np.maximum(excess, 0.0)
    distance = np.linalg.norm(outside)
    if distance > 0:
        return distance, np.sign(relative) * outside / distance
    face = int(np.argmax(excess))
    normal = np.zeros(3)
    normal[face] = 1.0 if relative[face] >= 0 else -1.0
    return excess[face], normal


def body_spheres(geometry):
    """Cover each link capsule and each tool box with spheres."""
    joints = geometry.joints
    delta = np.diff(joints, axis=0)
    points = joints[:-1, None, :] + delta[:, None, :] * LINK_FRACTIONS
    radii = LINK_RADII_M + np.linalg.norm(delta, axis=1) / 8
    centers = [points.reshape(-1, 3)]
    sizes = [np.repeat(radii, len(LINK_FRACTIONS))]
    for box in geometry.tool_boxes:
        quarter = np.asarray(box.size) / 4
        centers.append((OCTANTS * quarter) @ box.transform[:3, :3].T + box.transform[:3, 3])
        sizes.append(np.full(len(OCTANTS), np.linalg.norm(quarter)))
    return np.vstack(centers), np.concatenate(sizes)


def body_clearances(centers, radii, obstacle_center):
    excess = np.abs(centers - obstacle_center) - HALF_SIDE_M
    outside = np.linalg.norm(np.maximum(excess, 0), axis=1)
    signed = outside + np.minimum(np.max(excess, axis=1), 0)
    return signed - radii


def closest_body_spheres(clearances):
    first = np.arange(0, 6 * len(LINK_FRACTIONS), len(LINK_FRACTIONS))
    tools = np.arange(6 * len(LINK_FRACTIONS), len(clearances), len(OCTANTS))
    starts = np.r_[first, tools, len(clearances)]
    return [lo + int(np.argmin(clearances[lo:hi])) for lo, hi in zip(starts, starts[1:])]
