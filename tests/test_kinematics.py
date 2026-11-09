import unittest
import json
from pathlib import Path

import numpy as np

from mycobot_id.kinematics import (
    forward_kinematics,
    self_collision,
    validate_angles,
    validate_path,
    ToolGeometry,
    ToolBox,
    segment_box_distance,
    _origin_matrix,
)


class KinematicsTests(unittest.TestCase):
    def test_zero_pose_matches_jn_reference(self):
        points = forward_kinematics([0.0] * 6).joints
        np.testing.assert_allclose(points[-1], [0.0456, -0.06462, 0.43614], atol=1e-8)

    def test_measured_vertical_pose_matches_firmware_xyz(self):
        # Captured from the actual JN while physically extended vertically.
        angles = [30.05, -3.42, 9.58, -5.71, -74.35, 6.76]
        firmware_xyz_m = np.array([60.7, -88.8, 420.2]) / 1000.0
        endpoint = forward_kinematics(angles).joints[-1]
        self.assertLess(np.linalg.norm(endpoint - firmware_xyz_m), 0.020)

    def test_zero_pose_is_not_reported_as_collision(self):
        self.assertIsNone(self_collision([0.0] * 6))

    def test_default_model_contains_no_end_effector(self):
        geometry = forward_kinematics([0.0] * 6)
        self.assertEqual(geometry.joints.shape, (7, 3))
        self.assertEqual(geometry.gripper.shape, (0, 3))
        self.assertEqual(geometry.camera.shape, (0, 3))

    def test_conservative_limits(self):
        self.assertTrue(validate_angles([0.0] * 6).ok)
        self.assertFalse(validate_angles([166.0, 0, 0, 0, 0, 0]).ok)
        self.assertFalse(validate_angles([0, 141.0, 0, 0, 0, 0]).ok)
        self.assertFalse(validate_angles([0, 0, 0, 0, 0, 176.0]).ok)

    def test_short_path_from_zero_is_valid(self):
        result = validate_path([0.0] * 6, [2.0] * 6)
        self.assertTrue(result.ok, result.message)

    def test_folded_pose_collision_is_detected(self):
        folded = [-163.13, -62.05, -141.17, -10.59, -3.28, -83.89]
        self.assertIsNotNone(self_collision(folded))

    def test_segment_box_distance_checks_segment_interior(self):
        box = ToolBox("test", np.eye(4), np.array([2.0, 2.0, 2.0]))
        self.assertEqual(segment_box_distance(np.array([-3., 0, 0]), np.array([3., 0, 0]), box), 0)
        self.assertAlmostEqual(segment_box_distance(np.array([-3., 2, 0]), np.array([3., 2, 0]), box), 1)

    def test_configured_camera_can_block_otherwise_clear_pose(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/robot_pose_gui.json").read_text())
        tool = ToolGeometry(**config["tool"])
        pose = [17.81, 4.39, -16.48, 140.22, 81.8, -94.83]
        self.assertIsNone(self_collision(pose))
        self.assertIn("camera", self_collision(pose, tool))
        self.assertFalse(validate_path(pose, pose, tool=tool).ok)

    def test_camera_box_matches_composed_official_mesh_bounds(self):
        tool = ToolGeometry(include_camera=True)
        geometry = forward_kinematics([0] * 6, tool)
        flange = geometry.transforms[-1]
        joint = _origin_matrix((0, 0, .01), (1.579, 0, 0))
        mesh = _origin_matrix((.024, -.008, 0), (0, 3.14159, 0))
        bounds = [(-.0414825439453125, .0414825439453125),
                  (0, .016), (-.016482545852661103, .0174825458526611)]
        expected = np.array([(flange @ joint @ mesh @ np.array([x,y,z,1]))[:3]
                             for x in bounds[0] for y in bounds[1] for z in bounds[2]])
        np.testing.assert_allclose(geometry.tool_boxes[0].corners(), expected, atol=1e-10)

    def test_configured_tools_are_rigid_and_clear_at_measured_pose(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/robot_pose_gui.json").read_text())
        tool = ToolGeometry(**config["tool"])
        pose = [-3.42, 5.97, 1.66, -5.09, 2.1, -45.87]
        self.assertIsNone(self_collision(pose, tool))
        geometry = forward_kinematics(pose, tool)
        self.assertEqual(len(geometry.tool_boxes), 2)
        flange = geometry.transforms[-1]
        for box in geometry.tool_boxes:
            local = np.linalg.inv(flange) @ box.transform
            center = tool.gripper_center_m if box.name == "gripper" else tool.camera_offset_m
            np.testing.assert_allclose(local[:3, 3], center, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
