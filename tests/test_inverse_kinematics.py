import json
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from mycobot_id.inverse_kinematics import solve_closed_tip
from mycobot_id.kinematics import ToolGeometry, forward_kinematics, self_collision, validate_angles


class InverseKinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c = json.loads((Path(__file__).resolve().parents[1] / "configs/robot_pose_gui.json").read_text())
        cls.cal, cls.tool = c["floor_calibration"], ToolGeometry(**c["tool"])
        cls.seed = [36.82,-54.84,-66.53,33.48,-1.31,-58.35]

    def test_reachable_tip_maps_back_to_requested_position_and_downward_yaw(self):
        target = np.array([243.19,80.36,10.])
        result = solve_closed_tip(target, 5.16, self.cal, self.tool, self.seed)
        self.assertTrue(result.ok, result.message)
        t = forward_kinematics(result.angles_deg, self.tool).transforms[-1]
        p = (t @ np.r_[self.cal["closed_tip_midpoint_flange_m"],1.])[:3] * 1000
        p[2] += self.cal["model_to_table_z_offset_m"] * 1000
        self.assertLess(np.linalg.norm(p-target), .5)
        expected = Rotation.from_euler("z",5.16,degrees=True).as_matrix() @ np.diag([1,-1,-1])
        self.assertLess(Rotation.from_matrix(expected.T @ t[:3,:3]).magnitude(), np.radians(1))
        self.assertTrue(validate_angles(result.angles_deg).ok)
        self.assertIsNone(self_collision(result.angles_deg, self.tool))

    def test_unreachable_and_below_table_inputs_do_not_produce_motion_targets(self):
        for point in ([2000,0,0], [200,0,-1], [float("nan"),0,0]):
            result = solve_closed_tip(point,0,self.cal,self.tool,self.seed)
            self.assertFalse(result.ok)
            self.assertEqual(result.angles_deg, ())


if __name__ == "__main__":
    unittest.main()
