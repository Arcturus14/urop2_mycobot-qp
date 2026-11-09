import unittest

import numpy as np

from mycobot_id.kinematics import forward_kinematics
from qp_control.run import PositionQP, grasp_axis


class QPFormulationTests(unittest.TestCase):
    def test_grasp_axis_supports_vertical_and_horizontal_angles(self):
        np.testing.assert_allclose(grasp_axis("vertical", 37.0), [0.0, 0.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(grasp_axis("horizontal", 0.0), [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(grasp_axis("horizontal", 90.0), [0.0, 1.0, 0.0], atol=1e-12)

    def test_velocity_and_delta_q_forms_produce_same_target(self):
        period = 0.05
        q = np.radians([4.0, -18.0, -42.0, 22.0, 78.0, -35.0])
        position = forward_kinematics(np.degrees(q)).transforms[-1][:3, 3]
        goal = position + np.array([0.012, -0.008, 0.006])
        previous_velocity = np.radians([1.0, -2.0, 0.5, 0.0, -1.0, 1.5])
        options = dict(
            position_weight=4.0,
            x_position_weight=4.0,
            brake_weight=1e-4,
            smoothness_weight=0.008,
            vertical_weight=0.2,
            max_speed=np.radians(10.0),
        )
        velocity_qp = PositionQP(period, decision_variable="velocity", **options)
        delta_qp = PositionQP(period, decision_variable="delta_q", **options)

        velocity_target, velocity_status = velocity_qp.command(
            q, position, goal, previous_velocity=previous_velocity
        )
        delta_target, delta_status = delta_qp.command(
            q, position, goal, previous_velocity=previous_velocity
        )

        self.assertEqual(velocity_status, "solved")
        self.assertEqual(delta_status, "solved")
        # OSQP's absolute stopping tolerance is expressed in differently scaled
        # variables, so the reconstructed joint targets may differ by sub-microradians.
        np.testing.assert_allclose(delta_target, velocity_target, atol=1e-6, rtol=0)

    def test_legacy_smoothness_is_recorded_as_physical_acceleration_cost(self):
        period = 0.05
        qp = PositionQP(period, smoothness_weight=0.008)
        self.assertAlmostEqual(qp.acceleration_weight, 0.008 * period**2)

    def test_hard_acceleration_limit_caps_command_velocity_change(self):
        period = 0.05
        max_acceleration = np.radians(100.0)
        q = np.zeros(6)
        position = forward_kinematics(np.degrees(q)).transforms[-1][:3, 3]
        qp = PositionQP(
            period,
            position_weight=100.0,
            max_speed=np.radians(10.0),
            max_acceleration=max_acceleration,
        )
        target, status = qp.command(q, position, position + [0.1, 0.1, 0.1])
        self.assertEqual(status, "solved")
        velocity = (target - q) / period
        self.assertLessEqual(
            np.max(np.abs(velocity)), max_acceleration * period + 1e-7
        )


if __name__ == "__main__":
    unittest.main()
