import argparse
import unittest

from scripts.jetson_cartesian_move import decode_ik_response, execute, make_plan


class FakeRobot:
    def __init__(self):
        self.coords = [200.0, 0.0, 250.0, 10.0, 20.0, 30.0]
        self.angles = [0.0] * 6
        self.ik = [100, -200, 300, -400, 500, -600]
        self.calls = []

    def get_coords(self):
        return list(self.coords)

    def get_angles(self):
        return list(self.angles)

    def solve_inv_kinematics(self, target, current):
        self.calls.append(("ik", list(target), list(current)))
        return list(self.ik)

    @staticmethod
    def _int2angle(value):
        return round(value / 100.0, 3)

    def sync_send_angles(self, target, speed, timeout):
        self.calls.append(("angles", list(target), speed, timeout))
        self.angles = list(target)
        self.coords[:3] = [205.0, 0.0, 250.0]
        return 1

    def sync_send_coords(self, target, speed, mode, timeout):
        self.calls.append(("coords", list(target), speed, mode, timeout))
        self.coords = list(target)
        self.angles = [self._int2angle(value) for value in self.ik]
        return 1

    def stop(self):
        self.calls.append(("stop",))


def args(method="angles", confirmation=""):
    return argparse.Namespace(
        port="mock", baud=1000000, xyz=[205.0, 0.0, 250.0], method=method,
        coord_mode=0, speed=5, timeout=30.0, max_cartesian_step_mm=30.0,
        max_joint_delta_deg=[15.0] * 6, joint_limit_margin_deg=2.0,
        position_tolerance_mm=5.0, orientation_tolerance_deg=5.0,
        joint_tolerance_deg=2.0, confirm_motion=confirmation,
    )


class CartesianMoveTests(unittest.TestCase):
    def test_integer_ik_response_is_decoded_from_centidegrees(self):
        decoded, response_format = decode_ik_response(
            [8241, -1234, 0, 100, -200, 300], FakeRobot()
        )
        self.assertEqual(decoded, [82.41, -12.34, 0.0, 1.0, -2.0, 3.0])
        self.assertEqual(response_format, "centidegree_int")

    def test_float_ik_response_is_not_scaled_again(self):
        decoded, response_format = decode_ik_response(
            [82.0, -12.0, 0.0, 1.0, -2.0, 3.0], FakeRobot()
        )
        self.assertEqual(decoded[0], 82.0)
        self.assertEqual(response_format, "degree_float")

    def test_repeated_8241_response_is_rejected_as_no_solution(self):
        with self.assertRaisesRegex(RuntimeError, "found no solution"):
            decode_ik_response([8241] * 6, FakeRobot())

    def test_plan_preserves_current_orientation(self):
        plan = make_plan(
            [200, 0, 250, 10, 20, 30], [0] * 6, [1, 2, 3, 4, 5, 6],
            [205, 1, 250], max_cartesian_step_mm=30,
            max_joint_delta_deg=[15] * 6, joint_limit_margin_deg=2,
        )
        self.assertEqual(plan["target_coords"], [205.0, 1.0, 250.0, 10.0, 20.0, 30.0])

    def test_excessive_joint_delta_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "J1"):
            make_plan(
                [200, 0, 250, 10, 20, 30], [0] * 6, [16, 0, 0, 0, 0, 0],
                [205, 0, 250], max_cartesian_step_mm=30,
                max_joint_delta_deg=[15] * 6, joint_limit_margin_deg=2,
            )

    def test_dry_run_calls_ik_but_does_not_move(self):
        robot = FakeRobot()
        result = execute(args(), lambda _port, _baud: robot)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual([call[0] for call in robot.calls], ["ik"])

    def test_angles_and_coords_are_separate_execution_modes(self):
        for method in ("angles", "coords"):
            with self.subTest(method=method):
                robot = FakeRobot()
                result = execute(args(method, "I_UNDERSTAND"), lambda _port, _baud: robot)
                self.assertEqual(result["status"], "reached")
                self.assertIn(method, [call[0] for call in robot.calls])


if __name__ == "__main__":
    unittest.main()
