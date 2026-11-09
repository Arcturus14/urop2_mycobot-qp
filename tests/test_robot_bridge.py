import math
import unittest
import argparse
import io
from itertools import chain, count, repeat
from unittest.mock import Mock, patch

from scripts import robot_bridge as bridge
from scripts.robot_bridge import valid_angles, validate_grip


class RobotBridgeTests(unittest.TestCase):
    def test_accepts_six_finite_angles(self):
        self.assertTrue(valid_angles([0, 1, 2, 3, 4, 5]))

    def test_rejects_bad_angle_samples(self):
        self.assertFalse(valid_angles([0] * 5))
        self.assertFalse(valid_angles([0, 0, 0, 0, 0, math.nan]))

    def test_invalid_gripper_values_are_rejected(self):
        for value, speed in [(-1, 20), (101, 20), (20, 0), (20, 101), (1.5, 20)]:
            with self.assertRaises(ValueError):
                validate_grip(value, speed)

    def run_transaction(self, final, record_motion=False, drift=None, final_only=False):
        robot = Mock()
        steady = final if drift is None else drift
        motion_samples = [] if final_only else [[0.5] * 6]
        robot.get_angles.side_effect = chain([[0] * 6, [0] * 6, *motion_samples, final], repeat(steady))
        robot.get_coords.side_effect = [[10, 20, 30, 40, 50, 60], [11, 21, 31, 41, 51, 61]]
        robot.is_moving.return_value = 0
        robot.is_moving.side_effect = chain([1], repeat(0))
        robot.is_gripper_moving.return_value = 0
        robot.get_gripper_value.return_value = 0
        args = argparse.Namespace(target=[1] * 6, speed=20, port="mock", baud=1000000, timeout=60)
        record = "true" if record_motion else "false"
        final_mode = "true" if final_only else "false"
        decision = io.StringIO('{"action":"PROCEED","grip":{"value":0,"speed":20},'
                               '"record_motion":' + record + ',"final_only":' + final_mode + '}\n')
        events = []
        traces = []
        with patch.object(bridge, "connect", return_value=robot), \
                patch.object(bridge.sys, "stdin", decision), \
                patch.object(bridge.select, "select", return_value=([], [], [])), \
                patch.object(bridge.time, "sleep"), \
                patch.object(bridge.time, "monotonic", side_effect=count(step=0.1)), \
                patch.object(bridge, "emit_trace", side_effect=lambda samples, target: traces.extend(samples)), \
                patch.object(bridge, "emit", side_effect=lambda event, **values: events.append(event)):
            try:
                bridge.transaction(args)
            except RuntimeError:
                return robot, events, False, traces
        return robot, events, True, traces

    def test_grasp_runs_only_after_verified_final_pose(self):
        robot, events, ok, traces = self.run_transaction([1] * 6)
        self.assertTrue(ok)
        self.assertEqual(events, ["current", "final", "gripper_sent", "gripper_final", "done"])
        robot.set_gripper_value.assert_called_once_with(0, 20, 1)
        self.assertGreater(robot.get_angles.call_count, 4)
        robot.send_angles.assert_called_once()
        self.assertEqual(robot.get_coords.call_count, 2)
        self.assertEqual(traces, [])

    def test_final_only_reads_angles_at_start_and_end_and_skips_grasp_on_error(self):
        robot, events, ok, traces = self.run_transaction([4] * 6, final_only=True)
        self.assertFalse(ok)
        self.assertEqual(events, ["current", "final"])
        self.assertEqual(robot.get_angles.call_count, 3)
        self.assertEqual(robot.is_moving.call_count, 2)
        robot.send_angles.assert_called_once()
        robot.set_gripper_value.assert_not_called()
        self.assertEqual(traces, [])

    def test_bad_coordinate_read_is_explicitly_missing(self):
        robot = Mock()
        robot.get_coords.return_value = None
        coords, error = bridge.read_coords(robot)
        self.assertIsNone(coords)
        self.assertIn("invalid", error)

    def test_joint_limits_observer_allows_off_path_pose_but_rejects_joint_limit(self):
        robot = Mock()
        samples = []
        for intermediate in ([0, 20, 0, 0, 0, 0], [0, 141, 0, 0, 0, 0]):
            robot.is_moving.side_effect = [1, 0]
            robot.get_angles.side_effect = [intermediate, [0, 20, 0, 0, 0, 0]]
            with patch.object(bridge, "stop_if_requested"), patch.object(bridge.time, "sleep"):
                if intermediate[1] == 141:
                    with self.assertRaisesRegex(RuntimeError, "joint limit exceeded \\(J2"):
                        bridge.observe_joint_limits(robot, 1, samples, 0)
                else:
                    self.assertEqual(bridge.observe_joint_limits(robot, 1, None, 0)[1], 20)

    def test_target_not_reached_skips_grasp(self):
        robot, events, ok, traces = self.run_transaction([10] * 6, record_motion=True)
        self.assertFalse(ok)
        robot.set_gripper_value.assert_not_called()
        self.assertNotIn("done", events)
        self.assertEqual([sample["phase"] for sample in traces[:3]], ["before", "moving", "holding"])
        self.assertEqual(traces[-1]["angles_deg"], [10] * 6)
        robot.stop.assert_called_once()

    def test_joint_drift_skips_grasp_even_without_recording(self):
        robot, events, ok, traces = self.run_transaction([1] * 6, drift=[1, 1, 1, -0.5, 1, 1])
        self.assertFalse(ok)
        robot.stop.assert_called_once()
        robot.set_gripper_value.assert_not_called()
        self.assertNotIn("done", events)
        self.assertEqual(traces, [])

    def test_j4_correction_waits_for_path_approval(self):
        start = [0, 0, 0, -3.42, 0, 0]
        target = [0, 0, 0, -2.31, 0, 0]
        args = argparse.Namespace(speed=10, timeout=60)
        for approval in ("ABORT", "CORRECT"):
            robot = Mock()
            robot.is_moving.return_value = 0
            corrected = [0, 0, 0, -3.07, 0, 0]
            with patch.object(bridge, "emit"), patch.object(bridge.time, "sleep"), \
                    patch.object(bridge, "sample_angles", return_value=start), \
                    patch.object(bridge, "supervise_move", return_value=(corrected, 1.87)), \
                    patch.object(bridge.sys, "stdin", io.StringIO(approval + "\n")), \
                    patch.object(bridge.select, "select", side_effect=[([True], [], []), ([], [], [])]):
                if approval == "ABORT":
                    with self.assertRaisesRegex(RuntimeError, "not approved"):
                        bridge.correct_j4_once(robot, start, target, args, [], 0)
                    robot.send_angle.assert_not_called()
                else:
                    self.assertEqual(bridge.correct_j4_once(robot, start, target, args, [], 0), (corrected, True))
                    robot.send_angle.assert_called_once()
                    joint, angle, speed = robot.send_angle.call_args.args
                    self.assertEqual((joint, speed), (4, 10))
                    self.assertAlmostEqual(angle, -1.20)

    def test_j4_correction_rejects_measured_j6_drift(self):
        target = [-0.96, -0.17, -0.7, -2.23, -0.79, -43.94]
        first = [-0.96, -0.17, -0.7, -3.33, -0.79, -43.06]
        corrected = [-0.96, -0.17, -0.7, -2.90, -0.79, -42.18]
        robot = Mock()
        robot.is_moving.return_value = 0
        args = argparse.Namespace(speed=10, timeout=60)
        with patch.object(bridge, "emit") as emit, patch.object(bridge.time, "sleep"), \
                patch.object(bridge, "sample_angles", return_value=first), \
                patch.object(bridge, "supervise_move", return_value=(corrected, 1.77)), \
                patch.object(bridge.sys, "stdin", io.StringIO("CORRECT\n")), \
                patch.object(bridge.select, "select", side_effect=[([True], [], []), ([], [], [])]):
            self.assertEqual(bridge.correct_j4_once(robot, first, target, args, [], 0), (corrected, False))
            self.assertAlmostEqual(emit.call_args.kwargs["other_joint_drift_deg"], 0.88)
            robot.send_angle.assert_called_once()
            robot.send_angles.assert_not_called()


if __name__ == "__main__":
    unittest.main()
