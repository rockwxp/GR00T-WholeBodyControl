import unittest

import numpy as np

from gear_sonic.utils.teleop.gem_smpl_live import (
    GemPoseSafetyFilter,
    LiveGemSonicPublisher,
    LivePoseBuffer,
    TimedSonicFrame,
    interpolate_frame,
)


def _frame(timestamp: float, value: float = 0.0) -> TimedSonicFrame:
    pose = np.zeros((21, 3), dtype=np.float32)
    pose[:, 2] = value
    joints = np.full((24, 3), value, dtype=np.float32)
    half_angle = value / 2.0
    root = np.array(
        [np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)], dtype=np.float32
    )
    joint_pos = np.full(29, value, dtype=np.float32)
    return TimedSonicFrame(timestamp, pose, joints, root, joint_pos)


class LiveGemSmplTests(unittest.TestCase):
    def test_interpolation_uses_rotation_aware_midpoint(self):
        result = interpolate_frame(_frame(1.0, 0.0), _frame(2.0, 1.0), 1.5)
        np.testing.assert_allclose(result.smpl_pose[:, 2], 0.5, atol=1e-6)
        np.testing.assert_allclose(result.smpl_joints, 0.5, atol=1e-6)
        np.testing.assert_allclose(result.joint_pos, 0.5, atol=1e-6)
        expected_root = np.array(
            [np.cos(0.25), 0.0, 0.0, np.sin(0.25)], dtype=np.float32
        )
        np.testing.assert_allclose(result.body_quat_w, expected_root, atol=1e-6)

    def test_buffer_interpolates_then_rejects_stale_input(self):
        buffer = LivePoseBuffer(interpolation_delay=0.1, stale_timeout=0.3)
        buffer.push(_frame(1.0, 0.0))
        buffer.push(_frame(1.2, 1.0))

        result, age = buffer.sample(1.25)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(age, 0.05)
        np.testing.assert_allclose(result.joint_pos, 0.75, atol=1e-6)

        result, age = buffer.sample(1.51)
        self.assertIsNone(result)
        self.assertAlmostEqual(age, 0.31)

    def test_protocol_batch_is_left_padded_and_monotonic(self):
        publisher = LiveGemSonicPublisher(window=5, target_fps=50)
        first = publisher._batch(_frame(1.0, 0.0))
        self.assertEqual(first["smpl_pose"].shape, (5, 21, 3))
        self.assertEqual(first["smpl_joints"].shape, (5, 24, 3))
        self.assertEqual(first["body_quat_w"].shape, (5, 4))
        self.assertEqual(first["joint_pos"].shape, (5, 29))
        self.assertEqual(first["joint_vel"].shape, (5, 29))
        self.assertEqual(first["frame_index"].tolist(), [0, 1, 2, 3, 4])

        second = publisher._batch(_frame(1.02, 0.1))
        self.assertEqual(second["frame_index"].tolist(), [1, 2, 3, 4, 5])
        np.testing.assert_allclose(second["joint_vel"][-1], 5.0, atol=1e-6)

    def test_buffer_rejects_out_of_order_timestamps(self):
        buffer = LivePoseBuffer(interpolation_delay=0.05, stale_timeout=0.3)
        buffer.push(_frame(2.0))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            buffer.push(_frame(2.0))

    def test_pose_safety_filter_rejects_large_rotation_jump(self):
        safety = GemPoseSafetyFilter(max_root_jump=0.5, max_joint_jump=1.0)

        def params(root_angle=0.0, joint_angle=0.0):
            pose = np.zeros((1, 21, 3), dtype=np.float32)
            pose[0, 3, 2] = joint_angle
            return {
                "body_pose": pose,
                "global_orient": np.array(
                    [[[0.0, 0.0, root_angle]]], dtype=np.float32
                ),
                "transl": np.zeros((1, 3), dtype=np.float32),
            }

        self.assertEqual(safety.check(params()), (True, None))
        accepted, reason = safety.check(params(root_angle=0.6))
        self.assertFalse(accepted)
        self.assertIn("root rotation jumped", reason)

        accepted, reason = safety.check(params(joint_angle=1.1))
        self.assertFalse(accepted)
        self.assertIn("body joint jumped", reason)


if __name__ == "__main__":
    unittest.main()
