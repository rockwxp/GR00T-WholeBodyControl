import socket
import threading
import unittest

import numpy as np
import zmq

from gear_sonic.utils.teleop.gem_smpl_live import (
    GemPoseSafetyFilter,
    LiveGemSonicPublisher,
    LivePoseBuffer,
    TimedSonicFrame,
    interpolate_frame,
)
from gear_sonic.scripts.run_gem_webcam_teleop import (
    InferenceReadiness,
    PublishControlServer,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE


class _FakePublisher:
    def __init__(self):
        self.publishing_enabled = False
        self.events = []

    def set_publishing_enabled(self, enabled):
        self.publishing_enabled = bool(enabled)
        self.events.append(("publishing", bool(enabled)))

    def set_sonic_streaming_enabled(self, enabled):
        self.events.append(("sonic_streaming", bool(enabled)))


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
    def test_publisher_sends_sonic_mode_commands_on_command_topic(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        publisher = LiveGemSonicPublisher(port=port)
        publisher.set_publishing_enabled(False)
        publisher.start()
        context = zmq.Context()
        subscriber = context.socket(zmq.SUB)
        subscriber.setsockopt(zmq.LINGER, 0)
        subscriber.setsockopt(zmq.RCVTIMEO, 2000)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"command")
        subscriber.connect(f"tcp://127.0.0.1:{port}")
        try:
            # Allow the PUB/SUB subscription to propagate before the first command.
            threading.Event().wait(0.2)
            publisher.set_sonic_streaming_enabled(True)
            messages = [subscriber.recv() for _ in range(3)]
            payload_offset = len(b"command") + HEADER_SIZE
            self.assertEqual(messages[-1][payload_offset + 2], 0)
            self.assertTrue(publisher.sonic_streaming_requested)

            publisher.set_sonic_streaming_enabled(False)
            messages = [subscriber.recv() for _ in range(3)]
            self.assertEqual(messages[-1][payload_offset + 2], 1)
            self.assertFalse(publisher.sonic_streaming_requested)
        finally:
            subscriber.close(linger=0)
            context.term()
            publisher.close()

    def test_inference_readiness_tracks_live_operator_validation(self):
        readiness = InferenceReadiness()
        self.assertFalse(readiness.ready)
        readiness.mark_ready()
        self.assertTrue(readiness.wait_until_ready(0.0))
        readiness.mark_not_ready()
        self.assertFalse(readiness.ready)

    def test_control_server_orders_sonic_mode_before_pose_publication(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        publisher = _FakePublisher()
        readiness = InferenceReadiness()
        server = PublishControlServer(publisher, readiness, port)
        server.start()
        context = zmq.Context()
        client = context.socket(zmq.REQ)
        client.setsockopt(zmq.LINGER, 0)
        client.connect(f"tcp://127.0.0.1:{port}")
        try:
            timer = threading.Timer(0.01, readiness.mark_ready)
            timer.start()
            client.send_json({"command": "prepare", "timeout_seconds": 1.0})
            prepared = client.recv_json()
            timer.join()
            self.assertTrue(prepared["ok"])
            self.assertEqual(prepared["inference"], "ready")
            self.assertFalse(prepared["publishing"])
            self.assertEqual(
                publisher.events[-2:],
                [("publishing", False), ("sonic_streaming", False)],
            )

            client.send_json({"command": "enable"})
            enabled = client.recv_json()
            self.assertTrue(enabled["ok"])
            self.assertTrue(enabled["publishing"])
            self.assertEqual(
                publisher.events[-2:],
                [("sonic_streaming", True), ("publishing", True)],
            )

            client.send_json({"command": "disable"})
            disabled = client.recv_json()
            self.assertTrue(disabled["ok"])
            self.assertEqual(disabled["inference"], "ready")
            self.assertEqual(
                publisher.events[-2:],
                [("publishing", False), ("sonic_streaming", False)],
            )
            self.assertTrue(readiness.ready)
        finally:
            client.close(linger=0)
            context.term()
            server.close()

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

    def test_buffer_clear_removes_pose_from_previous_operator(self):
        buffer = LivePoseBuffer(interpolation_delay=0.05, stale_timeout=0.3)
        buffer.push(_frame(2.0))
        buffer.clear()

        result, age = buffer.sample(2.1)
        self.assertIsNone(result)
        self.assertIsNone(age)

    def test_episode_gate_clears_buffer_when_disabled(self):
        publisher = LiveGemSonicPublisher(window=5, target_fps=50)
        publisher.buffer.push(_frame(2.0))

        publisher.set_publishing_enabled(False)

        self.assertFalse(publisher.publishing_enabled)
        result, age = publisher.buffer.sample(2.1)
        self.assertIsNone(result)
        self.assertIsNone(age)
        publisher.set_publishing_enabled(True)
        self.assertTrue(publisher.publishing_enabled)

    def test_episode_gate_discards_submissions_while_disabled(self):
        publisher = LiveGemSonicPublisher(window=5, target_fps=50)
        publisher.set_publishing_enabled(False)

        submitted = publisher.submit(
            {
                "body_pose": np.zeros((1, 21, 3), dtype=np.float32),
                "global_orient": np.zeros((1, 1, 3), dtype=np.float32),
                "transl": np.zeros((1, 3), dtype=np.float32),
            },
            timestamp=2.0,
        )

        self.assertFalse(submitted)
        result, age = publisher.buffer.sample(2.1)
        self.assertIsNone(result)
        self.assertIsNone(age)

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

        safety.reset()
        self.assertEqual(safety.check(params(joint_angle=1.1)), (True, None))


if __name__ == "__main__":
    unittest.main()
