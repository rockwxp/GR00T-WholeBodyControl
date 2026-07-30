import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gear_sonic.utils.teleop.gem_smpl_replay import (
    GemMotion,
    convert_to_sonic,
    load_gem_motion,
    resample_gem_motion,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE, pack_pose_message


class GemSmplReplayTest(unittest.TestCase):
    def test_load_resample_convert_and_pack(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_frames = 4
            body_pose = torch.zeros(source_frames, 63)
            body_pose[:, 47] = torch.linspace(0.0, 0.2, source_frames)
            global_orient = torch.zeros(source_frames, 3)
            transl = torch.zeros(source_frames, 3)
            path = Path(temp_dir) / "smpl_params.pt"
            torch.save(
                {
                    "fps": 30.0,
                    "result_ids": torch.tensor([0, 1, 3, 4]),
                    "body_params_global": {
                        "body_pose": body_pose,
                        "global_orient": global_orient,
                        "transl": transl,
                    }
                },
                path,
            )

            loaded = load_gem_motion(path)
            self.assertEqual(loaded.body_pose.shape, (source_frames, 63))
            self.assertEqual(loaded.fps, 30.0)
            self.assertEqual(loaded.frame_ids.tolist(), [0, 1, 3, 4])

            resampled = resample_gem_motion(loaded, source_fps=None, target_fps=50.0)
            self.assertEqual(resampled.num_frames, 8)

            converted = convert_to_sonic(resampled, fps=50.0)
            self.assertEqual(converted.smpl_pose.shape, (8, 21, 3))
            self.assertEqual(converted.smpl_joints.shape, (8, 24, 3))
            self.assertEqual(converted.body_quat_w.shape, (8, 4))
            self.assertEqual(converted.joint_pos.shape, (8, 29))
            self.assertTrue(
                np.allclose(np.linalg.norm(converted.body_quat_w, axis=1), 1.0, atol=1e-5)
            )

            batch = converted.frame_batch(end_index=0, window=5)
            self.assertEqual(batch["frame_index"].tolist(), [0, 0, 0, 0, 0])
            streaming_batch = converted.frame_batch(
                end_index=0, window=5, stream_end_index=9
            )
            self.assertEqual(streaming_batch["frame_index"].tolist(), [5, 6, 7, 8, 9])
            message = pack_pose_message(batch, topic="pose", version=3)
            self.assertTrue(message.startswith(b"pose"))
            self.assertGreater(len(message), len(b"pose") + HEADER_SIZE)

    def test_rejects_wrong_body_pose_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.pt"
            torch.save(
                {
                    "body_params_global": {
                        "body_pose": torch.zeros(2, 69),
                        "global_orient": torch.zeros(2, 3),
                    }
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "body_pose"):
                load_gem_motion(path)

    def test_resample_rejects_invalid_fps(self):
        motion = GemMotion(
            body_pose=np.zeros((2, 63), dtype=np.float32),
            global_orient=np.zeros((2, 3), dtype=np.float32),
            transl=np.zeros((2, 3), dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            resample_gem_motion(motion, source_fps=0.0, target_fps=50.0)


if __name__ == "__main__":
    unittest.main()
