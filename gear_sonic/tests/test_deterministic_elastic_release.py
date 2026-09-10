import unittest
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "gear_sonic" / "data" / "robot_model" / "model_data" / "g1" / "g1_29dof_with_hand.xml"


class DeterministicElasticReleaseTests(unittest.TestCase):
    def test_deferred_toggle_waits_for_simulation_thread(self):
        band = ElasticBand(deferred_release=True)

        band.toggle()

        self.assertTrue(band.enable)
        self.assertTrue(band.consume_release_request())
        self.assertFalse(band.consume_release_request())

    def test_release_sets_exact_root_and_clears_robot_velocity(self):
        env = DefaultEnv.__new__(DefaultEnv)
        env.config = {
            "DETERMINISTIC_ELASTIC_RELEASE": True,
            "ELASTIC_RELEASE_ROOT_POSITION": [0.0, 0.0, 0.793],
            "ELASTIC_RELEASE_ROOT_QUATERNION": [1.0, 0.0, 0.0, 0.0],
            "ELASTIC_RELEASE_BODY_JOINT_POSITIONS": [0.01] * 29,
            "ELASTIC_RELEASE_ZERO_ROBOT_VELOCITY": True,
        }
        env.mj_model = mujoco.MjModel.from_xml_path(str(MODEL))
        env.mj_data = mujoco.MjData(env.mj_model)
        env.mj_data.qpos[:7] = [0.12, -0.08, 0.86, 0.98, 0.0, 0.0, 0.2]
        env.mj_data.qvel[:] = 0.5
        env.mj_data.qacc[:] = 0.25
        env.mj_data.qacc_warmstart[:] = 0.125
        env.qvel_offset = 6
        env.qpos_offset = 7
        env.num_body_dof = 29
        env.num_hand_dof = 7
        env.band_attached_link = env.mj_model.body("torso_link").id
        env.elastic_band = ElasticBand(deferred_release=True)

        env._complete_elastic_band_release()

        np.testing.assert_allclose(env.mj_data.qpos[:3], [0.0, 0.0, 0.793])
        np.testing.assert_allclose(env.mj_data.qpos[3:7], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(env.mj_data.qpos[7:36], 0.01)
        np.testing.assert_allclose(env.mj_data.qvel, 0.0)
        self.assertFalse(env.elastic_band.enable)


if __name__ == "__main__":
    unittest.main()
