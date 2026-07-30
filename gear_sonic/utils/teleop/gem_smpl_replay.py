"""Convert offline GEM SMPL estimates into SONIC protocol-v3 pose batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa
from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
)


_HUMAN_JOINTS_INFO = (
    Path(__file__).resolve().parents[2] / "data" / "human" / "human_joints_info.pkl"
)


@dataclass(frozen=True)
class GemMotion:
    """GEM SMPL parameters normalized to one time dimension."""

    body_pose: np.ndarray
    global_orient: np.ndarray
    transl: np.ndarray
    fps: float | None = None
    frame_ids: np.ndarray | None = None

    @property
    def num_frames(self) -> int:
        return int(self.body_pose.shape[0])


@dataclass(frozen=True)
class SonicSmplMotion:
    """A complete SONIC protocol-v3 motion sampled at a fixed frame rate."""

    smpl_pose: np.ndarray
    smpl_joints: np.ndarray
    body_quat_w: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    fps: float

    @property
    def num_frames(self) -> int:
        return int(self.smpl_pose.shape[0])

    def frame_batch(
        self, end_index: int, window: int, *, stream_end_index: int | None = None
    ) -> dict[str, np.ndarray]:
        """Return a left-padded rolling window ending at ``end_index``.

        ``stream_end_index`` can provide a session-global monotonically
        increasing index independent of a looping source motion.
        """
        if not 0 <= end_index < self.num_frames:
            raise IndexError(f"end_index {end_index} outside [0, {self.num_frames})")
        if window < 1:
            raise ValueError("window must be at least 1")

        indices = np.arange(end_index - window + 1, end_index + 1)
        indices = np.clip(indices, 0, self.num_frames - 1)
        frame_indices = (
            np.arange(stream_end_index - window + 1, stream_end_index + 1)
            if stream_end_index is not None
            else indices
        )
        if frame_indices[0] < 0:
            raise ValueError("stream_end_index must be at least window - 1")
        return {
            "smpl_pose": np.ascontiguousarray(self.smpl_pose[indices], dtype=np.float32),
            "smpl_joints": np.ascontiguousarray(self.smpl_joints[indices], dtype=np.float32),
            "body_quat_w": np.ascontiguousarray(self.body_quat_w[indices], dtype=np.float32),
            "joint_pos": np.ascontiguousarray(self.joint_pos[indices], dtype=np.float32),
            "joint_vel": np.ascontiguousarray(self.joint_vel[indices], dtype=np.float32),
            "frame_index": frame_indices.astype(np.int64),
        }


def _to_numpy(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a numeric tensor/array") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return result


def _normalize_pose(value: Any) -> np.ndarray:
    pose = _to_numpy(value, "body_pose")
    while pose.ndim > 2 and pose.shape[0] == 1:
        pose = pose[0]
    if pose.ndim == 3 and pose.shape[-2:] == (21, 3):
        pose = pose.reshape(pose.shape[0], 63)
    if pose.ndim != 2 or pose.shape[1] != 63:
        raise ValueError(
            f"body_pose must have shape [T,63] or [T,21,3], got {pose.shape}"
        )
    return np.ascontiguousarray(pose, dtype=np.float32)


def _normalize_vector_sequence(value: Any, name: str, frames: int) -> np.ndarray:
    sequence = _to_numpy(value, name)
    while sequence.ndim > 2 and sequence.shape[0] == 1:
        sequence = sequence[0]
    if sequence.ndim == 1 and sequence.shape[0] == 3 and frames == 1:
        sequence = sequence[None]
    if sequence.shape != (frames, 3):
        raise ValueError(f"{name} must have shape [{frames},3], got {sequence.shape}")
    return np.ascontiguousarray(sequence, dtype=np.float32)


def load_gem_motion(
    path: str | Path,
    *,
    parameter_group: str = "body_params_global",
    allow_unsafe_load: bool = False,
) -> GemMotion:
    """Load and validate a ``smpl_params.pt`` produced by GEM."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"GEM SMPL file not found: {path}")

    try:
        data = torch.load(path, map_location="cpu", weights_only=not allow_unsafe_load)
    except Exception as exc:
        if not allow_unsafe_load:
            raise RuntimeError(
                "Safe torch.load failed. Only for a trusted local GEM output, retry with "
                "--allow-unsafe-load."
            ) from exc
        raise

    if not isinstance(data, dict) or parameter_group not in data:
        groups = sorted(data) if isinstance(data, dict) else []
        raise ValueError(
            f"Missing parameter group {parameter_group!r}; available top-level keys: {groups}"
        )
    params = data[parameter_group]
    if not isinstance(params, dict):
        raise ValueError(f"{parameter_group} must be a dictionary")
    for required in ("body_pose", "global_orient"):
        if required not in params:
            raise ValueError(f"{parameter_group} is missing {required!r}")

    body_pose = _normalize_pose(params["body_pose"])
    global_orient = _normalize_vector_sequence(
        params["global_orient"], "global_orient", body_pose.shape[0]
    )
    transl_value = params.get("transl", np.zeros((body_pose.shape[0], 3), dtype=np.float32))
    transl = _normalize_vector_sequence(transl_value, "transl", body_pose.shape[0])

    fps = data.get("fps")
    if isinstance(fps, torch.Tensor):
        fps = fps.item()
    if fps is not None:
        fps = float(fps)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"fps must be positive and finite, got {fps}")

    frame_ids_value = data.get("result_ids")
    frame_ids = None
    if frame_ids_value is not None:
        if isinstance(frame_ids_value, torch.Tensor):
            frame_ids_value = frame_ids_value.detach().cpu().numpy()
        frame_ids = np.asarray(frame_ids_value, dtype=np.int64).reshape(-1)
        if frame_ids.shape != (body_pose.shape[0],):
            raise ValueError(
                f"result_ids must have shape [{body_pose.shape[0]}], got {frame_ids.shape}"
            )
        if np.any(np.diff(frame_ids) <= 0):
            raise ValueError("result_ids must be strictly increasing")

    return GemMotion(
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        fps=fps,
        frame_ids=frame_ids,
    )


def _resample_rotvec_sequence(
    rotations: np.ndarray, source_times: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    """SLERP a ``[T,J,3]`` axis-angle sequence."""
    if len(source_times) == 1:
        return np.repeat(rotations[:1], len(target_times), axis=0).astype(np.float32)
    result = np.empty((len(target_times), rotations.shape[1], 3), dtype=np.float32)
    for joint_index in range(rotations.shape[1]):
        key_rotations = Rotation.from_rotvec(rotations[:, joint_index])
        result[:, joint_index] = Slerp(source_times, key_rotations)(target_times).as_rotvec()
    return result


def resample_gem_motion(
    motion: GemMotion, source_fps: float | None, target_fps: float
) -> GemMotion:
    """Resample GEM motion with rotation-aware interpolation."""
    source_fps = source_fps if source_fps is not None else motion.fps
    if source_fps is None:
        raise ValueError("source_fps is required when the GEM file has no fps metadata")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if motion.num_frames == 1:
        return motion

    source_frame_ids = (
        motion.frame_ids.astype(np.float64)
        if motion.frame_ids is not None
        else np.arange(motion.num_frames, dtype=np.float64)
    )
    source_times = (source_frame_ids - source_frame_ids[0]) / source_fps
    duration = source_times[-1]
    target_frames = int(round(duration * target_fps)) + 1
    target_times = np.arange(target_frames, dtype=np.float64) / target_fps
    target_times[-1] = min(target_times[-1], source_times[-1])

    body_pose = _resample_rotvec_sequence(
        motion.body_pose.reshape(-1, 21, 3), source_times, target_times
    ).reshape(-1, 63)
    global_orient = _resample_rotvec_sequence(
        motion.global_orient[:, None, :], source_times, target_times
    )[:, 0]
    transl = np.column_stack(
        [
            np.interp(target_times, source_times, motion.transl[:, axis])
            for axis in range(3)
        ]
    ).astype(np.float32)
    return GemMotion(
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        fps=float(target_fps),
        frame_ids=np.arange(target_frames, dtype=np.int64),
    )


def _smpl_wrist_targets(body_pose: np.ndarray) -> np.ndarray:
    """Approximate the six G1 wrist targets used by the PICO SMPL streamer."""
    pose = body_pose.reshape(-1, 21, 3)
    joint_pos = np.zeros((pose.shape[0], 29), dtype=np.float32)

    # The shared decomposition helper divides by the axis-angle magnitude.
    # Give exact identity rotations a numerically harmless epsilon axis.
    left_elbow = pose[:, 17].copy()
    right_elbow = pose[:, 18].copy()
    left_elbow[np.linalg.norm(left_elbow, axis=1) < 1e-8, 0] = 1e-8
    right_elbow[np.linalg.norm(right_elbow, axis=1) < 1e-8, 0] = 1e-8
    twist_axis = np.array([0.0, 1.0, 0.0])
    _, left_swing = decompose_rotation_aa(left_elbow, twist_axis)
    _, right_swing = decompose_rotation_aa(right_elbow, twist_axis)

    left_elbow_euler = Rotation.from_quat(
        left_swing[:, [1, 2, 3, 0]]
    ).as_euler("XYZ")
    right_elbow_euler = Rotation.from_quat(
        right_swing[:, [1, 2, 3, 0]]
    ).as_euler("XYZ")
    left_wrist_euler = Rotation.from_rotvec(pose[:, 19]).as_euler("XYZ")
    right_wrist_euler = Rotation.from_rotvec(pose[:, 20]).as_euler("XYZ")

    joint_pos[:, 23] = left_elbow_euler[:, 0] + left_wrist_euler[:, 0]
    joint_pos[:, 25] = left_wrist_euler[:, 1]
    joint_pos[:, 27] = left_elbow_euler[:, 2] + left_wrist_euler[:, 2]
    joint_pos[:, 24] = -(right_elbow_euler[:, 0] + right_wrist_euler[:, 0])
    joint_pos[:, 26] = -right_wrist_euler[:, 1]
    joint_pos[:, 28] = right_elbow_euler[:, 2] + right_wrist_euler[:, 2]
    return joint_pos


def convert_to_sonic(motion: GemMotion, *, fps: float, include_wrists: bool = True) -> SonicSmplMotion:
    """Apply SONIC's SMPL coordinate/FK conversion to a GEM motion."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    body_pose = torch.from_numpy(motion.body_pose).float()
    global_orient = torch.from_numpy(motion.global_orient).float()

    root_quat = smpl_root_ytoz_up(angle_axis_to_quaternion(global_orient))
    converted_global_orient = quaternion_to_angle_axis(root_quat)
    joints = compute_human_joints(
        body_pose=body_pose,
        global_orient=converted_global_orient,
        human_joints_info_path=str(_HUMAN_JOINTS_INFO),
    )

    root_quat = remove_smpl_base_rot(root_quat, w_last=False)
    inverse_root = quat_inv(root_quat).unsqueeze(1).expand(-1, joints.shape[1], -1)
    local_joints = quat_apply(inverse_root, joints)

    joint_pos = (
        _smpl_wrist_targets(motion.body_pose)
        if include_wrists
        else np.zeros((motion.num_frames, 29), dtype=np.float32)
    )
    if motion.num_frames > 1:
        joint_vel = np.gradient(joint_pos, 1.0 / fps, axis=0).astype(np.float32)
    else:
        joint_vel = np.zeros_like(joint_pos)

    return SonicSmplMotion(
        smpl_pose=motion.body_pose.reshape(-1, 21, 3).astype(np.float32),
        smpl_joints=local_joints.numpy().astype(np.float32),
        body_quat_w=root_quat.numpy().astype(np.float32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        fps=float(fps),
    )
