"""Thread-safe online GEM SMPL buffering for SONIC ZMQ streaming."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
import zmq
from scipy.spatial.transform import Rotation, Slerp

from gear_sonic.utils.teleop.gem_smpl_replay import (
    SonicSmplMotion,
    convert_to_sonic,
    gem_motion_from_params,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)


@dataclass(frozen=True)
class TimedSonicFrame:
    """One converted SONIC reference frame with a monotonic arrival time."""

    timestamp: float
    smpl_pose: np.ndarray
    smpl_joints: np.ndarray
    body_quat_w: np.ndarray
    joint_pos: np.ndarray


def _single_frame(motion: SonicSmplMotion, timestamp: float) -> TimedSonicFrame:
    if motion.num_frames != 1:
        raise ValueError("online conversion expects exactly one GEM frame")
    return TimedSonicFrame(
        timestamp=float(timestamp),
        smpl_pose=motion.smpl_pose[0].copy(),
        smpl_joints=motion.smpl_joints[0].copy(),
        body_quat_w=motion.body_quat_w[0].copy(),
        joint_pos=motion.joint_pos[0].copy(),
    )


def convert_live_gem_frame(
    params: dict,
    *,
    timestamp: float,
    include_wrists: bool = True,
) -> TimedSonicFrame:
    """Convert a one-frame GEM parameter dictionary to SONIC coordinates."""
    gem_motion = gem_motion_from_params(params)
    if gem_motion.num_frames != 1:
        raise ValueError(
            f"live GEM input must contain exactly one frame, got {gem_motion.num_frames}"
        )
    return _single_frame(
        convert_to_sonic(gem_motion, fps=50.0, include_wrists=include_wrists),
        timestamp,
    )


def _slerp_rotvec(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    shape = first.shape
    # scipy treats the leading dimensions as a rotation stack. Interpolate
    # each body joint independently to preserve the SMPL axis-angle encoding.
    result = np.empty_like(first, dtype=np.float32).reshape(-1, 3)
    first_flat = first.reshape(-1, 3)
    second_flat = second.reshape(-1, 3)
    for index in range(len(result)):
        pair = Rotation.from_rotvec(np.stack((first_flat[index], second_flat[index])))
        result[index] = Slerp([0.0, 1.0], pair)([alpha]).as_rotvec()[0]
    return result.reshape(shape)


def _slerp_wxyz(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    pair_xyzw = np.stack((first[[1, 2, 3, 0]], second[[1, 2, 3, 0]]))
    result_xyzw = Slerp([0.0, 1.0], Rotation.from_quat(pair_xyzw))([alpha]).as_quat()[0]
    return result_xyzw[[3, 0, 1, 2]].astype(np.float32)


def interpolate_frame(
    first: TimedSonicFrame,
    second: TimedSonicFrame,
    target_time: float,
) -> TimedSonicFrame:
    """Interpolate two converted frames at ``target_time``."""
    if second.timestamp <= first.timestamp:
        raise ValueError("frame timestamps must be strictly increasing")
    alpha = float(
        np.clip(
            (target_time - first.timestamp) / (second.timestamp - first.timestamp),
            0.0,
            1.0,
        )
    )
    return TimedSonicFrame(
        timestamp=float(target_time),
        smpl_pose=_slerp_rotvec(first.smpl_pose, second.smpl_pose, alpha),
        smpl_joints=(
            first.smpl_joints + alpha * (second.smpl_joints - first.smpl_joints)
        ).astype(np.float32),
        body_quat_w=_slerp_wxyz(first.body_quat_w, second.body_quat_w, alpha),
        joint_pos=(
            first.joint_pos + alpha * (second.joint_pos - first.joint_pos)
        ).astype(np.float32),
    )


class LivePoseBuffer:
    """Select and interpolate the latest online pose at a small fixed delay."""

    def __init__(self, *, interpolation_delay: float, stale_timeout: float):
        if interpolation_delay < 0:
            raise ValueError("interpolation_delay cannot be negative")
        if stale_timeout <= 0:
            raise ValueError("stale_timeout must be positive")
        self.interpolation_delay = float(interpolation_delay)
        self.stale_timeout = float(stale_timeout)
        self._frames: deque[TimedSonicFrame] = deque(maxlen=8)
        self._lock = threading.Lock()

    def push(self, frame: TimedSonicFrame) -> None:
        with self._lock:
            if self._frames and frame.timestamp <= self._frames[-1].timestamp:
                raise ValueError("live frame timestamps must be strictly increasing")
            self._frames.append(frame)

    def clear(self) -> None:
        """Discard poses from a lost operator before accepting a new one."""

        with self._lock:
            self._frames.clear()

    def sample(self, now: float) -> tuple[TimedSonicFrame | None, float | None]:
        """Return `(frame, source_age)` or `(None, age)` when input is stale."""
        with self._lock:
            frames = tuple(self._frames)
        if not frames:
            return None, None

        age = float(now - frames[-1].timestamp)
        if age > self.stale_timeout:
            return None, age

        target_time = now - self.interpolation_delay
        if target_time <= frames[0].timestamp:
            return frames[0], age
        if target_time >= frames[-1].timestamp:
            return frames[-1], age

        for first, second in zip(frames, frames[1:]):
            if first.timestamp <= target_time <= second.timestamp:
                return interpolate_frame(first, second, target_time), age
        return frames[-1], age


class GemPoseSafetyFilter:
    """Reject implausible one-result rotation jumps before publication."""

    def __init__(
        self,
        *,
        max_root_jump: float = 1.2,
        max_joint_jump: float = 1.5,
    ):
        if max_root_jump <= 0 or max_joint_jump <= 0:
            raise ValueError("pose jump limits must be positive")
        self.max_root_jump = float(max_root_jump)
        self.max_joint_jump = float(max_joint_jump)
        self._last_root: np.ndarray | None = None
        self._last_pose: np.ndarray | None = None

    def reset(self) -> None:
        """Accept the next valid pose as a new continuity baseline."""

        self._last_root = None
        self._last_pose = None

    @staticmethod
    def _rotation_jump(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        first_rotation = Rotation.from_rotvec(first.reshape(-1, 3))
        second_rotation = Rotation.from_rotvec(second.reshape(-1, 3))
        return (first_rotation.inv() * second_rotation).magnitude()

    def check(self, params: dict) -> tuple[bool, str | None]:
        motion = gem_motion_from_params(params)
        if motion.num_frames != 1:
            return False, f"expected one live frame, got {motion.num_frames}"
        root = motion.global_orient[0]
        pose = motion.body_pose[0].reshape(21, 3)
        if self._last_root is not None and self._last_pose is not None:
            root_jump = float(self._rotation_jump(self._last_root, root)[0])
            joint_jump = float(self._rotation_jump(self._last_pose, pose).max())
            if root_jump > self.max_root_jump:
                return (
                    False,
                    f"root rotation jumped {root_jump:.2f} rad "
                    f"(limit {self.max_root_jump:.2f})",
                )
            if joint_jump > self.max_joint_jump:
                return (
                    False,
                    f"body joint jumped {joint_jump:.2f} rad "
                    f"(limit {self.max_joint_jump:.2f})",
                )
        self._last_root = root.copy()
        self._last_pose = pose.copy()
        return True, None


class LiveGemSonicPublisher:
    """Publish online GEM results to SONIC protocol v3 at a fixed rate."""

    def __init__(
        self,
        *,
        port: int = 5556,
        topic: str = "pose",
        target_fps: float = 50.0,
        window: int = 5,
        interpolation_delay: float = 0.05,
        stale_timeout: float = 0.3,
        include_wrists: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not 1 <= port <= 65535:
            raise ValueError("port must be in [1, 65535]")
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if window < 1:
            raise ValueError("window must be at least 1")
        self.port = int(port)
        self.topic = topic
        self.target_fps = float(target_fps)
        self.window = int(window)
        self.include_wrists = include_wrists
        self.clock = clock
        self.buffer = LivePoseBuffer(
            interpolation_delay=interpolation_delay,
            stale_timeout=stale_timeout,
        )
        self._history: deque[tuple[TimedSonicFrame, np.ndarray]] = deque(
            maxlen=self.window
        )
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._stream_index = self.window - 1
        self._last_publish_time: float | None = None
        self._reset_requested = threading.Event()
        self._publishing_enabled = threading.Event()
        self._publishing_enabled.set()
        self._publish_gate_lock = threading.Lock()
        self._command_condition = threading.Condition()
        self._pending_commands: deque[tuple[int, bytes, bool]] = deque()
        self._next_command_id = 1
        self._last_sent_command_id = 0
        self._sonic_streaming_requested = False
        self.sent_messages = 0

    def reset_source(self) -> None:
        """Stop replaying a lost operator and clear interpolation history."""

        self.buffer.clear()
        self._reset_requested.set()

    @property
    def publishing_enabled(self) -> bool:
        """Return whether accepted GEM poses may be sent to SONIC."""

        return self._publishing_enabled.is_set()

    def set_publishing_enabled(self, enabled: bool) -> None:
        """Open or close the output gate without stopping GEM inference.

        Closing the gate also removes the last pose and interpolation history.
        This prevents the publisher thread from repeating a pre-Episode pose and
        makes the next Episode begin only after a fresh GEM result arrives.
        """

        with self._publish_gate_lock:
            if enabled:
                self._publishing_enabled.set()
            else:
                self._publishing_enabled.clear()
                self.reset_source()

    def submit(self, params: dict, *, timestamp: float | None = None) -> bool:
        """Convert one GEM result, returning whether the output gate accepted it."""
        with self._publish_gate_lock:
            if not self._publishing_enabled.is_set():
                return False
            timestamp = self.clock() if timestamp is None else float(timestamp)
            self.buffer.push(
                convert_live_gem_frame(
                    params,
                    timestamp=timestamp,
                    include_wrists=self.include_wrists,
                )
            )
            return True

    def set_sonic_streaming_enabled(
        self,
        enabled: bool,
        *,
        timeout: float = 1.0,
        repeat: int = 3,
    ) -> None:
        """Idempotently request SONIC streamed-motion or planner mode.

        The command uses the same PUB socket as pose data because ZeroMQ sockets
        are thread-affine. Repeating the idempotent command protects the Episode
        boundary from a transient PUB/SUB delivery loss.
        """

        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if repeat < 1:
            raise ValueError("repeat must be at least one")
        message = build_command_message(
            start=False,
            stop=False,
            planner=not enabled,
        )
        with self._command_condition:
            command_id = self._next_command_id
            self._next_command_id += 1
            for index in range(repeat):
                self._pending_commands.append(
                    (command_id, message, index == repeat - 1)
                )
            deadline = self.clock() + timeout
            while self._last_sent_command_id < command_id:
                if self._stop.is_set():
                    raise RuntimeError("publisher stopped before sending SONIC command")
                remaining = deadline - self.clock()
                if remaining <= 0 or not self._command_condition.wait(remaining):
                    raise TimeoutError("timed out sending SONIC streaming command")
            self._sonic_streaming_requested = bool(enabled)

    @property
    def sonic_streaming_requested(self) -> bool:
        """Return the most recent streamed-motion mode requested from SONIC."""

        with self._command_condition:
            return self._sonic_streaming_requested

    def _batch(self, frame: TimedSonicFrame) -> dict[str, np.ndarray]:
        previous = self._history[-1][0] if self._history else None
        joint_vel = np.zeros_like(frame.joint_pos)
        if previous is not None:
            joint_vel = (frame.joint_pos - previous.joint_pos) * self.target_fps
        self._history.append((frame, joint_vel.astype(np.float32)))
        history = list(self._history)
        history = [history[0]] * (self.window - len(history)) + history
        frames = [item[0] for item in history]
        velocities = [item[1] for item in history]
        start_index = self._stream_index - self.window + 1
        batch = {
            "smpl_pose": np.ascontiguousarray(
                np.stack([item.smpl_pose for item in frames]), dtype=np.float32
            ),
            "smpl_joints": np.ascontiguousarray(
                np.stack([item.smpl_joints for item in frames]), dtype=np.float32
            ),
            "body_quat_w": np.ascontiguousarray(
                np.stack([item.body_quat_w for item in frames]), dtype=np.float32
            ),
            "joint_pos": np.ascontiguousarray(
                np.stack([item.joint_pos for item in frames]), dtype=np.float32
            ),
            "joint_vel": np.ascontiguousarray(np.stack(velocities), dtype=np.float32),
            "frame_index": np.arange(
                start_index, self._stream_index + 1, dtype=np.int64
            ),
        }
        self._stream_index += 1
        return batch

    def start(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            raise RuntimeError("publisher is already started")
        self._thread = threading.Thread(
            target=self._run, name="gem-sonic-publisher", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("timed out waiting for the ZMQ publisher to bind")
        if self._error is not None:
            raise RuntimeError("ZMQ publisher failed to start") from self._error

    def _run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.bind(f"tcp://*:{self.port}")
            self._ready.set()
            period = 1.0 / self.target_fps
            deadline = self.clock()
            stale_reported = False
            while not self._stop.is_set():
                pending_command = None
                with self._command_condition:
                    if self._pending_commands:
                        pending_command = self._pending_commands.popleft()
                if pending_command is not None:
                    command_id, message, is_last_repeat = pending_command
                    socket.send(message)
                    if is_last_repeat:
                        with self._command_condition:
                            self._last_sent_command_id = command_id
                            self._command_condition.notify_all()
                if self._reset_requested.is_set():
                    self._history.clear()
                    self._last_publish_time = None
                    self._reset_requested.clear()
                    stale_reported = False
                now = self.clock()
                with self._publish_gate_lock:
                    publishing_enabled = self._publishing_enabled.is_set()
                    frame, age = (
                        self.buffer.sample(now)
                        if publishing_enabled
                        else (None, None)
                    )
                    if frame is not None:
                        if (
                            self._last_publish_time is not None
                            and now - self._last_publish_time > self.buffer.stale_timeout
                        ):
                            # Do not mix a pre-dropout pose into the recovered
                            # rolling window or derive a large synthetic velocity.
                            self._history.clear()
                        socket.send(
                            pack_pose_message(
                                self._batch(frame), topic=self.topic, version=3
                            )
                        )
                        self._last_publish_time = now
                        self.sent_messages += 1
                        stale_reported = False
                if not publishing_enabled:
                    deadline += period
                    remaining = deadline - self.clock()
                    if remaining > 0:
                        self._stop.wait(remaining)
                    else:
                        deadline = self.clock()
                    continue
                if frame is None and age is not None and not stale_reported:
                    print(
                        f"\n[Safety] GEM pose is stale ({age * 1000:.0f} ms); "
                        "stopped publishing until tracking recovers.",
                        flush=True,
                    )
                    stale_reported = True
                deadline += period
                remaining = deadline - self.clock()
                if remaining > 0:
                    self._stop.wait(remaining)
                else:
                    deadline = self.clock()
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            socket.close()
            context.term()

    def close(self) -> None:
        self._stop.set()
        with self._command_condition:
            self._command_condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._error is not None:
            raise RuntimeError("ZMQ publisher stopped after an error") from self._error

    def __enter__(self) -> "LiveGemSonicPublisher":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
