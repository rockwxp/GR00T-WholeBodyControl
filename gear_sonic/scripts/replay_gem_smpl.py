#!/usr/bin/env python3
"""Replay an offline GEM ``smpl_params.pt`` through SONIC's ZMQ v3 input."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import zmq

from gear_sonic.utils.teleop.gem_smpl_replay import (
    convert_to_sonic,
    load_gem_motion,
    resample_gem_motion,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay GEM SMPL estimates to a SONIC MuJoCo deployment over ZMQ."
    )
    parser.add_argument("smpl_params", type=Path, help="GEM output smpl_params.pt")
    parser.add_argument(
        "--source-fps",
        type=float,
        help="Override GEM video FPS (default: read fps from smpl_params.pt)",
    )
    parser.add_argument("--target-fps", type=float, default=50.0, help="SONIC stream FPS")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ PUB port")
    parser.add_argument("--topic", default="pose", help="ZMQ pose topic")
    parser.add_argument("--window", type=int, default=5, help="Frames per rolling v3 message")
    parser.add_argument(
        "--parameter-group",
        choices=("body_params_global", "body_params_incam"),
        default="body_params_global",
        help="GEM parameter group to replay",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="First source frame")
    parser.add_argument("--end-frame", type=int, help="Exclusive final source frame")
    parser.add_argument("--lead-in", type=float, default=1.0, help="Seconds to hold first pose")
    parser.add_argument("--hold-last", type=float, default=1.0, help="Seconds to hold final pose")
    parser.add_argument("--loop", action="store_true", help="Replay continuously")
    parser.add_argument("--no-wrists", action="store_true", help="Leave all G1 joint fields zero")
    parser.add_argument("--no-wait", action="store_true", help="Start without waiting for Enter")
    parser.add_argument("--dry-run", action="store_true", help="Validate and convert without ZMQ")
    parser.add_argument(
        "--allow-unsafe-load",
        action="store_true",
        help="Allow legacy pickle loading; only use for a trusted local GEM output",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")
    if args.window < 1:
        raise ValueError("--window must be at least 1")
    if args.start_frame < 0:
        raise ValueError("--start-frame cannot be negative")
    if args.end_frame is not None and args.end_frame <= args.start_frame:
        raise ValueError("--end-frame must be greater than --start-frame")
    if args.lead_in < 0 or args.hold_last < 0:
        raise ValueError("--lead-in and --hold-last cannot be negative")


def _sleep_until(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
        return deadline
    return time.monotonic()


def main() -> int:
    args = _parse_args()
    try:
        _validate_args(args)
        gem_motion = load_gem_motion(
            args.smpl_params,
            parameter_group=args.parameter_group,
            allow_unsafe_load=args.allow_unsafe_load,
        )
        end_frame = args.end_frame if args.end_frame is not None else gem_motion.num_frames
        if args.start_frame >= gem_motion.num_frames or end_frame > gem_motion.num_frames:
            raise ValueError(
                f"Requested source range [{args.start_frame}, {end_frame}) outside "
                f"{gem_motion.num_frames} frames"
            )
        gem_motion = type(gem_motion)(
            body_pose=gem_motion.body_pose[args.start_frame:end_frame],
            global_orient=gem_motion.global_orient[args.start_frame:end_frame],
            transl=gem_motion.transl[args.start_frame:end_frame],
            fps=gem_motion.fps,
            frame_ids=(
                gem_motion.frame_ids[args.start_frame:end_frame]
                if gem_motion.frame_ids is not None
                else None
            ),
        )
        gem_motion = resample_gem_motion(gem_motion, args.source_fps, args.target_fps)
        sonic_motion = convert_to_sonic(
            gem_motion, fps=args.target_fps, include_wrists=not args.no_wrists
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    duration = (sonic_motion.num_frames - 1) / sonic_motion.fps
    print(
        f"Loaded {args.smpl_params}: {sonic_motion.num_frames} frames at "
        f"{sonic_motion.fps:g} Hz ({duration:.2f} s)"
    )
    print(
        "Converted fields: smpl_pose {}, smpl_joints {}, body_quat_w {}, joint_pos {}".format(
            sonic_motion.smpl_pose.shape,
            sonic_motion.smpl_joints.shape,
            sonic_motion.body_quat_w.shape,
            sonic_motion.joint_pos.shape,
        )
    )
    if args.dry_run:
        batch = sonic_motion.frame_batch(sonic_motion.num_frames - 1, args.window)
        packed = pack_pose_message(batch, topic=args.topic, version=3)
        print(f"Dry run OK: final v3 message is {len(packed)} bytes")
        return 0

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.bind(f"tcp://*:{args.port}")
        print(f"Publishing SONIC protocol v3 on tcp://*:{args.port}, topic={args.topic!r}")
        print("Start MuJoCo and the C++ deployment, then enable ZMQ streaming with ENTER.")
        if not args.no_wait:
            input("Press Enter here to begin offline playback (Ctrl-C cancels)... ")
        else:
            time.sleep(0.2)

        lead_frames = int(round(args.lead_in * sonic_motion.fps))
        tail_frames = int(round(args.hold_last * sonic_motion.fps))
        period = 1.0 / sonic_motion.fps

        stream_index = args.window - 1
        while True:
            playback_indices = (
                [0] * lead_frames
                + list(range(sonic_motion.num_frames))
                + [sonic_motion.num_frames - 1] * tail_frames
            )
            deadline = time.monotonic()
            for frame_index in playback_indices:
                batch = sonic_motion.frame_batch(
                    frame_index, args.window, stream_end_index=stream_index
                )
                socket.send(pack_pose_message(batch, topic=args.topic, version=3))
                stream_index += 1
                deadline = _sleep_until(deadline + period)
            if not args.loop:
                break
        print("Playback complete; the final pose was held before the publisher stopped.")
        return 0
    except KeyboardInterrupt:
        print("\nPlayback interrupted.")
        return 130
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
