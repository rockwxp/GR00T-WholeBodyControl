#!/usr/bin/env python3
"""Run GEM's real-time webcam estimator and stream its SMPL output to SONIC."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from types import ModuleType

from gear_sonic.utils.teleop.gem_smpl_live import (
    GemPoseSafetyFilter,
    LiveGemSonicPublisher,
)


DEFAULT_GENMO_ROOT = Path("/home/xiaopeng/workspace/GENMO")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GEM webcam to SONIC real-time teleoperation bridge."
    )
    parser.add_argument(
        "--genmo-root",
        type=Path,
        default=DEFAULT_GENMO_ROOT,
        help=f"GEM/GENMO repository root (default: {DEFAULT_GENMO_ROOT})",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera-id", type=int, default=0, help="OpenCV camera ID")
    source.add_argument("--video", type=Path, help="Use a video instead of a live camera")
    parser.add_argument(
        "--context-frames",
        type=int,
        default=120,
        help="GEM sliding window length; must match the exported denoiser",
    )
    parser.add_argument("--yolo-period", type=int, default=5)
    parser.add_argument("--vitpose-period", type=int, default=1)
    parser.add_argument(
        "--no-imgfeat",
        action="store_true",
        help="Use GEM's faster no-image-feature denoiser",
    )
    parser.add_argument(
        "--no-async-pipeline",
        action="store_true",
        help="Run GEM synchronously for diagnostics",
    )
    parser.add_argument("--render", action="store_true", help="Enable GEM rendering")
    parser.add_argument(
        "--render-mode", choices=("opencv", "viser"), default="opencv"
    )
    parser.add_argument("--render-port", type=int, default=8012)
    parser.add_argument("--port", type=int, default=5556, help="SONIC ZMQ PUB port")
    parser.add_argument("--topic", default="pose", help="SONIC ZMQ topic")
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument(
        "--interpolation-delay",
        type=float,
        default=0.05,
        help="Seconds of buffering used to interpolate camera-rate results",
    )
    parser.add_argument(
        "--stale-timeout",
        type=float,
        default=0.3,
        help="Stop publishing after this many seconds without a new GEM result",
    )
    parser.add_argument(
        "--reacquire-timeout",
        type=float,
        default=0.75,
        help="Reset pose continuity after this many seconds without an accepted pose",
    )
    parser.add_argument(
        "--acquisition-results",
        type=int,
        default=3,
        help="Require this many consecutive valid GEM results before publishing",
    )
    parser.add_argument(
        "--min-visible-keypoints",
        type=int,
        default=8,
        help="Reject GEM results with fewer visible COCO-17 keypoints",
    )
    parser.add_argument(
        "--min-lower-body-keypoints",
        type=int,
        default=4,
        help="Require this many visible hips/knees/ankles (COCO indices 11-16)",
    )
    parser.add_argument(
        "--max-root-jump",
        type=float,
        default=1.2,
        help="Reject a root rotation jump larger than this many radians",
    )
    parser.add_argument(
        "--max-joint-jump",
        type=float,
        default=1.5,
        help="Reject any body-joint rotation jump larger than this many radians",
    )
    parser.add_argument("--no-wrists", action="store_true")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Start camera capture without waiting for Enter",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    demo_path = args.genmo_root / "scripts" / "demo" / "demo_webcam.py"
    if not demo_path.is_file():
        raise FileNotFoundError(f"GEM webcam demo not found: {demo_path}")
    if args.video is not None and not args.video.is_file():
        raise FileNotFoundError(f"input video not found: {args.video}")
    if args.context_frames < 2:
        raise ValueError("--context-frames must be at least 2")
    if args.yolo_period < 1 or args.vitpose_period < 1:
        raise ValueError("--yolo-period and --vitpose-period must be at least 1")
    if args.target_fps <= 0:
        raise ValueError("--target-fps must be positive")
    if args.window < 1:
        raise ValueError("--window must be at least 1")
    if args.interpolation_delay < 0:
        raise ValueError("--interpolation-delay cannot be negative")
    if args.stale_timeout <= 0:
        raise ValueError("--stale-timeout must be positive")
    if args.reacquire_timeout <= args.stale_timeout:
        raise ValueError("--reacquire-timeout must be greater than --stale-timeout")
    if args.acquisition_results < 1:
        raise ValueError("--acquisition-results must be at least 1")
    if not 0 <= args.min_visible_keypoints <= 17:
        raise ValueError("--min-visible-keypoints must be in [0, 17]")
    if not 0 <= args.min_lower_body_keypoints <= 6:
        raise ValueError("--min-lower-body-keypoints must be in [0, 6]")
    if args.max_root_jump <= 0 or args.max_joint_jump <= 0:
        raise ValueError("--max-root-jump and --max-joint-jump must be positive")


def _load_gem_webcam_module(genmo_root: Path) -> ModuleType:
    demo_dir = genmo_root / "scripts" / "demo"
    for path in (str(genmo_root), str(demo_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)
    # Import by a real, child-importable module name. GEM's OpenCV/Viser
    # renderer uses multiprocessing "spawn", which pickles worker functions
    # by module name and cannot restore functions from a synthetic spec name.
    return importlib.import_module("demo_webcam")


def _gem_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        camera_id=args.camera_id,
        video=str(args.video) if args.video is not None else None,
        context_frames=args.context_frames,
        yolo_period=args.yolo_period,
        vitpose_period=args.vitpose_period,
        no_imgfeat=args.no_imgfeat,
        render=args.render,
        render_mode=args.render_mode,
        render_port=args.render_port,
        save_output=False,
        output_root="outputs/webcam",
        async_pipeline=not args.no_async_pipeline,
        no_async_pipeline=args.no_async_pipeline,
    )


def _complete_global_params(result: dict) -> dict:
    """Merge GEM's global root rollout with its in-camera local body pose."""
    global_params = dict(result["body_params_global"])
    incam_params = result["body_params_incam"]
    for key in ("body_pose", "betas"):
        if key not in global_params and key in incam_params:
            global_params[key] = incam_params[key]
    return global_params


def main() -> int:
    args = _parse_args()
    try:
        _validate_args(args)
        genmo_root = args.genmo_root.resolve()
        # GEM's ONNX runners intentionally resolve model assets such as
        # inputs/onnx relative to the repository root.
        os.chdir(genmo_root)
        gem_webcam = _load_gem_webcam_module(genmo_root)
        demo = gem_webcam.WebcamGEMSMPLDemo(_gem_args(args))
        publisher = LiveGemSonicPublisher(
            port=args.port,
            topic=args.topic,
            target_fps=args.target_fps,
            window=args.window,
            interpolation_delay=args.interpolation_delay,
            stale_timeout=args.stale_timeout,
            include_wrists=not args.no_wrists,
        )
        safety_filter = GemPoseSafetyFilter(
            max_root_jump=args.max_root_jump,
            max_joint_jump=args.max_joint_jump,
        )
        publisher.start()
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    original_process_frame = demo.process_frame
    last_result_id = None
    submitted_results = 0
    rejected_results = 0
    acquisition_results = 0
    active_track_id = None
    last_accepted_time = None

    def process_and_publish(frame_bgr):
        nonlocal last_result_id, submitted_results, rejected_results
        nonlocal acquisition_results, active_track_id, last_accepted_time
        result = original_process_frame(frame_bgr)
        if result is None or not result.get("ready", False):
            return result
        result_id = result.get("_result_id")
        if result_id is not None and result_id == last_result_id:
            return result
        last_result_id = result_id
        now = time.monotonic()
        track_id = getattr(demo, "primary_track_id", None)
        track_changed = (
            active_track_id is not None
            and track_id is not None
            and track_id != active_track_id
        )
        pose_was_stale = (
            last_accepted_time is not None
            and now - last_accepted_time > args.reacquire_timeout
        )
        if track_changed or pose_was_stale:
            reason = "operator track changed" if track_changed else "pose stream was stale"
            safety_filter.reset()
            publisher.reset_source()
            acquisition_results = 0
            print(f"\n[Tracking] Reacquiring operator: {reason}", flush=True)
        if track_id is not None:
            active_track_id = track_id

        keypoints = getattr(demo, "_last_kp2d", None)
        if keypoints is not None:
            visible = keypoints[:, 2] > 0.5
            visible_count = int(visible.sum().item())
            lower_body_count = int(visible[11:17].sum().item())
            if (
                visible_count < args.min_visible_keypoints
                or lower_body_count < args.min_lower_body_keypoints
            ):
                rejected_results += 1
                print(
                    f"\n[Safety] Rejected GEM result {result_id}: visible "
                    f"keypoints={visible_count}/17, lower-body={lower_body_count}/6",
                    flush=True,
                )
                acquisition_results = 0
                return result
        params = _complete_global_params(result)
        accepted, reason = safety_filter.check(params)
        if not accepted:
            rejected_results += 1
            print(f"\n[Safety] Rejected GEM result {result_id}: {reason}", flush=True)
            acquisition_results = 0
            return result
        last_accepted_time = now
        acquisition_results += 1
        if acquisition_results < args.acquisition_results:
            print(
                f"\r[Tracking] Confirming operator "
                f"{acquisition_results}/{args.acquisition_results}",
                end="",
                flush=True,
            )
            return result
        try:
            publisher.submit(
                params,
                timestamp=now,
            )
        except ValueError as exc:
            print(f"\n[Safety] Rejected GEM result {result_id}: {exc}", flush=True)
            rejected_results += 1
            return result
        submitted_results += 1
        return result

    demo.process_frame = process_and_publish
    print(
        f"[SONIC] Publisher ready on tcp://*:{args.port}, topic={args.topic!r}, "
        f"{args.target_fps:g} Hz"
    )
    print(
        f"[Safety] New GEM results older than {args.stale_timeout * 1000:.0f} ms "
        "stop pose publication automatically."
    )
    try:
        if not args.no_wait:
            input(
                "Stand fully visible in a neutral pose. Press Enter to start GEM capture "
                "(Ctrl-C cancels)... "
            )
        demo.run()
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\n[Interrupted]")
        return 130
    finally:
        publisher.close()
        print(
            f"[SONIC] Stopped after {submitted_results} GEM results and "
            f"{publisher.sent_messages} ZMQ messages "
            f"({rejected_results} results rejected by safety gates)."
        )


if __name__ == "__main__":
    raise SystemExit(main())
