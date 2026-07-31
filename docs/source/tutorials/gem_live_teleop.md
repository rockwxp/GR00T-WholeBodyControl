# GEM Webcam Teleoperation in MuJoCo

This workflow streams GEM/GENMO's real-time webcam SMPL estimates through
SONIC to a G1 in MuJoCo. It is a simulation-only validation step; do not use
this command against a physical robot.

The bridge directly consumes each new result from GEM's `demo_webcam.py`. It
does not write or poll `smpl_params.pt`.

## 1. Prerequisites

The tested repositories and environment are:

```text
GR00T: /home/xiaopeng/workspace/GR00T-WholeBodyControl
GEM:   /home/xiaopeng/workspace/GENMO
Conda: /home/xiaopeng/miniforge3/envs/genmo
```

The GEM environment needs `pyzmq` in addition to its webcam/ONNX dependencies:

```bash
conda activate genmo
python -m pip install pyzmq
```

The tested machine already has this dependency installed. GEM's webcam ONNX
models must exist under `/home/xiaopeng/workspace/GENMO/inputs/onnx`.

## 2. Three-terminal startup

Follow the order exactly. Terminal 2 and Terminal 3 must use the same ZMQ port
and topic.

### Terminal 1: MuJoCo

```bash
cd /home/xiaopeng/workspace/GR00T-WholeBodyControl
git switch feature/gem-sonic-live-teleop
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

Leave the simulator running.

### Terminal 2: SONIC deployment

```bash
cd /home/xiaopeng/workspace/GR00T-WholeBodyControl/gear_sonic_deploy
bash deploy.sh \
  --input-type zmq \
  --zmq-host localhost \
  --zmq-port 5556 \
  --zmq-topic pose \
  sim
```

Then:

1. Enter `y` if confirmation is requested.
2. Wait for `Init Done`.
3. Press `]` to start the SONIC controller.
4. Click the MuJoCo window and press `9`.
5. Wait until G1 is standing stably.
6. Do not press Enter yet.

Uppercase `O` in this terminal is the emergency stop.

### Terminal 3: GEM webcam bridge

For the camera used by the tested `cam4` recording:

```bash
cd /home/xiaopeng/workspace/GR00T-WholeBodyControl
export PYTHONPATH=/home/xiaopeng/workspace/GR00T-WholeBodyControl

/home/xiaopeng/miniforge3/envs/genmo/bin/python \
  gear_sonic/scripts/run_gem_webcam_teleop.py \
  --genmo-root /home/xiaopeng/workspace/GENMO \
  --camera-id 4 \
  --no-imgfeat \
  --render \
  --render-mode opencv \
  --port 5556 \
  --topic pose
```

If `/dev/video4` is not the intended camera, change `--camera-id 4`. Remove
`--render` to reduce display overhead.

The script first loads and warms up the GEM ONNX models, binds the ZMQ
publisher, and displays:

```text
[SONIC] Publisher ready on tcp://*:5556
Stand fully visible in a neutral pose. Press Enter to start GEM capture
```

Stand approximately 2–4 metres from the camera with the complete body,
including both feet, visible. Press Enter in Terminal 3. GEM then fills its
120-frame sliding window. No pose is sent during this warm-up.

When Terminal 3 changes from:

```text
Warmup .../120
```

to:

```text
Frame ... | FPS ... | transl=(...)
```

press Enter once in Terminal 2 to enable SONIC ZMQ streaming. Start with slow
arm movements and shallow weight shifts before attempting stepping or turning.

To stop:

1. Press uppercase `O` in Terminal 2.
2. Press Ctrl-C in Terminal 3.
3. Stop Terminal 1 last.

## 3. Data path

For each unique GEM `_result_id`, the bridge:

1. merges the global root rollout with the local `body_pose`;
2. checks visible COCO-17 keypoints and lower-body visibility;
3. rejects implausible root/body rotation jumps;
4. converts SMPL Y-up data to SONIC's Z-up convention;
5. buffers results for 50 ms and uses SLERP/linear interpolation;
6. publishes a rolling five-frame SONIC protocol-v3 message at 50 Hz.

Repeated asynchronous GEM results are ignored. If no new accepted pose arrives
for 300 ms, publication stops until tracking recovers.

## 4. Parameters

### GEM input and performance

| Option | Default | Meaning |
|---|---:|---|
| `--genmo-root PATH` | `/home/xiaopeng/workspace/GENMO` | GEM repository containing `demo_webcam.py` and ONNX assets |
| `--camera-id N` | `0` | OpenCV webcam index |
| `--video PATH` | none | Use a recorded video instead of a camera |
| `--context-frames N` | `120` | GEM sliding-window length; must match the exported denoiser |
| `--yolo-period N` | `5` | Run person detection every N captured frames |
| `--vitpose-period N` | `1` | Run 2D pose estimation every N frames |
| `--no-imgfeat` | off | Skip HMR2 image features and use the faster no-image-feature denoiser |
| `--no-async-pipeline` | off | Disable GEM's asynchronous overlap for diagnostics |
| `--render` | off | Show the GEM body estimate |
| `--render-mode opencv` | `opencv` | Use an OpenCV overlay; `viser` is also supported |

The ONNX denoiser is exported for a specific context length. Do not change
`--context-frames` unless a matching model was exported.

### SONIC stream

| Option | Default | Meaning |
|---|---:|---|
| `--port PORT` | `5556` | Publisher port; must match Terminal 2 |
| `--topic TOPIC` | `pose` | ZMQ topic; must match Terminal 2 |
| `--target-fps FPS` | `50` | SONIC publication rate |
| `--window N` | `5` | Rolling protocol-v3 frame count |
| `--interpolation-delay SEC` | `0.05` | Small latency budget used to interpolate camera-rate results |
| `--stale-timeout SEC` | `0.3` | Stop publication after this long without a valid result |
| `--no-wrists` | off | Disable approximate G1 wrist targets |
| `--no-wait` | off | Start camera capture without the neutral-pose Enter gate |

### Safety gates

| Option | Default | Meaning |
|---|---:|---|
| `--min-visible-keypoints N` | `8` | Required visible COCO-17 keypoints |
| `--min-lower-body-keypoints N` | `4` | Required visible hips, knees, and ankles |
| `--max-root-jump RAD` | `1.2` | Reject larger root-rotation changes between accepted GEM results |
| `--max-joint-jump RAD` | `1.5` | Reject larger single-joint rotation changes |

A rejected result is not published. Repeated rejection will trigger the
stale-pose timeout and stop the reference stream. Do not loosen the gates until
the reason for rejection has been observed in MuJoCo.

## 5. Video-only bridge test

This checks GEM inference and SONIC message generation without a camera:

```bash
cd /home/xiaopeng/workspace/GR00T-WholeBodyControl
export PYTHONPATH=/home/xiaopeng/workspace/GR00T-WholeBodyControl

/home/xiaopeng/miniforge3/envs/genmo/bin/python \
  gear_sonic/scripts/run_gem_webcam_teleop.py \
  --video /absolute/path/to/rgb.mp4 \
  --no-imgfeat \
  --no-wait \
  --port 5566
```

Port `5566` keeps this diagnostic publisher separate from the normal SONIC
deployment on `5556`.
