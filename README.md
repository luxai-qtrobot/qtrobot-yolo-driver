# qtrobot-yolo-driver

A standalone C++ service for QTrobot that detects and tracks persons in real
time using YOLO pose estimation, streaming per-person bounding boxes and
COCO keypoints over ZMQ.

This is a deliberately narrow driver: it exposes **only** raw YOLO pose
detection + tracking, mirroring `qtrobot-realsense-driver`'s shape. It does
**not** do voice activity detection, DOA voice matching, 3D kinematic
projection, or engagement scoring — those stay in Python, either in other
demo projects or in the existing `human-detector` plugin in
`robot-sdk-python`, since kinematics and VAD are already implemented and
working there. This driver is the C++ replacement for the YOLO/tracking
piece of `qtrobot-human-detector` only.

---

## License

**GNU Affero General Public License v3.0 (AGPL-3.0)** — see [LICENSE](LICENSE).

This driver links [YOLOs-CPP](https://github.com/Geekgineer/YOLOs-CPP) and
[motcpp](https://github.com/Geekgineer/motcpp), both AGPL-3.0. If you're
building/distributing this commercially and want to avoid AGPL's source
disclosure obligations, contact Ultralytics (for the underlying YOLO model)
and Geekgineer (both repos' READMEs mention commercial licensing) about
alternative licenses before shipping. This driver is intentionally isolated
as its own process (talks to the rest of QTrobot only over magpie ZMQ, never
linked into another binary) specifically so AGPL stays contained to this one
component.

---

## Ports & Topics

| Purpose | Port | Topic |
|---|---|---|
| RPC | 50770 | — |
| Persons stream | 5781 | `/yolo/persons` |
| Annotated image stream (optional) | 5782 | `/yolo/annotated_image` |

The service advertises itself via Zeroconf under node ID **`qtrobot-yolo-driver`**,
and consumes the color image stream from `qtrobot-realsense-driver` (via
Zeroconf discovery of `camera.node_id`, or a direct `camera.endpoint` override).

---

## Output Schema

Each frame published on `/yolo/persons` is a `DictFrame`:

```python
{
  "persons": {
    "<track_id>": {                      # stable ID string, e.g. "1", "2"
      "bbox":       [x1, y1, x2, y2],     # pixels, image frame
      "confidence": float,                # YOLO detection confidence
      "keypoints": {
        "nose":           {"uv": [u, v], "conf": float},
        "left_eye":       {"uv": [u, v], "conf": float},
        # ... 17 COCO keypoints total (see below)
      }
    }
  }
}
```

### COCO Keypoints

`nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`, `left_shoulder`,
`right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`,
`left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`

No `xyz`/3D projection — that needs head-joint state + camera intrinsics
(`pixel_to_point`), deliberately kept out of this driver. If you need 3D
points, project `uv` yourself the same way `robot-sdk-python`'s
`HeadSolver.pixel_to_base()` does (pure trig, no model — see that file for
the exact math) or call the existing `kinematics` plugin from Python.

---

## Configuration

Config file: `/opt/luxai/qtrobot_yolo_driver/etc/config.yaml` (after install)
or `config/config.yaml` (dev). All parameters can be overridden from the
command line — run with `--help` for details.

Key fields:

| Field | Default | Notes |
|---|---|---|
| `vision.model` | `models/yolo26n-pose.onnx` | YOLO pose model file |
| `vision.framerate` | `5` | Max detection rate (Hz) |
| `vision.confidence` | `0.6` | YOLO detection threshold |
| `vision.stream_annotated_image` | `false` | Enables port+2 |
| `device` | `cpu` | `cpu` or `cuda` — see note below |
| `camera.node_id` | `qtrobot-realsense-driver` | Zeroconf discovery |
| `camera.endpoint` | `""` | Direct ZMQ override |
| `zmq.port` | `50770` | Base RPC port |

### About `device: cuda`

`YOLOPoseDetector` (from `YOLOs-CPP`) checks `Ort::GetAvailableProviders()`
at startup: if `device: cuda` is set and the linked ONNX Runtime build has
`CUDAExecutionProvider`, it uses it; otherwise it logs a warning and falls
back to CPU automatically. **On x86, this "just works"** since Microsoft
publishes a CUDA-EP-enabled `onnxruntime-linux-x64-gpu` release. **On
Jetson/aarch64, it currently won't** — Microsoft's generic aarch64 release is
CPU-only; getting CUDA EP on Jetson needs either NVIDIA's own Jetson-specific
ONNX Runtime build/wheel or building ONNX Runtime from source against
JetPack's CUDA/cuDNN. Given this driver doesn't need TensorRT-class
throughput for human-robot interaction, running CPU-only on Jetson at first
(and revisiting GPU EP only if the framerate isn't enough) is a perfectly
reasonable starting point — same `.onnx` model and code either way, no
recompilation needed to flip `device`.

---

## Installation

### Prerequisites

- `qtrobot-realsense-driver` running and reachable (only the color stream is
  used — no depth dependency)
- A `yolo26n-pose.onnx` model file (or any YOLOv8/v11/v26-pose `.onnx` model)

### Build

```bash
cd qtrobot-yolo-driver
mkdir build && cd build
cmake ..
make -j$(nproc)
```

`CMakeLists.txt` fetches `YOLOs-CPP` and `motcpp` source via `FetchContent`,
and downloads a prebuilt ONNX Runtime release tarball automatically (no
manual download needed) — the right architecture (`x64`/`aarch64`) is
detected from `CMAKE_SYSTEM_PROCESSOR`.

### Install on robot

```bash
sudo dpkg -i qtrobot-yolo-driver_*.deb
```

The service starts automatically after install. To check status:

```bash
systemctl status qtrobot-yolo-driver
journalctl -u qtrobot-yolo-driver -f
```

---

## Development

Run directly without installing:

```bash
cd build
./qtrobot_yolo_driver --config ../config/config.yaml
```

To connect directly to a known camera endpoint instead of relying on
Zeroconf discovery:

```bash
./qtrobot_yolo_driver --config ../config/config.yaml --camera.endpoint tcp://10.231.0.1:50750
```


