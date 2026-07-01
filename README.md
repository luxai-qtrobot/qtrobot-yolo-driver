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

This driver links [YOLOs-CPP](https://github.com/Geekgineer/YOLOs-CPP) (or
[YOLOs-CPP-TensorRT](https://github.com/Geekgineer/YOLOs-CPP-TensorRT),
depending on backend — see below) and
[motcpp](https://github.com/Geekgineer/motcpp), all AGPL-3.0. If you're
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
| Persons stream | 50771 | `/persons` |
| Annotated image stream (optional) | 50772 | `/annotated_image` |

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
        "nose": {
          "uv":    [u, v],               # pixel coordinates
          "conf":  float,                # keypoint confidence
          "depth": float,                # metres from camera (EMA-smoothed); -1.0 if no valid reading;
                                         # absent when depth.enabled=false
        },
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
| `vision.model` | `models/yolo26n-pose` | Model path, **no extension** — the backend compiled in appends `.onnx` or `.trt` |
| `vision.framerate` | `5` | Max detection rate (Hz) |
| `vision.confidence` | `0.6` | YOLO detection threshold |
| `vision.stream_annotated_image` | `false` | Enables port+2 |
| `device` | `cpu` | `cpu` or `cuda` — see note below |
| `camera.node_id` | `qtrobot-realsense-driver` | Zeroconf discovery |
| `camera.endpoint` | `""` | Direct ZMQ override |
| `zmq.port` | `50770` | Base RPC port |

### Inference backend: `onnx` vs `tensorrt`

There are two pose-inference backends, selected at **build time** via
`-DYOLO_DRIVER_BACKEND=onnx|tensorrt`, defaulting to `tensorrt` on
aarch64/Jetson and `onnx` on x86:

- **`onnx`** (`YOLOs-CPP`, ONNX Runtime): CPU by default
  (`device: cpu|cuda` config field, checked against
  `Ort::GetAvailableProviders()` at startup — falls back to CPU automatically
  if CUDA EP isn't in the linked ONNX Runtime build, which is the case for
  Microsoft's generic `onnxruntime-linux-aarch64` release used on Jetson).
  Measured ~19ms/frame on an x86 dev machine for `yolo26n-pose`.
- **`tensorrt`** (`YOLOs-CPP-TensorRT`, GPU only): uses the TensorRT + CUDA
  Toolkit that ship as part of JetPack itself — no extra install needed on a
  real Jetson. Needs a `.trt` engine file, not `.onnx` directly (see below).

**Why two backends, and why Jetson defaults to `tensorrt`**: measured
CPU-only inference on a Jetson AGX Orin came out to **~208ms/frame** — about
11x slower than the x86 number above, and already over budget for this
project's own default `vision.framerate: 5` target. Microsoft doesn't
publish a CUDA-EP-enabled ONNX Runtime build for aarch64 (no equivalent of
the `onnxruntime-linux-x64-gpu` release exists for Jetson), and building one
from source is a heavy (1-3+ hour), version-finicky undertaking. TensorRT,
by contrast, is already part of JetPack — `YOLOs-CPP-TensorRT`'s own
`CMakeLists.txt` finds it via the same paths JetPack installs into
(`/usr/lib/aarch64-linux-gnu`), and even auto-detects Jetson to set the right
CUDA architecture (`sm_87` for Orin, `sm_72` for Xavier) — no source build,
no container extraction, no version-matching risk the way ONNX Runtime's
CUDA EP would need.

### The `tensorrt` backend needs a `.trt` engine

`.trt` engines are **not portable** like `.onnx` is — they're tied to the
exact GPU and exact TensorRT version that built them, and TensorRT will
often refuse to load a mismatched one outright rather than degrade
gracefully. That matters if you're building a package meant to run on
varying/unknown hardware — but if you're building for one specific, known
device (e.g. a fixed-hardware robot you control, always the same JetPack
version), it's simplest to just convert the engine **once** and ship it
directly in `models/` alongside the `.onnx` — `install(DIRECTORY models/
...)` already packages whatever's in that directory, `.trt` included, no
extra CMake/postinst wiring needed. The only thing to remember: if this
device's JetPack/TensorRT version ever changes, regenerate the `.trt` and
rebuild the package — same as rebuilding for any other underlying platform
change.

`vision.model` itself has no extension (e.g. `models/yolo26n-pose`) —
whichever backend is compiled in appends its own: `.onnx` for the `onnx`
backend, `.trt` for `tensorrt`. So the same config value works unchanged
for either backend, as long as both `models/yolo26n-pose.onnx` and
`models/yolo26n-pose.trt` exist where it points.

To convert (on the exact target device, or a hardware/JetPack-identical one):

```bash
trtexec --onnx=models/yolo26n-pose.onnx \
        --saveEngine=models/yolo26n-pose.trt \
        --fp16 \
        --shapes=images:1x3x640x640
```

`--shapes` is required because this model's ONNX export has a **dynamic
input shape** — `trtexec` silently defaults unspecified dynamic dims to a
degenerate `1x1` and fails to build (an internal concat-layer assertion)
without it. Since this driver always runs at one fixed size
(`vision.image_size`, default `640`), a fixed-shape engine for that one size
is what we actually want anyway — adjust the `640`s if you change
`vision.image_size`. `images` is the input tensor name ultralytics' own YOLO
ONNX export uses.

---

## Installation

### Prerequisites

- `qtrobot-realsense-driver` running and reachable (only the color stream is
  used — no depth dependency)
- A `yolo26n-pose.onnx` model file (or any YOLOv8/v11/v26-pose `.onnx` model)
- `tensorrt` backend only: TensorRT + CUDA Toolkit (already present via
  JetPack on Jetson; on x86 you'd need to install TensorRT yourself, or just
  use the `onnx` backend instead, which is the x86 default)

### Build

```bash
cd qtrobot-yolo-driver
mkdir build && cd build
cmake ..                                       # backend auto-selected by platform
# cmake .. -DYOLO_DRIVER_BACKEND=onnx          # force ONNX Runtime instead
# cmake .. -DYOLO_DRIVER_BACKEND=tensorrt      # force TensorRT instead
make -j$(nproc)
```

`CMakeLists.txt` fetches `YOLOs-CPP`/`YOLOs-CPP-TensorRT` and `motcpp` source
via `FetchContent` (whichever matches the selected backend). The `onnx`
backend also downloads a prebuilt ONNX Runtime release tarball automatically
(no manual download needed) — the right architecture (`x64`/`aarch64`) is
detected from `CMAKE_SYSTEM_PROCESSOR`. The `tensorrt` backend instead finds
TensorRT/CUDA as ordinary system libraries (JetPack-provided on Jetson).

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


