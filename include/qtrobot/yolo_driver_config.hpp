#pragma once

#include <cstdint>
#include <string>

namespace qtrobot::yolo {

struct YoloDriverConfig {

    // ---------- Vision ----------
    std::string model       = "models/yolo26n-pose"; // no extension - backend appends .onnx or .trt
    int         framerate   = 10;     // Hz - max detection rate
    float       confidence  = 0.6f;   // YOLO detection threshold
    int         image_size  = 640;    // YOLO input size
    bool        stream_annotated_image = false;
    int         num_threads = 1;      // ONNX Runtime intra/inter-op thread count

    // ---------- Device ----------
    std::string device = "cpu";       // cpu | cuda - passed straight to YOLOPoseDetector's useGPU flag

    // ---------- Emotion recognition (optional) ----------
    bool        emotion_enabled = false;
    std::string emotion_model   = "models/enet_b0_8_best_vgaf.onnx"; // always onnx - EmotiEffLib has no TensorRT backend

    // ---------- Camera (qtrobot-realsense-driver connection) ----------
    std::string camera_node_id = "qtrobot-realsense-driver"; // Zeroconf discovery
    std::string camera_endpoint;                              // direct override, e.g. tcp://10.0.0.5:50655

    // ---------- ZMQ ----------
    std::string node_id  = "qtrobot-yolo-driver"; // used in descriptor + Zeroconf
    uint16_t    zmq_port = 5780;                  // base RPC port; streams use port+1, port+2
};

constexpr const char* paramsYaml = R"(
name: qtrobot-yolo-driver
description: QTrobot YOLO pose detection + tracking driver

parameters:
  - name: log_level
    type: string
    value: INFO
    scope: cli
    description: DEBUG, INFO, WARN

  # Vision (YOLO pose detection)
  - name: vision
    type: group
    description: Visual detection parameters
    parameters:
      - name: model
        type: string
        default: models/yolo26n-pose
        scope: cli
        description: YOLO pose model file, no extension - the backend compiled in (onnx or tensorrt) appends .onnx or .trt
      - name: framerate
        type: int
        default: 10
        scope: cli
        description: Max detection framerate (Hz)
      - name: confidence
        type: double
        default: 0.6
        scope: cli
        description: Minimum YOLO detection confidence
      - name: image_size
        type: int
        default: 640
        scope: cli
        description: YOLO input image size (pixels, smaller = faster e.g. 320, 416, 640)
      - name: stream_annotated_image
        type: bool
        default: false
        scope: cli
        description: Stream camera image with pose skeleton drawn on it (port+2)
      - name: num_threads
        type: int
        default: 1
        scope: cli
        description: ONNX Runtime intra/inter-op thread count. Kept low deliberately - ONNX Runtime's CPU thread pool busy-spins (100% CPU per thread) while idle waiting for work by default, so more threads than this model actually benefits from just burns CPU for no speed gain.

  - name: device
    type: string
    default: cpu
    scope: cli
    description: Inference device - cpu or cuda. Falls back to cpu automatically if CUDA is requested but unavailable in the linked ONNX Runtime build.

  # Emotion recognition (optional - off by default, adds one extra inference
  # call per detected person per frame)
  - name: emotion
    type: group
    description: Per-person emotion recognition parameters
    parameters:
      - name: enabled
        type: bool
        default: false
        scope: cli
        description: Run emotion recognition on each detected person's face crop
      - name: model
        type: string
        default: models/enet_b0_8_best_vgaf.onnx
        scope: cli
        description: EmotiEffLib ONNX model file (always .onnx - no TensorRT backend for this)

  # Camera / RealSense driver connection
  - name: camera
    type: group
    description: RealSense driver connection parameters
    parameters:
      - name: node_id
        type: string
        default: qtrobot-realsense-driver
        description: Zeroconf node_id of the realsense driver service
      - name: endpoint
        type: string
        default: ""
        description: Direct ZMQ endpoint (overrides node_id if set), e.g. tcp://10.0.0.5:50655

  # ZMQ communication (this service's own RPC and stream ports)
  - name: zmq
    type: group
    description: ZMQ communication parameters
    parameters:
      - name: node_id
        type: string
        value: qtrobot-yolo-driver
        description: Node id used for service discovery advertisement
      - name: port
        type: int
        default: 5780
        description: Base ZMQ RPC port. Persons stream is port+1, annotated image stream is port+2

)";

} // namespace qtrobot::yolo
