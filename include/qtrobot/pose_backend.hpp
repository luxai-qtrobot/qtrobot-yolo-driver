#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace qtrobot::yolo {

// Same 17 COCO keypoints, same order, as ultralytics/YOLOs-CPP use internally.
constexpr std::array<const char*, 17> COCO_KEYPOINTS = {
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
};

struct Keypoint {
    float u = 0.0f;
    float v = 0.0f;
    float confidence = 0.0f;
};

struct PersonDetection {
    int trackId = -1;
    float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
    float confidence = 0.0f;
    std::array<Keypoint, 17> keypoints;
};

// Wraps yolos::pose::YOLOPoseDetector (pose estimation) + motcpp::trackers::ByteTrack
// (stable per-person IDs across frames). Kept as our own plain types so the
// rest of the node doesn't depend on either library's internal result types.
class PoseBackend {
public:
    PoseBackend(const std::string& modelPath,
                float confidence,
                int imageSize,
                bool useGPU,
                int numThreads);
    ~PoseBackend();

    PoseBackend(const PoseBackend&) = delete;
    PoseBackend& operator=(const PoseBackend&) = delete;

    std::vector<PersonDetection> detect(const cv::Mat& bgrImage);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    float confidence_;
};

} // namespace qtrobot::yolo
