#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "qtrobot/pose_backend.hpp"

namespace qtrobot::yolo {

struct EmotionScore {
    std::string label;
    float score = 0.0f;
};

struct EmotionResult {
    std::string topLabel;
    std::vector<EmotionScore> scores; // all classes the model supports, with confidence
};

// Wraps EmotiEffLib::EmotiEffLibRecognizer (ONNX backend) for per-person
// facial emotion recognition, scoped to a single model
// (enet_b0_8_best_vgaf.onnx, 8 classes, non-MTL).
class EmotionRecognizer {
public:
    explicit EmotionRecognizer(const std::string& modelPath);
    ~EmotionRecognizer();

    EmotionRecognizer(const EmotionRecognizer&) = delete;
    EmotionRecognizer& operator=(const EmotionRecognizer&) = delete;

    EmotionResult predict(const cv::Mat& faceCrop);

    // Derives a face crop from pose keypoints (nose/eyes/ears), padded and
    // clamped to image bounds. Returns an empty Mat if too few of those
    // keypoints were detected with enough confidence to locate a face.
    static cv::Mat cropFace(const cv::Mat& bgrImage, const PersonDetection& person);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace qtrobot::yolo
