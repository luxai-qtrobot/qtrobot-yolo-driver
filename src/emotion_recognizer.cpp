#include "qtrobot/emotion_recognizer.hpp"

#include <algorithm>
#include <array>
#include <limits>

#include <emotiefflib/facial_analysis.h>

namespace qtrobot::yolo {

struct EmotionRecognizer::Impl {
    std::unique_ptr<EmotiEffLib::EmotiEffLibRecognizer> recognizer;
};

EmotionRecognizer::EmotionRecognizer(const std::string& modelPath)
    : impl_(std::make_unique<Impl>()) {
    impl_->recognizer = EmotiEffLib::EmotiEffLibRecognizer::createInstance("onnx", modelPath);
}

EmotionRecognizer::~EmotionRecognizer() = default;

EmotionResult EmotionRecognizer::predict(const cv::Mat& faceCrop) {
    // logits=false -> softmax applied, so scores are real confidence levels.
    auto res = impl_->recognizer->predictEmotions(faceCrop, /*logits=*/false);

    EmotionResult out;
    out.topLabel = res.labels.front();
    const int numClasses = static_cast<int>(res.scores.shape(1));
    out.scores.reserve(numClasses);
    for (int i = 0; i < numClasses; ++i) {
        out.scores.push_back({impl_->recognizer->getEmotionClassById(i), res.scores(0, i)});
    }
    return out;
}

cv::Mat EmotionRecognizer::cropFace(const cv::Mat& bgrImage, const PersonDetection& person) {
    constexpr std::array<int, 5> kFaceKeypoints = {0, 1, 2, 3, 4}; // nose, eyes, ears
    constexpr float kMinConfidence = 0.3f;

    float minU = std::numeric_limits<float>::max();
    float minV = std::numeric_limits<float>::max();
    float maxU = std::numeric_limits<float>::lowest();
    float maxV = std::numeric_limits<float>::lowest();
    int validCount = 0;

    for (int idx : kFaceKeypoints) {
        const Keypoint& kp = person.keypoints[idx];
        if (kp.confidence < kMinConfidence) continue;
        minU = std::min(minU, kp.u);
        minV = std::min(minV, kp.v);
        maxU = std::max(maxU, kp.u);
        maxV = std::max(maxV, kp.v);
        ++validCount;
    }
    if (validCount < 2) {
        return cv::Mat();
    }

    const float centerU = 0.5f * (minU + maxU);
    const float centerV = 0.5f * (minV + maxV);

    // Nose/eyes/ears span only the middle of the face — pad generously to
    // also cover forehead/chin/cheeks. Floor the span at a fraction of the
    // person's own bbox height so a near-frontal face (eyes/ears nearly
    // overlapping) still yields a reasonably sized crop, not a sliver.
    const float personHeight = person.y2 - person.y1;
    float side = std::max({maxU - minU, maxV - minV, 0.25f * personHeight});
    side *= 2.0f;

    int left   = static_cast<int>(std::round(centerU - 0.5f * side));
    int top    = static_cast<int>(std::round(centerV - 0.5f * side));
    int right  = static_cast<int>(std::round(centerU + 0.5f * side));
    int bottom = static_cast<int>(std::round(centerV + 0.5f * side));

    left   = std::clamp(left, 0, bgrImage.cols - 1);
    top    = std::clamp(top, 0, bgrImage.rows - 1);
    right  = std::clamp(right, left + 1, bgrImage.cols);
    bottom = std::clamp(bottom, top + 1, bgrImage.rows);

    return bgrImage(cv::Rect(left, top, right - left, bottom - top)).clone();
}

} // namespace qtrobot::yolo
