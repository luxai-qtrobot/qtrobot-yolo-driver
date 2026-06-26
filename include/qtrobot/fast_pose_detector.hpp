#pragma once

#include <string>
#include <vector>

#include <yolos/core/session_base.hpp>
#include <yolos/core/preprocessing.hpp>
#include <yolos/tasks/pose.hpp> // for yolos::pose::PoseResult (type only, not its detector)

namespace qtrobot::yolo {

// Replacement for yolos::pose::YOLOPoseDetector that gives us control over
// ONNX Runtime's thread pool spin-wait behavior, which YOLOPoseDetector's
// constructor never exposes (it only forwards modelPath/useGPU to
// OrtSessionBase, never numThreads, and never touches SessionOptions'
// allow_spinning config — both of which must be set before the Session is
// constructed, which happens inside OrtSessionBase's own constructor).
//
// We inherit OrtSessionBase (not YOLOPoseDetector) so we still get its node
// name / input shape discovery for free, then rebuild session_ ourselves
// right after with the SessionOptions we actually want. The model gets
// parsed twice at startup (once by the inherited constructor, once by us) —
// a one-time cost, not a per-frame one.
//
// Pre/postprocessing below is adapted from yolos::pose::YOLOPoseDetector
// (github.com/Geekgineer/YOLOs-CPP, AGPL-3.0) since that logic is tightly
// coupled to the class itself (protected members, not free functions) and
// couldn't be reused by composition.
class FastPoseDetector : public yolos::OrtSessionBase {
public:
    FastPoseDetector(const std::string& modelPath, bool useGPU, int numThreads);

    std::vector<yolos::pose::PoseResult> detect(const cv::Mat& image,
                                                 float confThreshold,
                                                 float iouThreshold);

private:
    static constexpr int kNumKeypoints = 17;
    static constexpr int kFeaturesPerKeypoint = 3;

    std::vector<yolos::pose::PoseResult> postprocessV8(const cv::Size& originalSize,
                                                         const cv::Size& resizedShape,
                                                         const float* rawOutput,
                                                         const std::vector<int64_t>& outputShape,
                                                         float confThreshold,
                                                         float iouThreshold);
    std::vector<yolos::pose::PoseResult> postprocessV26(const cv::Size& originalSize,
                                                          const cv::Size& resizedShape,
                                                          const float* rawOutput,
                                                          const std::vector<int64_t>& outputShape,
                                                          float confThreshold);

    yolos::preprocessing::InferenceBuffer buffer_;
};

} // namespace qtrobot::yolo
