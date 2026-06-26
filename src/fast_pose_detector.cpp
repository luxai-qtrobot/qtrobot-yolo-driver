#include <qtrobot/fast_pose_detector.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

#include <yolos/core/nms.hpp>
#include <yolos/core/utils.hpp>

namespace qtrobot::yolo {

FastPoseDetector::FastPoseDetector(const std::string& modelPath, bool useGPU, int numThreads)
    // Runs OrtSessionBase's normal (default-threaded, spinning) init first —
    // wasted work, but it's what gives us env_/inputShape_/input-output node
    // names for free. We immediately replace session_ below with one built
    // from our own SessionOptions before any frame is ever processed.
    : OrtSessionBase(modelPath, useGPU, numThreads)
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(numThreads > 0 ? numThreads : 1);
    opts.SetInterOpNumThreads(numThreads > 0 ? numThreads : 1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // The actual fix: tell ORT's thread pools to block (OS-level wait) when
    // idle instead of busy-spinning. This can only be set on SessionOptions
    // before the Session is constructed — there is no way to change it on an
    // already-built Session, which is exactly why OrtSessionBase's own
    // initSession() (called above) can't be patched into doing this from
    // outside; it has to happen here, on a session we build ourselves.
    opts.AddConfigEntry("session.intra_op.allow_spinning", "0");
    opts.AddConfigEntry("session.inter_op.allow_spinning", "0");

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    bool hasCuda = std::find(availableProviders.begin(), availableProviders.end(),
                              "CUDAExecutionProvider") != availableProviders.end();
    if (useGPU && hasCuda) {
        OrtCUDAProviderOptions cudaOptions{};
        opts.AppendExecutionProvider_CUDA(cudaOptions);
    }

#ifdef _WIN32
    std::wstring wModelPath(modelPath.begin(), modelPath.end());
    session_ = Ort::Session(env_, wModelPath.c_str(), opts);
#else
    session_ = Ort::Session(env_, modelPath.c_str(), opts);
#endif
    sessionOptions_ = std::move(opts);

    buffer_.ensureCapacity(inputShape_.height, inputShape_.width, 3);
}

std::vector<yolos::pose::PoseResult> FastPoseDetector::detect(const cv::Mat& image,
                                                                float confThreshold,
                                                                float iouThreshold) {
    cv::Size actualSize;
    yolos::preprocessing::letterBoxToBlob(image, buffer_, inputShape_, actualSize, isDynamicInputShape_);

    std::vector<int64_t> inputTensorShape = {1, 3, actualSize.height, actualSize.width};
    Ort::Value inputTensor = createInputTensor(buffer_.blob.data(), inputTensorShape);

    std::vector<Ort::Value> outputTensors = runInference(inputTensor);

    const float* rawOutput = outputTensors[0].GetTensorData<float>();
    const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

    const int expectedFeaturesV8 = 4 + 1 + kNumKeypoints * kFeaturesPerKeypoint;       // 56
    const int expectedFeaturesV26 = 4 + 1 + 1 + kNumKeypoints * kFeaturesPerKeypoint;  // 57

    if (outputShape.size() == 3 && outputShape[2] == expectedFeaturesV26) {
        return postprocessV26(image.size(), actualSize, rawOutput, outputShape, confThreshold);
    } else if (outputShape.size() == 3 && outputShape[1] == expectedFeaturesV8) {
        return postprocessV8(image.size(), actualSize, rawOutput, outputShape, confThreshold, iouThreshold);
    }

    std::cerr << "[ERROR] Unsupported pose model output shape" << std::endl;
    return {};
}

std::vector<yolos::pose::PoseResult> FastPoseDetector::postprocessV8(
    const cv::Size& originalSize, const cv::Size& resizedShape,
    const float* rawOutput, const std::vector<int64_t>& outputShape,
    float confThreshold, float iouThreshold)
{
    std::vector<yolos::pose::PoseResult> results;
    const size_t numDetections = outputShape[2];

    float scale, padX, padY;
    yolos::preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
    const float invScale = 1.0f / scale;

    std::vector<yolos::BoundingBox> boxes;
    std::vector<float> confidences;
    std::vector<std::vector<yolos::KeyPoint>> allKeypoints;
    boxes.reserve(64);
    confidences.reserve(64);
    allKeypoints.reserve(64);

    for (size_t d = 0; d < numDetections; ++d) {
        const float objConfidence = rawOutput[4 * numDetections + d];
        if (objConfidence < confThreshold) continue;

        const float cx = rawOutput[0 * numDetections + d];
        const float cy = rawOutput[1 * numDetections + d];
        const float w = rawOutput[2 * numDetections + d];
        const float h = rawOutput[3 * numDetections + d];

        yolos::BoundingBox box;
        box.x = yolos::utils::clamp(static_cast<int>((cx - w * 0.5f - padX) * invScale), 0, originalSize.width - 1);
        box.y = yolos::utils::clamp(static_cast<int>((cy - h * 0.5f - padY) * invScale), 0, originalSize.height - 1);
        box.width = yolos::utils::clamp(static_cast<int>(w * invScale), 1, originalSize.width - box.x);
        box.height = yolos::utils::clamp(static_cast<int>(h * invScale), 1, originalSize.height - box.y);

        std::vector<yolos::KeyPoint> keypoints;
        keypoints.reserve(kNumKeypoints);
        for (int k = 0; k < kNumKeypoints; ++k) {
            const int offset = 5 + k * kFeaturesPerKeypoint;
            yolos::KeyPoint kpt;
            kpt.x = (rawOutput[offset * numDetections + d] - padX) * invScale;
            kpt.y = (rawOutput[(offset + 1) * numDetections + d] - padY) * invScale;
            const float rawConf = rawOutput[(offset + 2) * numDetections + d];
            kpt.confidence = 1.0f / (1.0f + std::exp(-rawConf));

            kpt.x = yolos::utils::clamp(kpt.x, 0.0f, static_cast<float>(originalSize.width - 1));
            kpt.y = yolos::utils::clamp(kpt.y, 0.0f, static_cast<float>(originalSize.height - 1));
            keypoints.push_back(kpt);
        }

        boxes.push_back(box);
        confidences.push_back(objConfidence);
        allKeypoints.push_back(std::move(keypoints));
    }

    if (boxes.empty()) return results;

    std::vector<int> indices;
    yolos::nms::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, indices);

    results.reserve(indices.size());
    for (int idx : indices) {
        results.emplace_back(boxes[idx], confidences[idx], 0, allKeypoints[idx]);
    }
    return results;
}

std::vector<yolos::pose::PoseResult> FastPoseDetector::postprocessV26(
    const cv::Size& originalSize, const cv::Size& resizedShape,
    const float* rawOutput, const std::vector<int64_t>& outputShape,
    float confThreshold)
{
    std::vector<yolos::pose::PoseResult> results;
    const size_t numDetections = outputShape[1];
    const size_t numFeatures = outputShape[2];

    float scale, padX, padY;
    yolos::preprocessing::getScalePad(originalSize, resizedShape, scale, padX, padY);
    const float invScale = 1.0f / scale;

    for (size_t d = 0; d < numDetections; ++d) {
        const size_t base = d * numFeatures;

        const float x1 = rawOutput[base + 0];
        const float y1 = rawOutput[base + 1];
        const float x2 = rawOutput[base + 2];
        const float y2 = rawOutput[base + 3];
        const float conf = rawOutput[base + 4];
        if (conf < confThreshold) continue;

        yolos::BoundingBox box;
        box.x = yolos::utils::clamp(static_cast<int>((x1 - padX) * invScale), 0, originalSize.width - 1);
        box.y = yolos::utils::clamp(static_cast<int>((y1 - padY) * invScale), 0, originalSize.height - 1);
        const int x2_scaled = yolos::utils::clamp(static_cast<int>((x2 - padX) * invScale), 0, originalSize.width - 1);
        const int y2_scaled = yolos::utils::clamp(static_cast<int>((y2 - padY) * invScale), 0, originalSize.height - 1);
        box.width = std::max(1, x2_scaled - box.x);
        box.height = std::max(1, y2_scaled - box.y);

        std::vector<yolos::KeyPoint> keypoints;
        keypoints.reserve(kNumKeypoints);
        for (int k = 0; k < kNumKeypoints; ++k) {
            const size_t kptBase = base + 6 + k * kFeaturesPerKeypoint;
            yolos::KeyPoint kpt;
            kpt.x = (rawOutput[kptBase + 0] - padX) * invScale;
            kpt.y = (rawOutput[kptBase + 1] - padY) * invScale;
            kpt.confidence = rawOutput[kptBase + 2];

            kpt.x = yolos::utils::clamp(kpt.x, 0.0f, static_cast<float>(originalSize.width - 1));
            kpt.y = yolos::utils::clamp(kpt.y, 0.0f, static_cast<float>(originalSize.height - 1));
            keypoints.push_back(kpt);
        }

        results.emplace_back(box, conf, 0, std::move(keypoints));
    }
    return results;
}

} // namespace qtrobot::yolo
