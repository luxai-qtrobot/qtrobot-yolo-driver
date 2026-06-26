#include <qtrobot/pose_backend.hpp>

#include <yolos/tasks/pose.hpp>
#include <motcpp/trackers/bytetrack.hpp>

namespace qtrobot::yolo {

namespace {
// vision.model defaults to a .onnx path (same config field used by the onnx
// backend, and the .onnx file postinst's trtexec conversion reads from) — the
// TensorRT backend loads the sibling .trt engine instead, so this derives
// "<dir>/<name>.trt" from "<dir>/<name>.onnx" without needing a separate
// config field. If modelPath doesn't end in .onnx, it's used unmodified
// (assume the user already pointed it straight at a .trt engine).
std::string deriveEnginePath(const std::string& modelPath) {
    constexpr const char* kOnnxSuffix = ".onnx";
    const size_t suffixLen = 5; // strlen(".onnx")
    if (modelPath.size() > suffixLen &&
        modelPath.compare(modelPath.size() - suffixLen, suffixLen, kOnnxSuffix) == 0) {
        return modelPath.substr(0, modelPath.size() - suffixLen) + ".trt";
    }
    return modelPath;
}
}

struct PoseBackend::Impl {
    yolos::pose::YOLOPoseDetector detector;
    motcpp::trackers::ByteTrack tracker;

    Impl(const std::string& enginePath)
        : detector(enginePath, /*labelsPath=*/"", /*dlaCore=*/-1)
        , tracker()
    {}
};

PoseBackend::PoseBackend(const std::string& modelPath,
                          float confidence,
                          int /*imageSize*/,
                          bool /*useGPU*/,    // TensorRT is GPU-only by construction
                          int /*numThreads*/) // no CPU thread pool concept here
    : impl_(std::make_unique<Impl>(deriveEnginePath(modelPath)))
    , confidence_(confidence)
{}

PoseBackend::~PoseBackend() = default;

std::vector<PersonDetection> PoseBackend::detect(const cv::Mat& bgrImage) {
    std::vector<PersonDetection> out;

    auto poses = impl_->detector.detect(bgrImage, confidence_, /*iouThreshold=*/0.5f);
    if (poses.empty()) {
        impl_->tracker.update(Eigen::MatrixXf(0, 6), bgrImage);
        return out;
    }

    Eigen::MatrixXf dets(static_cast<int>(poses.size()), 6);
    for (int i = 0; i < static_cast<int>(poses.size()); ++i) {
        const auto& box = poses[i].box;
        dets(i, 0) = static_cast<float>(box.x);
        dets(i, 1) = static_cast<float>(box.y);
        dets(i, 2) = static_cast<float>(box.x + box.width);
        dets(i, 3) = static_cast<float>(box.y + box.height);
        dets(i, 4) = poses[i].conf;
        dets(i, 5) = static_cast<float>(poses[i].classId);
    }

    Eigen::MatrixXf tracks = impl_->tracker.update(dets, bgrImage);

    out.reserve(static_cast<size_t>(tracks.rows()));
    for (int i = 0; i < tracks.rows(); ++i) {
        const int detIdx = static_cast<int>(tracks(i, 7));
        if (detIdx < 0 || detIdx >= static_cast<int>(poses.size())) {
            continue;
        }

        PersonDetection p;
        p.x1 = tracks(i, 0);
        p.y1 = tracks(i, 1);
        p.x2 = tracks(i, 2);
        p.y2 = tracks(i, 3);
        p.trackId = static_cast<int>(tracks(i, 4));
        p.confidence = tracks(i, 5);

        const auto& kpts = poses[detIdx].keypoints;
        for (size_t k = 0; k < p.keypoints.size() && k < kpts.size(); ++k) {
            p.keypoints[k].u = kpts[k].x;
            p.keypoints[k].v = kpts[k].y;
            p.keypoints[k].confidence = kpts[k].confidence;
        }

        out.push_back(p);
    }

    return out;
}

} // namespace qtrobot::yolo
