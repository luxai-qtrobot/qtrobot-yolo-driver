#include <qtrobot/pose_backend.hpp>

#include <qtrobot/fast_pose_detector.hpp>
#include <motcpp/trackers/bytetrack.hpp>

namespace qtrobot::yolo {

struct PoseBackend::Impl {
    FastPoseDetector detector;
    motcpp::trackers::ByteTrack tracker;

    Impl(const std::string& modelPath, bool useGPU, int numThreads)
        : detector(modelPath, useGPU, numThreads)
        // Defaults match motcpp's own ByteTrack defaults; per_class/nr_classes
        // are irrelevant here since we only ever feed it class 0 (person).
        , tracker()
    {}
};

PoseBackend::PoseBackend(const std::string& modelPath,
                          float confidence,
                          int /*imageSize*/,
                          bool useGPU,
                          int numThreads)
    : impl_(std::make_unique<Impl>(modelPath, useGPU, numThreads))
    , confidence_(confidence)
{}

PoseBackend::~PoseBackend() = default;

std::vector<PersonDetection> PoseBackend::detect(const cv::Mat& bgrImage) {
    std::vector<PersonDetection> out;

    auto poses = impl_->detector.detect(bgrImage, confidence_, /*iouThreshold=*/0.5f);
    if (poses.empty()) {
        // Still call update() with an empty matrix so the tracker ages out
        // and removes stale tracks instead of freezing their last state.
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
            continue; // shouldn't happen, but don't index out of bounds
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
