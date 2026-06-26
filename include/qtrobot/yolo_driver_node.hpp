#pragma once

#include <memory>
#include <string>

#include <qtrobot/yolo_driver_config.hpp>
#include <qtrobot/pose_backend.hpp>
#include <qtrobot/emotion_recognizer.hpp>

#include <magpie/nodes/base_node.hpp>
#include <magpie/nodes/server_node.hpp>
#include <magpie/transport/zmq_stream_reader.hpp>
#include <magpie/transport/zmq_stream_writer.hpp>
#include <magpie/transport/zmq_rpc_responder.hpp>

namespace qtrobot::yolo {

class YoloDriverNode : public magpie::BaseNode {
public:
    explicit YoloDriverNode(
        YoloDriverConfig& config,
        std::string name = "yolo-driver"
    );

    ~YoloDriverNode() override = default;

protected:
    void setup() override;

    void process() override;

    void cleanup() override;

private:
    magpie::Value onRequest(const magpie::Value& req);
    std::string resolveCameraEndpoint();
    magpie::Value buildPersonsValue(const std::vector<PersonDetection>& persons,
                                     const std::vector<EmotionResult>& emotions);
    void writeAnnotatedImage(const cv::Mat& bgrImage, const std::vector<PersonDetection>& persons);

private:
    YoloDriverConfig config_;
    uint16_t rpcPort_     = 0;
    uint16_t streamPort_  = 0;
    uint16_t imagePort_   = 0;
    double frameIntervalSec_ = 0.0;
    double lastFrameTs_      = 0.0;

    std::shared_ptr<magpie::ZmqRpcResponder> responder_;
    std::unique_ptr<magpie::ServerNode> server_;
    std::unique_ptr<magpie::ZmqStreamReader> cameraReader_;
    std::unique_ptr<magpie::ZmqStreamWriter> personsWriter_;
    std::unique_ptr<magpie::ZmqStreamWriter> imageWriter_;
    std::unique_ptr<PoseBackend> backend_;
    std::unique_ptr<EmotionRecognizer> emotionRecognizer_;

    magpie::Value sysDescription_;
};

} // namespace qtrobot::yolo
