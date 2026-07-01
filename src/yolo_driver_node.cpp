#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include <magpie/utils/logger.hpp>
#include <magpie/frames/image_frame.hpp>
#include <magpie/frames/primitive_frames.hpp>
#include <magpie/transport/zmq_rpc_requester.hpp>
#include <magpie/discovery/zconf_discovery.hpp>

#include <paramify/paramify.hpp>

#include <qtrobot/yolo_driver_node.hpp>
#include <qtrobot/utility.hpp>

using namespace magpie;

namespace qtrobot::yolo {

namespace {
    // stream topics (ours)
    constexpr const char* personsTopic = "/persons";
    constexpr const char* imageTopic   = "/annotated_image";

    // stream topics we consume from qtrobot-realsense-driver
    constexpr const char* cameraColorTopic        = "/camera/color/image";
    constexpr const char* cameraDepthAlignedTopic = "/camera/depth/aligned/image";

    // Standard COCO 17-keypoint skeleton, indices into COCO_KEYPOINTS — used
    // only for the optional annotated-image overlay.
    constexpr std::pair<int, int> kSkeleton[] = {
        {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
        {5, 11},  {6, 12},  {5, 6},   {5, 7},   {6, 8},
        {7, 9},   {8, 10},  {1, 2},   {0, 1},   {0, 2},
        {1, 3},   {2, 4},   {3, 5},   {4, 6},
    };
    constexpr float kKeypointMinConf = 0.3f;

    // Parses "tcp://host:port" and rebuilds it with a different port.
    std::string withPort(const std::string& endpoint, uint16_t port) {
        auto schemeEnd = endpoint.find("://");
        std::string host = endpoint;
        if (schemeEnd != std::string::npos) {
            host = endpoint.substr(schemeEnd + 3);
        }
        auto colon = host.find_last_of(':');
        if (colon != std::string::npos) {
            host = host.substr(0, colon);
        }
        return "tcp://" + host + ":" + std::to_string(port);
    }
}


YoloDriverNode::YoloDriverNode(
    YoloDriverConfig& config,
    std::string name
)
    : BaseNode(config, false, std::move(name))
    , config_(std::move(config))
{}


std::string YoloDriverNode::resolveCameraStreamEndpoint(uint16_t portOffset) {
    if (!config_.camera_endpoint.empty()) {
        // camera_endpoint is the realsense driver's base RPC endpoint.
        // Color stream = base+1, depth_aligned stream = base+2.
        auto colon = config_.camera_endpoint.find_last_of(':');
        uint16_t basePort = 0;
        if (colon != std::string::npos) {
            basePort = static_cast<uint16_t>(std::stoi(config_.camera_endpoint.substr(colon + 1)));
        }
        return withPort(config_.camera_endpoint, static_cast<uint16_t>(basePort + portOffset));
    }

    ZconfDiscovery disc;
    disc.start();
    ZconfDiscovery::NodeInfo info;
    bool ok = disc.resolve_node(config_.camera_node_id, info, 5.0);
    disc.close();

    if (!ok || !info.is_resolved()) {
        throw std::runtime_error(
            "Could not discover camera node_id='" + config_.camera_node_id +
            "' — set camera.endpoint directly if discovery is unavailable."
        );
    }

    std::string ip = ZconfDiscovery::pick_best_ip(info);
    return "tcp://" + ip + ":" + std::to_string(info.port + portOffset);
}


float YoloDriverNode::sampleDepth(const cv::Mat& depth16, int u, int v, int patch) {
    if (depth16.empty()) return 0.0f;
    const int h = depth16.rows, w = depth16.cols;
    u = std::clamp(u, patch, w - patch - 1);
    v = std::clamp(v, patch, h - patch - 1);
    std::vector<uint16_t> vals;
    vals.reserve((2*patch+1) * (2*patch+1));
    for (int r = v - patch; r <= v + patch; ++r) {
        const uint16_t* row = depth16.ptr<uint16_t>(r);
        for (int c = u - patch; c <= u + patch; ++c) {
            if (row[c] > 0) vals.push_back(row[c]);
        }
    }
    if (vals.empty()) return 0.0f;
    auto mid = vals.begin() + vals.size() / 2;
    std::nth_element(vals.begin(), mid, vals.end());
    return static_cast<float>(*mid);
}


void YoloDriverNode::setup()
{
    rpcPort_    = config_.zmq_port;
    streamPort_ = config_.zmq_port + 1;
    imagePort_  = config_.zmq_port + 2;
    frameIntervalSec_ = 1.0 / static_cast<double>(config_.framerate);

    std::ostringstream desc;
    desc << "node_id: \"" << config_.node_id << "\"\n"
         << "rpc: {}\n"
         << "stream:\n"
         << "  " << personsTopic << ":\n"
         << "    direction: out\n"
         << "    frame_type: DictFrame\n"
         << "    transports:\n"
         << "      zmq:\n"
         << "        endpoint: \"tcp://*:" << streamPort_ << "\"\n"
         << "        delivery: latest\n"
         << "        queue_size: 1\n";
    if (config_.stream_annotated_image) {
        desc << "\n"
             << "  " << imageTopic << ":\n"
             << "    direction: out\n"
             << "    frame_type: ImageFrameRaw\n"
             << "    transports:\n"
             << "      zmq:\n"
             << "        endpoint: \"tcp://*:" << imagePort_ << "\"\n"
             << "        delivery: latest\n"
             << "        queue_size: 1\n";
    }
    sysDescription_ = value_from_yaml(desc.str());

    Logger::info("YoloDriverNode: loading model '" + config_.model + "' (device=" + config_.device +
                 ", num_threads=" + std::to_string(config_.num_threads) + ")");
    backend_ = std::make_unique<PoseBackend>(
        config_.model, config_.confidence, config_.image_size, config_.device == "cuda", config_.num_threads
    );

    std::string cameraEndpoint = resolveCameraStreamEndpoint(1);
    Logger::info("YoloDriverNode: reading camera color stream from " + cameraEndpoint);
    cameraReader_ = std::make_unique<ZmqStreamReader>(
        cameraEndpoint, cameraColorTopic, /*queueSize=*/1, /*bind=*/false, "latest"
    );

    if (config_.depth_enabled) {
        // Fetch the real depth scale from the realsense driver via RPC before
        // opening the stream — avoids hardcoding a default that may differ per device.
        try {
            std::string rpcEndpoint = resolveCameraStreamEndpoint(0);
            ZmqRpcRequester scaleReq(rpcEndpoint);
            Value req = Value::fromDict({
                {"name",  Value::fromString("/camera/depth/scale")},
                {"args",  Value::fromDict({})}
            });
            Value resp = scaleReq.call(req, 5.0);
            scaleReq.close();
            const auto& respDict = resp.asDict();
            if (respDict.count("status") && respDict.at("status").asBool() &&
                respDict.count("response")) {
                const auto& inner = respDict.at("response").asDict();
                if (inner.count("scale")) {
                    depthScale_ = static_cast<float>(inner.at("scale").asDouble());
                }
            }
            Logger::info("YoloDriverNode: depth scale from realsense driver = " +
                         std::to_string(depthScale_) + " m/unit");
        } catch (const std::exception& e) {
            Logger::warning(std::string("YoloDriverNode: could not fetch depth scale via RPC (") +
                            e.what() + "), using default " + std::to_string(depthScale_));
        }

        std::string depthEndpoint = resolveCameraStreamEndpoint(2);
        Logger::info("YoloDriverNode: reading depth_aligned stream from " + depthEndpoint);
        depthReader_ = std::make_unique<ZmqStreamReader>(
            depthEndpoint, cameraDepthAlignedTopic, /*queueSize=*/1, /*bind=*/false, "latest"
        );
    }

    personsWriter_ = std::make_unique<ZmqStreamWriter>(
        "tcp://*:" + std::to_string(streamPort_), 1, true, "latest"
    );
    Logger::info("YoloDriverNode: publishing persons on " + std::string(personsTopic) +
                 " (port " + std::to_string(streamPort_) + ")");

    if (config_.stream_annotated_image) {
        imageWriter_ = std::make_unique<ZmqStreamWriter>(
            "tcp://*:" + std::to_string(imagePort_), 1, true, "latest"
        );
        Logger::info("YoloDriverNode: publishing annotated image on " + std::string(imageTopic) +
                     " (port " + std::to_string(imagePort_) + ")");
    }

    responder_ = std::make_shared<ZmqRpcResponder>("tcp://*:" + std::to_string(rpcPort_));
    server_ = std::make_unique<ServerNode>(
        responder_,
        [this](const Value& req) { return this->onRequest(req); }
    );
    server_->start();
}


void YoloDriverNode::process()
{
    std::unique_ptr<Frame> frame;
    std::string topic;

    try {
        bool ok = cameraReader_->read(frame, topic, 1.0);
        if (!ok || !frame) {
            Logger::debug("YoloDriverNode: no camera frame within timeout");
            return;
        }
    } catch (const std::exception& e) {
        Logger::warning(std::string("YoloDriverNode: camera read error: ") + e.what());
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return;
    }
    // Logger::debug("YoloDriverNode: got camera frame on topic '" + topic + "'");

    auto now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (now - lastFrameTs_ < frameIntervalSec_) {
        return;
    }

    lastFrameTs_ = now;

    auto* img = dynamic_cast<ImageFrameRaw*>(frame.get());
    if (!img || img->format() != "raw") {
        Logger::warning("YoloDriverNode: unexpected camera frame type/format");
        return;
    }
    if (img->channels() != 3) {
        Logger::warning("YoloDriverNode: expected a 3-channel color frame, got " +
                         std::to_string(img->channels()) + " channels");
        return;
    }

    cv::Mat bgr(img->height(), img->width(), CV_8UC3, img->data().data());
    Logger::debug("YoloDriverNode: running detection on " + std::to_string(bgr.cols) +
                  "x" + std::to_string(bgr.rows) + " frame");

    // Non-blocking grab of the latest depth_aligned frame (timeout=0).
    cv::Mat depthImg;
    if (depthReader_ != nullptr) {
        std::unique_ptr<Frame> depthFrame;
        std::string depthTopic;
        if (depthReader_->read(depthFrame, depthTopic, 0.0) && depthFrame) {
            auto* dImg = dynamic_cast<ImageFrameRaw*>(depthFrame.get());
            if (dImg && dImg->channels() == 1) {
                depthImg = cv::Mat(dImg->height(), dImg->width(), CV_16UC1,
                                   const_cast<void*>(static_cast<const void*>(dImg->data().data()))).clone();
            }
        }
    }

    auto persons = backend_->detect(bgr);
    Logger::debug("YoloDriverNode: detected " + std::to_string(persons.size()) + " person(s)");
    for (const auto& p : persons) {
        const auto& nose = p.keypoints[0]; // COCO_KEYPOINTS[0] == "nose"
        std::ostringstream pLog;
        pLog << "  track_id=" << p.trackId
             << " bbox=[" << p.x1 << "," << p.y1 << "," << p.x2 << "," << p.y2 << "]"
             << " conf=" << p.confidence
             << " nose_uv=[" << nose.u << "," << nose.v << "] nose_conf=" << nose.confidence;
        Logger::debug(pLog.str());
    }

    // Evict stale track IDs from depth EMA map.
    if (!depthEma_.empty()) {
        std::unordered_map<int, std::array<float, 17>> keep;
        for (const auto& p : persons) {
            auto it = depthEma_.find(p.trackId);
            if (it != depthEma_.end()) keep.insert(*it);
        }
        depthEma_.swap(keep);
    }

    if (!persons.empty()) {
        personsWriter_->write(DictFrame(buildPersonsValue(persons, depthImg).asDict()), personsTopic);
    }

    if (imageWriter_ != nullptr) {
        writeAnnotatedImage(bgr, persons);
    }
}


void YoloDriverNode::cleanup()
{
    Logger::info("YoloDriverNode: cleanup");
    if (cameraReader_ != nullptr) { cameraReader_->close(); }
    if (depthReader_  != nullptr) { depthReader_->close(); }
    if (personsWriter_ != nullptr) { personsWriter_->close(); }
    if (imageWriter_ != nullptr) { imageWriter_->close(); }
    if (server_ != nullptr) { server_->terminate(); }
}


magpie::Value YoloDriverNode::buildPersonsValue(
    const std::vector<PersonDetection>& persons,
    const cv::Mat& depthImg)
{
    constexpr float kDepthEmaAlpha = 0.4f;
    const bool haveDepth = config_.depth_enabled && !depthImg.empty();

    Value::Dict personsDict;

    for (const auto& p : persons) {
        Value::Dict personEntry;

        Value::List bbox;
        bbox.push_back(Value::fromDouble(p.x1));
        bbox.push_back(Value::fromDouble(p.y1));
        bbox.push_back(Value::fromDouble(p.x2));
        bbox.push_back(Value::fromDouble(p.y2));
        personEntry["bbox"] = Value::fromList(bbox);
        personEntry["confidence"] = Value::fromDouble(p.confidence);

        // Depth EMA state for this track (initialised to 0 = no reading yet).
        auto& ema = depthEma_[p.trackId];

        Value::Dict keypointsDict;
        for (size_t k = 0; k < COCO_KEYPOINTS.size(); ++k) {
            const auto& kp = p.keypoints[k];
            Value::Dict kpEntry;

            Value::List uv;
            uv.push_back(Value::fromDouble(kp.u));
            uv.push_back(Value::fromDouble(kp.v));
            kpEntry["uv"] = Value::fromList(uv);
            kpEntry["conf"] = Value::fromDouble(kp.confidence);

            if (haveDepth) {
                float raw = sampleDepth(depthImg, static_cast<int>(kp.u), static_cast<int>(kp.v));
                if (raw > 0.0f) {
                    float rawM = raw * depthScale_;
                    ema[k] = (ema[k] > 0.0f)
                             ? kDepthEmaAlpha * rawM + (1.0f - kDepthEmaAlpha) * ema[k]
                             : rawM;
                }
                // Publish smoothed depth if we have a reading, otherwise -1 (no valid depth).
                kpEntry["depth"] = Value::fromDouble(ema[k] > 0.0f ? ema[k] : -1.0);
            }

            keypointsDict[COCO_KEYPOINTS[k]] = Value::fromDict(kpEntry);
        }
        personEntry["keypoints"] = Value::fromDict(keypointsDict);

        personsDict[std::to_string(p.trackId)] = Value::fromDict(personEntry);
    }

    Value::Dict root;
    root["persons"] = Value::fromDict(personsDict);
    return Value::fromDict(root);
}


void YoloDriverNode::writeAnnotatedImage(const cv::Mat& bgrImage, const std::vector<PersonDetection>& persons)
{
    cv::Mat annotated = bgrImage.clone();
    const cv::Scalar color(0, 100, 255);

    for (const auto& p : persons) {
        cv::rectangle(annotated,
                      cv::Point(static_cast<int>(p.x1), static_cast<int>(p.y1)),
                      cv::Point(static_cast<int>(p.x2), static_cast<int>(p.y2)),
                      color, 2);
        cv::putText(annotated, "ID " + std::to_string(p.trackId),
                    cv::Point(static_cast<int>(p.x1), std::max(0, static_cast<int>(p.y1) - 8)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);

        for (const auto& edge : kSkeleton) {
            const auto& a = p.keypoints[edge.first];
            const auto& b = p.keypoints[edge.second];
            if (a.confidence < kKeypointMinConf || b.confidence < kKeypointMinConf) continue;
            cv::line(annotated, cv::Point(static_cast<int>(a.u), static_cast<int>(a.v)),
                     cv::Point(static_cast<int>(b.u), static_cast<int>(b.v)), color, 2);
        }
        for (const auto& kp : p.keypoints) {
            if (kp.confidence < kKeypointMinConf) continue;
            cv::circle(annotated, cv::Point(static_cast<int>(kp.u), static_cast<int>(kp.v)), 3, color, -1);
        }
    }

    ImageFrameRaw outFrame(
        std::vector<std::uint8_t>(annotated.data, annotated.data + annotated.total() * annotated.elemSize()),
        "raw", annotated.cols, annotated.rows, annotated.channels(), "bgr8"
    );
    imageWriter_->write(outFrame, imageTopic);
}


magpie::Value YoloDriverNode::onRequest(const magpie::Value& req_)
{
    Value::Dict response;

    try {
        const Value::Dict& request = req_.asDict();
        std::string name;
        auto name_it = request.find("name");
        if (name_it != request.end() && name_it->second.type() == Value::Type::String)
            name = name_it->second.asString();

        Value payload;
        if (name.empty()) {
            payload = filterSysDescriptor(sysDescription_, req_);
        } else {
            throw std::runtime_error("unknown service name: " + name);
        }

        response["status"]   = Value::fromBool(true);
        response["response"] = payload;
    }
    catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "YoloDriverNode::onRequest wrong request: "
            << req_.toDebugString() << " (" << e.what() << ")";
        Logger::warning(oss.str());

        response["status"]   = Value::fromBool(false);
        response["response"] = Value();
    }

    return Value::fromDict(response);
}


std::atomic<bool> running{true};

} // namespace qtrobot::yolo


int main(int argc, char** argv) {
    using namespace qtrobot::yolo;

    std::signal(SIGINT,  [](int) { running.store(false, std::memory_order_relaxed); });
    std::signal(SIGTERM, [](int) { running.store(false, std::memory_order_relaxed); });

    paramify::Paramify params;
    params.load_from_yaml_string(paramsYaml);

    if (!params.apply_cli_or_exit(argc, argv)) {
        return 1;
    }

    Logger::setLevel(params.get_string("log_level"));

    YoloDriverConfig config;
    try {
        config.model      = params.get_string("vision.model");
        config.framerate  = (int64_t) params["vision.framerate"];
        config.confidence = (double) params["vision.confidence"];
        config.image_size = (int64_t) params["vision.image_size"];
        config.stream_annotated_image = (bool) params["vision.stream_annotated_image"];
        config.num_threads = (int64_t) params["vision.num_threads"];
        config.device      = params.get_string("device");
        config.depth_enabled = (bool) params["depth.enabled"];
        config.camera_node_id = params.get_string("camera.node_id");
        config.camera_endpoint = params.get_string("camera.endpoint");
        config.zmq_port  = (uint16_t) params.get_int("zmq.port");
        config.node_id   = params.get_string("zmq.node_id");
    }
    catch (paramify::ParamError e) {
        Logger::error(e.what());
        return 1;
    }

    YoloDriverNode node(config);
    node.start();

    const std::string nodeName = params.get_string("zmq.node_id");
    ZconfDiscovery disc;
    disc.advertise(nodeName, config.zmq_port, R"({"version":"0.1.0"})");

    while (running.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    disc.close();
    node.terminate();

    return 0;
}
