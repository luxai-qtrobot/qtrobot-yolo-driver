
#pragma once

#include <string>
#include <magpie/serializer/value.hpp>

namespace qtrobot::yolo {

magpie::Value value_from_yaml(const std::string& yaml_text);

// Filter and/or strip the full system descriptor based on request args.
//
// Supported args (all optional):
//   detail  bool   default false  — if true, include params/returns/descriptions
//   filter  str    default ""     — prefix filter on path (e.g. "/yolo/")
//   scope   str    default "all"  — "rpc", "stream", or "all"
//
// Default (no args): all RPCs with transports only + all streams in full.
inline magpie::Value filterSysDescriptor(const magpie::Value& full, const magpie::Value& req) {
    using Type = magpie::Value::Type;

    bool detail = false;
    std::string filter_prefix;
    std::string scope = "all";

    if (req.type() == Type::Dict) {
        const auto& req_dict = req.asDict();
        auto args_it = req_dict.find("args");
        if (args_it != req_dict.end() && args_it->second.type() == Type::Dict) {
            const auto& args = args_it->second.asDict();

            auto it = args.find("detail");
            if (it != args.end() && it->second.type() == Type::Bool)
                detail = it->second.asBool();

            it = args.find("filter");
            if (it != args.end() && it->second.type() == Type::String)
                filter_prefix = it->second.asString();

            it = args.find("scope");
            if (it != args.end() && it->second.type() == Type::String)
                scope = it->second.asString();
        }
    }

    if (detail && filter_prefix.empty() && scope == "all")
        return full;

    if (full.type() != Type::Dict)
        return full;

    const auto& full_dict = full.asDict();
    magpie::Value::Dict result;

    for (const auto& kv : full_dict) {
        if (kv.first != "rpc" && kv.first != "stream")
            result[kv.first] = kv.second;
    }

    auto matches_filter = [&](const std::string& path) -> bool {
        if (filter_prefix.empty()) return true;
        return path.rfind(filter_prefix, 0) == 0;
    };

    if (scope == "all" || scope == "rpc") {
        auto rpc_it = full_dict.find("rpc");
        if (rpc_it != full_dict.end() && rpc_it->second.type() == Type::Dict) {
            magpie::Value::Dict filtered_rpc;
            for (const auto& kv : rpc_it->second.asDict()) {
                if (!matches_filter(kv.first)) continue;
                if (detail) {
                    filtered_rpc[kv.first] = kv.second;
                } else {
                    magpie::Value::Dict basic;
                    if (kv.second.type() == Type::Dict) {
                        auto trans_it = kv.second.asDict().find("transports");
                        if (trans_it != kv.second.asDict().end())
                            basic["transports"] = trans_it->second;
                    }
                    filtered_rpc[kv.first] = magpie::Value::fromDict(basic);
                }
            }
            result["rpc"] = magpie::Value::fromDict(filtered_rpc);
        }
    }

    if (scope == "all" || scope == "stream") {
        auto stream_it = full_dict.find("stream");
        if (stream_it != full_dict.end() && stream_it->second.type() == Type::Dict) {
            magpie::Value::Dict filtered_stream;
            for (const auto& kv : stream_it->second.asDict()) {
                if (!matches_filter(kv.first)) continue;
                filtered_stream[kv.first] = kv.second;
            }
            result["stream"] = magpie::Value::fromDict(filtered_stream);
        }
    }

    return magpie::Value::fromDict(result);
}

}
