

#include <unordered_map>
#include <stdexcept>

#include <yaml-cpp/yaml.h>
#include <string>
#include <cstdint>

#include <qtrobot/utility.hpp>

namespace qtrobot::yolo {

using namespace magpie;

static Value convert_yaml_node(const YAML::Node& node) {
    if (!node || node.IsNull()) {
        return Value::null();
    }

    if (node.IsSequence()) {
        Value::List out;
        out.reserve(node.size());
        for (const auto& it : node) {
            out.push_back(convert_yaml_node(it));
        }
        return Value::fromList(out);
    }

    if (node.IsMap()) {
        Value::Dict out;
        for (const auto& kv : node) {
            const YAML::Node& k = kv.first;
            const YAML::Node& v = kv.second;

            if (!k.IsScalar()) {
                throw std::runtime_error("YAML: map key is not a scalar string");
            }

            const std::string key = k.as<std::string>();
            out.emplace(key, convert_yaml_node(v));
        }
        return Value::fromDict(out);
    }

    // Scalar
    // Try in a safe order: bool -> int -> double -> string
    try {
        bool b = node.as<bool>();
        return Value::fromBool(b);
    } catch (...) {}

    try {
        std::int64_t i = node.as<std::int64_t>();
        return Value::fromInt(i);
    } catch (...) {}

    try {
        double d = node.as<double>();
        return Value::fromDouble(d);
    } catch (...) {}

    // Fallback to string (also covers things like "tcp://*:5780")
    return Value::fromString(node.as<std::string>());
}

magpie::Value value_from_yaml(const std::string& yaml_text) {
    YAML::Node root = YAML::Load(yaml_text);
    return convert_yaml_node(root);
}

}
