#include "alice/jsp/config/yaml_loader.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace alice::jsp::config {
namespace {

std::string Trim(std::string value) {
    const auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    while (!value.empty() && is_space(value.front())) {
        value.erase(value.begin());
    }
    while (!value.empty() && is_space(value.back())) {
        value.pop_back();
    }
    return value;
}

std::vector<std::string> Split(const std::string& value, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, delimiter)) {
        parts.push_back(Trim(item));
    }
    return parts;
}

void AddFeature(const std::string& value, ExperimentConfig& config) {
    if (!value.empty()) {
        config.features.push_back(value);
    }
}

void ApplyScalar(const std::string& key, const std::string& value, DataConfig& data) {
    if (key == "name") {
        data.name = value;
        return;
    }
    if (key == "generator") {
        data.generator = value;
        return;
    }
    if (key == "instances") {
        data.instances = std::stoi(value);
        return;
    }
    if (key == "durationLB") {
        data.duration_lb = std::stoi(value);
        return;
    }
    if (key == "durationUB") {
        data.duration_ub = std::stoi(value);
        return;
    }
    if (key == "set") {
        data.set = value;
        return;
    }
    if (key == "root") {
        data.root = value;
        return;
    }
    if (key == "file") {
        data.file = value;
        return;
    }
}

void ApplyInstanceSize(const std::string& key, const std::string& value, DataConfig& data) {
    if (key == "jobs") {
        data.jobs = std::stoi(value);
    } else if (key == "machines") {
        data.machines = std::stoi(value);
    }
}

void ApplyExperimentScalar(const std::string& key, const std::string& value, ExperimentConfig& config) {
    if (key == "domain") {
        config.domain = value;
        return;
    }
    if (key == "features") {
        config.features_specified = true;
        if (!value.empty() && value.front() == '[') {
            std::string list_value = value;
            if (list_value.back() == ']') {
                list_value = list_value.substr(1, list_value.size() - 2);
            } else {
                list_value = list_value.substr(1);
            }
            for (const auto& entry : Split(list_value, ',')) {
                AddFeature(entry, config);
            }
        } else {
            AddFeature(value, config);
        }
    }
}

}  // namespace

ExperimentConfig LoadExperimentConfig(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open config file: " + path.string());
    }

    ExperimentConfig config;
    std::string line;
    bool in_data = false;
    bool in_instance_size = false;
    int instance_size_indent = 0;
    bool in_features = false;
    int features_indent = 0;

    while (std::getline(input, line)) {
        if (line.find('#') != std::string::npos) {
            line = line.substr(0, line.find('#'));
        }
        if (Trim(line).empty()) {
            continue;
        }

        int indent = 0;
        while (indent < static_cast<int>(line.size()) && std::isspace(static_cast<unsigned char>(line[indent]))) {
            indent++;
        }

        std::string trimmed = Trim(line);
        if (trimmed == "data:") {
            in_data = true;
            in_instance_size = false;
            continue;
        }
        if (trimmed == "features:") {
            in_features = true;
            features_indent = indent;
            config.features_specified = true;
            continue;
        }
        if (indent == 0) {
            in_data = false;
            in_instance_size = false;
        }
        if (in_features && indent <= features_indent) {
            in_features = false;
        }

        if (in_instance_size && indent <= instance_size_indent) {
            in_instance_size = false;
        }

        const auto colon = trimmed.find(':');
        if (colon == std::string::npos) {
            if (in_features && trimmed.rfind("-", 0) == 0) {
                auto feature_name = Trim(trimmed.substr(1));
                config.features_specified = true;
                AddFeature(feature_name, config);
            }
            continue;
        }

        std::string key = Trim(trimmed.substr(0, colon));
        std::string value = Trim(trimmed.substr(colon + 1));

        if (!in_data) {
            ApplyExperimentScalar(key, value, config);
            continue;
        }

        if (key == "instance_size") {
            if (!value.empty() && value.front() == '{') {
                if (value.back() == '}') {
                    value = value.substr(1, value.size() - 2);
                } else {
                    value = value.substr(1);
                }
                for (const auto& entry : Split(value, ',')) {
                    const auto map_colon = entry.find(':');
                    if (map_colon == std::string::npos) {
                        continue;
                    }
                    auto map_key = Trim(entry.substr(0, map_colon));
                    auto map_value = Trim(entry.substr(map_colon + 1));
                    ApplyInstanceSize(map_key, map_value, config.data);
                }
            } else {
                in_instance_size = true;
                instance_size_indent = indent;
            }
            continue;
        }

        if (in_instance_size) {
            ApplyInstanceSize(key, value, config.data);
            continue;
        }

        ApplyScalar(key, value, config.data);
    }

    return config;
}

}  // namespace alice::jsp::config



