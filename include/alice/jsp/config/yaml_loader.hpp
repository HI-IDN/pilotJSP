#pragma once

#include <filesystem>

#include "alice/jsp/config/experiment_config.hpp"

namespace alice::jsp::config {

ExperimentConfig LoadExperimentConfig(const std::filesystem::path& path);

}  // namespace alice::jsp::config



