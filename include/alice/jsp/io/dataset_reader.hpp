#pragma once

#include <filesystem>

#include "alice/jsp/config/experiment_config.hpp"
#include "alice/jsp/data/dataset.hpp"

namespace alice::jsp::io {

class DatasetReader {
  public:
    static data::Dataset LoadFromConfig(const config::ExperimentConfig& config,
                                        const std::filesystem::path& config_path);
    static data::Dataset LoadFromFile(const std::filesystem::path& file_path,
                                      int max_instances = 0);
};

}  // namespace alice::jsp::io



