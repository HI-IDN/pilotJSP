#pragma once

#include <string>
#include <vector>

namespace alice::jsp::config {

struct DataConfig {
    std::string name;
    std::string generator;
    int jobs = 0;
    int machines = 0;
    int instances = 0;
    int duration_lb = 0;
    int duration_ub = 0;
    std::string set = "train";
    std::string root;
    std::string file;
};

struct ExperimentConfig {
    DataConfig data;
    std::string domain;
    std::vector<std::string> features;
    bool features_specified = false;
};

}  // namespace alice::jsp::config



