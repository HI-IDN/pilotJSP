#include "alice/jsp/io/dataset_reader.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace alice::jsp::io {
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

bool HasAlpha(const std::string& value) {
    return std::any_of(value.begin(), value.end(), [](unsigned char ch) { return std::isalpha(ch) != 0; });
}

std::vector<int> ParseInts(const std::string& line) {
    std::string trimmed = Trim(line);
    if (trimmed.empty()) {
        return {};
    }
    if (HasAlpha(trimmed)) {
        return {};
    }

    std::vector<int> values;
    std::stringstream stream(trimmed);
    int number = 0;
    while (stream >> number) {
        values.push_back(number);
    }
    return values;
}

bool ParseNamedEntity(const std::string& line, char prefix, int& index, std::string& name) {
    std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed.front() != prefix) {
        return false;
    }

    std::size_t pos = 1;
    while (pos < trimmed.size() && std::isdigit(static_cast<unsigned char>(trimmed[pos]))) {
        pos++;
    }
    if (pos == 1) {
        return false;
    }

    index = std::stoi(trimmed.substr(1, pos - 1));
    name = Trim(trimmed.substr(pos));
    return !name.empty();
}

std::string NormalizePrefix(std::string prefix) {
    if (prefix.find('.') != std::string::npos) {
        return prefix;
    }
    if (prefix.size() >= 2) {
        return prefix.substr(0, 1) + "." + prefix.substr(1);
    }
    return prefix;
}

std::string BuildBaseName(const config::DataConfig& data) {
    if (!data.name.empty()) {
        if (data.name.find('.') != std::string::npos) {
            return data.name;
        }
        const auto underscore = data.name.find('_');
        if (underscore != std::string::npos) {
            const auto prefix = NormalizePrefix(data.name.substr(0, underscore));
            const auto dimension = data.name.substr(underscore + 1);
            if (!dimension.empty()) {
                return prefix + "." + dimension;
            }
            return prefix;
        }
        return NormalizePrefix(data.name);
    }

    if (!data.generator.empty() && data.jobs > 0 && data.machines > 0) {
        const auto prefix = NormalizePrefix(data.generator);
        return prefix + "." + std::to_string(data.jobs) + "x" + std::to_string(data.machines);
    }

    return {};
}

std::filesystem::path ResolveDataFile(const config::ExperimentConfig& config,
                                     const std::filesystem::path& config_path) {
    if (!config.data.file.empty()) {
        return config.data.file;
    }

    const auto base_name = BuildBaseName(config.data);
    if (base_name.empty()) {
        throw std::runtime_error("Config missing data.name or data.generator/instance_size");
    }

    const std::string set = config.data.set.empty() ? "train" : config.data.set;
    const auto filename = base_name + "." + set + ".txt";

    std::vector<std::filesystem::path> roots;
    if (!config.data.root.empty()) {
        roots.push_back(config.data.root);
    }

    const auto config_dir = config_path.parent_path();
    roots.push_back(config_dir / ".." / "data");
    roots.push_back(config_dir / ".." / "Data" / "Raw");
    roots.push_back(config_dir / ".." / ".." / "Data" / "Raw");
    roots.push_back(config_dir / ".." / ".." / ".." / "Data" / "Raw");

    for (const auto& root : roots) {
        auto candidate = root / filename;
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    std::string attempted;
    for (const auto& root : roots) {
        attempted += (root / filename).string() + "\n";
    }
    throw std::runtime_error("Could not find data file. Tried:\n" + attempted);
}

}  // namespace

alice::jsp::data::Dataset DatasetReader::LoadFromConfig(const config::ExperimentConfig& config,
                                                      const std::filesystem::path& config_path) {
    const auto file_path = ResolveDataFile(config, config_path);
    return LoadFromFile(file_path, config.data.instances);
}

alice::jsp::data::Dataset DatasetReader::LoadFromFile(const std::filesystem::path& file_path,
                                                    int max_instances) {
    std::ifstream input(file_path);
    if (!input) {
        throw std::runtime_error("Failed to open data file: " + file_path.string());
    }

    alice::jsp::data::Dataset dataset;
    std::string line;

    std::string current_name;
    std::string current_given_name;
    int jobs = -1;
    int macs = -1;
    int current_job = 0;
    std::vector<int> sigma;
    std::vector<int> procs;
    std::vector<std::string> job_names;
    std::vector<std::string> machine_names;

    auto finalize_instance = [&]() {
        if (jobs <= 0 || macs <= 0 || current_job != jobs) {
            return;
        }

        model::ProblemInstance instance(jobs, macs, sigma, procs);
        for (int job = 0; job < static_cast<int>(job_names.size()); job++) {
            if (!job_names[job].empty()) {
                instance.SetJobName(job, job_names[job]);
            }
        }
        for (int mac = 0; mac < static_cast<int>(machine_names.size()); mac++) {
            if (!machine_names[mac].empty()) {
                instance.SetMachineName(mac, machine_names[mac]);
            }
        }

        data::DatasetInstance entry;
        entry.name = current_name.empty() ? current_given_name : current_name;
        entry.given_name = current_given_name;
        entry.problem = std::move(instance);
        dataset.Add(std::move(entry));
    };

    while (std::getline(input, line)) {
        const auto trimmed = Trim(line);
        if (trimmed.rfind("instance ", 0) == 0) {
            finalize_instance();
            if (max_instances > 0 && static_cast<int>(dataset.Size()) >= max_instances) {
                return dataset;
            }

            current_given_name = Trim(trimmed.substr(std::string("instance ").size()));
            current_name.clear();
            jobs = -1;
            macs = -1;
            current_job = 0;
            sigma.clear();
            procs.clear();
            job_names.clear();
            machine_names.clear();
            continue;
        }

        if (jobs > 0) {
            int index = -1;
            std::string name;
            if (ParseNamedEntity(trimmed, 'J', index, name)) {
                if (index >= 0 && index < jobs) {
                    if (job_names.size() != static_cast<std::size_t>(jobs)) {
                        job_names.resize(static_cast<std::size_t>(jobs));
                    }
                    job_names[static_cast<std::size_t>(index)] = name;
                }
                continue;
            }
            if (ParseNamedEntity(trimmed, 'M', index, name)) {
                if (index >= 0 && index < macs) {
                    if (machine_names.size() != static_cast<std::size_t>(macs)) {
                        machine_names.resize(static_cast<std::size_t>(macs));
                    }
                    machine_names[static_cast<std::size_t>(index)] = name;
                }
                continue;
            }
        }

        auto values = ParseInts(trimmed);
        if (values.empty()) {
            continue;
        }

        if (values.size() == 2 && jobs < 0) {
            jobs = values[0];
            macs = values[1];
            sigma.assign(static_cast<std::size_t>(jobs * macs), 0);
            procs.assign(static_cast<std::size_t>(jobs * macs), 0);
            current_job = 0;
            job_names.resize(static_cast<std::size_t>(jobs));
            machine_names.resize(static_cast<std::size_t>(macs));
            continue;
        }

        if (jobs > 0 && macs > 0 && values.size() == static_cast<std::size_t>(2 * macs) &&
            current_job < jobs) {
            for (int mac = 0; mac < macs; mac++) {
                const auto offset = static_cast<std::size_t>(current_job * macs + mac);
                sigma[offset] = values[static_cast<std::size_t>(mac * 2)];
                procs[offset] = values[static_cast<std::size_t>(mac * 2 + 1)];
            }
            current_job++;
        }
    }

    finalize_instance();
    return dataset;
}

}  // namespace alice::jsp::io



