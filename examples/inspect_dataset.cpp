#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "alice/jsp/config/yaml_loader.hpp"
#include "alice/jsp/io/dataset_reader.hpp"

namespace {

void PrintMatrix(const std::string& title, int jobs, int machines,
                 const std::function<int(int, int)>& getter) {
    std::cout << title << "\n";
    const int cell_width = 4;
    std::cout << std::setw(5) << " ";
    for (int m = 0; m < machines; ++m) {
        std::ostringstream header;
        header << "M_" << m;
        std::cout << std::setw(cell_width) << header.str();
    }
    std::cout << "\n";
    for (int j = 0; j < jobs; ++j) {
        std::ostringstream row_label;
        row_label << "J_" << j;
        std::cout << std::setw(5) << row_label.str();
        for (int m = 0; m < machines; ++m) {
            std::cout << std::setw(cell_width) << getter(j, m);
        }
        std::cout << "\n";
    }
}

bool ValidateSigmaRow(const alice::jsp::model::ProblemInstance& instance, int job) {
    const int machines = instance.NumMachines();
    std::vector<bool> seen(static_cast<std::size_t>(machines), false);
    for (int m = 0; m < machines; ++m) {
        int value = instance.Sigma(job, m);
        if (value < 0 || value >= machines) {
            return false;
        }
        if (seen[static_cast<std::size_t>(value)]) {
            return false;
        }
        seen[static_cast<std::size_t>(value)] = true;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path config_path = "config/experiment.yaml";
    if (argc > 1) {
        config_path = argv[1];
    }

    try {
        auto config = alice::jsp::config::LoadExperimentConfig(config_path);
        auto dataset = alice::jsp::io::DatasetReader::LoadFromConfig(config, config_path);

        std::cout << "Loaded " << dataset.Size() << " instances\n";
        if (!dataset.Instances().empty()) {
            const int expected_jobs = config.data.jobs;
            const int expected_machines = config.data.machines;
            const int lower_bound = config.data.duration_lb;
            const int upper_bound = config.data.duration_ub;

            std::size_t size_mismatch = 0;
            std::size_t sigma_invalid = 0;
            std::size_t proc_out_of_bounds = 0;

            for (const auto& entry : dataset.Instances()) {
                const auto& instance = entry.problem;
                if (expected_jobs > 0 && instance.NumJobs() != expected_jobs) {
                    size_mismatch++;
                }
                if (expected_machines > 0 && instance.NumMachines() != expected_machines) {
                    size_mismatch++;
                }

                for (int j = 0; j < instance.NumJobs(); ++j) {
                    if (!ValidateSigmaRow(instance, j)) {
                        sigma_invalid++;
                        break;
                    }
                }

                if (lower_bound > 0 && upper_bound > 0) {
                    for (int j = 0; j < instance.NumJobs(); ++j) {
                        for (int m = 0; m < instance.NumMachines(); ++m) {
                            int value = instance.Proc(j, m);
                            if (value < lower_bound || value > upper_bound) {
                                proc_out_of_bounds++;
                                j = instance.NumJobs();
                                break;
                            }
                        }
                    }
                }
            }

            const auto& first = dataset.Instances().front();
            std::cout << "First instance: " << first.given_name << "\n";
            std::cout << "Jobs: " << first.problem.NumJobs()
                      << ", Machines: " << first.problem.NumMachines() << "\n";
            std::cout << "Size mismatches: " << size_mismatch << "\n";
            std::cout << "Sigma permutation issues: " << sigma_invalid << "\n";
            std::cout << "Processing times out of bounds: " << proc_out_of_bounds << "\n";

            if (size_mismatch > 0 || sigma_invalid > 0 || proc_out_of_bounds > 0) {
                std::cerr << "Validation failed for one or more instances.\n";
                return 2;
            }

            std::cout << "Show matrices for first instance? (y/n): ";
            char answer = 'n';
            std::cin >> answer;
            if (answer == 'y' || answer == 'Y') {
                PrintMatrix("Sigma matrix", first.problem.NumJobs(), first.problem.NumMachines(),
                            [&](int j, int m) { return first.problem.Sigma(j, m); });
                PrintMatrix("Proc matrix", first.problem.NumJobs(), first.problem.NumMachines(),
                            [&](int j, int m) { return first.problem.Proc(j, m); });
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}



