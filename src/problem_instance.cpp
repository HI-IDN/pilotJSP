#include "alice/jsp/model/problem_instance.hpp"

#include <stdexcept>

namespace alice::jsp::model {

ProblemInstance::ProblemInstance(int num_jobs, int num_machines, std::vector<int> sigma,
                                 std::vector<int> procs)
    : num_jobs_(num_jobs),
      num_machines_(num_machines),
      sigma_(std::move(sigma)),
      procs_(std::move(procs)),
      job_names_(num_jobs_),
      machine_names_(num_machines_) {
    if (num_jobs_ < 0 || num_machines_ < 0) {
        throw std::invalid_argument("Job and machine counts must be non-negative");
    }
    if (sigma_.size() != static_cast<std::size_t>(num_jobs_ * num_machines_)) {
        throw std::invalid_argument("Sigma size does not match dimensions");
    }
    if (procs_.size() != static_cast<std::size_t>(num_jobs_ * num_machines_)) {
        throw std::invalid_argument("Processing time size does not match dimensions");
    }
}

int ProblemInstance::NumJobs() const {
    return num_jobs_;
}

int ProblemInstance::NumMachines() const {
    return num_machines_;
}

int ProblemInstance::Sigma(int job, int mac) const {
    return sigma_.at(Index(job, mac));
}

int ProblemInstance::Proc(int job, int mac) const {
    return procs_.at(Index(job, mac));
}

void ProblemInstance::SetJobName(int job, std::string name) {
    if (job < 0 || job >= num_jobs_) {
        return;
    }
    job_names_[static_cast<std::size_t>(job)] = std::move(name);
}

void ProblemInstance::SetMachineName(int machine, std::string name) {
    if (machine < 0 || machine >= num_machines_) {
        return;
    }
    machine_names_[static_cast<std::size_t>(machine)] = std::move(name);
}

const std::vector<std::string>& ProblemInstance::JobNames() const {
    return job_names_;
}

const std::vector<std::string>& ProblemInstance::MachineNames() const {
    return machine_names_;
}

std::size_t ProblemInstance::Index(int job, int mac) const {
    if (job < 0 || job >= num_jobs_ || mac < 0 || mac >= num_machines_) {
        throw std::out_of_range("Job or machine index out of range");
    }
    return static_cast<std::size_t>(job * num_machines_ + mac);
}

}  // namespace alice::jsp::model



