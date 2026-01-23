#pragma once

#include <string>
#include <vector>

namespace alice::jsp::model {

class ProblemInstance {
  public:
    ProblemInstance() = default;
    ProblemInstance(int num_jobs, int num_machines, std::vector<int> sigma, std::vector<int> procs);

    int NumJobs() const;
    int NumMachines() const;
    int Sigma(int job, int mac) const;
    int Proc(int job, int mac) const;

    void SetJobName(int job, std::string name);
    void SetMachineName(int machine, std::string name);
    const std::vector<std::string>& JobNames() const;
    const std::vector<std::string>& MachineNames() const;

  private:
    int num_jobs_ = 0;
    int num_machines_ = 0;
    std::vector<int> sigma_;
    std::vector<int> procs_;
    std::vector<std::string> job_names_;
    std::vector<std::string> machine_names_;

    std::size_t Index(int job, int mac) const;
};

}  // namespace alice::jsp::model



