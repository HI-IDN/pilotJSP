#pragma once

#include <string>
#include <vector>

#include "alice/jsp/model/problem_instance.hpp"

namespace alice::jsp::data {

struct DatasetInstance {
    std::string name;
    std::string given_name;
    model::ProblemInstance problem;
};

class Dataset {
  public:
    void Add(DatasetInstance instance);
    const std::vector<DatasetInstance>& Instances() const;
    std::size_t Size() const;

  private:
    std::vector<DatasetInstance> instances_;
};

}  // namespace alice::jsp::data



