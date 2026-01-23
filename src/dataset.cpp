#include "alice/jsp/data/dataset.hpp"

namespace alice::jsp::data {

void Dataset::Add(DatasetInstance instance) {
    instances_.push_back(std::move(instance));
}

const std::vector<DatasetInstance>& Dataset::Instances() const {
    return instances_;
}

std::size_t Dataset::Size() const {
    return instances_.size();
}

}  // namespace alice::jsp::data



