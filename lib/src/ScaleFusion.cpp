#include "ScaleFusion.hpp"

#include <cmath>

namespace armvo
{

std::optional<float> ScaleFusion::fuse(const std::vector<ScaleMeasurement>& measurements)
{
    if (measurements.empty())
    {
        return std::nullopt;
    }
    if (measurements.size() == 1)
    {
        return measurements.front().value;
    }

    float weightedLogScale = 0.0f;
    float totalWeight = 0.0f;
    for (const auto& measurement : measurements)
    {
        weightedLogScale += measurement.weight * std::log(measurement.value);
        totalWeight += measurement.weight;
    }
    return std::exp(weightedLogScale / totalWeight);
}

} // namespace armvo
