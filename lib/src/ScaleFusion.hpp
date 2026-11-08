#pragma once

#include <optional>
#include <vector>

#include "ScaleMeasurement.hpp"

namespace armvo
{

/// @brief Combines already accepted scale observations without keeping temporal state
class ScaleFusion
{
public:
    /// @brief Computes the weighted geometric mean, preserving a single measurement as-is
    /// @pre Measurements supplied for logarithmic fusion have positive values and weights.
    /// @return Fused scale in meters, or std::nullopt when no measurements are supplied
    static std::optional<float> fuse(const std::vector<ScaleMeasurement>& measurements);
};

} // namespace armvo
