#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "ScaleMeasurement.hpp"

namespace armvo
{

/// @brief Applies motion limits and maintains the last accepted fused scale
class TemporalScaleFilter
{
public:
    /// @param fps Camera frame rate, used with maximum speed to bound scale
    /// @param maxVehicleSpeed Maximum vehicle speed in km/h
    /// @throws std::invalid_argument if maximum speed is non-positive
    TemporalScaleFilter(float fps, float maxVehicleSpeed);

    /// @brief Rejects each measurement against speed/jump limits before fusing survivors
    /// @return Fresh fused scale, or the previous scale if no measurement survives.
    /// Returns std::nullopt until the first accepted measurement. The previous scale
    /// is retained indefinitely, preserving the existing scale-estimation behavior.
    std::optional<float> update(const std::vector<ScaleMeasurement>& measurements);

private:
    class Impl;
    std::shared_ptr<Impl> mImpl;
};

} // namespace armvo
