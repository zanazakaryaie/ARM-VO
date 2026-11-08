#pragma once

namespace armvo
{

/// @brief A fresh metric-scale observation for the current frame pair
struct ScaleMeasurement
{
    /// Multiplier in meters for the unit-length translation direction
    float value;
    /// Relative weight in log-scale fusion; larger values have more influence.
    /// This is a score, not a calibrated probability or inverse scale variance.
    /// Sources combined by ScaleFusion must use comparable weights.
    float weight;
};

} // namespace armvo
