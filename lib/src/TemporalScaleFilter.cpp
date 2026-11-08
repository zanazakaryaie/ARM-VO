#include "TemporalScaleFilter.hpp"

#include <stdexcept>

#include "ScaleFusion.hpp"

namespace armvo
{

TemporalScaleFilter::TemporalScaleFilter(float fps, float maxVehicleSpeed)
{
    if (maxVehicleSpeed <= 0)
    {
        throw std::invalid_argument("Max vehicle speed is expected to be larger than zero");
    }
    mMaxPossibleScale = (maxVehicleSpeed * 1000 / 3600) / fps;
}

std::optional<float> TemporalScaleFilter::update(const std::vector<ScaleMeasurement>& measurements)
{
    std::vector<ScaleMeasurement> accepted;
    accepted.reserve(measurements.size());
    for (const auto& measurement : measurements)
    {
        if (accepts(measurement.value))
        {
            accepted.push_back(measurement);
        }
    }

    const auto scale = ScaleFusion::fuse(accepted);
    if (scale.has_value())
    {
        mPreviousScale = scale;
    }
    return mPreviousScale;
}

bool TemporalScaleFilter::accepts(float scale) const
{
    constexpr float JUMP_UP_SCALE_RATIO = 2.5f;
    constexpr float JUMP_DOWN_SCALE_RATIO = 0.4f;

    if (scale > mMaxPossibleScale)
    {
        return false;
    }
    if (mPreviousScale.has_value())
    {
        const float prevScale = mPreviousScale.value();
        if (scale > JUMP_UP_SCALE_RATIO * prevScale || scale < JUMP_DOWN_SCALE_RATIO * prevScale)
        {
            return false;
        }
    }
    return true;
}

} // namespace armvo
