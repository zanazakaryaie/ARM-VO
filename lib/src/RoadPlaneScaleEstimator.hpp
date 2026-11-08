#pragma once

#include <memory>

#include "IScaleEstimator.hpp"
#include "Types.hpp"

namespace armvo
{

class KeypointTracker;

/// @brief Estimates scale from camera height and forward/backward road homographies
/// @note Measurement weights are 1 / (1e-6 + mean squared Sampson residual).
class RoadPlaneScaleEstimator final : public IScaleEstimator
{
public:
    /// @param keypointTracker Non-owning tracker; must outlive this estimator and have
    /// previous/current frames set before estimate() is called.
    /// @throws std::invalid_argument if camera height is non-positive
    RoadPlaneScaleEstimator(const CameraConfig& cameraConfig, KeypointTracker& keypointTracker);
    ~RoadPlaneScaleEstimator() = default;

    void initialize(const ScaleEstimationInput& frame) override;
    void setCurrentFrameData(const ScaleEstimationInput& frame) override;
    std::vector<ScaleMeasurement> estimate(const cv::Matx33f& rotation,
                                          const cv::Matx31f& translation) override;
    void step() override;

private:
    class Impl;
    std::shared_ptr<Impl> mImpl;
};

} // namespace armvo
