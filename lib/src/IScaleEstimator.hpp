#pragma once

#include <vector>

#include <opencv2/core/mat.hpp>

#include "KeypointDetector.hpp"
#include "ScaleMeasurement.hpp"

namespace armvo
{

/// @brief Frame observations shared by scale-estimation sources
/// @note References need only remain valid for the duration of the receiving call.
struct ScaleEstimationInput
{
    const std::vector<Keypoint>& keypoints;
    /// CV_8UC1 class labels at input-image resolution
    const cv::Mat& segmentationMap;
};

/// @brief Produces fresh scale measurements without temporal filtering or fallback
class IScaleEstimator
{
public:
    virtual ~IScaleEstimator() = default;

    /// @brief Initializes observations for the previous frame
    virtual void initialize(const ScaleEstimationInput& frame) = 0;

    /// @brief Supplies observations for the current frame
    virtual void setCurrentFrameData(const ScaleEstimationInput& frame) = 0;

    /// @brief Estimates scale for a previous-to-current rotation and unit translation
    /// @return Fresh measurements, or an empty vector when geometry cannot be estimated.
    /// Multiple measurements allow sources such as forward/backward road tracking
    /// to be validated individually before fusion. No previous scale is reused here.
    virtual std::vector<ScaleMeasurement> estimate(const cv::Matx33f& rotation,
                                                  const cv::Matx31f& translation) = 0;

    /// @brief Promotes current observations to the previous frame after a successful update
    virtual void step() = 0;
};

} // namespace armvo
