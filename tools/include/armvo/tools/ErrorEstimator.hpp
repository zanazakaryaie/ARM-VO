#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <armvo/Types.hpp>

namespace armvo::tools
{

/// @brief Normalized KITTI odometry segment error
struct Error
{
    /// Index of the segment's first frame
    int32_t startFrameIndex;
    /// Rotation error in radians per meter
    float rotationErr;
    /// Translation error normalized by segment length
    float translationErr;
    /// Segment length in meters
    float length;

    /// @brief Creates a segment error record
    /// @param idx Start-frame index
    /// @param rotErr Rotation error in radians per meter
    /// @param trErr Translation error normalized by segment length
    /// @param len Segment length in meters
    Error(int32_t idx, float rotErr, float trErr, float len);
};

/// @brief Computes KITTI odometry-style trajectory errors
class ErrorEstimator
{
public:
    ErrorEstimator();

    /// @brief Estimates rotational and translation errors
    /// @details This is based on the KITTI odometry devkit metric
    /// @param estimatedPoses Estimated camera trajectory
    /// @param groundTruthPoses Ground-truth camera trajectory
    /// @return Segment errors for all valid KITTI segment windows
    /// @throws Throws exception if the trajectory lengths differ
    std::vector<Error> estimate(const std::vector<armvo::Pose>& estimatedPoses, const std::vector<armvo::Pose>& groundTruthPoses);

private:
    class Impl;
    std::shared_ptr<Impl> mImpl;
};

} // namespace armvo::tools
