#pragma once

#include "Types.hpp"

namespace armvo
{

/// @brief Semantic segmentation interface
class ISemanticSegmentor
{
public:
    /// @brief Runs segmentation on an input frame
    /// @param img Input frame
    /// @param pixelformat Pixel format of @a img
    /// @return Segmentation map of type CV_8UC1 where each pixel stores a class label
    virtual cv::Mat segment(const cv::Mat& img, PixelFormat pixelformat) = 0;

    /// @brief Returns whether segmentation uses a non-CPU co-processor, such as a GPU
    virtual bool runsOnCoProcessor() = 0;

    /// @brief Builds a road mask from a segmentation map
    /// @param segmentationMap Segmentation output from segment()
    /// @return Binary mask where road pixels are 255 and non-road pixels are 0
    /// @note This is used for scale estimation
    virtual cv::Mat getRoadMask(const cv::Mat& segmentationMap) = 0;

    /// @brief Builds a static-scene mask from a segmentation map
    /// @param segmentationMap Segmentation output from segment()
    /// @return Binary mask where static pixels are 255 and dynamic pixels are 0
    /// @note This is used to favor static keypoints over dynamic ones for motion estimation
    virtual cv::Mat getStaticMask(const cv::Mat& segmentationMap) = 0;

    virtual ~ISemanticSegmentor() = default;
};

} // namespace armvo
