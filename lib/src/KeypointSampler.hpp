#pragma once

#include <vector>
#include <cstdint>
#include <memory>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "detector.hpp"

namespace armvo
{

/// @brief Samples detector keypoints by semantic mask, image grid, and spatial diversity
class KeypointSampler
{
public:

    KeypointSampler();

    /// @brief Creates a grid of shape @a numRows by @a numCols
    /// @param height Grid height in pixels
    /// @param width Grid width in pixels
    /// @param numRows Number of grid rows
    /// @param numCols Number of grid columns
    /// @return Non-overlapping ROIs that cover the image, or an empty vector if the grid is denser than the image
    static std::vector<cv::Rect> createROIs(uint32_t height, uint32_t width, uint32_t numRows, uint32_t numCols);

    /// @brief Samples points using a binary mask
    /// @details Points that satisfy the mask are ranked by response before farthest-point sampling is applied
    /// @param keypoints Detected keypoints from KeypointDetector
    /// @param mask Binary mask where accepted pixels are 255 and rejected pixels are 0
    /// @param numPoints Desired number of sampled points
    /// @return Sampled point coordinates
    /// @note The function returns fewer than @a numPoints when too few keypoints fall inside positive mask regions
    std::vector<cv::Point2f> run(const std::vector<Keypoint>& keypoints, const cv::Mat& mask, uint32_t numPoints);

    /// @brief Samples keypoints uniformly across image ROIs
    /// @details In each ROI, points with higher response scores are kept. Static points get a score boost while dynamic points are penalized
    /// @param keypoints Detected keypoints from KeypointDetector
    /// @param ROIs Non-overlapping image cells, created with KeypointSampler::createROIs()
    /// @param staticMask Binary mask where static pixels are 255 and dynamic pixels are 0
    /// @param numPoints Desired number of sampled points
    /// @return Sampled point coordinates
    /// @note The function can return fewer than @a numPoints when @a numPoints is not divisible by the number of ROIs
    std::vector<cv::Point2f> run(const std::vector<Keypoint>& keypoints, const std::vector<cv::Rect>& ROIs, const cv::Mat& staticMask, uint32_t numPoints);

private:
    class Impl;
    std::shared_ptr<Impl> mImpl;
};

} // namespace armvo
