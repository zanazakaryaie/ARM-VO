#pragma once

#include <memory>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "Types.hpp"

namespace armvo
{

/// @brief Tracks 2D keypoints between consecutive grayscale frames using pyramidal Lucas-Kanade optical flow
class KeypointTracker
{
public:
    /// @brief Constructor
    /// @param config Tracker configuration
    /// @throws Throws exception if the window size is smaller than 3 or evens
    explicit KeypointTracker(const KeypointTrackerConfig& config);

    /// @brief Sets the previous frame
    /// @param image Grayscale input image
    void setPreviousFrame(const cv::Mat& image);

    /// @brief Sets the current frame
    /// @param image Grayscale input image with the same size as the previous frame
    void setCurrentFrame(const cv::Mat& image);

    /// @brief Tracks keypoints from the previous frame to the current frame
    /// @param points Keypoints in the previous frame. Untracked or out-of-bounds points are removed in place
    /// @return Corresponding keypoints in the current frame
    /// @pre setPreviousFrame() and setCurrentFrame() must have been called before this function
    std::vector<cv::Point2f> trackForward(std::vector<cv::Point2f>& points);

    /// @brief Tracks keypoints from the current frame back to the previous frame
    /// @param points Keypoints in the current frame. Untracked or out-of-bounds points are removed in place
    /// @return Corresponding keypoints in the previous frame
    /// @pre setPreviousFrame() and setCurrentFrame() must have been called before this function
    std::vector<cv::Point2f> trackBackward(std::vector<cv::Point2f>& points);

    /// @brief Promotes the current frame to the previous frame
    /// @note This function must be called before the next call to setCurrentFrame()
    void step();

private:
    class Impl;
    std::shared_ptr<Impl> mImpl;
};

} // namespace armvo
