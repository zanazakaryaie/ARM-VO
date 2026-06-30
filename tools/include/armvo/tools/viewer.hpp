#pragma once

#include <memory>
#include <armvo/Types.hpp>

namespace armvo::tools
{

/// @brief Window plotter for image frames and estimated trajectory
class Plotter
{
public:
    Plotter();

    /// @brief Adds a frame and its estimated pose
    /// @param frame Latest input frame
    /// @param pose Estimated camera pose
    void update(const cv::Mat& frame, const armvo::Pose& pose);

    /// @brief Adds a frame, its estimated pose, and its ground-truth pose
    /// @param frame Latest input frame
    /// @param pose Estimated camera pose
    /// @param gt Ground-truth camera pose
    void update(const cv::Mat& frame, const armvo::Pose& pose, const armvo::Pose& gt);

    /// @brief Plots the latest frame and camera trajectory
    /// @pre update() must have been called at least once
    void plot();

private:
    class Impl;
    std::shared_ptr<Impl> mImpl;
};

} // namespace armvo::tools
