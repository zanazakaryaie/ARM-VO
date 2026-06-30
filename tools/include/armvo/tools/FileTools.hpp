#pragma once

#include <vector>
#include <string>
#include <opencv2/core/mat.hpp>

#include <armvo/Types.hpp>

namespace armvo::tools
{

/// @brief File and image-loading helpers
class FileTools
{
public:
    /// @brief Finds image frames in a directory and sorts them by filename in ascending order
    /// @param dir Directory containing .png, .jpg, or .jpeg files
    /// @return Sorted file paths
    /// @throws Throws exception if @a dir does not exist or is not a directory
    static std::vector<std::string> listFramesInDir(const std::string& dir);

    /// @brief Loads an image in the requested pixel format
    /// @param filepath Path to the image file
    /// @param desiredPixelFormat Desired output pixel format
    /// @return Loaded image
    /// @throws Throws exception if the image cannot be loaded or @a desiredPixelFormat is unsupported
    static cv::Mat loadImage(const std::string& filepath, PixelFormat desiredPixelFormat);

    /// @brief Loads ground-truth poses from a KITTI odometry text file
    /// @param filepath Path to a text file with one 3x4 pose matrix per line
    /// @return Parsed poses
    /// @throws Throws exception if the path does not exist, is not a file, or cannot be opened
    static std::vector<armvo::Pose> loadKittiGroundTruthPoses(const std::string& filepath);
};

} // namespace armvo::tools
