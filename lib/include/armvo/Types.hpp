#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <opencv2/core/mat.hpp>

namespace armvo
{

/// @brief Pixel format of input frames
enum class PixelFormat : uint8_t
{
    /// Single-channel 8-bit grayscale image
    Gray,
    /// Three-channel 8-bit image in BGR order
    BGR,
    /// Three-channel 8-bit image in RGB order
    RGB
};

/// @brief Camera calibration and input image metadata
struct CameraConfig
{
    /// Camera intrinsic matrix with fx, fy, cx, and cy
    cv::Matx33f intrinsics;
    PixelFormat pixelFormat = PixelFormat::Gray;
    float fps = 10;
    /// Camera height above the ground in meters
    float height = 1.7;
    /// Distortion coefficients
    std::optional<cv::Mat> distortions;
};

/// @brief Keypoint detection and image-grid sampling parameters
struct KeypointDetectorConfig
{
    /// Maximum number of sampled keypoints retained per frame
    uint16_t maxNumberOfPoints = 500;
    /// Corner response threshold. Must be in the open interval (0, 255)
    uint8_t responseThreshold = 5;
    /// Number of rows in the image grid used to distribute keypoints
    uint8_t numberOfImageGridRows = 6;
    /// Number of columns in the image grid used to distribute keypoints
    uint8_t numberOfImageGridCols = 12;
};

/// @brief Lucas-Kanade optical-flow tracker parameters
struct KeypointTrackerConfig
{
    /// Odd tracking-window size in pixels. Must be at least 3
    uint8_t windowSize = 15;
};

/// @brief Complete ARM-VO runtime configuration
struct ArmVoConfig
{
    CameraConfig camera;
    KeypointDetectorConfig keypointDetector;
    KeypointTrackerConfig keypointTracker;
    /// Maximum expected vehicle speed in km/h, used to reject impossible scale estimates
    float maxVehicleSpeed = 90;

    /// @brief Loads ARM-VO configuration from a YAML or JSON file
    /// @param filepath Path to a configuration file
    /// @return Parsed configuration
    /// @throws Throws exceptions if the file cannot be opened, or if required nodes are invalid
    static ArmVoConfig load(const std::string& filepath);
};

/// @brief Camera pose in world coordinates
struct Pose
{
    cv::Matx33f rotation;
    cv::Matx31f translation;
};

enum class Status : uint8_t
{
    /// Operation completed successfully
    SUCCESS = 0,
    /// Input frame is empty, has the wrong type, or changes resolution after initialization
    INVALID_FRAME,
    /// ArmVo::initialize() has not been called, or didn't complete successfully
    NOT_INITIALIZED,
    /// Too few keypoints were detected or retained
    NOT_ENOUGH_KEYPOINTS,
    /// Too few previously tracked keypoints remain
    TRACK_LOST,
    /// Frame was intentionally skipped because motion was too small to estimate reliably
    FRAME_SKIPPED,
    /// Scale estimation failed for the current frame
    SCALE_ESTIMATION_FAILURE,
};

} // namespace armvo
