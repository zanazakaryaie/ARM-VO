#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "KeypointTracker.hpp"
#include "KeypointDetector.hpp"
#include "Types.hpp"

namespace armvo
{

/// @brief Estimates a fundamental matrix using RANSAC
/// @param points1 Keypoints in the first frame
/// @param points2 Corresponding keypoints in the second frame
/// @param reprojectionError Maximum RANSAC reprojection error in pixels
/// @param mask Output 8-bit mask with non-zero values for inlier correspondences
/// @return Estimated 3x3 fundamental matrix
/// @throws Throws exception if estimation fails
cv::Matx33f estimateFundamentalMatrix(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2, float reprojectionError, cv::Mat& mask);

/// @brief Estimates a rough homography matrix using USAC
/// @param points1 Keypoints in the first frame
/// @param points2 Corresponding keypoints in the second frame
/// @return Estimated 3x3 homography matrix
/// @throws Throws exception if estimation fails
cv::Matx33f estimateHomographyMatrix(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2);

/// @brief Undistorts image points and maps them back into pixel coordinates
/// @param points Distorted image points
/// @param intrinsics Camera intrinsic matrix
/// @param distortions Distortion coefficients
/// @return Undistorted image points
std::vector<cv::Point2f> undistortPoints(const std::vector<cv::Point2f>& points, const cv::Matx33f& intrinsics, const cv::Mat& distortions);

/// @brief Geometric Robust Information Criterion model scorer
class GRIC
{
public:
    GRIC();

    /// @brief Computes the cost of modeling correspondences with a fundamental matrix
    /// @param points0 Keypoints in the first frame
    /// @param points1 Corresponding keypoints in the second frame
    /// @param fundamentalMatrix Fundamental matrix to score
    /// @param sigma Pixel-level noise standard deviation
    /// @return GRIC cost. Lower is better
    float calculateFcost(const std::vector<cv::Point2f>& points0,
                         const std::vector<cv::Point2f>& points1,
                         const cv::Matx33f& fundamentalMatrix,
                         float sigma);

    /// @brief Computes the cost of modeling correspondences with a homography matrix
    /// @param points0 Keypoints in the first frame
    /// @param points1 Corresponding keypoints in the second frame
    /// @param homographyMatrix Homography matrix to score
    /// @param sigma Pixel-level noise standard deviation
    /// @return GRIC cost. Lower is better
    float calculateHcost(const std::vector<cv::Point2f>& points0,
                         const std::vector<cv::Point2f>& points1,
                         const cv::Matx33f& homographyMatrix,
                         float sigma);
private:
    class Impl;
    std::shared_ptr<Impl> mImpl;
};

/// @brief Recovers relative camera motion from epipolar geometry
class MotionEstimator
{
public:
    /// @brief Constructor
    /// @param cameraMatrix Camera intrinsic matrix
    explicit MotionEstimator(const cv::Matx33f& cameraMatrix);

    /// @brief Estimates inter-frame rotation and translation direction
    /// @param fundamentalMatrix Estimated fundamental matrix
    /// @param points1 Keypoints detected in the first frame
    /// @param points2 Corresponding keypoints in the second frame
    /// @param mask Binary correspondence mask where non-zero values are inliers
    /// @return Estimated relative pose
    /// @note The translation vector is unit length; metric scale is resolved by ScaleEstimator.
    Pose estimate(const cv::Matx33f& fundamentalMatrix, const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2, const cv::Mat &mask);

private:
    class Impl;
    std::shared_ptr<Impl> mImpl;
};

/// @brief Estimates metric scale for monocular motion using road-plane homographies
class ScaleEstimator
{
public:
    /// @brief Constructor
    /// @param camConfig Camera configuration
    /// @param maxVehicleSpeed Maximum vehicle speed in km/h
    /// @param keyPointTracker Tracker used for forward/backward road-point tracking
    /// @note The @a keyPointTracker object must have previous and current frames set before estimate() is called
    /// @throws Throws exception if camera height or maximum speed is non-positive
    ScaleEstimator(const CameraConfig& camConfig, float maxVehicleSpeed, std::unique_ptr<KeypointTracker>& keyPointTracker);

    /// @brief Initializes road points from the previous frame
    /// @param points Detected keypoints in the previous frame
    /// @param roadMask Binary mask where road pixels are 255 and non-road pixels are 0
    void initialize(const std::vector<Keypoint>& points, const cv::Mat& roadMask);

    /// @brief Feeds current-frame road points to the scale estimator
    /// @param points Detected keypoints in the current frame
    /// @param roadMask Binary mask where road pixels are 255 and non-road pixels are 0
    void setCurrentFrameData(const std::vector<Keypoint>& points, const cv::Mat& roadMask);

    /// @brief Estimates metric scale for the latest motion estimate
    /// @pre initialize() and setCurrentFrameData() must have been called first
    /// @param rotation Rotation from the previous frame to the current frame
    /// @param translation Unit-length translation direction from the previous frame to the current frame
    /// @return Estimated scale in meters, or std::nullopt if scale cannot be estimated
    std::optional<float> estimate(const cv::Matx33f& rotation, const cv::Matx31f& translation);

    /// @brief Promotes current-frame road points to previous-frame road points
    /// @note This must be called before the next calls to setCurrentFrameData() and estimate()
    void step();

private:
    class Impl;
    std::shared_ptr<Impl> mImpl;
};

} // namespace armvo
