#pragma once

#include <memory>
#include <vector>

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

/// @brief Estimates a homography and returns its inlier mask
/// @param accurate Uses the existing MAGSAC fit when true, or the fast fit otherwise
cv::Matx33f estimateHomographyMatrixWithInliers(const std::vector<cv::Point2f>& points1,
                                                const std::vector<cv::Point2f>& points2,
                                                std::vector<uchar>& mask, bool accurate);

/// @brief Computes squared Sampson residuals for a homography and corresponding points
std::vector<float> sampsonDistanceSquaredH(const cv::Matx33f& homographyMatrix,
                                           const std::vector<cv::Point2f>& points0,
                                           const std::vector<cv::Point2f>& points1);

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
    /// @note The translation vector is unit length; metric scale is resolved separately from the recovered motion.
    Pose estimate(const cv::Matx33f& fundamentalMatrix, const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2, const cv::Mat &mask);

private:
    class Impl;
    std::shared_ptr<Impl> mImpl;
};

} // namespace armvo
