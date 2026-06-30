#include <catch2/catch.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "Geometry.hpp"

namespace
{

constexpr float kTightEps = 1e-4f;
constexpr float kEps = 1e-3f;

cv::Matx33f makeIntrinsics()
{
    return cv::Matx33f(700.0f, 0.0f, 320.0f,
                       0.0f, 710.0f, 240.0f,
                       0.0f, 0.0f, 1.0f);
}

cv::Matx33f rodrigues(const cv::Vec3f& rotationVector)
{
    cv::Matx33f rotation;
    cv::Rodrigues(rotationVector, rotation);
    return rotation;
}

cv::Matx33f skew(const cv::Vec3f& translation)
{
    return cv::Matx33f(0.0f, -translation[2], translation[1],
                       translation[2], 0.0f, -translation[0],
                       -translation[1], translation[0], 0.0f);
}

std::vector<cv::Point3f> makeNonPlanarScenePoints()
{
    std::vector<cv::Point3f> points;
    points.reserve(108);

    for (int y = 0; y < 9; y++)
    {
        for (int x = 0; x < 12; x++)
        {
            const float worldX = (static_cast<float>(x) - 5.5f) * 0.35f;
            const float worldY = (static_cast<float>(y) - 4.0f) * 0.25f;
            const float worldZ = 4.0f + 0.08f * static_cast<float>(x) +
                                 0.11f * static_cast<float>(y) +
                                 0.07f * static_cast<float>((x + y) % 3);
            points.emplace_back(worldX, worldY, worldZ);
        }
    }

    return points;
}

cv::Point2f projectPoint(const cv::Matx33f& intrinsics,
                         const cv::Point3f& point,
                         const cv::Matx33f& rotation = cv::Matx33f::eye(),
                         const cv::Vec3f& translation = cv::Vec3f(0.0f, 0.0f, 0.0f))
{
    const cv::Vec3f cameraPoint = rotation * cv::Vec3f(point.x, point.y, point.z) + translation;
    const cv::Vec3f pixel = intrinsics * cameraPoint;
    return cv::Point2f(pixel[0] / pixel[2], pixel[1] / pixel[2]);
}

std::vector<cv::Point2f> projectPoints(const cv::Matx33f& intrinsics,
                                       const std::vector<cv::Point3f>& points,
                                       const cv::Matx33f& rotation = cv::Matx33f::eye(),
                                       const cv::Vec3f& translation = cv::Vec3f(0.0f, 0.0f, 0.0f))
{
    std::vector<cv::Point2f> projected;
    projected.reserve(points.size());

    for (const auto& point : points)
    {
        projected.push_back(projectPoint(intrinsics, point, rotation, translation));
    }

    return projected;
}

cv::Matx33f makeFundamentalMatrix(const cv::Matx33f& intrinsics,
                                  const cv::Matx33f& rotation,
                                  const cv::Vec3f& translation)
{
    return intrinsics.inv().t() * skew(translation) * rotation * intrinsics.inv();
}

float dotMat(const cv::Matx33f& lhs, const cv::Matx33f& rhs)
{
    float sum = 0.0f;
    for (uint8_t i = 0; i < 9; i++)
    {
        sum += lhs.val[i] * rhs.val[i];
    }
    return sum;
}

bool sameProjectiveMatrix(cv::Matx33f lhs, cv::Matx33f rhs, float tolerance)
{
    lhs /= cv::norm(lhs);
    rhs /= cv::norm(rhs);

    if (dotMat(lhs, rhs) < 0.0f)
    {
        rhs *= -1.0f;
    }

    return cv::norm(lhs - rhs) < tolerance;
}

cv::Point2f applyHomography(const cv::Matx33f& homography, const cv::Point2f& point)
{
    const cv::Vec3f warped = homography * cv::Vec3f(point.x, point.y, 1.0f);
    return cv::Point2f(warped[0] / warped[2], warped[1] / warped[2]);
}

std::vector<cv::Point2f> makeImageGridPoints()
{
    std::vector<cv::Point2f> points;
    points.reserve(48);

    for (int y = 0; y < 6; y++)
    {
        for (int x = 0; x < 8; x++)
        {
            points.emplace_back(60.0f + 30.0f * static_cast<float>(x),
                                50.0f + 25.0f * static_cast<float>(y));
        }
    }

    return points;
}

float maxHomographyReprojectionError(const cv::Matx33f& homography,
                                     const std::vector<cv::Point2f>& source,
                                     const std::vector<cv::Point2f>& expected)
{
    float maxError = 0.0f;
    for (size_t i = 0; i < source.size(); i++)
    {
        const float error = static_cast<float>(cv::norm(applyHomography(homography, source[i]) - expected[i]));
        maxError = std::max(maxError, error);
    }
    return maxError;
}

float maxPointDistance(const std::vector<cv::Point2f>& points0, const std::vector<cv::Point2f>& points1)
{
    REQUIRE(points0.size() == points1.size());

    float maxDistance = 0.0f;
    for (size_t i = 0; i < points0.size(); i++)
    {
        const float distance = static_cast<float>(cv::norm(points0[i] - points1[i]));
        maxDistance = std::max(maxDistance, distance);
    }
    return maxDistance;
}

float dotVec(const cv::Matx31f& lhs, const cv::Matx31f& rhs)
{
    return lhs(0) * rhs(0) + lhs(1) * rhs(1) + lhs(2) * rhs(2);
}

std::unique_ptr<armvo::KeypointTracker> makeKeypointTracker()
{
    armvo::KeypointTrackerConfig trackerConfig;
    trackerConfig.windowSize = 15;
    return std::make_unique<armvo::KeypointTracker>(trackerConfig);
}

armvo::CameraConfig makeCameraConfig()
{
    armvo::CameraConfig config;
    config.intrinsics = makeIntrinsics();
    config.fps = 10.0f;
    config.height = 1.6f;
    return config;
}

cv::Matx33f outerProduct(const cv::Matx31f& lhs, const cv::Matx31f& rhs)
{
    cv::Matx33f output;
    for (int r = 0; r < 3; r++)
    {
        for (int c = 0; c < 3; c++)
        {
            output(r, c) = lhs(r) * rhs(c);
        }
    }
    return output;
}

cv::Matx33f makePlanarPixelHomography(const armvo::CameraConfig& cameraConfig,
                                      const cv::Matx33f& rotation,
                                      const cv::Matx31f& unitTranslation,
                                      const cv::Matx31f& planeNormal,
                                      float scale)
{
    const cv::Matx33f normalizedHomography =
        rotation - (scale / cameraConfig.height) * outerProduct(unitTranslation, planeNormal);
    return cameraConfig.intrinsics * normalizedHomography * cameraConfig.intrinsics.inv();
}

std::vector<cv::Point2f> makeRoadFeaturePoints()
{
    std::vector<cv::Point2f> points;
    points.reserve(48);

    for (int y = 0; y < 6; y++)
    {
        for (int x = 0; x < 8; x++)
        {
            points.emplace_back(110.0f + 55.0f * static_cast<float>(x),
                                120.0f + 42.0f * static_cast<float>(y));
        }
    }

    return points;
}

cv::Mat renderRoadTexture(const cv::Size& size, const std::vector<cv::Point2f>& points)
{
    cv::Mat image(size, CV_8UC1, cv::Scalar(0));

    for (size_t i = 0; i < points.size(); i++)
    {
        const cv::Point center(cvRound(points[i].x), cvRound(points[i].y));
        const int intensity = 150 + static_cast<int>((i * 37) % 90);
        cv::circle(image, center, 5, cv::Scalar(intensity), cv::FILLED);
        cv::line(image, center + cv::Point(-7, 2), center + cv::Point(8, 2), cv::Scalar(255), 1);
        cv::line(image, center + cv::Point(2, -7), center + cv::Point(2, 8), cv::Scalar(220), 1);
    }

    return image;
}

std::vector<cv::Point2f> warpPoints(const cv::Matx33f& homography, const std::vector<cv::Point2f>& points)
{
    std::vector<cv::Point2f> warped;
    warped.reserve(points.size());

    for (const auto& point : points)
    {
        warped.push_back(applyHomography(homography, point));
    }

    return warped;
}

std::vector<armvo::Keypoint> makeKeypoints(const std::vector<cv::Point2f>& points)
{
    std::vector<armvo::Keypoint> keypoints;
    keypoints.reserve(points.size());

    for (size_t i = 0; i < points.size(); i++)
    {
        keypoints.emplace_back(points[i].x, points[i].y, 100.0f + static_cast<float>(i));
    }

    return keypoints;
}

} // namespace

TEST_CASE("estimateFundamentalMatrix recovers epipolar geometry for a non-planar scene")
{
    const cv::Matx33f intrinsics = makeIntrinsics();
    const cv::Matx33f rotation = rodrigues(cv::Vec3f(0.02f, -0.03f, 0.05f));
    const cv::Vec3f translation(0.45f, -0.02f, 0.12f);

    const auto scenePoints = makeNonPlanarScenePoints();
    const auto points0 = projectPoints(intrinsics, scenePoints);
    const auto points1 = projectPoints(intrinsics, scenePoints, rotation, translation);

    cv::Mat mask;
    const cv::Matx33f estimated = armvo::estimateFundamentalMatrix(points0, points1, 0.5f, mask);
    const cv::Matx33f expected = makeFundamentalMatrix(intrinsics, rotation, translation);

    CHECK(mask.rows == static_cast<int>(points0.size()));
    CHECK(cv::countNonZero(mask) == static_cast<int>(points0.size()));
    CHECK(sameProjectiveMatrix(estimated, expected, 5e-3f));
}

TEST_CASE("estimateFundamentalMatrix throws when correspondence counts differ")
{
    const std::vector<cv::Point2f> points0 =
    {
        {10.0f, 10.0f}, {20.0f, 12.0f}, {30.0f, 14.0f}, {40.0f, 18.0f},
        {50.0f, 22.0f}, {60.0f, 26.0f}, {70.0f, 30.0f}, {80.0f, 34.0f}
    };
    const std::vector<cv::Point2f> points1 =
    {
        {10.0f, 10.0f}, {20.0f, 12.0f}, {30.0f, 14.0f}, {40.0f, 18.0f},
        {50.0f, 22.0f}, {60.0f, 26.0f}, {70.0f, 30.0f}
    };

    cv::Mat mask;
    CHECK_THROWS(armvo::estimateFundamentalMatrix(points0, points1, 1.0f, mask));
}

TEST_CASE("estimateHomographyMatrix recovers a projective image transform")
{
    const cv::Matx33f expected(1.02f, -0.08f, 24.0f,
                               0.05f, 0.98f, -12.0f,
                               0.00015f, -0.00009f, 1.0f);

    const auto points0 = makeImageGridPoints();
    std::vector<cv::Point2f> points1;
    points1.reserve(points0.size());

    for (const auto& point : points0)
    {
        points1.push_back(applyHomography(expected, point));
    }

    const cv::Matx33f estimated = armvo::estimateHomographyMatrix(points0, points1);

    CHECK(maxHomographyReprojectionError(estimated, points0, points1) < 1e-2f);
}

TEST_CASE("estimateHomographyMatrix throws when there are too few correspondences")
{
    const std::vector<cv::Point2f> points0 = {{0.0f, 0.0f}, {10.0f, 0.0f}, {0.0f, 10.0f}};
    const auto points1 = points0;

    CHECK_THROWS(armvo::estimateHomographyMatrix(points0, points1));
}

TEST_CASE("undistortPoints matches OpenCV's camera-space correction")
{
    const cv::Matx33f intrinsics = makeIntrinsics();
    const cv::Mat distortions = (cv::Mat_<double>(1, 5) << 0.08, -0.03, 0.001, -0.002, 0.01);
    const std::vector<cv::Point2f> points =
    {
        {120.0f, 90.0f},
        {320.0f, 240.0f},
        {520.0f, 380.0f},
        {610.0f, 130.0f}
    };

    std::vector<cv::Point2f> expected;
    cv::undistortPoints(points, expected, intrinsics, distortions, cv::noArray(), intrinsics);

    const auto actual = armvo::undistortPoints(points, intrinsics, distortions);

    CHECK(maxPointDistance(actual, expected) < kTightEps);
}

TEST_CASE("GRIC assigns lower cost to the correct model than to a wrong model")
{
    armvo::GRIC gric;

    SECTION("fundamental matrix")
    {
        const cv::Matx33f intrinsics = makeIntrinsics();
        const cv::Matx33f rotation = rodrigues(cv::Vec3f(0.01f, 0.02f, -0.04f));
        const cv::Vec3f translation(0.3f, 0.04f, 0.1f);

        const auto scenePoints = makeNonPlanarScenePoints();
        const auto points0 = projectPoints(intrinsics, scenePoints);
        const auto points1 = projectPoints(intrinsics, scenePoints, rotation, translation);

        const cv::Matx33f correctFundamental = makeFundamentalMatrix(intrinsics, rotation, translation);
        const cv::Matx33f wrongFundamental = cv::Matx33f::eye();

        CHECK(gric.calculateFcost(points0, points1, correctFundamental, 1.0f) <
              gric.calculateFcost(points0, points1, wrongFundamental, 1.0f));
    }

    SECTION("homography")
    {
        const cv::Matx33f correctHomography(1.02f, -0.08f, 24.0f,
                                            0.05f, 0.98f, -12.0f,
                                            0.00015f, -0.00009f, 1.0f);
        const auto points0 = makeImageGridPoints();

        std::vector<cv::Point2f> points1;
        points1.reserve(points0.size());
        for (const auto& point : points0)
        {
            points1.push_back(applyHomography(correctHomography, point));
        }

        CHECK(gric.calculateHcost(points0, points1, correctHomography, 1.0f) <
              gric.calculateHcost(points0, points1, cv::Matx33f::eye(), 1.0f));
    }

    SECTION("homography is preferred for planar correspondences")
    {
        const armvo::CameraConfig cameraConfig = makeCameraConfig();
        const cv::Matx33f rotation = rodrigues(cv::Vec3f(0.0f, 0.01f, 0.0f));
        const cv::Matx31f unitTranslation(1.0f, 0.0f, 0.0f);
        const cv::Matx31f planeNormal(0.0f, 1.0f, 0.0f);
        const float scale = 0.2f;

        const cv::Matx33f correctHomography =
            makePlanarPixelHomography(cameraConfig, rotation, unitTranslation, planeNormal, scale);
        const cv::Vec3f translation(scale * unitTranslation(0),
                                    scale * unitTranslation(1),
                                    scale * unitTranslation(2));
        const cv::Matx33f fundamental =
            makeFundamentalMatrix(cameraConfig.intrinsics, rotation, translation);

        const auto points0 = makeRoadFeaturePoints();
        const auto points1 = warpPoints(correctHomography, points0);

        const float hCost = gric.calculateHcost(points0, points1, correctHomography, 1.0f);
        const float fCost = gric.calculateFcost(points0, points1, fundamental, 1.0f);

        CHECK(hCost < fCost);
    }

    SECTION("fundamental matrix is preferred for non-planar correspondences")
    {
        const cv::Matx33f intrinsics = makeIntrinsics();
        const cv::Matx33f rotation = rodrigues(cv::Vec3f(0.02f, -0.04f, 0.03f));
        const cv::Vec3f translation(1.6f, -0.08f, 0.35f);

        auto scenePoints = makeNonPlanarScenePoints();
        for (size_t i = 0; i < scenePoints.size(); i++)
        {
            scenePoints[i].z += (i % 2 == 0) ? -1.0f : 1.5f;
        }
        const auto points0 = projectPoints(intrinsics, scenePoints);
        const auto points1 = projectPoints(intrinsics, scenePoints, rotation, translation);

        const cv::Matx33f correctFundamental = makeFundamentalMatrix(intrinsics, rotation, translation);
        const cv::Matx33f fittedHomography = armvo::estimateHomographyMatrix(points0, points1);

        const float fCost = gric.calculateFcost(points0, points1, correctFundamental, 1.0f);
        const float hCost = gric.calculateHcost(points0, points1, fittedHomography, 1.0f);

        CHECK(fCost < hCost);
    }
}

TEST_CASE("MotionEstimator recovers rotation and unit translation direction from known epipolar geometry")
{
    const cv::Matx33f intrinsics = makeIntrinsics();
    const cv::Matx33f rotation = rodrigues(cv::Vec3f(0.015f, -0.025f, 0.04f));
    const cv::Vec3f translation(0.35f, -0.03f, 0.08f);

    const auto scenePoints = makeNonPlanarScenePoints();
    const auto points0 = projectPoints(intrinsics, scenePoints);
    const auto points1 = projectPoints(intrinsics, scenePoints, rotation, translation);
    const cv::Mat mask(static_cast<int>(points0.size()), 1, CV_8U, cv::Scalar(255));

    const cv::Matx33f fundamental = makeFundamentalMatrix(intrinsics, rotation, translation);
    armvo::MotionEstimator estimator(intrinsics);

    const armvo::Pose pose = estimator.estimate(fundamental, points0, points1, mask);

    cv::Matx31f expectedTranslation(translation[0], translation[1], translation[2]);
    expectedTranslation /= cv::norm(expectedTranslation);

    CHECK(cv::norm(pose.rotation - rotation) < kEps);
    CHECK(std::abs(cv::norm(pose.translation) - 1.0f) < kTightEps);
    CHECK(dotVec(pose.translation, expectedTranslation) > 0.99f);
}

TEST_CASE("ScaleEstimator validates constructor inputs")
{
    SECTION("valid configuration")
    {
        auto tracker = makeKeypointTracker();
        const armvo::CameraConfig cameraConfig = makeCameraConfig();

        CHECK_NOTHROW(armvo::ScaleEstimator(cameraConfig, 90.0f, tracker));
    }

    SECTION("invalid camera height")
    {
        auto tracker = makeKeypointTracker();
        armvo::CameraConfig cameraConfig = makeCameraConfig();
        cameraConfig.height = 0.0f;

        CHECK_THROWS(armvo::ScaleEstimator(cameraConfig, 90.0f, tracker));
    }

    SECTION("invalid max vehicle speed")
    {
        auto tracker = makeKeypointTracker();
        const armvo::CameraConfig cameraConfig = makeCameraConfig();

        CHECK_THROWS(armvo::ScaleEstimator(cameraConfig, 0.0f, tracker));
    }
}

TEST_CASE("ScaleEstimator estimates scale from tracked planar road points")
{
    auto tracker = makeKeypointTracker();
    const armvo::CameraConfig cameraConfig = makeCameraConfig();

    const cv::Matx33f rotation = cv::Matx33f::eye();
    const cv::Matx31f unitTranslation(1.0f, 0.0f, 0.0f);
    const cv::Matx31f roadPlaneNormal(0.0f, 1.0f, 0.0f);
    const float expectedScale = 0.2f;

    const cv::Size imageSize(640, 480);
    const cv::Matx33f homography = makePlanarPixelHomography(
        cameraConfig, rotation, unitTranslation, roadPlaneNormal, expectedScale);

    const auto previousPoints = makeRoadFeaturePoints();
    const auto currentPoints = warpPoints(homography, previousPoints);
    const cv::Mat previousFrame = renderRoadTexture(imageSize, previousPoints);

    cv::Mat currentFrame;
    cv::warpPerspective(previousFrame, currentFrame, cv::Mat(homography), imageSize);

    const cv::Mat roadMask(imageSize, CV_8UC1, cv::Scalar(255));

    tracker->setPreviousFrame(previousFrame);
    tracker->setCurrentFrame(currentFrame);

    armvo::ScaleEstimator estimator(cameraConfig, 90.0f, tracker);
    estimator.initialize(makeKeypoints(previousPoints), roadMask);
    estimator.setCurrentFrameData(makeKeypoints(currentPoints), roadMask);

    const std::optional<float> scale = estimator.estimate(rotation, unitTranslation);

    REQUIRE(scale.has_value());
    CHECK(scale.value() == Approx(expectedScale).margin(0.05f));
}

TEST_CASE("ScaleEstimator returns no scale when no road points are available")
{
    auto tracker = makeKeypointTracker();
    const armvo::CameraConfig cameraConfig = makeCameraConfig();

    const cv::Size imageSize(320, 240);
    const cv::Mat previousFrame(imageSize, CV_8UC1, cv::Scalar(0));
    const cv::Mat currentFrame(imageSize, CV_8UC1, cv::Scalar(0));
    const cv::Mat emptyRoadMask(imageSize, CV_8UC1, cv::Scalar(0));

    tracker->setPreviousFrame(previousFrame);
    tracker->setCurrentFrame(currentFrame);

    armvo::ScaleEstimator estimator(cameraConfig, 90.0f, tracker);
    estimator.initialize(makeKeypoints(makeRoadFeaturePoints()), emptyRoadMask);
    estimator.setCurrentFrameData(makeKeypoints(makeRoadFeaturePoints()), emptyRoadMask);

    const std::optional<float> scale = estimator.estimate(cv::Matx33f::eye(), cv::Matx31f(1.0f, 0.0f, 0.0f));

    CHECK_FALSE(scale.has_value());
}
