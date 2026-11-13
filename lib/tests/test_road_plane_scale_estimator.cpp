#include <catch2/catch.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <memory>
#include <vector>

#include "KeypointTracker.hpp"
#include "RoadPlaneScaleEstimator.hpp"
#include "TemporalScaleFilter.hpp"

namespace
{

cv::Matx33f makeIntrinsics()
{
    return cv::Matx33f(700.0f, 0.0f, 320.0f,
                       0.0f, 710.0f, 240.0f,
                       0.0f, 0.0f, 1.0f);
}

cv::Point2f applyHomography(const cv::Matx33f& homography, const cv::Point2f& point)
{
    const cv::Vec3f warped = homography * cv::Vec3f(point.x, point.y, 1.0f);
    return cv::Point2f(warped[0] / warped[2], warped[1] / warped[2]);
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

TEST_CASE("RoadPlaneScaleEstimator validates constructor inputs")
{
    SECTION("valid configuration")
    {
        auto tracker = makeKeypointTracker();
        const armvo::CameraConfig cameraConfig = makeCameraConfig();

        CHECK_NOTHROW(armvo::RoadPlaneScaleEstimator(cameraConfig, *tracker));
    }

    SECTION("invalid camera height")
    {
        auto tracker = makeKeypointTracker();
        armvo::CameraConfig cameraConfig = makeCameraConfig();
        cameraConfig.height = 0.0f;

        CHECK_THROWS(armvo::RoadPlaneScaleEstimator(cameraConfig, *tracker));
    }
}

TEST_CASE("RoadPlaneScaleEstimator estimates scale from tracked planar road points")
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

    const cv::Mat segmentationMap(imageSize, CV_8UC1, cv::Scalar(0));
    const cv::Mat noRoadMap(imageSize, CV_8UC1, cv::Scalar(1));

    bool forwardOnly = false;
    bool backwardOnly = false;
    SECTION("both directions") {}
    SECTION("only previous frame has road observations") { forwardOnly = true; }
    SECTION("only current frame has road observations") { backwardOnly = true; }

    tracker->setPreviousFrame(previousFrame);
    tracker->setCurrentFrame(currentFrame);

    std::unique_ptr<armvo::IScaleEstimator> estimator =
        std::make_unique<armvo::RoadPlaneScaleEstimator>(cameraConfig, *tracker);
    estimator->initialize({makeKeypoints(previousPoints), backwardOnly ? noRoadMap : segmentationMap});
    estimator->setCurrentFrameData({makeKeypoints(currentPoints), forwardOnly ? noRoadMap : segmentationMap});

    const auto measurements = estimator->estimate(rotation, unitTranslation);
    REQUIRE(measurements.size() == (forwardOnly || backwardOnly ? 1 : 2));
    for (const auto& measurement : measurements)
    {
        CHECK(measurement.value == Approx(expectedScale).margin(0.05f));
        CHECK(std::isfinite(measurement.weight));
        CHECK(measurement.weight > 0.0f);
    }
    armvo::TemporalScaleFilter temporalScaleFilter(cameraConfig.fps, 90.0f);
    const auto scale = temporalScaleFilter.update(measurements);

    REQUIRE(scale.has_value());
    CHECK(scale.value() == Approx(expectedScale).margin(0.05f));

    // Only forward tracking can supply the next estimate. Its points must come
    // from the promoted current observations, not from the original frame.
    cv::Mat nextFrame;
    cv::warpPerspective(currentFrame, nextFrame, cv::Mat(homography), imageSize);
    tracker->step();
    estimator->step();
    tracker->setCurrentFrame(nextFrame);
    estimator->setCurrentFrameData({{}, noRoadMap});
    const auto nextMeasurements = estimator->estimate(rotation, unitTranslation);
    REQUIRE(nextMeasurements.size() == (forwardOnly ? 0 : 1));
    if (!nextMeasurements.empty())
    {
        CHECK(nextMeasurements.front().value == Approx(expectedScale).margin(0.05f));
    }
    const auto nextScale = temporalScaleFilter.update(nextMeasurements);
    REQUIRE(nextScale.has_value());

    // Once both frames lack road observations, only the tracker supplies fallback.
    tracker->step();
    estimator->step();
    tracker->setCurrentFrame(nextFrame);
    estimator->setCurrentFrameData({{}, noRoadMap});
    const auto missingMeasurements = estimator->estimate(rotation, unitTranslation);
    CHECK(missingMeasurements.empty());
    CHECK(temporalScaleFilter.update(missingMeasurements).value() == nextScale.value());
}

TEST_CASE("RoadPlaneScaleEstimator recovers forward motion with projectively normalized homographies")
{
    auto tracker = makeKeypointTracker();
    const auto camera = makeCameraConfig();
    constexpr float expectedScale = 0.6f;
    float roadTilt = 0;
    float cameraPitch = 0;
    float direction = 1;
    SECTION("level road") {}
    SECTION("tilted road") { roadTilt = 0.12f; }
    SECTION("camera pitching") { cameraPitch = 0.02f; }
    SECTION("tilted road and camera pitching") { roadTilt = 0.12f; cameraPitch = 0.02f; }
    SECTION("reverse motion") { direction = -1; }
    CAPTURE(roadTilt, cameraPitch, direction);

    const cv::Matx33f rotation(1, 0, 0,
                             0, std::cos(cameraPitch), -std::sin(cameraPitch),
                             0, std::sin(cameraPitch), std::cos(cameraPitch));
    const cv::Matx31f normal(0, -std::cos(roadTilt), -std::sin(roadTilt));
    // Motion tangent to the plane preserves camera height in both frames.
    const cv::Matx31f translation = -direction * rotation * cv::Matx31f(0, -std::sin(roadTilt), std::cos(roadTilt));
    const auto homography = makePlanarPixelHomography(camera, rotation, translation, normal, expectedScale);
    REQUIRE(std::abs(homography(2, 2) - 1.0f) > 0.05f);

    std::vector<cv::Point2f> previousPoints;
    for (int y = 300; y <= 405; y += 35)
    {
        for (int x = 150; x <= 500; x += 50)
        {
            previousPoints.emplace_back(x, y);
        }
    }
    const cv::Size imageSize(640, 480);
    const auto currentPoints = warpPoints(homography, previousPoints);
    // Non-repeating texture avoids ambiguous matches between identical grid markers.
    cv::Mat texture(120, 160, CV_8UC1);
    cv::RNG rng(42);
    rng.fill(texture, cv::RNG::UNIFORM, 0, 256);
    cv::Mat previousFrame;
    cv::resize(texture, previousFrame, imageSize, 0, 0, cv::INTER_CUBIC);
    cv::GaussianBlur(previousFrame, previousFrame, cv::Size(3, 3), 0.8);
    cv::Mat currentFrame;
    cv::warpPerspective(previousFrame, currentFrame, cv::Mat(homography), imageSize);
    const cv::Mat roadMask(imageSize, CV_8UC1, cv::Scalar(0));
    tracker->setPreviousFrame(previousFrame);
    tracker->setCurrentFrame(currentFrame);
    armvo::RoadPlaneScaleEstimator estimator(camera, *tracker);
    estimator.initialize({makeKeypoints(previousPoints), roadMask});
    estimator.setCurrentFrameData({makeKeypoints(currentPoints), roadMask});

    const auto measurements = estimator.estimate(rotation, translation);
    REQUIRE(measurements.size() == 2);
    // Check each direction independently: fusion can conceal opposing scale biases.
    for (const auto& measurement : measurements)
    {
        // Allow 5% for image resampling and optical-flow error in this end-to-end test.
        CHECK(measurement.value == Approx(expectedScale).margin(0.03f));
        CHECK(std::isfinite(measurement.weight));
        CHECK(measurement.weight > 0);
    }
}

TEST_CASE("RoadPlaneScaleEstimator returns no scale when no road points are available")
{
    auto tracker = makeKeypointTracker();
    const armvo::CameraConfig cameraConfig = makeCameraConfig();

    const cv::Size imageSize(320, 240);
    const cv::Mat previousFrame(imageSize, CV_8UC1, cv::Scalar(0));
    const cv::Mat currentFrame(imageSize, CV_8UC1, cv::Scalar(0));
    const cv::Mat segmentationMap(imageSize, CV_8UC1, cv::Scalar(1));

    tracker->setPreviousFrame(previousFrame);
    tracker->setCurrentFrame(currentFrame);

    armvo::RoadPlaneScaleEstimator estimator(cameraConfig, *tracker);
    estimator.initialize({makeKeypoints(makeRoadFeaturePoints()), segmentationMap});
    estimator.setCurrentFrameData({makeKeypoints(makeRoadFeaturePoints()), segmentationMap});

    const auto measurements = estimator.estimate(cv::Matx33f::eye(), cv::Matx31f(1.0f, 0.0f, 0.0f));

    CHECK(measurements.empty());
}
