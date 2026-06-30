#include <catch2/catch.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "Tracker.hpp"

namespace
{

constexpr float kTrackingTolerance = 0.35f;

class OpenCvThreadGuard
{
public:
    OpenCvThreadGuard() : mOriginalThreadCount(cv::getNumThreads())
    {
    }

    ~OpenCvThreadGuard()
    {
        cv::setNumThreads(mOriginalThreadCount);
    }

private:
    int mOriginalThreadCount;
};

armvo::KeypointTrackerConfig makeTrackerConfig(uint8_t windowSize = 21)
{
    armvo::KeypointTrackerConfig config;
    config.windowSize = windowSize;
    return config;
}

std::vector<cv::Point2f> makeFeaturePoints()
{
    return {
        {60.0f, 50.0f},
        {130.0f, 55.0f},
        {215.0f, 70.0f},
        {80.0f, 135.0f},
        {170.0f, 145.0f},
        {245.0f, 155.0f},
        {105.0f, 215.0f},
        {205.0f, 225.0f},
    };
}

cv::Mat makeTrackingPattern()
{
    cv::Mat image(280, 320, CV_8UC1, cv::Scalar(0));
    const auto points = makeFeaturePoints();

    for (size_t i = 0; i < points.size(); i++)
    {
        const cv::Point center(cvRound(points[i].x), cvRound(points[i].y));
        const int intensity = 170 + static_cast<int>((i * 17) % 70);

        cv::circle(image, center, 8, cv::Scalar(intensity), cv::FILLED);
        cv::rectangle(image, cv::Rect(center.x - 4, center.y - 4, 9, 9), cv::Scalar(35), cv::FILLED);
        cv::line(image, center + cv::Point(-11, 0), center + cv::Point(11, 0), cv::Scalar(255), 1);
        cv::line(image, center + cv::Point(0, -11), center + cv::Point(0, 11), cv::Scalar(255), 1);
    }

    cv::GaussianBlur(image, image, cv::Size(5, 5), 0.0);
    return image;
}

cv::Mat translateImage(const cv::Mat& image, const cv::Point2f& shift)
{
    const cv::Mat transform = (cv::Mat_<double>(2, 3) << 1.0, 0.0, shift.x,
                                                        0.0, 1.0, shift.y);
    cv::Mat translated;
    cv::warpAffine(image, translated, transform, image.size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(0));
    return translated;
}

std::vector<cv::Point2f> translatePoints(const std::vector<cv::Point2f>& points,
                                         const cv::Point2f& shift)
{
    std::vector<cv::Point2f> translated;
    translated.reserve(points.size());

    for (const auto& point : points)
    {
        translated.push_back(point + shift);
    }

    return translated;
}

float maxPointDistance(const std::vector<cv::Point2f>& points0,
                       const std::vector<cv::Point2f>& points1)
{
    REQUIRE(points0.size() == points1.size());

    float maxDistance = 0.0f;
    for (size_t i = 0; i < points0.size(); i++)
    {
        maxDistance = std::max(maxDistance, static_cast<float>(cv::norm(points0[i] - points1[i])));
    }

    return maxDistance;
}

} // namespace

TEST_CASE("KeypointTracker validates window size")
{
    CHECK_THROWS(armvo::KeypointTracker(makeTrackerConfig(0)));
    CHECK_THROWS(armvo::KeypointTracker(makeTrackerConfig(2)));
    CHECK_THROWS(armvo::KeypointTracker(makeTrackerConfig(4)));

    CHECK_NOTHROW(armvo::KeypointTracker(makeTrackerConfig(3)));
    CHECK_NOTHROW(armvo::KeypointTracker(makeTrackerConfig(15)));
}

TEST_CASE("KeypointTracker tracks points forward through translated frames")
{
    OpenCvThreadGuard guard;
    cv::setNumThreads(1);

    const cv::Mat previousFrame = makeTrackingPattern();
    const cv::Point2f shift(5.0f, -3.0f);
    const cv::Mat currentFrame = translateImage(previousFrame, shift);

    armvo::KeypointTracker tracker(makeTrackerConfig());
    tracker.setPreviousFrame(previousFrame);
    tracker.setCurrentFrame(currentFrame);

    std::vector<cv::Point2f> previousPoints = makeFeaturePoints();
    const std::vector<cv::Point2f> expectedPreviousPoints = previousPoints;
    const std::vector<cv::Point2f> expectedCurrentPoints = translatePoints(previousPoints, shift);

    const std::vector<cv::Point2f> currentPoints = tracker.trackForward(previousPoints);

    REQUIRE(currentPoints.size() == expectedCurrentPoints.size());
    CHECK(maxPointDistance(previousPoints, expectedPreviousPoints) < kTrackingTolerance);
    CHECK(maxPointDistance(currentPoints, expectedCurrentPoints) < kTrackingTolerance);
}

TEST_CASE("KeypointTracker tracks points backward through translated frames")
{
    OpenCvThreadGuard guard;
    cv::setNumThreads(1);

    const cv::Mat previousFrame = makeTrackingPattern();
    const cv::Point2f shift(5.0f, -3.0f);
    const cv::Mat currentFrame = translateImage(previousFrame, shift);

    armvo::KeypointTracker tracker(makeTrackerConfig());
    tracker.setPreviousFrame(previousFrame);
    tracker.setCurrentFrame(currentFrame);

    const std::vector<cv::Point2f> expectedPreviousPoints = makeFeaturePoints();
    std::vector<cv::Point2f> currentPoints = translatePoints(expectedPreviousPoints, shift);
    const std::vector<cv::Point2f> expectedCurrentPoints = currentPoints;

    const std::vector<cv::Point2f> previousPoints = tracker.trackBackward(currentPoints);

    REQUIRE(previousPoints.size() == expectedPreviousPoints.size());
    CHECK(maxPointDistance(currentPoints, expectedCurrentPoints) < kTrackingTolerance);
    CHECK(maxPointDistance(previousPoints, expectedPreviousPoints) < kTrackingTolerance);
}

TEST_CASE("KeypointTracker step promotes current frame to previous frame")
{
    OpenCvThreadGuard guard;
    cv::setNumThreads(1);

    const cv::Mat frame0 = makeTrackingPattern();
    const cv::Point2f firstShift(4.0f, 2.0f);
    const cv::Point2f secondShift(-3.0f, 5.0f);
    const cv::Mat frame1 = translateImage(frame0, firstShift);
    const cv::Mat frame2 = translateImage(frame0, firstShift + secondShift);

    armvo::KeypointTracker tracker(makeTrackerConfig());
    tracker.setPreviousFrame(frame0);
    tracker.setCurrentFrame(frame1);
    tracker.step();
    tracker.setCurrentFrame(frame2);

    std::vector<cv::Point2f> frame1Points = translatePoints(makeFeaturePoints(), firstShift);
    const std::vector<cv::Point2f> expectedFrame1Points = frame1Points;
    const std::vector<cv::Point2f> expectedFrame2Points = translatePoints(frame1Points, secondShift);

    const std::vector<cv::Point2f> frame2Points = tracker.trackForward(frame1Points);

    REQUIRE(frame2Points.size() == expectedFrame2Points.size());
    CHECK(maxPointDistance(frame1Points, expectedFrame1Points) < kTrackingTolerance);
    CHECK(maxPointDistance(frame2Points, expectedFrame2Points) < kTrackingTolerance);
}

TEST_CASE("KeypointTracker removes points that cannot be tracked")
{
    OpenCvThreadGuard guard;
    cv::setNumThreads(1);

    const cv::Mat blankFrame(100, 120, CV_8UC1, cv::Scalar(0));

    armvo::KeypointTracker tracker(makeTrackerConfig());
    tracker.setPreviousFrame(blankFrame);
    tracker.setCurrentFrame(blankFrame);

    std::vector<cv::Point2f> points = {
        {30.0f, 30.0f},
        {60.0f, 40.0f},
        {90.0f, 70.0f},
    };

    const std::vector<cv::Point2f> trackedPoints = tracker.trackForward(points);

    CHECK(points.empty());
    CHECK(trackedPoints.empty());
}
