#include <catch2/catch.hpp>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "KeypointDetector.hpp"

namespace
{

constexpr uint8_t kDefaultThreshold = 10;

using Location = std::pair<int, int>;

cv::Mat makeBlankGray(int width, int height, uint8_t value = 0)
{
    return cv::Mat(height, width, CV_8UC1, cv::Scalar(value));
}

cv::Mat blur5x5(const cv::Mat& image)
{
    cv::Mat blurred;
    cv::GaussianBlur(image, blurred, cv::Size(5, 5), 0.0);
    return blurred;
}

cv::Mat makeBlurredGridPattern(int width, int height)
{
    cv::Mat image = makeBlankGray(width, height);

    constexpr int margin = 32;
    constexpr int step = 48;
    constexpr int boxSize = 14;

    for (int y = margin; y <= height - margin - boxSize; y += step)
    {
        for (int x = margin; x <= width - margin - boxSize; x += step)
        {
            cv::rectangle(image, cv::Rect(x, y, boxSize, boxSize), cv::Scalar(255), cv::FILLED);
        }
    }

    return blur5x5(image);
}

cv::Mat makeBlurredLowContrastSquare()
{
    cv::Mat image = makeBlankGray(320, 240, 128);
    cv::rectangle(image, cv::Rect(100, 70, 96, 96), cv::Scalar(150), cv::FILLED);
    return blur5x5(image);
}

std::vector<cv::KeyPoint> detectOpenCvFast(const cv::Mat& image, uint8_t threshold)
{
    std::vector<cv::KeyPoint> keypoints;
    cv::FAST(image, keypoints, threshold, true, cv::FastFeatureDetector::TYPE_9_16);
    return keypoints;
}

std::vector<Location> sortedLocations(const std::vector<armvo::Keypoint>& keypoints)
{
    std::vector<Location> locations;
    locations.reserve(keypoints.size());

    for (const auto& keypoint : keypoints)
    {
        locations.emplace_back(
            static_cast<int>(std::lround(keypoint.x * 1000.0f)),
            static_cast<int>(std::lround(keypoint.y * 1000.0f)));
    }

    std::sort(locations.begin(), locations.end());
    return locations;
}

std::vector<Location> sortedLocations(const std::vector<cv::KeyPoint>& keypoints)
{
    std::vector<Location> locations;
    locations.reserve(keypoints.size());

    for (const auto& keypoint : keypoints)
    {
        locations.emplace_back(
            static_cast<int>(std::lround(keypoint.pt.x * 1000.0f)),
            static_cast<int>(std::lround(keypoint.pt.y * 1000.0f)));
    }

    std::sort(locations.begin(), locations.end());
    return locations;
}

bool hasSameLocations(const std::vector<armvo::Keypoint>& actual,
                      const std::vector<cv::KeyPoint>& expected)
{
    return sortedLocations(actual) == sortedLocations(expected);
}

bool allLocationsInside(const std::vector<armvo::Keypoint>& keypoints, const cv::Mat& image)
{
    return std::all_of(keypoints.begin(), keypoints.end(),
        [&](const armvo::Keypoint& keypoint)
        {
            return keypoint.x >= 0.0f &&
                   keypoint.y >= 0.0f &&
                   keypoint.x < static_cast<float>(image.cols) &&
                   keypoint.y < static_cast<float>(image.rows);
        });
}

bool allScoresValid(const std::vector<armvo::Keypoint>& keypoints)
{
    return std::all_of(keypoints.begin(), keypoints.end(),
        [](const armvo::Keypoint& keypoint)
        {
            return std::isfinite(keypoint.score) && keypoint.score >= 0.0f;
        });
}

} // namespace

TEST_CASE("KeypointDetector validates response threshold")
{
    CHECK_THROWS(armvo::KeypointDetector(0));
    CHECK_THROWS(armvo::KeypointDetector(255));

    CHECK_NOTHROW(armvo::KeypointDetector(1));
    CHECK_NOTHROW(armvo::KeypointDetector(254));
}

TEST_CASE("KeypointDetector returns no keypoints for a blank image")
{
    armvo::KeypointDetector detector(kDefaultThreshold);

    const cv::Mat image = makeBlankGray(320, 240);
    const auto keypoints = detector.detect(image);

    CHECK(keypoints.empty());
}

TEST_CASE("KeypointDetector matches OpenCV FAST locations on a deterministic pattern")
{
    armvo::KeypointDetector detector(kDefaultThreshold);
    const cv::Mat image = makeBlurredGridPattern(320, 240);

    const auto actual = detector.detect(image);
    const auto expected = detectOpenCvFast(image, kDefaultThreshold);

    REQUIRE_FALSE(expected.empty());
    REQUIRE(actual.size() == expected.size());
    CHECK(hasSameLocations(actual, expected));
    CHECK(allLocationsInside(actual, image));
    CHECK(allScoresValid(actual));
}

TEST_CASE("KeypointDetector response threshold controls sensitivity")
{
    const cv::Mat image = makeBlurredLowContrastSquare();

    armvo::KeypointDetector lowThresholdDetector(10);
    armvo::KeypointDetector highThresholdDetector(15);

    const auto lowThresholdKeypoints = lowThresholdDetector.detect(image);
    const auto highThresholdKeypoints = highThresholdDetector.detect(image);

    CHECK(hasSameLocations(lowThresholdKeypoints, detectOpenCvFast(image, 10)));
    CHECK(hasSameLocations(highThresholdKeypoints, detectOpenCvFast(image, 15)));

    REQUIRE_FALSE(lowThresholdKeypoints.empty());
    CHECK(highThresholdKeypoints.size() < lowThresholdKeypoints.size());
}

TEST_CASE("KeypointDetector gives the same result for non-contiguous ROI and continuous clone")
{
    const int width = 320;
    const int height = 240;

    const cv::Mat pattern = makeBlurredGridPattern(width, height);
    cv::Mat padded = makeBlankGray(width + 37, height);
    pattern.copyTo(padded(cv::Rect(0, 0, width, height)));

    const cv::Mat roi = padded(cv::Rect(0, 0, width, height));
    const cv::Mat continuous = roi.clone();

    REQUIRE_FALSE(roi.isContinuous());
    REQUIRE(continuous.isContinuous());

    armvo::KeypointDetector detector(kDefaultThreshold);
    const auto roiKeypoints = detector.detect(roi);
    const auto continuousKeypoints = detector.detect(continuous);

    REQUIRE(roiKeypoints.size() == continuousKeypoints.size());
    CHECK(sortedLocations(roiKeypoints) == sortedLocations(continuousKeypoints));
    CHECK(hasSameLocations(roiKeypoints, detectOpenCvFast(continuous, kDefaultThreshold)));
}
