#include <catch2/catch.hpp>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "KeypointSampler.hpp"

namespace
{

using Location = std::pair<int, int>;

std::vector<Location> sortedLocations(const std::vector<cv::Point2f>& points)
{
    std::vector<Location> locations;
    locations.reserve(points.size());

    for (const auto& point : points)
    {
        locations.emplace_back(static_cast<int>(point.x), static_cast<int>(point.y));
    }

    std::sort(locations.begin(), locations.end());
    return locations;
}

cv::Mat makeMask(int width, int height, uint8_t value = 0)
{
    return cv::Mat(height, width, CV_8UC1, cv::Scalar(value));
}

void setMaskValue(cv::Mat& mask, const armvo::Keypoint& keypoint, uint8_t value)
{
    mask.at<uint8_t>(static_cast<int>(keypoint.y), static_cast<int>(keypoint.x)) = value;
}

} // namespace

TEST_CASE("KeypointSampler creates non-overlapping image grid ROIs")
{
    const std::vector<cv::Rect> rois = armvo::KeypointSampler::createROIs(10, 12, 2, 3);

    REQUIRE(rois.size() == 6);
    CHECK(rois[0] == cv::Rect(0, 0, 4, 5));
    CHECK(rois[1] == cv::Rect(4, 0, 4, 5));
    CHECK(rois[2] == cv::Rect(8, 0, 4, 5));
    CHECK(rois[3] == cv::Rect(0, 5, 4, 5));
    CHECK(rois[4] == cv::Rect(4, 5, 4, 5));
    CHECK(rois[5] == cv::Rect(8, 5, 4, 5));
}

TEST_CASE("KeypointSampler creates ROIs that cover non-divisible dimensions")
{
    const std::vector<cv::Rect> rois = armvo::KeypointSampler::createROIs(10, 11, 3, 2);

    REQUIRE(rois.size() == 6);
    CHECK(rois[0] == cv::Rect(0, 0, 5, 3));
    CHECK(rois[1] == cv::Rect(5, 0, 6, 3));
    CHECK(rois[2] == cv::Rect(0, 3, 5, 3));
    CHECK(rois[3] == cv::Rect(5, 3, 6, 3));
    CHECK(rois[4] == cv::Rect(0, 6, 5, 4));
    CHECK(rois[5] == cv::Rect(5, 6, 6, 4));
}

TEST_CASE("KeypointSampler returns no ROIs when grid is denser than image")
{
    CHECK(armvo::KeypointSampler::createROIs(4, 8, 5, 2).empty());
    CHECK(armvo::KeypointSampler::createROIs(4, 8, 2, 9).empty());
}

TEST_CASE("KeypointSampler samples masked keypoints with farthest point sampling")
{
    const std::vector<armvo::Keypoint> keypoints = {
        {1.0f, 1.0f, 100.0f},
        {8.0f, 1.0f, 90.0f},
        {1.0f, 8.0f, 80.0f},
        {8.0f, 8.0f, 70.0f},
        {5.0f, 5.0f, 60.0f},
        {15.0f, 15.0f, 1000.0f},
    };

    cv::Mat mask = makeMask(20, 20);
    for (size_t i = 0; i < keypoints.size() - 1; i++)
    {
        setMaskValue(mask, keypoints[i], 255);
    }

    armvo::KeypointSampler sampler;
    const std::vector<cv::Point2f> sampled = sampler.run(keypoints, mask, 3);

    REQUIRE(sampled.size() == 3);
    CHECK(sortedLocations(sampled) == std::vector<Location>{{1, 1}, {8, 1}, {8, 8}});
}

TEST_CASE("KeypointSampler samples equally from ROIs and favors static keypoints")
{
    const std::vector<armvo::Keypoint> keypoints = {
        {1.0f, 1.0f, 10.0f},
        {2.0f, 1.0f, 5.0f},
        {3.0f, 1.0f, 4.0f},
        {4.0f, 1.0f, 9.0f},
        {11.0f, 1.0f, 18.0f},
        {12.0f, 1.0f, 10.0f},
        {13.0f, 1.0f, 16.0f},
        {14.0f, 1.0f, 8.0f},
    };

    cv::Mat staticMask = makeMask(20, 10);
    setMaskValue(staticMask, keypoints[1], 255);
    setMaskValue(staticMask, keypoints[2], 255);
    setMaskValue(staticMask, keypoints[5], 255);

    const std::vector<cv::Rect> rois = {
        cv::Rect(0, 0, 10, 10),
        cv::Rect(10, 0, 10, 10),
    };

    armvo::KeypointSampler sampler;
    const std::vector<cv::Point2f> sampled = sampler.run(keypoints, rois, staticMask, 4);

    REQUIRE(sampled.size() == 4);
    CHECK(sortedLocations(sampled) == std::vector<Location>{{2, 1}, {3, 1}, {11, 1}, {12, 1}});
}

TEST_CASE("KeypointSampler returns empty output for empty inputs")
{
    const std::vector<armvo::Keypoint> keypoints = {
        {1.0f, 1.0f, 10.0f},
    };
    const std::vector<cv::Rect> rois = {
        cv::Rect(0, 0, 5, 5),
    };
    const cv::Mat mask = makeMask(5, 5, 255);

    armvo::KeypointSampler sampler;

    CHECK(sampler.run({}, mask, 1).empty());
    CHECK(sampler.run(keypoints, cv::Mat(), 1).empty());
    CHECK(sampler.run(keypoints, mask, 0).empty());
    CHECK(sampler.run({}, rois, mask, 1).empty());
    CHECK(sampler.run(keypoints, std::vector<cv::Rect>{}, mask, 1).empty());
    CHECK(sampler.run(keypoints, rois, mask, 0).empty());
}
