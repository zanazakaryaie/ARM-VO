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

TEST_CASE("NonOverlappingGrid reports the number of grid cells")
{
    armvo::NonOverlappingGrid grid(10, 12, 2, 3);

    CHECK(grid.getNumCells() == 6);
}

TEST_CASE("NonOverlappingGrid assigns points to image cells")
{
    armvo::NonOverlappingGrid grid(10, 12, 2, 3);

    CHECK(grid.findCellId({0.0f, 0.0f, 1.0f}) == 0);
    CHECK(grid.findCellId({3.0f, 4.0f, 1.0f}) == 0);
    CHECK(grid.findCellId({4.0f, 0.0f, 1.0f}) == 1);
    CHECK(grid.findCellId({8.0f, 0.0f, 1.0f}) == 2);
    CHECK(grid.findCellId({0.0f, 5.0f, 1.0f}) == 3);
    CHECK(grid.findCellId({11.0f, 9.0f, 1.0f}) == 5);
}

TEST_CASE("NonOverlappingGrid assigns points in non-divisible image dimensions")
{
    armvo::NonOverlappingGrid grid(10, 11, 3, 2);

    CHECK(grid.findCellId({0.0f, 0.0f, 1.0f}) == 0);
    CHECK(grid.findCellId({5.0f, 3.0f, 1.0f}) == 0);
    CHECK(grid.findCellId({6.0f, 3.0f, 1.0f}) == 1);
    CHECK(grid.findCellId({0.0f, 4.0f, 1.0f}) == 2);
    CHECK(grid.findCellId({6.0f, 4.0f, 1.0f}) == 3);
    CHECK(grid.findCellId({0.0f, 7.0f, 1.0f}) == 4);
    CHECK(grid.findCellId({10.0f, 9.0f, 1.0f}) == 5);
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

    armvo::NonOverlappingGrid grid(10, 20, 1, 2);

    armvo::KeypointSampler sampler;
    const std::vector<cv::Point2f> sampled = sampler.run(keypoints, grid, staticMask, 4);

    REQUIRE(sampled.size() == 4);
    CHECK(sortedLocations(sampled) == std::vector<Location>{{2, 1}, {3, 1}, {11, 1}, {12, 1}});
}

TEST_CASE("KeypointSampler grid sampling uses signed response scores")
{
    const std::vector<armvo::Keypoint> keypoints = {
        {1.0f, 1.0f, 10.0f},
        {2.0f, 1.0f, -100.0f},
        {11.0f, 1.0f, 20.0f},
        {12.0f, 1.0f, -200.0f},
    };

    cv::Mat staticMask = makeMask(20, 10);
    armvo::NonOverlappingGrid grid(10, 20, 1, 2);

    armvo::KeypointSampler sampler;
    const std::vector<cv::Point2f> sampled = sampler.run(keypoints, grid, staticMask, 2);

    REQUIRE(sampled.size() == 2);
    CHECK(sortedLocations(sampled) == std::vector<Location>{{1, 1}, {11, 1}});
}

TEST_CASE("KeypointSampler returns empty output for empty inputs")
{
    const std::vector<armvo::Keypoint> keypoints = {
        {1.0f, 1.0f, 10.0f},
    };
    const cv::Mat mask = makeMask(5, 5, 255);
    armvo::NonOverlappingGrid grid(5, 5, 1, 1);
    armvo::NonOverlappingGrid denseGrid(5, 5, 2, 2);

    armvo::KeypointSampler sampler;

    CHECK(sampler.run({}, mask, 1).empty());
    CHECK(sampler.run(keypoints, cv::Mat(), 1).empty());
    CHECK(sampler.run(keypoints, mask, 0).empty());
    CHECK(sampler.run({}, grid, mask, 1).empty());
    CHECK(sampler.run(keypoints, grid, mask, 0).empty());
    CHECK(sampler.run(keypoints, denseGrid, mask, 1).empty());
}
