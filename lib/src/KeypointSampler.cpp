#include "KeypointSampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>

namespace armvo
{

class FarthestPointSampler
{
public:

    static std::vector<cv::Point2f> run(const std::vector<Keypoint>& points, uint32_t numDesiredPoints)
    {
        std::vector<cv::Point2f> output;
        const uint32_t numPoints = points.size();
        if (numPoints == 0 || numDesiredPoints == 0)
        {
            return output;
        }

        numDesiredPoints = std::min(numDesiredPoints, numPoints);
        output.reserve(numDesiredPoints);

        if (numPoints == numDesiredPoints)
        {
            for (const auto& point : points)
            {
                output.emplace_back(point.x, point.y);
            }
            return output;
        }

        std::vector<uint8_t> selected(numPoints, 0);
        std::vector<float> nearestDist2(numPoints, std::numeric_limits<float>::infinity());

        const uint32_t firstIdx = findFirstPointIndex(points);
        output.emplace_back(points[firstIdx].x, points[firstIdx].y);
        selected[firstIdx] = 1;

        updateDistances(points, firstIdx, selected, nearestDist2);

        for (uint32_t k = 1; k < numDesiredPoints; k++)
        {
            uint32_t farthestIdx = points.size();
            float maxD2 = -1.0f;

            for (uint32_t i = 0; i < numPoints; i++)
            {
                if (!selected[i] && nearestDist2[i] > maxD2)
                {
                    maxD2 = nearestDist2[i];
                    farthestIdx = i;
                }
            }

            output.emplace_back(points[farthestIdx].x, points[farthestIdx].y);
            selected[farthestIdx] = 1;

            updateDistances(points, farthestIdx, selected, nearestDist2);
        }

        return output;
    }

private:

    static inline float dist2(const Keypoint& pt1, const Keypoint& pt2)
    {
        const float dx = pt1.x - pt2.x;
        const float dy = pt1.y - pt2.y;
        return dx*dx + dy*dy;
    }

    static void updateDistances(const std::vector<Keypoint>& pts,
                                uint32_t newIdx,
                                const std::vector<uint8_t>& selected,
                                std::vector<float>& nearestDist2)
    {
        const Keypoint& newPt = pts[newIdx];
        for (uint32_t i = 0; i < pts.size(); i++)
        {
            if (!selected[i])
            {
                float d2 = dist2(newPt, pts[i]);
                if (d2 < nearestDist2[i])
                {
                    nearestDist2[i] = d2;
                }
            }
        }
    }

    static uint32_t findFirstPointIndex(const std::vector<Keypoint>& pts)
    {
        uint32_t bestIdx = 0;
        float bestScore = pts[0].score;
        for (uint32_t i = 1; i < pts.size(); i++)
        {
            if (pts[i].score > bestScore)
            {
                bestScore = pts[i].score;
                bestIdx = i;
            }
        }
        return bestIdx;
    }
};


class NonOverlappingGrid::Impl
{
    uint32_t mHeight;
    uint32_t mWidth;
    uint32_t mNumRows;
    uint32_t mNumCols;

public:

    Impl(uint32_t height, uint32_t width, uint32_t numRows, uint32_t numCols)
        : mHeight(height), mWidth(width), mNumRows(numRows), mNumCols(numCols)
    {
    }

    uint32_t getNumCells() const
    {
        return mNumRows * mNumCols;
    }

    uint32_t findCellId(const Keypoint& kpt) const
    {
        const uint32_t cellY = (kpt.y * mNumRows)/mHeight;
        const uint32_t cellX = (kpt.x * mNumCols)/mWidth;
        return cellY * mNumCols + cellX;
    }
};

class KeypointSampler::Impl
{
    std::mt19937 mRNG{42};

public:

    Impl() = default;

    std::vector<cv::Point2f> run(const std::vector<Keypoint>& keypoints, const cv::Mat& mask, uint32_t maxNumPoints)
    {
        if (keypoints.empty() || mask.empty() || maxNumPoints == 0)
        {
            return {};
        }

        std::vector<Keypoint> pointsInROI;
        for (const auto& point : keypoints)
        {
            if (mask.at<uchar>(static_cast<int>(point.y), static_cast<int>(point.x)) != 0)
            {
                pointsInROI.push_back(point);
            }
        }

        const uint32_t numGoodScorePointsToKeep = std::min(maxNumPoints*3, static_cast<uint32_t>(pointsInROI.size()));
        keepStrongestKeypoints(pointsInROI, numGoodScorePointsToKeep);

        std::vector<cv::Point2f> sampledPoints = FarthestPointSampler::run(pointsInROI, maxNumPoints);

        shufflePointsDeterministically(sampledPoints); // shuffle (in a determinstic way) to de-bias RANSAC/USAC

        return sampledPoints;
    }

    std::vector<cv::Point2f> run(const std::vector<Keypoint>& keypoints, NonOverlappingGrid& grid, const cv::Mat& staticMask, uint32_t maxNumPoints)
    {
        const uint32_t numCells = grid.getNumCells();
        if (keypoints.empty() || numCells == 0 || maxNumPoints == 0)
        {
            return {};
        }

        maxNumPoints = std::min(static_cast<uint32_t>(keypoints.size()), maxNumPoints);

        const uint32_t numPointsPerCell = maxNumPoints / numCells;
        if (numPointsPerCell == 0)
        {
            return {};
        }

        std::vector<cv::Point2f> sampledPoints;
        sampledPoints.reserve(maxNumPoints);

        constexpr float staticBoost = 5.f;
        constexpr float dynamicPenalty = 2.f;

        std::vector<std::vector<Keypoint>> buckets(numCells);
        for (const Keypoint& kpt : keypoints)
        {
            const uint32_t cellId = grid.findCellId(kpt);
            const bool isStatic = staticMask.at<uchar>(static_cast<int>(kpt.y), static_cast<int>(kpt.x)) == 255;
            const float modifiedScore = kpt.score + (isStatic ? staticBoost : -dynamicPenalty);
            buckets[cellId].emplace_back(kpt.x, kpt.y, modifiedScore);
        }

        for (std::vector<Keypoint>& bucket : buckets)
        {
            auto strongEnd = sortStrongestKeypoints(bucket, numPointsPerCell);
            for (auto it = bucket.begin(); it != strongEnd; ++it)
            {
                sampledPoints.emplace_back(it->x, it->y);
            }
        }

        shufflePointsDeterministically(sampledPoints); // shuffle (in a determinstic way) to de-bias RANSAC/USAC

        return sampledPoints;
    }

private:

    static void keepStrongestKeypoints(std::vector<Keypoint>& keypoints, std::size_t desiredNumberOfKeypoints)
    {
        auto nth = sortStrongestKeypoints(keypoints, desiredNumberOfKeypoints);
        if (nth != keypoints.end())
        {
            keypoints.erase(nth, keypoints.end());
        }
    }

    static std::vector<Keypoint>::iterator sortStrongestKeypoints(std::vector<Keypoint>& keypoints, std::size_t desiredNumberOfKeypoints)
    {
        if (keypoints.size() <= desiredNumberOfKeypoints)
        {
            return keypoints.end();
        }
        std::vector<Keypoint>::iterator nth = keypoints.begin() + desiredNumberOfKeypoints;
        std::nth_element(keypoints.begin(), nth, keypoints.end(), [](const Keypoint& a, const Keypoint& b) {
            return a.score > b.score;
        });
        return nth;
    }

    void shufflePointsDeterministically(std::vector<cv::Point2f>& pts)
    {
        std::sort(pts.begin(), pts.end(), pointComparator);
        std::shuffle(pts.begin(), pts.end(), mRNG);
    }

    static bool pointComparator(const cv::Point2f& pt1, const cv::Point2f& pt2) noexcept
    {
        const uint32_t x1 = static_cast<uint32_t>(pt1.x);
        const uint32_t x2 = static_cast<uint32_t>(pt2.x);
        if (x1 != x2)
        {
            return x1 < x2;
        }
        return pt1.y < pt2.y;
    }
};

NonOverlappingGrid::NonOverlappingGrid(uint32_t height, uint32_t width, uint32_t numRows, uint32_t numCols)
{
    mImpl = std::make_shared<Impl>(height, width, numRows, numCols);
}

uint32_t NonOverlappingGrid::getNumCells()
{
    return mImpl->getNumCells();
}

uint32_t NonOverlappingGrid::findCellId(const Keypoint& keypoint)
{
    return mImpl->findCellId(keypoint);
}

KeypointSampler::KeypointSampler()
{
    mImpl = std::make_shared<Impl>();
}

std::vector<cv::Point2f> KeypointSampler::run(const std::vector<Keypoint>& keypoints, const cv::Mat& mask, uint32_t maxNumPoints)
{
    return mImpl->run(keypoints, mask, maxNumPoints);
}

std::vector<cv::Point2f> KeypointSampler::run(const std::vector<Keypoint>& keypoints, NonOverlappingGrid& grid, const cv::Mat& mask, uint32_t maxNumPoints)
{
    return mImpl->run(keypoints, grid, mask, maxNumPoints);
}

}
