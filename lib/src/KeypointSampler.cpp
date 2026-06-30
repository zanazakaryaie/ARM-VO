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


class KeypointSampler::Impl
{
    std::mt19937 mRNG{42};

public:

    Impl() = default;

    static std::vector<cv::Rect> createROIs(uint32_t height, uint32_t width, uint32_t numRows, uint32_t numCols)
    {
        if (numRows > height || numCols > width)
        {
            return {};
        }

        std::vector<cv::Rect> ROIs;
        ROIs.reserve(numRows * numCols);

        for (uint32_t celly=0; celly<numRows; celly++)
        {
            const uint32_t startRow = (celly*height)/numRows;
            const uint32_t endRow = ((celly+1)*height)/numRows;

            for (uint32_t cellx=0; cellx<numCols; cellx++)
            {
                const uint32_t startCol = (cellx*width)/numCols;
                const uint32_t endCol = ((cellx+1)*width)/numCols;

                ROIs.emplace_back(startCol, startRow, endCol - startCol, endRow - startRow);
            }
        }

        return ROIs;
    }

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

        // To get identical results in each run (the detector may shuffle the points because its multi-threaded)
        shufflePointsDeterministically(sampledPoints);

        return sampledPoints;
    }

    // We assume that ROIs are non-overlapping and sampling is done equally
    std::vector<cv::Point2f> run(const std::vector<Keypoint>& keypoints, const std::vector<cv::Rect>& ROIs, const cv::Mat& staticMask, uint32_t maxNumPoints)
    {
        if (keypoints.empty() || ROIs.empty() || maxNumPoints == 0)
        {
            return {};
        }

        maxNumPoints = std::min(static_cast<uint32_t>(keypoints.size()), maxNumPoints);

        const uint32_t numPointsPerCell = maxNumPoints / ROIs.size();

        std::vector<cv::Point2f> sampledPoints;
        sampledPoints.reserve(maxNumPoints);

        constexpr float staticBoost = 5.f;
        constexpr float dynamicPenalty = 2.f;

        std::vector<Keypoint> pointsInROI;
        for (const auto& roi : ROIs)
        {
            for (const auto& point : keypoints)
            {
                if (isInside(point, roi))
                {
                    const bool isStatic = staticMask.at<uchar>(static_cast<int>(point.y), static_cast<int>(point.x)) == 255;
                    const float modifiedScore = point.score + (isStatic ? staticBoost : -dynamicPenalty);
                    pointsInROI.emplace_back(point.x, point.y, modifiedScore);
                }
            }
            keepStrongestKeypoints(pointsInROI, numPointsPerCell);
            for (const auto& point : pointsInROI)
            {
                sampledPoints.emplace_back(point.x, point.y);
            }
            pointsInROI.clear();
        }

        // To get identical results in each run (the detector may shuffle the points because its multi-threaded)
        shufflePointsDeterministically(sampledPoints);

        return sampledPoints;
    }

private:

    static void keepStrongestKeypoints(std::vector<Keypoint>& keypoints, std::size_t desiredNumberOfKeypoints)
    {
        if (keypoints.size() > desiredNumberOfKeypoints)
        {
            std::vector<Keypoint>::iterator nth = keypoints.begin() + desiredNumberOfKeypoints;
            std::nth_element(keypoints.begin(), nth, keypoints.end(), [](const Keypoint& a, const Keypoint& b) {
                return std::abs(a.score) > std::abs(b.score);
            });
            keypoints.erase(nth, keypoints.end());
        }
    }

    static bool isInside(const Keypoint& pt, const cv::Rect& roi) noexcept
    {
        return (pt.x >= roi.x) && (pt.x < roi.x + roi.width) && (pt.y >= roi.y) && (pt.y < roi.y + roi.height);
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

KeypointSampler::KeypointSampler()
{
    mImpl = std::make_shared<Impl>();
}

std::vector<cv::Rect> KeypointSampler::createROIs(uint32_t height, uint32_t width, uint32_t numRows, uint32_t numCols)
{
    return Impl::createROIs(height, width, numRows, numCols);
}

std::vector<cv::Point2f> KeypointSampler::run(const std::vector<Keypoint>& keypoints, const cv::Mat& mask, uint32_t maxNumPoints)
{
    return mImpl->run(keypoints, mask, maxNumPoints);
}

std::vector<cv::Point2f> KeypointSampler::run(const std::vector<Keypoint>& keypoints, const std::vector<cv::Rect>& ROIs, const cv::Mat& mask, uint32_t maxNumPoints)
{
    return mImpl->run(keypoints, ROIs, mask, maxNumPoints);
}

}
