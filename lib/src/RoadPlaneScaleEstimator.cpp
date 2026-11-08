#include "RoadPlaneScaleEstimator.hpp"

#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>

#include <opencv2/core.hpp>

#include "Geometry.hpp"
#include "KeypointSampler.hpp"
#include "KeypointTracker.hpp"

namespace armvo
{

class RoadPlaneScaleEstimator::Impl
{
    CameraConfig mCamConfig;
    cv::Matx33f mKinv;
    KeypointSampler mKeypointSampler;
    static constexpr uint8_t ROAD_LABEL = 0;
    static constexpr uint32_t NUM_ROAD_POINTS_TO_USE = 100;

    std::vector<cv::Point2f> mPrevRoadPoints;
    std::vector<cv::Point2f> mCurrRoadPoints;
    KeypointTracker& mKeyPointTracker;

public:

    Impl(const CameraConfig& camConfig, KeypointTracker& keyPointTracker)
        : mCamConfig(camConfig), mKeyPointTracker(keyPointTracker)
    {
        if (camConfig.height <= 0)
        {
            throw std::invalid_argument("Camera height is expected to be larger than zero");
        }
        mKinv = camConfig.intrinsics.inv();
    }

    void initialize(const ScaleEstimationInput& frame)
    {
        const cv::Mat roadMask = frame.segmentationMap == ROAD_LABEL;
        std::vector<cv::Point2f> rawRoadPoints = mKeypointSampler.run(frame.keypoints, roadMask, NUM_ROAD_POINTS_TO_USE);
        if (mCamConfig.distortions.has_value())
        {
            mPrevRoadPoints = undistortPoints(rawRoadPoints, mCamConfig.intrinsics, mCamConfig.distortions.value());
        }
        else
        {
            mPrevRoadPoints = std::move(rawRoadPoints);
        }
    }

    void setCurrentFrameData(const ScaleEstimationInput& frame)
    {
        const cv::Mat roadMask = frame.segmentationMap == ROAD_LABEL;
        std::vector<cv::Point2f> rawRoadPoints = mKeypointSampler.run(frame.keypoints, roadMask, NUM_ROAD_POINTS_TO_USE);
        if (mCamConfig.distortions.has_value())
        {
            mCurrRoadPoints = undistortPoints(rawRoadPoints, mCamConfig.intrinsics, mCamConfig.distortions.value());
        }
        else
        {
            mCurrRoadPoints = std::move(rawRoadPoints);
        }
    }

    std::vector<ScaleMeasurement> estimate(const cv::Matx33f& rotation, const cv::Matx31f& translation)
    {
        std::vector<ScaleMeasurement> measurements;
        const auto scaleForward = estimateForward(rotation, translation);
        const auto scaleBackward = estimateBackward(rotation, translation);
        if (scaleForward.has_value())
        {
            measurements.push_back(scaleForward.value());
        }
        if (scaleBackward.has_value())
        {
            measurements.push_back(scaleBackward.value());
        }
        return measurements;
    }

    void step()
    {
        mPrevRoadPoints = std::move(mCurrRoadPoints);
    }

private:

    std::optional<ScaleMeasurement> estimateForward(const cv::Matx33f& rotation, const cv::Matx31f& translation)
    {
        try
        {
            const auto trackedRoadPoints = mKeyPointTracker.trackForward(mPrevRoadPoints);
            return estimateScale(mPrevRoadPoints, trackedRoadPoints, rotation, translation);
        }
        catch (...)
        {
            return std::nullopt;
        }
    }

    std::optional<ScaleMeasurement> estimateBackward(const cv::Matx33f& rotation, const cv::Matx31f& translation)
    {
        try
        {
            const auto trackedRoadPoints = mKeyPointTracker.trackBackward(mCurrRoadPoints);
            cv::Matx33f backwardRotation = rotation.t();
            cv::Matx31f backwardTranslation = -backwardRotation * translation;
            return estimateScale(mCurrRoadPoints, trackedRoadPoints, backwardRotation, backwardTranslation);
        }
        catch (...)
        {
            return std::nullopt;
        }
    }

    std::optional<ScaleMeasurement> estimateScale(const std::vector<cv::Point2f>& prevRoadPoints,
                                       const std::vector<cv::Point2f>& currRoadPoints,
                                       const cv::Matx33f& rotation,
                                       const cv::Matx31f& translation) const
    {
        static const bool accurateHomo = true;
        cv::Matx33f Homography;
        std::vector<uchar> mask;
        try
        {
            Homography = estimateHomographyMatrixWithInliers(prevRoadPoints, currRoadPoints, mask, accurateHomo);
        }
        catch (...)
        {
            return std::nullopt;
        }

        const cv::Matx33f Hnorm = mKinv * Homography * mCamConfig.intrinsics;

        if (isHomographyDegenerate(Hnorm))
        {
            return std::nullopt;
        }

        cv::Matx33f M = rotation - Hnorm;
        const cv::Matx31f v = M.t() * translation;
        const float scale = mCamConfig.height * cv::norm(v);

        ScaleMeasurement output;
        output.value = scale;

        std::vector<cv::Point2f> inlierPrevPoints;
        std::vector<cv::Point2f> inlierCurrPoints;
        const uint32_t numInliers = cv::countNonZero(mask);
        inlierPrevPoints.reserve(numInliers);
        inlierCurrPoints.reserve(numInliers);
        for (uint32_t i=0; i<prevRoadPoints.size(); i++)
        {
            if (mask[i])
            {
                inlierPrevPoints.push_back(prevRoadPoints[i]);
                inlierCurrPoints.push_back(currRoadPoints[i]);
            }
        }

        const std::vector<float> residuals = sampsonDistanceSquaredH(Homography, inlierPrevPoints, inlierCurrPoints);
        const float sum = std::accumulate(residuals.begin(), residuals.end(), 0.0f);
        const float sampsonDistance = sum / residuals.size();
        output.weight = 1.0f / (1e-6f + sampsonDistance);

        return output;
    }

    static bool isHomographyDegenerate(const cv::Matx33f& homography)
    {
        cv::SVD svd(homography, cv::SVD::MODIFY_A);
        return (svd.w.at<float>(2) / svd.w.at<float>(1)) < 1e-3f;
    }
};

RoadPlaneScaleEstimator::RoadPlaneScaleEstimator(const CameraConfig& cameraConfig, KeypointTracker& keypointTracker)
{
    mImpl = std::make_unique<Impl>(cameraConfig, keypointTracker);
}

void RoadPlaneScaleEstimator::initialize(const ScaleEstimationInput& frame)
{
    mImpl->initialize(frame);
}

void RoadPlaneScaleEstimator::setCurrentFrameData(const ScaleEstimationInput& frame)
{
    mImpl->setCurrentFrameData(frame);
}

std::vector<ScaleMeasurement> RoadPlaneScaleEstimator::estimate(const cv::Matx33f& rotation,
                                                               const cv::Matx31f& translation)
{
    return mImpl->estimate(rotation, translation);
}

void RoadPlaneScaleEstimator::step()
{
    mImpl->step();
}

} // namespace armvo
