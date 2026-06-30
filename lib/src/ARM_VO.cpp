#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "ARM_VO.hpp"
#include "Tracker.hpp"
#include "detector.hpp"
#include "Geometry.hpp"
#include "KeypointSampler.hpp"
#include "SemanticSegmentorNcnn.hpp"

#include <opencv2/imgproc.hpp>

namespace armvo
{

class ArmVo::Impl
{
    const ArmVoConfig mConfig;
    std::unique_ptr<KeypointDetector> mKeypointDetector;
    std::unique_ptr<KeypointSampler> mKeypointSampler;
    std::unique_ptr<KeypointTracker> mKeypointTracker;
    std::unique_ptr<GRIC> mGRIC;
    std::unique_ptr<MotionEstimator> mMotionEstimator;
    std::unique_ptr<ISemanticSegmentor> mSemanticSegmentor;
    std::unique_ptr<ScaleEstimator> mScaleEstimator;
    cv::Size mCameraResolution;
    std::vector<cv::Rect> mImageCells;
    std::vector<cv::Point2f> mPreviousKeypoints;
    Pose mCurrentPose;
    static const uint32_t kMinRequiredKeypoints = 200;

public:

    bool isInitialized = false;

    Impl(const ArmVoConfig& config) : mConfig(config)
    {
        if (config.keypointDetector.numberOfImageGridCols == 0 || config.keypointDetector.numberOfImageGridRows == 0)
        {
            throw std::logic_error("number of grid rows or cols cannot be zero");
        }

        mKeypointDetector = std::make_unique<KeypointDetector>(config.keypointDetector.responseThreshold);
        mKeypointSampler = std::make_unique<KeypointSampler>();
        mKeypointTracker = std::make_unique<KeypointTracker>(config.keypointTracker);
        mGRIC = std::make_unique<GRIC>();
        mMotionEstimator = std::make_unique<MotionEstimator>(config.camera.intrinsics);
        mSemanticSegmentor = std::make_unique<SemanticSegmentorNcnn>();
        mScaleEstimator = std::make_unique<ScaleEstimator>(config.camera, config.maxVehicleSpeed, mKeypointTracker);
    }

    Status initialize(const cv::Mat& frame, Pose& pose)
    {
        this->isInitialized = false; // so that update won't work if re-initializing fails

        if (!isFrameValid(frame))
        {
            return Status::INVALID_FRAME;
        }

        std::future<cv::Mat> segmentationHandle;
        if (mSemanticSegmentor->runsOnCoProcessor())
        {
            segmentationHandle = std::async(std::launch::async, &ISemanticSegmentor::segment, mSemanticSegmentor.get(), std::cref(frame), mConfig.camera.pixelFormat);
        }

        cv::Mat src = convertToGray(frame);
        mCameraResolution = src.size();

        const auto keypoints = mKeypointDetector->detect(src);
        if (keypoints.size() < kMinRequiredKeypoints)
        {
            return Status::NOT_ENOUGH_KEYPOINTS;
        }

        mKeypointTracker->setPreviousFrame(src);
        mImageCells = KeypointSampler::createROIs(src.rows, src.cols, mConfig.keypointDetector.numberOfImageGridRows, mConfig.keypointDetector.numberOfImageGridCols);

        cv::Mat segmentationMap;
        if (mSemanticSegmentor->runsOnCoProcessor())
        {
            segmentationMap = segmentationHandle.get();
        }
        else
        {
            segmentationMap = mSemanticSegmentor->segment(frame, mConfig.camera.pixelFormat);
        }

        if (segmentationMap.empty())
        {
            return Status::NOT_INITIALIZED;
        }

        cv::Mat staticMask = mSemanticSegmentor->getStaticMask(segmentationMap);
        mPreviousKeypoints = mKeypointSampler->run(keypoints, mImageCells, staticMask, mConfig.keypointDetector.maxNumberOfPoints);

        cv::Mat roadMask = mSemanticSegmentor->getRoadMask(segmentationMap);
        mScaleEstimator->initialize(keypoints, roadMask);

        mCurrentPose.rotation = cv::Matx33f::eye();
        mCurrentPose.translation = cv::Matx31f::zeros();
        pose = mCurrentPose;

        this->isInitialized = true;

        return Status::SUCCESS;
    }

    Status update(const cv::Mat& frame, Pose& pose)
    {
        if (!this->isInitialized)
        {
            return Status::NOT_INITIALIZED;
        }

        if (mPreviousKeypoints.size() < kMinRequiredKeypoints)
        {
            return Status::NOT_ENOUGH_KEYPOINTS;
        }

        if (!isFrameValid(frame))
        {
            return Status::INVALID_FRAME;
        }

        std::future<cv::Mat> segmentationHandle;
        if (mSemanticSegmentor->runsOnCoProcessor())
        {
            segmentationHandle = std::async(std::launch::async, &ISemanticSegmentor::segment, mSemanticSegmentor.get(), std::cref(frame), mConfig.camera.pixelFormat);
        }

        cv::Mat currentFrame = convertToGray(frame);

        mKeypointTracker->setCurrentFrame(currentFrame);
        std::vector<cv::Point2f> currentKeypoints = mKeypointTracker->trackForward(mPreviousKeypoints);
        if (currentKeypoints.size() < kMinRequiredKeypoints)
        {
            return Status::TRACK_LOST;
        }

        // Handle distorted frames
        std::vector<cv::Point2f> undistortedCurrentKeypoints;
        std::vector<cv::Point2f> undistortedPreviousKeypoints;
        if (mConfig.camera.distortions.has_value())
        {
            const cv::Mat distortions = mConfig.camera.distortions.value();
            undistortedCurrentKeypoints = undistortPoints(currentKeypoints, mConfig.camera.intrinsics, distortions);
            undistortedPreviousKeypoints = undistortPoints(mPreviousKeypoints,  mConfig.camera.intrinsics, distortions);
        }
        else
        {
            undistortedCurrentKeypoints = std::move(currentKeypoints);
            undistortedPreviousKeypoints = mPreviousKeypoints;
        }

        cv::Mat mask;
        std::future<cv::Matx33f> homographyMatrix = std::async(std::launch::async, &estimateHomographyMatrix, std::cref(undistortedPreviousKeypoints), std::cref(undistortedCurrentKeypoints));
        const cv::Matx33f fundamentalMatrix = estimateFundamentalMatrix(undistortedPreviousKeypoints, undistortedCurrentKeypoints, 1, mask);

        constexpr float SIGMA = 1.0f;
        auto HomographyMatrix = homographyMatrix.get();
        const float fCost = mGRIC->calculateFcost(undistortedPreviousKeypoints, undistortedCurrentKeypoints, fundamentalMatrix, SIGMA);
        const float hCost = mGRIC->calculateHcost(undistortedPreviousKeypoints, undistortedCurrentKeypoints, HomographyMatrix, SIGMA);

        if (hCost < fCost)
        {
            // Very small translation. We output the last estimated pose
            pose = mCurrentPose;
            return Status::FRAME_SKIPPED;
        }

        // Refine Fundamental Matrix

        std::vector<cv::Point2f> previousInliers;
        std::vector<cv::Point2f> currentInliers;
        const uint32_t numInliers = cv::countNonZero(mask);
        previousInliers.reserve(numInliers);
        currentInliers.reserve(numInliers);

        for (size_t i=0; i < undistortedPreviousKeypoints.size(); i++)
        {
            if (mask.data[i] == 1)
            {
                previousInliers.push_back(undistortedPreviousKeypoints[i]);
                currentInliers.push_back(undistortedCurrentKeypoints[i]);
            }
        }
        const cv::Matx33f moreAccurateFundamentalMatrix = estimateFundamentalMatrix(previousInliers, currentInliers, 0.5, mask);
        const auto estimatedMotion = mMotionEstimator->estimate(moreAccurateFundamentalMatrix, previousInliers, currentInliers, mask);
        const auto keypoints = mKeypointDetector->detect(currentFrame);
        if (keypoints.size() < kMinRequiredKeypoints)
        {
            return Status::NOT_ENOUGH_KEYPOINTS;
        }

        cv::Mat segmentationMap;
        if (mSemanticSegmentor->runsOnCoProcessor())
        {
            segmentationMap = segmentationHandle.get();
        }
        else
        {
            segmentationMap = mSemanticSegmentor->segment(frame, mConfig.camera.pixelFormat);
        }

        if (segmentationMap.empty())
        {
            return Status::SCALE_ESTIMATION_FAILURE;
        }

        cv::Mat roadMask = mSemanticSegmentor->getRoadMask(segmentationMap);
        mScaleEstimator->setCurrentFrameData(keypoints, roadMask);
        const std::optional<float> scaleFactor = mScaleEstimator->estimate(estimatedMotion.rotation, estimatedMotion.translation);

        if (!scaleFactor.has_value())
        {
            return Status::SCALE_ESTIMATION_FAILURE;
        }

        cv::Matx33f RotInv = estimatedMotion.rotation.t();
        cv::Matx31f TansInv = -RotInv * estimatedMotion.translation;
        mCurrentPose.translation += scaleFactor.value() * (mCurrentPose.rotation * TansInv);
        mCurrentPose.rotation = mCurrentPose.rotation * RotInv;

        // Prepare for next frame
        cv::Mat staticMask = mSemanticSegmentor->getStaticMask(segmentationMap);
        mPreviousKeypoints = mKeypointSampler->run(keypoints, mImageCells, staticMask, mConfig.keypointDetector.maxNumberOfPoints);
        mKeypointTracker->step();
        mScaleEstimator->step();

        pose = mCurrentPose;

        return Status::SUCCESS;
    }

    bool isFrameValid(const cv::Mat& frame) const
    {
        if (frame.empty())
        {
            return false;
        }

        if (this->isInitialized)
        {
            if (frame.rows != mCameraResolution.height || frame.cols != mCameraResolution.width)
            {
                return false;
            }
        }

        switch (mConfig.camera.pixelFormat)
        {
            case PixelFormat::Gray:
                return frame.type() == CV_8UC1;
            case PixelFormat::RGB:
            case PixelFormat::BGR:
                return frame.type() == CV_8UC3;
            default:
                return false;
        }
    }

    cv::Mat convertToGray(const cv::Mat& frame) const
    {
        cv::Mat src;
        switch (mConfig.camera.pixelFormat)
        {
            case PixelFormat::Gray:
                src = frame;
                break;
            case PixelFormat::BGR:
                cv::cvtColor(frame, src, cv::COLOR_BGR2GRAY);
                break;
            case PixelFormat::RGB:
                cv::cvtColor(frame, src, cv::COLOR_RGB2GRAY);
                break;
        }
        return src;
    }
};


ArmVo::ArmVo(const ArmVoConfig& config)
{
    mImpl = std::make_shared<Impl>(config);
}

Status ArmVo::initialize(const cv::Mat& frame, Pose& pose)
{
    return mImpl->initialize(frame, pose);
}

bool ArmVo::isInitialized() const noexcept
{
    return mImpl->isInitialized;
}

Status ArmVo::update(const cv::Mat& frame, Pose& pose)
{
    return mImpl->update(frame, pose);
}

} // namespace armvo
