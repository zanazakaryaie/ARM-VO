#include "ErrorEstimator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>

#include <opencv2/core.hpp>

namespace armvo::tools
{

class ErrorEstimator::Impl
{
    std::array<float, 8> mLengths{100, 200, 300, 400, 500, 600, 700, 800};

public:
    Impl() = default;

    std::vector<Error> estimate(const std::vector<armvo::Pose>& estimatedPoses, const std::vector<armvo::Pose>& groundTruthPoses) const
    {
        if (estimatedPoses.empty() && groundTruthPoses.empty())
        {
            return {};
        }

        if (estimatedPoses.size() != groundTruthPoses.size())
        {
            throw std::logic_error("Number of estimated poses and ground truth poses mismatch");
        }

        constexpr int32_t STEP_SIZE = 10; // every second
        std::vector<float> distances = trajectoryDistances(groundTruthPoses);
        std::vector<Error> errors;

        for (int32_t firstFrame=0; firstFrame<groundTruthPoses.size(); firstFrame+=STEP_SIZE)
        {
            for (const float len : mLengths)
            {
                const int32_t lastFrame = lastFrameFromSegmentLength(distances, firstFrame, len);

                if (lastFrame == -1)
                {
                    continue;
                }

                cv::Matx44f gtPoseFirstFrame = getPoseAs4x4(groundTruthPoses[firstFrame]);
                cv::Matx44f gtPoseLastFrame = getPoseAs4x4(groundTruthPoses[lastFrame]);
                cv::Matx44f estimatedPoseFirstFrame = getPoseAs4x4(estimatedPoses[firstFrame]);
                cv::Matx44f estimatedPoseLastFrame = getPoseAs4x4(estimatedPoses[lastFrame]);

                cv::Matx44f deltaPoseGroundTruth = gtPoseFirstFrame.inv() * gtPoseLastFrame;
                cv::Matx44f deltaPoseEstimated = estimatedPoseFirstFrame.inv() * estimatedPoseLastFrame;
                cv::Matx44f poseError = deltaPoseEstimated.inv() * deltaPoseGroundTruth;
                float rotationErr = rotationError(poseError);
                float translationErr = translationError(poseError);

                errors.emplace_back(firstFrame, rotationErr/len, translationErr/len, len);
            }
        }
        return errors;
    }

    static cv::Matx44f getPoseAs4x4(const armvo::Pose& pose)
    {
        cv::Matx44f output = cv::Matx44f::eye();
        for (uint8_t i=0; i<3; i++)
        {
            for (uint8_t j=0; j<3; j++)
            {
                output(i, j) = pose.rotation(i, j);
            }
            output(i, 3) = pose.translation(i);
        }
        return output;
    }

    static std::vector<float> trajectoryDistances(const std::vector<armvo::Pose>& poses)
    {
        std::vector<float> dist;
        dist.reserve(poses.size());
        dist.push_back(0);
        for (int32_t i=1; i<poses.size(); i++)
        {
            const cv::Matx31f T1 = poses[i-1].translation;
            const cv::Matx31f T2 = poses[i].translation;
            dist.push_back(dist[i-1] + cv::norm(T1, T2));
        }
        return dist;
    }

    static int32_t lastFrameFromSegmentLength(const std::vector<float>& dist, int32_t startFrameIdx, float len)
    {
        for (int32_t i=startFrameIdx; i<dist.size(); i++)
        {
            if (dist[i] > dist[startFrameIdx]+len)
            {
                return i;
            }
        }
        return -1;
    }

    static float rotationError(const cv::Matx44f& poseError)
    {
        const float a = poseError(0, 0);
        const float b = poseError(1, 1);
        const float c = poseError(2, 2);
        const float d = 0.5*(a+b+c-1.0);
        return std::acos(std::max(std::min(d, 1.0f), -1.0f));
    }

    static float translationError(const cv::Matx44f& poseError)
    {
        const float dx = poseError(0, 3);
        const float dy = poseError(1, 3);
        const float dz = poseError(2, 3);
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
};

Error::Error(int32_t idx, float rotErr, float trErr, float len)
    : startFrameIndex(idx), rotationErr(rotErr), translationErr(trErr), length(len) {}

ErrorEstimator::ErrorEstimator()
{
    mImpl = std::make_shared<Impl>();
}

std::vector<Error> ErrorEstimator::estimate(const std::vector<armvo::Pose>& estimatedPoses, const std::vector<armvo::Pose>& groundTruthPoses)
{
    return mImpl->estimate(estimatedPoses, groundTruthPoses);
}

}
