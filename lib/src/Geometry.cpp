#include "Geometry.hpp"
#include "KeypointSampler.hpp"

#include <algorithm>
#include <cmath>
#include <future>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <tuple>

#include <opencv2/calib3d.hpp>

namespace
{

cv::Matx34f createProjectionMatrix(const cv::Matx33f& rotation, const cv::Matx31f& translation)
{
    cv::Matx34f projectionMatrix;
    for (int i=0; i<3; i++)
    {
        for (int j=0; j<3; j++)
        {
            projectionMatrix(i, j) = rotation(i, j);
        }
    }
    for (int i=0; i<3; i++)
    {
        projectionMatrix(i, 3) = translation(i);
    }
    return projectionMatrix;
}

std::vector<float> sampsonDistanceSquaredH(const cv::Matx33f& homographyMatrix,
                                           const std::vector<cv::Point2f>& pts0,
                                           const std::vector<cv::Point2f>& pts1)
{
    float G0[3], G1[3], alg[2];
    const float* const H = homographyMatrix.val;
    const uint32_t npts = pts0.size();
    std::vector<float> residuals(npts);

    for(uint32_t i=0; i<npts; ++i)
    {
        const float m0_x = pts0[i].x;
        const float m0_y = pts0[i].y;

        const float m1_x = pts1[i].x;
        const float m1_y = pts1[i].y;

        G0[0]= H[0] - m1_x * H[6];
        G0[1]= H[1] - m1_x * H[7];
        G0[2]=-m0_x * H[6] - m0_y * H[7] - H[8];

        G1[0]= H[3] - m1_y * H[6];
        G1[1]= H[4] - m1_y * H[7];
        G1[2]=-m0_x * H[6] - m0_y * H[7] - H[8];

        const float magG0 = sqrt(G0[0]*G0[0] + G0[1]*G0[1] + G0[2]*G0[2]);
        const float magG1 = sqrt(G1[0]*G1[0] + G1[1]*G1[1] + G1[2]*G1[2]);
        const float magG0G1 = G0[0]*G1[0] + G0[1]*G1[1];

        const float alpha = acos(magG0G1 /(magG0*magG1));

        /* algebraic distance */
        alg[0] = m0_x*H[0] + m0_y*H[1] + H[2] -
                    m1_x*(m0_x*H[6] + m0_y*H[7] + H[8]);

        alg[1] = m0_x*H[3] + m0_y*H[4] + H[5] -
                    m1_y*(m0_x*H[6] + m0_y*H[7] + H[8]);

        const float D1 = alg[0]/magG0;
        const float D2 = alg[1]/magG1;

        residuals[i] = (D1*D1 + D2*D2 - 2.0*D1*D2*cosf(alpha))/sinf(alpha);
    }

    return residuals;
}

std::vector<float> sampsonDistanceSquaredF(const cv::Matx33f& fundamentalMatrix,
                                           const std::vector<cv::Point2f>& pts0,
                                           const std::vector<cv::Point2f>& pts1)
{
    float Fm0[3], Ftm1[3];
    const uint32_t npts = pts0.size();
    const float* const F = fundamentalMatrix.val;
    std::vector<float> residuals(npts);

    for(uint32_t i=0; i<npts; ++i)
    {
        const float m0_x = pts0[i].x;
        const float m0_y = pts0[i].y;

        const float m1_x = pts1[i].x;
        const float m1_y = pts1[i].y;

        /* F*m0 */
        Fm0[0] = F[0]*m0_x + F[1]*m0_y + F[2];
        Fm0[1] = F[3]*m0_x + F[4]*m0_y + F[5];
        Fm0[2] = F[6]*m0_x + F[7]*m0_y + F[8];

        /* F'*m1 */
        Ftm1[0] = F[0]*m1_x + F[3]*m1_y + F[6];
        Ftm1[1] = F[1]*m1_x + F[4]*m1_y + F[7];
        // Ftm1[2] = F[2]*m1_x + F[5]*m1_y + F[8];

        /* m1'*F*m0 */
        const float m1Fm0 = Fm0[0]*m1_x + Fm0[1]*m1_y +  Fm0[2];

        residuals[i] = m1Fm0*m1Fm0/(Fm0[0]*Fm0[0] + Fm0[1]*Fm0[1] + Ftm1[0]*Ftm1[0] + Ftm1[1]*Ftm1[1]);
    }

    return residuals;
}

cv::Matx33f convertMat(cv::Mat src)
{
    cv::Matx33f dst;
    const double* const srcPtr = src.ptr<double>();
    for (uint8_t i=0; i<9; i++)
    {
        dst.val[i] = static_cast<float>(srcPtr[i]);
    }
    return dst;
}

cv::Matx33f estimateHomographyMatrix(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2, std::vector<uchar>& mask, bool accurate)
{
    cv::Mat homographyMatrixDouble;
    if (accurate)
    {
        homographyMatrixDouble = cv::findHomography(points1, points2, mask, cv::USAC_MAGSAC);
    }
    else
    {
        homographyMatrixDouble = cv::findHomography(points1, points2, cv::USAC_FAST, 10, mask, 1500);
    }

    if (homographyMatrixDouble.empty())
    {
        throw std::runtime_error("Failed to compute homography matrix");
    }
    return convertMat(homographyMatrixDouble);
}

}

namespace armvo
{

cv::Matx33f estimateFundamentalMatrix(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2, float reprojectionError, cv::Mat& mask)
{
    constexpr int METHOD = cv::FM_RANSAC;
    constexpr double CONFIDENCE = 0.99;
    cv::Mat fundamentalMatrixDouble = cv::findFundamentalMat(points1, points2, mask, METHOD, reprojectionError, CONFIDENCE);
    if (fundamentalMatrixDouble.empty())
    {
        throw std::runtime_error("Failed to compute fundamental matrix");
    }
    return convertMat(fundamentalMatrixDouble);
}

cv::Matx33f estimateHomographyMatrix(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2)
{
    std::vector<uchar> mask;
    static const bool accurate = false;
    return ::estimateHomographyMatrix(points1, points2, mask, accurate);
}

std::vector<cv::Point2f> undistortPoints(const std::vector<cv::Point2f>& points, const cv::Matx33f& intrinsics, const cv::Mat& distortions)
{
    std::vector<cv::Point2f> undistortedPoints;
    cv::undistortPoints(points, undistortedPoints, intrinsics, distortions, cv::noArray(), intrinsics);
    return undistortedPoints;
}

class GRIC::Impl
{
public:
    Impl() = default;

    float calculateFcost(const std::vector<cv::Point2f>& points0,
                          const std::vector<cv::Point2f>& points1,
                          const cv::Matx33f& fundamentalMatrix,
                          float sigma)
    {
        const std::vector<float> residuals = sampsonDistanceSquaredF(fundamentalMatrix, points0, points1);
        return calculateCost(residuals, sigma, false);
    }

    float calculateHcost(const std::vector<cv::Point2f>& points0,
                          const std::vector<cv::Point2f>& points1,
                          const cv::Matx33f& homographyMatrix,
                          float sigma)
    {
        const std::vector<float> residuals = sampsonDistanceSquaredH(homographyMatrix, points0, points1);
        return calculateCost(residuals, sigma, true);
    }

private:

    float calculateCost(const std::vector<float>& residuals, float sigma, bool isHomography)
    {
        const int R = 4; // data dimension (image point pairs)
        const float invSigmaSquared = 1.0/(sigma*sigma);

        int K = 7; // number of parameters
        int D = 3; // dimension of the manifold
        if (isHomography)
        {
            K = 8;
            D = 2;
        }

        const float lam3RD = 2.0*(R-D);

        float sum = 0.0f;
        for (const float res : residuals)
        {
            sum += std::min(res*invSigmaSquared, lam3RD);
        }

        const int n = residuals.size();
        sum += n*D*log((float)R) + K*log((float)(R*n));

        return sum;
    }
};

class MotionEstimator::Impl
{
    const cv::Matx33f mCameraMatrix;
    const cv::Matx33f mCameraMatrixTransposed;

public:
    Impl(const cv::Matx33f& cameraMatrix) : mCameraMatrix(cameraMatrix), mCameraMatrixTransposed(cameraMatrix.t()) {}

    Pose estimate(const cv::Matx33f& fundamentalMatrix, const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2, const cv::Mat &mask)
    {
        const cv::Matx33f essentialMatrix = mCameraMatrixTransposed * fundamentalMatrix * mCameraMatrix;
        const auto [R1, R2, t] = decomposeEssentialMatrix(essentialMatrix);

        const auto normalizedPoints1 = normalizePoints(points1);
        const auto normalizedPoints2 = normalizePoints(points2);

        std::future<uint32_t> cheiralCheck1 = std::async(std::launch::async, &Impl::cheiralCheck, R1, t, std::cref(normalizedPoints1), std::cref(normalizedPoints2), std::cref(mask));
        std::future<uint32_t> cheiralCheck2 = std::async(std::launch::async, &Impl::cheiralCheck, R2, t, std::cref(normalizedPoints1), std::cref(normalizedPoints2), std::cref(mask));
        std::future<uint32_t> cheiralCheck3 = std::async(std::launch::async, &Impl::cheiralCheck, R1, -t, std::cref(normalizedPoints1), std::cref(normalizedPoints2), std::cref(mask));
        const auto goodness4 = cheiralCheck(R2, -t, normalizedPoints1, normalizedPoints2, mask);

        const uint32_t goodness1 = cheiralCheck1.get();
        const uint32_t goodness2 = cheiralCheck2.get();
        const uint32_t goodness3 = cheiralCheck3.get();

        if (goodness1 >= goodness2 && goodness1 >= goodness3 && goodness1 >= goodness4)
        {
            return {R1, t};
        }
        else if (goodness2 >= goodness1 && goodness2 >= goodness3 && goodness2 >= goodness4)
        {
            return {R2, t};
        }
        else if (goodness3 >= goodness1 && goodness3 >= goodness2 && goodness3 >= goodness4)
        {
            return {R1, -t};
        }
        else
        {
            return {R2, -t};
        }
    }

    std::vector<cv::Point2f> normalizePoints(const std::vector<cv::Point2f>& points) const
    {
        const float& fx = mCameraMatrix(0,0);
        const float& cx = mCameraMatrix(0,2);
        const float& fy = mCameraMatrix(1,1);
        const float& cy = mCameraMatrix(1,2);

        std::vector<cv::Point2f> normalizedPoints;
        normalizedPoints.reserve(points.size());
        for (const auto& point : points)
        {
            normalizedPoints.emplace_back((point.x - cx)/fx, (point.y - cy)/fy);
        }
        return normalizedPoints;
    }

    static std::tuple<cv::Matx33f, cv::Matx33f, cv::Matx31f> decomposeEssentialMatrix(const cv::Matx33f& essentialMatrix)
    {
        cv::Matx33f U, Vt;
        cv::Vec3f D;
        cv::SVD::compute(essentialMatrix, D, U, Vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        if (cv::determinant(U) < 0)
        {
            U *= -1.;
        }

        if (cv::determinant(Vt) < 0)
        {
            Vt *= -1.;
        }

        static const cv::Matx33f W(0, 1, 0, -1, 0, 0, 0, 0, 1);
        static const cv::Matx33f WTransposed = W.t();

        cv::Matx33f R1 = U * W * Vt;
        cv::Matx33f R2 = U * WTransposed * Vt;
        cv::Matx31f t = U.col(2) * 1.0;
        return {R1, R2, t / cv::norm(t)};
    }


    static uint32_t cheiralCheck(const cv::Matx33f& rotation,
                                 const cv::Matx31f& translation,
                                 const std::vector<cv::Point2f>& points1,
                                 const std::vector<cv::Point2f>& points2,
                                 const cv::Mat& mask)
    {
        static const cv::Matx34f projectionMatrix0 = cv::Matx34f::eye();
        const cv::Matx34f projectionMatrix1 = createProjectionMatrix(rotation, translation);
        cv::Mat Q;
        cv::triangulatePoints(projectionMatrix0, projectionMatrix1, points1, points2, Q);

        cv::Mat mask1 = Q.row(2).mul(Q.row(3)) >= 0;
        Q.row(0) /= Q.row(3);
        Q.row(1) /= Q.row(3);
        Q.row(2) /= Q.row(3);
        Q.row(3) /= Q.row(3);

        // Filter distant points because their depth may flip sign near infinity.
        constexpr float DIST = 50.0;
        mask1 = (Q.row(2) <= DIST) & mask1;
        Q = projectionMatrix1 * Q;
        mask1 = (Q.row(2) >= 0) & mask1;
        mask1 = (Q.row(2) <= DIST) & mask1;

        mask1 = mask1.t();

        cv::bitwise_and(mask, mask1, mask1);

        return cv::countNonZero(mask1);
    }
};


class ScaleEstimator::Impl
{
    CameraConfig mCamConfig;
    cv::Matx33f mKinv;
    float mMaxPossibleScale;
    KeypointSampler mKeypointSampler;
    static constexpr uint32_t NUM_ROAD_POINTS_TO_USE = 100;
    static constexpr float JUMP_UP_SCALE_RATIO = 2.5f;
    static constexpr float JUMP_DOWN_SCALE_RATIO = 0.4f;

    std::vector<cv::Point2f> mPrevRoadPoints;
    std::vector<cv::Point2f> mCurrRoadPoints;
    std::unique_ptr<KeypointTracker>& mKeyPointTracker;
    std::optional<float> mPreviousScale = std::nullopt;

    struct Scale
    {
        float value;
        float weight;
    };

public:

    Impl(const CameraConfig& camConfig, float maxVehicleSpeed, std::unique_ptr<KeypointTracker>& keyPointTracker)
        : mCamConfig(camConfig), mKeyPointTracker(keyPointTracker)
    {
        if (camConfig.height <= 0)
        {
            throw std::invalid_argument("Camera height is expected to be larger than zero");
        }
        if (maxVehicleSpeed <= 0)
        {
            throw std::invalid_argument("Max vehicle speed is expected to be larger than zero");
        }
        mKinv = camConfig.intrinsics.inv();
        mMaxPossibleScale = (maxVehicleSpeed * 1000 / 3600) / camConfig.fps;
    }

    void initialize(const std::vector<Keypoint>& points, const cv::Mat& roadMask)
    {
        std::vector<cv::Point2f> rawRoadPoints = mKeypointSampler.run(points, roadMask, NUM_ROAD_POINTS_TO_USE);
        if (mCamConfig.distortions.has_value())
        {
            mPrevRoadPoints = undistortPoints(rawRoadPoints, mCamConfig.intrinsics, mCamConfig.distortions.value());
        }
        else
        {
            mPrevRoadPoints = std::move(rawRoadPoints);
        }
    }

    void setCurrentFrameData(const std::vector<Keypoint>& points, const cv::Mat& roadMask)
    {
        std::vector<cv::Point2f> rawRoadPoints = mKeypointSampler.run(points, roadMask, NUM_ROAD_POINTS_TO_USE);
        if (mCamConfig.distortions.has_value())
        {
            mCurrRoadPoints = undistortPoints(rawRoadPoints, mCamConfig.intrinsics, mCamConfig.distortions.value());
        }
        else
        {
            mCurrRoadPoints = std::move(rawRoadPoints);
        }
    }

    std::optional<float> estimate(const cv::Matx33f& rotation, const cv::Matx31f& translation)
    {
        const std::optional<Scale> scaleForward = estimateForward(rotation, translation);
        const std::optional<Scale> scaleBackward = estimateBackward(rotation, translation);
        auto scale = fuseScales(scaleForward, scaleBackward);

        if (!scale.has_value())
        {
            scale = mPreviousScale;
        }

        mPreviousScale = scale;

        return scale;
    }

    void step()
    {
        mPrevRoadPoints = std::move(mCurrRoadPoints);
    }

private:

    std::optional<Scale> estimateForward(const cv::Matx33f& rotation, const cv::Matx31f& translation)
    {
        try
        {
            const auto trackedRoadPoints = mKeyPointTracker->trackForward(mPrevRoadPoints);
            return estimateScale(mPrevRoadPoints, trackedRoadPoints, rotation, translation);
        }
        catch (...)
        {
            return std::nullopt;
        }
    }

    std::optional<Scale> estimateBackward(const cv::Matx33f& rotation, const cv::Matx31f& translation)
    {
        try
        {
            const auto trackedRoadPoints = mKeyPointTracker->trackBackward(mCurrRoadPoints);
            cv::Matx33f backwardRotation = rotation.t();
            cv::Matx31f backwardTranslation = -backwardRotation * translation;
            return estimateScale(mCurrRoadPoints, trackedRoadPoints, backwardRotation, backwardTranslation);
        }
        catch (...)
        {
            return std::nullopt;
        }
    }

    std::optional<Scale> estimateScale(const std::vector<cv::Point2f>& prevRoadPoints,
                                       const std::vector<cv::Point2f>& currRoadPoints,
                                       const cv::Matx33f& rotation,
                                       const cv::Matx31f& translation) const
    {
        static const bool accurateHomo = true;
        cv::Matx33f Homography;
        std::vector<uchar> mask;
        try
        {
            Homography = ::estimateHomographyMatrix(prevRoadPoints, currRoadPoints, mask, accurateHomo);
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

        if (scale > mMaxPossibleScale)
        {
            return std::nullopt;
        }

        if (mPreviousScale.has_value())
        {
            const float prevScale = mPreviousScale.value();
            if (scale > JUMP_UP_SCALE_RATIO*prevScale || scale < JUMP_DOWN_SCALE_RATIO*prevScale)
            {
                return std::nullopt;
            }
        }

        Scale output;
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

    static std::optional<float> fuseScales(const std::optional<Scale>& scaleForward, const std::optional<Scale>& scaleBackward)
    {
        if (scaleForward.has_value() && scaleBackward.has_value())
        {
            const Scale scaleFwd = scaleForward.value();
            const Scale scaleBwd = scaleBackward.value();
            const float lnScaleFused = (scaleFwd.weight * std::log(scaleFwd.value) + scaleBwd.weight * std::log(scaleBwd.value)) / (scaleFwd.weight + scaleBwd.weight);
            return std::exp(lnScaleFused);
        }
        else if (scaleForward.has_value() && !scaleBackward.has_value())
        {
            return scaleForward.value().value;
        }
        else if (!scaleForward.has_value() && scaleBackward.has_value())
        {
            return scaleBackward.value().value;
        }
        else
        {
            return std::nullopt;
        }
    }
};


GRIC::GRIC()
{
    mImpl = std::make_shared<Impl>();
}

float GRIC::calculateFcost(const std::vector<cv::Point2f>& points0,
                            const std::vector<cv::Point2f>& points1,
                            const cv::Matx33f& fundamentalMatrix,
                            float sigma)
{
    return mImpl->calculateFcost(points0, points1, fundamentalMatrix, sigma);
}

float GRIC::calculateHcost(const std::vector<cv::Point2f>& points0,
                            const std::vector<cv::Point2f>& points1,
                            const cv::Matx33f& homographyMatrix,
                            float sigma)
{
    return mImpl->calculateHcost(points0, points1, homographyMatrix, sigma);
}

MotionEstimator::MotionEstimator(const cv::Matx33f& cameraMatrix)
{
    mImpl = std::make_shared<Impl>(cameraMatrix);
}

Pose MotionEstimator::estimate(const cv::Matx33f& fundamentalMatrix, const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2, const cv::Mat &mask)
{
    return mImpl->estimate(fundamentalMatrix, points1, points2, mask);
}

ScaleEstimator::ScaleEstimator(const CameraConfig& camConfig, float maxVehicleSpeed, std::unique_ptr<KeypointTracker>& keyPointTracker)
{
    mImpl = std::make_shared<Impl>(camConfig, maxVehicleSpeed, keyPointTracker);
}

void ScaleEstimator::initialize(const std::vector<Keypoint>& points, const cv::Mat& roadMask)
{
    mImpl->initialize(points, roadMask);
}

void ScaleEstimator::setCurrentFrameData(const std::vector<Keypoint>& points, const cv::Mat& roadMask)
{
    mImpl->setCurrentFrameData(points, roadMask);
}

std::optional<float> ScaleEstimator::estimate(const cv::Matx33f& rotation, const cv::Matx31f& translation)
{
    return mImpl->estimate(rotation, translation);
}

void ScaleEstimator::step()
{
    mImpl->step();
}

}
