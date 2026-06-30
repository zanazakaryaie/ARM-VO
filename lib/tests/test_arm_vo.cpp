#include <catch2/catch.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "ARM_VO.hpp"

namespace
{

armvo::ArmVoConfig makeConfig(armvo::PixelFormat pixelFormat = armvo::PixelFormat::Gray)
{
    armvo::ArmVoConfig config;
    config.camera.intrinsics = cv::Matx33f(700.0f, 0.0f, 320.0f,
                                           0.0f, 710.0f, 240.0f,
                                           0.0f, 0.0f, 1.0f);
    config.camera.pixelFormat = pixelFormat;
    config.camera.fps = 10.0f;
    config.camera.height = 1.6f;

    config.keypointDetector.maxNumberOfPoints = 500;
    config.keypointDetector.responseThreshold = 10;
    config.keypointDetector.numberOfImageGridRows = 6;
    config.keypointDetector.numberOfImageGridCols = 12;

    config.keypointTracker.windowSize = 21;
    config.maxVehicleSpeed = 90.0f;

    return config;
}

cv::Mat makeBlankGrayFrame()
{
    return cv::Mat(480, 640, CV_8UC1, cv::Scalar(0));
}

cv::Mat makeHighTextureGrayFrame()
{
    cv::Mat image(480, 640, CV_8UC1, cv::Scalar(0));

    constexpr int margin = 24;
    constexpr int step = 24;
    constexpr int boxSize = 10;

    for (int y = margin; y <= image.rows - margin - boxSize; y += step)
    {
        for (int x = margin; x <= image.cols - margin - boxSize; x += step)
        {
            const uint8_t value = static_cast<uint8_t>(180 + ((x + 3 * y) % 75));
            cv::rectangle(image, cv::Rect(x, y, boxSize, boxSize), cv::Scalar(value), cv::FILLED);
        }
    }

    cv::GaussianBlur(image, image, cv::Size(5, 5), 0.0);
    return image;
}

void checkIdentityPose(const armvo::Pose& pose)
{
    CHECK(cv::norm(pose.rotation - cv::Matx33f::eye()) < 1e-6f);
    CHECK(cv::norm(pose.translation) < 1e-6f);
}

} // namespace

TEST_CASE("ArmVo validates grid configuration")
{
    SECTION("grid rows must be nonzero")
    {
        armvo::ArmVoConfig config = makeConfig();
        config.keypointDetector.numberOfImageGridRows = 0;

        CHECK_THROWS(armvo::ArmVo(config));
    }

    SECTION("grid columns must be nonzero")
    {
        armvo::ArmVoConfig config = makeConfig();
        config.keypointDetector.numberOfImageGridCols = 0;

        CHECK_THROWS(armvo::ArmVo(config));
    }
}

TEST_CASE("ArmVo reports initialization and update preconditions")
{
    armvo::ArmVo vo(makeConfig());
    armvo::Pose pose;

    CHECK_FALSE(vo.isInitialized());
    CHECK(vo.update(makeBlankGrayFrame(), pose) == armvo::Status::NOT_INITIALIZED);

    CHECK(vo.initialize(cv::Mat(), pose) == armvo::Status::INVALID_FRAME);
    CHECK_FALSE(vo.isInitialized());

    const cv::Mat wrongPixelFormat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    CHECK(vo.initialize(wrongPixelFormat, pose) == armvo::Status::INVALID_FRAME);
    CHECK_FALSE(vo.isInitialized());

    CHECK(vo.initialize(makeBlankGrayFrame(), pose) == armvo::Status::NOT_ENOUGH_KEYPOINTS);
    CHECK_FALSE(vo.isInitialized());
}

TEST_CASE("ArmVo initializes pose from a valid textured frame")
{
    armvo::ArmVo vo(makeConfig());
    armvo::Pose pose;

    const cv::Mat frame = makeHighTextureGrayFrame();
    REQUIRE(frame.isContinuous());

    REQUIRE(vo.initialize(frame, pose) == armvo::Status::SUCCESS);

    CHECK(vo.isInitialized());
    checkIdentityPose(pose);

    CHECK(vo.initialize(cv::Mat(), pose) == armvo::Status::INVALID_FRAME);
    CHECK_FALSE(vo.isInitialized());
}
