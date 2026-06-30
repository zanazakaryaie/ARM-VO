#include <catch2/catch.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <cstdint>

namespace fs = std::filesystem;

#include "FileTools.hpp"

namespace
{

class TempDir
{
public:
    TempDir()
    {
        const auto base = fs::temp_directory_path();

        const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::mt19937_64 rng(static_cast<uint64_t>(now));
        std::uniform_int_distribution<uint64_t> dist;

        mDir = base / ("filetools_test_" + std::to_string(now) + "_" + std::to_string(dist(rng)));
        fs::create_directories(mDir);
    }

    ~TempDir()
    {
        std::error_code errCode;
        fs::remove_all(mDir, errCode);
    }

    const fs::path& path() const
    {
        return mDir;
    }

private:
    fs::path mDir;
};

void writeTextFile(const fs::path& path, const std::string& content)
{
    std::ofstream out(path);
    REQUIRE(out.is_open());
    out << content;
    out.close();
}

void writeBgrImage(const fs::path& path, uint32_t width, uint32_t height, const cv::Vec3b& bgr)
{
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(bgr[0], bgr[1], bgr[2]));
    REQUIRE(cv::imwrite(path.string(), img));
}

void checkMatx33Approx(const cv::Matx33f& src1, const cv::Matx33f& src2, float eps = 1e-6f)
{
    for (uint8_t r = 0; r < 3; r++)
    {
        for (uint8_t c = 0; c < 3; c++)
        {
            auto diff = std::abs(src1(r, c) - src2(r, c));
            CHECK(diff < eps);
        }
    }
}

void checkMatx31Approx(const cv::Matx31f& src1, const cv::Matx31f& src2, float eps = 1e-6f)
{
    for (uint8_t r = 0; r < 3; r++)
    {
        auto diff = std::abs(src1(r, 0) - src2(r, 0));
        CHECK(diff < eps);
    }
}

}


TEST_CASE("FileTools::listFramesInDir throws if dir does not exist")
{
    std::string missingDir;
    {
        TempDir tmp;
        missingDir = tmp.path().string();
    }
    REQUIRE_THROWS_AS(armvo::tools::FileTools::listFramesInDir(missingDir), std::invalid_argument);
}

TEST_CASE("FileTools::listFramesInDir lists only png/jpg and sorts ascending by filename")
{
    TempDir tmp;
    const fs::path frames = tmp.path() / "frames";
    fs::create_directories(frames);

    // Create valid images
    writeBgrImage(frames / "0003.png", 2, 2, {10, 20, 30});
    writeBgrImage(frames / "0001.jpg", 2, 2, {10, 20, 30});
    writeBgrImage(frames / "0002.png", 2, 2, {10, 20, 30});
    writeBgrImage(frames / "0004.jpeg", 2, 2, {10, 20, 30});

    // Non-frame file
    writeTextFile(frames / "0000.txt", "not an image");

    const auto result = armvo::tools::FileTools::listFramesInDir(frames.string());

    REQUIRE(result.size() == 4);
    CHECK(result[0] == (frames / "0001.jpg").string());
    CHECK(result[1] == (frames / "0002.png").string());
    CHECK(result[2] == (frames / "0003.png").string());
    CHECK(result[3] == (frames / "0004.jpeg").string());
}

TEST_CASE("FileTools::loadImage throws if file does not exist")
{
    TempDir tmp;
    const fs::path missing = tmp.path() / "nope.png";
    REQUIRE_THROWS_AS(armvo::tools::FileTools::loadImage(missing.string(), armvo::PixelFormat::BGR), std::runtime_error);
}

TEST_CASE("FileTools::loadImage(BGR) loads 3-channel image and preserves BGR ordering")
{
    TempDir tmp;
    const fs::path imgPath = tmp.path() / "img.png";

    const cv::Vec3b bgr{11, 22, 33};
    writeBgrImage(imgPath, 2, 2, bgr);

    const cv::Mat img = armvo::tools::FileTools::loadImage(imgPath.string(), armvo::PixelFormat::BGR);

    REQUIRE_FALSE(img.empty());
    REQUIRE(img.channels() == 3);
    REQUIRE(img.type() == CV_8UC3);

    const cv::Vec3b px = img.at<cv::Vec3b>(0, 0);
    CHECK(px[0] == 11);
    CHECK(px[1] == 22);
    CHECK(px[2] == 33);
}

TEST_CASE("FileTools::loadImage(RGB) loads 3-channel image and swaps BGR->RGB")
{
    TempDir tmp;
    const fs::path imgPath = tmp.path() / "img.png";

    const cv::Vec3b bgr{11, 22, 33};
    writeBgrImage(imgPath, 2, 2, bgr);

    const cv::Mat img = armvo::tools::FileTools::loadImage(imgPath.string(), armvo::PixelFormat::RGB);

    REQUIRE_FALSE(img.empty());
    REQUIRE(img.channels() == 3);
    REQUIRE(img.type() == CV_8UC3);

    // Expect RGB ordering: [R,G,B] = [33,22,11]
    const cv::Vec3b px = img.at<cv::Vec3b>(0, 0);
    CHECK(px[0] == 33);
    CHECK(px[1] == 22);
    CHECK(px[2] == 11);
}

TEST_CASE("FileTools::loadImage(Gray) loads single-channel and matches OpenCV BGR->Gray conversion")
{
    TempDir tmp;
    const fs::path imgPath = tmp.path() / "img.png";

    const cv::Vec3b bgr{11, 22, 33};
    writeBgrImage(imgPath, 2, 2, bgr);

    const cv::Mat gray = armvo::tools::FileTools::loadImage(imgPath.string(), armvo::PixelFormat::Gray);

    REQUIRE_FALSE(gray.empty());
    REQUIRE(gray.channels() == 1);
    REQUIRE(gray.type() == CV_8UC1);

    // Compute expected gray using OpenCV from the same saved image
    cv::Mat bgrImg = cv::imread(imgPath.string(), cv::IMREAD_COLOR);
    REQUIRE_FALSE(bgrImg.empty());
    cv::Mat expectedGray;
    cv::cvtColor(bgrImg, expectedGray, cv::COLOR_BGR2GRAY);

    const uint8_t got = gray.at<uint8_t>(0, 0);
    const uint8_t exp = expectedGray.at<uint8_t>(0, 0);
    CHECK(got == exp);
}

TEST_CASE("FileTools::loadKittiGroundTruthPoses throws if file does not exist")
{
    std::string missingGtFile;
    {
        TempDir tmp;
        missingGtFile = (tmp.path() / "gt.txt").string();
    }
    REQUIRE_THROWS(armvo::tools::FileTools::loadKittiGroundTruthPoses(missingGtFile));
}

TEST_CASE("FileTools::loadKittiGroundTruthPoses parses rotations/translations correctly")
{
    TempDir tmp;
    const fs::path gt = tmp.path() / "gt.txt";

    // Pose 0: identity rotation, t=(1,2,3)
    // Pose 1: rotation = [0 -1 0; 1 0 0; 0 0 1], t=(4,5,6)
    const std::string content =
        "1 0 0 1   0 1 0 2   0 0 1 3\n"
        "0 -1 0 4  1 0 0 5   0 0 1 6\n";
    writeTextFile(gt, content);

    const auto poses = armvo::tools::FileTools::loadKittiGroundTruthPoses(gt.string());
    REQUIRE(poses.size() == 2);

    const armvo::Pose& p0 = poses[0];
    const armvo::Pose& p1 = poses[1];

    const cv::Matx33f R0_expected(
        1,0,0,
        0,1,0,
        0,0,1
    );
    const cv::Matx31f t0_expected(1, 2, 3);

    const cv::Matx33f R1_expected(
        0,-1,0,
        1, 0,0,
        0, 0,1
    );
    const cv::Matx31f t1_expected(4, 5, 6);

    checkMatx33Approx(p0.rotation, R0_expected);
    checkMatx31Approx(p0.translation, t0_expected);

    checkMatx33Approx(p1.rotation, R1_expected);
    checkMatx31Approx(p1.translation, t1_expected);
}