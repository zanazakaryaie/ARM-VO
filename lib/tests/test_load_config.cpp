#include <fstream>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>

#include <catch2/catch.hpp>
#include <opencv2/core.hpp>

#include "Types.hpp"

namespace fs = std::filesystem;

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

        mDir = base / ("armvo_config_test_" + std::to_string(now) + "_" + std::to_string(dist(rng)));
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

std::string makeSampleConfig(const std::string& pixelFormat, const std::string& cameraExtra = "")
{
    const std::string cameraExtraBlock = cameraExtra.empty() ? "" : cameraExtra + "\n";
    return R"(%YAML:1.0

Vehicle:
  max_speed: 80

Camera:
  fx: 718.856
  fy: 718.856
  cx: 607.1928
  cy: 185.2157
  fps: 15
  height: 1.6
  pixel_format: ")" + pixelFormat + R"("
)" + cameraExtraBlock + R"(

Keypoint_Detector:
  num_points: 300
  response_threshold: 12
  num_image_grid_rows: 8
  num_image_grid_cols: 10

Keypoint_Tracker:
  window_size: 21
)";
}

void writeConfigFile(const fs::path& filepath, const std::string& content)
{
    std::ofstream file(filepath);
    REQUIRE(file.is_open());
    file << content;
}

}


TEST_CASE("ArmVoConfig")
{
    SECTION("load throws when file is wrong")
    {
        REQUIRE_THROWS(armvo::ArmVoConfig::load("foo.yaml"));
    }

    SECTION("load works fine when file is correct")
    {
        TempDir tmp;
        const auto configPath = tmp.path() / "bar.yaml";
        writeConfigFile(configPath, makeSampleConfig("gray"));

        armvo::ArmVoConfig config = armvo::ArmVoConfig::load(configPath.string());

        CHECK(config.maxVehicleSpeed == 80);

        CHECK(config.camera.fps == 15);
        CHECK(config.camera.height == 1.6f);
        CHECK(config.camera.distortions == std::nullopt);
        CHECK(config.camera.pixelFormat == armvo::PixelFormat::Gray);

        cv::Matx33f expectedIntrinsics(718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1);
        cv::Mat diff;
        cv::absdiff(config.camera.intrinsics, expectedIntrinsics, diff);
        CHECK(cv::norm(diff) < 1e-6);

        CHECK(config.keypointDetector.maxNumberOfPoints == 300);
        CHECK(config.keypointDetector.responseThreshold == 12);
        CHECK(config.keypointDetector.numberOfImageGridRows == 8);
        CHECK(config.keypointDetector.numberOfImageGridCols == 10);

        CHECK(config.keypointTracker.windowSize == 21);
    }

    SECTION("load parses bgr pixel format")
    {
        TempDir tmp;
        const auto configPath = tmp.path() / "bgr.yaml";
        writeConfigFile(configPath, makeSampleConfig("bgr"));

        const armvo::ArmVoConfig config = armvo::ArmVoConfig::load(configPath.string());

        CHECK(config.camera.pixelFormat == armvo::PixelFormat::BGR);
    }

    SECTION("load parses rgb pixel format")
    {
        TempDir tmp;
        const auto configPath = tmp.path() / "rgb.yaml";
        writeConfigFile(configPath, makeSampleConfig("rgb"));

        const armvo::ArmVoConfig config = armvo::ArmVoConfig::load(configPath.string());

        CHECK(config.camera.pixelFormat == armvo::PixelFormat::RGB);
    }

    SECTION("load throws when pixel format is unsupported")
    {
        TempDir tmp;
        const auto configPath = tmp.path() / "invalid_pixel_format.yaml";
        writeConfigFile(configPath, makeSampleConfig("mono"));

        REQUIRE_THROWS(armvo::ArmVoConfig::load(configPath.string()));
    }

    SECTION("load parses optional camera distortions")
    {
        TempDir tmp;
        const auto configPath = tmp.path() / "distortions.yaml";
        writeConfigFile(configPath, makeSampleConfig("gray", "  distortions: [0.1, -0.2, 0.003, 0.004, 0.05]"));

        const armvo::ArmVoConfig config = armvo::ArmVoConfig::load(configPath.string());

        REQUIRE(config.camera.distortions.has_value());

        const cv::Mat distortions = config.camera.distortions.value();
        REQUIRE(distortions.total() == 5);

        CHECK(distortions.at<double>(0) == Approx(0.1));
        CHECK(distortions.at<double>(1) == Approx(-0.2));
        CHECK(distortions.at<double>(2) == Approx(0.003));
        CHECK(distortions.at<double>(3) == Approx(0.004));
        CHECK(distortions.at<double>(4) == Approx(0.05));
    }

    SECTION("load throws when Camera section is missing")
    {
        TempDir tmp;
        auto filepath = tmp.path() / "missing_camera.yaml";
        writeConfigFile(filepath, R"(%YAML:1.0

Vehicle:
  max_speed: 80

Keypoint_Detector:
  num_points: 300
  response_threshold: 12
  num_image_grid_rows: 8
  num_image_grid_cols: 10

Keypoint_Tracker:
  window_size: 21
)");
        REQUIRE_THROWS(armvo::ArmVoConfig::load(filepath.string()));
    }

    SECTION("load throws when Keypoint_Detector section is missing")
    {
        TempDir tmp;
        auto filepath = tmp.path() / "missing_detector.yaml";
        writeConfigFile(filepath, R"(%YAML:1.0

Vehicle:
  max_speed: 80

Camera:
  fx: 718.856
  fy: 718.856
  cx: 607.1928
  cy: 185.2157
  fps: 15
  height: 1.6
  pixel_format: "gray"

Keypoint_Tracker:
  window_size: 21
)");
        REQUIRE_THROWS(armvo::ArmVoConfig::load(filepath.string()));
    }

    SECTION("load throws when Keypoint_Tracker section is missing")
    {
        TempDir tmp;
        auto filepath = tmp.path() / "missing_tracker.yaml";
        writeConfigFile(filepath, R"(%YAML:1.0

Vehicle:
  max_speed: 80

Camera:
  fx: 718.856
  fy: 718.856
  cx: 607.1928
  cy: 185.2157
  fps: 15
  height: 1.6
  pixel_format: "gray"

Keypoint_Detector:
  num_points: 300
  response_threshold: 12
  num_image_grid_rows: 8
  num_image_grid_cols: 10
)");
        REQUIRE_THROWS(armvo::ArmVoConfig::load(filepath.string()));
    }

    SECTION("load throws when Vehicle section is missing")
    {
        TempDir tmp;
        auto filepath = tmp.path() / "missing_vehicle.yaml";
        writeConfigFile(filepath, R"(%YAML:1.0

Camera:
  fx: 718.856
  fy: 718.856
  cx: 607.1928
  cy: 185.2157
  fps: 15
  height: 1.6
  pixel_format: "gray"

Keypoint_Detector:
  num_points: 300
  response_threshold: 12
  num_image_grid_rows: 8
  num_image_grid_cols: 10

Keypoint_Tracker:
  window_size: 21
)");
        REQUIRE_THROWS(armvo::ArmVoConfig::load(filepath.string()));
    }
}
