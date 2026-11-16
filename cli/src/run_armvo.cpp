#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core/persistence.hpp>

#include "cxxopts.hpp"

#include <armvo/ARM_VO.hpp> 
#include <armvo/tools/Plotter.hpp>
#include <armvo/tools/FileTools.hpp>
#include <armvo/tools/ErrorEstimator.hpp>

namespace fs = std::filesystem;

class CLI
{
    struct Stats
    {
        std::optional<double> minFps;
        std::optional<double> maxFps;
        std::optional<double> medianFps;
        std::optional<double> meanFps;
        std::optional<double> throughputFps;
        std::optional<double> translationError;
        std::optional<double> rotationError;
    };

    static constexpr float kPi = 3.14159265358979323846f;
    static constexpr int32_t kWarmupFrames = 5;

    std::vector<std::chrono::microseconds> mFrameProcessingTimes;
    cv::Mat mFrame;
    std::vector<armvo::Pose> mEstimatedPoses;
    std::vector<armvo::Pose> mGroundTruthPoses;
    armvo::tools::ErrorEstimator mErrorEstimator;
    int32_t mCurrentFrameIndex = -1;
    armvo::tools::Plotter mPlotter;

public:

    CLI() = default;

    int run(int argc, char* argv[])
    {
        cxxopts::Options options(argv[0], "Runs ARM-VO on a sequence of frames");

        options.add_options()
            ("i,image_folder", "Path to the folder that contains the sequence of frames", cxxopts::value<std::string>())
            ("c,config", "Path to .yaml config file", cxxopts::value<std::string>())
            ("g,gt_poses", "Path to .txt file that stores the ground-truth poses (optional)", cxxopts::value<std::string>())
            ("d,display", "Enable/Disable the camera view and trajectory", cxxopts::value<bool>()->default_value("true"))
            ("s,stats_output_file", "Write summary statistics to a JSON file", cxxopts::value<std::string>())
            ("help", "Print help");

        auto result = options.parse(argc, argv);

        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            return 0;
        }

        if (!result.count("image_folder") || !result.count("config"))
        {
            std::cerr << "Missing arguments! Run with --help to see the required arguments" << std::endl;
            return -1;
        }

        std::string imageDir = fs::canonical(result["image_folder"].as<std::string>());
        std::string configFilepath = fs::canonical(result["config"].as<std::string>());

        if (result.count("gt_poses"))
        {
            std::string gtPosesFilepath = fs::canonical(result["gt_poses"].as<std::string>());
            mGroundTruthPoses = armvo::tools::FileTools::loadKittiGroundTruthPoses(gtPosesFilepath);
        }

        const auto filepaths = armvo::tools::FileTools::listFramesInDir(imageDir);
        if (filepaths.empty())
        {
            std::cout << "No images were found in " << imageDir << std::endl;
            return -1;
        }

        if (!mGroundTruthPoses.empty() && mGroundTruthPoses.size() != filepaths.size())
        {
            std::cerr << "Number of images mismatch number of ground truth poses" << std::endl;
            return -1;
        }

        auto config = armvo::ArmVoConfig::load(configFilepath);
        armvo::ArmVo VO(config);

        const std::string statsPath = result.count("stats_output_file") ? result["stats_output_file"].as<std::string>() : "";
        mEstimatedPoses.reserve(filepaths.size());
        mFrameProcessingTimes.reserve(filepaths.size());
        bool isDisplayEnabled = result["display"].as<bool>();
        for (const auto& filepath : filepaths)
        {
            std::cout << "\rProcessing frame " << mCurrentFrameIndex+1 << "/" << filepaths.size()-1 << " " << std::flush;
            const bool status = processFrame(VO, filepath, config.camera.pixelFormat);
            if (!status)
            {
                return -1;
            }
            if (isDisplayEnabled)
            {
                plot();
            }
        }
        std::cout << "\nFinished" << std::endl;

        const auto stats = calculateStats();
        printStats(stats);
        if (!statsPath.empty())
        {
            writeStats(statsPath, stats);
        }

        return 0;
    }

private:

    bool processFrame(armvo::ArmVo& VO, const std::string& filepath, armvo::PixelFormat pixelFormat)
    {
        mFrame = armvo::tools::FileTools::loadImage(filepath, pixelFormat);
        mCurrentFrameIndex++;
        bool logNewLine = false;

        std::chrono::microseconds processingTime{0};
        armvo::Pose estimatedPose;

        if (!VO.isInitialized())
        {
            const auto t0 = std::chrono::steady_clock::now();
            const auto status = VO.initialize(mFrame, estimatedPose);
            processingTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0);
            switch (status)
            {
                case armvo::Status::SUCCESS:
                    std::cout << "Initialization done successfully. ";
                    logNewLine = true;
                    break;
                case armvo::Status::INVALID_FRAME:
                    std::cerr << "Initialization failed (invalid frame)" << std::endl;
                    return false;
                case armvo::Status::NOT_ENOUGH_KEYPOINTS:
                    std::cerr << "Initialization failed (not enough keypoints were detected)" << std::endl;
                    return false;
                case armvo::Status::NOT_INITIALIZED:
                case armvo::Status::TRACK_LOST:
                case armvo::Status::FRAME_SKIPPED:
                case armvo::Status::SCALE_ESTIMATION_FAILURE:
                    std::cerr << "Initialization failed with unexpected status" << std::endl;
                    return false;
            }
        }
        else
        {
            const auto t0 = std::chrono::steady_clock::now();
            const auto status = VO.update(mFrame, estimatedPose);
            processingTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0);
            switch (status)
            {
                case armvo::Status::SUCCESS:
                    break;
                case armvo::Status::INVALID_FRAME:
                    std::cerr << "Invalid frame" << std::endl;
                    return false;
                case armvo::Status::NOT_INITIALIZED:
                    std::cerr << "ARM-VO is not initialized" << std::endl;
                    return false;
                case armvo::Status::NOT_ENOUGH_KEYPOINTS:
                    std::cerr << "Not enough keypoints were detected in the previous frame" << std::endl;
                    return false;
                case armvo::Status::TRACK_LOST:
                    std::cerr << "Track lost" << std::endl;
                    return false;
                case armvo::Status::SCALE_ESTIMATION_FAILURE:
                    std::cerr << "Failed to estimate scale" << std::endl;
                    return false;
                case armvo::Status::FRAME_SKIPPED:
                    std::cout << "Frame skipped. ";
                    logNewLine = true;
                    break;
            }
        }

        mEstimatedPoses.push_back(estimatedPose);
        if (mCurrentFrameIndex > kWarmupFrames && processingTime.count() > 0)
        {
            mFrameProcessingTimes.push_back(processingTime);
        }

        std::cout << "FPS: ";
        if (processingTime.count() > 0)
        {
            std::cout << std::fixed << std::setprecision(2) << 1'000'000.0 / processingTime.count();
        }
        else
        {
            std::cout << "N/A";
        }
        if (logNewLine)
        {
            std::cout << "\n";
        }

        return true;
    }

    void plot()
    {
        if (!mGroundTruthPoses.empty())
        {
            mPlotter.update(mFrame, mEstimatedPoses.back(), mGroundTruthPoses[mCurrentFrameIndex]);
        }
        else
        {
            mPlotter.update(mFrame, mEstimatedPoses.back());
        }
        mPlotter.plot();
    }

    Stats calculateStats()
    {
        Stats stats;
        if (!mFrameProcessingTimes.empty())
        {
            std::vector<double> fps;
            fps.reserve(mFrameProcessingTimes.size());
            for (const auto processingTime : mFrameProcessingTimes)
            {
                fps.push_back(1'000'000.0 / processingTime.count());
            }
            std::sort(fps.begin(), fps.end());
            stats.minFps = fps.front();
            stats.maxFps = fps.back();
            const size_t mid = fps.size() / 2;
            stats.medianFps = fps.size() % 2 ? fps[mid] : (fps[mid - 1] + fps[mid]) / 2.0;
            stats.meanFps = std::accumulate(fps.begin(), fps.end(), 0.0) / fps.size();
            const auto totalTime = std::accumulate(mFrameProcessingTimes.begin(), mFrameProcessingTimes.end(),
                                                   std::chrono::microseconds{0});
            stats.throughputFps = mFrameProcessingTimes.size() * 1'000'000.0 / totalTime.count();
        }

        if (!mGroundTruthPoses.empty())
        {
            const auto errors = mErrorEstimator.estimate(mEstimatedPoses, mGroundTruthPoses);
            if (!errors.empty())
            {
                double translation = 0.0, rotation = 0.0;
                for (const auto& err : errors)
                {
                    translation += err.translationErr;
                    rotation += err.rotationErr;
                }
                stats.translationError = translation * 100.0 / errors.size();
                stats.rotationError = rotation * 180.0 / (errors.size() * kPi);
            }
        }

        return stats;
    }

    void printStats(const Stats& stats)
    {
        const auto printValue = [](const char* label, const std::optional<double>& value, const char* unit = "")
        {
            std::cout << label;
            if (value)
            {
                std::cout << *value << unit;
            }
            else
            {
                std::cout << "N/A";
            }
            std::cout << '\n';
        };
        std::cout << "\n=======================================\n"
                  << std::fixed << std::setprecision(2);
        printValue("Min FPS: ", stats.minFps);
        printValue("Max FPS: ", stats.maxFps);
        printValue("Median FPS: ", stats.medianFps);
        printValue("Average FPS: ", stats.meanFps);
        printValue("Throughput FPS: ", stats.throughputFps);
        std::cout << std::setprecision(4);
        printValue("Average Translation Error: ", stats.translationError, "%");
        printValue("Average Rotation Error: ", stats.rotationError, " [deg/m]");
        std::cout << std::flush;
    }

    void writeStats(const std::string& statsPath, const Stats& stats)
    {
        cv::FileStorage out(statsPath, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
        if (!out.isOpened())
        {
            throw std::runtime_error("Cannot open stats output file: " + statsPath);
        }

        const auto writeValue = [&out](const char* key, const std::optional<double>& value)
        {
            if (value && std::isfinite(*value))
            {
                out << key << *value;
            }
        };
        writeValue("fps_min", stats.minFps);
        writeValue("fps_max", stats.maxFps);
        writeValue("fps_median", stats.medianFps);
        writeValue("fps_mean", stats.meanFps);
        writeValue("fps_throughput", stats.throughputFps);
        writeValue("translation_error_percent", stats.translationError);
        writeValue("rotation_error_deg_per_m", stats.rotationError);
    }

};

int main(int argc, char* argv[])
{
    CLI app;
    try
    {
        return app.run(argc, argv);
    }
    catch (std::exception& ex)
    {
        std::cerr << "Something went wrong" << std::endl;
        std::cerr << ex.what() << std::endl;
    }
    return -1;
}
