#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "cxxopts.hpp"

#include <armvo/ARM_VO.hpp> 
#include <armvo/tools/Plotter.hpp>
#include <armvo/tools/FileTools.hpp>
#include <armvo/tools/ErrorEstimator.hpp>

namespace fs = std::filesystem;

class CLI
{
    static constexpr float kPi = 3.14159265358979323846f;

    float mSumFps = 0.f;
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

        mEstimatedPoses.reserve(filepaths.size());
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

        printStats(filepaths.size());

        return 0;
    }

private:

    bool processFrame(armvo::ArmVo& VO, const std::string& filepath, armvo::PixelFormat pixelFormat)
    {
        mFrame = armvo::tools::FileTools::loadImage(filepath, pixelFormat);
        mCurrentFrameIndex++;
        bool logNewLine = false;

        const auto t0 = std::chrono::high_resolution_clock::now();
        armvo::Pose estimatedPose;

        if (!VO.isInitialized())
        {
            const auto status = VO.initialize(mFrame, estimatedPose);
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
            const auto status = VO.update(mFrame, estimatedPose);
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

        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        mEstimatedPoses.push_back(estimatedPose);

        const float fps = 1000.f / duration;
        mSumFps += fps;

        std::cout << "FPS: " << std::fixed << std::setprecision(2) << fps;
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

    void printStats(uint32_t numFrames)
    {
        std::cout << "=======================================" << std::endl;
        std::cout << "Average FPS: " << std::fixed << std::setprecision(2) << 1.0f*mSumFps/numFrames << std::endl;

        if (!mGroundTruthPoses.empty())
        {
            const auto errors = mErrorEstimator.estimate(mEstimatedPoses, mGroundTruthPoses);
            if (errors.empty())
            {
                std::cout << "No KITTI error segments available for this trajectory length" << std::endl;
                return;
            }

            float avgRotationError = 0.f;
            float avgTranslationError = 0.f;
            for (const auto& err : errors)
            {
                avgRotationError += err.rotationErr;
                avgTranslationError += err.translationErr;
            }
            std::cout << "Average Rotation Error: " << std::setprecision(4) << (avgRotationError * 180.f) / (errors.size() * kPi) << " [deg/m]" << std::endl;
            std::cout << "Average Translation Error: " << std::setprecision(4) << avgTranslationError*100.f/errors.size() << "%" << std::endl;
        }
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
