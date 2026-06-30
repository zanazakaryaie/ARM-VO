#include "FileTools.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

namespace
{

armvo::Pose parseKittiPose(const std::string& poseStr)
{
    float tokens[12];
    std::istringstream ss(poseStr);
    for (uint8_t i=0; i<12; i++)
    {
        ss >> tokens[i];
    }
    armvo::Pose pose;
    pose.rotation = cv::Matx33f(tokens[0], tokens[1], tokens[2],
                                tokens[4], tokens[5], tokens[6],
                                tokens[8], tokens[9], tokens[10]);
    pose.translation = cv::Matx31f(tokens[3], tokens[7], tokens[11]);
    return pose;
}

}

namespace armvo::tools
{

std::vector<std::string> FileTools::listFramesInDir(const std::string& dir)
{
    if (!fs::is_directory(dir))
    {
        throw std::invalid_argument(dir + " is not a directory");
    }

    std::vector<std::string> filepaths;
    for (const auto& entry : fs::directory_iterator(dir))
    {
        if (fs::is_regular_file(entry))
        {
            const std::string ext = entry.path().extension();
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg")
            {
                filepaths.push_back(entry.path().string());
            }
        }
    }

    std::sort(filepaths.begin(), filepaths.end());

    return filepaths;
}

cv::Mat FileTools::loadImage(const std::string& filepath, PixelFormat desiredPixelFormat)
{
    cv::Mat image;
    switch (desiredPixelFormat)
    {
        case PixelFormat::Gray:
            image = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
            break;
        case PixelFormat::BGR:
        case PixelFormat::RGB:
            image = cv::imread(filepath, cv::IMREAD_COLOR);
            break;
        default:
            throw std::logic_error("Unsupported pixel format");
    }

    if (image.empty())
    {
        throw std::runtime_error("Failed to load " + filepath);
    }

    if (desiredPixelFormat == PixelFormat::RGB)
    {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    }

    return image;
}

std::vector<armvo::Pose> FileTools::loadKittiGroundTruthPoses(const std::string& filepath)
{
    if (!fs::exists(filepath))
    {
        throw std::invalid_argument(filepath + " does not exist");
    }

    if (!fs::is_regular_file(filepath))
    {
        throw std::invalid_argument(filepath + " is not a file");
    }

    std::ifstream file(filepath);
    if (!file)
    {
        throw std::runtime_error("Failed to open " + filepath);
    }

    std::vector<armvo::Pose> poses;

    std::string line;
    while (std::getline(file, line))
    {
        poses.push_back(parseKittiPose(line));
    }

    return poses;
}

}
