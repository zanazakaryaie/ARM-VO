#include <stdexcept>
#include <opencv2/core/persistence.hpp>
#include "Types.hpp"

namespace armvo
{

ArmVoConfig ArmVoConfig::load(const std::string& filepath)
{
    cv::FileStorage fs(filepath, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        throw std::runtime_error("Could not open config file: " + filepath);
    }

    ArmVoConfig config;

    const auto& cameraNode = fs["Camera"];
    if (!cameraNode.isMap())
    {
        throw std::logic_error("Config file is expected to have \"Camera\" node");
    }
    cameraNode["height"] >> config.camera.height;

    std::string pixelFormat;
    cameraNode["pixel_format"] >> pixelFormat;
    if (pixelFormat == "gray")
    {
        config.camera.pixelFormat = PixelFormat::Gray;
    }
    else if (pixelFormat == "rgb")
    {
        config.camera.pixelFormat = PixelFormat::RGB;
    }
    else if (pixelFormat == "bgr")
    {
        config.camera.pixelFormat = PixelFormat::BGR;
    }
    else
    {
        throw std::invalid_argument("Invalid pixel format. Supported formats are \"gray\", \"rgb\", and \"bgr\"");
    }

    cameraNode["fps"] >> config.camera.fps;

    if (!cameraNode["distortions"].empty())
    {
        const cv::FileNode distortionsNode = cameraNode["distortions"];
        if (!distortionsNode.isSeq())
        {
            throw std::logic_error("\"distortions\" is expected to be a sequence: [...]");
        }
        size_t numCoeffs = distortionsNode.size();
        cv::Mat distortions(1, numCoeffs, CV_64F);
        for (size_t i = 0; i < numCoeffs; i++)
        {
            distortions.at<double>(i) = distortionsNode[i];
        }
        config.camera.distortions = distortions.clone();
    }

    config.camera.intrinsics = cv::Matx33f((float)cameraNode["fx"], 0, (float)cameraNode["cx"],
                                            0, (float)cameraNode["fy"], (float)cameraNode["cy"],
                                            0, 0, 1);


    const auto& detectorNode = fs["Keypoint_Detector"];
    if (!detectorNode.isMap())
    {
        throw std::logic_error("Config file is expected to have \"Keypoint_Detector\" node");
    }
    detectorNode["num_points"] >> config.keypointDetector.maxNumberOfPoints;
    detectorNode["response_threshold"] >> config.keypointDetector.responseThreshold;
    detectorNode["num_image_grid_rows"] >> config.keypointDetector.numberOfImageGridRows;
    detectorNode["num_image_grid_cols"] >> config.keypointDetector.numberOfImageGridCols;

    const auto& trackerNode = fs["Keypoint_Tracker"];
    if (!trackerNode.isMap())
    {
        throw std::logic_error("Config file is expected to have \"Keypoint_Tracker\" node");
    }
    trackerNode["window_size"] >> config.keypointTracker.windowSize;

    const auto& vehicleNode = fs["Vehicle"];
    if (!vehicleNode.isMap())
    {
        throw std::logic_error("Config file is expected to have \"Vehicle\" node");
    }
    vehicleNode["max_speed"] >> config.maxVehicleSpeed;

    fs.release();

    return config;
}

}
