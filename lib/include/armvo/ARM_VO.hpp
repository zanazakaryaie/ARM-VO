#pragma once

#include <memory>

#include <armvo/Types.hpp>

namespace armvo
{

/// @brief Monocular visual odometry pipeline optimized for ARM processors
class ArmVo
{
public:
    /// @brief Creates an ARM-VO pipeline from a config
    /// @param config Runtime configuration. This can be created manually or loaded with ArmVoConfig::load()
    /// @throws Throws exceptions if the configuration is invalid
    explicit ArmVo(const ArmVoConfig& config);

    ArmVo(const ArmVo&) = delete;
    ArmVo& operator=(const ArmVo&) = delete;

    /// @brief Initializes tracking from the first frame and sets pose to identity
    /// @param frame Input image. Expected type is CV_8UC1 for Gray and CV_8UC3 for RGB/BGR
    /// @param pose Initial pose, valid only when Status::SUCCESS is returned
    /// @return Initialization status
    [[nodiscard]] Status initialize(const cv::Mat& frame, Pose& pose);

    /// @brief Returns whether initialization has completed successfully
    /// @return True after initialize() returns Status::SUCCESS, otherwise false
    [[nodiscard]] bool isInitialized() const noexcept;

    /// @brief Updates the camera pose with a new frame
    /// @param frame Input image. Expected type is CV_8UC1 for Gray and CV_8UC3 for RGB/BGR
    /// @param pose Estimated pose, valid only when Status::SUCCESS or Status::FRAME_SKIPPED is returned
    /// @return Update status
    /// @pre initialize() must have returned Status::SUCCESS before calling this function
    [[nodiscard]] Status update(const cv::Mat& frame, Pose& pose);

private:
    class Impl;
    std::shared_ptr<Impl> mImpl;
};

} // namespace armvo
