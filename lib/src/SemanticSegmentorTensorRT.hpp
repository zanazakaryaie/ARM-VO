#pragma once

#include <memory>

#include "SemanticSegmentorBase.hpp"
#include "Types.hpp"

namespace armvo
{

/// @brief TensorRT-backed semantic segmentor
class SemanticSegmentorTensorRT final : public SemanticSegmentorBase
{
public:

    /// @brief Loads the staged TensorRT engine
    /// @throws Throws exception if the model shape or engine file cannot be loaded
    SemanticSegmentorTensorRT();

    /// @copydoc ISemanticSegmentor::runsOnCoProcessor()
    bool runsOnCoProcessor() override;

    /// @copydoc ISemanticSegmentor::segment()
    cv::Mat segment(const cv::Mat& frame, PixelFormat pixelFormat) override;

private:
    class Impl;
    std::shared_ptr<Impl> mImpl;
};

} // namespace armvo
