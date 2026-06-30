#pragma once

#include <memory>
#include "ISemanticSegmentor.hpp"
#include "Types.hpp"

namespace armvo
{

/// @brief NCNN-backed semantic segmentor
class SemanticSegmentorNcnn : public ISemanticSegmentor
{
public:

    /// @brief Loads the staged NCNN model files
    /// @throws Throws exception if the model shape, param, or bin files cannot be loaded
    SemanticSegmentorNcnn();

    /// @copydoc ISemanticSegmentor::runsOnCoProcessor()
    bool runsOnCoProcessor() override;

    /// @copydoc ISemanticSegmentor::segment()
    cv::Mat segment(const cv::Mat& frame, PixelFormat pixelFormat) override;

    /// @copydoc ISemanticSegmentor::getRoadMask()
    cv::Mat getRoadMask(const cv::Mat& segmentationMap) override;

    /// @copydoc ISemanticSegmentor::getStaticMask()
    cv::Mat getStaticMask(const cv::Mat& segmentationMap) override;

private:
    class Impl;
    std::shared_ptr<Impl> mImpl;
};

} // namespace armvo
