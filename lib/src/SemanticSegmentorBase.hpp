#pragma once

#include <cstddef>
#include <cstdint>

#include "ISemanticSegmentor.hpp"

namespace armvo
{

class SemanticSegmentorBase : public ISemanticSegmentor
{
public:
    /// @copydoc ISemanticSegmentor::getRoadMask()
    cv::Mat getRoadMask(const cv::Mat& segmentationMap) override
    {
        constexpr int ROAD_LABEL = 0;
        return segmentationMap == ROAD_LABEL;
    }

    /// @copydoc ISemanticSegmentor::getStaticMask()
    cv::Mat getStaticMask(const cv::Mat& segmentationMap) override
    {
        cv::Mat mask(segmentationMap.size(), CV_8UC1);
        const uint8_t* const src = segmentationMap.ptr<uint8_t>();
        uint8_t* dst = mask.ptr<uint8_t>();
        for (size_t i = 0; i < mask.total(); i++)
        {
            dst[i] = (src[i] <= 10 ? 255 : 0);
        }
        return mask;
    }
};

} // namespace armvo
