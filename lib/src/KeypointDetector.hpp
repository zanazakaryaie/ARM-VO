#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace armvo
{

struct Keypoint
{
    float x;
    float y;
    float score;

    Keypoint(float x, float y, float score) : x(x), y(y), score(score) {}
};

class KeypointDetector
{
public:

    explicit KeypointDetector(uint8_t responseThreshold);

    std::vector<Keypoint> detect(const cv::Mat& image);

private:
    class Impl;
    std::shared_ptr<Impl> mImpl;
};

} // namespace armvo
