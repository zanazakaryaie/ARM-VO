#pragma once

#include <opencv2/core.hpp>

template<class T> struct idx_cmp
{
    idx_cmp(const T arr) : arr(arr) {}
    bool operator()(const size_t a, const size_t b) const
    {
        return arr[a] < arr[b];
    }
    const T arr;
};


class scaleEstimator
{
public:
    void setCamera(const float height, const float pitch);
    float estimate(const cv::Mat &X);

private:
    void smallerThanMedian (const cv::Mat &X, float &median);
    float cam_height;
    float cam_pitch;
    float motion_threshold = 100.0;
    float prev_scale = 1;
    cv::Mat n_transpose;
};
