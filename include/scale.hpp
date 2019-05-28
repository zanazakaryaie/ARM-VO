#ifndef SCALE_HPP_INCLUDED
#define SCALE_HPP_INCLUDED

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
    scaleEstimator(float a, float b);
    float estimate(cv::Mat &X);
private:
    float cam_height;
    float cam_pitch;
    float motion_threshold = 100.0;
    float prev_scale = 1;
    cv::Mat n, n_transpose;
    void smallerThanMedian (cv::Mat &X,float &median);
};


#endif // SCALE_HPP_INCLUDED
