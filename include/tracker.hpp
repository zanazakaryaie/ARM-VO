#pragma once

#include <opencv2/video/tracking.hpp>

class tracker
{
public:
    void setWindowSize(const int windowSize);
    void track(const cv::Mat &img_1, const cv::Mat &img_2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);

private:
    cv::Size winSize;
    void calcSharrDeriv(const cv::Mat& src, cv::Mat& dst);
    void calc_OpticalFlowPyrLK( cv::InputArray _prevImg, cv::InputArray _nextImg,
                            cv::InputArray _prevPts, cv::InputOutputArray _nextPts,
                            cv::OutputArray _status,
                            cv::Size winSize, int maxLevel,
                            cv::TermCriteria criteria,
                            int flags, double minEigThreshold );
};
