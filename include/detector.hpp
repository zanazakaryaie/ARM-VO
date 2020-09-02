#pragma once

#include <vector>
#include <iostream>
#include <opencv2/features2d.hpp>


class gridFASTdetector
{
public:
    void setMaxFeatures(const int n);
    void setThreshold(const int thresh);
    void setGridSize(const cv::Size& gridSize);
    bool detect( const cv::Mat& image, std::vector<cv::Point2f>& points);

private:
    int nFeatures;
    int threshold;
    cv::Size grid;
};



struct ResponseComparator
{
    bool operator() (const cv::KeyPoint& a, const cv::KeyPoint& b)
    {
        return abs(a.response) > abs(b.response);
    }
};

void keepStrongest( int N, std::vector<cv::KeyPoint>& keypoints ); // Selects the N strongest keypoints detected based on their response



class GridAdaptedFeatureDetectorInvoker : public cv::ParallelLoopBody
{
public:
    GridAdaptedFeatureDetectorInvoker(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const int maxPerCell, const int gridRows, const int gridCols, const int thresh, cv::Mutex* kptLock);
    void operator() (const cv::Range& range) const;

private:
    int gridRows_, gridCols_;
    int maxPerCell_;
    int threshold_;
    std::vector<cv::KeyPoint>& keypoints_;
    const cv::Mat& image_;
    cv::Mutex* kptLock_;
};
