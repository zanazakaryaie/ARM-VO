#ifndef DETECTOR_HPP_INCLUDED
#define DETECTOR_HPP_INCLUDED

#include <vector>
#include <opencv2/features2d.hpp>

struct ResponseComparator
{
    bool operator() (const cv::KeyPoint& a, const cv::KeyPoint& b)
    {
        return abs(a.response) > abs(b.response);
    }
};

// Selects the N strongest keypoints detected based on their response
void keepStrongest( int N, std::vector<cv::KeyPoint>& keypoints );

class gridFASTdetector
{
public:
    gridFASTdetector(int n, int thresh, int grid_x, int grid_y);
    void detect( const cv::Mat& image, std::vector<cv::Point2f>& points);

private:
    int nFeatures;
    int threshold;
    int nRows;
    int nCols;
};

#endif // DETECTOR_HPP_INCLUDED
