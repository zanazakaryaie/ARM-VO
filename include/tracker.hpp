#ifndef TRACKER_HPP_INCLUDED
#define TRACKER_HPP_INCLUDED

#include "opencv2/video/tracking.hpp"

using namespace std;
using namespace cv;

class tracker
{
public:
    tracker(int windowSize);
    void track(const Mat &img_1, const Mat &img_2, vector<Point2f>& points1, vector<Point2f>& points2);
private:
    Size winSize;
    TermCriteria termcrit;
    void calcSharrDeriv(const Mat& src, Mat& dst);
    void calc_OpticalFlowPyrLK( InputArray _prevImg, InputArray _nextImg,
                            InputArray _prevPts, InputOutputArray _nextPts,
                            OutputArray _status,
                            Size winSize, int maxLevel,
                            TermCriteria criteria,
                            int flags, double minEigThreshold );
};

#endif // TRACKER_HPP_INCLUDED
