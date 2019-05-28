#include "detector.hpp"
#include "FAST_NEON.hpp"
#include <opencv2/features2d.hpp>

void keepStrongest( int N, std::vector<cv::KeyPoint>& keypoints )
{
    if( (int)keypoints.size() > N )
    {
        std::vector<cv::KeyPoint>::iterator nth = keypoints.begin() + N;
        nth_element( keypoints.begin(), nth, keypoints.end(), ResponseComparator() );
        keypoints.erase( nth, keypoints.end() );
    }
}


class GridAdaptedFeatureDetectorInvoker : public cv::ParallelLoopBody
{
private:
    int gridRows_, gridCols_;
    int maxPerCell_;
    int threshold_;
    std::vector<cv::KeyPoint>& keypoints_;
    const cv::Mat& image_;
    cv::Mutex* kptLock_;

    GridAdaptedFeatureDetectorInvoker& operator=(const GridAdaptedFeatureDetectorInvoker&);

public:

    GridAdaptedFeatureDetectorInvoker(const cv::Mat& image,
                                      std::vector<cv::KeyPoint>& keypoints, int maxPerCell, int gridRows, int gridCols, int threshold,
                                      cv::Mutex* kptLock)
        : gridRows_(gridRows), gridCols_(gridCols), maxPerCell_(maxPerCell), threshold_(threshold),
          keypoints_(keypoints), image_(image),
          kptLock_(kptLock)
    {
    }

    void operator() (const cv::Range& range) const
    {

        for (int i = range.start; i < range.end; ++i)
        {
            int celly = i / gridCols_;
            int cellx = i - celly * gridCols_;

            cv::Range row_range((celly*image_.rows)/gridRows_, ((celly+1)*image_.rows)/gridRows_);
            cv::Range col_range((cellx*image_.cols)/gridCols_, ((cellx+1)*image_.cols)/gridCols_);

            cv::Mat sub_image = image_(row_range, col_range);

            std::vector<cv::KeyPoint> sub_keypoints;
            FAST9_NEON(sub_image, sub_keypoints, threshold_);
            keepStrongest( maxPerCell_, sub_keypoints );

            std::vector<cv::KeyPoint>::iterator it = sub_keypoints.begin(),
                                                end = sub_keypoints.end();
            for( ; it != end; ++it )
            {
                it->pt.x += col_range.start;
                it->pt.y += row_range.start;
            }

            cv::AutoLock join_keypoints(*kptLock_);
            keypoints_.insert( keypoints_.end(), sub_keypoints.begin(), sub_keypoints.end() );
        }
    }
};



gridFASTdetector::gridFASTdetector(int n, int thresh, int grid_x, int grid_y)
{
    nFeatures = n;
    threshold = thresh;
    nRows = grid_x;
    nCols = grid_y;
}

void gridFASTdetector::detect( const cv::Mat& image, std::vector<cv::Point2f>& points)
{
    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(nFeatures);
    int maxPerCell = nFeatures / (nRows * nCols);

    cv::Mutex kptLock;
    cv::parallel_for_(cv::Range(0, nRows * nCols),
                      GridAdaptedFeatureDetectorInvoker(image, keypoints, maxPerCell, nRows, nCols, threshold ,&kptLock));

    if (keypoints.size() < 100)
    {
        printf("\nFew keypoints in the scene! Lowering FAST threshold to 5 for this frame\n");

        keypoints.clear();
        keypoints.reserve(nFeatures);
        cv::parallel_for_(cv::Range(0, nRows * nCols),
                      GridAdaptedFeatureDetectorInvoker(image, keypoints, maxPerCell, nRows, nCols, 5 ,&kptLock));

        if (keypoints.size() < 100)
        {
            printf("\nToo few keypoints in the scene!!!\n");
            abort();
        }
    }

    points.clear();
    cv::KeyPoint::convert(keypoints, points, std::vector<int>());
}
