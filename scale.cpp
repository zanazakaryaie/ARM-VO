#include "scale.hpp"

scaleEstimator::scaleEstimator(float a, float b)
{
    cam_height = a;
    cam_pitch = b;
    n = (cv::Mat_<float>(2, 1) << cosf(-cam_pitch), sinf(-cam_pitch));
    n_transpose = n.t();
}

void scaleEstimator::smallerThanMedian(cv::Mat &X,float &median)
{
    // set distance and index vector
    std::vector<float> dist;
    std::vector<int32_t> idx;
    for (int32_t i=0; i<X.cols; i++)
    {
        dist.push_back(fabs(X.at<float>(0,i))+fabs(X.at<float>(1,i))+fabs(X.at<float>(2,i)));
        idx.push_back(i);
    }

    // sort elements
    sort(idx.begin(),idx.end(),idx_cmp<std::vector<float>&>(dist));

    // get median
    int32_t num_elem_half = idx.size()/2;
    median = dist[idx[num_elem_half]];
}


float scaleEstimator::estimate(cv::Mat &X)
{
    std::vector<int32_t> pos_idx;
    pos_idx.reserve(X.cols);

    for (int32_t j=0; j<X.cols; j++)
        if (X.at<float>(2,j)>0)
            pos_idx.push_back(j);

    int nCols = pos_idx.size();
    cv::Mat X_plane(4,nCols,5);
    for (int32_t j=0; j<nCols; j++)
        X.col(pos_idx[j]).copyTo(X_plane.col(j));

    // get elements closer than median
    float median;
    smallerThanMedian(X_plane,median);

    cv::Mat x_plane(2,nCols,5);
    X_plane.row(1).copyTo(x_plane.row(0));
    X_plane.row(2).copyTo(x_plane.row(1));

    cv::Mat   d        = n_transpose*x_plane;
    float   sigma    = median/50.f;
    float   weight   = 1.f/(2.f*sigma*sigma);
    float   best_sum = 0.f;
    int32_t  best_idx = 0;

    float sum, dist;
    float temporary = median/motion_threshold;
    // find best plane
    for (int32_t j=0; j<nCols; j++)
    {
        if (d.at<float>(0,j)> temporary)
        {
            sum = 0;

            for (int32_t p=0; p<nCols; p++)
            {
                if (p!=j)
                {
                    dist = d.at<float>(p)-d.at<float>(j);
                    sum += exp(-dist*dist*weight);
                }
            }

            if (sum>best_sum)
            {
                best_sum = sum;
                best_idx = j;
            }
        }
    }

    float scale = cam_height/d.at<float>(best_idx);

    if (scale < 0 || scale/prev_scale > 1.3)
        scale = prev_scale;

    prev_scale = scale;
    return scale;
}
