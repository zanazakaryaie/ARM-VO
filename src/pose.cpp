#include "pose.hpp"


// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
void decomposeEssentialMat( const cv::Mat& E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t )
{
    cv::Mat D, U, Vt;
    cv::SVD::compute(E, D, U, Vt);

    if (determinant(U) < 0) U *= -1.;
    if (determinant(Vt) < 0) Vt *= -1.;

    static const cv::Mat W = (cv::Mat_<float>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
    static const cv::Mat W_trans = W.t();

    R1 = U * W * Vt;
    R2 = U * W_trans * Vt;
    t = U.col(2) * 1.0;
}


// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
int cheiralCheck(const cv::Mat &P1, const cv::Mat &points1, const cv::Mat &points2,
                 float dist, cv::Mat &mask, cv::Mat &X)
{
    static const cv::Mat P0 = cv::Mat::eye(3, 4, 5);

    cv::Mat Q;
    triangulatePoints(P0, P1, points1, points2, Q);

    cv::Mat mask1 = Q.row(2).mul(Q.row(3)) >= 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    X=Q.clone();


    mask1 = (Q.row(2) <= dist) & mask1;
    Q = P1 * Q;
    mask1 = (Q.row(2) >= 0) & mask1;
    mask1 = (Q.row(2) <= dist) & mask1;

    mask1 = mask1.t();

    cv::bitwise_and(mask, mask1, mask1);

    return countNonZero(mask1);
}


// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
void recoverPose( cv::Mat F, const cv::InputArray _points1, const cv::InputArray _points2,
                 const cv::Mat &cameraMatrix, cv::OutputArray _R, cv::OutputArray _t, cv::Mat &mask, cv::Mat &X)
{
    F.convertTo(F, CV_32F);

    static const cv::Mat cameraMatrixTransposed = cameraMatrix.t();

    const cv::Mat E = cameraMatrixTransposed*F*cameraMatrix; //Essential Matrix

    cv::Mat points1, points2;

    _points1.getMat().convertTo(points1, CV_32F);
    _points2.getMat().convertTo(points2, CV_32F);

    int npoints = points1.checkVector(2);

    points1 = points1.reshape(1, npoints);
    points2 = points2.reshape(1, npoints);

    static const float fx = cameraMatrix.at<float>(0,0);
    static const float fy = cameraMatrix.at<float>(1,1);
    static const float cx = cameraMatrix.at<float>(0,2);
    static const float cy = cameraMatrix.at<float>(1,2);


    points1.col(0) = (points1.col(0) - cx) / fx;
    points2.col(0) = (points2.col(0) - cx) / fx;
    points1.col(1) = (points1.col(1) - cy) / fy;
    points2.col(1) = (points2.col(1) - cy) / fy;

    points1 = points1.t();
    points2 = points2.t();

    cv::Mat R1, R2, t;

    decomposeEssentialMat(E, R1, R2, t);


    cv::Mat P1(3, 4, 5);
    std::vector<cv::Mat> P;
    P.resize(4);

    P1(cv::Range::all(), cv::Range(0, 3)) = R1 * 1.0;
    P1.col(3) = t * 1.0;
    P1.copyTo(P[0]);
    P1(cv::Range::all(), cv::Range(0, 3)) = R2 * 1.0; //P1.col(3) = t * 1.0;
    P1.copyTo(P[1]);
    P1(cv::Range::all(), cv::Range(0, 3)) = R1 * 1.0;
    P1.col(3) = -t * 1.0;
    P1.copyTo(P[2]);
    P1(cv::Range::all(), cv::Range(0, 3)) = R2 * 1.0; //P1.col(3) = -t * 1.0;
    P1.copyTo(P[3]);


    // Do the cheirality check.
    // Notice here a threshold dist is used to filter
    // out far away points (i.e. infinite points) since
    // there depth may vary between postive and negtive.

    float dist = 50.0;
    int goodness[4];
    std::vector<cv::Mat> Q;
    Q.resize(4);

    #pragma omp parallel for num_threads(4)
    for (unsigned int i=0; i<4; i++)
        goodness[i]=cheiralCheck(P[i], points1, points2, dist, mask, Q[i]);


    _R.create(3, 3, 5);
    _t.create(3, 1, 5);

    if (goodness[0] >= goodness[1] && goodness[0] >= goodness[2] && goodness[0] >= goodness[3])
    {
        R1.copyTo(_R);
        t.copyTo(_t);
        Q[0].copyTo(X);
    }
    else if (goodness[1] >= goodness[0] && goodness[1] >= goodness[2] && goodness[1] >= goodness[3])
    {
        R2.copyTo(_R);
        t.copyTo(_t);
        Q[1].copyTo(X);
    }
    else if (goodness[2] >= goodness[0] && goodness[2] >= goodness[1] && goodness[2] >= goodness[3])
    {
        t = -t;
        R1.copyTo(_R);
        t.copyTo(_t);
        Q[2].copyTo(X);
    }
    else
    {
        t = -t;
        R2.copyTo(_R);
        t.copyTo(_t);
        Q[3].copyTo(X);
    }
}
