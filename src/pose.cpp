#include "pose.hpp"


void decomposeEssentialMat( cv::Mat E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t )
{

    cv::Mat D, U, Vt;
    cv::SVD::compute(E, D, U, Vt);

    if (determinant(U) < 0) U *= -1.;
    if (determinant(Vt) < 0) Vt *= -1.;

    static cv::Mat W = (cv::Mat_<float>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
    static cv::Mat W_trans = W.t();

    R1 = U * W * Vt;
    R2 = U * W_trans * Vt;
    t = U.col(2) * 1.0;
}



void cvTriangulate_Points(CvMat* projMatr1, CvMat* projMatr2, CvMat* projPoints1, CvMat* projPoints2, CvMat* points4D)
{

    int numPoints = projPoints1->cols;

    cv::Matx<float, 4, 4> matrA;
    cv::Matx<float, 4, 4> matrU;
    cv::Matx<float, 4, 1> matrW;
    cv::Matx<float, 4, 4> matrV;

    CvMat* projPoints[2] = {projPoints1, projPoints2};
    CvMat* projMatrs[2] = {projMatr1, projMatr2};


    float x,y;
    /* Solve system for each point */
    int i,j;
    for( i = 0; i < numPoints; i++ )/* For each point */
    {
        /* Fill matrix for current point */
        for( j = 0; j < 2; j++ )/* For each view */
        {
            x = cvmGet(projPoints[j],0,i);
            y = cvmGet(projPoints[j],1,i);
            for( int k = 0; k < 4; k++ )
            {
                matrA(j*2+0, k) = x * cvmGet(projMatrs[j],2,k) - cvmGet(projMatrs[j],0,k);
                matrA(j*2+1, k) = y * cvmGet(projMatrs[j],2,k) - cvmGet(projMatrs[j],1,k);
            }
        }
        /* Solve system for current point */
        {
            cv::SVD::compute(matrA, matrW, matrU, matrV);

            /* Copy computed point */
            cvmSet(points4D,0,i,matrV(3,0));/* X */
            cvmSet(points4D,1,i,matrV(3,1));/* Y */
            cvmSet(points4D,2,i,matrV(3,2));/* Z */
            cvmSet(points4D,3,i,matrV(3,3));/* W */
        }
    }
}





void triangulate_Points( cv::Mat &projMatr1, cv::Mat &projMatr2,
                         cv::InputArray _projPoints1, cv::InputArray _projPoints2,
                         cv::OutputArray _points4D )
{
    cv::Mat points1 = _projPoints1.getMat(), points2 = _projPoints2.getMat();

    CvMat cvMatr1 = projMatr1, cvMatr2 = projMatr2;
    CvMat cvPoints1 = points1, cvPoints2 = points2;

    _points4D.create(4, points1.cols, 5);
    CvMat cvPoints4D = _points4D.getMat();

    cvTriangulate_Points(&cvMatr1, &cvMatr2, &cvPoints1, &cvPoints2, &cvPoints4D);
}




int cheiralCheck(cv::Mat &P1, cv::Mat &points1, cv::Mat &points2, float dist, cv::Mat &mask, cv::Mat &X)
{
    static cv::Mat P0 = cv::Mat::eye(3, 4, 5);

    cv::Mat Q;
    triangulate_Points(P0, P1, points1, points2, Q);

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




void recoverPose( cv::Mat F, cv::InputArray _points1, cv::InputArray _points2, cv::Mat &cameraMatrix,
                  cv::OutputArray _R, cv::OutputArray _t, cv::Mat &mask, cv::Mat &X)
{

    F.convertTo(F, CV_32F);

    static cv::Mat cameraMatrixTransposed = cameraMatrix.t();

    cv::Mat E = cameraMatrixTransposed*F*cameraMatrix; //Essential Matrix

    cv::Mat points1, points2;

    _points1.getMat().convertTo(points1, CV_32F);
    _points2.getMat().convertTo(points2, CV_32F);

    int npoints = points1.checkVector(2);

    points1 = points1.reshape(1, npoints);
    points2 = points2.reshape(1, npoints);

    static float fx = cameraMatrix.at<float>(0,0);
    static float fy = cameraMatrix.at<float>(1,1);
    static float cx = cameraMatrix.at<float>(0,2);
    static float cy = cameraMatrix.at<float>(1,2);


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
    P[0]=P1.clone();
    P1(cv::Range::all(), cv::Range(0, 3)) = R2 * 1.0; //P1.col(3) = t * 1.0;
    P[1] = P1.clone();
    P1(cv::Range::all(), cv::Range(0, 3)) = R1 * 1.0;
    P1.col(3) = -t * 1.0;
    P[2] = P1.clone();
    P1(cv::Range::all(), cv::Range(0, 3)) = R2 * 1.0; //P1.col(3) = -t * 1.0;
    P[3] = P1.clone();


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
        X=Q[0].clone();
    }
    else if (goodness[1] >= goodness[0] && goodness[1] >= goodness[2] && goodness[1] >= goodness[3])
    {
        R2.copyTo(_R);
        t.copyTo(_t);
        X=Q[1].clone();
    }
    else if (goodness[2] >= goodness[0] && goodness[2] >= goodness[1] && goodness[2] >= goodness[3])
    {
        t = -t;
        R1.copyTo(_R);
        t.copyTo(_t);
        X=Q[2].clone();
    }
    else
    {
        t = -t;
        R2.copyTo(_R);
        t.copyTo(_t);
        X=Q[3].clone();
    }


}
