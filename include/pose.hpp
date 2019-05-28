#ifndef POSE_HPP_INCLUDED
#define POSE_HPP_INCLUDED


#include <opencv2/imgproc.hpp>


void decomposeEssentialMat( cv::Mat E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t );


void cvTriangulate_Points(CvMat* projMatr1, CvMat* projMatr2, CvMat* projPoints1, CvMat* projPoints2, CvMat* points4D);


void triangulate_Points( cv::Mat &projMatr1, cv::Mat &projMatr2,
                         cv::InputArray _projPoints1, cv::InputArray _projPoints2,
                         cv::OutputArray _points4D );


int cheiralCheck(cv::Mat &P1, cv::Mat &points1, cv::Mat &points2, float dist, cv::Mat &mask, cv::Mat &X);


void recoverPose( cv::Mat E, cv::InputArray _points1, cv::InputArray _points2, cv::Mat &cameraMatrix,
                  cv::OutputArray _R, cv::OutputArray _t, cv::Mat &mask, cv::Mat &X);


#endif // POSE_HPP_INCLUDED
