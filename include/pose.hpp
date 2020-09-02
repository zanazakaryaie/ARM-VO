#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

void decomposeEssentialMat( const cv::Mat& E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t );

int cheiralCheck(const cv::Mat &P1, const cv::Mat &points1, const cv::Mat &points2,
                 float dist, cv::Mat &mask, cv::Mat &X);

void recoverPose( cv::Mat F, const cv::InputArray _points1, const cv::InputArray _points2,
                 const cv::Mat &cameraMatrix, cv::OutputArray _R, cv::OutputArray _t, cv::Mat &mask, cv::Mat &X);

