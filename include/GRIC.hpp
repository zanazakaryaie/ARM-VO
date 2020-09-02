#pragma once

#include <vector>
#include <opencv2/core.hpp>

float calc_GRIC(float *res, float sigma, int n, int model);

void sampsonF_dsqr(std::vector<float> &F, std::vector<cv::Point2f> &pts0, std::vector<cv::Point2f> &pts1, int npts, float *res);


// squared geometric distance to the H manifold.
void sampsonH_dsqr(std::vector<float> &H, std::vector<cv::Point2f> &pts0, std::vector<cv::Point2f> &pts1, int npts, float *res);



/* Given a set of matching points (including outliers) and estimates of fundamental
 * and homography matrices F & H resp., computes Torr's GRIC
 *
 * A model with lower GRIC is more likely
 */
void GRIC(std::vector<cv::Point2f> &pts0, std::vector<cv::Point2f> &pts1, int nmatches,
          cv::Mat &Fundamental, cv::Mat &Homography, float sigma, float &gricF, float &gricH);
