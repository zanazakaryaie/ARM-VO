#ifndef FAST_NEON_HPP_INCLUDED
#define FAST_NEON_HPP_INCLUDED

#include <arm_neon.h>
#include <opencv2/features2d.hpp>

int cornerScore(const uchar* ptr, const int pixel[], int threshold);

void makeOffsets(int pixel[25], const int rowStride);

uint16_t movemask ( uint8x16_t input );

void FAST9_NEON(const cv::Mat &img, std::vector<cv::KeyPoint>& keypoints, int threshold);

#endif // FAST_NEON_HPP_INCLUDED
