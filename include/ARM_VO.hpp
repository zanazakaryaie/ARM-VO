#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>

#include "scale.hpp"
#include "GRIC.hpp"
#include "pose.hpp"
#include "tracker.hpp"
#include "detector.hpp"
#include "viewer.hpp"

class ARM_VO
{
public:

    ARM_VO(void); //Default Constructor
    ARM_VO(const std::string& paramsFileName); //Constructor
    void loadSetting(const std::string& paramsFileName);
    void init(const cv::Mat& firstFrame); //Initializes the VO with features in the first frame. Camera pose is set to Zero
    void update(const cv::Mat& currentFrame); //Updates Camera pose with current frame

    bool initialized = false;
    cv::Mat R_f, t_f; //Current pose of the camera
    std::vector<cv::Point2f> prev_inliers, curr_inliers; //Required for Viewer

private:

    int maxFeatures; //Maximum number of keypoints to be extracted from each frame
    cv::Mat cameraMatrix;
    cv::Mat prev_frame;
    std::vector<cv::Point2f> prev_keypoints;
    gridFASTdetector Detector;
    tracker KLT;
    scaleEstimator Scale;
};
