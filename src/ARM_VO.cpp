#include "ARM_VO.hpp"

// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
ARM_VO::ARM_VO(void)
{

}


// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
ARM_VO::ARM_VO(const std::string& paramsFileName)
{
    loadSetting(paramsFileName);
}


// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
void ARM_VO::loadSetting(const std::string& paramsFileName)
{
    cv::FileStorage fs(paramsFileName, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        std::cout << "Error loading params file" << std::endl;
        std::abort();
    }

    float fx = fs["fx"], fy = fs["fy"], cx = fs["cx"], cy = fs["cy"];
    cameraMatrix=(cv::Mat_<float>(3,3)<<fx,0,cx,0,fy,cy,0,0,1);

    maxFeatures = fs["nFeatures"];
    Detector.setMaxFeatures(maxFeatures);
    Detector.setThreshold(fs["threshold"]);
    Detector.setGridSize(cv::Size(fs["nRows"], fs["nCols"]));

    KLT.setWindowSize(fs["window_size"]);

    Scale.setCamera(fs["height"], fs["pitch_angle"]);

    fs.release();
}


// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
void ARM_VO::init(const cv::Mat& firstFrame)
{
    bool detectionStatus = Detector.detect(firstFrame, prev_keypoints);

    if (detectionStatus)
    {
        std::cout << "Initialized" << std::endl;

        initialized = true;

        firstFrame.copyTo(prev_frame);

        prev_inliers.reserve(maxFeatures);
        curr_inliers.reserve(maxFeatures);

        R_f = cv::Mat::eye(3,3,CV_32FC1);
        t_f = cv::Mat::zeros(3,1,CV_32FC1);
    }
    else
        std::cerr << "Failed to initialize! Will try the next image" << std::endl;
}


// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
void ARM_VO::update(const cv::Mat& currentFrame)
{
    std::vector<cv::Point2f> curr_keypoints;
    KLT.track(prev_frame, currentFrame, prev_keypoints, curr_keypoints);

    cv::Mat F, H, mask;

    //Parallel Homo and F-matrix estimation
    #pragma omp parallel sections
    {
        #pragma omp section
        F = cv::findFundamentalMat(prev_keypoints, curr_keypoints, mask, 8, 1, 0.99); //FM_RANSAC
        #pragma omp section
        H = cv::findHomography(prev_keypoints, curr_keypoints, 8, 10); //RANSAC
    }

    float Fcriteria, Hcriteria;
    GRIC(prev_keypoints, curr_keypoints, prev_keypoints.size(), F, H, 0.8, Fcriteria, Hcriteria);

    if (Fcriteria < Hcriteria)
    {
        prev_inliers.clear();
        curr_inliers.clear();

        //Refine Fundamental Matrix
        for (size_t j=0; j<prev_keypoints.size(); j++)
        {
            if (mask.at<uchar>(j)==1)
            {
                prev_inliers.push_back(prev_keypoints[j]);
                curr_inliers.push_back(curr_keypoints[j]);
            }
        }

        F = cv::findFundamentalMat(prev_inliers, curr_inliers, mask, 8, 0.5, 0.99); //FM_RANSAC

        cv::Mat R, t, _3dPoints;
        recoverPose(F, prev_inliers, curr_inliers, cameraMatrix, R, t, mask, _3dPoints);

        //Recover Scale from camera height and pitch angle
        float scale_factor = Scale.estimate(_3dPoints);

        t_f = t_f + scale_factor*(R_f*t);
        R_f = R*R_f;

        bool detectionStatus = Detector.detect(currentFrame, prev_keypoints);

        if (!detectionStatus)
        {
            std::cerr << "Track lost!" << std::endl;
            abort();
        }

        currentFrame.copyTo(prev_frame);

        std::cout << "Position: [" << t_f.at<float>(0) << ", " << t_f.at<float>(1) << ", " << t_f.at<float>(2) << "]" << std::endl;
    }
    else
        std::cerr << "Skipping" << std::endl;
}
