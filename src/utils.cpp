#include <opencv2/opencv.hpp>
#include "utils.hpp"


void read_params(std::string filename, Parameters &p)
{
    cv::FileStorage fs(filename,cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        printf("\nError loading params file!\n");
        abort();
    }

    ///Calibration parameters
    float fx = fs["fx"];
    float fy = fs["fy"];
    float cx = fs["cx"];
    float cy = fs["cy"];
    p.cameraMatrix=(cv::Mat_<float>(3,3)<<fx,0,cx,
                                           0,fy,cy,
                                           0,0,1);

    p.cameraHeight = fs["height"]; //in meter
    p.cameraPitchAngle = fs["pitch_angle"]; //in radian

    ///Detector parameters
    p.nFeatures = fs["nFeatures"];
    p.threshold = fs["threshold"];
    p.nRows = fs["nRows"];
    p.nCols = fs["nCols"];


    ///Tracker parameters
    p.winSize = fs["window_size"];
}




cv::Mat Viewer::draw_tracked_kpts(const cv::Mat &frame, std::vector<cv::Point2f> &prev_keypoints, std::vector<cv::Point2f> &curr_keypoints)
{
    cv::Mat frameWithFeatures;
    cv::cvtColor(frame, frameWithFeatures, CV_GRAY2BGR);

    for (size_t i=0; i<curr_keypoints.size(); i++)
    {
        cv::line(frameWithFeatures, prev_keypoints[i], curr_keypoints[i], cv::Scalar(255,0,0), 2);
        cv::circle(frameWithFeatures, curr_keypoints[i], 3, CV_RGB(0,255,0), -1);
    }

    return frameWithFeatures;
}


cv::Point2i Viewer::convertToImageCoordinates(const cv::Point2f pointInMeters)
{
    cv::Point2f pt = (pointInMeters - cv::Point2f(min_x, min_y)) * scale;
    return cv::Point2i(pt.x, imageSize - pt.y); // reverse y for more intuitive top-down plot
}

cv::Mat Viewer::draw_trajectory(void)
{
    float x = path.back().x;
    float y = path.back().y;

    if (x - frameScale < min_x)
        min_x = x - frameScale;
    if (y - frameScale < min_y)
        min_y = y - frameScale;
    if (x + frameScale > max_x)
        max_x = x + frameScale;
    if (y + frameScale > max_y)
        max_y = y + frameScale;

    cv::Mat traj(imageSize, imageSize, CV_8UC3, cv::Scalar::all(0));
    scale = std::min(imageSize / (max_x - min_x), imageSize / (max_y - min_y));

    for (size_t i = 0; i + 1 < path.size(); )
    {
        cv::Point2i p0 = convertToImageCoordinates(path[i]);
        cv::Point2i p1 = convertToImageCoordinates(path[i + 1]);

        cv::line(traj,p0,p1,cv::Scalar(0,255,0),3);

        i++;
    }

    return traj;
}

void Viewer::show(const cv::Mat &frame, std::vector<cv::Point2f> prev_keypoints, std::vector<cv::Point2f> curr_keypoints, int FPS, const cv::Mat translation)
{
    cv::Mat frameWithFeatures;
    if (curr_keypoints.size() > 0)
        frameWithFeatures = draw_tracked_kpts(frame, prev_keypoints, curr_keypoints);
    else
        cv::cvtColor(frame, frameWithFeatures, CV_GRAY2BGR);

    cv::resize(frameWithFeatures, frameWithFeatures, cv::Size(), 0.5, 0.5);

    char text[20];
    sprintf(text, "FPS: %d", FPS);
    putText(frameWithFeatures, text, cv::Point(8,30),CV_FONT_HERSHEY_SIMPLEX,1.0,cv::Scalar(0,0,255),3);

    cv::imshow("Frames", frameWithFeatures);


    float x =  translation.at<float>(0);
    float y = -translation.at<float>(2);
    path.push_back(cv::Point2f(x,y));

    cv::Mat trajectory = draw_trajectory();
    sprintf(text, "X[m]: %.2f", path.back().x);
    putText(trajectory, text, cv::Point(350,20),CV_FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar::all(255),2);

    sprintf(text, "Y[m]: %.2f", path.back().y);
    putText(trajectory, text, cv::Point(350,50),CV_FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar::all(255),2);

    cv::imshow("Trajectory", trajectory);
    cv::waitKey(1);
}
