#pragma once

#include "../../../include/ARM_VO.hpp"

#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

void getQuaternion(const cv::Mat& R, float Q[]);

class ARM_VO_Node
{
public:

    ARM_VO_Node(ros::NodeHandle &nh, const std::string& paramsFileName); //Constructs ARM_VO, image subscriber and pose publisher

private:

    ARM_VO VO;
    Viewer Results;

    image_transport::Subscriber imageSubscriber;
    ros::Publisher posePublisher;

    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
};
