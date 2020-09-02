#include "ARM_VO_Node.hpp"

// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
ARM_VO_Node::ARM_VO_Node(ros::NodeHandle &nh, const std::string& paramsFileName)
{
    image_transport::ImageTransport it(nh);
    imageSubscriber = it.subscribe("kitti/image", 1, &ARM_VO_Node::imageCallback, this);

    VO.loadSetting(paramsFileName);

    posePublisher = nh.advertise<geometry_msgs::Pose>("arm_vo/pose", 1);
}


// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
void ARM_VO_Node::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    cv::Mat curr_frame = cv_bridge::toCvShare(msg, "mono8")->image;

    if (curr_frame.empty())
    {
        std::cerr << "Can't read image" << std::endl;
        ros::shutdown();
    }

    if (!VO.initialized)
    {
        VO.init(curr_frame);
    }
    else
    {
        clock_t start = clock();

        VO.update(curr_frame);

        clock_t finish = clock();
        int FPS = 1000 / (1000*(finish-start)/CLOCKS_PER_SEC);

        float quat[4];
        getQuaternion(VO.R_f, quat);

        geometry_msgs::Pose pose_msg;
        pose_msg.position.x = VO.t_f.at<float>(0);
        pose_msg.position.y = VO.t_f.at<float>(1);
        pose_msg.position.z = VO.t_f.at<float>(2);
        pose_msg.orientation.x = quat[0];
        pose_msg.orientation.y = quat[1];
        pose_msg.orientation.z = quat[2];
        pose_msg.orientation.w = quat[3];

        posePublisher.publish(pose_msg);

        Results.show(curr_frame, VO.prev_inliers, VO.curr_inliers, FPS, VO.t_f);
    }
}


// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
void getQuaternion(const cv::Mat& R, float Q[])
{
    float trace = R.at<float>(0,0) + R.at<float>(1,1) + R.at<float>(2,2);

    if (trace > 0.0)
    {
        float s = sqrtf(trace + 1.0);
        Q[3] = (s * 0.5);
        s = 0.5 / s;
        Q[0] = ((R.at<float>(2,1) - R.at<float>(1,2)) * s);
        Q[1] = ((R.at<float>(0,2) - R.at<float>(2,0)) * s);
        Q[2] = ((R.at<float>(1,0) - R.at<float>(0,1)) * s);
    }

    else
    {
        int i = R.at<float>(0,0) < R.at<float>(1,1) ? (R.at<float>(1,1) < R.at<float>(2,2) ? 2 : 1) : (R.at<float>(0,0) < R.at<float>(2,2) ? 2 : 0);
        int j = (i + 1) % 3;
        int k = (i + 2) % 3;

        float s = sqrtf(R.at<float>(i, i) - R.at<float>(j,j) - R.at<float>(k,k) + 1.0);
        Q[i] = s * 0.5;
        s = 0.5 / s;

        Q[3] = (R.at<float>(k,j) - R.at<float>(j,k)) * s;
        Q[j] = (R.at<float>(j,i) + R.at<float>(i,j)) * s;
        Q[k] = (R.at<float>(k,i) + R.at<float>(i,k)) * s;
    }
}
