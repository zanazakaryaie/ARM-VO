#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Few input arguments!" << std::endl;
        std::cout << "Usage: rosrun ARM_VO ImagePublisher pathToDataFolder" << std::endl;
        return -1;
    }

    std::string imageDir = argv[1];
    char imageName[100];
    unsigned int imageCounter = 0;

    ros::init(argc, argv, "ImagePublisher");
    ros::NodeHandle nh;

    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("kitti/image", 1);

    ros::Rate loop_rate(10);

    while (nh.ok())
    {
        sprintf(imageName,"%06d.png",imageCounter);
        cv::Mat image = cv::imread(imageDir+imageName, 0); //Load as grayscale

        if (image.empty())
        {
            std::cerr << "Can't read " << imageName << std::endl;
            break;
        }

        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", image).toImageMsg();

        pub.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();

        imageCounter++;
    }

    return 1;
}
