#include "ARM_VO_Node.hpp"

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Few input arguments!" << std::endl;
        std::cout << "Usage: rosrun ARM_VO ARM_VO paramsFile" << std::endl;
        return -1;
    }

    ros::init(argc, argv, "ARM_VO");
    ros::NodeHandle nh;

    ARM_VO_Node node = ARM_VO_Node(nh, argv[1]);

    ros::spin();

    cv::destroyAllWindows();

    return 1;
}
