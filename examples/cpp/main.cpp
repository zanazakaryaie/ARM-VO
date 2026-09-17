#include <armvo/ARM_VO.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: my_app config.yaml frame1.png [frame2.png ...]\n";
        return 1;
    }

    armvo::ArmVo vo(armvo::ArmVoConfig::load(argv[1]));
    armvo::Pose pose;
    for (int i = 2; i < argc; i++)
    {
        const auto frame = cv::imread(argv[i], cv::IMREAD_COLOR);

        if (!vo.isInitialized())
        {
            const auto status = vo.initialize(frame, pose);
            if (status != armvo::Status::SUCCESS)
            {
                std::cerr << "Initialization failed!" << std::endl;
                return 1;
            }
        }
        else
        {
            const auto status = vo.update(frame, pose);
            if (status != armvo::Status::SUCCESS && status != armvo::Status::FRAME_SKIPPED)
            {
                std::cerr << "Something went wrong!" << std::endl;
                return 1;
            }
        }

        std::cout << pose.rotation << '\n' << pose.translation << '\n';
    }
}
