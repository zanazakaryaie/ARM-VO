#include "ARM_VO.hpp"
#include <time.h>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    if (argc<3)
    {
        cerr << "Few input arguments!" << endl;
        cout << "Usage: ./exe pathToDataFolder paramsFile" << endl;
        return -1;
    }

    string imageDir = argv[1];
    string paramsFileName = argv[2];

    ARM_VO VO(paramsFileName);
    Viewer Results;

    unsigned int FPS, sum_fps = 0;
    unsigned int imageCounter = 0;

    char imageName[100];

    for (;;)
    {
        sprintf(imageName,"%06d.png",imageCounter);
        Mat curr_frame = imread(imageDir+imageName, 0); //Load as grayscale

        cout << "Processing image " << imageName << ": ";

        if (curr_frame.empty())
        {
            cerr << "Can't read the image " << endl;
            break;
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
            FPS = 1000 / (1000*(finish-start)/CLOCKS_PER_SEC);
            sum_fps+=FPS;

            Results.show(curr_frame, VO.prev_inliers, VO.curr_inliers, FPS, VO.t_f);
        }

        imageCounter++;
    }

    cout << "Finished" << endl;
    cout << "Average FPS: " << setprecision(2) << 1.0f*sum_fps/imageCounter << endl;

    destroyAllWindows();

    return 1;
}
