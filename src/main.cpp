#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#include "scale.hpp"
#include "GRIC.hpp"
#include "pose.hpp"
#include "tracker.hpp"
#include "detector.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{

    if (argc<3)
    {
        printf("Few input arguments!\nUsage: ./exe pathToDataFolder paramsFile\n");
        return -1;
    }

    string data_dir = argv[1];
    string paramsFileName = argv[2];

    Parameters p;
    read_params(paramsFileName, p);

    gridFASTdetector Detector(p.nFeatures, p.threshold, p.nRows, p.nCols);
    tracker KLT(p.winSize);
    scaleEstimator Scale(p.cameraHeight, p.cameraPitchAngle);
    Viewer results;

    Mat prev_frame, curr_frame, F, H, R, t, mask, _3dPoints;
    vector<Point2f> prev_keypoints, curr_keypoints, prev_inliers, curr_inliers;
    clock_t start, finish;
    Mat R_f = Mat::eye(3,3,CV_32FC1), t_f = Mat::zeros(3,1,CV_32FC1), traj = Mat::zeros(800, 800, CV_8UC3);
    float scale_factor, Fcriteria, Hcriteria;
    int FPS, sum_fps = 0;
    unsigned int i = 1;


    printf("\n\rProcessing Frame %d", 0);
    fflush(stdout);

    char text[100];
    sprintf(text,"%06d.png",0);
    prev_frame = imread(data_dir+text, CV_LOAD_IMAGE_GRAYSCALE);
    if (prev_frame.empty())
    {
        printf("\nCan't read %s!\n", text);
        return -1;
    }

    Detector.detect(prev_frame, prev_keypoints);

    for (;;)
    {

        sprintf(text,"%06d.png",i);
        curr_frame = imread(data_dir+text,CV_LOAD_IMAGE_GRAYSCALE);

        if (curr_frame.empty())
            break;

        start = clock();

        KLT.track(prev_frame, curr_frame, prev_keypoints, curr_keypoints);

        ///Parallel Homo and F-matrix estimation
        #pragma omp parallel sections
        {
            #pragma omp section
            F = findFundamentalMat(prev_keypoints, curr_keypoints, mask, FM_RANSAC, 1, 0.99);
            #pragma omp section
            H = findHomography(prev_keypoints, curr_keypoints, CV_RANSAC, 10);
        }

        GRIC(prev_keypoints, curr_keypoints, prev_keypoints.size(), F, H, 0.8, Fcriteria, Hcriteria);

        prev_inliers.clear();
        curr_inliers.clear();

        if (Fcriteria < Hcriteria)
        {

            printf("\rProcessing Frame %d", i);
            fflush(stdout);

            ///Refine Fundamental Matrix
            for (size_t j=0; j<prev_keypoints.size(); j++)
            {
                if (mask.at<uchar>(j)==1)
                {
                    prev_inliers.push_back(prev_keypoints[j]);
                    curr_inliers.push_back(curr_keypoints[j]);
                }
            }

            F = findFundamentalMat(prev_inliers, curr_inliers, mask, FM_RANSAC, 0.5, 0.99);

            recoverPose(F, prev_inliers, curr_inliers, p.cameraMatrix, R, t, mask, _3dPoints);

            /// Recover Scale from camera height and pitch angle
            scale_factor = Scale.estimate(_3dPoints);

            t_f = t_f + scale_factor*(R_f*t);
            R_f = R*R_f;

            Detector.detect(curr_frame, prev_keypoints);

            curr_frame.copyTo(prev_frame);

        }
        else
            printf("\nSkipping Frame %d!\n", i);

        finish = clock();
        FPS = 1000 / (1000*(finish-start)/CLOCKS_PER_SEC);
        sum_fps+=FPS;


        results.show(curr_frame, prev_inliers, curr_inliers, FPS, t_f);
        i++;
    }

    printf("\nFinished\n");
    printf("Average FPS: %.2f\n", 1.0f*sum_fps/i);
    destroyAllWindows();
    return 0;
}
