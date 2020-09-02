#include "GRIC.hpp"


// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
float calc_GRIC(float *res, float sigma, int n, int model)
{
    register int i;
    int K, D; // number of parameters, dimension of the manifold
    const int R=4; // data dimension (image point pairs)
    const float sigmasq1=1.0/(sigma*sigma);
    float lam3RD;
    register float sum;

    switch(model)
    {
    case 0: // fundamental
        K=7;
        D=3;
        break;

    case 1: // homography
        K=8;
        D=2;
        break;

    default:
        fprintf(stderr, "unkown model %d specified to calc_GRIC()\n", model);
        exit(1);
    }

    lam3RD=2.0*(R-D); // lam3==2.0
    float tmp;
    for(i=0, sum=0.0; i<n; ++i)
    {
        tmp=res[i]*sigmasq1;
        sum+=(tmp<=lam3RD)? tmp : lam3RD;
    }

    sum+=n*D*log((float)R) + K*log((float)(R*n));

    return sum;
}


// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
void sampsonF_dsqr(std::vector<float> &F, std::vector<cv::Point2f> &pts0,
                   std::vector<cv::Point2f> &pts1, int npts, float *res)
{
    register int i;
    float  Fm0[3], Ftm1[3], m1Fm0;
    float m0_x, m0_y, m1_x, m1_y;

    for(i=0; i<npts; ++i)
    {
        m0_x = pts0[i].x;
        m0_y = pts0[i].y;

        m1_x = pts1[i].x;
        m1_y = pts1[i].y;

        /* F*m0 */
        Fm0[0]=F[0]*m0_x + F[1]*m0_y + F[2];
        Fm0[1]=F[3]*m0_x + F[4]*m0_y + F[5];
        Fm0[2]=F[6]*m0_x + F[7]*m0_y + F[8];

        /* F'*m1 */
        Ftm1[0]=F[0]*m1_x + F[3]*m1_y + F[6];
        Ftm1[1]=F[1]*m1_x + F[4]*m1_y + F[7];
        Ftm1[2]=F[2]*m1_x + F[5]*m1_y + F[8];

        /* m1'*F*m0 */
        m1Fm0=Fm0[0]*m1_x + Fm0[1]*m1_y +  Fm0[2];

        res[i]=m1Fm0*m1Fm0/(Fm0[0]*Fm0[0] + Fm0[1]*Fm0[1] + Ftm1[0]*Ftm1[0] + Ftm1[1]*Ftm1[1]);
    }
}


// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
void sampsonH_dsqr(std::vector<float> &H, std::vector<cv::Point2f> &pts0,
                   std::vector<cv::Point2f> &pts1, int npts, float *res)
{
    register int i;
    float m0_x, m0_y, m1_x, m1_y;
    float G0[3], G1[3], magG0, magG1, magG0G1, alpha, D1, D2, alg[2];

    for(i=0; i<npts; ++i)
    {
        m0_x = pts0[i].x;
        m0_y = pts0[i].y;

        m1_x = pts1[i].x;
        m1_y = pts1[i].y;

        G0[0]= H[0] - m1_x * H[6];
        G0[1]= H[1] - m1_x * H[7];
        G0[2]=-m0_x * H[6] - m0_y * H[7] - H[8];

        G1[0]= H[3] - m1_y * H[6];
        G1[1]= H[4] - m1_y * H[7];
        G1[2]=-m0_x * H[6] - m0_y * H[7] - H[8];

        magG0=sqrt(G0[0]*G0[0] + G0[1]*G0[1] + G0[2]*G0[2]);
        magG1=sqrt(G1[0]*G1[0] + G1[1]*G1[1] + G1[2]*G1[2]);
        magG0G1=G0[0]*G1[0] + G0[1]*G1[1];

        alpha=acos(magG0G1 /(magG0*magG1));

        /* algebraic distance */
        alg[0]=   m0_x*H[0] + m0_y*H[1] + H[2] -
                  m1_x*(m0_x*H[6] + m0_y*H[7] + H[8]);

        alg[1]=   m0_x*H[3] + m0_y*H[4] + H[5] -
                  m1_y*(m0_x*H[6] + m0_y*H[7] + H[8]);

        D1=alg[0]/magG0;
        D2=alg[1]/magG1;

        res[i]=(D1*D1 + D2*D2 - 2.0*D1*D2*cosf(alpha))/sinf(alpha);
    }
}


// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
void GRIC(std::vector<cv::Point2f> &pts0, std::vector<cv::Point2f> &pts1, int nmatches,
          cv::Mat &Fundamental, cv::Mat &Homography, float sigma, float &gricF, float &gricH)
{

    Fundamental.convertTo(Fundamental, CV_32F);
    Homography.convertTo(Homography, CV_32F);

    std::vector<float> F, H;
    F.assign((float*)Fundamental.datastart, (float*)Fundamental.dataend);
    H.assign((float*)Homography.datastart, (float*)Homography.dataend);

    float *res;

    res=(float *)malloc(nmatches*sizeof(float));
    if(!res)
    {
        fprintf(stderr, "memory allocation request failed in fundest_GRIC()\n");
        exit(1);
    }

    sampsonF_dsqr(F, pts0, pts1, nmatches, res);
    gricF=calc_GRIC(res, sigma, nmatches, 0); ///Fundamental


    sampsonH_dsqr(H, pts0, pts1, nmatches, res);
    gricH=calc_GRIC(res, sigma, nmatches, 1); ///Homography

    free(res);
}
