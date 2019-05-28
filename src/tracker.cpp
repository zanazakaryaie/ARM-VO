#include "tracker.hpp"
#include "lkpyramid.hpp"

tracker::tracker(int windowSize)
{
    winSize = Size(windowSize, windowSize);
    termcrit =TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
}


void tracker::calcSharrDeriv(const Mat& src, Mat& dst)
{
    using cv::detail::deriv_type;
    const int rows = src.rows, cols = src.cols, cn = 1, colsn = cols;

    dst.create(rows, cols, 11);

    unsigned int x, y, delta = (unsigned int)alignSize(cols + 2, 16);

    AutoBuffer<deriv_type> _tempBuf(delta*2 + 64);
    deriv_type *trow0 = alignPtr(_tempBuf + cn, 16), *trow1 = alignPtr(trow0 + delta, 16);


    unsigned int x0 = 1, x1 = cols-2;

    for( y = 0; y < rows; y++ )
    {
        const uchar* srow0 = src.ptr<uchar>(y > 0 ? y-1 : rows > 1 ? 1 : 0);
        const uchar* srow1 = src.ptr<uchar>(y);
        const uchar* srow2 = src.ptr<uchar>(y < rows-1 ? y+1 : rows > 1 ? rows-2 : 0);
        deriv_type* drow = dst.ptr<deriv_type>(y);

        // do vertical convolution
        x = 0;

        const uint16x8_t q8 = vdupq_n_u16(3);
        const uint8x8_t d18 = vdup_n_u8(10);

        const int16x8_t q8i = vdupq_n_s16(3);
        const int16x8_t q9 = vdupq_n_s16(10);

        for( ; x <= colsn - 8; x += 8)
        {
            uint8x8_t d0 = vld1_u8((const uint8_t*)&srow0[x]);
            uint8x8_t d1 = vld1_u8((const uint8_t*)&srow1[x]);
            uint8x8_t d2 = vld1_u8((const uint8_t*)&srow2[x]);
            uint16x8_t q4 = vaddl_u8(d0, d2);
            uint16x8_t q11 = vsubl_u8(d2, d0);
            uint16x8_t q5 = vmulq_u16(q4, q8);
            uint16x8_t q6 = vmull_u8(d1, d18);
            uint16x8_t q10 = vaddq_u16(q6, q5);
            vst1q_u16((uint16_t*)&trow0[x], q10);
            vst1q_u16((uint16_t*)&trow1[x], q11);

        }

        for( ; x < colsn; x++ )
        {
            int t0 = (srow0[x] + srow2[x])*3 + srow1[x]*10;
            int t1 = srow2[x] - srow0[x];
            trow0[x] = (deriv_type)t0;
            trow1[x] = (deriv_type)t1;
        }

        trow0[-cn] = trow0[x0];
        trow0[colsn] = trow0[x1];
        trow1[-cn] = trow1[x0];
        trow1[colsn] = trow1[x1];

        // do horizontal convolution, interleave the results and store them to dst
        x = 0;

        for( ; x <= colsn - 8; x += 8 )
        {

            int16x8_t q0 = vld1q_s16((const int16_t*)&trow0[x+cn]);
            int16x8_t q1 = vld1q_s16((const int16_t*)&trow0[x-cn]);
            int16x8_t q2 = vld1q_s16((const int16_t*)&trow1[x+cn]);
            int16x8_t q3 = vld1q_s16((const int16_t*)&trow1[x-cn]);
            int16x8_t q5 = vsubq_s16(q0, q1);
            int16x8_t q6 = vaddq_s16(q2, q3);
            int16x8_t q4 = vld1q_s16((const int16_t*)&trow1[x]);
            int16x8_t q7 = vmulq_s16(q6, q8i);
            int16x8_t q10 = vmulq_s16(q4, q9);
            int16x8_t q11 = vaddq_s16(q7, q10);
            int16x4_t d22 = vget_low_s16(q11);
            int16x4_t d23 = vget_high_s16(q11);
            int16x4_t d11 = vget_high_s16(q5);
            int16x4_t d10 = vget_low_s16(q5);
            int16x4x2_t q5x2, q11x2;
            q5x2.val[0] = d10;
            q5x2.val[1] = d22;
            q11x2.val[0] = d11;
            q11x2.val[1] = d23;
            vst2_s16((int16_t*)&drow[x*2], q5x2);
            vst2_s16((int16_t*)&drow[(x*2)+8], q11x2);

        }

        for( ; x < colsn; x++ )
        {
            deriv_type t0 = (deriv_type)(trow0[x+cn] - trow0[x-cn]);
            deriv_type t1 = (deriv_type)((trow1[x+cn] + trow1[x-cn])*3 + trow1[x]*10);
            drow[x*2] = t0;
            drow[x*2+1] = t1;
        }
    }
}



void tracker::calc_OpticalFlowPyrLK(InputArray _prevImg, InputArray _nextImg,
                            InputArray _prevPts, InputOutputArray _nextPts,
                            OutputArray _status,
                            Size winSize, int maxLevel,
                            TermCriteria criteria,
                            int flags, double minEigThreshold)
{
    Mat prevPtsMat = _prevPts.getMat();
    const int derivDepth = DataType<cv::detail::deriv_type>::depth;

    unsigned int npoints = prevPtsMat.checkVector(2, CV_32F, true);

    Mat nextPtsMat = _nextPts.getMat();

    const Point2f* prevPts = (const Point2f*)prevPtsMat.data;
    Point2f* nextPts = (Point2f*)nextPtsMat.data;

    _status.create((int)npoints, 1, CV_8U, -1, true);
    Mat statusMat = _status.getMat(), errMat;
    uchar* status = statusMat.data;

    for(unsigned int i = 0; i < npoints; i++ )
        status[i] = true;

    vector<Mat> prevPyr, nextPyr;

    maxLevel = buildOpticalFlowPyramid(_prevImg, prevPyr, winSize, maxLevel, false);
    maxLevel = buildOpticalFlowPyramid(_nextImg, nextPyr, winSize, maxLevel, false);

    criteria.epsilon *= criteria.epsilon;

    Mat derivIBuf;
    derivIBuf.create(prevPyr[0].rows + winSize.height*2, prevPyr[0].cols + winSize.width*2, CV_MAKETYPE(derivDepth, prevPyr[0].channels() * 2));

    for(int level = maxLevel; level >= 0; level-- )
    {
        Mat derivI;

        Size imgSize = prevPyr[level].size();
        Mat _derivI( imgSize.height + winSize.height*2,
                     imgSize.width + winSize.width*2, derivIBuf.type(), derivIBuf.data );
        derivI = _derivI(Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));

        calcSharrDeriv(prevPyr[level], derivI);
        copyMakeBorder(derivI, _derivI, winSize.height, winSize.height, winSize.width, winSize.width, BORDER_CONSTANT|BORDER_ISOLATED);

        typedef cv::detail::LKTrackerInvoker LKTrackerInvoker;

        parallel_for_(Range(0, npoints), LKTrackerInvoker(prevPyr[level], derivI,
                      nextPyr[level], prevPts, nextPts,
                      status,
                      winSize, criteria, level, maxLevel,
                      flags, (float)minEigThreshold));
    }

}



void tracker::track(const Mat &img_1, const Mat &img_2, vector<Point2f>& points1, vector<Point2f>& points2)
{
    vector<uchar> status;
    points2 = points1;
    calc_OpticalFlowPyrLK(img_1, img_2, points1, points2, status, winSize, 3, termcrit, OPTFLOW_USE_INITIAL_FLOW, 0.001);
    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    unsigned int indexCorrection = 0;
    Point2f pt;
    for( size_t i=0; i<status.size(); i++)
    {
        pt = points2[i- indexCorrection];
        if ((status[i] == 0)||(pt.x<0)||(pt.y<0))
        {
            points1.erase (points1.begin() + i - indexCorrection);
            points2.erase (points2.begin() + i - indexCorrection);
            indexCorrection++;
        }

    }
}
