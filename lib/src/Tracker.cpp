#include "Tracker.hpp"

#include <cassert>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>

#ifdef ARM
#include <arm_neon.h>

namespace
{

#define CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

class LKTrackerInvoker
{
    const cv::Mat* mPrevImg;
    const cv::Mat* mNextImg;
    const cv::Mat* mPrevDeriv;
    const cv::Point2f* mPrevPts;
    cv::Point2f* mNextPts;
    uchar* mStatus;
    cv::Size mWinSize;
    const cv::TermCriteria mCriteria{cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.0001};
    int mLevel;
    int mMaxLevel;
    const float mMinEigThreshold = 1e-4;

    const float FLT_SCALE = 1.f/(1 << 20);
    static const int W_BITS = 14;
    static const int W_BITS1 = 14;

public:

    LKTrackerInvoker(const cv::Mat& prevImg, const cv::Mat& prevDeriv, const cv::Mat& nextImg,
                     const cv::Point2f* prevPts, cv::Point2f* nextPts,
                     uchar* status,
                     cv::Size winSize,
                     int level, int maxLevel)
        : mPrevImg(&prevImg), mNextImg(&nextImg), mPrevDeriv(&prevDeriv),
          mPrevPts(prevPts), mNextPts(nextPts), mStatus(status), mWinSize(winSize),
          mLevel(level), mMaxLevel(maxLevel) {}

    void invoke(const cv::Range& range) const
    {
        cv::Point2f halfWin((mWinSize.width-1)*0.5f, (mWinSize.height-1)*0.5f);
        const cv::Mat& I = *mPrevImg;
        const cv::Mat& J = *mNextImg;
        const cv::Mat& derivI = *mPrevDeriv;
        const int cn = 1;
        const int cn2 = 2;
        int j;
        cv::AutoBuffer<int16_t> _buf(mWinSize.area()*(cn + cn2));

        cv::Mat IWinBuf(mWinSize, CV_16SC1, _buf.data());
        cv::Mat derivIWinBuf(mWinSize, CV_16SC2, _buf.data() + mWinSize.area());

        const int dstep = (int)(derivI.step/derivI.elemSize1());
        const int stepI = (int)(I.step/I.elemSize1());
        const int stepJ = (int)(J.step/J.elemSize1());

        for (int ptidx = range.start; ptidx < range.end; ptidx++ )
        {
            cv::Point2f prevPt = mPrevPts[ptidx]*(1.f/(1 << mLevel));
            cv::Point2f nextPt;

            if (mLevel == mMaxLevel)
            {
                nextPt = mNextPts[ptidx]*(1.f/(1 << mLevel));
            }
            else
            {
                nextPt = mNextPts[ptidx]*2.f;
            }

            mNextPts[ptidx] = nextPt;

            cv::Point2i iprevPt, inextPt;
            prevPt -= halfWin;
            iprevPt.x = cvFloor(prevPt.x);
            iprevPt.y = cvFloor(prevPt.y);

            if (iprevPt.x < -mWinSize.width || iprevPt.x >= derivI.cols ||
                iprevPt.y < -mWinSize.height || iprevPt.y >= derivI.rows )
            {
                if (mLevel == 0)
                {
                    mStatus[ptidx] = false;
                }
                continue;
            }

            float a = prevPt.x - iprevPt.x;
            float b = prevPt.y - iprevPt.y;

            int iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
            int iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
            int iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
            int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

            float iA11 = 0, iA12 = 0, iA22 = 0;

            float CV_DECL_ALIGNED(16) nA11[] = {0, 0, 0, 0}, nA12[] = {0, 0, 0, 0}, nA22[] = {0, 0, 0, 0};
            const int shifter1 = -(W_BITS - 5); //negative so it shifts right
            const int shifter2 = -(W_BITS);

            const int16x4_t d26 = vdup_n_s16((int16_t)iw00);
            const int16x4_t d27 = vdup_n_s16((int16_t)iw01);
            const int16x4_t d28 = vdup_n_s16((int16_t)iw10);
            const int16x4_t d29 = vdup_n_s16((int16_t)iw11);
            const int32x4_t q11 = vdupq_n_s32((int32_t)shifter1);
            const int32x4_t q12 = vdupq_n_s32((int32_t)shifter2);

            // extract the patch from the first image, compute covariation matrix of derivatives
            int x, y;
            for (y = 0; y < mWinSize.height; y++)
            {
                const uchar* src = I.ptr() + (y + iprevPt.y)*stepI + iprevPt.x;
                const int16_t* dsrc = derivI.ptr<int16_t>() + (y + iprevPt.y)*dstep + iprevPt.x*2;

                int16_t* Iptr = IWinBuf.ptr<int16_t>(y);
                int16_t* dIptr = derivIWinBuf.ptr<int16_t>(y);

                x = 0;

                for (; x <= mWinSize.width - 4; x += 4, dsrc += 4*2, dIptr += 4*2)
                {
                    uint8x8_t d0 = vld1_u8(&src[x]);
                    uint8x8_t d2 = vld1_u8(&src[x+cn]);
                    uint16x8_t q0 = vmovl_u8(d0);
                    uint16x8_t q1 = vmovl_u8(d2);

                    int32x4_t q5 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q0)), d26);
                    int32x4_t q6 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q1)), d27);

                    uint8x8_t d4 = vld1_u8(&src[x + stepI]);
                    uint8x8_t d6 = vld1_u8(&src[x + stepI + cn]);
                    uint16x8_t q2 = vmovl_u8(d4);
                    uint16x8_t q3 = vmovl_u8(d6);

                    int32x4_t q7 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q2)), d28);
                    int32x4_t q8 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q3)), d29);

                    q5 = vaddq_s32(q5, q6);
                    q7 = vaddq_s32(q7, q8);
                    q5 = vaddq_s32(q5, q7);

                    int16x4x2_t d0d1 = vld2_s16(dsrc);
                    int16x4x2_t d2d3 = vld2_s16(&dsrc[cn2]);

                    q5 = vqrshlq_s32(q5, q11);

                    int32x4_t q4 = vmull_s16(d0d1.val[0], d26);
                    q6 = vmull_s16(d0d1.val[1], d26);

                    int16x4_t nd0 = vmovn_s32(q5);

                    q7 = vmull_s16(d2d3.val[0], d27);
                    q8 = vmull_s16(d2d3.val[1], d27);

                    vst1_s16(&Iptr[x], nd0);

                    int16x4x2_t d4d5 = vld2_s16(&dsrc[dstep]);
                    int16x4x2_t d6d7 = vld2_s16(&dsrc[dstep+cn2]);

                    q4 = vaddq_s32(q4, q7);
                    q6 = vaddq_s32(q6, q8);

                    q7 = vmull_s16(d4d5.val[0], d28);
                    int32x4_t q14 = vmull_s16(d4d5.val[1], d28);
                    q8 = vmull_s16(d6d7.val[0], d29);
                    int32x4_t q15 = vmull_s16(d6d7.val[1], d29);

                    q7 = vaddq_s32(q7, q8);
                    q14 = vaddq_s32(q14, q15);

                    q4 = vaddq_s32(q4, q7);
                    q6 = vaddq_s32(q6, q14);

                    float32x4_t nq0 = vld1q_f32(nA11);
                    float32x4_t nq1 = vld1q_f32(nA12);
                    float32x4_t nq2 = vld1q_f32(nA22);

                    q4 = vqrshlq_s32(q4, q12);
                    q6 = vqrshlq_s32(q6, q12);

                    q7 = vmulq_s32(q4, q4);
                    q8 = vmulq_s32(q4, q6);
                    q15 = vmulq_s32(q6, q6);

                    nq0 = vaddq_f32(nq0, vcvtq_f32_s32(q7));
                    nq1 = vaddq_f32(nq1, vcvtq_f32_s32(q8));
                    nq2 = vaddq_f32(nq2, vcvtq_f32_s32(q15));

                    vst1q_f32(nA11, nq0);
                    vst1q_f32(nA12, nq1);
                    vst1q_f32(nA22, nq2);

                    int16x4_t d8 = vmovn_s32(q4);
                    int16x4_t d12 = vmovn_s32(q6);

                    int16x4x2_t d8d12;
                    d8d12.val[0] = d8;
                    d8d12.val[1] = d12;
                    vst2_s16(dIptr, d8d12);
                }

                for (; x < mWinSize.width; x++, dsrc += 2, dIptr += 2)
                {
                    int ival = CV_DESCALE(src[x]*iw00 + src[x+cn]*iw01 +
                                          src[x+stepI]*iw10 + src[x+stepI+cn]*iw11, W_BITS1-5);
                    int ixval = CV_DESCALE(dsrc[0]*iw00 + dsrc[cn2]*iw01 +
                                           dsrc[dstep]*iw10 + dsrc[dstep+cn2]*iw11, W_BITS1);
                    int iyval = CV_DESCALE(dsrc[1]*iw00 + dsrc[cn2+1]*iw01 + dsrc[dstep+1]*iw10 +
                                           dsrc[dstep+cn2+1]*iw11, W_BITS1);

                    Iptr[x] = (short)ival;
                    dIptr[0] = (short)ixval;
                    dIptr[1] = (short)iyval;

                    iA11 += (float)(ixval*ixval);
                    iA12 += (float)(ixval*iyval);
                    iA22 += (float)(iyval*iyval);
                }
            }


            iA11 += (nA11[0] + nA11[1] + nA11[2] + nA11[3]);
            iA12 += (nA12[0] + nA12[1] + nA12[2] + nA12[3]);
            iA22 += (nA22[0] + nA22[1] + nA22[2] + nA22[3]);

            float A11 = iA11*FLT_SCALE;
            float A12 = iA12*FLT_SCALE;
            float A22 = iA22*FLT_SCALE;

            float D = A11*A22 - A12*A12;
            const float minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                                  4.f*A12*A12))/(2*mWinSize.width*mWinSize.height);



            if (minEig < mMinEigThreshold || D < FLT_EPSILON)
            {
                if (mLevel == 0)
                {
                    mStatus[ptidx] = false;
                }
                continue;
            }

            D = 1.f/D;

            nextPt -= halfWin;
            cv::Point2f prevDelta;

            for (j = 0; j < mCriteria.maxCount; j++)
            {
                inextPt.x = cvFloor(nextPt.x);
                inextPt.y = cvFloor(nextPt.y);

                if (inextPt.x < -mWinSize.width || inextPt.x >= J.cols ||
                    inextPt.y < -mWinSize.height || inextPt.y >= J.rows)
                {
                    if (mLevel == 0)
                    {
                        mStatus[ptidx] = false;
                    }
                    break;
                }

                a = nextPt.x - inextPt.x;
                b = nextPt.y - inextPt.y;
                iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
                iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
                iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
                iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
                float ib1 = 0, ib2 = 0;

                float CV_DECL_ALIGNED(16) nB1[] = {0,0,0,0}, nB2[] = {0,0,0,0};

                const int16x4_t d26_2 = vdup_n_s16((int16_t)iw00);
                const int16x4_t d27_2 = vdup_n_s16((int16_t)iw01);
                const int16x4_t d28_2 = vdup_n_s16((int16_t)iw10);
                const int16x4_t d29_2 = vdup_n_s16((int16_t)iw11);

                for (y = 0; y < mWinSize.height; y++)
                {
                    const uchar* Jptr = J.ptr() + (y + inextPt.y)*stepJ + inextPt.x;
                    const int16_t* Iptr = IWinBuf.ptr<int16_t>(y);
                    const int16_t* dIptr = derivIWinBuf.ptr<int16_t>(y);

                    x = 0;

                    for (; x <= mWinSize.width - 8; x += 8, dIptr += 8*2)
                    {
                        uint8x8_t d0 = vld1_u8(&Jptr[x]);
                        uint8x8_t d2 = vld1_u8(&Jptr[x+cn]);
                        uint8x8_t d4 = vld1_u8(&Jptr[x+stepJ]);
                        uint8x8_t d6 = vld1_u8(&Jptr[x+stepJ+cn]);

                        uint16x8_t q0 = vmovl_u8(d0);
                        uint16x8_t q1 = vmovl_u8(d2);
                        uint16x8_t q2 = vmovl_u8(d4);
                        uint16x8_t q3 = vmovl_u8(d6);

                        int32x4_t nq4 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q0)), d26_2);
                        int32x4_t nq5 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q0)), d26_2);

                        int32x4_t nq6 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q1)), d27_2);
                        int32x4_t nq7 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q1)), d27_2);

                        int32x4_t nq8 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q2)), d28_2);
                        int32x4_t nq9 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q2)), d28_2);

                        int32x4_t nq10 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q3)), d29_2);
                        int32x4_t nq11 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q3)), d29_2);

                        nq4 = vaddq_s32(nq4, nq6);
                        nq5 = vaddq_s32(nq5, nq7);
                        nq8 = vaddq_s32(nq8, nq10);
                        nq9 = vaddq_s32(nq9, nq11);

                        int16x8_t q6 = vld1q_s16(&Iptr[x]);

                        nq4 = vaddq_s32(nq4, nq8);
                        nq5 = vaddq_s32(nq5, nq9);

                        nq8 = vmovl_s16(vget_high_s16(q6));
                        nq6 = vmovl_s16(vget_low_s16(q6));

                        nq4 = vqrshlq_s32(nq4, q11);
                        nq5 = vqrshlq_s32(nq5, q11);

                        int16x8x2_t q0q1 = vld2q_s16(dIptr);
                        float32x4_t nB1v = vld1q_f32(nB1);
                        float32x4_t nB2v = vld1q_f32(nB2);

                        nq4 = vsubq_s32(nq4, nq6);
                        nq5 = vsubq_s32(nq5, nq8);

                        int32x4_t nq2 = vmovl_s16(vget_low_s16(q0q1.val[0]));
                        int32x4_t nq3 = vmovl_s16(vget_high_s16(q0q1.val[0]));

                        nq7 = vmovl_s16(vget_low_s16(q0q1.val[1]));
                        nq8 = vmovl_s16(vget_high_s16(q0q1.val[1]));

                        nq9 = vmulq_s32(nq4, nq2);
                        nq10 = vmulq_s32(nq5, nq3);

                        nq4 = vmulq_s32(nq4, nq7);
                        nq5 = vmulq_s32(nq5, nq8);

                        nq9 = vaddq_s32(nq9, nq10);
                        nq4 = vaddq_s32(nq4, nq5);

                        nB1v = vaddq_f32(nB1v, vcvtq_f32_s32(nq9));
                        nB2v = vaddq_f32(nB2v, vcvtq_f32_s32(nq4));

                        vst1q_f32(nB1, nB1v);
                        vst1q_f32(nB2, nB2v);
                    }


                    for (; x < mWinSize.width; x++, dIptr += 2)
                    {
                        int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                              Jptr[x+stepJ]*iw10 + Jptr[x+stepJ+cn]*iw11,
                                              W_BITS1-5) - Iptr[x];
                        ib1 += (float)(diff*dIptr[0]);
                        ib2 += (float)(diff*dIptr[1]);
                    }
                }


                ib1 += (float)(nB1[0] + nB1[1] + nB1[2] + nB1[3]);
                ib2 += (float)(nB2[0] + nB2[1] + nB2[2] + nB2[3]);


                float b1 = ib1*FLT_SCALE;
                float b2 = ib2*FLT_SCALE;

                cv::Point2f delta ((float)((A12*b2 - A22*b1) * D),
                                   (float)((A12*b1 - A11*b2) * D));

                nextPt += delta;
                mNextPts[ptidx] = nextPt + halfWin;

                if (delta.ddot(delta) <= mCriteria.epsilon)
                {
                    break;
                }

                if (j > 0 && std::abs(delta.x + prevDelta.x) < 0.01 &&
                    std::abs(delta.y + prevDelta.y) < 0.01)
                {
                    mNextPts[ptidx] -= delta*0.5f;
                    break;
                }
                prevDelta = delta;
            }
        }
    }
};

}

#endif

namespace armvo
{

class KeypointTracker::Impl
{
    const KeypointTrackerConfig mConfig;
    int mMaxLevel = 3;
    int mRows;
    int mCols;

#ifdef ARM
    std::vector<cv::Mat> mPrevPyr;
    std::vector<cv::Mat> mCurrPyr;
#else
    cv::Mat mPrevImg;
    cv::Mat mCurrImg;
#endif

public:

    Impl(const KeypointTrackerConfig& config) : mConfig(config)
    {
        if (config.windowSize <= 2)
        {
            throw std::invalid_argument("Tracker's window size must be at least 3");
        }
        if (config.windowSize % 2 != 1)
        {
            throw std::invalid_argument("Tracker's window size must be an odd number");
        }
    }

    void setPreviousFrame(const cv::Mat& image)
    {
        mRows = image.rows;
        mCols = image.cols;
#ifdef ARM
        mMaxLevel = 3;
        mMaxLevel = cv::buildOpticalFlowPyramid(image, mPrevPyr, cv::Size(mConfig.windowSize, mConfig.windowSize), mMaxLevel, false);
#else
        image.copyTo(mPrevImg);
#endif
    }

    void setCurrentFrame(const cv::Mat& image)
    {
#ifdef ARM
        mMaxLevel = cv::buildOpticalFlowPyramid(image, mCurrPyr, cv::Size(mConfig.windowSize, mConfig.windowSize), mMaxLevel, false);
#else
        image.copyTo(mCurrImg);
#endif
    }

    std::vector<cv::Point2f> trackForward(std::vector<cv::Point2f>& points)
    {
        std::vector<cv::Point2f> trackedPoints;
#ifdef ARM
        trackedPoints = track(mPrevPyr, mCurrPyr, points);
#else
        trackedPoints = track(mPrevImg, mCurrImg, points);
#endif
        return trackedPoints;
    }

    std::vector<cv::Point2f> trackBackward(std::vector<cv::Point2f>& points)
    {
        std::vector<cv::Point2f> trackedPoints;
#ifdef ARM
        trackedPoints = track(mCurrPyr, mPrevPyr, points);
#else
        trackedPoints = track(mCurrImg, mPrevImg, points);
#endif
        return trackedPoints;
    }

    void step()
    {
#ifdef ARM
        mPrevPyr = std::move(mCurrPyr);
#else
        mPrevImg = std::move(mCurrImg);
#endif
    }

private:

#ifdef ARM

    std::vector<cv::Point2f> track(const std::vector<cv::Mat>& pyr1, const std::vector<cv::Mat>& pyr2, std::vector<cv::Point2f>& points) const
    {
        std::vector<uchar> status;
        const size_t numPoints = points.size();
        std::vector<cv::Point2f> trackedPoints = points;

        status.resize(numPoints, 1);

        cv::Mat derivIBuf;
        derivIBuf.create(pyr1[0].rows + mConfig.windowSize*2, pyr1[0].cols + mConfig.windowSize*2, CV_16SC2);

        const cv::Size winSize(mConfig.windowSize, mConfig.windowSize);

        for(int level = mMaxLevel; level >= 0; level--)
        {
            const cv::Size imgSize = pyr1[level].size();
            cv::Mat _derivI(imgSize.height + winSize.height*2, imgSize.width + winSize.width*2, CV_16SC2, derivIBuf.data);
            cv::Mat derivI = _derivI(cv::Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));

            calculateScharrDerivatives(pyr1[level], derivI);
            cv::copyMakeBorder(derivI, _derivI, winSize.height, winSize.height, winSize.width, winSize.width, cv::BORDER_CONSTANT|cv::BORDER_ISOLATED);

            assert(pyr1[level].size() == pyr2[level].size());
            assert(pyr1[level].type() == pyr2[level].type());

            LKTrackerInvoker invoker(pyr1[level], derivI,
                                     pyr2[level], points.data(), trackedPoints.data(),
                                     status.data(),
                                     winSize, level, mMaxLevel);

            cv::parallel_for_(cv::Range(0, numPoints), [=](const cv::Range& range){ invoker.invoke(range); });
        }
        deleteOutOfBoundTracks(points, trackedPoints, status);
        return trackedPoints;
    }

    static void calculateScharrDerivatives(const cv::Mat& src, cv::Mat& dst)
    {
        const int rows = src.rows;
        const int cols = src.cols;
        const int cn = 1;

        dst.create(rows, cols, CV_16SC2);

        const int delta = static_cast<int>(cv::alignSize(cols + 2, 16));

        cv::AutoBuffer<int16_t> tempBuffer(delta*2 + 64);
        int16_t *trow0 = cv::alignPtr(tempBuffer.data() + cn, 16);
        int16_t *trow1 = cv::alignPtr(trow0 + delta, 16);

        const int x0 = 1;
        const int x1 = cols-2;

        for (int y = 0; y < rows; y++)
        {
            const uchar* srow0 = src.ptr<uchar>(y > 0 ? y-1 : rows > 1 ? 1 : 0);
            const uchar* srow1 = src.ptr<uchar>(y);
            const uchar* srow2 = src.ptr<uchar>(y < rows-1 ? y+1 : rows > 1 ? rows-2 : 0);
            int16_t* drow = dst.ptr<int16_t>(y);

            // do vertical convolution
            int x = 0;

            const uint16x8_t q8 = vdupq_n_u16(3);
            const uint8x8_t d18 = vdup_n_u8(10);

            const int16x8_t q8i = vdupq_n_s16(3);
            const int16x8_t q9 = vdupq_n_s16(10);

            for ( ; x <= cols - 8; x += 8)
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

            // leftovers
            for (; x < cols; x++)
            {
                int t0 = (srow0[x] + srow2[x])*3 + srow1[x]*10;
                int t1 = srow2[x] - srow0[x];
                trow0[x] = (int16_t)t0;
                trow1[x] = (int16_t)t1;
            }

            trow0[-cn] = trow0[x0];
            trow0[cols] = trow0[x1];
            trow1[-cn] = trow1[x0];
            trow1[cols] = trow1[x1];

            // do horizontal convolution, interleave the results and store them to dst
            x = 0;

            for (; x <= cols - 8; x += 8)
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

            for (; x < cols; x++)
            {
                int16_t t0 = static_cast<int16_t>(trow0[x+cn] - trow0[x-cn]);
                int16_t t1 = static_cast<int16_t>((trow1[x+cn] + trow1[x-cn])*3 + trow1[x]*10);
                drow[x*2] = t0;
                drow[x*2+1] = t1;
            }
        }
    }

#else

    std::vector<cv::Point2f> track(const cv::Mat& frame1, const cv::Mat& frame2, std::vector<cv::Point2f>& points) const
    {
        std::vector<uchar> status;
        std::vector<cv::Point2f> trackedPoints = points;
        std::vector<float> error;
        cv::calcOpticalFlowPyrLK(frame1, frame2, points, trackedPoints, status, error,
                                 cv::Size(mConfig.windowSize, mConfig.windowSize), mMaxLevel,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.0001),
                                 cv::OPTFLOW_USE_INITIAL_FLOW);
        deleteOutOfBoundTracks(points, trackedPoints, status);
        return trackedPoints;
    }

#endif

    void deleteOutOfBoundTracks(std::vector<cv::Point2f>& points,
                                std::vector<cv::Point2f>& trackedPoints,
                                const std::vector<uchar>& status) const
    {
        size_t dst = 0;
        for (size_t src = 0; src < status.size(); src++)
        {
            if (status[src] && inBounds(trackedPoints[src]))
            {
                points[dst] = points[src];
                trackedPoints[dst] = trackedPoints[src];
                dst++;
            }
        }
        points.resize(dst);
        trackedPoints.resize(dst);
    }

    bool inBounds(const cv::Point2f& pt) const noexcept
    {
        return pt.x >= 0 && pt.x < mCols && pt.y >= 0 && pt.y < mRows;
    }
};

KeypointTracker::KeypointTracker(const KeypointTrackerConfig& config)
{
    mImpl = std::make_shared<Impl>(config);
}

void KeypointTracker::setPreviousFrame(const cv::Mat& image)
{
    mImpl->setPreviousFrame(image);
}

void KeypointTracker::setCurrentFrame(const cv::Mat& image)
{
    mImpl->setCurrentFrame(image);
}

std::vector<cv::Point2f> KeypointTracker::trackForward(std::vector<cv::Point2f>& points)
{
    return mImpl->trackForward(points);
}

std::vector<cv::Point2f> KeypointTracker::trackBackward(std::vector<cv::Point2f>& points)
{
    return mImpl->trackBackward(points);
}

void KeypointTracker::step()
{
    mImpl->step();
}

}
