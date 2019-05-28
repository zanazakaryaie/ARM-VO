#pragma once

#include <arm_neon.h>
#include "opencv2/video/tracking.hpp"

#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

namespace cv
{
namespace detail
{

typedef short deriv_type;

struct LKTrackerInvoker : ParallelLoopBody
{
    LKTrackerInvoker( const Mat& _prevImg, const Mat& _prevDeriv, const Mat& _nextImg,
                      const Point2f* _prevPts, Point2f* _nextPts,
                      uchar* _status,
                      Size _winSize, TermCriteria _criteria,
                      int _level, int _maxLevel, int _flags, float _minEigThreshold )
    {
        prevImg = &_prevImg;
        prevDeriv = &_prevDeriv;
        nextImg = &_nextImg;
        prevPts = _prevPts;
        nextPts = _nextPts;
        status = _status;
        winSize = _winSize;
        criteria = _criteria;
        level = _level;
        maxLevel = _maxLevel;
        flags = _flags;
        minEigThreshold = _minEigThreshold;
    }

    void operator()(const Range& range) const
    {
        Point2f halfWin((winSize.width-1)*0.5f, (winSize.height-1)*0.5f);
        const Mat& I = *prevImg;
        const Mat& J = *nextImg;
        const Mat& derivI = *prevDeriv;
        int j, cn = 1, cn2 = 2;
        cv::AutoBuffer<deriv_type> _buf(winSize.area()*(cn + cn2));
        int derivDepth = DataType<deriv_type>::depth;

        Mat IWinBuf(winSize, 3, (deriv_type*)_buf);
        Mat derivIWinBuf(winSize, 11, (deriv_type*)_buf + winSize.area());

        int dstep = (int)(derivI.step/derivI.elemSize1());
        int stepI = (int)(I.step/I.elemSize1());
        int stepJ = (int)(J.step/J.elemSize1());

        for( int ptidx = range.start; ptidx < range.end; ptidx++ )
        {
            Point2f prevPt = prevPts[ptidx]*(1.f/(1 << level));
            Point2f nextPt;

            if( level == maxLevel )
                nextPt = nextPts[ptidx]*(1.f/(1 << level));
            else
                nextPt = nextPts[ptidx]*2.f;

            nextPts[ptidx] = nextPt;

            Point2i iprevPt, inextPt;
            prevPt -= halfWin;
            iprevPt.x = cvFloor(prevPt.x);
            iprevPt.y = cvFloor(prevPt.y);

            if( iprevPt.x < -winSize.width || iprevPt.x >= derivI.cols ||
                    iprevPt.y < -winSize.height || iprevPt.y >= derivI.rows )
            {
                if( level == 0 )
                    status[ptidx] = false;

                continue;
            }

            float a = prevPt.x - iprevPt.x;
            float b = prevPt.y - iprevPt.y;

            int iw00 = cvRound((1.f - a)*(1.f - b)*rep);
            int iw01 = cvRound(a*(1.f - b)*rep);
            int iw10 = cvRound((1.f - a)*b*rep);
            int iw11 = rep - iw00 - iw01 - iw10;


            float A11 = 0, A12 = 0, A22 = 0;

            int CV_DECL_ALIGNED(16) nA11[] = {0, 0, 0, 0}, nA12[] = {0, 0, 0, 0}, nA22[] = {0, 0, 0, 0};
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
            for( y = 0; y < winSize.height; y++ )
            {
                const uchar* src = (const uchar*)I.data + (y + iprevPt.y)*stepI + iprevPt.x;
                const deriv_type* dsrc = (const deriv_type*)derivI.data + (y + iprevPt.y)*dstep + iprevPt.x*2;

                deriv_type* Iptr = (deriv_type*)(IWinBuf.data + y*IWinBuf.step);
                deriv_type* dIptr = (deriv_type*)(derivIWinBuf.data + y*derivIWinBuf.step);

                x = 0;

                for( ; x <= winSize.width*cn - 4; x += 4, dsrc += 4*2, dIptr += 4*2 )
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
                    int32x4_t nq0 = vmull_s16(d4d5.val[1], d28);
                    q8 = vmull_s16(d6d7.val[0], d29);
                    int32x4_t q15 = vmull_s16(d6d7.val[1], d29);

                    q7 = vaddq_s32(q7, q8);
                    nq0 = vaddq_s32(nq0, q15);

                    q4 = vaddq_s32(q4, q7);
                    q6 = vaddq_s32(q6, nq0);

                    int32x4_t nq1 = vld1q_s32(nA12);
                    int32x4_t nq2 = vld1q_s32(nA22);
                    nq0 = vld1q_s32(nA11);

                    q4 = vqrshlq_s32(q4, q12);
                    q6 = vqrshlq_s32(q6, q12);

                    q7 = vmulq_s32(q4, q4);
                    q8 = vmulq_s32(q4, q6);
                    q15 = vmulq_s32(q6, q6);

                    nq0 = vaddq_s32(nq0, q7);
                    nq1 = vaddq_s32(nq1, q8);
                    nq2 = vaddq_s32(nq2, q15);

                    vst1q_s32(nA11, nq0);
                    vst1q_s32(nA12, nq1);
                    vst1q_s32(nA22, nq2);

                    int16x4_t d8 = vmovn_s32(q4);
                    int16x4_t d12 = vmovn_s32(q6);

                    int16x4x2_t d8d12;
                    d8d12.val[0] = d8;
                    d8d12.val[1] = d12;
                    vst2_s16(dIptr, d8d12);
                }

                for( ; x < winSize.width; x++, dsrc += 2, dIptr += 2 )
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

                    A11 += (float)(ixval*ixval);
                    A12 += (float)(ixval*iyval);
                    A22 += (float)(iyval*iyval);
                }
            }


            A11 += (float)(nA11[0] + nA11[1] + nA11[2] + nA11[3]);
            A12 += (float)(nA12[0] + nA12[1] + nA12[2] + nA12[3]);
            A22 += (float)(nA22[0] + nA22[1] + nA22[2] + nA22[3]);

            A11 *= FLT_SCALE;
            A12 *= FLT_SCALE;
            A22 *= FLT_SCALE;

            float D = A11*A22 - A12*A12;
            float minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                                                  4.f*A12*A12))/(2*winSize.width*winSize.height);



            if( minEig < minEigThreshold || D < FLT_EPSILON )
            {
                if( level == 0)
                    status[ptidx] = false;
                continue;
            }

            D = 1.f/D;

            nextPt -= halfWin;
            Point2f prevDelta;

            for( j = 0; j < criteria.maxCount; j++ )
            {
                inextPt.x = cvFloor(nextPt.x);
                inextPt.y = cvFloor(nextPt.y);

                if( inextPt.x < -winSize.width || inextPt.x >= J.cols ||
                        inextPt.y < -winSize.height || inextPt.y >= J.rows )
                {
                    if( level == 0 )
                        status[ptidx] = false;
                    break;
                }

                a = nextPt.x - inextPt.x;
                b = nextPt.y - inextPt.y;
                iw00 = cvRound((1.f - a)*(1.f - b)*rep);
                iw01 = cvRound(a*(1.f - b)*rep);
                iw10 = cvRound((1.f - a)*b*rep);
                iw11 = rep - iw00 - iw01 - iw10;
                float b1 = 0, b2 = 0;

                int CV_DECL_ALIGNED(16) nB1[] = {0,0,0,0}, nB2[] = {0,0,0,0};

                const int16x4_t d26_2 = vdup_n_s16((int16_t)iw00);
                const int16x4_t d27_2 = vdup_n_s16((int16_t)iw01);
                const int16x4_t d28_2 = vdup_n_s16((int16_t)iw10);
                const int16x4_t d29_2 = vdup_n_s16((int16_t)iw11);

                for( y = 0; y < winSize.height; y++ )
                {
                    const uchar* Jptr = (const uchar*)J.data + (y + inextPt.y)*stepJ + inextPt.x;
                    const deriv_type* Iptr = (const deriv_type*)(IWinBuf.data + y*IWinBuf.step);
                    const deriv_type* dIptr = (const deriv_type*)(derivIWinBuf.data + y*derivIWinBuf.step);

                    x = 0;

                    for( ; x <= winSize.width - 8; x += 8, dIptr += 8*2 )
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
                        nq11 = vld1q_s32(nB1);
                        int32x4_t nq15 = vld1q_s32(nB2);

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

                        nq11 = vaddq_s32(nq11, nq9);
                        nq15 = vaddq_s32(nq15, nq4);

                        vst1q_s32(nB1, nq11);
                        vst1q_s32(nB2, nq15);
                    }


                    for( ; x < winSize.width; x++, dIptr += 2 )
                    {
                        int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                              Jptr[x+stepJ]*iw10 + Jptr[x+stepJ+cn]*iw11,
                                              W_BITS1-5) - Iptr[x];
                        b1 += (float)(diff*dIptr[0]);
                        b2 += (float)(diff*dIptr[1]);
                    }
                }


                b1 += (float)(nB1[0] + nB1[1] + nB1[2] + nB1[3]);
                b2 += (float)(nB2[0] + nB2[1] + nB2[2] + nB2[3]);


                b1 *= FLT_SCALE;
                b2 *= FLT_SCALE;

                Point2f delta( (float)((A12*b2 - A22*b1) * D),
                               (float)((A12*b1 - A11*b2) * D));

                nextPt += delta;
                nextPts[ptidx] = nextPt + halfWin;

                if( delta.ddot(delta) <= criteria.epsilon )
                    break;

                if( j > 0 && std::abs(delta.x + prevDelta.x) < 0.01 &&
                        std::abs(delta.y + prevDelta.y) < 0.01 )
                {
                    nextPts[ptidx] -= delta*0.5f;
                    break;
                }
                prevDelta = delta;
            }

        }
    }


    const float FLT_SCALE = 1.f/(1 << 20);
    const int W_BITS = 14, W_BITS1 = 14;
    const int rep = (1 << W_BITS);

    const Mat* prevImg;
    const Mat* nextImg;
    const Mat* prevDeriv;
    const Point2f* prevPts;
    Point2f* nextPts;
    uchar* status;
    Size winSize;
    TermCriteria criteria;
    int level;
    int maxLevel;
    int flags;
    float minEigThreshold;
};

}// namespace detail
}// namespace cv
