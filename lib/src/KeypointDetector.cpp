#include "KeypointDetector.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#ifdef x86
#include <opencv2/features2d.hpp>
#else
#include <arm_neon.h>
#endif

namespace armvo
{

class KeypointDetector::Impl
{
    const uint8_t mResponseThreshold;

public:
    explicit Impl(uint8_t responseThreshold) : mResponseThreshold(responseThreshold)
    {
        if (responseThreshold == 0 || responseThreshold == 255)
        {
            throw std::logic_error("Keypoint response threshold must be BETWEEN 0-255");
        }
    }

    std::vector<Keypoint> detect(const cv::Mat& image) const
    {
#ifdef x86
        return FAST9_x86(image, mResponseThreshold);
#else
        const int numStripes = std::max(1, std::min(image.rows, cv::getNumberOfCPUs()));

        const int rowsPerStripe = image.rows / numStripes;
        const int extraRows = image.rows % numStripes;
        const int stripeBorder = 4; // Each strip is padded to ensure points are not missed in the strip borders

        cv::Mutex kptLock;
        std::vector<Keypoint> keypoints;

        cv::parallel_for_(cv::Range(0, numStripes), [&](const cv::Range& range)
        {
            for (int stripeIdx = range.start; stripeIdx < range.end; stripeIdx++)
            {
                const int y0Orig = stripeIdx * rowsPerStripe + std::min(stripeIdx, extraRows);
                const int y1Orig = y0Orig + rowsPerStripe + (stripeIdx < extraRows ? 1 : 0);

                const int y0Ext = std::max(0, y0Orig - stripeBorder);
                const int y1Ext = std::min(image.rows, y1Orig + stripeBorder);

                if (y0Ext >= y1Ext)
                {
                    continue;
                }

                cv::Mat sub = image.rowRange(y0Ext, y1Ext).clone();
                auto subKeypoints = FAST9_NEON(sub, mResponseThreshold);

                for (auto& keypoint : subKeypoints)
                {
                    keypoint.y += y0Ext;
                }

                subKeypoints.erase(
                std::remove_if(subKeypoints.begin(), subKeypoints.end(),
                    [&](const Keypoint& k)
                    {
                        return k.y < y0Orig || k.y >= y1Orig;
                    }),
                subKeypoints.end());

                cv::AutoLock joinKeypoints(kptLock);
                keypoints.insert(keypoints.end(), subKeypoints.begin(), subKeypoints.end());
            }
        });

        return keypoints;
#endif
    }

private:

#ifdef x86
    static std::vector<Keypoint> FAST9_x86(const cv::Mat &img, uint8_t threshold)
    {
        std::vector<cv::KeyPoint> cvKeypoints;
        cv::FAST(img, cvKeypoints, threshold, true, cv::FastFeatureDetector::TYPE_9_16);
        std::vector<Keypoint> output;
        output.reserve(cvKeypoints.size());
        for (const auto& cvKeypoint : cvKeypoints)
        {
            output.emplace_back(cvKeypoint.pt.x, cvKeypoint.pt.y, cvKeypoint.response);
        }
        return output;
    }
#endif

#ifdef ARM
    static std::vector<Keypoint> FAST9_NEON(const cv::Mat &img, uint8_t threshold)
    {
        std::vector<Keypoint> keypoints;
        keypoints.reserve(500);

        constexpr int32_t K = 8, N = 16 + K + 1;
        long int i, j, k, pixel[N];
        makeOffsets(pixel, img.cols);
        for (k = 16; k < N; k++)
        {
            pixel[k] = pixel[k - 16];
        }

        uint8x16_t delta = vdupq_n_u8(128);
        uint8x16_t t = vdupq_n_u8(threshold);
        uint8x16_t K16 = vdupq_n_u8((uint8_t)K);

        uint8_t threshold_tab[512];
        for (i = -255; i <= 255; i++)
        {
            threshold_tab[i+255] = static_cast<uint8_t>(i < -threshold ? 1 : i > threshold ? 2 : 0);
        }

        std::vector<uint8_t> _buf((img.cols+16)*3*(sizeof(long int) + sizeof(uint8_t)) + 128);
        uint8_t* buf[3];
        buf[0] = &_buf[0]; buf[1] = buf[0] + img.cols; buf[2] = buf[1] + img.cols;
        long int* cpbuf[3];
        cpbuf[0] = (long int*)alignPtr(buf[2] + img.cols, sizeof(long int)) + 1;
        cpbuf[1] = cpbuf[0] + img.cols + 1;
        cpbuf[2] = cpbuf[1] + img.cols + 1;
        memset(buf[0], 0, img.cols*3);

        for (i = 3; i < (long int)img.rows-2; i++)
        {
            const uint8_t* ptr = img.ptr(i) + 3;
            uint8_t* curr = buf[(i - 3)%3];
            long int* cornerpos = cpbuf[(i - 3)%3];
            memset(curr, 0, img.cols);
            long int ncorners = 0;

            if (i < (long int)img.rows - 3)
            {
                j = 3;

                for (; j < (long int)img.cols - 16 - 3; j += 16, ptr += 16)
                {
                    prefetch(ptr);
                    prefetch(ptr + pixel[0]);
                    prefetch(ptr + pixel[2]);

                    uint8x16_t v0 = vld1q_u8(ptr);
                    int8x16_t v1 = vreinterpretq_s8_u8(veorq_u8(vqsubq_u8(v0, t), delta));
                    int8x16_t v2 = vreinterpretq_s8_u8(veorq_u8(vqaddq_u8(v0, t), delta));

                    int8x16_t x0 = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(ptr + pixel[0]), delta));
                    int8x16_t x1 = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(ptr + pixel[4]), delta));
                    int8x16_t x2 = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(ptr + pixel[8]), delta));
                    int8x16_t x3 = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(ptr + pixel[12]), delta));

                    uint8x16_t m0 =   vandq_u8(vcgtq_s8(x0, v2), vcgtq_s8(x1, v2));
                    uint8x16_t m1 =   vandq_u8(vcgtq_s8(v1, x0), vcgtq_s8(v1, x1));
                    m0 = vorrq_u8(m0, vandq_u8(vcgtq_s8(x1, v2), vcgtq_s8(x2, v2)));
                    m1 = vorrq_u8(m1, vandq_u8(vcgtq_s8(v1, x1), vcgtq_s8(v1, x2)));
                    m0 = vorrq_u8(m0, vandq_u8(vcgtq_s8(x2, v2), vcgtq_s8(x3, v2)));
                    m1 = vorrq_u8(m1, vandq_u8(vcgtq_s8(v1, x2), vcgtq_s8(v1, x3)));
                    m0 = vorrq_u8(m0, vandq_u8(vcgtq_s8(x3, v2), vcgtq_s8(x0, v2)));
                    m1 = vorrq_u8(m1, vandq_u8(vcgtq_s8(v1, x3), vcgtq_s8(v1, x0)));
                    m0 = vorrq_u8(m0, m1);

                    uint64_t mask[2];
                    vst1q_u64(mask, vreinterpretq_u64_u8(m0));

                    if (mask[0] == 0)
                    {
                        if (mask[1] != 0)
                        {
                            j -= 8;
                            ptr -= 8;
                        }
                        continue;
                    }

                    uint8x16_t c0 = vmovq_n_u8(0);
                    uint8x16_t c1 = vmovq_n_u8(0);
                    uint8x16_t max0 = vmovq_n_u8(0);
                    uint8x16_t max1 = vmovq_n_u8(0);
                    for (k = 0; k < N; k++)
                    {
                        int8x16_t x = vreinterpretq_s8_u8(veorq_u8(vld1q_u8(ptr + pixel[k]), delta));
                        m0 = vcgtq_s8(x, v2);
                        m1 = vcgtq_s8(v1, x);

                        c0 = vandq_u8(vsubq_u8(c0, m0), m0);
                        c1 = vandq_u8(vsubq_u8(c1, m1), m1);

                        max0 = vmaxq_u8(max0, c0);
                        max1 = vmaxq_u8(max1, c1);
                    }

                    max0 = vmaxq_u8(max0, max1);
                    uint8_t m[16];
                    vst1q_u8(m, vcgtq_u8(max0, K16));

                    for (k = 0; k < 16; ++k)
                    {
                        if (m[k])
                        {
                            cornerpos[ncorners++] = j+k;
                            curr[j+k] = cornerScore(ptr+k, pixel);
                        }
                    }

                }

                for (; j < (int32_t)img.cols - 3; j++, ptr++)
                {
                    int32_t v = ptr[0];
                    const uint8_t* tab = &threshold_tab[0] - v + 255;
                    int32_t d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];

                    if (d == 0)
                    {
                        continue;
                    }

                    d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
                    d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
                    d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];

                    if (d == 0)
                    {
                        continue;
                    }

                    d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
                    d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
                    d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
                    d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];

                    if (d & 1)
                    {
                        int32_t vt = v - threshold, count = 0;

                        for (k = 0; k < N; k++)
                        {
                            int32_t x = ptr[pixel[k]];
                            if (x < vt)
                            {
                                if (++count > K)
                                {
                                    cornerpos[ncorners++] = j;
                                    curr[j] = cornerScore(ptr, pixel);
                                    break;
                                }
                            }
                            else
                            {
                                count = 0;
                            }
                        }
                    }

                    if (d & 2)
                    {
                        int32_t vt = v + threshold, count = 0;

                        for (k = 0; k < N; k++)
                        {
                            int32_t x = ptr[pixel[k]];
                            if (x > vt)
                            {
                                if (++count > K)
                                {
                                    cornerpos[ncorners++] = j;
                                    curr[j] = cornerScore(ptr, pixel);
                                    break;
                                }
                            }
                            else
                            {
                                count = 0;
                            }
                        }
                    }
                }
            }

            cornerpos[-1] = ncorners;

            if (i == 3)
            {
                continue;
            }

            const uint8_t* prev = buf[(i - 4 + 3)%3];
            const uint8_t* pprev = buf[(i - 5 + 3)%3];
            cornerpos = cpbuf[(i - 4 + 3)%3];
            ncorners = cornerpos[-1];

            for (k = 0; k < ncorners; k++)
            {
                j = cornerpos[k];
                int32_t score = prev[j];
                if ((score > prev[j+1] && score > prev[j-1] && score > pprev[j-1] &&
                     score > pprev[j] && score > pprev[j+1] && score > curr[j-1] &&
                     score > curr[j] && score > curr[j+1]))
                {
                    keypoints.emplace_back((float)j, (float)(i-1), float(score));
                }
            }
        }

        return keypoints;
    }

    static void makeOffsets(long int pixel[], long int row_stride)
    {
        pixel[0] = 0 + row_stride * 3;
        pixel[1] = 1 + row_stride * 3;
        pixel[2] = 2 + row_stride * 2;
        pixel[3] = 3 + row_stride * 1;
        pixel[4] = 3 + row_stride * 0;
        pixel[5] = 3 + row_stride * -1;
        pixel[6] = 2 + row_stride * -2;
        pixel[7] = 1 + row_stride * -3;
        pixel[8] = 0 + row_stride * -3;
        pixel[9] = -1 + row_stride * -3;
        pixel[10] = -2 + row_stride * -2;
        pixel[11] = -3 + row_stride * -1;
        pixel[12] = -3 + row_stride * 0;
        pixel[13] = -3 + row_stride * 1;
        pixel[14] = -2 + row_stride * 2;
        pixel[15] = -1 + row_stride * 3;
    }

    template<typename T>
    static inline T* alignPtr(T* ptr, size_t n=sizeof(T))
    {
        return (T*)(((size_t)ptr + n-1) & -n);
    }

    static inline void prefetch(const void *ptr, size_t offset = 32*10)
    {
        __builtin_prefetch(reinterpret_cast<const char*>(ptr) + offset);
    }

    static uint8_t cornerScore(const uint8_t* ptr, const long int pixel[])
    {
        constexpr int32_t K = 8, N = 16 + K + 1;
        int32_t k, v = ptr[0];
        int16_t d[(N + 7) & ~7];
        for (k = 0; k < N; k++)
        {
            d[k] = (int16_t)(v - ptr[pixel[k]]);
        }

        int16x8_t q0 = vdupq_n_s16((int16_t)(-1000));
        int16x8_t q1 = vdupq_n_s16((int16_t)(1000));

        int16x8_t d0_7   = vld1q_s16(d +  0);
        int16x8_t d8_15  = vld1q_s16(d +  8);
        int16x8_t d16_23 = vld1q_s16(d + 16);
        int16x8_t d24    = vld1q_s16(d + 24);

        //k == 0
        int16x8_t v0k0 = vextq_s16(d0_7, d8_15, 1);
        int16x8_t v1k0 = vextq_s16(d0_7, d8_15, 2);
        int16x8_t ak0 = vminq_s16(v0k0, v1k0);
        int16x8_t bk0 = vmaxq_s16(v0k0, v1k0);

        v0k0 = vextq_s16(d0_7, d8_15, 3);
        ak0 = vminq_s16(ak0, v0k0);
        bk0 = vmaxq_s16(bk0, v0k0);

        v1k0 = vextq_s16(d0_7, d8_15, 4);
        ak0 = vminq_s16(ak0, v1k0);
        bk0 = vmaxq_s16(bk0, v1k0);

        v0k0 = vextq_s16(d0_7, d8_15, 5);
        ak0 = vminq_s16(ak0, v0k0);
        bk0 = vmaxq_s16(bk0, v0k0);

        v1k0 = vextq_s16(d0_7, d8_15, 6);
        ak0 = vminq_s16(ak0, v1k0);
        bk0 = vmaxq_s16(bk0, v1k0);

        v0k0 = vextq_s16(d0_7, d8_15, 7);
        ak0 = vminq_s16(ak0, v0k0);
        bk0 = vmaxq_s16(bk0, v0k0);

        ak0 = vminq_s16(ak0, d8_15);
        bk0 = vmaxq_s16(bk0, d8_15);

        q0 = vmaxq_s16(q0, vminq_s16(ak0, d0_7));
        q1 = vminq_s16(q1, vmaxq_s16(bk0, d0_7));

        v1k0 = vextq_s16(d8_15, d16_23, 1);
        q0 = vmaxq_s16(q0, vminq_s16(ak0, v1k0));
        q1 = vminq_s16(q1, vmaxq_s16(bk0, v1k0));

        //k == 8
        int16x8_t v0k8 = v1k0;
        int16x8_t v1k8 = vextq_s16(d8_15, d16_23, 2);
        int16x8_t ak8 = vminq_s16(v0k8, v1k8);
        int16x8_t bk8 = vmaxq_s16(v0k8, v1k8);

        v0k8 = vextq_s16(d8_15, d16_23, 3);
        ak8 = vminq_s16(ak8, v0k8);
        bk8 = vmaxq_s16(bk8, v0k8);

        v1k8 = vextq_s16(d8_15, d16_23, 4);
        ak8 = vminq_s16(ak8, v1k8);
        bk8 = vmaxq_s16(bk8, v1k8);

        v0k8 = vextq_s16(d8_15, d16_23, 5);
        ak8 = vminq_s16(ak8, v0k8);
        bk8 = vmaxq_s16(bk8, v0k8);

        v1k8 = vextq_s16(d8_15, d16_23, 6);
        ak8 = vminq_s16(ak8, v1k8);
        bk8 = vmaxq_s16(bk8, v1k8);

        v0k8 = vextq_s16(d8_15, d16_23, 7);
        ak8 = vminq_s16(ak8, v0k8);
        bk8 = vmaxq_s16(bk8, v0k8);

        ak8 = vminq_s16(ak8, d16_23);
        bk8 = vmaxq_s16(bk8, d16_23);

        q0 = vmaxq_s16(q0, vminq_s16(ak8, d8_15));
        q1 = vminq_s16(q1, vmaxq_s16(bk8, d8_15));

        v1k8 = vextq_s16(d16_23, d24, 1);
        q0 = vmaxq_s16(q0, vminq_s16(ak8, v1k8));
        q1 = vminq_s16(q1, vmaxq_s16(bk8, v1k8));

        //fin
        int16x8_t q = vmaxq_s16(q0, vsubq_s16(vmovq_n_s16(0), q1));
        int16x4_t q2 = vmax_s16(vget_low_s16(q), vget_high_s16(q));
        int32x4_t q2w = vmovl_s16(q2);
        int32x2_t q4 = vmax_s32(vget_low_s32(q2w), vget_high_s32(q2w));
        int32x2_t q8 = vmax_s32(q4, vreinterpret_s32_s64(vshr_n_s64(vreinterpret_s64_s32(q4), 32)));

        return (uint8_t)(vget_lane_s32(q8, 0) - 1);
    }
#endif
};

KeypointDetector::KeypointDetector(uint8_t responseThreshold)
{
    mImpl = std::make_shared<Impl>(responseThreshold);
}

std::vector<Keypoint> KeypointDetector::detect(const cv::Mat& image)
{
    return mImpl->detect(image);
}

} // namespace armvo
