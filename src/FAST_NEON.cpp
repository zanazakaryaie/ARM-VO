#include "FAST_NEON.hpp"
#include <arm_neon.h>


static const int offsets16[][2] =
{
    {0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
    {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}
};

const int (*offsets)[2] = offsets16;

const uint8_t __attribute__ (( aligned (16) )) _Powers [16] = { 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };

static uint8x16_t powers = vld1q_u8 ( _Powers ); //~ Set the powers of 2 (do it once for all , if applicable )


// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
int cornerScore(const uchar* ptr, const int pixel[], int threshold)
{
    const int K = 8, N = K*3 + 1;
    int k, v = ptr[0];
    short d[N];
    for( k = 0; k < N; k++ )
        d[k] = (short)(v - ptr[pixel[k]]);

    int16x8_t  q0 = vdupq_n_s16(-1000), q1 = vdupq_n_s16(1000);

    for( k = 0; k < 16; k += 8 )
    {
        int16x8_t v0 = vld1q_s16((const int16_t*)(d+k+1));
        int16x8_t v1 = vld1q_s16((const int16_t*)(d+k+2));
        int16x8_t a = vminq_s16(v0,v1);
        int16x8_t b = vmaxq_s16(v0,v1);
        v0 = vld1q_s16((const int16_t*)(d+k+3));
        a = vminq_s16(a,v1);
        b = vmaxq_s16(b,v0);
        v0 = vld1q_s16((const int16_t*)(d+k+4));
        a = vminq_s16(a,v0);
        b = vmaxq_s16(b,v0);
        v0 = vld1q_s16((const int16_t*)(d+k+5));
        a = vminq_s16(a,v0);
        b = vmaxq_s16(b,v0);
        v0 = vld1q_s16((const int16_t*)(d+k+6));
        a = vminq_s16(a,v0);
        b = vmaxq_s16(b,v0);
        v0 = vld1q_s16((const int16_t*)(d+k+7));
        a = vminq_s16(a,v0);
        b = vmaxq_s16(b,v0);
        v0 = vld1q_s16((const int16_t*)(d+k+8));
        a = vminq_s16(a,v0);
        b = vmaxq_s16(b,v0);
        v0 = vld1q_s16((const int16_t*)(d+k));
        q0 = vmaxq_s16(q0, vminq_s16(a, v0));
        q1 = vminq_s16(q1, vmaxq_s16(b, v0));
        v0 = vld1q_s16((const int16_t*)(d+k+9));
        q0 = vmaxq_s16(q0, vminq_s16(a, v0));
        q1 = vminq_s16(q1, vmaxq_s16(b, v0));
    }

    q0 = vmaxq_s16 (q0 , vextq_s16 (q0 , q0 , 4));
    q0 = vmaxq_s16 (q0 , vextq_s16 (q0 , q0 , 2));
    q0 = vmaxq_s16 (q0 , vextq_s16 (q0 , q0 , 1));
    vst1q_lane_s16 (( int16_t *) & threshold , q0 , 0);
    threshold --;


    return threshold;
}


// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
void makeOffsets(int pixel[25], const int rowStride)
{

    unsigned int k = 0;
    for( ; k < 16; k++ )
        pixel[k] = offsets[k][0] + offsets[k][1] * rowStride;
    for( ; k < 25; k++ )
        pixel[k] = pixel[k - 16];
}


// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
uint16_t movemask ( uint8x16_t input )
{
    //~ Compute mask from input
    uint64x2_t _Mask = vpaddlq_u32 ( vpaddlq_u16 ( vpaddlq_u8 ( vandq_u8 (input, powers ))));

    //~ Get resulting bytes
    uint16_t mask ;
    vst1q_lane_u8 (( uint8_t *)& mask + 0, ( uint8x16_t )_Mask, 0);
    vst1q_lane_u8 (( uint8_t *)& mask + 1, ( uint8x16_t )_Mask, 8);

    return mask ;
}


// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
void FAST9_NEON(const cv::Mat &img, std::vector<cv::KeyPoint>& keypoints, int threshold)
{
    const int K = 8, N = 25;

    int i, j, k, pixel[25];
    makeOffsets(pixel, (int)img.step);

    keypoints.clear();
    keypoints.reserve(5000);

    uint8x16_t delta = vdupq_n_u8(-128), t = vdupq_n_u8((char)threshold), K16 = vdupq_n_u8((char)K);
    (void)K16;
    (void)delta;
    (void)t;

    uchar threshold_tab[512];
    for( i = -255; i <= 255; i++ )
        threshold_tab[i+255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);

    cv::AutoBuffer<uchar> _buf((img.cols+16)*3*(sizeof(int) + sizeof(uchar)) + 128);
    uchar* buf[3];
    buf[0] = _buf;
    buf[1] = buf[0] + img.cols;
    buf[2] = buf[1] + img.cols;
    int* cpbuf[3];
    cpbuf[0] = (int*)cv::alignPtr(buf[2] + img.cols, sizeof(int)) + 1;
    cpbuf[1] = cpbuf[0] + img.cols + 1;
    cpbuf[2] = cpbuf[1] + img.cols + 1;
    memset(buf[0], 0, img.cols*3);

    for(i = 3; i < img.rows-2; i++)
    {
        const uchar* ptr = img.ptr<uchar>(i) + 3;
        uchar* curr = buf[(i - 3)%3];
        int* cornerpos = cpbuf[(i - 3)%3];
        memset(curr, 0, img.cols);
        int ncorners = 0;

        if( i < img.rows - 3 )
        {
            j = 3;

            for(; j < img.cols - 16 - 3; j += 16, ptr += 16)
            {
                uint8x16_t m0,m1;
                uint8x16_t v0 = vld1q_u8((const uint8_t*)ptr);
                uint8x16_t v1 = vqsubq_u8(v0,t);
                v0 = vqaddq_u8(v0,t);

                uint8x16_t x0 = vld1q_u8((const uint8_t*)(ptr + pixel[0]));
                uint8x16_t x1 = vld1q_u8((const uint8_t*)(ptr + pixel[4]));
                uint8x16_t x2 = vld1q_u8((const uint8_t*)(ptr + pixel[8]));
                uint8x16_t x3 = vld1q_u8((const uint8_t*)(ptr + pixel[12]));

                m0 = vandq_u8(vcgtq_u8(x0,v0), vcgtq_u8(x1,v0));
                m1 = vandq_u8(vcgtq_u8(v1,x0), vcgtq_u8(v1,x1));
                m0 = vorrq_u8(m0, vandq_u8(vcgtq_u8(x1, v0), vcgtq_u8(x2, v0)));
                m1 = vorrq_u8(m1, vandq_u8(vcgtq_u8(v1, x1), vcgtq_u8(v1, x2)));
                m0 = vorrq_u8(m0, vandq_u8(vcgtq_u8(x2, v0), vcgtq_u8(x3, v0)));
                m1 = vorrq_u8(m1, vandq_u8(vcgtq_u8(v1, x2), vcgtq_u8(v1, x3)));
                m0 = vorrq_u8(m0, vandq_u8(vcgtq_u8(x3, v0), vcgtq_u8(x0, v0)));
                m1 = vorrq_u8(m1, vandq_u8(vcgtq_u8(v1, x3), vcgtq_u8(v1, x0)));
                m0 = vorrq_u8(m0, m1);

                uint32x2_t m_half = vqmovn_u64 (( uint64x2_t ) m0);
                uint32_t mask_array [2];
                vst1_u32 ( mask_array , m_half );

                if (( mask_array [0] == 0) && ( mask_array [1] == 0))
                    continue ;
                if ( mask_array [0] == 0)
                {
                    j -= 8;
                    ptr -= 8;
                    continue ;
                }

                uint8x16_t c0 = vdupq_n_u8(0), c1 = c0, max0 = c0, max1 = c0;
                for( k = 0; k < N; k++ )
                {
                    uint8x16_t x = vld1q_u8((const uint8_t*)(ptr + pixel[k]));
                    m0 = vcgtq_u8(x, v0);
                    m1 = vcgtq_u8(v1, x0);

                    c0 = vandq_u8(vsubq_u8(c0,m0),m0);
                    c1 = vandq_u8(vsubq_u8(c1,m1),m1);

                    max0 = vmaxq_u8(max0, c0);
                    max1 = vmaxq_u8(max1, c1);
                }

                max0 = vmaxq_u8(max0, max1);


                uint8_t m_array [16];
                vst1q_u8 ( m_array , ( vcgtq_u8 (max0 , K16)));

                for( k = 0; k < 16; k++)
                    if( m_array [k] != 0)
                    {
                        cornerpos [ ncorners ++] = j+k;
                        curr [j+k] = ( uint8_t ) cornerScore (ptr +k, pixel , threshold );
                    }
            }

            for( ; j < img.cols - 3; j++, ptr++ )
            {
                int v = ptr[0];
                const uchar* tab = &threshold_tab[0] - v + 255;
                int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];

                if( d == 0 )
                    continue;

                d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
                d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
                d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];

                if( d == 0 )
                    continue;

                d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
                d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
                d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
                d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];

                if( d & 1 )
                {
                    int vt = v - threshold, count = 0;

                    for( k = 0; k < N; k++ )
                    {
                        int x = ptr[pixel[k]];
                        if(x < vt)
                        {
                            if( ++count > K )
                            {
                                cornerpos[ncorners++] = j;
                                curr[j] = (uchar)cornerScore(ptr, pixel, threshold);
                                break;
                            }
                        }
                        else
                            count = 0;
                    }
                }

                if( d & 2 )
                {
                    int vt = v + threshold, count = 0;

                    for( k = 0; k < N; k++ )
                    {
                        int x = ptr[pixel[k]];
                        if(x > vt)
                        {
                            if( ++count > K )
                            {
                                cornerpos[ncorners++] = j;
                                curr[j] = (uchar)cornerScore(ptr, pixel, threshold);
                                break;
                            }
                        }
                        else
                            count = 0;
                    }
                }
            }
        }

        cornerpos[-1] = ncorners;

        if( i == 3 )
            continue;

        const uchar* prev = buf[(i - 4 + 3)%3];
        const uchar* pprev = buf[(i - 5 + 3)%3];
        cornerpos = cpbuf[(i - 4 + 3)%3];
        ncorners = cornerpos[-1];

        for( k = 0; k < ncorners; k++ )
        {
            j = cornerpos[k];
            int score = prev[j];

            if(
                    (score > prev[j+1] && score > prev[j-1] &&
                     score > pprev[j-1] && score > pprev[j] && score > pprev[j+1] &&
                     score > curr[j-1] && score > curr[j] && score > curr[j+1]) )
            {
                keypoints.push_back(cv::KeyPoint((float)j, (float)(i-1), 7.f, -1, (float)score));
            }
        }
    }
}
