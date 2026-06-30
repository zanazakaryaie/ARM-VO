#include <catch2/catch.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <vector>

#include "SemanticSegmentorNcnn.hpp"

namespace
{

constexpr int kNumCityscapesClasses = 19;

armvo::SemanticSegmentorNcnn& segmentor()
{
    static armvo::SemanticSegmentorNcnn instance;
    return instance;
}

cv::Mat makeGrayFrame()
{
    cv::Mat image(96, 160, CV_8UC1);

    for (int y = 0; y < image.rows; y++)
    {
        uint8_t* row = image.ptr<uint8_t>(y);
        for (int x = 0; x < image.cols; x++)
        {
            row[x] = static_cast<uint8_t>((3 * x + 5 * y) % 256);
        }
    }

    cv::circle(image, cv::Point(48, 40), 18, cv::Scalar(230), cv::FILLED);
    cv::rectangle(image, cv::Rect(92, 28, 42, 34), cv::Scalar(35), cv::FILLED);
    return image;
}

cv::Mat makeBgrFrame()
{
    const cv::Mat gray = makeGrayFrame();
    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);

    for (int y = 0; y < bgr.rows; y++)
    {
        cv::Vec3b* row = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < bgr.cols; x++)
        {
            row[x][0] = static_cast<uint8_t>((row[x][0] + x) % 256);
            row[x][1] = static_cast<uint8_t>((row[x][1] + y) % 256);
            row[x][2] = static_cast<uint8_t>((row[x][2] + x + y) % 256);
        }
    }

    return bgr;
}

cv::Mat makeRgbFrame()
{
    const cv::Mat bgr = makeBgrFrame();
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

void checkSegmentationMap(const cv::Mat& segmentationMap, const cv::Size& expectedSize)
{
    REQUIRE_FALSE(segmentationMap.empty());
    CHECK(segmentationMap.size() == expectedSize);
    CHECK(segmentationMap.type() == CV_8UC1);

    double minLabel = 0.0;
    double maxLabel = 0.0;
    cv::minMaxLoc(segmentationMap, &minLabel, &maxLabel);

    CHECK(minLabel >= 0.0);
    CHECK(maxLabel < kNumCityscapesClasses);
}

std::vector<uint8_t> rowValues(const cv::Mat& mask)
{
    REQUIRE(mask.rows == 1);
    REQUIRE(mask.type() == CV_8UC1);

    std::vector<uint8_t> values;
    values.reserve(static_cast<size_t>(mask.cols));

    for (int x = 0; x < mask.cols; x++)
    {
        values.push_back(mask.at<uint8_t>(0, x));
    }

    return values;
}

} // namespace

TEST_CASE("SemanticSegmentorNcnn creates road and static masks from label maps")
{
    const cv::Mat labels = (cv::Mat_<uint8_t>(1, 8) << 0, 1, 2, 10, 11, 12, 18, 0);

    const cv::Mat roadMask = segmentor().getRoadMask(labels);
    const cv::Mat staticMask = segmentor().getStaticMask(labels);

    CHECK(roadMask.size() == labels.size());
    CHECK(roadMask.type() == CV_8UC1);
    CHECK(rowValues(roadMask) == std::vector<uint8_t>{255, 0, 0, 0, 0, 0, 0, 255});

    CHECK(staticMask.size() == labels.size());
    CHECK(staticMask.type() == CV_8UC1);
    CHECK(rowValues(staticMask) == std::vector<uint8_t>{255, 255, 255, 255, 0, 0, 0, 255});
}

TEST_CASE("SemanticSegmentorNcnn returns an empty map for non-contiguous frames")
{
    cv::Mat padded(96, 168, CV_8UC1, cv::Scalar(0));
    cv::Mat roi = padded(cv::Rect(0, 0, 160, 96));

    REQUIRE_FALSE(roi.isContinuous());

    const cv::Mat segmentationMap = segmentor().segment(roi, armvo::PixelFormat::Gray);

    CHECK(segmentationMap.empty());
}

TEST_CASE("SemanticSegmentorNcnn segments continuous frames")
{
    SECTION("gray input")
    {
        const cv::Mat frame = makeGrayFrame();
        REQUIRE(frame.isContinuous());

        const cv::Mat segmentationMap = segmentor().segment(frame, armvo::PixelFormat::Gray);

        checkSegmentationMap(segmentationMap, frame.size());
    }

    SECTION("bgr input")
    {
        const cv::Mat frame = makeBgrFrame();
        REQUIRE(frame.isContinuous());

        const cv::Mat segmentationMap = segmentor().segment(frame, armvo::PixelFormat::BGR);

        checkSegmentationMap(segmentationMap, frame.size());
    }

    SECTION("rgb input")
    {
        const cv::Mat frame = makeRgbFrame();
        REQUIRE(frame.isContinuous());

        const cv::Mat segmentationMap = segmentor().segment(frame, armvo::PixelFormat::RGB);

        checkSegmentationMap(segmentationMap, frame.size());
    }
}
