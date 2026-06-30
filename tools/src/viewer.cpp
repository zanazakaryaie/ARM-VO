#include "viewer.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace armvo::tools
{

class Plotter::Impl
{
    cv::Mat mFrame;
    static constexpr uint16_t IMAGE_SIZE = 500;
    cv::Mat mPlottedTrajectory;
    std::vector<cv::Matx31f> mEstimatedTrajectory;
    std::vector<cv::Matx31f> mGroundTruthTrajectory;
    float mMinX = -0.5f;
    float mMinY = -0.5f;
    float mMaxX = 0.5f;
    float mMaxY = 0.5f;
    static constexpr float BORDER = 5.0f;
    static constexpr int LINE_THICKNESS = 2;
    bool mWindowsAreOpen = false;

public:
    Impl() : mPlottedTrajectory(IMAGE_SIZE, IMAGE_SIZE, CV_8UC3, cv::Scalar(0))
    {
        // Minimum number of frames in Kitti Odometry dataset is 271 (Seq. 04)
        mEstimatedTrajectory.reserve(271);
        mGroundTruthTrajectory.reserve(271);
    }

    void update(const cv::Mat& frame, const armvo::Pose& pose)
    {
        frame.copyTo(mFrame);
        mEstimatedTrajectory.push_back(pose.translation);
        updateRange(pose.translation);
    }

    void update(const cv::Mat& frame, const armvo::Pose& pose, const armvo::Pose& gt)
    {
        frame.copyTo(mFrame);
        mEstimatedTrajectory.emplace_back(pose.translation);
        updateRange(pose.translation);
        mGroundTruthTrajectory.emplace_back(gt.translation);
        updateRange(gt.translation);
    }

    void plot()
    {
        mPlottedTrajectory.setTo(cv::Scalar(0));

        const float scale = IMAGE_SIZE / std::max((mMaxX - mMinX), (mMaxY - mMinY));
        drawTrajectory(mEstimatedTrajectory, scale, cv::Scalar(0, 0, 255));
        drawTrajectory(mGroundTruthTrajectory, scale, cv::Scalar(0, 255, 0));

        drawLegend();

        cv::imshow("Frame", mFrame);
        cv::imshow("Trajectory", mPlottedTrajectory);
        cv::waitKey(1);

        mWindowsAreOpen = true;
    }

    void updateRange(const cv::Matx31f& position)
    {
        const float x = position(0);
        const float y = position(2);

        if (x - BORDER < mMinX)
        {
            mMinX = x - BORDER;
        }

        if (y - BORDER < mMinY)
        {
            mMinY = y - BORDER;
        }

        if (x + BORDER > mMaxX)
        {
            mMaxX = x + BORDER;
        }

        if (y + BORDER > mMaxY)
        {
            mMaxY = y + BORDER;
        }
    }

    void drawTrajectory(const std::vector<cv::Matx31f>& trajectory, float scale, cv::Scalar color)
    {
        if (trajectory.empty())
        {
            return;
        }
        for (size_t i = 0; i < trajectory.size()-1; i++)
        {
            const auto p0 = transformToImageCoordinates(trajectory[i], scale);
            const auto p1 = transformToImageCoordinates(trajectory[i + 1], scale);
            cv::line(mPlottedTrajectory, p0, p1, color, LINE_THICKNESS);
        }
        constexpr int RADIUS = 6;
        constexpr int CIRCLE_THICKNESS = 1;
        const auto lastPoint = transformToImageCoordinates(trajectory.back(), scale);
        cv::circle(mPlottedTrajectory, lastPoint, RADIUS, color, CIRCLE_THICKNESS);
    }

    cv::Point2i transformToImageCoordinates(const cv::Matx31f& position, float scale) const noexcept
    {
        const int x = (position(0) - mMinX) * scale;
        const int y = IMAGE_SIZE - (position(2) - mMinY) * scale; // reverse y for more intuitive top-down plot
        return cv::Point2i(x, y);
    }

    void drawLegend()
    {
        if (mEstimatedTrajectory.empty())
        {
            return;
        }

        static const int font = cv::FONT_HERSHEY_SIMPLEX;
        const float fontScale = 0.6;

        std::array<char, 32> xText{};
        int baseline = 0;
        const cv::Matx31f currentPosition = mEstimatedTrajectory.back();

        std::snprintf(xText.data(), xText.size(), "X(m): %.2f", currentPosition(0));
        const cv::Size xTextSize = cv::getTextSize(xText.data(), font, fontScale, LINE_THICKNESS, &baseline);

        std::array<char, 32> yText{};
        std::snprintf(yText.data(), yText.size(), "Y(m): %.2f", currentPosition(1));
        const cv::Size yTextSize = cv::getTextSize(yText.data(), font, fontScale, LINE_THICKNESS, &baseline);

        std::array<char, 32> zText{};
        std::snprintf(zText.data(), zText.size(), "Z(m): %.2f", currentPosition(2));
        const cv::Size zTextSize = cv::getTextSize(zText.data(), font, fontScale, LINE_THICKNESS, &baseline);

        static const int topMargin = 5;
        static const int bottomMargin = 5;
        static const int lineSpace = 10;

        const int legendHeight = topMargin + xTextSize.height + lineSpace + yTextSize.height + lineSpace + zTextSize.height + bottomMargin + baseline;
        const int legendWidth = std::max(std::max(zTextSize.width, yTextSize.width), xTextSize.width);

        cv::Mat legend(legendHeight, legendWidth, CV_8UC3, cv::Scalar::all(255));
        cv::putText(legend, xText.data(), cv::Point(0, topMargin + xTextSize.height), font, fontScale, cv::Scalar(0), LINE_THICKNESS);
        cv::putText(legend, yText.data(), cv::Point(0, topMargin + xTextSize.height + lineSpace + yTextSize.height), font, fontScale, cv::Scalar(0), LINE_THICKNESS);
        cv::putText(legend, zText.data(), cv::Point(0, topMargin + xTextSize.height + lineSpace + yTextSize.height + lineSpace + zTextSize.height), font, fontScale, cv::Scalar(0), LINE_THICKNESS);

        legend.copyTo(mPlottedTrajectory(cv::Rect(IMAGE_SIZE - legend.cols, 0, legend.cols, legend.rows)));
    }

    ~Impl()
    {
        if (mWindowsAreOpen)
        {
            try
            {
                cv::destroyWindow("Frame");
                cv::destroyWindow("Trajectory");
            }
            catch (...)
            {
            }
        }
    }
};

Plotter::Plotter()
{
    mImpl = std::make_shared<Impl>();
}

void Plotter::update(const cv::Mat& frame, const armvo::Pose& pose)
{
    mImpl->update(frame, pose);
}

void Plotter::update(const cv::Mat& frame, const armvo::Pose& pose, const armvo::Pose& gt)
{
    mImpl->update(frame, pose, gt);
}

void Plotter::plot()
{
    mImpl->plot();
}

}
