#include <catch2/catch.hpp>

#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ErrorEstimator.hpp"

namespace
{

constexpr float kEpsTight = 1e-6f;
constexpr float kEps = 1e-5f;

constexpr std::array<float, 8> kLengths =
{
    100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f
};

bool nearFloat(float a, float b, float eps)
{
    return std::abs(a - b) < eps;
}

cv::Matx33f rotZ(float yawRad)
{
    const float c = std::cos(yawRad);
    const float s = std::sin(yawRad);

    return cv::Matx33f(
        c,   -s,  0.0f,
        s,    c,  0.0f,
        0.0f, 0.0f, 1.0f
    );
}

armvo::Pose makePose(float x, float y, float z, float yawRad)
{
    armvo::Pose pose;
    pose.rotation = rotZ(yawRad);
    pose.translation = cv::Matx31f(x, y, z);
    return pose;
}

// GT: identity rotation, translation along +X (meters)
std::vector<armvo::Pose> makeLinearGt(int32_t numFrames, float stepMeters)
{
    std::vector<armvo::Pose> poses;
    poses.reserve(numFrames);
    for (int32_t i = 0; i < numFrames; i++)
    {
        poses.push_back(makePose(i * stepMeters, 0.0f, 0.0f, 0.0f));
    }
    return poses;
}

// Estimated: identity rotation, translation along +X but scaled
std::vector<armvo::Pose> makeLinearScaledEst(int32_t numFrames, float stepMeters, float scale)
{
    std::vector<armvo::Pose> poses;
    poses.reserve(numFrames);
    for (int32_t i = 0; i < numFrames; i++)
    {
        poses.push_back(makePose(i * stepMeters * scale, 0.0f, 0.0f, 0.0f));
    }
    return poses;
}

// Estimated: same as GT, but modify ONLY one frame's rotation (used to lock in strict '>')
std::vector<armvo::Pose> makeGtWithSingleFrameYaw(
    const std::vector<armvo::Pose>& gt,
    int32_t frameIdxToRotate,
    float yawRad)
{
    auto est = gt;
    est.at(frameIdxToRotate).rotation = rotZ(yawRad);
    return est;
}

const armvo::tools::Error* findError(const std::vector<armvo::tools::Error>& errors, int32_t startFrameIndex, float length)
{
    for (const auto& err : errors)
    {
        if (err.startFrameIndex == startFrameIndex && err.length == length)
        {
            return &err;
        }
    }
    return nullptr;
}

// For the special case used in these tests:
// - GT is linear along X with step=1.0 -> dist[i] = i
// - lastFrame is the first i such that i > start + len -> last = start + len + 1
// - dx = (last - start) * step = len + 1
float expectedNormalizedTranslationErrorForScale(float scale, float len)
{
    const float dx = len + 1.0f;
    const float tErr = std::abs(1.0f - scale) * dx;
    return tErr / len;
}

float expectedNormalizedRotationError(float thetaRad, float len)
{
    return std::abs(thetaRad) / len;
}

} // namespace

TEST_CASE("ErrorEstimator::estimate - empty sizes and mismatches")
{
    armvo::tools::ErrorEstimator estimator;

    SECTION("Both empty -> empty output")
    {
        const std::vector<armvo::Pose> est;
        const std::vector<armvo::Pose> gt;
        const auto errors = estimator.estimate(est, gt);
        CHECK(errors.empty());
    }

    SECTION("Size mismatch -> throws std::logic_error")
    {
        const auto est = makeLinearGt(/*numFrames=*/2, /*stepMeters=*/1.0f);
        const auto gt  = makeLinearGt(/*numFrames=*/3, /*stepMeters=*/1.0f);
        REQUIRE_THROWS_AS(estimator.estimate(est, gt), std::logic_error);
    }
}

TEST_CASE("ErrorEstimator::estimate - strict single-segment case locks in KITTI '>' last-frame logic")
{
    // With step=1m and dist[i]=i:
    // for start=0 and len=100 => last is i > 100 => 101
    // choose numFrames=102 so ONLY (start=0, len=100) is valid -> exactly one error record
    const int32_t numFrames = 102;
    const float stepMeters  = 1.0f;

    armvo::tools::ErrorEstimator estimator;

    SECTION("Perfect estimate -> exactly one entry, zero errors")
    {
        const auto gt  = makeLinearGt(numFrames, stepMeters);
        const auto est = gt;

        const auto errors = estimator.estimate(est, gt);

        REQUIRE(errors.size() == 1);

        const auto& err = errors[0];
        CHECK(err.startFrameIndex == 0);
        CHECK(nearFloat(err.length, 100.0f, kEpsTight));

        CHECK(std::abs(err.rotationErr) < kEpsTight);
        CHECK(std::abs(err.translationErr) < kEpsTight);
    }

    SECTION("Scale drift -> exact translationErr normalization uses dx = len+1 (because of strict '>')")
    {
        const auto gt = makeLinearGt(numFrames, stepMeters);

        const float scale = 1.10f;
        const auto est = makeLinearScaledEst(numFrames, stepMeters, scale);

        const auto errors = estimator.estimate(est, gt);

        REQUIRE(errors.size() == 1);

        const auto& err = errors[0];
        REQUIRE(err.startFrameIndex == 0);
        REQUIRE(nearFloat(err.length, 100.0f, kEpsTight));

        const float expected = expectedNormalizedTranslationErrorForScale(scale, /*len=*/100.0f);

        CHECK(std::abs(err.rotationErr) < kEpsTight);
        CHECK(std::abs(err.translationErr - expected) < kEps);
    }

    SECTION("Rotation drift at frame 101 -> non-zero rotationErr proves lastFrame==101 (strict '>')")
    {
        const auto gt = makeLinearGt(numFrames, stepMeters);

        // If lastFrame were 100 (i.e., using >=), rotating frame 101 would not affect the segment,
        // and rotationErr would come out ~0. This test therefore pins down the exact behavior.
        const float theta = 0.2f;
        const auto est = makeGtWithSingleFrameYaw(gt, /*frameIdxToRotate=*/101, theta);

        const auto errors = estimator.estimate(est, gt);

        REQUIRE(errors.size() == 1);

        const auto& err = errors[0];
        REQUIRE(err.startFrameIndex == 0);
        REQUIRE(nearFloat(err.length, 100.0f, kEpsTight));

        const float expected = expectedNormalizedRotationError(theta, /*len=*/100.0f);

        CHECK(std::abs(err.translationErr) < kEpsTight);
        CHECK(std::abs(err.rotationErr - expected) < kEps);
    }
}

TEST_CASE("ErrorEstimator::estimate - long trajectory: exact count, ordering, boundaries, and exact per-length errors")
{
    // Choose long enough so many (start,len) are valid, and boundary is interesting for len=800.
    // With step=1, last = start + len + 1 must be < numFrames.
    const int32_t numFrames = 1002; // indices [0..1001]
    const float stepMeters  = 1.0f;

    armvo::tools::ErrorEstimator estimator;

    SECTION("Perfect estimate -> exact count + all errors near zero + expected ordering at start=0")
    {
        const auto gt  = makeLinearGt(numFrames, stepMeters);
        const auto est = gt;

        const auto errors = estimator.estimate(est, gt);

        size_t expectedCount = 0;
        for (int32_t start = 0; start < numFrames; start += 10)
        {
            for (const float len : kLengths)
            {
                const int32_t last = start + static_cast<int32_t>(len) + 1;
                if (last < numFrames)
                {
                    expectedCount++;
                }
            }
        }

        REQUIRE(errors.size() == expectedCount);

        // Ordering for the first startFrame=0: lengths in ascending order (all valid here)
        REQUIRE(errors.size() >= kLengths.size());
        for (size_t i = 0; i < kLengths.size(); ++i)
        {
            CHECK(errors[i].startFrameIndex == 0);
            CHECK(nearFloat(errors[i].length, kLengths[i], kEpsTight));
        }

        // Boundary check for len=800:
        // start=200 -> last=1001 valid
        // start=210 -> last=1011 invalid
        CHECK(findError(errors, /*startFrameIndex=*/200, /*length=*/800.0f) != nullptr);
        CHECK(findError(errors, /*startFrameIndex=*/210, /*length=*/800.0f) == nullptr);

        // Perfect => all errors are ~0 and invariants hold
        for (const auto& err : errors)
        {
            CHECK(err.startFrameIndex % 10 == 0);
            CHECK(err.startFrameIndex >= 0);
            CHECK(err.startFrameIndex < numFrames);

            bool lengthAllowed = false;
            for (const float len : kLengths)
            {
                if (err.length == len)
                {
                    lengthAllowed = true;
                    break;
                }
            }
            CHECK(lengthAllowed);

            CHECK(err.length > 0.0f);

            CHECK(std::abs(err.rotationErr) < kEpsTight);
            CHECK(std::abs(err.translationErr) < kEpsTight);
        }
    }

    SECTION("Scale drift -> for each length, translationErr matches exact formula for ALL segments")
    {
        const auto gt = makeLinearGt(numFrames, stepMeters);

        const float scale = 0.90f; // 10% too short
        const auto est = makeLinearScaledEst(numFrames, stepMeters, scale);

        const auto errors = estimator.estimate(est, gt);
        REQUIRE_FALSE(errors.empty());

        for (const float len : kLengths)
        {
            const float expected = expectedNormalizedTranslationErrorForScale(scale, len);

            for (const auto& err : errors)
            {
                if (err.length != len)
                {
                    continue;
                }

                CHECK(std::abs(err.rotationErr) < kEpsTight);
                CHECK(std::abs(err.translationErr - expected) < kEps);
            }
        }
    }
}