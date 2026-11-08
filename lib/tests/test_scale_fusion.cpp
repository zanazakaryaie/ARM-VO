#include <catch2/catch.hpp>

#include "ScaleFusion.hpp"

TEST_CASE("ScaleFusion returns no scale without measurements")
{
    CHECK_FALSE(armvo::ScaleFusion::fuse({}).has_value());
}

TEST_CASE("ScaleFusion preserves a single measurement regardless of its weight")
{
    const auto scale = armvo::ScaleFusion::fuse({{0.123456f, 0.01f}});
    REQUIRE(scale.has_value());
    CHECK(scale.value() == 0.123456f);
}

TEST_CASE("ScaleFusion combines measurements in log space with relative weights")
{
    SECTION("equal weights give the geometric mean")
    {
        const auto scale = armvo::ScaleFusion::fuse({{1.0f, 1.0f}, {4.0f, 1.0f}});
        REQUIRE(scale.has_value());
        CHECK(scale.value() == Approx(2.0f));
    }

    SECTION("unequal weights and common weight scaling")
    {
        for (const float weightFactor : {1.0f, 10.0f})
        {
            const auto scale = armvo::ScaleFusion::fuse({{1.0f, 3.0f * weightFactor},
                                                        {16.0f, weightFactor}});
            REQUIRE(scale.has_value());
            CHECK(scale.value() == Approx(2.0f));
        }
    }

    SECTION("more than two measurements")
    {
        const auto scale = armvo::ScaleFusion::fuse({{1.0f, 1.0f}, {2.0f, 1.0f}, {4.0f, 1.0f}});
        REQUIRE(scale.has_value());
        CHECK(scale.value() == Approx(2.0f));
    }
}
