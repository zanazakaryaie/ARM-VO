#include <catch2/catch.hpp>

#include <stdexcept>

#include "TemporalScaleFilter.hpp"

TEST_CASE("TemporalScaleFilter rejects non-positive maximum speed")
{
    CHECK_THROWS_AS(armvo::TemporalScaleFilter(10.0f, 0.0f), std::invalid_argument);
    CHECK_THROWS_AS(armvo::TemporalScaleFilter(10.0f, -1.0f), std::invalid_argument);
}

TEST_CASE("TemporalScaleFilter applies the speed limit before fusion")
{
    // 90 km/h at 10 fps permits a scale of at most 2.5 meters.
    armvo::TemporalScaleFilter tracker(10.0f, 90.0f);
    CHECK_FALSE(tracker.update({}).has_value());
    CHECK_FALSE(tracker.update({{2.51f, 1.0f}}).has_value());

    const auto scale = tracker.update({{2.5f, 1.0f}, {3.0f, 1000.0f}});
    REQUIRE(scale.has_value());
    CHECK(scale.value() == 2.5f);
}

TEST_CASE("TemporalScaleFilter rejects individual jumps before fusion")
{
    armvo::TemporalScaleFilter tracker(10.0f, 360.0f);
    REQUIRE(tracker.update({{1.0f, 1.0f}}).value() == 1.0f);

    SECTION("large upward jump cannot overwhelm a valid measurement")
    {
        const auto scale = tracker.update({{3.0f, 1000.0f}, {1.5f, 1.0f}});
        REQUIRE(scale.has_value());
        CHECK(scale.value() == 1.5f);
    }

    SECTION("large downward jump cannot overwhelm a valid measurement")
    {
        const auto scale = tracker.update({{0.1f, 1000.0f}, {1.5f, 1.0f}});
        REQUIRE(scale.has_value());
        CHECK(scale.value() == 1.5f);
    }

    SECTION("both jump boundaries are inclusive")
    {
        const auto scale = tracker.update({{0.4f, 1.0f}, {2.5f, 1.0f}});
        REQUIRE(scale.has_value());
        CHECK(scale.value() == Approx(1.0f));
    }

    SECTION("all candidates are compared with the same previous fused scale")
    {
        const auto scale = tracker.update({{0.5f, 1.0f}, {2.0f, 1.0f}});
        REQUIRE(scale.has_value());
        CHECK(scale.value() == Approx(1.0f));
    }
}

TEST_CASE("TemporalScaleFilter preserves fallback through missing or rejected measurements and recovers")
{
    armvo::TemporalScaleFilter tracker(10.0f, 360.0f);
    REQUIRE(tracker.update({{1.0f, 1.0f}}).value() == 1.0f);

    for (int frame = 0; frame < 3; ++frame)
    {
        CHECK(tracker.update({}).value() == 1.0f);
        CHECK(tracker.update({{0.1f, 1.0f}, {3.0f, 1.0f}}).value() == 1.0f);
    }

    CHECK(tracker.update({{2.0f, 1.0f}}).value() == 2.0f);
    // This exceeds the jump limit relative to 1.0, but is valid relative to 2.0.
    CHECK(tracker.update({{3.0f, 1.0f}}).value() == 3.0f);
    CHECK(tracker.update({}).value() == 3.0f);
}
