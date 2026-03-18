from __future__ import annotations

import unittest

from smart_tracker.environment import LandCoverClass, SeasonalVariation
from smart_tracker.visibility import SensorVisibilityModel


class TestLandCoverClassEnum(unittest.TestCase):
    """Verify that all LandCoverClass enum values exist with correct integer values."""

    def test_original_values_unchanged(self) -> None:
        self.assertEqual(int(LandCoverClass.OPEN), 0)
        self.assertEqual(int(LandCoverClass.URBAN), 1)
        self.assertEqual(int(LandCoverClass.FOREST), 2)
        self.assertEqual(int(LandCoverClass.WATER), 3)

    def test_new_enum_values(self) -> None:
        self.assertEqual(int(LandCoverClass.SCRUB), 4)
        self.assertEqual(int(LandCoverClass.WETLAND), 5)
        self.assertEqual(int(LandCoverClass.ROCKY), 6)
        self.assertEqual(int(LandCoverClass.SNOW), 7)
        self.assertEqual(int(LandCoverClass.ROAD), 8)

    def test_total_member_count(self) -> None:
        self.assertEqual(len(LandCoverClass), 9)

    def test_legend_includes_all_members(self) -> None:
        legend = LandCoverClass.legend()
        for member in LandCoverClass:
            self.assertIn(member.name.lower(), legend)
            self.assertEqual(legend[member.name.lower()], int(member))


class TestSensorVisibilityModelMappings(unittest.TestCase):
    """Verify SensorVisibilityModel has entries for every LandCoverClass member."""

    def setUp(self) -> None:
        self.model = SensorVisibilityModel()

    def test_attenuation_has_all_land_cover_types(self) -> None:
        for member in LandCoverClass:
            self.assertIn(
                int(member),
                self.model.attenuation_by_land_cover,
                f"Missing attenuation entry for {member.name}",
            )

    def test_noise_has_all_land_cover_types(self) -> None:
        for member in LandCoverClass:
            self.assertIn(
                int(member),
                self.model.noise_multiplier_by_land_cover,
                f"Missing noise entry for {member.name}",
            )

    def test_new_attenuation_values(self) -> None:
        att = self.model.attenuation_by_land_cover
        self.assertAlmostEqual(att[int(LandCoverClass.SCRUB)], 0.95)
        self.assertAlmostEqual(att[int(LandCoverClass.WETLAND)], 0.88)
        self.assertAlmostEqual(att[int(LandCoverClass.ROCKY)], 0.98)
        self.assertAlmostEqual(att[int(LandCoverClass.SNOW)], 0.97)
        self.assertAlmostEqual(att[int(LandCoverClass.ROAD)], 1.0)

    def test_new_noise_values(self) -> None:
        noise = self.model.noise_multiplier_by_land_cover
        self.assertAlmostEqual(noise[int(LandCoverClass.SCRUB)], 1.15)
        self.assertAlmostEqual(noise[int(LandCoverClass.WETLAND)], 1.25)
        self.assertAlmostEqual(noise[int(LandCoverClass.ROCKY)], 1.05)
        self.assertAlmostEqual(noise[int(LandCoverClass.SNOW)], 1.08)
        self.assertAlmostEqual(noise[int(LandCoverClass.ROAD)], 1.0)

    def test_optical_default_also_complete(self) -> None:
        model = SensorVisibilityModel.optical_default()
        for member in LandCoverClass:
            self.assertIn(int(member), model.attenuation_by_land_cover)
            self.assertIn(int(member), model.noise_multiplier_by_land_cover)


class TestSeasonalVariation(unittest.TestCase):
    """Test SeasonalVariation dataclass and its from_month factory."""

    def test_from_month_all_twelve_months(self) -> None:
        for month in range(1, 13):
            sv = SeasonalVariation.from_month(month)
            self.assertIn(sv.season, ("spring", "summer", "autumn", "winter"))
            self.assertGreaterEqual(sv.foliage_density_factor, 0.0)
            self.assertLessEqual(sv.foliage_density_factor, 1.0)
            self.assertIsInstance(sv.snow_cover, bool)

    def test_winter_months_have_snow_cover(self) -> None:
        for month in (12, 1, 2):
            sv = SeasonalVariation.from_month(month)
            self.assertEqual(sv.season, "winter")
            self.assertTrue(sv.snow_cover, f"Month {month} should have snow_cover=True")

    def test_winter_has_low_foliage(self) -> None:
        winter = SeasonalVariation.from_month(1)
        summer = SeasonalVariation.from_month(7)
        self.assertLess(winter.foliage_density_factor, summer.foliage_density_factor)

    def test_summer_has_highest_foliage(self) -> None:
        summer = SeasonalVariation.from_month(7)
        for month in range(1, 13):
            sv = SeasonalVariation.from_month(month)
            self.assertLessEqual(
                sv.foliage_density_factor,
                summer.foliage_density_factor,
                f"Month {month} ({sv.season}) should not exceed summer foliage",
            )

    def test_summer_foliage_is_one(self) -> None:
        summer = SeasonalVariation.from_month(6)
        self.assertAlmostEqual(summer.foliage_density_factor, 1.0)

    def test_non_winter_no_snow(self) -> None:
        for month in range(3, 12):  # March through November
            sv = SeasonalVariation.from_month(month)
            self.assertFalse(sv.snow_cover, f"Month {month} should have snow_cover=False")

    def test_season_mapping(self) -> None:
        expected = {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn",
        }
        for month, season in expected.items():
            sv = SeasonalVariation.from_month(month)
            self.assertEqual(sv.season, season, f"Month {month} should be {season}")

    def test_invalid_month_raises(self) -> None:
        with self.assertRaises(ValueError):
            SeasonalVariation.from_month(0)
        with self.assertRaises(ValueError):
            SeasonalVariation.from_month(13)

    def test_invalid_season_raises(self) -> None:
        with self.assertRaises(ValueError):
            SeasonalVariation(season="rainy", foliage_density_factor=0.5, snow_cover=False)

    def test_frozen(self) -> None:
        sv = SeasonalVariation.from_month(6)
        with self.assertRaises(AttributeError):
            sv.season = "winter"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
