import copy
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_trips import load_policy, read_trip_source, normalize_trips, filter_trips
from prep_joins import join_zones, join_weather

FIXTURES = ROOT / "tests/fixtures/data_prep"


class Joins(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy(ROOT / "configs/data_prep_demo.json")
        raw, source = read_trip_source(FIXTURES / "trips.csv", "synthetic_trips")
        normalized, _ = normalize_trips(raw, source, self.policy)
        self.trips, _ = filter_trips(normalized, self.policy)
        self.zones = pd.read_csv(FIXTURES / "zones.csv")
        self.weather = pd.read_csv(FIXTURES / "weather.csv")

    def test_zone_left_join_preserves_trip_ids(self):
        result, audit = join_zones(self.trips, self.zones)
        self.assertEqual(result.trip_id.tolist(), self.trips.trip_id.tolist())
        self.assertEqual(audit["input_rows"], audit["output_rows"])
        unknown = result[result.source_row_number == 9].iloc[0]
        self.assertFalse(unknown.pickup_zone_matched)
        self.assertTrue(unknown.dropoff_zone_matched)
        self.assertTrue(pd.isna(unknown.pickup_zone))
        with self.assertRaisesRegex(ValueError, "LocationID"):
            join_zones(self.trips, pd.concat([self.zones, self.zones.iloc[:1]]))
        bad = self.zones.copy()
        bad.loc[0, "LocationID"] = float("nan")
        with self.assertRaisesRegex(ValueError, "LocationID"):
            join_zones(self.trips, bad)

    def test_nearest_is_descriptive_and_has_tolerance(self):
        result, audit = join_weather(self.trips, self.weather, self.policy)
        self.assertEqual(result.trip_id.tolist(), self.trips.trip_id.tolist())
        tie = result[result.source_row_number == 0].iloc[0]
        self.assertEqual(tie.weather_record_id, "SYN_W1")
        self.assertEqual(tie.weather_lag_minutes, -30)
        self.assertTrue(result.weather_mode.eq("nearest_observation").all())
        self.assertTrue(result.weather_descriptive_only.all())
        gap = result[result.source_row_number == 10].iloc[0]
        self.assertFalse(gap.weather_matched)
        self.assertTrue(pd.isna(gap.temperature_c))
        future = self.trips.iloc[:1].copy()
        future["pickup_utc"] = pd.to_datetime(["2024-01-15T15:50:00Z"])
        future, _ = join_weather(future, self.weather, self.policy)
        self.assertEqual(future.weather_record_id.iloc[0], "SYN_W2")
        self.assertTrue(future.weather_observation_after_pickup.iloc[0])
        self.assertTrue(future.weather_available_after_pickup.iloc[0])
        self.assertEqual(audit["output_rows"], len(self.trips))

    def test_availability_mode_is_rejected(self):
        policy = copy.deepcopy(self.policy)
        policy["weather"]["mode"] = "asof_available"
        for weather in (self.weather, self.weather.drop(columns="available_at")):
            with self.assertRaisesRegex(ValueError, "asof_available"):
                join_weather(self.trips, weather, policy)
        result, _ = join_weather(self.trips, self.weather.drop(columns="available_at"), self.policy)
        self.assertTrue(result.weather_available_at.isna().all())
        self.assertTrue(result.weather_available_after_pickup.isna().all())

    def test_weather_duplicate_keys_fail(self):
        duplicated = pd.concat([self.weather, self.weather.iloc[:1]], ignore_index=True)
        duplicated.loc[2, "weather_record_id"] = "SYN_ALIAS"
        duplicated.loc[2, "observed_at"] = "2024-01-15T10:00:00-05:00"
        with self.assertRaisesRegex(ValueError, "station.*observed"):
            join_weather(self.trips, duplicated, self.policy)

    def test_weather_requires_offset_and_true_release_order(self):
        for field, bad in (("observed_at", "2024-01-15 10:00:00"),
                           ("available_at", "2024-01-15T14:00:00Z")):
            weather = self.weather.copy()
            weather.loc[0, field] = bad
            with self.assertRaises(ValueError):
                join_weather(self.trips, weather, self.policy)

    def test_empty_weather_retains_unmatched_rows(self):
        result, audit = join_weather(self.trips, self.weather.iloc[:0], self.policy)
        self.assertEqual(len(result), len(self.trips))
        self.assertFalse(result.weather_matched.any())
        self.assertEqual(audit["unmatched_rows"], len(self.trips))


if __name__ == "__main__":
    unittest.main()
