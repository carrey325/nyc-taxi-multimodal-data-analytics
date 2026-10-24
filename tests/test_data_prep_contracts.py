import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_trips as prep
from build_source_manifest import build_input_manifest

FIXTURES = ROOT / "tests/fixtures/data_prep"


class Contracts(unittest.TestCase):
    def setUp(self):
        self.policy = prep.load_policy(ROOT / "configs/data_prep_demo.json")
        self.raw, self.source = prep.read_trip_source(FIXTURES / "trips.csv", "synthetic_trips")

    def test_lineage_survives_filtering_and_rename(self):
        frame, _ = prep.normalize_trips(self.raw, self.source, self.policy)
        kept, ledger = prep.filter_trips(frame, self.policy)
        self.assertIn(0, kept.source_row_number.tolist())
        self.assertNotIn(3, kept.source_row_number.tolist())
        self.assertIn(9, kept.source_row_number.tolist())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "renamed.csv"
            shutil.copyfile(FIXTURES / "trips.csv", path)
            raw2, source2 = prep.read_trip_source(path, "alias")
            frame2, _ = prep.normalize_trips(raw2, source2, self.policy)
        self.assertEqual(frame.trip_id.tolist(), frame2.trip_id.tolist())
        chunks = [prep.normalize_trips(chunk, self.source, self.policy)[0]
                  for chunk in (self.raw.iloc[10:], self.raw.iloc[:10])]
        self.assertEqual(set(frame.trip_id), set(pd.concat(chunks).trip_id))
        self.assertEqual(frame.fare_amount.tolist(), self.raw.loc[frame.index, "fare_amount"].tolist())
        self.assertEqual(len(frame), len(kept) + ledger.loc[ledger.is_primary].trip_id.nunique())

    def test_duplicate_source_and_trip_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate source"):
            build_input_manifest([FIXTURES / "trips.csv"] * 2, {})
        with tempfile.TemporaryDirectory() as directory:
            alias = Path(directory) / "alias.csv"
            shutil.copyfile(FIXTURES / "trips.csv", alias)
            with self.assertRaisesRegex(ValueError, "duplicate source"):
                build_input_manifest([FIXTURES / "trips.csv", alias], {})
        frame, _ = prep.normalize_trips(self.raw, self.source, self.policy)
        with self.assertRaisesRegex(ValueError, "trip_id"):
            prep.filter_trips(pd.concat([frame, frame.iloc[:1]]), self.policy)

    def test_identical_content_is_not_automatically_deleted(self):
        frame, _ = prep.normalize_trips(self.raw, self.source, self.policy)
        kept, _ = prep.filter_trips(frame, self.policy)
        pair = kept[kept.source_row_number.isin([0, 1])]
        self.assertEqual(len(pair), 2)
        self.assertEqual(pair.trip_id.nunique(), 2)
        self.assertEqual(pair.record_fingerprint.nunique(), 1)

    def test_tip_policy_and_units(self):
        frame, _ = prep.normalize_trips(self.raw, self.source, self.policy)
        self.assertAlmostEqual(frame.loc[0, "tip_rate_fare"], .2)
        self.assertAlmostEqual(frame.loc[0, "trip_duration_minutes"], 15)
        self.assertAlmostEqual(frame.loc[0, "average_speed_mph"], 8)
        self.assertTrue(pd.isna(frame.loc[8, "tip_or_not"]))
        kept, ledger = prep.filter_trips(frame, self.policy)
        self.assertIn(2, kept.source_row_number.tolist())
        self.assertNotIn(8, kept.source_row_number.tolist())
        self.assertTrue({"fare_not_positive_finite", "tip_negative_or_nonfinite",
                         "distance_not_positive_finite", "duration_not_positive_finite"} <= set(ledger.reason))
        positive = copy.deepcopy(self.policy)
        positive["filter"]["sample_policy"] = "positive_tip"
        kept_positive, _ = prep.filter_trips(frame, positive)
        self.assertNotIn(2, kept_positive.source_row_number.tolist())
        self.assertTrue((kept_positive.tip_usd > 0).all())
        overlap = ledger[ledger.source_row_number == 13]
        self.assertEqual(overlap.is_primary.sum(), 1)
        self.assertEqual(len(overlap), 3)

    def test_dst_requires_explicit_resolution(self):
        frame, ledger = prep.normalize_trips(self.raw, self.source, self.policy)
        self.assertNotIn(11, frame.source_row_number.tolist())
        self.assertNotIn(12, frame.source_row_number.tolist())
        self.assertTrue({"pickup_time_nonexistent", "pickup_time_ambiguous"} <= set(ledger.reason))
        raw = self.raw.iloc[:2].copy()
        raw["tpep_pickup_datetime"] = ["2024-11-03T01:30:00-04:00", "2024-11-03T05:30:00Z"]
        raw["tpep_dropoff_datetime"] = ["2024-11-03T01:45:00-04:00", "2024-11-03T05:45:00Z"]
        policy = copy.deepcopy(self.policy)
        policy["time"]["input_kind"] = "aware"
        aware, errors = prep.normalize_trips(raw, self.source, policy)
        self.assertTrue(errors.empty)
        self.assertEqual(aware.pickup_utc.iloc[0], aware.pickup_utc.iloc[1])
        self.assertEqual(aware.pickup_utc.iloc[0], pd.Timestamp("2024-11-03T05:30:00Z"))

    def test_manifest_counts_csv_records_and_parquet_footer(self):
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "multiline.csv"
            csv.write_text('a,b\n1,"two\nlines"\n2,test\n', encoding="utf-8")
            pq = Path(directory) / "small.parquet"
            pd.DataFrame({"a": [1, 2, 3]}).to_parquet(pq)
            manifest = build_input_manifest([csv, pq], {})
            self.assertEqual(manifest.row_count.tolist(), [2, 3])
            self.assertNotIn(str(Path(directory)), manifest.to_json())
            self.assertEqual(manifest.release_info.tolist(), ["unknown", "unknown"])

    def test_policy_rejects_unknown_fields_and_implicit_units(self):
        for mutate in (lambda p: p.update({"typo": True}),
                       lambda p: p["units"].update({"money": "EUR"}),
                       lambda p: p["time"].update({"ambiguous": "infer"}),
                       lambda p: p["filter"].update({"sample_policy": "all_cash"}),
                       lambda p: p["sampling"].update({"seed": 1.5})):
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                policy = copy.deepcopy(self.policy)
                mutate(policy)
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(policy), encoding="utf-8")
                with self.assertRaises(ValueError):
                    prep.load_policy(path)

    def test_missing_columns_and_reserved_lineage_fail(self):
        with self.assertRaisesRegex(ValueError, "synthetic_trips.*fare_amount"):
            prep.normalize_trips(self.raw.drop(columns="fare_amount"), self.source, self.policy)
        bad = self.raw.copy()
        bad["trip_id"] = "collision"
        with self.assertRaisesRegex(ValueError, "reserved"):
            prep.normalize_trips(bad, self.source, self.policy)


if __name__ == "__main__":
    unittest.main()
