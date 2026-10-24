import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_trip_sample import run_preparation

FIXTURES = ROOT / "tests/fixtures/data_prep"
CONFIG = ROOT / "configs/data_prep_demo.json"


class CLI(unittest.TestCase):
    def test_cli_outputs_and_conservation(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            command = [sys.executable, str(ROOT / "scripts/build_trip_sample.py"),
                       "--config", str(CONFIG), "--input-dir", str(FIXTURES), "--output-dir", str(out)]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            required = {"trip_sample.parquet", "source_manifest.json", "quality_report.json",
                        "sample_flow.csv", "effective_policy.json", "exclusion_ledger.csv", "sampling_audit.csv"}
            self.assertEqual({p.name for p in out.iterdir()}, required)
            quality = json.loads((out / "quality_report.json").read_text())
            flow = pd.read_csv(out / "sample_flow.csv")
            self.assertEqual(quality["input_rows"], 33)
            self.assertEqual(quality["normalized_rows"], 30)
            self.assertEqual(quality["filtered_rows"], 23)
            self.assertTrue((flow.input_rows == flow.output_rows + flow.removed_rows).all())
            self.assertEqual(flow.output_rows.iloc[:-1].tolist(), flow.input_rows.iloc[1:].tolist())
            joins = flow[flow.stage.isin(["zone_join", "weather_join"])]
            self.assertTrue(joins.removed_rows.eq(0).all())
            self.assertEqual(quality["joins"]["zones"]["pickup_unmatched_rows"], 1)
            self.assertEqual(quality["joins"]["weather"]["unmatched_rows"], 2)
            sample = pd.read_parquet(out / "trip_sample.parquet")
            self.assertEqual(len(sample), quality["sample_rows"])
            self.assertEqual(sample.trip_id.nunique(), len(sample))
            self.assertTrue((sample.groupby("record_fingerprint").split_label.nunique() <= 1).all())
            ledger = pd.read_csv(out / "exclusion_ledger.csv")
            self.assertEqual(ledger.is_primary.sum(), 10)
            manifest = json.loads((out / "source_manifest.json").read_text())
            self.assertEqual(len(manifest["sources"]), 3)
            self.assertEqual(len(manifest["config_sha256"]), 64)
            self.assertTrue(all(s["status"] == "synthetic" for s in manifest["sources"]))
            for name in required - {"trip_sample.parquet"}:
                self.assertNotIn(str(ROOT), (out / name).read_text())
                self.assertNotIn(str(Path(directory)), (out / name).read_text())

    def test_two_runs_match_selected_ids_and_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            qa = run_preparation(CONFIG, FIXTURES, a)
            qb = run_preparation(CONFIG, FIXTURES, b)
            self.assertEqual(qa, qb)
            pd.testing.assert_frame_equal(pd.read_parquet(a / "trip_sample.parquet"),
                                          pd.read_parquet(b / "trip_sample.parquet"))
            pd.testing.assert_frame_equal(pd.read_csv(a / "sampling_audit.csv"),
                                          pd.read_csv(b / "sampling_audit.csv"))

    def test_refuses_existing_output_and_protected_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "occupied"
            out.mkdir()
            sentinel = out / "keep.txt"
            sentinel.write_text("keep")
            with self.assertRaisesRegex(ValueError, "empty"):
                run_preparation(CONFIG, FIXTURES, out)
            self.assertEqual(sentinel.read_text(), "keep")
            with self.assertRaises(ValueError):
                run_preparation(CONFIG, FIXTURES, FIXTURES / "output")
            with self.assertRaises(ValueError):
                run_preparation(CONFIG, FIXTURES, ROOT / "results" / "new_preparation")

    def test_missing_columns_fail_without_partial_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, out = Path(directory) / "inputs", Path(directory) / "out"
            shutil.copytree(FIXTURES, inputs)
            trips = pd.read_csv(inputs / "trips.csv")
            trips.drop(columns="fare_amount").to_csv(inputs / "trips.csv", index=False)
            with self.assertRaisesRegex(ValueError, "synthetic_trips.*fare_amount"):
                run_preparation(CONFIG, inputs, out)
            self.assertFalse(out.exists())

    def test_explicit_match_filter_and_positive_tip_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = json.loads(CONFIG.read_text())
            policy["match_filter"]["require_weather"] = True
            policy["filter"]["sample_policy"] = "positive_tip"
            config = Path(directory) / "positive.json"
            config.write_text(json.dumps(policy))
            quality = run_preparation(config, FIXTURES, Path(directory) / "output")
            sample = pd.read_parquet(Path(directory) / "output/trip_sample.parquet")
            self.assertTrue(sample.tip_usd.gt(0).all())
            self.assertTrue(sample.weather_matched.all())
            self.assertEqual(quality["primary_exclusion_counts"]["unmatched_weather"], 2)

    def test_all_excluded_still_writes_empty_audited_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, out = Path(directory) / "inputs", Path(directory) / "out"
            shutil.copytree(FIXTURES, inputs)
            raw = pd.read_csv(inputs / "trips.csv")
            raw["payment_type"] = 2
            raw.to_csv(inputs / "trips.csv", index=False)
            quality = run_preparation(CONFIG, inputs, out)
            self.assertEqual(quality["sample_rows"], 0)
            self.assertTrue(pd.read_parquet(out / "trip_sample.parquet").empty)


if __name__ == "__main__":
    unittest.main()
