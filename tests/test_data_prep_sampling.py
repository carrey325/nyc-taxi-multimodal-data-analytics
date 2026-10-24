import copy
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_trips import load_policy
from prep_sampling import assign_splits, sample_trips
from weather_effects_models import collapse_categories


class Sampling(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy(ROOT / "configs/data_prep_demo.json")
        self.frame = pd.DataFrame({
            "trip_id": [f"synthetic-{i:03}" for i in range(200)],
            "record_fingerprint": ["same", "same"] + [f"content-{i}" for i in range(2, 200)],
            "tip_or_not": [i % 2 for i in range(200)],
        })

    def test_sampling_invariant_to_input_order(self):
        split = assign_splits(self.frame, self.policy)
        sample, audit = sample_trips(split, self.policy)
        reordered = pd.concat([self.frame.iloc[70:], self.frame.iloc[:70]]).sample(frac=1, random_state=10)
        other, other_audit = sample_trips(assign_splits(reordered, self.policy), self.policy)
        pd.testing.assert_frame_equal(sample, other)
        pd.testing.assert_frame_equal(audit, other_audit)
        changed = copy.deepcopy(self.policy)
        changed["sampling"]["seed"] = 43
        different, _ = sample_trips(split, changed)
        self.assertNotEqual(set(sample.trip_id), set(different.trip_id))

    def test_split_is_stable_and_disjoint(self):
        frame = assign_splits(self.frame, self.policy)
        self.assertEqual(frame.loc[0, "split_bucket"], frame.loc[1, "split_bucket"])
        self.assertEqual(frame.groupby("record_fingerprint").split_label.nunique().max(), 1)
        self.assertEqual(frame.trip_id.nunique(), len(self.frame))
        preserved = assign_splits(frame, self.policy)
        pd.testing.assert_frame_equal(frame, preserved)
        for value in (-1, 100, 70.5, float("nan"), True):
            bad = self.frame.copy()
            bad["split_bucket"] = value
            with self.assertRaisesRegex(ValueError, "bucket"):
                assign_splits(bad, self.policy)
        bad = self.frame.copy()
        bad["split_bucket"] = 0
        bad.loc[1, "split_bucket"] = 99
        with self.assertRaisesRegex(ValueError, "duplicate group"):
            assign_splits(bad, self.policy)
        with self.assertRaisesRegex(ValueError, "trip_id"):
            assign_splits(pd.concat([self.frame, self.frame.iloc[:1]]), self.policy)

    def test_sample_caps_and_empty_strata(self):
        split = assign_splits(self.frame, self.policy)
        sample, audit = sample_trips(split, self.policy)
        self.assertEqual(len(audit), 6)
        self.assertEqual(len(sample), int(audit.sample_n.sum()))
        self.assertEqual(len(sample), sample.trip_id.nunique())
        self.assertTrue((audit.sample_n <= audit.eligible_n).all())
        self.assertEqual(audit.sample_n.tolist(), [4, 4, 3, 3, 3, 3])
        empty, empty_audit = sample_trips(split.iloc[:0], self.policy)
        self.assertTrue(empty.empty)
        self.assertEqual(empty_audit.sample_n.sum(), 0)
        self.assertEqual(len(empty_audit), 6)
        only_one, small_audit = sample_trips(split.iloc[:1], self.policy)
        self.assertEqual(len(only_one), 1)
        self.assertEqual(small_audit.eligible_n.sum(), 1)

    def test_split_boundary_and_category_fit(self):
        frame = self.frame.iloc[2:8].copy()
        frame["split_bucket"] = [0, 69, 70, 84, 85, 99]
        assigned = assign_splits(frame, self.policy)
        self.assertEqual(assigned.split_label.tolist(), ["train", "train", "validation", "validation", "test", "test"])
        self.assertEqual(assigned.split_bucket.tolist(), frame.split_bucket.tolist())
        self.assertTrue(assigned.split_policy.eq("external_bucket").all())
        train = pd.DataFrame({"PULocationID": [164, 186], "DOLocationID": [132, 132]})
        test = pd.DataFrame({"PULocationID": [999], "DOLocationID": [999]})
        collapse_categories(train, [train, test])
        self.assertEqual(train.PULocationID_top.tolist(), ["164", "186"])
        self.assertEqual(test.PULocationID_top.tolist(), ["OTHER"])
        self.assertEqual(test.route_pair_top.tolist(), ["OTHER"])


if __name__ == "__main__":
    unittest.main()
