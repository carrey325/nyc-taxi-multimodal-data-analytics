"""Order-invariant, within-period splits and fixed-cap sampling."""
from __future__ import annotations

import numpy as np
import pandas as pd

from prepare_trips import assert_unique_ids, digest_payload, validate_policy

SPLITS = ("train", "validation", "test")


def ids_digest(ids) -> str:
    return digest_payload(["sorted-trip-ids-v1", sorted(map(str, ids))])


def _labels(buckets: pd.Series) -> pd.Series:
    return pd.Series(np.select([buckets < 70, buckets < 85], ["train", "validation"],
                               default="test"), index=buckets.index)


def assign_splits(frame: pd.DataFrame, policy: dict) -> pd.DataFrame:
    validate_policy(policy)
    assert_unique_ids(frame)
    if "record_fingerprint" not in frame or frame.record_fingerprint.isna().any():
        raise ValueError("A nonmissing record_fingerprint is required")
    result = frame.copy()
    cfg = policy["split"]
    existing = "split_bucket" in result
    if existing:
        original = result.split_bucket
        buckets = pd.to_numeric(original, errors="coerce")
        if (original.map(lambda v: isinstance(v, (bool, np.bool_))).any()
                or buckets.isna().any() or (~np.isfinite(buckets)).any()
                or buckets.mod(1).ne(0).any() or not buckets.between(0, 99).all()):
            raise ValueError("Existing split bucket must be an integer from 0 to 99")
        result["split_bucket"] = buckets.astype("int64")
    else:
        if {"split_label", "split_policy", "split_version", "split_seed"} & set(result):
            raise ValueError("Existing split metadata is missing its bucket")
        result["split_bucket"] = [
            int(digest_payload([cfg["version"], cfg["seed"], str(group)]), 16) % 100
            for group in result.record_fingerprint
        ]
        result["split_bucket"] = result.split_bucket.astype("int64")
    labels = _labels(result.split_bucket)
    if "split_label" in result and not result.split_label.eq(labels).all():
        raise ValueError("Existing split label conflicts with bucket")
    result["split_label"] = labels
    if (result.groupby("record_fingerprint", observed=True).split_label.nunique() > 1).any():
        raise ValueError("An existing duplicate group crosses splits")
    metadata = {
        "split_policy": cfg["existing_policy"] if existing else cfg["mode"],
        "split_version": cfg["existing_version"] if existing else cfg["version"],
        "split_seed": cfg["existing_seed"] if existing else cfg["seed"],
    }
    for key, value in metadata.items():
        if key not in result:
            result[key] = str(value)
        elif result[key].isna().any():
            raise ValueError(f"Existing {key} cannot be missing")
    return result


def sample_trips(frame: pd.DataFrame, policy: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_policy(policy)
    assert_unique_ids(frame)
    # Also validates externally supplied split labels and duplicate groups.
    frame = assign_splits(frame, policy)
    cfg = policy["sampling"]
    stratum = cfg["stratum_column"]
    if stratum not in frame or frame[stratum].isna().any() or not frame[stratum].isin(cfg["strata"]).all():
        raise ValueError("Sampling requires complete, configured strata")
    if {"sample_hash", "sample_rank", "sample_version", "sample_seed"} & set(frame):
        raise ValueError("Input contains existing sample annotations")
    frame = frame.copy()
    frame["sample_hash"] = [digest_payload([cfg["version"], cfg["seed"], str(key)]) for key in frame.trip_id]
    samples, audits = [], []
    for split in SPLITS:
        for value in cfg["strata"]:
            group = frame.loc[frame.split_label.eq(split) & frame[stratum].eq(value)].sort_values(["sample_hash", "trip_id"])
            selected = group.head(cfg["caps"][split]).copy()
            selected["sample_rank"] = np.arange(1, len(selected) + 1, dtype=np.int64)
            selected["sample_version"] = cfg["version"]
            selected["sample_seed"] = cfg["seed"]
            samples.append(selected)
            audits.append(dict(split_label=split, stratum_column=stratum, stratum=value,
                               eligible_n=len(group), sample_n=len(selected),
                               selected_id_sha256=ids_digest(selected.trip_id)))
    sample = pd.concat(samples, ignore_index=True)
    assert_unique_ids(sample)
    return sample, pd.DataFrame(audits)
