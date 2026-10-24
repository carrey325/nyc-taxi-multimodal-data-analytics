"""Explicit preparation rules for a small, reconstructed public demonstration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

POLICY_KEYS = {
    "inputs": {"trips", "zones", "weather", "metadata"},
    "columns": {"pickup_time", "dropoff_time", "fare_usd", "tip_usd", "distance_miles",
                "payment_code", "pu_location_id", "do_location_id"},
    "units": {"money", "distance", "duration", "speed", "tip_rate"},
    "time": {"input_kind", "source_timezone", "feature_timezone", "ambiguous", "nonexistent"},
    "filter": {"card_values", "sample_policy", "max_duration_minutes", "max_fare_usd",
               "max_tip_usd", "max_distance_miles", "max_speed_mph", "max_tip_rate"},
    "weather": {"mode", "tolerance_minutes", "station_by_pickup_zone", "observed_at_kind",
                "available_at_semantics", "units"},
    "match_filter": {"require_pickup_zone", "require_dropoff_zone", "require_weather"},
    "split": {"mode", "version", "seed", "train_end", "validation_end", "existing_bucket",
              "existing_policy", "existing_version", "existing_seed"},
    "sampling": {"version", "seed", "stratum_column", "strata", "caps"},
}
LEDGER_COLUMNS = ["trip_id", "source_id", "source_row_number", "stage", "reason", "is_primary"]
GENERATED_COLUMNS = set(POLICY_KEYS["columns"]) | {
    "trip_id", "source_id", "source_file_sha256", "source_file_name", "source_row_number",
    "record_fingerprint", "pickup_utc", "dropoff_utc", "pickup_local", "pickup_hour",
    "pickup_weekday", "trip_duration_minutes", "average_speed_mph", "tip_rate_fare", "tip_or_not",
}


def digest_payload(value) -> str:
    """Versioned callers supply JSON-safe values; encoding is canonical UTF-8."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                    sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def load_policy(path: Path) -> dict:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_policy(policy)
    return policy


def validate_policy(policy: dict) -> None:
    expected = set(POLICY_KEYS) | {"version", "status", "provenance"}
    if set(policy) != expected:
        raise ValueError("Policy has missing or unknown top-level fields")
    for section, keys in POLICY_KEYS.items():
        if not isinstance(policy[section], dict) or set(policy[section]) != keys:
            raise ValueError(f"Policy {section}: missing or unknown fields")
    if policy["version"] != "data-prep-demo-v1" or policy["status"] != "reconstructed_public_demo":
        raise ValueError("Only the versioned reconstructed public demo is supported")
    if policy["units"] != dict(money="USD", distance="mile", duration="minute", speed="mph", tip_rate="fraction"):
        raise ValueError("Unsupported or implicit units")
    columns = policy["columns"]
    if any(not isinstance(v, str) or not v for v in columns.values()) or len(set(columns.values())) != len(columns):
        raise ValueError("Column mapping must contain distinct raw column names")
    if set(columns.values()) & (GENERATED_COLUMNS | {"_source_row_number"}):
        raise ValueError("Raw column mapping collides with reserved prepared columns")
    t = policy["time"]
    if t["input_kind"] not in {"naive", "aware"} or t["ambiguous"] != "quarantine" or t["nonexistent"] != "quarantine":
        raise ValueError("Time requires naive/aware input and DST quarantine")
    for key in ("source_timezone", "feature_timezone"):
        ZoneInfo(t[key])
    f = policy["filter"]
    if f["sample_policy"] not in {"all_card_nonnegative", "positive_tip"}:
        raise ValueError("Unknown card sample policy")
    if not f["card_values"] or any(type(v) is not int for v in f["card_values"]):
        raise ValueError("card_values must explicitly list integer payment codes")
    for key, value in f.items():
        if key.startswith("max_") and value is not None:
            if type(value) not in {int, float} or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be null or positive finite")
    w = policy["weather"]
    if w["mode"] != "nearest_observation":
        raise ValueError("asof_available is not implemented; select descriptive nearest_observation")
    if w["observed_at_kind"] != "aware":
        raise ValueError("Weather observed_at must use explicit offsets")
    if type(w["tolerance_minutes"]) not in {int, float} or not np.isfinite(w["tolerance_minutes"]) or w["tolerance_minutes"] < 0:
        raise ValueError("Weather tolerance must be finite and nonnegative")
    if w["units"] != {"temperature_c": "degC", "precipitation_mm": "mm"}:
        raise ValueError("Weather units must be explicit degC and mm")
    if w["available_at_semantics"] not in {"synthetic_release_time", "documented_release_time", "unknown"}:
        raise ValueError("Unknown availability semantics")
    if not isinstance(w["station_by_pickup_zone"], dict) or any(
        not str(k).isdigit() or not isinstance(v, str) or not v for k, v in w["station_by_pickup_zone"].items()
    ):
        raise ValueError("An explicit pickup-zone to station mapping is required")
    if any(type(v) is not bool for v in policy["match_filter"].values()):
        raise ValueError("Match-filter decisions must be boolean")
    s = policy["split"]
    if (s["mode"], s["version"], s["existing_bucket"]) != ("hash_group", "sha256-group-v1", "preserve"):
        raise ValueError("Only hash_group v1 with existing-bucket preservation is supported")
    if (s["train_end"], s["validation_end"]) != (70, 85):
        raise ValueError("Supported bucket boundaries are 70 and 85")
    a = policy["sampling"]
    if a["version"] != "sha256-rank-v1" or a["stratum_column"] != "tip_or_not" or a["strata"] != [0, 1]:
        raise ValueError("Sampling v1 uses explicit tip_or_not strata [0,1]")
    for value in (s["seed"], a["seed"]):
        if type(value) is not int:
            raise ValueError("Seeds must be integers")
    if set(a["caps"]) != {"train", "validation", "test"} or any(type(v) is not int or v < 0 for v in a["caps"].values()):
        raise ValueError("Caps must be nonnegative integers for all three splits")
    inputs = policy["inputs"]
    if not isinstance(inputs["trips"], list) or not inputs["trips"]:
        raise ValueError("At least one trip input is required")
    for name in inputs["trips"] + [inputs[k] for k in ("zones", "weather", "metadata")]:
        if not isinstance(name, str) or not name or "/" in name or "\\" in name or ":" in name or name in {".", ".."}:
            raise ValueError("Input names must be local basenames")
    provenance = policy["provenance"]
    required_rules = {"columns_and_units", "card_target", "time_resolution", "economic_filters",
                      "identity_and_duplicates", "zone_join", "weather_join", "match_retention",
                      "split_and_sampling", "learning_boundary"}
    if set(provenance) != required_rules or any(
        set(v) != {"status", "source", "note"} or v["status"] not in {"recovered", "documented", "demo_assumption"}
        or not v["source"] or not v["note"] for v in provenance.values()
    ):
        raise ValueError("Each rule requires a provenance label, source and explanation")


def read_trip_source(path: Path, source_id: str) -> tuple[pd.DataFrame, dict]:
    from build_source_manifest import build_input_manifest
    entry = build_input_manifest([path], {Path(path).name: {"source_id": source_id}}).iloc[0].to_dict()
    if Path(path).suffix.lower() == ".csv":
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        raw = pd.read_parquet(path)
    if "_source_row_number" in raw:
        raise ValueError("Source contains reserved _source_row_number")
    # Original zero-based data-record ordinal, before any filtering or chunking.
    raw["_source_row_number"] = np.arange(len(raw), dtype=np.int64)
    return raw, entry


def assert_unique_ids(frame: pd.DataFrame) -> None:
    if "trip_id" not in frame or frame.trip_id.isna().any() or frame.trip_id.duplicated().any():
        raise ValueError("Missing or duplicate trip_id")


def exclusion_ledger(frame: pd.DataFrame, reasons: dict[str, pd.Series], stage: str) -> pd.DataFrame:
    records = []
    first = set()
    for reason, mask in reasons.items():
        for _, row in frame.loc[mask.fillna(False)].iterrows():
            records.append(dict(trip_id=row.trip_id, source_id=row.source_id,
                                source_row_number=int(row.source_row_number), stage=stage,
                                reason=reason, is_primary=row.trip_id not in first))
            first.add(row.trip_id)
    return pd.DataFrame(records, columns=LEDGER_COLUMNS)


def _parse_time(value, policy: dict) -> tuple[pd.Timestamp, str | None]:
    try:
        if not isinstance(value, (str, pd.Timestamp)) and not hasattr(value, "tzinfo"):
            return pd.NaT, "invalid"
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return pd.NaT, "invalid"
        aware = stamp.tzinfo is not None
        if aware != (policy["input_kind"] == "aware"):
            return pd.NaT, "timezone_kind_mismatch"
        if not aware:
            localized = stamp.tz_localize(policy["source_timezone"], ambiguous="NaT", nonexistent="NaT")
            if pd.isna(localized):
                # Probe only to classify the error; never retain a guessed fold.
                probe = stamp.tz_localize(policy["source_timezone"], ambiguous=True, nonexistent="NaT")
                return pd.NaT, "nonexistent" if pd.isna(probe) else "ambiguous"
            stamp = localized
        return stamp.tz_convert("UTC"), None
    except Exception as exc:
        kind = type(exc).__name__
        if kind == "AmbiguousTimeError":
            return pd.NaT, "ambiguous"
        if kind == "NonExistentTimeError":
            return pd.NaT, "nonexistent"
        return pd.NaT, "invalid"


def normalize_trips(raw: pd.DataFrame, source: dict, policy: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_policy(policy)
    required = set(policy["columns"].values()) | {"_source_row_number"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Source {source['source_id']} missing columns: {missing}")
    if set(raw) & GENERATED_COLUMNS:
        raise ValueError("Input contains reserved prepared columns")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", source["sha256"]):
        raise ValueError("Source sha256 must be a file digest")
    ordinals = pd.to_numeric(raw["_source_row_number"], errors="coerce")
    if ordinals.isna().any() or (ordinals < 0).any() or (ordinals % 1 != 0).any() or ordinals.duplicated().any():
        raise ValueError("Invalid or duplicate source record ordinal")
    frame = raw.drop(columns="_source_row_number").copy()
    frame["source_row_number"] = ordinals.astype("int64")
    frame["source_file_sha256"] = source["sha256"].lower()
    frame["source_file_name"] = source["file_name"]
    frame["source_id"] = source["source_id"]
    frame["trip_id"] = [digest_payload(["source-row-v1", source["sha256"].lower(), int(n)]) for n in ordinals]
    assert_unique_ids(frame)
    raw_fields = [policy["columns"][k] for k in sorted(policy["columns"])]
    frame["record_fingerprint"] = [
        digest_payload(["record-v1", [(k, None if pd.isna(v) or v == "" else str(v))
                                     for k, v in zip(sorted(policy["columns"]), row)]])
        for row in raw[raw_fields].itertuples(index=False, name=None)
    ]
    reasons = {}
    for key in ("pickup_time", "dropoff_time"):
        original = raw[policy["columns"][key]]
        frame[key] = original
        parsed = [_parse_time(v, policy["time"]) for v in original]
        target = key.replace("_time", "_utc")
        frame[target] = pd.Series(pd.to_datetime([v[0] for v in parsed], utc=True), index=frame.index)
        errors = pd.Series([v[1] for v in parsed], index=frame.index)
        for error in ("invalid", "timezone_kind_mismatch", "ambiguous", "nonexistent"):
            reasons[f"{key}_{error}"] = errors.eq(error)
    for key in ("fare_usd", "tip_usd", "distance_miles", "payment_code", "pu_location_id", "do_location_id"):
        values = pd.to_numeric(raw[policy["columns"][key]], errors="coerce")
        if key in {"payment_code", "pu_location_id", "do_location_id"}:
            values = values.where(np.isfinite(values) & values.mod(1).eq(0)).astype("Int64")
        frame[key] = values
    frame["trip_duration_minutes"] = (frame.dropoff_utc - frame.pickup_utc).dt.total_seconds() / 60
    frame["average_speed_mph"] = frame.distance_miles / (frame.trip_duration_minutes.where(frame.trip_duration_minutes > 0) / 60)
    card = frame.payment_code.isin(policy["filter"]["card_values"])
    valid_tip = card & np.isfinite(frame.tip_usd) & frame.tip_usd.ge(0)
    frame["tip_or_not"] = frame.tip_usd.gt(0).astype("Int64").where(valid_tip)
    frame["tip_rate_fare"] = (frame.tip_usd / frame.fare_usd.where(np.isfinite(frame.fare_usd) & frame.fare_usd.gt(0))).where(valid_tip)
    frame["pickup_local"] = frame.pickup_utc.dt.tz_convert(policy["time"]["feature_timezone"])
    frame["pickup_hour"] = frame.pickup_local.dt.hour.astype("Int64")
    frame["pickup_weekday"] = frame.pickup_local.dt.day_name()
    ledger = exclusion_ledger(frame, reasons, "normalize")
    return frame.loc[~frame.trip_id.isin(ledger.trip_id)].copy(), ledger


def filter_trips(frame: pd.DataFrame, policy: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_policy(policy)
    assert_unique_ids(frame)
    f = policy["filter"]
    reasons = {
        "fare_not_positive_finite": frame.fare_usd.isna() | ~np.isfinite(frame.fare_usd) | frame.fare_usd.le(0),
        "tip_negative_or_nonfinite": frame.tip_usd.isna() | ~np.isfinite(frame.tip_usd) | frame.tip_usd.lt(0),
        "distance_not_positive_finite": frame.distance_miles.isna() | ~np.isfinite(frame.distance_miles) | frame.distance_miles.le(0),
        "duration_not_positive_finite": frame.trip_duration_minutes.isna() | ~np.isfinite(frame.trip_duration_minutes) | frame.trip_duration_minutes.le(0),
    }
    # Nullable Parquet values must fail closed; finite inputs can also overflow
    # during division. Record that separately from invalid source values.
    valid_target_inputs = (~reasons["fare_not_positive_finite"]
                           & ~reasons["tip_negative_or_nonfinite"]
                           & frame.payment_code.isin(f["card_values"]))
    valid_speed_inputs = (~reasons["distance_not_positive_finite"]
                          & ~reasons["duration_not_positive_finite"])
    reasons["tip_rate_nonfinite"] = valid_target_inputs & ~np.isfinite(frame.tip_rate_fare).fillna(False)
    reasons["speed_nonfinite"] = valid_speed_inputs & ~np.isfinite(frame.average_speed_mph).fillna(False)
    caps = {"max_duration_minutes": "trip_duration_minutes", "max_fare_usd": "fare_usd",
            "max_tip_usd": "tip_usd", "max_distance_miles": "distance_miles",
            "max_speed_mph": "average_speed_mph", "max_tip_rate": "tip_rate_fare"}
    for cap, column in caps.items():
        if f[cap] is not None:
            reasons[cap + "_exceeded"] = frame[column].gt(f[cap])
    reasons["non_card_or_unknown_payment"] = ~frame.payment_code.isin(f["card_values"])
    if f["sample_policy"] == "positive_tip":
        reasons["zero_tip_excluded"] = frame.tip_usd.eq(0)
    ledger = exclusion_ledger(frame, reasons, "filter")
    return frame.loc[~frame.trip_id.isin(ledger.trip_id)].copy(), ledger
