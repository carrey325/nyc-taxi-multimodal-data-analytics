"""Cardinality-checked left joins; weather observations are descriptive only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from prepare_trips import assert_unique_ids, validate_policy


def _check_preserved(before: pd.DataFrame, after: pd.DataFrame) -> None:
    assert_unique_ids(after)
    if before.trip_id.tolist() != after.trip_id.tolist():
        raise ValueError("Left join changed trip identities or order")


def _missing_rates(frame: pd.DataFrame) -> dict:
    return {col: float(frame[col].isna().mean()) if len(frame) else None for col in frame}


def join_zones(trips: pd.DataFrame, zones: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    assert_unique_ids(trips)
    required = {"LocationID", "Zone", "Borough"}
    if not required <= set(zones):
        raise ValueError(f"Zones missing columns: {sorted(required - set(zones))}")
    keys = pd.to_numeric(zones.LocationID, errors="coerce")
    if keys.isna().any() or (~np.isfinite(keys)).any() or keys.mod(1).ne(0).any() or keys.duplicated().any():
        raise ValueError("Zone LocationID must be finite, integer, nonmissing and unique")
    right = zones[["Zone", "Borough"]].copy()
    right["LocationID"] = keys.astype("Int64")
    result = trips.copy()
    audit = dict(input_rows=len(trips), right_rows=len(zones), right_duplicate_rows=0,
                 cardinality="many_to_one", join="left", removed_rows=0)
    for side, key in (("pickup", "pu_location_id"), ("dropoff", "do_location_id")):
        names = {"LocationID": key, "Zone": f"{side}_zone", "Borough": f"{side}_borough"}
        generated = {f"{side}_zone", f"{side}_borough", f"{side}_zone_matched"}
        if generated & set(result):
            raise ValueError("Zone join would overwrite prepared columns")
        marker = f"_{side}_merge"
        result = result.merge(right.rename(columns=names), on=key, how="left", sort=False,
                              validate="many_to_one", indicator=marker)
        result[f"{side}_zone_matched"] = result.pop(marker).eq("both")
        audit[f"{side}_matched_rows"] = int(result[f"{side}_zone_matched"].sum())
        audit[f"{side}_unmatched_rows"] = len(result) - audit[f"{side}_matched_rows"]
    _check_preserved(trips, result)
    audit.update(output_rows=len(result), missing_rates=_missing_rates(result))
    return result, audit


def _aware_times(values: pd.Series, field: str, allow_missing: bool = False) -> pd.Series:
    output = []
    for value in values:
        if allow_missing and (pd.isna(value) or value == ""):
            output.append(pd.NaT)
            continue
        try:
            timestamp = pd.Timestamp(value)
        except Exception as exc:
            raise ValueError(f"Weather {field} contains an invalid timestamp") from exc
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise ValueError(f"Weather {field} requires valid offset-aware timestamps")
        output.append(timestamp.tz_convert("UTC"))
    return pd.Series(pd.to_datetime(output, utc=True), index=values.index)


def join_weather(trips: pd.DataFrame, weather: pd.DataFrame, policy: dict) -> tuple[pd.DataFrame, dict]:
    validate_policy(policy)
    assert_unique_ids(trips)
    required = {"weather_record_id", "station_id", "observed_at", "temperature_c", "precipitation_mm"}
    if not required <= set(weather):
        raise ValueError(f"Weather missing columns: {sorted(required - set(weather))}")
    w = weather.copy().reset_index(drop=True)
    for column in ("weather_record_id", "station_id"):
        if w[column].isna().any() or w[column].astype(str).eq("").any():
            raise ValueError(f"Weather {column} cannot be missing")
        w[column] = w[column].astype(str)
    w["observed_at"] = _aware_times(w.observed_at, "observed_at")
    if w.duplicated(["station_id", "observed_at"]).any():
        raise ValueError("Duplicate weather station/observed_at key")
    if w.weather_record_id.duplicated().any():
        raise ValueError("Duplicate weather_record_id")
    w["available_at"] = _aware_times(w.available_at, "available_at", True) if "available_at" in w else pd.Series(pd.NaT, index=w.index, dtype="datetime64[ns, UTC]")
    if (w.available_at < w.observed_at).any():
        raise ValueError("Observed weather available_at cannot precede observed_at")
    for field in ("temperature_c", "precipitation_mm"):
        w[field] = pd.to_numeric(w[field], errors="raise")
        if (~np.isfinite(w[field].dropna())).any():
            raise ValueError(f"Weather {field} contains nonfinite values")
    groups = {str(station): group.sort_values("observed_at").reset_index(drop=True)
              for station, group in w.groupby("station_id", observed=True)}
    cfg = policy["weather"]
    tolerance = pd.Timedelta(minutes=cfg["tolerance_minutes"])
    matches = []
    for _, row in trips.iterrows():
        zone = str(int(row.pu_location_id)) if pd.notna(row.pu_location_id) else None
        station = cfg["station_by_pickup_zone"].get(zone)
        candidates = groups.get(station)
        selected = None
        if candidates is not None and pd.notna(row.pickup_utc):
            times = pd.DatetimeIndex(candidates.observed_at)
            position = times.searchsorted(row.pickup_utc)
            neighbors = [i for i in (position - 1, position) if 0 <= i < len(times)]
            if neighbors:
                best = min(neighbors, key=lambda i: (abs(times[i] - row.pickup_utc), times[i]))
                if abs(times[best] - row.pickup_utc) <= tolerance:
                    selected = candidates.iloc[best]
        matched = selected is not None
        matches.append({
            "weather_station_id": station,
            "weather_record_id": selected.weather_record_id if matched else None,
            "weather_observed_at": selected.observed_at if matched else pd.NaT,
            "weather_available_at": selected.available_at if matched else pd.NaT,
            "temperature_c": selected.temperature_c if matched else np.nan,
            "precipitation_mm": selected.precipitation_mm if matched else np.nan,
            "weather_matched": matched,
        })
    columns = ["weather_station_id", "weather_record_id", "weather_observed_at", "weather_available_at",
               "temperature_c", "precipitation_mm", "weather_matched"]
    extra = pd.DataFrame(matches, columns=columns, index=trips.index)
    if set(columns) & set(trips):
        raise ValueError("Weather join would overwrite prepared columns")
    result = pd.concat([trips.copy(), extra], axis=1)
    for col in ("weather_observed_at", "weather_available_at"):
        result[col] = pd.to_datetime(result[col], utc=True)
    result["weather_matched"] = result.weather_matched.astype(bool)
    result["weather_lag_minutes"] = (result.weather_observed_at - result.pickup_utc).dt.total_seconds() / 60
    result["weather_observation_after_pickup"] = result.weather_lag_minutes.gt(0).astype("boolean").where(result.weather_matched)
    result["weather_available_after_pickup"] = result.weather_available_at.gt(result.pickup_utc).astype("boolean").where(result.weather_available_at.notna())
    result["weather_mode"] = cfg["mode"]
    result["weather_descriptive_only"] = True
    _check_preserved(trips, result)
    lags = result.weather_lag_minutes.dropna()
    audit = dict(input_rows=len(trips), output_rows=len(result), removed_rows=0,
                 right_rows=len(weather), right_duplicate_rows=0, mode=cfg["mode"],
                 matched_rows=int(result.weather_matched.sum()),
                 unmatched_rows=int((~result.weather_matched).sum()),
                 observation_after_pickup_rows=int(result.weather_observation_after_pickup.sum()),
                 available_after_pickup_rows=int(result.weather_available_after_pickup.sum()),
                 availability_unknown_rows=int(result.weather_available_at.isna().sum()),
                 available_at_semantics=cfg["available_at_semantics"],
                 lag_minutes_min=float(lags.min()) if len(lags) else None,
                 lag_minutes_max=float(lags.max()) if len(lags) else None,
                 missing_rates=_missing_rates(result))
    return result, audit
