"""Build an auditable, small in-memory trip sample without network access."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import pandas as pd

from build_source_manifest import build_input_manifest, file_sha256
from prepare_trips import (
    assert_unique_ids, digest_payload, exclusion_ledger, filter_trips, load_policy,
    normalize_trips, read_trip_source,
)
from prep_joins import join_weather, join_zones, missing_rates
from prep_sampling import assign_splits, ids_digest, sample_trips
from research_config import PROJECT_ROOT


def _snapshot() -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                                check=True, capture_output=True, text=True).stdout
        git = {"head": head, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        git = {"head": "unknown", "dirty": "unknown"}
    files = ["prepare_trips.py", "prep_joins.py", "prep_sampling.py",
             "build_trip_sample.py", "build_source_manifest.py", "research_config.py"]
    git["preparation_code_sha256"] = digest_payload(
        {name: file_sha256(Path(__file__).parent / name) for name in files})
    return git


def _stage(name: str, before: int, frame: pd.DataFrame) -> dict:
    relationships = {"pickup": "pickup_zone_matched", "dropoff": "dropoff_zone_matched",
                     "weather": "weather_matched"}
    flags = [flag for flag in relationships.values() if flag in frame]
    unmatched = ~frame[flags] if flags else None
    return dict(stage=name, input_rows=before, output_rows=len(frame), removed_rows=before-len(frame),
                duplicate_rows=int(frame.record_fingerprint.duplicated().sum()) if "record_fingerprint" in frame else 0,
                unmatched_rows=int(unmatched.any(axis=1).sum()) if flags else None,
                **{f"{side}_unmatched_rows": int(unmatched[flag].sum()) if flag in frame else None
                   for side, flag in relationships.items()},
                missing_rates=missing_rates(frame))


def _validate_output(config: Path, inputs: Path, output: Path) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty")
    if (output == inputs or output.is_relative_to(inputs) or inputs.is_relative_to(output)
            or config.is_relative_to(output)):
        raise ValueError("Output must be independent of input/config locations")
    for protected in ("results", "scripts", "reports", "tests", "configs", "notebooks", ".git"):
        root = PROJECT_ROOT / protected
        if output == root or output.is_relative_to(root):
            raise ValueError(f"Output is inside the protected {protected} directory")


def run_preparation(config_path: Path, input_dir: Path, output_dir: Path) -> dict:
    config_path, input_dir, output_dir = (Path(p).resolve() for p in (config_path, input_dir, output_dir))
    _validate_output(config_path, input_dir, output_dir)
    policy = load_policy(config_path)
    names = policy["inputs"]
    metadata_path = input_dir / names["metadata"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    paths = [input_dir / n for n in names["trips"] + [names["zones"], names["weather"]]]
    if any(not p.resolve().is_relative_to(input_dir) for p in paths + [metadata_path]):
        raise ValueError("An input resolves outside the selected input directory")
    manifest = build_input_manifest(paths, metadata)
    frames, ledgers, raw_frames = [], [], []
    for name in names["trips"]:
        source = manifest.loc[manifest.file_name.eq(name)].iloc[0].to_dict()
        raw, actual = read_trip_source(input_dir / name, source["source_id"])
        if actual["sha256"] != source["sha256"]:
            raise ValueError(f"Input changed during preparation: {name}")
        frame, ledger = normalize_trips(raw, source, policy)
        raw_frames.append(raw)
        frames.append(frame)
        ledgers.append(ledger)
    raw = pd.concat(raw_frames, ignore_index=True)
    normalized = pd.concat(frames, ignore_index=True)
    assert_unique_ids(normalized)
    stages = [_stage("source_records", len(raw), raw), _stage("normalize", len(raw), normalized)]
    filtered, ledger = filter_trips(normalized, policy)
    ledgers.append(ledger)
    stages.append(_stage("filter", len(normalized), filtered))
    zones = pd.read_csv(input_dir / names["zones"])
    weather = pd.read_csv(input_dir / names["weather"])
    zoned, zone_audit = join_zones(filtered, zones)
    stages.append(_stage("zone_join", len(filtered), zoned))
    joined, weather_audit = join_weather(zoned, weather, policy)
    stages.append(_stage("weather_join", len(zoned), joined))
    reasons = {}
    for required, flag, reason in (
        ("require_pickup_zone", "pickup_zone_matched", "unmatched_pickup_zone"),
        ("require_dropoff_zone", "dropoff_zone_matched", "unmatched_dropoff_zone"),
        ("require_weather", "weather_matched", "unmatched_weather"),
    ):
        if policy["match_filter"][required]:
            reasons[reason] = ~joined[flag]
    match_ledger = exclusion_ledger(joined, reasons, "match_filter")
    ledgers.append(match_ledger)
    eligible = joined.loc[~joined.trip_id.isin(match_ledger.trip_id)].copy()
    stages.append(_stage("match_filter", len(joined), eligible))
    split = assign_splits(eligible, policy)
    stages.append(_stage("split", len(eligible), split))
    sample, sample_audit = sample_trips(split, policy)
    stages.append(_stage("sample", len(split), sample))
    ledger = pd.concat(ledgers, ignore_index=True)
    primary = ledger.loc[ledger.is_primary.astype(bool)]
    if len(raw) != len(eligible) + len(primary) or primary.trip_id.duplicated().any():
        raise AssertionError("Exclusion ledger does not conserve rows")
    quality = {
        "status": policy["status"], "input_rows": len(raw), "normalized_rows": len(normalized),
        "filtered_rows": len(filtered), "eligible_rows": len(eligible), "sample_rows": len(sample),
        "primary_exclusion_counts": {k: int(v) for k, v in primary.reason.value_counts().sort_index().items()},
        "overlapping_reason_counts": {k: int(v) for k, v in ledger.reason.value_counts().sort_index().items()},
        "suspected_duplicate_rows": int(normalized.record_fingerprint.duplicated().sum()),
        "confirmed_duplicate_imports": 0,
        "joins": {"zones": zone_audit, "weather": weather_audit}, "stages": stages,
        "eligible_id_sha256": ids_digest(eligible.trip_id),
        "sample_id_sha256": ids_digest(sample.trip_id),
        "sample_audit": sample_audit.to_dict(orient="records"),
        "scope": "synthetic demonstration; no historical model or scale verification",
    }
    manifest_payload = {
        "status": policy["status"], "sources": manifest.to_dict(orient="records"),
        "config_file": config_path.name, "config_sha256": file_sha256(config_path),
        "metadata_file": metadata_path.name, "metadata_sha256": file_sha256(metadata_path),
        "effective_policy_sha256": digest_payload(policy), "code": _snapshot(),
        "versions": {"python": platform.python_version(), **{
            package: importlib.metadata.version(package) for package in ("pandas", "numpy", "pyarrow")
        }},
    }
    # Validate everything before creating output. Refuse a late race with another run.
    _validate_output(config_path, input_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(output_dir / "trip_sample.parquet", index=False)
    sample_audit.to_csv(output_dir / "sampling_audit.csv", index=False)
    ledger.to_csv(output_dir / "exclusion_ledger.csv", index=False)
    pd.DataFrame([{k: v for k, v in row.items() if k != "missing_rates"} for row in stages]).to_csv(
        output_dir / "sample_flow.csv", index=False)
    for filename, data in (("source_manifest.json", manifest_payload), ("quality_report.json", quality),
                           ("effective_policy.json", policy)):
        (output_dir / filename).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    quality = run_preparation(args.config, args.input_dir, args.output_dir)
    print(json.dumps({key: quality[key] for key in ("status", "input_rows", "eligible_rows", "sample_rows", "sample_id_sha256")}, indent=2))


if __name__ == "__main__":
    main()
