from __future__ import annotations

from pathlib import Path

import pandas as pd

from research_config import OUTPUT_ROOT, SOURCE_MANIFEST_SPECS, ensure_workspace_dirs
from research_utils import infer_row_count


def build_manifest() -> pd.DataFrame:
    rows = []
    for source_id, path, family, purpose in SOURCE_MANIFEST_SPECS:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Required source file is missing: {resolved}")
        rows.append(
            {
                "source_id": source_id,
                "source_family": family,
                "path": str(resolved),
                "file_name": resolved.name,
                "suffix": resolved.suffix.lower(),
                "size_bytes": resolved.stat().st_size,
                "modified_time": pd.Timestamp(resolved.stat().st_mtime, unit="s"),
                "row_count": infer_row_count(resolved),
                "purpose": purpose,
            }
        )
    return pd.DataFrame(rows).sort_values(["source_family", "source_id"]).reset_index(drop=True)


def main() -> None:
    ensure_workspace_dirs()
    manifest = build_manifest()
    manifest.to_csv(OUTPUT_ROOT / "source_manifest.csv", index=False)
    print("Saved source manifest.")
    print(f"Source files tracked: {len(manifest)}")


if __name__ == "__main__":
    main()
