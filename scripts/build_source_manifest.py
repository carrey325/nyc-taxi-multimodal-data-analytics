from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import re

import pandas as pd
import pyarrow.parquet as pq

from research_config import OUTPUT_ROOT, SOURCE_MANIFEST_SPECS, ensure_workspace_dirs


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_input_manifest(paths: list[Path], metadata: dict) -> pd.DataFrame:
    """Inventory only selected sources, without publishing machine paths."""
    rows, seen = [], set()
    for path in map(Path, paths):
        digest = file_sha256(path)
        if digest in seen:
            raise ValueError(f"Confirmed duplicate source import: {path.name}")
        seen.add(digest)
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                columns = next(reader, [])
                if not columns or len(set(columns)) != len(columns):
                    raise ValueError(f"Invalid CSV columns: {path.name}")
                count = 0
                for row in reader:
                    if len(row) != len(columns):
                        raise ValueError(f"Invalid CSV record {count}: {path.name}")
                    count += 1
        elif path.suffix.lower() == ".parquet":
            footer = pq.ParquetFile(path)
            count, columns = footer.metadata.num_rows, footer.schema_arrow.names
        else:
            raise ValueError(f"Unsupported source format: {path.name}")
        info = metadata.get(path.name, {})
        row = dict(file_name=path.name, sha256=digest, row_count=count,
                   size_bytes=path.stat().st_size, columns=columns)
        for field in ("source_id", "status", "release_info", "acquired_at", "public_url", "schema_version"):
            value = info.get(field, path.stem if field == "source_id" else "unknown")
            if not isinstance(value, str) or re.search(r"[A-Za-z]:[\\/]|/(?:Users|home)/", value):
                raise ValueError(f"Nonpublic metadata field: {field} in {path.name}")
            row[field] = value
        if row["public_url"] not in {"unknown", "synthetic"}:
            from urllib.parse import urlsplit
            url = urlsplit(row["public_url"])
            if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password or url.query:
                raise ValueError("public_url must be a public URL without credentials or query parameters")
        rows.append(row)
    return pd.DataFrame(rows)


def build_manifest() -> pd.DataFrame:
    from research_utils import infer_row_count
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
