#!/usr/bin/env python3
"""Prepare four-source SPC inputs while keeping trusted rules private."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from comattack.spc_query_suffix import prepare_blind_inputs


ROOT = Path(__file__).parent


def load_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON array or JSONL objects: {path}")
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_lineage(manifest_path: Path, private_path: Path, public_path: Path,
                     private_rows: list[dict], public_rows: list[dict]) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("lineage manifest must be a JSON object")
    checks = {
        "private_samples_sha256": sha256_file(private_path),
        "public_surrogates_sha256": sha256_file(public_path),
    }
    if any(manifest.get(key) != value for key, value in checks.items()):
        raise ValueError("private/public inputs differ from the lineage manifest")
    if (manifest.get("private_count"), manifest.get("public_count")) != (
        len(private_rows), len(public_rows)
    ):
        raise ValueError("private/public counts differ from the lineage manifest")
    lineage = manifest.get("lineage_sha256")
    if not isinstance(lineage, str) or any(
        row.get("lineage_sha256") != lineage for row in private_rows + public_rows
    ):
        raise ValueError("private/public rows differ from the declared transformation lineage")
    return lineage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Toy SPC inputs are intentionally not bundled; require explicit, freshly
    # transformed paths instead of retaining defaults to deleted artifacts.
    parser.add_argument("--private-data", required=True)
    parser.add_argument("--public-pool", required=True)
    parser.add_argument("--lineage-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    private_path, public_path = Path(args.private_data), Path(args.public_pool)
    private_rows, public_rows = load_rows(private_path), load_rows(public_path)
    lineage = validate_lineage(Path(args.lineage_manifest), private_path, public_path,
                               private_rows, public_rows)
    rows = prepare_blind_inputs(private_rows, public_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} blind SPC inputs to {output} (lineage {lineage})")


if __name__ == "__main__":
    main()
