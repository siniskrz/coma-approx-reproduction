#!/usr/bin/env python3
"""Prepare four-source SPC inputs while keeping trusted rules private."""

from __future__ import annotations

import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Toy SPC inputs are intentionally not bundled; require explicit, freshly
    # transformed paths instead of retaining defaults to deleted artifacts.
    parser.add_argument("--private-data", required=True)
    parser.add_argument("--public-pool", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = prepare_blind_inputs(load_rows(Path(args.private_data)), load_rows(Path(args.public_pool)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} blind SPC inputs to {output}")


if __name__ == "__main__":
    main()
