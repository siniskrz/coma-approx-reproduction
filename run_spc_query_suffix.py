#!/usr/bin/env python3
"""Create boundary-safe suffix artifacts for the committed toy SPC dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from comattack.spc_query_suffix import build_artifact, validate_blind_row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Blind JSON/JSONL from prepare_spc_blind_inputs.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True, help="Local tokenizer snapshot")
    parser.add_argument("--suffix", default="compression robustness test marker")
    parser.add_argument("--compression-rate", type=float, choices=(0.5, 0.6, 0.7), default=0.6)
    parser.add_argument("--max-suffix-tokens", type=int, default=32)
    parser.add_argument("--surrogate-model", required=True)
    parser.add_argument("--surrogate-revision", required=True)
    parser.add_argument("--surrogate-weight-sha256", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data).resolve()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    text = data_path.read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    for row in rows:
        validate_blind_row(row)
    expected_ids = {f"toy-perm-{number:02d}" for number in range(1, 21)}
    if len(rows) != 20 or {row.get("sample_id") for row in rows} != expected_ids:
        raise ValueError("toy dataset identity check failed")

    surrogate = {
        "model": args.surrogate_model,
        "revision": args.surrogate_revision,
        "weight_sha256": args.surrogate_weight_sha256,
    }
    artifacts = [
        build_artifact(
            row,
            args.suffix,
            tokenizer,
            compression_rate=args.compression_rate,
            surrogate=surrogate,
            max_suffix_tokens=args.max_suffix_tokens,
        )
        for row in rows
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in artifacts) + "\n", encoding="utf-8")
    print(f"wrote {len(artifacts)} boundary-safe toy artifacts to {output}")


if __name__ == "__main__":
    main()
