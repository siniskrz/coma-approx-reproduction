#!/usr/bin/env python3
"""Run a clearly labelled formal Stage-I proxy for the generated low-risk rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comattack.spc_stages import select_dropout_target
from run_spc_asr import LLMLingua2, build_joint_prompt, load_records


def evidence(label: str, prompt: str) -> dict:
    answer = "DENIED" if label == "NO" else "APPROVED"
    return {
        "label": label,
        "backend": {"request": {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()},
                    "response": {"content": answer}, "content": answer, "error": None,
                    "model": "deterministic-formal-policy-v1"},
        "judge": {"request": {"response": answer}, "response": {"label": label},
                  "content": label, "error": None,
                  "model": "deterministic-formal-judge-v1"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compressor-snapshot", required=True)
    parser.add_argument("--compressor-revision", required=True)
    parser.add_argument("--compressor-weight-sha256", required=True)
    parser.add_argument("--compression-rate", type=float, default=0.6)
    args = parser.parse_args()

    compressor = LLMLingua2(args.compressor_snapshot, args.compressor_revision,
                            args.compression_rate, args.compressor_weight_sha256)
    output = []
    for row in load_records(args.blind_inputs):
        joint = build_joint_prompt(
            row["surrogate_prefix"] + "\n" + "\n".join(row["surrogate_guardrails"]),
            "", row["surrogate_query"]
        )
        compressed = compressor.compress(joint)
        target = row["critical_candidates"][0]

        def simulate(prompt: str) -> dict:
            retained = re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)", prompt,
                                 flags=re.IGNORECASE) is not None
            return evidence("NO" if retained else "YES", prompt)

        stage1 = select_dropout_target(
            compressed["text"], row["critical_candidates"], simulate,
            surrogate_guardrails=row["surrogate_guardrails"]
        )
        stage1["evidence_scope"] = "FORMAL_PROXY_NOT_VICTIM_VALIDATED"
        stage1["clean_joint_prompt"] = joint
        stage1["clean_compression"] = compressed
        output.append({**row, "stage1": stage1})

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n"
                                   for row in output), encoding="utf-8")
    counts = {}
    for row in output:
        status = row["stage1"]["status"]
        counts[status] = counts.get(status, 0) + 1
    print(json.dumps({"n": len(output), "stage1_status": counts,
                      "evidence_scope": "FORMAL_PROXY_NOT_VICTIM_VALIDATED"}, indent=2))


if __name__ == "__main__":
    main()
