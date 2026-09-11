#!/usr/bin/env python3
"""Disabled legacy SPC runner.

This entry point edited trusted system-prompt text and therefore did not
implement COMA's query-suffix threat model. It is retained only to fail closed
and direct callers to the auditable pipeline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


LEGACY_SPC_ERROR = (
    "run_guardrail_attack.py is disabled because it modifies trusted system "
    "prompt content. Use prepare_spc_blind_inputs.py, run_spc_stage1.py, "
    "run_spc_stage2.py, and run_spc_asr.py; only that query-suffix, "
    "shared-budget path may be reported as SPC."
)


def validate_surrogate(args) -> dict:
    """Retained for provenance validation used by existing callers/tests."""
    if not args.surrogate_revision and not args.surrogate_weight_sha256:
        return {"mode": "UNVERIFIED_MODEL_NAME_ONLY"}
    if not args.surrogate_revision or not args.surrogate_weight_sha256:
        raise ValueError(
            "pinning requires both --surrogate-revision and "
            "--surrogate-weight-sha256"
        )
    snapshot = Path(args.surrogate_model).resolve()
    weight = snapshot / "model.safetensors"
    if (
        not snapshot.is_dir()
        or snapshot.name != args.surrogate_revision
        or not weight.is_file()
    ):
        raise ValueError(
            "pinned surrogate must be a local revision directory containing "
            "model.safetensors"
        )
    digest = hashlib.sha256()
    with weight.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual.lower() != args.surrogate_weight_sha256.lower():
        raise ValueError("surrogate model.safetensors SHA-256 mismatch")
    return {
        "mode": "PINNED_LOCAL_SNAPSHOT",
        "snapshot": str(snapshot),
        "revision": args.surrogate_revision,
        "model_safetensors_sha256": actual,
    }


def reject_legacy_spc() -> None:
    raise RuntimeError(LEGACY_SPC_ERROR)


def main() -> None:
    raise SystemExit(f"ERROR: {LEGACY_SPC_ERROR}")


if __name__ == "__main__":
    main()
