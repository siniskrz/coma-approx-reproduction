"""Safe query-suffix artifacts for the public toy SPC dataset.

This module deliberately contains no attack optimisation.  It establishes the
paper threat-model boundary first: trusted fields are hashed for provenance but
never copied into an attack artifact, and the only emitted mutation is a short
suffix for the untrusted query.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Sequence


PROTOCOL = "PUBLIC_SURROGATE_QUERY_SUFFIX_V1"
FORBIDDEN_ARTIFACT_FIELDS = {
    "attacked_context",
    "attacked_prompt",
    "attacked_system",
    "attacked_system_prompt",
    "system_prompt",
    "context",
    "context_text",
    "external_context",
    "guardrail_list",
}
BLIND_FIELDS = {
    "sample_id",
    "public_surrogate_id",
    "source_hash",
    "original_query",
    "surrogate_query",
    "surrogate_prefix",
    "surrogate_guardrails",
    "critical_candidates",
}


def configure_right_padding(tokenizer):
    """Make individually computed token spans valid in a padded batch."""
    tokenizer.padding_side = "right"
    if tokenizer.padding_side != "right":
        raise ValueError("query-suffix batching requires right-padding tokenizer")
    return tokenizer


def validate_prompt_batch(prompts, guardrail_sentences, guardrail_keywords) -> None:
    """Reject empty or misaligned batched attack inputs before ``zip`` truncates."""
    lengths = tuple(map(len, (prompts, guardrail_sentences, guardrail_keywords)))
    if not lengths[0] or len(set(lengths)) != 1:
        raise ValueError(f"prompt, guardrail, and keyword batches must be non-empty and equal-length: {lengths}")


def _query(row: dict) -> str:
    value = row.get("adversarial_query", row.get("violation_query"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("toy row must contain a non-empty adversarial_query or violation_query")
    return value


def canonical_source_hash(row: dict) -> str:
    """Hash trusted source data without reproducing it in the attack artifact."""
    source = [
        row.get("system_prompt", ""),
        row.get("external_context", row.get("context_text", row.get("context", ""))),
        _query(row),
        row.get("guardrail_list", []),
    ]
    encoded = json.dumps(source, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _public_text(pool_row: dict) -> str:
    guardrails = pool_row.get("surrogate_guardrails", [])
    return "\n".join([str(pool_row.get("surrogate_prefix", "")), *map(str, guardrails)])


def prepare_blind_inputs(private_rows: list[dict], public_pool: list[dict]) -> list[dict]:
    """Join private provenance hashes to independent public surrogate rules."""
    if not private_rows or not public_pool:
        raise ValueError("private toy rows and public surrogate pool must be non-empty")
    private_fragments = []
    for row in private_rows:
        private_fragments.append(str(row.get("system_prompt", "")).strip())
        private_fragments.extend(
            str(item.get("sentence", "")).strip()
            for item in row.get("guardrail_list", [])
            if isinstance(item, dict)
        )
    private_fragments = [text.casefold() for text in private_fragments if len(text) >= 16]

    for pool_row in public_pool:
        public_text = _public_text(pool_row).casefold()
        if not public_text.strip():
            raise ValueError("public surrogate pool entries must be non-empty")
        if any(fragment in public_text or public_text in fragment for fragment in private_fragments):
            raise ValueError("public surrogate pool copies private toy policy text")

    blind = []
    for index, row in enumerate(private_rows):
        public = public_pool[index % len(public_pool)]
        public_id = public.get("pool_id")
        if not isinstance(public_id, str) or not public_id.strip():
            public_id = hashlib.sha256(json.dumps(public, ensure_ascii=False, sort_keys=True,
                                                  separators=(",", ":")).encode("utf-8")).hexdigest()
        candidate = {
            "sample_id": str(row.get("sample_id", row.get("id", ""))),
            "public_surrogate_id": public_id,
            "source_hash": canonical_source_hash(row),
            "original_query": _query(row),
            "surrogate_query": public.get("surrogate_query"),
            "surrogate_prefix": public.get("surrogate_prefix"),
            "surrogate_guardrails": public.get("surrogate_guardrails"),
            "critical_candidates": public.get("critical_candidates"),
        }
        validate_blind_row(candidate)
        blind.append(candidate)
    if len({row["sample_id"] for row in blind}) != len(blind):
        raise ValueError("private toy sample ids must be unique")
    return blind


def validate_blind_row(row: dict) -> None:
    extras = set(row) - BLIND_FIELDS
    missing = BLIND_FIELDS - set(row)
    if extras or missing:
        raise ValueError(f"blind row fields differ: missing={sorted(missing)}, extra={sorted(extras)}")
    if (not row["sample_id"] or not isinstance(row["public_surrogate_id"], str) or
            not row["public_surrogate_id"].strip() or
            any(not isinstance(row[key], str) or not row[key].strip()
                for key in ("original_query", "surrogate_query"))):
        raise ValueError("blind row requires sample_id, original_query, and surrogate_query")
    if not isinstance(row["source_hash"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["source_hash"]):
        raise ValueError("blind row source_hash must be lowercase SHA-256")
    for key in ("surrogate_guardrails", "critical_candidates"):
        if not isinstance(row[key], list) or not row[key] or any(
            not isinstance(value, str) or not value.strip() for value in row[key]
        ):
            raise ValueError(f"blind row {key} must be a non-empty string list")
    if not isinstance(row["surrogate_prefix"], str) or not row["surrogate_prefix"].strip():
        raise ValueError("blind row surrogate_prefix must be non-empty")


def suffix_token_ids(tokenizer, suffix: str, max_tokens: int = 32) -> list[int]:
    if not isinstance(suffix, str) or not suffix.strip():
        raise ValueError("attack_suffix must be a non-empty string")
    if not 1 <= max_tokens <= 32:
        raise ValueError("max suffix tokens must be between 1 and 32")
    ids = tokenizer.encode(suffix, add_special_tokens=False)
    if not ids or any(not isinstance(token, int) or isinstance(token, bool) for token in ids):
        raise ValueError("tokenizer must return non-empty integer token ids")
    if len(ids) > max_tokens:
        raise ValueError(f"attack_suffix is {len(ids)} tokens; limit is {max_tokens}")
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    if tokenizer.encode(decoded, add_special_tokens=False) != ids:
        raise ValueError("attack_suffix tokenizer encode/decode round trip is unstable")
    return ids


def append_suffix(original_query: str, attack_suffix: str) -> str:
    """The sole allowed mutation: append text to an untrusted query."""
    return f"{original_query.rstrip()} {attack_suffix.strip()}"


def build_artifact(
    row: dict,
    attack_suffix: str,
    tokenizer,
    *,
    compression_rate: float,
    surrogate: dict,
    max_suffix_tokens: int = 32,
) -> dict:
    """Build a non-mutating, provenance-bound toy query-suffix artifact."""
    validate_blind_row(row)
    if not 0 < compression_rate <= 1:
        raise ValueError("compression_rate must be in (0, 1]")
    required = ("model", "revision", "weight_sha256")
    if any(not isinstance(surrogate.get(key), str) or not surrogate[key].strip() for key in required):
        raise ValueError("surrogate model, revision, and weight_sha256 are required")

    ids = suffix_token_ids(tokenizer, attack_suffix, max_suffix_tokens)
    artifact = {
        "id": row["sample_id"],
        "protocol": PROTOCOL,
        "source_hash": row["source_hash"],
        "original_query": row["original_query"],
        "public_surrogate_prefix": row["surrogate_prefix"],
        "attack_suffix": attack_suffix.strip(),
        "suffix_token_ids": ids,
        "suffix_token_count": len(ids),
        "budget": {
            "compression_rate": compression_rate,
            "max_suffix_tokens": max_suffix_tokens,
        },
        "stage1": {
            "status": "NOT_RUN_ENGINEERING_BOUNDARY_ONLY",
            "surrogate_guardrails": row["surrogate_guardrails"],
            "critical_candidates": row["critical_candidates"],
            "trials": [],
        },
        "stage2": {
            "status": "NOT_RUN_ENGINEERING_BOUNDARY_ONLY",
            "max_steps": 0,
            "steps_run": 0,
            "best_loss": None,
            "candidates": [],
            "validated": False,
            "suffix_roundtrip_stable": True,
        },
        "surrogate": {key: surrogate[key] for key in required},
    }
    forbidden = FORBIDDEN_ARTIFACT_FIELDS.intersection(artifact)
    if forbidden:
        raise AssertionError(f"trusted fields leaked into artifact: {sorted(forbidden)}")
    return artifact


def target_span_mask(
    target_slices: Sequence[slice], attention_rows: Sequence[Sequence[int]]
) -> list[list[bool]]:
    """Return a right-padded mask that points only at the requested targets."""
    if len(target_slices) != len(attention_rows):
        raise ValueError("one target slice is required per prompt")
    masks = []
    for target, attention in zip(target_slices, attention_rows):
        width = len(attention)
        if target.start is None or target.stop is None or not 0 <= target.start < target.stop <= width:
            raise ValueError("target slice is outside the tokenized prompt")
        row = [False] * width
        for position in range(target.start, target.stop):
            row[position] = bool(attention[position])
        if not any(row):
            raise ValueError("target slice contains no attended tokens")
        masks.append(row)
    return masks


def require_validated_artifact(artifact: dict) -> None:
    """Prevent engineering-only artifacts from entering ASR calculations."""
    if artifact.get("protocol") != PROTOCOL:
        raise ValueError("unsupported query-suffix artifact protocol")
    if artifact.get("stage1", {}).get("status") != "COMPLETE":
        raise ValueError("Stage-I evidence is incomplete")
    if artifact.get("stage2", {}).get("validated") is not True:
        raise ValueError("Stage-II artifact is not validated")
