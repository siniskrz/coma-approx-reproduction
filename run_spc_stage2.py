#!/usr/bin/env python3
"""Optimize and validate low-risk toy SPC query suffixes on a public surrogate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from comattack.spc_query_suffix import FORBIDDEN_ARTIFACT_FIELDS, suffix_token_ids
from comattack.spc_stages import (make_stage_two_artifact, optimize_suffix_checkpoints,
                                  public_attack_prompt, stage_two_inputs)
from run_spc_asr import (LLMLingua2, load_records, model_auxiliary_manifest,
                         model_weight_manifest)


def target_absent(text: str, target: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)", text,
                     flags=re.IGNORECASE) is None


def run_stage2(rows, attacker, tokenizer, compress_at_rate, surrogate,
               *, initial_suffix: str, checkpoint_every: int = 25,
               max_suffix_tokens: int = 32) -> list[dict]:
    """Run 500 iterative steps, then require removal at all three budgets."""
    initial_ids = suffix_token_ids(tokenizer, initial_suffix, max_suffix_tokens)
    if not initial_ids:
        raise ValueError("initial suffix produced no tokens")
    results = []
    for row in rows:
        leaked_input = FORBIDDEN_ARTIFACT_FIELDS.intersection(row)
        if leaked_input:
            raise ValueError(f"Stage-II input leaks trusted fields: {sorted(leaked_input)}")
        if row.get("stage1", {}).get("status") != "COMPLETE":
            results.append({"id": str(row.get("sample_id", "")), "skip": True,
                            "reason": "Stage-I did not produce a validated behavior flip"})
            continue
        prefix, query, target_sentence, target_text = stage_two_inputs(row)
        guardrails = row["stage1"]["surrogate_guardrails"]
        render = lambda suffix: public_attack_prompt(prefix, guardrails, query, suffix)
        if hasattr(attacker, "best_loss"):
            attacker.best_loss = float("inf")
        if hasattr(attacker, "best_candidates"):
            attacker.best_candidates = None
        candidates, steps_run = optimize_suffix_checkpoints(
            attacker, render(initial_suffix), target_sentence, target_text, tokenizer,
            max_steps=500, checkpoint_every=checkpoint_every,
            max_suffix_tokens=max_suffix_tokens, render_prompt=render,
        )

        def validate(suffix: str, rate: float) -> dict:
            prompt = render(suffix)
            compressed = compress_at_rate(prompt, rate)
            text = compressed.get("text") if isinstance(compressed, dict) else None
            if not isinstance(text, str):
                raise ValueError("Stage-II compressor returned no text")
            return {"target_removed": target_absent(text, target_text),
                    "compressed_text": text, "compressor_raw": compressed.get("raw")}

        artifact = make_stage_two_artifact(
            row, candidates, validate, surrogate, steps_run=steps_run,
            max_suffix_tokens=max_suffix_tokens,
            raw_provenance={"optimizer": type(attacker).__name__,
                            "initial_suffix_token_ids": initial_ids,
                            "checkpoint_every": checkpoint_every,
                            "budgets": [0.5, 0.6, 0.7],
                            "python_version": sys.version,
                            "sample_batch_size": getattr(getattr(attacker, "config", None),
                                                         "sample_batch_size", None),
                            "top_k": getattr(getattr(attacker, "config", None), "top_k", None),
                            "eval_batch_size": getattr(getattr(attacker, "config", None),
                                                       "eval_batch_size", None),
                            "seed": getattr(getattr(attacker, "config", None), "seed", None)},
        )
        leaked = FORBIDDEN_ARTIFACT_FIELDS.intersection(artifact)
        if leaked:
            raise AssertionError(f"Stage-II artifact leaked trusted fields: {sorted(leaked)}")
        results.append(artifact)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--surrogate-snapshot", required=True)
    parser.add_argument("--surrogate-model", required=True,
                        help="canonical public surrogate model name")
    parser.add_argument("--surrogate-revision", required=True)
    parser.add_argument("--surrogate-weight-sha256", required=True)
    parser.add_argument("--initial-suffix", default="robustness marker test marker")
    parser.add_argument("--max-suffix-tokens", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--sample-batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-items", type=int,
                        help="optional smoke-test limit")
    args = parser.parse_args()
    if not 1 <= args.max_suffix_tokens <= 32:
        parser.error("--max-suffix-tokens must be between 1 and 32")
    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items must be at least 1")

    snapshot = Path(args.surrogate_snapshot).resolve()
    if not snapshot.is_dir() or snapshot.name != args.surrogate_revision:
        raise ValueError("surrogate snapshot directory name must equal its pinned revision")
    actual_hash, weight_files = model_weight_manifest(snapshot)
    if actual_hash.lower() != args.surrogate_weight_sha256.lower():
        raise ValueError("surrogate weight SHA-256 mismatch")

    from comattack.attacks.coma_utils import AttackConfig
    from comattack.attacks.extractive_suffix import AttackforLLMLingua2
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
    initial_ids = suffix_token_ids(tokenizer, args.initial_suffix, args.max_suffix_tokens)
    config = AttackConfig(model_name=str(snapshot), suffix_length=len(initial_ids),
                          num_steps=500, sample_batch_size=args.sample_batch_size,
                          top_k=args.top_k, eval_batch_size=args.eval_batch_size,
                          seed=args.seed)
    validator = LLMLingua2(str(snapshot), args.surrogate_revision, 0.6, actual_hash)
    # Reuse the validator's token-classification model and tokenizer. Loading a
    # second identical checkpoint can otherwise exhaust an 8 GB toy-run GPU.
    attacker = AttackforLLMLingua2(config=config, model=validator.compressor.model,
                                   tokenizer=validator.compressor.tokenizer)

    def compress_at_rate(text: str, rate: float) -> dict:
        raw = validator.compressor.compress_prompt(text, rate=rate)
        compressed = raw.get("compressed_prompt") if isinstance(raw, dict) else None
        if not isinstance(compressed, str):
            raise ValueError("LLMLingua2 returned no compressed_prompt")
        return {"text": compressed, "raw": raw}

    surrogate = {"model": args.surrogate_model, "revision": args.surrogate_revision,
                 "weight_sha256": actual_hash, "weight_files": weight_files,
                 "auxiliary_files": model_auxiliary_manifest(snapshot)}
    artifacts = run_stage2(load_records(args.stage1_results)[:args.max_items], attacker, tokenizer,
                           compress_at_rate, surrogate,
                           initial_suffix=args.initial_suffix,
                           checkpoint_every=args.checkpoint_every,
                           max_suffix_tokens=args.max_suffix_tokens)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n"
                              for row in artifacts),
                      encoding="utf-8")
    print(json.dumps({"n": len(artifacts),
                      "validated": sum(row.get("stage2", {}).get("validated") is True
                                       for row in artifacts),
                      "skipped": sum(row.get("skip") is True for row in artifacts)}, indent=2))


if __name__ == "__main__":
    main()
