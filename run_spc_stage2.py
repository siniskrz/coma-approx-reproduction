#!/usr/bin/env python3
"""Optimize and validate low-risk toy SPC query suffixes on a public surrogate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from comattack.spc_query_suffix import FORBIDDEN_ARTIFACT_FIELDS, suffix_token_ids
from comattack.spc_stages import (RATES, make_stage_two_artifact, optimize_suffix_checkpoints,
                                  public_attack_prompt, stage_two_inputs,
                                  validate_stage_one_surrogate_identity)
from run_spc_asr import (LLMLingua2, load_records, model_auxiliary_manifest,
                         model_weight_manifest)


def target_absent(text: str, target: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)", text,
                     flags=re.IGNORECASE) is None


def validate_stage1_surrogate_identity(rows: list[dict], surrogate: dict) -> None:
    """Bind Stage-I compression evidence to the Stage-II surrogate snapshot."""
    for row in rows:
        stage1 = row.get("stage1", {})
        if stage1.get("status") != "COMPLETE":
            continue
        validate_stage_one_surrogate_identity(stage1, surrogate)


def run_stage2(rows, attacker, tokenizer, compress_at_rate, surrogate,
               *, initial_suffix: str, checkpoint_every: int = 25,
               max_suffix_tokens: int = 32, max_steps: int = 500,
               diagnose_target=None,
               evidence_class: str = "SIMULATED_OR_CUSTOM") -> list[dict]:
    """Run 500 iterative steps, then require removal at all three budgets."""
    if evidence_class not in {"LIVE_MODEL", "SIMULATED_OR_CUSTOM"}:
        raise ValueError("unsupported Stage-II evidence class")
    if evidence_class == "LIVE_MODEL" and diagnose_target is None:
        raise ValueError("live Stage-II requires target score/cutoff diagnostics")
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
        diagnostic_cache = {}

        def diagnostics(suffix: str) -> dict:
            if suffix not in diagnostic_cache:
                diagnostic_cache[suffix] = (diagnose_target(
                    render(suffix), target_text, target_sentence) if diagnose_target else {})
            return diagnostic_cache[suffix]

        clean_diagnostics = diagnostics("")
        clean_metrics = {item["compression_rate"]: item
                         for item in clean_diagnostics.get("budget_metrics", [])}
        clean_trials = []
        for rate in RATES:
            compressed = compress_at_rate(render(""), rate)
            text = compressed.get("text") if isinstance(compressed, dict) else None
            if not isinstance(text, str) or compressed.get("raw") is None:
                raise ValueError("Stage-II compressor returned no text/raw evidence")
            metric = clean_metrics.get(rate, {})
            clean_trials.append({"compression_rate": rate,
                                 "target_retained": not target_absent(text, target_text),
                                 **({"score": clean_diagnostics["target_keep_score"],
                                     "cutoff": metric["cutoff"],
                                     "margin": metric["margin"]} if metric else {}),
                                 "compressed_text": text, "compressor_raw": compressed["raw"]})
        if not all(trial["target_retained"] for trial in clean_trials):
            results.append({"id": str(row.get("sample_id", "")), "skip": True,
                            "reason": "target is not retained by clean compression at every budget",
                            "clean_budget_trials": clean_trials})
            continue
        if hasattr(attacker, "best_loss"):
            attacker.best_loss = float("inf")
        if hasattr(attacker, "best_candidates"):
            attacker.best_candidates = None
        if hasattr(attacker, "best_metrics"):
            attacker.best_metrics = None
        def validate(suffix: str, rate: float) -> dict:
            prompt = render(suffix)
            compressed = compress_at_rate(prompt, rate)
            text = compressed.get("text") if isinstance(compressed, dict) else None
            if not isinstance(text, str) or compressed.get("raw") is None:
                raise ValueError("Stage-II compressor returned no text/raw evidence")
            model = diagnostics(suffix)
            metric = next((item for item in model.get("budget_metrics", [])
                           if item["compression_rate"] == rate), None)
            return {"target_removed": target_absent(text, target_text),
                    **({"score": model["target_keep_score"],
                        "cutoff": metric["cutoff"],
                        "margin": metric["margin"]} if metric else {}),
                    "compressed_text": text, "compressor_raw": compressed.get("raw")}

        def calibrate(checkpoint: dict) -> dict:
            trials = [{"compression_rate": rate, **validate(checkpoint["suffix"], rate)}
                      for rate in RATES]
            removed = sum(trial["target_removed"] is True for trial in trials)
            return {"budget_trials": trials, "target_removed_count": removed,
                    "all_rates_removed": removed == len(trials),
                    "decision_source": "REAL_COMPRESSOR_TARGET_REMOVAL"}

        candidates, steps_run, loss_history = optimize_suffix_checkpoints(
            attacker, render(initial_suffix), target_sentence, target_text, tokenizer,
            max_steps=max_steps, checkpoint_every=checkpoint_every,
            max_suffix_tokens=max_suffix_tokens, render_prompt=render,
            calibrate=calibrate,
        )

        artifact = make_stage_two_artifact(
            row, candidates, validate, surrogate, steps_run=steps_run,
            max_suffix_tokens=max_suffix_tokens,
            raw_provenance={"optimizer": type(attacker).__name__,
                            "evidence_class": evidence_class,
                            "initial_suffix_token_ids": initial_ids,
                            "checkpoint_every": checkpoint_every,
                            "budgets": list(RATES),
                            "proxy_loss_role": "PROPOSAL_ONLY",
                            "validation_decision": "REAL_COMPRESSOR_TARGET_REMOVAL_ALL_BUDGETS",
                            "python_version": sys.version,
                            "sample_batch_size": getattr(getattr(attacker, "config", None),
                                                         "sample_batch_size", None),
                            "top_k": getattr(getattr(attacker, "config", None), "top_k", None),
                            "eval_batch_size": getattr(getattr(attacker, "config", None),
                                                       "eval_batch_size", None),
                            "target_loss_weight": getattr(attacker, "target_loss_weight", 0.0),
                            "margin_hinge_weight": getattr(attacker, "margin_hinge_weight", 0.0),
                            "margin_delta": getattr(attacker, "margin_delta", 0.0),
                            "coordinate_width": getattr(attacker, "coordinate_width", 1),
                            "optimization_rate": getattr(attacker, "compression_rates", RATES),
                            "seed": getattr(getattr(attacker, "config", None), "seed", None)},
        )
        artifact["stage2"]["loss_history"] = loss_history
        artifact["stage2"]["clean_budget_trials"] = clean_trials
        if not artifact["stage2"]["validated"] and loss_history:
            artifact["stage2"]["best_loss"] = loss_history[-1]["best_objective_loss"]
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
    parser.add_argument("--max-steps", type=int, default=500,
                        help="experimental step limit; default is paper-aligned 500")
    parser.add_argument("--target-loss-weight", type=float, default=0.0)
    parser.add_argument("--margin-hinge-weight", type=float, default=0.0)
    parser.add_argument("--margin-delta", type=float, default=0.0)
    parser.add_argument("--optimization-rate", type=float, choices=RATES,
                        help="budget used by the differentiable optimizer; hard validation remains all three rates")
    parser.add_argument("--coordinate-width", type=int, choices=(1, 2), default=1,
                        help="mutated suffix coordinates per proposal")
    parser.add_argument("--require-clean-target-retention", action="store_true",
                        default=True,
                        help="required: skip samples whose clean compression drops the target")
    args = parser.parse_args()
    if not 1 <= args.max_suffix_tokens <= 32:
        parser.error("--max-suffix-tokens must be between 1 and 32")
    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items must be at least 1")
    if not 1 <= args.max_steps <= 500:
        parser.error("--max-steps must be between 1 and 500")

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
                          seed=args.seed, target_loss_weight=args.target_loss_weight,
                          margin_hinge_weight=args.margin_hinge_weight,
                          margin_delta=args.margin_delta,
                          coordinate_width=args.coordinate_width,
                          require_clean_target_retention=args.require_clean_target_retention)
    validator = LLMLingua2(str(snapshot), args.surrogate_revision, 0.6, actual_hash)
    # Reuse the validator's token-classification model and tokenizer. Loading a
    # second identical checkpoint can otherwise exhaust an 8 GB toy-run GPU.
    attacker = AttackforLLMLingua2(config=config, model=validator.compressor.model,
                                   tokenizer=validator.compressor.tokenizer,
                                   rank_tokenizer=validator.compressor.oai_tokenizer,
                                   compression_rates=((args.optimization_rate,)
                                                      if args.optimization_rate else RATES))

    def compress_at_rate(text: str, rate: float) -> dict:
        return validator.compress_at_rate(text, rate)

    surrogate = {"model": args.surrogate_model, "revision": args.surrogate_revision,
                 "weight_sha256": actual_hash, "weight_files": weight_files,
                 "auxiliary_files": model_auxiliary_manifest(snapshot)}
    stage1_rows = load_records(args.stage1_results)[:args.max_items]
    validate_stage1_surrogate_identity(stage1_rows, surrogate)
    artifacts = run_stage2(stage1_rows, attacker, tokenizer,
                           compress_at_rate, surrogate,
                           initial_suffix=args.initial_suffix,
                           checkpoint_every=args.checkpoint_every,
                           max_suffix_tokens=args.max_suffix_tokens,
                           max_steps=args.max_steps,
                           diagnose_target=validator.target_diagnostics,
                           evidence_class="LIVE_MODEL")
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
