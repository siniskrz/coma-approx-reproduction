"""Two-stage SPC helpers operating only on public surrogate prompts."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable


RATES = (0.5, 0.6, 0.7)


def content_suffix_bounds(special_tokens_mask: list[int], suffix_length: int) -> tuple[int, int]:
    """Locate final contiguous content tokens without consuming EOS or padding."""
    if suffix_length < 1:
        raise ValueError("suffix_length must be positive")
    content = [index for index, special in enumerate(special_tokens_mask) if not special]
    if len(content) < suffix_length:
        raise ValueError("prompt is shorter than the requested suffix")
    selected = content[-suffix_length:]
    if selected != list(range(selected[0], selected[-1] + 1)):
        raise ValueError("suffix content tokens are not contiguous")
    return selected[0], selected[-1] + 1


def delete_first_occurrence(text: str, candidate: str) -> tuple[str, dict] | tuple[None, None]:
    """Delete one exact word/phrase occurrence and retain its source position."""
    if not candidate or not candidate.strip():
        raise ValueError("critical candidate must be nonempty")
    match = re.search(r"(?<!\w)" + re.escape(candidate.strip()) + r"(?!\w)", text,
                      flags=re.IGNORECASE)
    if match is None:
        return None, None
    changed = text[:match.start()] + text[match.end():]
    occurrence = {"text": match.group(), "start": match.start(), "end": match.end()}
    return changed, occurrence


def select_dropout_target(
    compressed_prompt: str,
    critical_candidates: Iterable[str],
    simulate: Callable[[str], dict],
    *,
    surrogate_guardrails: Iterable[str],
) -> dict:
    """Greedily delete candidates until the surrogate Judge first flips NO→YES."""
    guardrails = list(surrogate_guardrails)
    if not guardrails or any(not isinstance(value, str) or not value.strip() for value in guardrails):
        raise ValueError("Stage-I requires non-empty public surrogate_guardrails")
    baseline = simulate(compressed_prompt)
    if baseline.get("label") != "NO":
        return {
            "status": "BASELINE_NOT_STABLE",
            "baseline_label": baseline.get("label", "UNKNOWN"),
            "counterfactual_label": None,
            "selected_target": None,
            "critical_occurrences": [],
            "baseline": baseline,
            "surrogate_guardrails": guardrails,
            "trials": [],
        }

    current = compressed_prompt
    trials = []
    deleted = []
    for candidate in critical_candidates:
        changed, occurrence = delete_first_occurrence(current, str(candidate))
        if changed is None:
            trials.append({"candidate": str(candidate), "status": "NOT_PRESENT"})
            continue
        outcome = simulate(changed)
        trial = {"candidate": str(candidate), "status": "EVALUATED",
                 "deleted_occurrence": occurrence, "target_prompt": changed,
                 "outcome": outcome}
        trials.append(trial)
        deleted.append(occurrence)
        current = changed
        if outcome.get("label") == "YES":
            return {
                "status": "COMPLETE",
                "baseline_label": "NO",
                "counterfactual_label": "YES",
                "selected_target": current,
                "critical_occurrences": deleted,
                "baseline": baseline,
                "surrogate_guardrails": guardrails,
                "trials": trials,
            }
        if outcome.get("label") not in {"NO", "YES"}:
            return {
                "status": "JUDGE_UNKNOWN",
                "baseline_label": "NO",
                "counterfactual_label": "UNKNOWN",
                "selected_target": None,
                "critical_occurrences": deleted,
                "baseline": baseline,
                "surrogate_guardrails": guardrails,
                "trials": trials,
            }

    return {
        "status": "NO_FEASIBLE_TARGET",
        "baseline_label": "NO",
        "counterfactual_label": "NO",
        "selected_target": None,
        "critical_occurrences": deleted,
        "baseline": baseline,
        "surrogate_guardrails": guardrails,
        "trials": trials,
    }


def stage_two_inputs(row: dict) -> tuple[str, str, str, str]:
    """Return public prefix/query/target sentence/target text from completed Stage-I."""
    stage1 = row.get("stage1")
    if not isinstance(stage1, dict) or stage1.get("status") != "COMPLETE":
        raise ValueError("Stage-II requires a completed Stage-I record")
    guardrails = stage1.get("surrogate_guardrails")
    occurrences = stage1.get("critical_occurrences")
    if not isinstance(guardrails, list) or not guardrails:
        raise ValueError("Stage-I record is missing surrogate_guardrails")
    if not isinstance(occurrences, list) or not occurrences:
        raise ValueError("Stage-I record is missing critical_occurrences")
    target = occurrences[-1].get("text") if isinstance(occurrences[-1], dict) else None
    prefix = row.get("surrogate_prefix", row.get("public_surrogate_prefix"))
    query = row.get("surrogate_query")
    if not all(isinstance(value, str) and value.strip() for value in (prefix, query, target)):
        raise ValueError("Stage-II public prefix, query, or selected target is missing")
    sentence = next((value for value in guardrails
                     if isinstance(value, str) and target.casefold() in value.casefold()), None)
    if sentence is None:
        raise ValueError("selected target is absent from surrogate_guardrails")
    return prefix, query, sentence, target


def public_attack_prompt(prefix: str, guardrails: Iterable[str], query: str, suffix: str) -> str:
    """Serialize only public surrogate material and the untrusted query suffix."""
    return ("<PUBLIC_SURROGATE>\n" + prefix + "\n" + "\n".join(guardrails) +
            "\n</PUBLIC_SURROGATE>\n<USER>\n" + query.rstrip() + " " + suffix.strip())


def optimize_suffix_checkpoints(
    attacker,
    prompt: str,
    target_sentence: str,
    target_text: str,
    tokenizer,
    *,
    max_steps: int = 500,
    checkpoint_every: int = 25,
    max_suffix_tokens: int = 32,
    render_prompt: Callable[[str], str] | None = None,
) -> tuple[list[dict], int]:
    """Run the existing optimizer and retain stable, bounded checkpoints."""
    if max_steps != 500:
        raise ValueError("paper-aligned Stage-II max_steps must be 500")
    if checkpoint_every < 1 or not 1 <= max_suffix_tokens <= 32:
        raise ValueError("invalid Stage-II checkpoint or suffix-token budget")
    candidates, seen = [], set()
    current_prompt = prompt
    for step in range(1, max_steps + 1):
        loss, raw_ids = attacker.step([current_prompt], [target_sentence], [target_text])
        ids = [int(value) for value in raw_ids]
        suffix = tokenizer.decode(ids, skip_special_tokens=True).strip()
        actual = tokenizer.encode(suffix, add_special_tokens=False) if suffix else []
        stable = actual == ids and 1 <= len(ids) <= max_suffix_tokens
        if stable and render_prompt is not None:
            current_prompt = render_prompt(suffix)
        checkpoint = {
            "step": step,
            "scheduled_checkpoint": step == 1 or step == max_steps or step % checkpoint_every == 0,
            "suffix": suffix,
            "suffix_token_ids": ids,
            "suffix_token_count": len(ids),
            "best_loss": float(loss),
            "suffix_roundtrip_stable": stable,
        }
        if stable and tuple(ids) not in seen:
            candidates.append(checkpoint)
            seen.add(tuple(ids))
    return candidates, max_steps


def make_stage_two_artifact(
    row: dict,
    candidates: list[dict],
    validate: Callable[[str, float], dict],
    surrogate: dict,
    *,
    steps_run: int = 500,
    max_suffix_tokens: int = 32,
    evaluation_rate: float = 0.6,
    raw_provenance: dict | None = None,
) -> dict:
    """Validate checkpoints at all paper budgets and emit the strict ASR schema."""
    result = validate_budget_candidates(candidates, validate)
    selected = result["selected"]
    stage2 = {
        "status": "COMPLETE" if selected else "NO_VALIDATED_SUFFIX",
        "max_steps": 500,
        "steps_run": steps_run,
        "best_loss": selected.get("best_loss") if selected else None,
        "budget_trials": selected.get("budget_trials", []) if selected else [],
        "suffix_roundtrip_stable": bool(selected and selected.get("suffix_roundtrip_stable")),
        "validated": result["validated"],
        "selected": selected,
        "candidates": result["candidates"],
        "raw_provenance": raw_provenance or {},
    }
    artifact = {
        "id": str(row.get("sample_id", row.get("id", ""))),
        "protocol": "PUBLIC_SURROGATE_QUERY_SUFFIX_V1",
        "source_hash": row.get("source_hash"),
        "original_query": row.get("original_query"),
        "attack_suffix": selected.get("suffix", "") if selected else "",
        "suffix_token_ids": selected.get("suffix_token_ids", []) if selected else [],
        "suffix_token_count": selected.get("suffix_token_count", 0) if selected else 0,
        "budget": {"compression_rate": evaluation_rate, "max_suffix_tokens": max_suffix_tokens},
        "stage1": row.get("stage1"),
        "stage2": stage2,
        "surrogate": surrogate,
    }
    return artifact


def validate_budget_candidates(
    candidates: Iterable[dict],
    validate: Callable[[str, float], dict],
    *,
    rates: tuple[float, ...] = RATES,
) -> dict:
    """Choose the shortest suffix that removes the target at every surrogate rate."""
    if tuple(rates) != RATES:
        raise ValueError("paper-aligned surrogate rates must be 0.5, 0.6, and 0.7")
    evaluated = []
    winners = []
    for candidate in candidates:
        suffix = candidate.get("suffix")
        token_ids = candidate.get("suffix_token_ids")
        if (not isinstance(suffix, str) or not suffix.strip() or
                not isinstance(token_ids, list) or not token_ids or
                any(not isinstance(token, int) or isinstance(token, bool) for token in token_ids) or
                candidate.get("suffix_token_count") != len(token_ids) or len(token_ids) > 32 or
                candidate.get("suffix_roundtrip_stable") is not True):
            raise ValueError("each Stage-II candidate needs suffix and suffix_token_ids")
        trials = []
        for rate in rates:
            result = validate(suffix, rate)
            trials.append({"compression_rate": rate, **result})
        record = {**candidate, "budget_trials": trials,
                  "stable": all(trial.get("target_removed") is True for trial in trials)}
        evaluated.append(record)
        if record["stable"]:
            winners.append(record)
    winners.sort(key=lambda row: (len(row["suffix_token_ids"]), row.get("best_loss", float("inf"))))
    winner = winners[0] if winners else None
    return {"validated": winner is not None, "selected": winner,
            "candidates": evaluated, "rates": list(rates)}
