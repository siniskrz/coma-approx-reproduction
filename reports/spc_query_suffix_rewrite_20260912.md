# SPC query-suffix rewrite and verification (2026-09-12)

## Scope and conclusion

This revision replaces the repository's reportable SPC path.  The retired path
edited trusted system-prompt text and therefore did not satisfy the paper's
attacker boundary.  The new approximation exposes only the untrusted query and
independent public-surrogate text to attack generation.  The victim evaluator
serializes system, context, and query once and compresses that single text under
one budget.

This is an auditable toy/development approximation, not a strict reproduction
of the paper's ASR.  The previously reported 20-item API pilot used the retired
direct-system route and must not be relabeled as a result of this revision.

## Implemented protocol

1. `prepare_spc_blind_inputs.py` replaces private victim policy text with a
   SHA-256 provenance binding and independent public toy surrogate rules.
2. `run_spc_stage1.py` compresses the complete public surrogate prompt and
   accepts a target only after recorded surrogate-backend/Judge evidence shows
   an exact `NO -> YES` behavior flip.
3. `run_spc_stage2.py` performs 500 iterative LLMLingua-2 optimization steps,
   keeps suffixes at or below 32 verified tokenizer tokens, and accepts a
   candidate only when the selected public target is absent at rates 0.5, 0.6,
   and 0.7.
4. `run_spc_asr.py` rejects legacy trusted-text edits, mismatched provenance,
   forged token counts, reserved prompt markers, incomplete Stage evidence, and
   matched surrogate/victim weights in black-box mode.  It reports paired
   A/B/C/D results, `UNKNOWN`, interaction effects, and the stable `A=B=C=NO`
   denominator.

The old `run_guardrail_attack.py` and SPC branches in the RQ1/RQ3 launchers now
fail closed.  CTRR is reported only when occurrence-level source positions are
available; otherwise it is explicitly unavailable.

## Verification performed

- 45 offline unit/integration tests passed.
- The integration test traverses blind preparation, Stage I, 500-step Stage II,
  and four-condition evaluation with deterministic fake model clients.
- All four new command-line entry points start and expose their help text.
- Package dependency checks and Python compilation passed.
- A local NVIDIA GPU loaded the pinned real
  `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` snapshot (revision
  `ebaba9b0e874dadd3003ffcff828e4397e568089`, weight SHA-256
  `a33a153b2493bff6be06af6921e69de9c0d0bb6ff06fe5bbb68670ba8d980ae2`).

The real-model Stage-II smoke test used one fictional permission item, 500
steps, candidate batch 8, top-k 8, and checkpoints every 100 steps.  It
completed without a crash and produced two round-trip-stable checkpoint
candidates, but neither removed the selected target at all three budgets:

- attempted: 1
- completed 500 steps: 1
- validated suffixes: 0
- ASR: not measured (the input contained synthetic Stage-I evidence solely to
  exercise Stage-II, so the result is deliberately ineligible for ASR)

The first real attempt exposed a scalar/per-candidate loss aggregation bug that
the offline tests had missed.  The implementation was corrected to calculate a
loss vector per candidate; the same 500-step smoke test then completed.

## Remaining evidence gap

A corrected ASR batch still requires real Stage-I backend/Judge calls and a
separate black-box victim compressor/backend.  No API key is stored in the
repository or result files.  Until that run is made, the only honest conclusion
is that the corrected code path and real Stage-II model execution work; there
is no new paper-comparable ASR estimate yet.
