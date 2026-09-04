# SPC mechanism-validation implementation note

## Existing path verified before changes

Track A currently reads the pinned reconstructed SPC rows, runs the released
`run_guardrail_attack.py`, and stores `attacked_context`. The pinned runner
passes the system prompt to the extractive context editor and does not consume
`adversarial_query` during optimization. The real victim pass then compresses
`system_prompt`, `target_prompt`, and `attacked_context` with the cached
LLMLingua-2 XLM model. Backend requests use the compressed text as a `system`
message and the generated query as a `user` message; Judge requests use the
actual recorded backend response.

## New implementation

- Add a separate mechanism-validation experiment path under
  `results/spc/mechanism_validation/`.
- Add a compact evaluator for NoComp+COMA controls, compressed controls,
  critical-span retention/CTRR, joint mechanism categories, and Wilson
  intervals.
- Add explicit per-sample critical-span annotations without changing existing
  frozen rows.
- Add a separate candidate-expansion script using the existing deterministic
  source extraction and two-state backend/Judge filter; it writes new
  intermediate/frozen paths only.
- Add a Track B scaffold that freezes the system prompt and records an
  untrusted query suffix, while leaving unresolved algorithmic details
  explicit.
- Make backend and Judge model names configurable in new request builders.
- Add a new mechanism-validation report; the existing reproduction report,
  attack outputs, compressed outputs, and prior diagnostic outputs remain
  untouched.
