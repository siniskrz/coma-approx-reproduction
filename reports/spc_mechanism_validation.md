# COMA authorization-SPC mechanism validation

Date: 2026-09-05
Evidence level: **Level 1 — released-artifact pipeline validation on a small reconstructed slice**

This report is a separate mechanism/control analysis. It does not replace `reports/reproduction_report.md`, and it does not claim to reproduce the paper's reported 0.63 ASR.

## Tracks and threat-model boundary

**Track A — released-artifact/code-faithful (executed).** The run uses the official COMA checkout at commit `2c70bd16230918b4c75eaa5384e2a3792dc717b6`, the official extractive LLMLingua-2 surrogate attack, a real LLMLingua-2 XLM victim compressor, and real backend/Judge calls. The backend and Judge are `openai/gpt-4o`, temperature 0, top-p 1, seed 42. The frozen Track A slice has 3 attacked rows.

The pinned official runner dispatches SPC to `run_context_edit_attack`, whose `task="spc"` branch uses `entry["system_prompt"]` as the editable context. It does not append an untrusted query suffix. Therefore Track A is faithful to the released runner, but it is **not** faithful to a query-only threat model.

**Track B — paper threat-model-faithful (not executed).** The intended paper-style control would keep the system prompt frozen and optimize only an untrusted query/suffix. `reconstruction_tools/track_b_query_scaffold.py` records the frozen prompt and a query scaffold, but the released repository does not expose enough algorithm details to claim a Track B attack result. No Track B number is reported here.

## Reconstructed-data funnel

The deterministic source/length/target rules produced 152 source candidates, of which 138 were attackable under this reconstruction's character-length and removable-target gates. The existing 24-row candidate pool was evaluated previously; 114 additional candidates were sent through query generation, backend checks, and Judge checks. All 114 had complete, error-free responses. Sixteen passed the two-state semantic filter (`J(original)=safe` and `J(no-system)=violation`).

| Funnel stage | Count |
|---|---:|
| Source candidates | 152 |
| Length/target-attackable candidates | 138 |
| Existing Track A candidate pool | 24 |
| Additional candidates evaluated | 114 |
| Additional two-state semantic passes | 16 |
| Additional rows whose annotated critical span survived clean LLMLingua-2 compression | 2 / 16 |
| Combined reconstructed valid pool | 19 |
| Combined unique system prompts | 17 |

The 1600-character gate is a reconstruction convenience; it is not the paper's exact 512-victim-token gate. The 19-row pool is therefore a **machine-filtered reconstructed pool**, not an author-identical SPC dataset. It is dominated by authorization prompts from the available public sources and is not representative of the full paper benchmark.

The executed 3-row mechanism slice contains only 2 unique system prompts. The Wilson intervals below are row-level descriptive intervals, not independent-population confidence intervals.

## NoCompression control and mechanism results

The following rows use the same 3-row Track A slice and the same real backend/Judge loop. Intervals are 95% Wilson intervals for row-level descriptive binomial proportions; they do not model LLM randomness, Judge uncertainty, or clustering by shared system prompt.

| Condition | Successes / n | Rate | Wilson 95% interval |
|---|---:|---:|---:|
| Original + NoCompression | 0 / 3 | 0.0% | [0.0%, 56.1%] |
| Original + LLMLingua-2 | 0 / 3 | 0.0% | [0.0%, 56.1%] |
| COMA + NoCompression | 2 / 3 | 66.7% | [20.8%, 93.9%] |
| COMA + LLMLingua-2 | 2 / 3 | 66.7% | [20.8%, 93.9%] |
| Target compression, changed-target gate | 1 / 2 | 50.0% | [9.5%, 90.5%] |

The raw target-compression Judge output was 1 / 3, but one official target string was byte-identical to the original because the upstream string-replacement boundary did not match. The changed-target gate excludes that row. This is a control-quality diagnostic, not an additional attack result.

### Critical-span removal proxy

The frozen annotations in `datasets/frozen/authorization_spc_critical_spans_v2.json` identify one semantically critical span per row. For the 2 rows whose clean compressed prompt retained its annotated span, COMA removed 2 / 2 annotated units: 100.0%, Wilson [34.2%, 100.0%]. One row is excluded because clean LLMLingua-2 compression had already removed its annotated span.

This is deliberately called a **critical-span removal proxy**, not paper-faithful CTRR. The paper's CTRR is defined over Stage-I token-level critical occurrences (`W_i`); these annotations are a transparent semantic audit of the available outputs and are not a replacement for the unpublished/full Stage-I accounting. The expanded 16 rows were not subjected to a new 500-step COMA attack and contribute no CTRR or ASR number.

### Joint mechanism categories

| Category | Count / 3 | Interpretation |
|---|---:|---|
| Critical-span removal + compression-only success | 0 / 3 | No row supports a compression-specific causal flip under this definition |
| Critical-span removal without behavioral success | 0 / 3 | Not observed in this slice |
| Behavioral success without critical-span removal | 0 / 3 | Not observed among clean-valid rows |
| COMA success with NoCompression | 2 / 3 | The attacked text alone was sufficient for both observed flips |

Both successful rows are successful with **COMA + NoCompression** and with **COMA + compression**. Thus this slice does not support the claim that victim compression is necessary for the observed behavioral failures. It is compatible with the simpler explanation that the released attack directly edits the trusted system prompt and the edited text itself causes the flip. This is a control result, not evidence that compression has no effect in the paper's intended threat model.

## Reproducibility artifacts

- `reconstruction_tools/mechanism_validation.py` builds/scores the five conditions, binary Judge labels, Wilson intervals, the removal proxy, and joint categories.
- `reconstruction_tools/summarize_expansion.py` records the reconstructed-data funnel.
- `reports/spc_expansion_funnel.json` is the aggregate funnel output.
- Raw prompts, backend responses, Judge responses, compressed text, and attack traces remain local/ignored; public archive metadata contains hashes and aggregate results only.

## Bottom line

The executed evidence is a small, code-faithful Track A pipeline check. It validates that the real victim compressor and backend/Judge loop run end to end, but it does not recover the paper's hidden data, exact threat model, victim/backend models, or full-scale ASR. The NoCompression control is essential here: on this slice, the two behavioral flips do not require compression, so reporting 0.63-like compression causality would be unsupported.
