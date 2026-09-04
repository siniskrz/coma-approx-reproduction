# Literature Search: COMA SPC data and mechanism diagnosis

Date: 2026-09-05
Search purpose: understand COMA's central mechanism, compare reproducible prompt/security benchmarks, and design the next SPC diagnosis phase.
Target venue/family: LLM security, prompt compression, agent benchmarks
Source-quality policy: applied; primary paper/repository/dataset pages preferred; MDPI excluded.

## Summary

- Closest-work clusters: COMA/LLMLingua compression boundary; SystemCheck/RealGuardrails-style system-prompt corpora; AgentDojo/ASB end-to-end agent security; span-level prompt auditing; prompt-injection over-defense controls.
- Opportunity map: the current gap is mechanism and dataset provenance, not another generic attack run.
- Strongest baselines: COMA's own NoCompression and No-Attack controls; paired clean/attacked evaluation; cluster-disjoint data splits from guardrail benchmarks.
- Recommended next action: audit all 152 candidates and run clean-compression diffs before selecting any new COMA attack rows.

## Paper and project table

| # | Title/project | Year | Venue/source | Link | Type | Insight | Completeness | Numeric evidence | Overall | Relevance |
|---|---|---:|---|---|---|---:|---:|---:|---|---|
| 1 | When Compression Becomes an Attack Surface (COMA) | 2026 | arXiv v4 / official artifact | https://arxiv.org/abs/2510.22963 | pure method | 5 | 4 | 5 | A | Defines AIL, two-stage target/dropout + preimage search, and requires NoCompression controls. |
| 2 | Comattack | 2026 | official GitHub artifact | https://github.com/zsLiu2003/Comattack | system/tool | 5 | 4 | 4 | A | Reproduction scripts expose the released runner, SPC preprocessing, and retention entry points. |
| 3 | LLMLingua / LLMLingua-2 | 2023–2024 | Microsoft official repository / ACL | https://github.com/microsoft/LLMLingua | system/tool | 4 | 5 | 5 | A | Defines the real victim compressor behavior and token-budget constraints used in the audit. |
| 4 | SystemCheck: A Closer Look at System Prompt Reliability | 2024 | dataset/project | https://huggingface.co/datasets/normster/SystemCheck | method + benchmark | 4 | 4 | 4 | A | Uses deduplication, language/tool filtering, conflicting/aligned queries, handwritten tests, and distractors. |
| 5 | SystemPromptIndex / AISPA | 2026 | public audited corpus/project | https://github.com/SystemPromptIndex/SystemPromptIndex | pure benchmark | 4 | 4 | 3 | B | Stores exact span offsets, dimensions, scores, and notes; directly informs critical-span provenance. |
| 6 | AgentDojo | 2024 | NeurIPS Datasets and Benchmarks | https://github.com/ethz-spylab/agentdojo | pure benchmark | 5 | 5 | 5 | A | Separates user tasks, injections, environment state, utility, and security outcomes in an end-to-end loop. |
| 7 | Agent Security Bench (ASB) | 2025 | ICLR | https://github.com/agiresearch/ASB | pure benchmark | 4 | 5 | 5 | A | Broad agent attack/defense coverage and multiple security/utility metrics; useful as a protocol reference. |
| 8 | Prompt Leakage effect and defense strategies | 2024 | EMNLP artifact | https://github.com/salesforce/prompt-leakage | method + benchmark | 3 | 4 | 4 | B | Useful provenance and real-output evaluation patterns, but studies leakage rather than compression-induced loss. |
| 9 | InjecGuard / NotInject | 2024 | arXiv + official GitHub | https://github.com/InjecGuard/InjecGuard | method + benchmark | 4 | 4 | 4 | B | Controls trigger-word shortcut and over-defense with benign/malicious paired examples. |
| 10 | SecurityLingua | 2025 | COLM / OpenReview | https://openreview.net/forum?id=tybbSo6wba | pure method | 3 | 4 | 4 | B | Relevant compression-security counterpoint: compression can be designed as a defense, not only an attack surface. |

## Closest-work clusters

### Cluster 1: Compression as a security boundary

- Representative papers: COMA; LLMLingua/LLMLingua-2; SecurityLingua.
- What this cluster already solves: formalizes shared-budget compression, attacker-side surrogate optimization, and downstream behavioral validation.
- Remaining gap: a public, byte-reconstructible SPC corpus and a clean separation between compression-specific failure and direct trusted-prompt editing.
- Possible rescue route: report a compression-specific differential estimand only for rows where NoCompression remains safe and clean compression is valid; keep released-runner results as a separate artifact-faithful track.
- How it affects this project: do not start with more 500-step attacks; first establish whether the reconstructed rows contain compression-sensitive cases.

### Cluster 2: Real system-prompt corpora and span audits

- Representative projects: SystemCheck, SystemPromptIndex, Comattack's RealGuardrails inputs.
- What this cluster already solves: broad prompt collection, deduplication/language filtering, aligned/conflicting queries, and exact span-level audit records.
- Remaining gap: the COMA paper's final 1,563 SPC IDs and full intermediate filtering provenance are not public in a directly verifiable form.
- Possible rescue route: freeze a broader source corpus, cluster near duplicates, annotate authorization spans with exact offsets and pre-registered semantic roles, then treat behavior checks as eligibility controls rather than the data definition.
- How it affects this project: the 19-row pool is a reconstructed diagnostic slice, not a distributional recovery of the paper dataset.

### Cluster 3: Agent security evaluation protocols

- Representative projects: AgentDojo, ASB, PromptLeakage.
- What this cluster already solves: paired clean/attack runs, explicit environment/task state, utility versus security metrics, and end-to-end outcome recording.
- Remaining gap: compression-stage retention and trusted/untrusted channel competition are usually not isolated as a causal factor.
- Possible rescue route: use a fixed task/query/environment and compare NoCompression, clean compression, COMA+NoCompression, and COMA+Compression with paired transitions.
- How it affects this project: ASR alone is insufficient; preserve raw responses and classify safe→safe, safe→violation, violation→safe, violation→violation, and non-evaluable.

## Opportunity map

| Cluster | Status | Open gap | Possible direction | Evidence needed | Risk |
|---|---|---|---|---|---|
| Compression boundary | mechanism gap | Is the observed failure caused by compressor steering or direct prompt corruption? | compression-specific differential ASR | paired NoCompression/Compression and token-level target accounting | high threat-model mismatch risk |
| SPC corpus | benchmark/provenance gap | Are reconstructed prompts structurally comparable to paper SPC? | source-balanced corpus with cluster and complexity metadata | all-candidate funnel, source/cluster counts, exact revisions | cannot claim recovery of 1,563 hidden IDs |
| Critical-span audit | measurement gap | Did compression remove authorization semantics or merely background? | offset-based span diff and cue retention | pre-registered spans, token/character diff, cue categories | proxy is not official CTRR |
| Agent evaluation | protocol gap | Does a change alter security behavior under the same task state? | paired transition report | identical query/backend/Judge and raw outputs | Judge uncertainty and model dependence |

## Benchmark and dataset candidates

| Name | Link | Task | Metrics | Fit | Risks |
|---|---|---|---|---|---|
| SystemCheck | https://huggingface.co/datasets/normster/SystemCheck | system-prompt reliability | aligned/conflicting behavior, distractors | strongest corpus/filter reference | not compression-specific |
| SystemPromptIndex | https://github.com/SystemPromptIndex/SystemPromptIndex | span-level system-prompt audit | offsets, dimensions, protective/problematic spans | strongest annotation schema reference | 2026 public corpus; not COMA task labels |
| AgentDojo | https://github.com/ethz-spylab/agentdojo | agent prompt injection | utility/security outcomes | strongest dynamic end-to-end protocol | different threat and task domains |
| ASB | https://github.com/agiresearch/ASB | multi-scenario agent security | attack/defense and utility metrics | broad external sanity check | larger framework than current question |
| Comattack | https://github.com/zsLiu2003/Comattack | COMA ATS/QA/SPC | ASR, CTRR, controls | authoritative implementation | released SPC data provenance remains incomplete |

## Positioning cautions

- The paper's SPC filter is behavioral: keep `J(original)=0` and `J(-sys)=1`; it is not a “compression semantic consistency” filter.
- The paper's threat model only perturbs untrusted query/context while trusted system input remains hidden and frozen; the pinned released SPC runner edits `system_prompt` directly, so those tracks must not be conflated.
- Paper Table 1 reports LLMLingua-2 SPC COMA ASR 0.63 and near-zero NoCompression control; the current 3-row result does not reproduce that control relation.
- Paper CTRR is defined from Stage-I token occurrences `W_i`, not an after-the-fact phrase-presence proxy.
