# COMA Authorization-SPC Approximate Reproduction

This repository is an approximate, method-faithful reproduction of COMA's
system-prompt corruption (SPC) component, specialized to authorization-critical
rules. The official SPC JSON, the original 1,563-row ordering, GPT artifacts,
and complete semantic-filter records are not public, so this is not an exact
or official-data reproduction.

Pipeline:

```
public system-prompt sources at fixed revisions
  -> authorization-rule extraction
  -> GPT-4o-mini query generation
  -> backend + fixed Judge semantic filter
  -> reconstructed_attack_input (official runner schema)
  -> pinned COMA gradient optimization
  -> real LLMLingua-2 compressor
  -> backend LLM + authorization Judge
```

The pinned COMA checkout is commit
`2c70bd16230918b4c75eaa5384e2a3792dc717b6`. The run keeps the official
attack implementation unchanged. A project-local Transformers compatibility
shim only removes an unsupported `device_map="auto"` argument for
`BertForTokenClassification` and places the unchanged model on the selected
CUDA device.

The current frozen authorization SPC file contains only rows that pass the
predeclared two-state filter (`J(original)=safe/confirmation` and
`J(without rule)=allow`) and have a removable negation span required by the
pinned runner. If fewer rows pass, the count is reported rather than padded.

All reconstruction choices, source revisions, raw artifact hashes, commands,
and deviations are recorded under `manifests/`, `commands/`, `reports/`, and
`datasets/frozen/`. No Mem0/A-Mem experiment is included in this stage.
