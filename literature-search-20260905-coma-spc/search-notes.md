# Search Notes

## Safe Queries Used

- `prompt compression security adversarial information loss LLMLingua attack GitHub`
- `system prompt leakage benchmark guardrail robustness GitHub dataset`
- `prompt injection benchmark system prompt security agents paper official repository`
- `LLMLingua prompt compression evaluation GitHub`
- `AgentDojo benchmark official GitHub prompt injection paper`
- `SystemPromptIndex GitHub span annotation system prompts`
- `InjecGuard benchmark GitHub prompt injection official`
- `SecurityLingua official paper OpenReview prompt compression security`

## Sources Checked

- COMA paper HTML and abstract: https://arxiv.org/html/2510.22963v4 and https://arxiv.org/abs/2510.22963
- COMA official artifact: https://github.com/zsLiu2003/Comattack
- Microsoft LLMLingua repository: https://github.com/microsoft/LLMLingua
- SystemCheck dataset card: https://huggingface.co/datasets/normster/SystemCheck
- SystemPromptIndex: https://github.com/SystemPromptIndex/SystemPromptIndex
- AgentDojo: https://github.com/ethz-spylab/agentdojo
- Agent Security Bench: https://proceedings.iclr.cc/paper_files/paper/2025/hash/5750f91d8fb9d5c02bd8ad2c3b44456b-Abstract-Conference.html
- Salesforce PromptLeakage: https://github.com/salesforce/prompt-leakage
- InjecGuard: https://github.com/InjecGuard/InjecGuard
- SecurityLingua: https://openreview.net/forum?id=tybbSo6wba

## Excluded Sources

- MDPI search result was excluded by the source policy.
- Reddit and generic blog results were used only for discovery or omitted; they are not evidence for the plan.

## Unknowns

- The authors' exact 1,563-row SPC IDs and complete intermediate filtering artifacts remain unavailable from the public artifact.
- The released SPC runner's direct `system_prompt` edit is code-faithful but differs from the paper's query-only perturbation threat model.
- A public source can establish the protocol and reported values, but cannot establish that the reconstructed 19-row pool matches the authors' hidden distribution.

## Handoff Notes

- For experiment design: prioritize a 152-row data/distribution audit and clean-compression diff before new COMA attacks.
- For paper review: treat NoCompression equality as a failed causal-control replication, not as evidence against COMA itself.
- For writing: separate `released-artifact faithful`, `paper threat-model faithful`, and `source-distribution reconstructed` estimands.
- For direction scouting: the strongest open gap is compression-specific causal measurement with provenance-preserving authorization spans.
