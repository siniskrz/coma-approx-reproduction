# COMA: When Compression Becomes an Attack Surface

Artifact for the paper **"When Compression Becomes an Attack Surface: Black-Box Attacks on Prompt-Compressed LLM Agents"**.

## Overview

COMA is a black-box adversarial framework that exploits prompt compression as an attack surface in LLM agent pipelines. It demonstrates that an adversary can craft inputs that, after compression, selectively remove or corrupt critical information -- causing downstream LLM agents to have misbehavior.

The framework implements a two-stage attack:

- **Stage I (Target Selection):** use a surrogate backend and semantic Judge to
  identify a behavior-changing compression target.
- **Stage II (Preimage Search):** optimize a short suffix that is appended only
  to the untrusted query and makes compression drop that target.

### Tasks
| Task | Abbrev. | Description |
|------|---------|-------------|
| Agent Tool Selection | ATS | Manipulate which tool/product the agent recommends |
| Question Answering | QA | Suppress answer spans so the agent cannot answer correctly |
| System Prompt Corruption | SPC | Append an adversarial query suffix so shared-budget compression drops behavior-critical trusted content |

### Compressors Evaluated
| Type | Compressor | Surrogate Model |
|------|-----------|-----------------|
| Extractive | LLMLingua-1 | Llama-2-7B |
| Extractive | LLMLingua-2 | xlm-roberta-large |
| Extractive | SelectiveContext | Llama-2-7B |
| Abstractive | Qwen3-4B | Qwen3-4B |
| Abstractive | Llama-3.2-3B | Llama-3.2-3B-Instruct |
| Abstractive | Gemma-3-4B | Gemma-3-4B |

## Quick Start

### 1. Install

```bash
conda create -n coma python=3.10 -y && conda activate coma
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
cd artifact/
pip install -e ".[all]"
```

## Reproducing Paper Results

Each RQ has a dedicated reproduction script:

| RQ | Script | Description |
|----|--------|-------------|
| RQ1 | `scripts/reproduce_rq1.sh` | Effectiveness launcher for ATS/QA; legacy SPC is disabled
| RQ2 | `scripts/reproduce_rq2.sh` | Generalization: budget sweep + backend LLMs
| RQ3 | `scripts/reproduce_rq3.sh` | Surrogate mismatch + token retention
| RQ4 | `scripts/reproduce_rq4.sh` | Case studies: VSCode Cline, LangChain+Ollama
| RQ5 | `scripts/reproduce_rq5.sh` | Defense evaluation

### Reportable SPC approximation: query suffix only

This is the only route in this repository whose output may be reported as an
SPC approximation. The adversary receives a blind input containing the
untrusted query and independent public-surrogate policy text; it never receives
or changes the victim system prompt. At evaluation time, system, context, and
query share one compression budget. The included dataset contains 20 low-risk,
fictional permission examples, so these results are a development experiment,
not the paper's reported ASR or a strict reproduction.

Install the limited SPC dependencies, pin every local model snapshot and hash,
and provide API keys only through the named environment variables:

```bash
pip install -r requirements-spc.txt

python prepare_spc_blind_inputs.py \
  --private-data data/toy_spc_permissions.json \
  --public-pool data/public_toy_surrogate_pool.json \
  --output results/spc/blind.jsonl

python run_spc_stage1.py \
  --blind-inputs results/spc/blind.jsonl \
  --output results/spc/stage1.jsonl \
  --compressor-snapshot /path/to/pinned/compressor/snapshot \
  --compressor-revision COMPRESSOR_REVISION \
  --compressor-weight-sha256 COMPRESSOR_SHA256 \
  --compression-rate 0.6 \
  --backend-url https://example.invalid/v1 \
  --backend-model SURROGATE_BACKEND_MODEL \
  --backend-key-env SURROGATE_BACKEND_API_KEY \
  --judge-url https://example.invalid/v1 \
  --judge-model SURROGATE_JUDGE_MODEL \
  --judge-key-env SURROGATE_JUDGE_API_KEY

python run_spc_stage2.py \
  --stage1-results results/spc/stage1.jsonl \
  --output results/spc/attacks.jsonl \
  --surrogate-snapshot /path/to/pinned/surrogate/snapshot \
  --surrogate-model SURROGATE_MODEL \
  --surrogate-revision SURROGATE_REVISION \
  --surrogate-weight-sha256 SURROGATE_SHA256 \
  --initial-suffix "robustness marker test marker" \
  --max-suffix-tokens 32 --sample-batch-size 256 --top-k 64 \
  --eval-batch-size 128 --checkpoint-every 25

python run_spc_asr.py \
  --data data/toy_spc_permissions.json \
  --attack-results results/spc/attacks.jsonl \
  --output results/spc/asr.json \
  --compressor llmlingua2 \
  --compressor-snapshot /path/to/pinned/compressor/snapshot \
  --compressor-revision COMPRESSOR_REVISION \
  --compressor-weight-sha256 COMPRESSOR_SHA256 \
  --compression-rate 0.6 --max-suffix-tokens 32 \
  --transfer-mode black_box \
  --attack-surrogate-model SURROGATE_MODEL \
  --attack-surrogate-snapshot /path/to/pinned/surrogate/snapshot \
  --attack-surrogate-revision SURROGATE_REVISION \
  --attack-surrogate-weight-sha256 SURROGATE_SHA256 \
  --backend-url https://example.invalid/v1 --backend-model BACKEND_MODEL \
  --judge-url https://example.invalid/v1 --judge-model JUDGE_MODEL \
  --backend-key-env BACKEND_API_KEY --judge-key-env JUDGE_API_KEY
```

Stage II is fixed to 500 optimization steps, a suffix of at most 32
tokens, and validation at compression rates 0.5, 0.6, and 0.7. `run_spc_asr.py`
rejects incomplete Stage-I evidence, unvalidated Stage-II candidates, modified
trusted fields, provenance mismatches, and suffixes over the limit. It stores
raw backend/Judge evidence, maps non-exact labels to `UNKNOWN`, and reports the
paired A/B/C/D design plus the baseline-stable subset (`A=B=C=NO`). Use
`--max-items 2` only as a paid-API smoke test.

The retired `run_guardrail_attack.py`, `scripts/run_guardrail_attack.sh`, the
SPC branch of `scripts/reproduce_rq1.sh`, and SPC in
`run_surrogate_mismatch.py` fail closed because those paths edited trusted
system-prompt text. The engineering boundary generator
`run_spc_query_suffix.py` does not run Stage I/II and its artifacts are not
eligible for ASR reporting.

Run the offline protocol checks with:

```bash
python -m unittest discover -s tests -v
```

## Output Format

Reportable SPC attack results are saved as JSONL files, one entry per line.
Trusted victim system/context text is deliberately absent:

```json
{
    "id": "toy-perm-01",
    "protocol": "PUBLIC_SURROGATE_QUERY_SUFFIX_V1",
    "source_hash": "sha256-of-private-source",
    "original_query": "untrusted query text...",
    "attack_suffix": "validated suffix...",
    "suffix_token_count": 12,
    "stage1": {"status": "COMPLETE"},
    "stage2": {"status": "COMPLETE", "max_steps": 500, "validated": true}
}
```

## File Structure

```
artifact/
  README.md                 
  config.py                 # Configuration (env-var based, no hardcoded paths)
  requirements.txt          # Python dependencies
  .gitignore

  comattack/                # Core Python package
    __init__.py
    attacks/                # Stage II: preimage search (extractive + abstractive)
    compressors/            # Compressor wrappers
    defense/                # Defense baselines
    evaluation/             # End-to-end evaluation, metrics, compliance
    llm/                    # LLM provider abstraction (vLLM, OpenAI, Ollama)
    targets/                # Stage I: Target generation per task
    data/                   # Package data (prompts, templates)

  prepare_spc_blind_inputs.py # Blind private victim rules from attack generation
  run_spc_stage1.py         # Behavior-based target selection
  run_spc_stage2.py         # Query-suffix optimization and multi-budget validation
  run_spc_asr.py            # Paired shared-budget evaluation
  run_guardrail_attack.py   # Disabled legacy direct-system SPC path
  run_qa_attack.py          # Entry point: QA task
  run_pref_attack.py        # Entry point: ATS task
  run_surrogate_mismatch.py # Surrogate mismatch (legacy SPC path disabled)
  run_pref_attack.sh        # Batch launcher: ATS (all compressors)
  run_qa_attack.sh          # Batch launcher: QA (all compressors)
  run_surrogate_mismatch.sh  # Batch launcher: surrogate grid

```

