# COMA: When Compression Becomes an Attack Surface

Artifact for the paper **"When Compression Becomes an Attack Surface: Black-Box Attacks on Prompt-Compressed LLM Agents"**.

## Overview

COMA is a black-box adversarial framework that exploits prompt compression as an attack surface in LLM agent pipelines. It demonstrates that an adversary can craft inputs that, after compression, selectively remove or corrupt critical information -- causing downstream LLM agents to have misbehavior.

The framework implements a two-stage attack:
- **Stage I (Target Selection):** Identify which information to suppress (answer spans, guardrail negations, preference-critical keywords)
- **Stage II (Preimage Search):** Use COMA-based optimization to find adversarial inputs that, after compression, match the target

### Tasks
| Task | Abbrev. | Description |
|------|---------|-------------|
| Agent Tool Selection | ATS | Manipulate which tool/product the agent recommends |
| Question Answering | QA | Suppress answer spans so the agent cannot answer correctly |
| System Prompt Corruption | SPC | Remove guardrail negations (e.g., "do not" -> "") to disable safety rules |

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
| RQ1 | `scripts/reproduce_rq1.sh` | Effectiveness: 3 tasks x 6 compressors
| RQ2 | `scripts/reproduce_rq2.sh` | Generalization: budget sweep + backend LLMs
| RQ3 | `scripts/reproduce_rq3.sh` | Surrogate mismatch + token retention
| RQ4 | `scripts/reproduce_rq4.sh` | Case studies: VSCode Cline, LangChain+Ollama
| RQ5 | `scripts/reproduce_rq5.sh` | Defense evaluation

### Auditable SPC toy run

`run_spc_asr.py` evaluates low-risk fictional permission rules with a paired
four-condition design: clean/attacked system prompts, each with and without
LLMLingua-2 compression. It preserves the system/user role boundary, saves raw
backend and judge evidence, treats non-exact judge answers as `UNKNOWN`, and
reports both all-pair rates and the baseline-stable subset (`A=B=C=0`). This is
explicitly a public approximation, not a claim of reproducing the paper's ASR.

The included `data/toy_spc_permissions.json` contains 20 harmless synthetic
examples. API keys are read only from the environment. A pinned local model
snapshot is required and its directory name must equal the supplied revision.
Install `requirements-spc.txt` for this limited route; it avoids the full
repository's vLLM dependency. The ASR runner itself uses Python's HTTP client,
while `openai` remains pinned for the repository's other API entry points.

```bash
pip install -r requirements-spc.txt

python run_guardrail_attack.py \
  --data data/toy_spc_permissions.json \
  --output results/toy-spc-attack \
  --compressor llmlingua2 \
  --surrogate-model /path/to/pinned/snapshot \
  --surrogate-revision SNAPSHOT_DIRECTORY_NAME \
  --surrogate-weight-sha256 EXPECTED_MODEL_SAFETENSORS_SHA256 \
  --num-steps 10 --batch-size 32 --topk 16 \
  --eval-batch-size 8 --test-steps 2 --edit-radius 4 --seed 42

python run_spc_asr.py \
  --data data/toy_spc_permissions.json \
  --attack-results results/toy-spc-attack/guardrail_extractive_results.jsonl \
  --output results/toy-spc-asr/result.json \
  --compressor-snapshot /path/to/pinned/snapshot \
  --compressor-revision SNAPSHOT_DIRECTORY_NAME \
  --compressor-weight-sha256 EXPECTED_MODEL_SAFETENSORS_SHA256 \
  --compression-rate 0.6 \
  --backend-url https://example.invalid/v1 --backend-model BACKEND_MODEL \
  --judge-url https://example.invalid/v1 --judge-model JUDGE_MODEL \
  --backend-key-env BACKEND_API_KEY --judge-key-env JUDGE_API_KEY
```

Use `--max-items 2` for a paid-API smoke test. Run the offline protocol tests
with `python -m unittest tests.test_spc_asr -v`.

## Output Format

Attack results are saved as JSONL files, one entry per line:

```json
{
    "context": "original input text...",
    "attacked_context": "adversarial input text...",
    "best_loss": 0.023,
    "converged": true,
    "steps": 142
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

  run_guardrail_attack.py   # Entry point: SPC task
  run_qa_attack.py          # Entry point: QA task
  run_pref_attack.py        # Entry point: ATS task
  run_surrogate_mismatch.py # Entry point: surrogate mismatch
  run_pref_attack.sh        # Batch launcher: ATS (all compressors)
  run_qa_attack.sh          # Batch launcher: QA (all compressors)
  run_surrogate_mismatch.sh  # Batch launcher: surrogate grid

```

