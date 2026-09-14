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
pip install -r requirements-spc.txt
```

## Experiment Launchers

Each RQ has a dedicated reproduction script:

| RQ | Script | Description |
|----|--------|-------------|
| RQ1 | `scripts/reproduce_rq1.sh` | Effectiveness launcher for ATS/QA
| RQ2 | `scripts/reproduce_rq2.sh` | Generalization: budget sweep + backend LLMs
| RQ3 | `scripts/reproduce_rq3.sh` | Surrogate mismatch + token retention
| RQ4 | `scripts/reproduce_rq4.sh` | Case studies: VSCode Cline, LangChain+Ollama
| RQ5 | `scripts/reproduce_rq5.sh` | Defense evaluation

### Pinned public assets

Use `reconstruction_tools/fetch_public_assets.py --output <assets-dir> --models` to fetch the four source repositories and fixed Hugging Face snapshots. The command records exact revisions and weight hashes in `public_assets_manifest.json`; run subsequent stages with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

### Reportable SPC approximation: query suffix only

This is the only route in this repository whose output may be reported as an
SPC approximation. The adversary receives a blind input containing the
untrusted query and independent public-surrogate policy text; it never receives
or changes the victim system prompt. At evaluation time, system, context, and
query share one compression budget. No toy SPC dataset is bundled; construct a
fresh, provenance-bound dataset from pinned public sources before running ASR.

Install the limited SPC dependencies, then run the five modules in this order.
No module accepts a trusted victim prompt from the attack path.

| Module | One responsibility | Hard stop before the next module |
|---|---|---|
| `fetch_public_assets.py` | Pin sources and both model snapshots | missing revision or hash |
| `build_four_source_spc_candidates.py` + `transform_four_source_spc.py` | Build a balanced, provenance-bound private/public split | lineage, count, or leakage mismatch |
| `prepare_spc_blind_inputs.py` | Give the attack only public surrogate text and the query | private-field or lineage mismatch |
| `run_spc_stage1.py` + `run_spc_stage2.py` | Select a clean three-budget target, then search and validate a suffix | no real three-budget evidence |
| `run_spc_asr.py` | Bind assets, independent victim, live backend/Judge, and A/B/C/D | any precondition failure; ASR stays `None` |

Provide API keys only through the named environment variables. Take model
revision/hash placeholders from `public_assets_manifest.json`.

```bash
pip install -r requirements-spc.txt

python reconstruction_tools/fetch_public_assets.py \
  --output results/spc/assets --models

python reconstruction_tools/build_four_source_spc_candidates.py \
  --source-root results/spc/assets/sources \
  --output results/spc/candidates.json \
  --per-source 8

python reconstruction_tools/transform_four_source_spc.py \
  --input results/spc/candidates.json \
  --output-dir results/spc/transformed \
  --deterministic --model deterministic-v1

python prepare_spc_blind_inputs.py \
  --private-data results/spc/transformed/private_samples.json \
  --public-pool results/spc/transformed/public_surrogates.json \
  --lineage-manifest results/spc/transformed/lineage_manifest.json \
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
  --data results/spc/transformed/private_samples.json \
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
  --asset-manifest results/spc/assets/public_assets_manifest.json \
  --backend-url https://example.invalid/v1 --backend-model BACKEND_MODEL \
  --judge-url https://example.invalid/v1 --judge-model JUDGE_MODEL \
  --backend-key-env BACKEND_API_KEY --judge-key-env JUDGE_API_KEY
```

Stage II always runs 500 optimization steps and accepts a suffix only when the
real compressor removes the target at 0.5, 0.6, and 0.7. Proxy loss proposes
candidates; it never proves success. Each candidate carries its score, cutoff,
margin, compressed text, and raw compressor output. The final runner accepts
only a complete A/B/C/D batch whose clean baseline is stable (`A=B=C=NO`);
otherwise ASR is `None`. Use `--max-items 2` only for a paid-API smoke test.

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
  run_qa_attack.py          # Entry point: QA task
  run_pref_attack.py        # Entry point: ATS task
  run_surrogate_mismatch.py # ATS/QA surrogate mismatch
  run_pref_attack.sh        # Batch launcher: ATS (all compressors)
  run_qa_attack.sh          # Batch launcher: QA (all compressors)
  run_surrogate_mismatch.sh  # Batch launcher: surrogate grid

```
