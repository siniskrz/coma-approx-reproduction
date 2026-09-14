set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=false
DETECTOR="ppl"
LLM_MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --detector) DETECTOR="$2"; shift 2 ;;
        --llm-model) LLM_MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

case "$DETECTOR" in
    ppl|llm) ;;
    bleu)
        echo "ERROR: the repository has no BLEU detector implementation; refusing to fabricate BLEU scores." >&2
        exit 2
        ;;
    *) echo "ERROR: unknown detector: $DETECTOR" >&2; exit 1 ;;
esac

if [[ "$DETECTOR" == "llm" && -z "$LLM_MODEL" ]]; then
    echo "ERROR: --llm-model is required for the repository's llm_inference detector." >&2
    exit 2
fi

run_cmd() {
    if $DRY_RUN; then echo "$@"; else echo ">>> $@"; "$@"; fi
}

echo "===== RQ5: ${DETECTOR} defense evaluation ====="
echo "NOTE: This requires attack results from RQ1."

for TASK_CFG in \
    "qa:results/rq1/qa/extractive_llmlingua1/qa_extractive_results.jsonl" \
    "pref:results/rq1/pref/extractive_llmlingua1/pref_extractive_results.jsonl"; do

    IFS=: read -r TASK RESULTS_FILE <<< "$TASK_CFG"
    if [[ ! -f "$RESULTS_FILE" ]] && ! $DRY_RUN; then
        echo "ERROR: missing RQ1 results: $RESULTS_FILE" >&2
        exit 2
    fi

    if [[ "$DETECTOR" == "ppl" ]]; then
        run_cmd python -c "
from comattack.defense.bleu_detector import AttackDetector
import json, os

detector = AttackDetector()
results = []
with open('${RESULTS_FILE}', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        attacked = entry.get('attacked_context') or entry.get('attacked_prompt', '')
        original = entry.get('original_context') or entry.get('context') or entry.get('system_prompt', '')
        if attacked and original:
            original_score, attacked_score = detector.detect([original, attacked])
            results.append({'original': original_score, 'attacked': attacked_score})

out = 'results/rq5/ppl_detection/${TASK}'
os.makedirs(out, exist_ok=True)
with open(out + '/scores.json', 'w', encoding='utf-8') as f:
    json.dump({'task': '${TASK}', 'n': len(results), 'scores': results}, f, indent=2)
print(f'  PPL/anomaly detection on ${TASK}: {len(results)} entries scored')
"
    else
        run_cmd python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from comattack.defense.llm_detector import llm_inference
import json, os

model = AutoModelForCausalLM.from_pretrained('${LLM_MODEL}', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('${LLM_MODEL}')
results = []
with open('${RESULTS_FILE}', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        attacked = entry.get('attacked_context') or entry.get('attacked_prompt', '')
        original = entry.get('original_context') or entry.get('context') or entry.get('system_prompt', '')
        if attacked and original:
            results.append({
                'original': llm_inference(model=model, tokenizer=tokenizer, text=original),
                'attacked': llm_inference(model=model, tokenizer=tokenizer, text=attacked),
            })

out = 'results/rq5/llm_detection/${TASK}'
os.makedirs(out, exist_ok=True)
with open(out + '/scores.json', 'w', encoding='utf-8') as f:
    json.dump({'task': '${TASK}', 'n': len(results), 'detections': results}, f, indent=2)
print(f'  LLM detection on ${TASK}: {len(results)} entries scored')
"
    fi
done

echo "===== RQ5 reproduction complete ====="
