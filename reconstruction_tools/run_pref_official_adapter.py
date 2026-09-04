#!/usr/bin/env python3
"""Run the pinned official LLMLingua-2 attack class on reconstructed ATS rows.

The official preference runner's fallback joins unordered words and can skip
every row when that phrase is not contiguous. This adapter keeps the attack
class and parameters unchanged, supplying one deterministic removable span.
"""
import argparse, json, re
from pathlib import Path

def context(x): return "\n\n".join(x[f"demo_{i}"] for i in range(1, 6))

def anchor(x):
    best, target = x[x["best"]], x[x["target"]]
    target_words = {w.casefold() for w in re.findall(r"[A-Za-z0-9]+", target)}
    for w in re.findall(r"[A-Za-z0-9]+", best):
        if len(w) >= 3 and w.casefold() not in target_words:
            pos = best.find(w)
            return w, pos
    return None, -1

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--surrogate-model", required=True); ap.add_argument("--max-entries", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=500); ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--topk", type=int, default=64); ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--test-steps", type=int, default=20); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--edit-radius", type=int, default=-1)
    a = ap.parse_args()
    from comattack.attacks.coma_utils import AttackConfig, NPEncoder
    from comattack.attacks.extractive_context_edit import ContextEditAttackLLMLingua2
    rows = json.loads(Path(a.data).read_text())[:a.max_entries]
    cfg = AttackConfig(model_name=a.surrogate_model, num_steps=a.num_steps,
                       sample_batch_size=a.batch_size, top_k=a.topk,
                       eval_batch_size=a.eval_batch_size, test_steps=a.test_steps,
                       seed=a.seed)
    attacker = ContextEditAttackLLMLingua2(config=cfg)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    path = out / "pref_extractive_adapter_results.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            text = context(row); span, _ = anchor(row); start = text.find(span) if span else -1
            if not span or start < 0:
                rec = {**row, "skip": True, "skip_reason": "no_deterministic_anchor", "attacked_context": text}
            else:
                attacker.best_loss = float("inf"); attacker.best_control_ids = None
                result = attacker.attack(text, span, start, a.edit_radius, a.num_steps)
                rec = {**row, "original_context": text, "removed_span": span,
                       "span_start": start, "target_context": text[:start] + text[start + len(span):],
                       **result, "adapter": "official_ContextEditAttackLLMLingua2"}
            fh.write(json.dumps(rec, cls=NPEncoder, ensure_ascii=False) + "\n"); fh.flush()
            print(f"{i}/{len(rows)} {row.get('reconstruction_id')} {'skip' if rec.get('skip') else 'ok'}", flush=True)
if __name__ == "__main__": main()
