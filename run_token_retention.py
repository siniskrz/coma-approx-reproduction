"""Auditable critical-token removal measurement.

A compressor must return one ``source_map`` entry per source-word occurrence:
``{"start": int, "end": int, "kept": bool}``. Without it CTRR is unavailable.
"""

import argparse
import csv
import glob
import json
import logging
import os
import re
import sys
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("rq3_token_retention")
ALL_TASKS = ["prom", "deg", "qa", "spc"]
CTRR_AVAILABLE = "CTRR_AVAILABLE"
CTRR_UNAVAILABLE = "CTRR_UNAVAILABLE"


def _word_occurrences(text):
    return [{"word": m.group(0).lower(), "start": m.start(), "end": m.end()}
            for m in re.finditer(r"\b\w+\b", text)]


def _explicit_ranges(entry, field):
    value = entry.get(field)
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    ranges = []
    for item in value:
        if not isinstance(item, dict):
            return None
        start, end = item.get("start"), item.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            return None
        ranges.append((start, end))
    return ranges


def critical_ranges(task, entry, text, *, attacked=False):
    """Find critical character ranges, returning None when provenance is ambiguous."""
    explicit = _explicit_ranges(entry, "critical_spans_attack" if attacked else "critical_spans_benign")
    if explicit is None:
        explicit = _explicit_ranges(entry, "critical_spans")
    if explicit is not None:
        return explicit
    if task == "qa":
        removed, start = entry.get("removed_span", ""), entry.get("span_start", -1)
        if not removed:
            answers = entry.get("answers", {})
            texts, starts = answers.get("text", []), answers.get("answer_start", [])
            if texts and starts:
                removed, start = texts[0], starts[0]
        if isinstance(start, int) and start >= 0 and text[start:start + len(removed)] == removed:
            return [(start, start + len(removed))]
        return None
    if task == "spc":
        phrases = entry.get("removed_phrases", [])
        phrases = [phrases] if isinstance(phrases, str) else phrases
        ranges = {(m.start(), m.end()) for phrase in phrases if phrase
                  for m in re.finditer(re.escape(phrase), text)}
        return sorted(ranges) or None
    # A deleted word value cannot identify which repeated occurrence was critical.
    return None


def measure_ctrr(original_text, source_map, ranges):
    """Compute per-sample CTRR = 1 - retained / total source occurrences."""
    if not isinstance(source_map, list):
        return {"status": CTRR_UNAVAILABLE, "reason": "missing_source_map"}
    if not ranges:
        return {"status": CTRR_UNAVAILABLE, "reason": "missing_critical_source_ranges"}
    occurrences = _word_occurrences(original_text)
    if len(source_map) != len(occurrences):
        return {"status": CTRR_UNAVAILABLE, "reason": "incomplete_source_map"}
    for expected, supplied in zip(occurrences, source_map):
        if not isinstance(supplied, dict):
            return {"status": CTRR_UNAVAILABLE, "reason": "invalid_source_map_entry"}
        if (supplied.get("start"), supplied.get("end")) != (expected["start"], expected["end"]):
            return {"status": CTRR_UNAVAILABLE, "reason": "source_map_position_mismatch"}
        if not isinstance(supplied.get("kept"), bool):
            return {"status": CTRR_UNAVAILABLE, "reason": "source_map_kept_not_boolean"}
    critical = [item for item in source_map
                if any(item["start"] < end and item["end"] > start for start, end in ranges)]
    if not critical:
        return {"status": CTRR_UNAVAILABLE, "reason": "no_critical_word_occurrences"}
    retained = sum(item["kept"] for item in critical)
    return {"status": CTRR_AVAILABLE, "ctrr": 1 - retained / len(critical),
            "critical_occurrences": len(critical),
            "retained_critical_occurrences": retained}


def presence_proxy_retained_fraction(original_text, compressed_text, ranges):
    """Unaudited bag-of-words presence diagnostic; deliberately not CTRR."""
    words = [item["word"] for item in _word_occurrences(original_text)
             if any(item["start"] < end and item["end"] > start for start, end in ranges)]
    if not words:
        return None
    available = Counter(item["word"] for item in _word_occurrences(compressed_text))
    return sum(min(count, available[word]) for word, count in Counter(words).items()) / len(words)


def make_compressor(name):
    from comattack.evaluation.e2e_eval import make_compressor as factory
    return factory(name)


def _compressed_text(result):
    return result.get("compressed_prompt", result.get("compressed_text", ""))


def run_retention_analysis(args):
    with open(args.attack_results, encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    if args.task == "spc":
        raise ValueError(
            "SPC CTRR rejects legacy system-prompt/attacked_context artifacts; "
            "query-suffix shared-budget results need occurrence-level joint-prompt source maps"
        )
    if args.max_entries > 0:
        entries = entries[:args.max_entries]
    compressor, results = make_compressor(args.compressor), []
    for idx, entry in enumerate(entries):
        original = (entry.get("system_prompt") if args.task == "spc"
                    else entry.get("context") or entry.get("original_context", ""))
        attacked = entry.get("attacked_context") or entry.get("attacked_prompt") or original
        benign_output = compressor.compress(original, rate=args.compression_rate)
        attack_output = compressor.compress(attacked, rate=args.compression_rate)
        benign_ranges = critical_ranges(args.task, entry, original)
        attack_ranges = critical_ranges(args.task, entry, attacked, attacked=True)
        benign = measure_ctrr(original, benign_output.get("source_map"), benign_ranges)
        attack = measure_ctrr(attacked, attack_output.get("source_map"), attack_ranges)
        row = {"idx": idx, "ctrr_benign": benign, "ctrr_attack": attack}
        if args.presence_proxy:
            row["presence_proxy_retained_fraction_benign"] = presence_proxy_retained_fraction(
                original, _compressed_text(benign_output), benign_ranges or [])
            row["presence_proxy_retained_fraction_attack"] = presence_proxy_retained_fraction(
                attacked, _compressed_text(attack_output), attack_ranges or [])
        results.append(row)

    paired = [row for row in results
              if row["ctrr_benign"]["status"] == CTRR_AVAILABLE
              and row["ctrr_attack"]["status"] == CTRR_AVAILABLE]
    summary = {
        "compressor": args.compressor, "task": args.task,
        "compression_rate": args.compression_rate,
        "ctrr_definition": "1-retained_critical_source_occurrences/total_critical_source_occurrences",
        "ctrr_aggregation": "sample_macro_mean_on_complete_pairs",
        "ctrr_status": CTRR_AVAILABLE if paired else CTRR_UNAVAILABLE,
        "n_input": len(results), "n_ctrr_paired": len(paired),
        "n_ctrr_unavailable": len(results) - len(paired),
    }
    if paired:
        benign_values = [row["ctrr_benign"]["ctrr"] for row in paired]
        attack_values = [row["ctrr_attack"]["ctrr"] for row in paired]
        benign_mean = sum(benign_values) / len(benign_values)
        attack_mean = sum(attack_values) / len(attack_values)
        summary.update(ctrr_benign=round(benign_mean, 4), ctrr_attack=round(attack_mean, 4),
                       ctrr_increase=round(attack_mean - benign_mean, 4))
    else:
        summary["reason"] = "No complete pair had occurrence-level source maps and critical spans"

    out_dir = os.path.join(args.output, args.compressor, args.task)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "retention_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(os.path.join(out_dir, "retention_per_instance.jsonl"), "w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")
    log.info("%s: %d/%d complete pairs", summary["ctrr_status"], len(paired), len(results))
    return summary


def aggregate_retention(output_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(output_dir, "*/*/retention_summary.json"))):
        with open(path, encoding="utf-8") as handle:
            rows.append(json.load(handle))
    if not rows:
        log.warning("No retention_summary.json found under %s", output_dir)
        return
    csv_rows = [["compressor", "task", "ctrr_status", "n_ctrr_paired",
                 "ctrr_benign", "ctrr_attack", "ctrr_increase"]]
    for row in sorted(rows, key=lambda item: (item["compressor"], item["task"])):
        print(row["compressor"], row["task"], row["ctrr_status"],
              row.get("ctrr_benign", "-"), row.get("ctrr_attack", "-"))
        csv_rows.append([row["compressor"], row["task"], row["ctrr_status"],
                         row["n_ctrr_paired"], row.get("ctrr_benign", ""),
                         row.get("ctrr_attack", ""), row.get("ctrr_increase", "")])
    with open(os.path.join(output_dir, "token_retention_table.csv"), "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(csv_rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Auditable critical-token removal study")
    parser.add_argument("--attack-results")
    parser.add_argument("--compressor")
    parser.add_argument("--task", choices=ALL_TASKS)
    parser.add_argument("--compression-rate", type=float, default=0.6)
    parser.add_argument("--max-entries", type=int, default=-1)
    parser.add_argument("--output", default="results/rq3_retention/")
    parser.add_argument("--presence-proxy", action="store_true",
                        help="Report an unaudited text-presence proxy (never labeled CTRR)")
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.aggregate:
        aggregate_retention(args.output)
    elif not args.attack_results or not args.compressor or not args.task:
        log.error("--attack-results, --compressor, --task required")
        sys.exit(1)
    else:
        run_retention_analysis(args)


if __name__ == "__main__":
    main()
