import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_token_retention as retention


def source_map(text, kept):
    return [{"start": item["start"], "end": item["end"], "kept": flag}
            for item, flag in zip(retention._word_occurrences(text), kept)]


class TokenRetentionTests(unittest.TestCase):
    def test_ctrr_counts_repeated_occurrences_by_position(self):
        text = "deny box deny"
        measured = retention.measure_ctrr(
            text, source_map(text, [True, True, False]), [(0, 4), (9, 13)])
        self.assertEqual(measured["status"], retention.CTRR_AVAILABLE)
        self.assertEqual(measured["critical_occurrences"], 2)
        self.assertEqual(measured["retained_critical_occurrences"], 1)
        self.assertEqual(measured["ctrr"], 0.5)

    def test_missing_or_incomplete_map_is_unavailable(self):
        text = "must not approve"
        self.assertEqual(retention.measure_ctrr(text, None, [(5, 8)])["status"],
                         retention.CTRR_UNAVAILABLE)
        measured = retention.measure_ctrr(text, source_map(text, [True, False]), [(5, 8)])
        self.assertEqual(measured["status"], retention.CTRR_UNAVAILABLE)
        self.assertEqual(measured["reason"], "incomplete_source_map")

    def test_legacy_spc_artifacts_fail_closed(self):
        class Compressor:
            def compress(self, text, rate):
                return {"compressed_text": text.replace("not ", "")}

        entry = {"system_prompt": "must not approve",
                 "attacked_context": "must not approve suffix", "removed_phrases": ["not"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attacks.jsonl"
            path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
            args = SimpleNamespace(attack_results=str(path), compressor="fake", task="spc",
                                   compression_rate=0.6, max_entries=-1, output=directory,
                                   presence_proxy=True)
            with self.assertRaisesRegex(ValueError, "rejects legacy"):
                retention.run_retention_analysis(args)

    def test_run_uses_sample_macro_not_pooled_occurrences(self):
        entries = [
            {"context": "deny box", "attacked_context": "deny box",
             "answers": {"text": ["deny"], "answer_start": [0]}},
            {"context": "deny deny box", "attacked_context": "deny deny box",
             "answers": {"text": ["deny"], "answer_start": [0]}},
        ]

        class Compressor:
            def compress(self, text, rate):
                kept = [False, True] if len(retention._word_occurrences(text)) == 2 else [True, True, True]
                return {"compressed_text": text, "source_map": source_map(text, kept)}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attacks.jsonl"
            path.write_text("".join(json.dumps(x) + "\n" for x in entries), encoding="utf-8")
            args = SimpleNamespace(attack_results=str(path), compressor="fake", task="qa",
                                   compression_rate=0.6, max_entries=-1, output=directory,
                                   presence_proxy=False)
            with patch.object(retention, "make_compressor", return_value=Compressor()):
                summary = retention.run_retention_analysis(args)
            self.assertEqual(summary["ctrr_status"], retention.CTRR_AVAILABLE)
            self.assertEqual(summary["ctrr_benign"], 0.5)
            self.assertEqual(summary["ctrr_attack"], 0.5)
            self.assertEqual(summary["n_ctrr_paired"], 2)


if __name__ == "__main__":
    unittest.main()
