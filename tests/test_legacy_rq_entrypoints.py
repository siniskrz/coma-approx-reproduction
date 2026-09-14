import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import run_pref_attack
import run_qa_attack


class LegacyRQEntrypointTest(unittest.TestCase):
    def test_pref_target_is_one_real_contiguous_span(self):
        entry = {
            "demo_1": "ordinary basic tool",
            "demo_2": "specialized accurate instrument",
            "best": "demo_2",
            "target": "demo_1",
        }
        result = run_pref_attack.compute_pref_target_offline(entry)

        self.assertTrue(result["success"])
        self.assertEqual(len(result["deleted_words"]), 1)
        removed = result["deleted_words"][0]
        self.assertEqual(result["original_context"].find(removed), result["span_start"])
        self.assertEqual(
            result["target_context"],
            result["original_context"][:result["span_start"]]
            + result["original_context"][result["span_start"] + len(removed):],
        )

    def test_selective_context_is_not_advertised_by_attack_runners(self):
        required = [
            "runner", "--data", "data.json", "--output", "out",
            "--compressor", "selective_context", "--surrogate-model", "model",
        ]
        for module in (run_qa_attack, run_pref_attack):
            with self.subTest(module=module.__name__):
                with patch.object(sys, "argv", required), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    module.parse_args()

    def test_rq_scripts_forward_real_parameters_and_paths(self):
        root = Path(__file__).resolve().parents[1]
        rq2 = (root / "scripts/reproduce_rq2.sh").read_text(encoding="utf-8")
        rq4 = (root / "scripts/reproduce_rq4.sh").read_text(encoding="utf-8")
        rq5 = (root / "scripts/reproduce_rq5.sh").read_text(encoding="utf-8")

        self.assertIn("run_surrogate_mismatch.py", rq2)
        self.assertIn('--compression-rate "$RATE"', rq2)
        self.assertIn('--backend-llm "$BACKEND"', rq2)
        self.assertIn("comattack/case_studies/case_study_cline.py", rq4)
        self.assertIn("comattack/case_studies/case_study_langchain_ollama.py", rq4)
        self.assertNotIn("PPLDetector", rq5)
        self.assertNotIn("BLEUDetector", rq5)
        self.assertNotIn("LLMDetector", rq5)
        self.assertIn("AttackDetector", rq5)
        self.assertIn("llm_inference", rq5)


if __name__ == "__main__":
    unittest.main()
