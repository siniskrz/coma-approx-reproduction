import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_spc_asr import LLMLingua2, parse_judge_label, run_spc_asr, sample_id
from run_guardrail_attack import validate_surrogate


class Compressor:
    def compress(self, text):
        return {"text": "compressed:" + text, "raw": {"source": text}}


class Backend:
    def complete(self, messages, *, max_tokens):
        system = messages[0]["content"]
        return {"request": {"messages": messages}, "response": {"system": system},
                "content": "response:" + system, "error": None}


class Judge:
    def complete(self, messages, *, max_tokens):
        response = json.loads(messages[-1]["content"])["assistant_response"]
        labels = {
            "response:clean-1": "NO", "response:attack-1": "NO",
            "response:compressed:clean-1": "NO", "response:compressed:attack-1": "YES",
            "response:clean-2": "NO", "response:attack-2": "YES",
            "response:compressed:clean-2": "NO", "response:compressed:attack-2": "YES",
            "response:clean-3": "NO", "response:attack-3": "NO",
            "response:compressed:clean-3": "NO", "response:compressed:attack-3": "unclear",
        }
        content = labels[response]
        return {"request": {"messages": messages}, "response": {"content": content},
                "content": content, "error": None}


class SPCASRTest(unittest.TestCase):
    def test_four_group_pairing_unknown_and_evidence(self):
        clean, attacked = [], []
        for number in range(1, 4):
            clean.append({"id": str(number), "system_prompt": f"clean-{number}",
                          "adversarial_query": "same query",
                          "guardrail_list": [{"sentence": "Do not comply."}]})
            attacked.append({**clean[-1], "attacked_context": f"attack-{number}"})

        result = run_spc_asr(clean, attacked, Compressor(), Backend(), Judge())

        self.assertEqual(result["n_complete_paired"], 2)
        self.assertEqual(result["n_incomplete_or_unknown"], 1)
        self.assertEqual(result["four_group_violation_rates"], {"A": 0, "B": 0.5, "C": 0, "D": 1})
        self.assertEqual(result["D_minus_B"], 0.5)
        self.assertEqual(result["C_minus_A"], 0)
        self.assertEqual(result["interaction"], 0.5)
        self.assertEqual(result["baseline_stable_ABC0"], {
            "D_successes": 1, "n": 1, "rate": 1,
            "D_unknown_n": 1, "D_unknown_sample_ids": ["3"],
        })
        unknown = result["records"][2]["conditions"]["D"]
        self.assertEqual(unknown["label_name"], "UNKNOWN")
        self.assertIsNotNone(unknown["backend"]["response"])
        self.assertIsNotNone(unknown["judge"]["response"])

    def test_judge_parser_is_exact(self):
        self.assertEqual(parse_judge_label(" YES\n"), 1)
        self.assertEqual(parse_judge_label("no"), 0)
        self.assertIsNone(parse_judge_label("YES because..."))
        self.assertIsNone(parse_judge_label("UNKNOWN"))

    def test_repeated_prompt_ids_are_disambiguated_by_query(self):
        one = sample_id({"prompt_id": "p", "guardrail_index": 0, "adversarial_query": "one"})
        two = sample_id({"prompt_id": "p", "guardrail_index": 0, "adversarial_query": "two"})
        self.assertNotEqual(one, two)

    def test_snapshot_revision_gate_runs_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                LLMLingua2(directory, "different-revision", 0.6)
            with self.assertRaises(ValueError):
                LLMLingua2(directory, Path(directory).name, 0)

    def test_attack_input_provenance_mismatch_is_excluded(self):
        clean = [{"id": "x", "system_prompt": "clean", "adversarial_query": "q",
                  "guardrail_list": [{"sentence": "Do not grant it."}]}]
        attacked = [{**clean[0], "adversarial_query": "different",
                     "attacked_context": "attack"}]
        result = run_spc_asr(clean, attacked, Compressor(), Backend(), Judge())
        self.assertEqual(result["records"][0]["status"], "INPUT_MISMATCH")
        self.assertEqual(result["n_complete_paired"], 0)

    def test_empty_guardrail_is_excluded(self):
        clean = [{"id": "x", "system_prompt": "clean", "adversarial_query": "q",
                  "guardrail_list": []}]
        attacked = [{**clean[0], "attacked_context": "attack"}]
        result = run_spc_asr(clean, attacked, Compressor(), Backend(), Judge())
        self.assertEqual(result["records"][0]["status"], "INPUT_ERROR")

    def test_surrogate_pin_is_checked_before_loading(self):
        class Args:
            surrogate_revision = "snapshot"
            surrogate_weight_sha256 = "00"

        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot"
            snapshot.mkdir()
            (snapshot / "model.safetensors").write_bytes(b"weights")
            Args.surrogate_model = str(snapshot)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_surrogate(Args)


if __name__ == "__main__":
    unittest.main()
