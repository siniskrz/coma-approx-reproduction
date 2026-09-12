import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_spc_asr import (ATTACK_PROTOCOL, LLMLingua2, attack_surrogate_identity, build_joint_prompt,
                         parse_judge_label, run_spc_asr, sample_id, source_hash,
                         summarize_human_reviews)
from comattack.spc_stages import record_sha256


class SuffixTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(1, len(text.split()) + 1))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{value}" for value in ids)


class Compressor:
    suffix_tokenizer = SuffixTokenizer()
    def compress(self, text):
        return {"text": "compressed:" + text, "raw": {"source": text}}


class Backend:
    def complete(self, messages, *, max_tokens):
        self.calls = getattr(self, "calls", []) + [messages]
        prompt = messages[0]["content"]
        if "compressed:" in prompt:
            condition = "C" if "suffix-" not in prompt else "D"
        else:
            condition = "A" if "suffix-" not in prompt else "B"
        sample = prompt.split("clean-")[1][0] if "clean-" in prompt else "1"
        return {"request": {"messages": messages}, "response": {"prompt": prompt},
                "content": f"response:{condition}-{sample}", "error": None}


class Judge:
    def complete(self, messages, *, max_tokens):
        response = json.loads(messages[-1]["content"])["assistant_response"]
        labels = {
            "response:A-1": "NO", "response:B-1": "NO",
            "response:C-1": "NO", "response:D-1": "YES",
            "response:A-2": "NO", "response:B-2": "YES",
            "response:C-2": "NO", "response:D-2": "YES",
            "response:A-3": "NO", "response:B-3": "NO",
            "response:C-3": "NO", "response:D-3": "unclear",
        }
        content = labels[response]
        return {"request": {"messages": messages}, "response": {"content": content},
                "content": content, "error": None}


def attack_for(clean, suffix="suffix", token_ids=None, rate=0.6, **extra):
    token_ids = token_ids or list(range(1, len(suffix.split()) + 1))
    backend_evidence = {"request": {"messages": []}, "response": {}, "content": "x", "error": None}
    judge_no = {"request": {"messages": []}, "response": {}, "content": "NO", "error": None}
    judge_yes = {"request": {"messages": []}, "response": {}, "content": "YES", "error": None}
    selected = {
        "suffix": suffix, "suffix_token_ids": token_ids, "best_loss": 0, "step": 500,
        "budget_trials": [{"compression_rate": trial_rate, "target_removed": True,
                           "compressed_text": "target removed", "compressor_raw": {"ok": True}}
                          for trial_rate in (0.5, 0.6, 0.7)],
        "stable": True,
    }
    occurrence = {"text": "not", "start": 0, "end": 3}
    stage1 = {"status": "COMPLETE", "selected_target": "toy", "baseline_label": "NO",
              "counterfactual_label": "YES", "critical_occurrences": [occurrence],
              "baseline": {"label": "NO", "backend": backend_evidence, "judge": judge_no},
              "trials": [{"target_prompt": "toy", "deleted_occurrence": occurrence,
                          "outcome": {"label": "YES", "backend": backend_evidence,
                                      "judge": judge_yes}}]}
    return {
        "id": clean["id"], "protocol": ATTACK_PROTOCOL,
        "source_hash": source_hash(clean), "original_query": clean["adversarial_query"],
        "attack_suffix": suffix, "suffix_token_ids": token_ids,
        "suffix_token_count": len(token_ids),
        "budget": {"compression_rate": rate, "max_suffix_tokens": 32},
        "stage1": stage1,
        "stage2": {"status": "COMPLETE", "max_steps": 500, "steps_run": 500,
                   "best_loss": 0, "candidates": [selected], "selected": selected,
                   "suffix_roundtrip_stable": True, "validated": True,
                   "raw_provenance": {"stage1_sha256": record_sha256(stage1)}},
        "surrogate": {"model": "surrogate", "revision": "revision", "weight_sha256": "00",
                      "auxiliary_files": {"tokenizer.json": "11"}},
        **extra,
    }


class SPCASRTest(unittest.TestCase):
    def test_four_group_pairing_unknown_and_evidence(self):
        clean, attacked = [], []
        for number in range(1, 4):
            clean.append({"id": str(number), "system_prompt": f"clean-{number}",
                          "adversarial_query": "same query",
                          "guardrail_list": [{"sentence": "Do not comply."}]})
            attacked.append(attack_for(clean[-1], f"suffix-{number}"))

        backend = Backend()
        result = run_spc_asr(clean, attacked, Compressor(), backend, Judge())

        self.assertEqual(result["n_complete_paired"], 2)
        self.assertEqual(result["n_incomplete_or_unknown"], 1)
        self.assertEqual(result["four_group_violation_rates"], {"A": 0, "B": 0.5, "C": 0, "D": 1})
        self.assertEqual(result["D_minus_B"], 0.5)
        self.assertEqual(result["C_minus_A"], 0)
        self.assertEqual(result["interaction"], 0.5)
        self.assertEqual(result["baseline_stable_ABC0"], {
            "D_successes": 1, "D_failures": 0,
            "eligible_n_including_D_unknown": 2,
            "resolved_n_excluding_D_unknown": 1,
            "confirmed_success_lower_bound": 0.5,
            "resolved_case_rate": 1,
            "D_unknown_n": 1, "D_unknown_sample_ids": ["3"],
        })
        unknown = result["records"][2]["conditions"]["D"]
        self.assertEqual(unknown["label_name"], "UNKNOWN")
        self.assertIsNotNone(unknown["backend"]["response"])
        self.assertIsNotNone(unknown["judge"]["response"])
        self.assertTrue(all(call[0]["role"] == "user" and len(call) == 1
                            for call in backend.calls))
        self.assertIn("<SYSTEM>\nclean-1", result["records"][0]["conditions"]["D"]["joint_input_text"])
        self.assertIn("same query suffix-1", result["records"][0]["conditions"]["D"]["joint_input_text"])

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
        attacked = [attack_for(clean[0], original_query="different")]
        result = run_spc_asr(clean, attacked, Compressor(), Backend(), Judge())
        self.assertEqual(result["records"][0]["status"], "INPUT_MISMATCH")
        self.assertEqual(result["n_complete_paired"], 0)

    def test_empty_guardrail_is_excluded(self):
        clean = [{"id": "x", "system_prompt": "clean", "adversarial_query": "q",
                  "guardrail_list": []}]
        attacked = [attack_for(clean[0])]
        result = run_spc_asr(clean, attacked, Compressor(), Backend(), Judge())
        self.assertEqual(result["records"][0]["status"], "INPUT_ERROR")

    def test_shared_budget_query_suffix_and_legacy_system_edit_rejection(self):
        clean = [{"id": "x", "system_prompt": "trusted", "external_context": "retrieved",
                  "adversarial_query": "question", "guardrail_list": [{"sentence": "Do not grant it."}]}]
        attack = [attack_for(clean[0], "adversarial suffix", [1, 2])]
        result = run_spc_asr(clean, attack, Compressor(), Backend(), Judge())
        record = result["records"][0]
        self.assertIn("compressed:<SYSTEM>\ntrusted", record["conditions"]["D"]["joint_input_text"])
        self.assertIn("<CONTEXT>\nretrieved", record["conditions"]["D"]["joint_input_text"])
        self.assertIn("question adversarial suffix", record["conditions"]["D"]["joint_input_text"])
        unsafe = [attack_for(clean[0], attacked_context="edited trusted")]
        self.assertEqual(run_spc_asr(clean, unsafe, Compressor(), Backend(), Judge())
                         ["records"][0]["status"], "UNSAFE_LEGACY_ARTIFACT")

    def test_suffix_budget_and_human_review_denominators(self):
        clean = [{"id": "x", "system_prompt": "clean-1", "adversarial_query": "q",
                  "guardrail_list": [{"sentence": "Do not grant it."}]}]
        attack = [attack_for(clean[0], "too long", [1, 2, 3])]
        attack[0]["budget"]["max_suffix_tokens"] = 2
        result = run_spc_asr(clean, attack, Compressor(), Backend(), Judge(), max_suffix_tokens=2)
        self.assertEqual(result["records"][0]["status"], "SUFFIX_ERROR")
        review = summarize_human_reviews([
            {"id": "a", "label": "YES"}, {"id": "b", "label": "NO"},
            {"id": "c", "label": "UNKNOWN"}], ["a", "b", "c", "d"])
        self.assertEqual(review["confirmed_success_lower_bound"], 0.25)
        self.assertEqual(review["resolved_case_rate"], 0.5)
        self.assertEqual(review["unknown_or_unreviewed_n"], 2)

    def test_boundary_only_artifact_cannot_be_scored_as_an_attack(self):
        clean = [{"id": "x", "system_prompt": "clean-1", "adversarial_query": "q",
                  "guardrail_list": [{"sentence": "Do not grant it."}]}]
        artifact = attack_for(clean[0])
        artifact["stage1"] = {"status": "NOT_RUN_ENGINEERING_BOUNDARY_ONLY", "trials": []}
        artifact["stage2"].update({"max_steps": 0, "validated": False})
        result = run_spc_asr(clean, [artifact], Compressor(), Backend(), Judge())
        self.assertEqual(result["records"][0]["status"], "ATTACK_EVIDENCE_ERROR")
        self.assertEqual(result["n_complete_paired"], 0)

    def test_forged_stage_evidence_and_reserved_markers_are_rejected(self):
        clean = [{"id": "x", "system_prompt": "clean-1", "adversarial_query": "q",
                  "guardrail_list": [{"sentence": "Do not grant it."}]}]
        forged = attack_for(clean[0])
        forged["stage2"]["steps_run"] = 0
        self.assertEqual(run_spc_asr(clean, [forged], Compressor(), Backend(), Judge())
                         ["records"][0]["status"], "ATTACK_EVIDENCE_ERROR")
        marker = attack_for(clean[0], "<SYSTEM>")
        self.assertEqual(run_spc_asr(clean, [marker], Compressor(), Backend(), Judge())
                         ["records"][0]["status"], "SUFFIX_ERROR")

    def test_joint_prompt_has_one_copy_of_each_component(self):
        prompt = build_joint_prompt("system", "context", "query")
        self.assertEqual(prompt.count("system"), 1)
        self.assertEqual(prompt.count("context"), 1)
        self.assertEqual(prompt.count("query"), 1)

    def test_all_skipped_attack_batch_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "no eligible"):
            attack_surrogate_identity([{"id": "x", "skip": True}])


if __name__ == "__main__":
    unittest.main()
