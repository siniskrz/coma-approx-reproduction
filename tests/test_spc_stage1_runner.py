import unittest

from run_spc_stage1 import run_stage1, validate_blind_row


class Compressor:
    def compress(self, text):
        return {"text": text, "raw": {"text": text}}


class Backend:
    def complete(self, messages, *, max_tokens):
        prompt = messages[-1]["content"]
        content = "APPROVED" if " not " not in prompt.lower() else "DENIED"
        return {"request": {"messages": messages}, "response": {"content": content},
                "content": content, "error": None}


class Judge:
    def complete(self, messages, *, max_tokens):
        content = "YES" if "APPROVED" in messages[-1]["content"] else "NO"
        return {"request": {"messages": messages}, "response": {"content": content},
                "content": content, "error": None}


class Stage1RunnerTest(unittest.TestCase):
    def test_blind_runner_observes_real_flip_without_trusted_fields(self):
        row = {"sample_id": "toy", "public_surrogate_id": "public-1",
               "source_hash": "ab" * 32, "original_query": "May it enter?",
               "surrogate_query": "May the public token enter?",
               "surrogate_prefix": "A public toy keeper follows this rule.",
               "surrogate_guardrails": ["must not approve entry"],
               "critical_candidates": ["not"]}
        result = run_stage1([row], Compressor(), Backend(), Judge())[0]
        self.assertEqual(result["stage1"]["status"], "COMPLETE")
        self.assertEqual(result["stage1"]["baseline_label"], "NO")
        self.assertEqual(result["stage1"]["counterfactual_label"], "YES")
        self.assertNotIn("system_prompt", result)
        self.assertTrue(result["stage1"]["trials"][0]["outcome"]["backend"]["response"])
        self.assertEqual(result["stage1"]["clean_joint_prompt"].count("must not approve entry"), 1)

    def test_private_system_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fields differ"):
            validate_blind_row({"system_prompt": "private"})

    def test_guardrail_not_just_prefix_is_in_stage_one_prompt(self):
        row = {"sample_id": "toy", "public_surrogate_id": "public-1",
               "source_hash": "ab" * 32, "original_query": "May it enter?",
               "surrogate_query": "May the public token enter?",
               "surrogate_prefix": "Public toy keeper rules:",
               "surrogate_guardrails": ["must not approve entry"],
               "critical_candidates": ["not"]}
        result = run_stage1([row], Compressor(), Backend(), Judge())[0]
        self.assertEqual(result["stage1"]["status"], "COMPLETE")
        self.assertIn("must not approve entry", result["stage1"]["clean_joint_prompt"])


if __name__ == "__main__":
    unittest.main()
