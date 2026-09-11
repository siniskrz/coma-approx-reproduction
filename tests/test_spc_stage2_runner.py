import unittest

from run_spc_stage2 import run_stage2, target_absent


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [len(word) for word in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join("x" * value for value in ids)


class Attacker:
    best_loss = float("inf")
    best_candidates = None

    def step(self, prompts, sentences, targets):
        self.calls = getattr(self, "calls", 0) + 1
        self.prompts = getattr(self, "prompts", []) + prompts
        return 1 / self.calls, [5, 6]


class Stage2RunnerTest(unittest.TestCase):
    def test_runs_500_steps_updates_prompt_and_validates_three_budgets(self):
        row = {"sample_id": "toy", "source_hash": "a" * 64,
               "original_query": "May it enter?", "surrogate_prefix": "Public rules:",
               "stage1": {"status": "COMPLETE", "surrogate_guardrails": ["must not enter"],
                          "critical_occurrences": [{"text": "not"}],
                          "baseline_label": "NO", "counterfactual_label": "YES"}}
        attacker = Attacker()
        calls = []

        def compress(prompt, rate):
            calls.append(rate)
            return {"text": prompt.replace("not", ""), "raw": {"rate": rate}}

        result = run_stage2([row], attacker, Tokenizer(), compress,
                            {"model": "m", "revision": "r", "weight_sha256": "h"},
                            initial_suffix="xxxx xxxxxx")[0]
        self.assertEqual(attacker.calls, 500)
        self.assertNotEqual(attacker.prompts[0], attacker.prompts[1])
        self.assertEqual(calls, [0.5, 0.6, 0.7])
        self.assertTrue(result["stage2"]["validated"])
        self.assertEqual(result["suffix_token_count"], 2)
        self.assertNotIn("system_prompt", result)

    def test_target_absence_uses_word_boundaries(self):
        self.assertTrue(target_absent("now allowed", "not"))
        self.assertFalse(target_absent("must not enter", "not"))
        self.assertTrue(target_absent("notable", "not"))

    def test_stage_two_rejects_private_system_input(self):
        with self.assertRaisesRegex(ValueError, "leaks trusted"):
            run_stage2([{"system_prompt": "private"}], Attacker(), Tokenizer(),
                       lambda prompt, rate: {"text": prompt},
                       {"model": "m", "revision": "r", "weight_sha256": "h"},
                       initial_suffix="xxxx")


if __name__ == "__main__":
    unittest.main()
