import unittest

from comattack.spc_stages import (
    content_suffix_bounds,
    make_stage_two_artifact,
    optimize_suffix_checkpoints,
    select_dropout_target,
    stage_two_inputs,
    validate_budget_candidates,
)


class SPCStagesTest(unittest.TestCase):
    @staticmethod
    def evidence(label):
        return {"label": label,
                "backend": {"request": {}, "response": {}, "content": "answer", "error": None},
                "judge": {"request": {}, "response": {}, "content": label, "error": None}}

    def test_suffix_bounds_exclude_bos_eos_and_padding(self):
        self.assertEqual(content_suffix_bounds([1, 0, 0, 0, 1, 1], 2), (2, 4))

    def test_stage_one_requires_observed_no_to_yes_flip(self):
        def simulate(prompt):
            return {"label": "YES" if "not" not in prompt.lower() else "NO",
                    "raw": {"prompt": prompt}}

        result = select_dropout_target(
            "Public toy keeper must not approve the square badge.",
            ["not", "badge"],
            simulate,
            surrogate_guardrails=["Public toy keeper must not approve the square badge."],
        )
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["baseline_label"], "NO")
        self.assertEqual(result["counterfactual_label"], "YES")
        self.assertEqual(result["critical_occurrences"][0]["text"], "not")
        self.assertTrue(result["trials"][0]["outcome"]["raw"])
        self.assertEqual(result["surrogate_guardrails"],
                         ["Public toy keeper must not approve the square badge."])

    def test_stage_one_does_not_invent_a_target(self):
        result = select_dropout_target(
            "Public toy text.", ["missing"], lambda prompt: {"label": "NO"},
            surrogate_guardrails=["A public toy rule."],
        )
        self.assertEqual(result["status"], "NO_FEASIBLE_TARGET")
        self.assertIsNone(result["selected_target"])

    def test_stage_one_candidates_are_independent_not_cumulative(self):
        seen = []

        def simulate(prompt):
            seen.append(prompt)
            return {"label": "YES" if "alpha" not in prompt and "beta" not in prompt else "NO"}

        result = select_dropout_target(
            "rule alpha beta", ["alpha", "beta"], simulate,
            surrogate_guardrails=["rule alpha beta"],
        )
        self.assertEqual(result["status"], "NO_FEASIBLE_TARGET")
        self.assertEqual(seen, ["rule alpha beta", "rule  beta", "rule alpha "])

    def test_stage_one_rejects_missing_surrogate_guardrails(self):
        with self.assertRaisesRegex(ValueError, "surrogate_guardrails"):
            select_dropout_target("Public toy text.", ["toy"],
                                  lambda prompt: {"label": "NO"},
                                  surrogate_guardrails=[])

    def test_stage_two_requires_all_three_rates(self):
        candidates = [{"suffix": "toy marker", "suffix_token_ids": [1, 2],
                       "suffix_token_count": 2, "suffix_roundtrip_stable": True,
                       "best_loss": 0.1}]
        result = validate_budget_candidates(
            candidates, lambda suffix, rate: {"target_removed": rate != 0.6})
        self.assertFalse(result["validated"])
        self.assertEqual(len(result["candidates"][0]["budget_trials"]), 3)
        with self.assertRaisesRegex(ValueError, "0.5"):
            validate_budget_candidates(candidates, lambda suffix, rate: {}, rates=(0.6,))

    def test_stage_two_emits_strict_selected_candidate_and_three_raw_trials(self):
        occurrence = {"text": "not", "start": 18, "end": 21}
        stage1 = {
            "status": "COMPLETE", "baseline_label": "NO", "counterfactual_label": "YES",
            "selected_target": "target", "critical_occurrences": [occurrence],
            "surrogate_guardrails": ["A public token is not admitted."],
            "baseline": self.evidence("NO"),
            "trials": [{"target_prompt": "target", "deleted_occurrence": occurrence,
                        "outcome": self.evidence("YES")}],
        }
        row = {"sample_id": "toy-perm-01", "source_hash": "ab" * 32,
               "original_query": "May the token enter?", "surrogate_query": "May the public token enter?",
               "surrogate_prefix": "Public toy policy.",
               "stage1": stage1}
        self.assertEqual(stage_two_inputs(row)[2:], ("A public token is not admitted.", "not"))

        artifact = make_stage_two_artifact(
            row,
            [{"suffix": "marker", "suffix_token_ids": [1], "suffix_token_count": 1,
              "best_loss": 0.1, "suffix_roundtrip_stable": True, "step": 500}],
            lambda suffix, rate: {"target_removed": True, "raw": {"rate": rate}},
            {"model": "public-surrogate", "revision": "r1", "weight_sha256": "cd" * 32},
            raw_provenance={"stage1_sha256": "ef" * 32},
        )
        self.assertTrue(artifact["stage2"]["validated"])
        self.assertEqual(artifact["stage2"]["status"], "COMPLETE")
        self.assertEqual(artifact["stage2"]["selected"]["budget_trials"][1]["raw"]["rate"], 0.6)
        self.assertEqual(artifact["attack_suffix"], "marker")

    def test_optimizer_runs_exactly_500_steps_and_drops_unstable_suffixes(self):
        class Attacker:
            calls = 0

            def step(self, prompts, sentences, targets):
                self.calls += 1
                del prompts, sentences, targets
                return 0.5, [1, 2]

        class Tokenizer:
            def decode(self, ids, skip_special_tokens=True):
                del ids, skip_special_tokens
                return "one two"

            def encode(self, text, add_special_tokens=False):
                del text, add_special_tokens
                return [1, 2]

        attacker = Attacker()
        candidates, steps, loss_history = optimize_suffix_checkpoints(
            attacker, "prompt", "sentence", "target", Tokenizer(), checkpoint_every=100)
        self.assertEqual(attacker.calls, 500)
        self.assertEqual(steps, 500)
        self.assertEqual(len(loss_history), 500)
        self.assertEqual(loss_history[0], {
            "step": 1, "step_objective_loss": 0.5, "best_objective_loss": 0.5,
            "step_loss": 0.5, "best_loss": 0.5, "loss": 0.5})
        self.assertEqual(len(candidates), 1)

    def test_optimizer_records_signed_margin_metrics(self):
        class Attacker:
            last_step_loss = 0.4
            last_step_metrics = {"objective_loss": 0.4}
            best_metrics = {"objective_loss": 0.3, "target_keep_score": 0.2,
                            "compression_rates": [0.7], "signed_margins": [0.05],
                            "worst_signed_margin": 0.05}

            def step(self, prompts, sentences, targets):
                del prompts, sentences, targets
                return 0.3, [1]

        class Tokenizer:
            def decode(self, ids, skip_special_tokens=True):
                del ids, skip_special_tokens
                return "x"

            def encode(self, text, add_special_tokens=False):
                del text, add_special_tokens
                return [1]

        candidates, _, history = optimize_suffix_checkpoints(
            Attacker(), "prompt", "sentence", "target", Tokenizer(), max_steps=1)
        self.assertEqual(history[0]["step_objective_loss"], 0.4)
        self.assertEqual(history[0]["margin_0.7"], 0.05)
        self.assertNotIn("margin_0.5", history[0])
        self.assertEqual(history[0]["worst_signed_margin"], 0.05)
        self.assertEqual(candidates[0]["target_keep_score"], 0.2)

    def test_optimizer_keeps_unique_stable_candidates_between_scheduled_checkpoints(self):
        class Attacker:
            calls = 0

            def step(self, prompts, sentences, targets):
                self.calls += 1
                del prompts, sentences, targets
                return 1 / self.calls, [self.calls, 2]

        class Tokenizer:
            def decode(self, ids, skip_special_tokens=True):
                del skip_special_tokens
                return " ".join(str(value) for value in ids)

            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [int(value) for value in text.split()]

        candidates, _, _ = optimize_suffix_checkpoints(
            Attacker(), "prompt", "sentence", "target", Tokenizer(),
            checkpoint_every=100,
        )
        self.assertEqual(len(candidates), 500)
        self.assertFalse(candidates[24]["scheduled_checkpoint"])
        self.assertTrue(candidates[99]["scheduled_checkpoint"])


if __name__ == "__main__":
    unittest.main()
