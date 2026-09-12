import unittest

from run_spc_stage2 import run_stage2, target_absent

try:
    import torch
    from comattack.attacks.extractive_suffix import AttackforLLMLingua2
except ImportError:
    torch = None
    AttackforLLMLingua2 = None


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
        self.last_step_metrics = {"objective_loss": 1 / self.calls}
        self.best_metrics = {"objective_loss": 1 / self.calls,
                             "target_keep_score": 0.2,
                             "signed_margins": [-0.1, -0.2, 0.05],
                             "worst_signed_margin": 0.05}
        return 1 / self.calls, [5, 6]


class Stage2RunnerTest(unittest.TestCase):
    @staticmethod
    def complete_stage1():
        backend = {"request": {}, "response": {}, "content": "answer", "error": None}
        no = {"request": {}, "response": {}, "content": "NO", "error": None}
        yes = {"request": {}, "response": {}, "content": "YES", "error": None}
        occurrence = {"text": "not", "start": 5, "end": 8}
        return {"status": "COMPLETE", "surrogate_guardrails": ["must not enter"],
                "critical_occurrences": [occurrence], "selected_target": "must enter",
                "baseline_label": "NO", "counterfactual_label": "YES",
                "baseline": {"label": "NO", "backend": backend, "judge": no},
                "trials": [{"target_prompt": "must enter", "deleted_occurrence": occurrence,
                            "outcome": {"label": "YES", "backend": backend, "judge": yes}}]}

    @unittest.skipUnless(torch is not None, "requires optional SPC torch dependency")
    def test_llmlingua2_filters_unstable_text_roundtrips_before_scoring(self):
        class RoundtripTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                del skip_special_tokens
                return "stable" if ids == [1, 2] else "changed"

            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [1, 2] if text == "stable" else [8]

        attacker = AttackforLLMLingua2.__new__(AttackforLLMLingua2)
        attacker.tokenizer = RoundtripTokenizer()
        candidates = torch.tensor([[1, 2], [3, 4]])
        kept = attacker._roundtrip_stable_candidates(candidates, torch.tensor([9, 9]))
        self.assertEqual(kept.tolist(), [[1, 2]])

    @unittest.skipUnless(torch is not None, "requires optional SPC torch dependency")
    def test_llmlingua2_budget_margin_prefers_target_below_all_cutoffs(self):
        class RankTokenizer:
            def encode(self, text, add_special_tokens=False):
                del text, add_special_tokens
                return [1]

        class MarginTokenizer:
            all_special_ids = [0, 2]

            def convert_ids_to_tokens(self, ids):
                table = {0: "<s>", 10: "▁forbi", 11: "dden", 20: "▁a",
                         21: "▁b", 22: "▁c", 23: "▁d", 2: "</s>"}
                return [table[value] for value in ids]

            def convert_tokens_to_string(self, tokens):
                return "".join(tokens).replace("▁", " ").strip()

        attacker = AttackforLLMLingua2.__new__(AttackforLLMLingua2)
        attacker.tokenizer = MarginTokenizer()
        attacker.rank_tokenizer = RankTokenizer()
        attacker.compression_rates = (0.5, 0.6, 0.7)
        ids = torch.tensor([[0, 10, 11, 20, 21, 22, 23, 2]] * 2)
        keep = torch.tensor([[0.5, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5],
                             [0.5, 0.1, 0.1, 0.8, 0.7, 0.6, 0.5, 0.5]])
        logits = torch.stack((torch.log1p(-keep), torch.log(keep)), dim=-1)
        attention = torch.ones_like(ids)
        target = torch.zeros_like(ids, dtype=torch.bool)
        target[:, 1:3] = True
        metrics = attacker._budget_margin_metrics(logits, ids, attention, target)
        losses = metrics["objective_loss"]
        self.assertGreater(losses[0].item(), 0)
        self.assertLessEqual(losses[1].item(), 0)
        self.assertLess(losses[1].item(), losses[0].item())
        self.assertEqual(tuple(metrics["signed_margins"].shape), (2, 3))
        self.assertTrue(torch.equal(
            metrics["worst_signed_margin"], metrics["signed_margins"].max(dim=1).values))
        self.assertTrue(torch.allclose(metrics["target_keep_score"], torch.tensor([0.9, 0.1])))

    @unittest.skipUnless(torch is not None, "requires optional SPC torch dependency")
    def test_llmlingua2_margin_backpropagates_through_quantile_cutoff(self):
        class MarginTokenizer:
            all_special_ids = [0, 2]

            def convert_ids_to_tokens(self, ids):
                table = {0: "<s>", 10: "▁target", 20: "▁a", 21: "▁b",
                         22: "▁c", 23: "▁d", 2: "</s>"}
                return [table[value] for value in ids]

        attacker = AttackforLLMLingua2.__new__(AttackforLLMLingua2)
        attacker.tokenizer = MarginTokenizer()
        attacker.rank_tokenizer = None
        attacker.compression_rates = (0.5,)
        ids = torch.tensor([[0, 10, 20, 21, 22, 23, 2]])
        keep = torch.tensor([[0.5, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5]])
        logits = torch.stack((torch.log1p(-keep), torch.log(keep)), dim=-1).requires_grad_()
        attention = torch.ones_like(ids)
        target = torch.zeros_like(ids, dtype=torch.bool)
        target[:, 1] = True

        attacker._budget_margin_losses(logits, ids, attention, target).sum().backward()

        self.assertGreater(logits.grad[0, 3:5].abs().sum().item(), 0)

    @unittest.skipUnless(torch is not None, "requires optional SPC torch dependency")
    def test_rank_weight_preserves_sentencepiece_boundary_marker(self):
        seen = []

        class RankTokenizer:
            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                seen.append(text)
                return [1, 2, 3] if text.startswith("▁") else [1]

        attacker = AttackforLLMLingua2.__new__(AttackforLLMLingua2)
        attacker.tokenizer = object()
        attacker.rank_tokenizer = RankTokenizer()
        self.assertEqual(attacker._rank_token_count(["▁forbi", "dden"]), 3)
        self.assertEqual(seen, ["▁forbidden"])

    def test_runs_500_steps_updates_prompt_and_validates_three_budgets(self):
        row = {"sample_id": "toy", "source_hash": "a" * 64,
               "original_query": "May it enter?", "surrogate_query": "May the public token enter?",
               "surrogate_prefix": "Public rules:",
               "stage1": self.complete_stage1()}
        attacker = Attacker()
        attacker.require_clean_target_retention = True
        calls = []

        def compress(prompt, rate):
            calls.append(rate)
            text = prompt.replace("not", "") if "xxxxx xxxxxx" in prompt else prompt
            return {"text": text, "raw": {"rate": rate}}

        result = run_stage2([row], attacker, Tokenizer(), compress,
                            {"model": "m", "revision": "r", "weight_sha256": "h"},
                            initial_suffix="xxxx xxxxxx")[0]
        self.assertEqual(attacker.calls, 500)
        self.assertNotEqual(attacker.prompts[0], attacker.prompts[1])
        self.assertEqual(calls, [0.5, 0.6, 0.7] * 2)
        self.assertTrue(result["stage2"]["validated"])
        self.assertEqual(len(result["stage2"]["loss_history"]), 500)
        self.assertEqual(result["stage2"]["loss_history"][-1]["margin_0.7"], 0.05)
        self.assertEqual(result["suffix_token_count"], 2)
        self.assertNotIn("system_prompt", result)

    def test_target_absence_uses_word_boundaries(self):
        self.assertTrue(target_absent("now allowed", "not"))
        self.assertFalse(target_absent("must not enter", "not"))
        self.assertTrue(target_absent("notable", "not"))

    def test_skips_target_not_stable_in_clean_compression(self):
        row = {"sample_id": "toy", "source_hash": "a" * 64,
               "original_query": "May it enter?", "surrogate_query": "May it enter?",
               "surrogate_prefix": "Public rules:", "stage1": self.complete_stage1()}

        def compress(prompt, rate):
            return {"text": prompt.replace("not", "") if rate == 0.7 else prompt,
                    "raw": {}}

        attacker = Attacker()
        attacker.require_clean_target_retention = True
        result = run_stage2(
            [row], attacker, Tokenizer(), compress,
            {"model": "m", "revision": "r", "weight_sha256": "h"},
            initial_suffix="xxxx xxxxxx", max_steps=1,
        )[0]
        self.assertTrue(result["skip"])
        self.assertFalse(result["clean_budget_trials"][-1]["target_retained"])
        self.assertFalse(hasattr(attacker, "calls"))

    def test_failed_artifact_keeps_optimizer_best_loss(self):
        row = {"sample_id": "toy", "source_hash": "a" * 64,
               "original_query": "May it enter?", "surrogate_query": "May it enter?",
               "surrogate_prefix": "Public rules:", "stage1": self.complete_stage1()}
        result = run_stage2(
            [row], Attacker(), Tokenizer(), lambda prompt, rate: {"text": prompt, "raw": {}},
            {"model": "m", "revision": "r", "weight_sha256": "h"},
            initial_suffix="xxxx xxxxxx", max_steps=1,
        )[0]
        self.assertFalse(result["stage2"]["validated"])
        self.assertEqual(result["stage2"]["best_loss"], 1.0)

    def test_stage_two_rejects_private_system_input(self):
        with self.assertRaisesRegex(ValueError, "leaks trusted"):
            run_stage2([{"system_prompt": "private"}], Attacker(), Tokenizer(),
                       lambda prompt, rate: {"text": prompt},
                       {"model": "m", "revision": "r", "weight_sha256": "h"},
                       initial_suffix="xxxx")


if __name__ == "__main__":
    unittest.main()
