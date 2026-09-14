import unittest

from comattack.evaluation.e2e_eval import (
    _parse_binary_judge,
    evaluate_dataset,
    evaluate_single,
    judge_spc,
)


class _Judge:
    def __init__(self, verdict):
        self.verdict = verdict

    def generate(self, *args, **kwargs):
        return self.verdict


class E2EIntegrityTest(unittest.TestCase):
    def test_spc_requires_exact_semantic_judge(self):
        entry = {"system_prompt": "Do not grant the toy badge."}
        with self.assertRaisesRegex(ValueError, "semantic judge"):
            judge_spc(entry, "A long answer that used to pass the length heuristic")
        self.assertIs(judge_spc(entry, "response", _Judge("YES")), True)
        self.assertIs(judge_spc(entry, "response", _Judge("NO")), False)
        self.assertIsNone(judge_spc(entry, "response", _Judge("NOT YES")))
        self.assertIsNone(_parse_binary_judge("YES because..."))

    def test_dataset_and_artifact_lengths_must_match(self):
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            evaluate_dataset([{}], [], None, None, "qa")

    def test_missing_compressed_text_is_not_silently_uncompressed(self):
        class Compressor:
            def compress(self, text, rate):
                return {}

        with self.assertRaisesRegex(ValueError, "no compressed text"):
            evaluate_single({}, "toy input", Compressor(), None, "qa")

    def test_all_unknown_has_no_asr(self):
        class Compressor:
            def compress(self, text, rate):
                return {"compressed_text": text}

        class Backend:
            def generate(self, *args, **kwargs):
                return "toy response"

        result = evaluate_dataset(
            [{"system_prompt": "Toy rule."}], ["toy input"],
            Compressor(), Backend(), "spc", judge_llm=_Judge("UNKNOWN"),
        )
        self.assertIsNone(result["asr"])
        self.assertEqual((result["total"], result["unknown"]), (0, 1))

    def test_asr_counts_only_clean_to_attacked_failures(self):
        class Compressor:
            def compress(self, text, rate):
                return {"compressed_text": text}

        class Backend:
            outputs = iter(["gold", "wrong", "wrong", "wrong"])

            def generate(self, *args, **kwargs):
                return next(self.outputs)

        dataset = [
            {"context": "clean one", "question": "q", "answers": {"text": ["gold"]}},
            {"context": "clean two", "question": "q", "answers": {"text": ["gold"]}},
        ]
        result = evaluate_dataset(
            dataset, ["attacked one", "attacked two"],
            Compressor(), Backend(), "qa",
        )

        self.assertEqual(result["asr"], 1.0)
        self.assertEqual((result["successes"], result["total"]), (1, 1))
        self.assertEqual(result["ineligible_clean_failures"], 1)
        self.assertEqual(
            [(r["clean_success"], r["attack_success"], r["eligible"])
             for r in result["per_instance"]],
            [(False, True, True), (True, True, False)],
        )

    def test_missing_clean_baseline_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "clean baseline"):
            evaluate_dataset(
                [{"answers": {"text": ["gold"]}}], ["attacked"],
                object(), object(), "qa",
            )


if __name__ == "__main__":
    unittest.main()
