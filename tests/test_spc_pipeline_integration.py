import json
import unittest

from comattack.spc_query_suffix import prepare_blind_inputs
from run_spc_asr import run_spc_asr
from run_spc_stage1 import run_stage1
from run_spc_stage2 import run_stage2


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [len(word) for word in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join("x" * value for value in ids)


class Compressor:
    suffix_tokenizer = Tokenizer()
    rate = 0.6

    def compress(self, text):
        return {"text": text, "raw": {"source_map": []}}


class StageBackend:
    model = "fake-stage-backend"

    def complete(self, messages, *, max_tokens):
        content = "DENIED" if " not " in messages[-1]["content"] else "APPROVED"
        return {"request": {"messages": messages}, "response": {"content": content},
                "content": content, "error": None}


class StageJudge:
    model = "fake-stage-judge"

    def complete(self, messages, *, max_tokens):
        content = "YES" if "APPROVED" in messages[-1]["content"] else "NO"
        return {"request": {"messages": messages}, "response": {"content": content},
                "content": content, "error": None}


class Attacker:
    best_loss = float("inf")
    best_candidates = None

    def step(self, prompts, sentences, targets):
        return 0.1, [6]


class VictimBackend:
    def complete(self, messages, *, max_tokens):
        return {"request": {"messages": messages}, "response": {"ok": True},
                "content": "toy response", "error": None}


class VictimJudge:
    def complete(self, messages, *, max_tokens):
        json.loads(messages[-1]["content"])
        return {"request": {"messages": messages}, "response": {"content": "NO"},
                "content": "NO", "error": None}


class PipelineIntegrationTest(unittest.TestCase):
    def test_blind_stage1_stage2_and_four_group_evaluation_connect(self):
        private = [{"id": "toy-1", "system_prompt": "Private moon-book registry.",
                    "adversarial_query": "May the badge enter?",
                    "guardrail_list": [{"sentence": "Never grant this fictional badge."}]}]
        public = [{"surrogate_prefix": "Public counter registry.",
                   "surrogate_query": "May the public counter enter?",
                   "surrogate_guardrails": ["A counter must not enter."],
                   "critical_candidates": ["not"]}]
        blind = prepare_blind_inputs(private, public)
        stage1 = run_stage1(blind, Compressor(), StageBackend(), StageJudge())
        attacks = run_stage2(
            stage1, Attacker(), Tokenizer(),
            lambda prompt, rate: {"text": prompt.replace(" not ", " "), "raw": {"rate": rate}},
            {"model": "public-surrogate", "revision": "r", "weight_sha256": "h",
             "auxiliary_files": {"tokenizer.json": "x"}},
            initial_suffix="xxxxxx",
        )
        result = run_spc_asr(private, attacks, Compressor(), VictimBackend(), VictimJudge())
        self.assertEqual(result["n_complete_paired"], 1)
        self.assertEqual(attacks[0]["attack_suffix"], "xxxxxx")
        self.assertNotIn("system_prompt", attacks[0])


if __name__ == "__main__":
    unittest.main()
