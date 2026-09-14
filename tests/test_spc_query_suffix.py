import copy
import json
import unittest

from comattack.spc_query_suffix import (
    FORBIDDEN_ARTIFACT_FIELDS,
    append_suffix,
    build_artifact,
    canonical_source_hash,
    configure_right_padding,
    prepare_blind_inputs,
    require_validated_artifact,
    suffix_token_ids,
    target_span_mask,
    validate_blind_row,
    validate_prompt_batch,
)


class ToyTokenizer:
    padding_side = "left"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(1, len(text.split()) + 1))

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(f"t{token}" for token in ids)


class QuerySuffixBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "sample_id": "toy-perm-01",
            "lineage_id": "private-01",
            "lineage_sha256": "cd" * 32,
            "system_prompt": "Trusted toy rule bytes must remain unchanged.",
            "context": "Public cardboard-box context.",
            "adversarial_query": "May the red token enter?",
            "guardrail_list": [{"sentence": "Do not admit it.", "keyword": "not"}],
        }
        self.public_pool = [{
            "pool_id": "public-01",
            "lineage_id": "public-01",
            "lineage_sha256": "cd" * 32,
            "surrogate_prefix": "Public policy for imaginary glass counters.",
            "surrogate_query": "May the glass counter enter?",
            "surrogate_guardrails": ["An amber shape lacks access to shelf omega."],
            "critical_candidates": ["lacks access"],
        }]
        self.surrogate = {"model": "toy-model", "revision": "r1", "weight_sha256": "ab" * 32}

    def test_artifact_only_appends_query_suffix_and_never_copies_trusted_text(self):
        before = json.dumps(self.row, ensure_ascii=False, sort_keys=True).encode()
        blind = prepare_blind_inputs([self.row], self.public_pool)[0]
        artifact = build_artifact(
            blind,
            "compression robustness marker",
            ToyTokenizer(),
            compression_rate=0.6,
            surrogate=self.surrogate,
        )
        after = json.dumps(self.row, ensure_ascii=False, sort_keys=True).encode()

        self.assertEqual(before, after)
        self.assertFalse(FORBIDDEN_ARTIFACT_FIELDS.intersection(artifact))
        self.assertNotIn(self.row["system_prompt"], json.dumps(artifact))
        self.assertEqual(
            append_suffix(artifact["original_query"], artifact["attack_suffix"]),
            "May the red token enter? compression robustness marker",
        )
        self.assertEqual(artifact["source_hash"], canonical_source_hash(self.row))
        self.assertFalse(artifact["stage2"]["validated"])
        self.assertTrue(artifact["stage2"]["suffix_roundtrip_stable"])
        with self.assertRaisesRegex(ValueError, "Stage-I"):
            require_validated_artifact(artifact)

    def test_blind_input_contains_only_public_surrogate_and_provenance(self):
        blind = prepare_blind_inputs([self.row], self.public_pool)[0]
        self.assertEqual(set(blind), {
            "sample_id", "public_surrogate_id", "lineage_sha256", "source_hash", "original_query", "surrogate_query", "surrogate_prefix",
            "surrogate_guardrails", "critical_candidates",
        })
        self.assertTrue(blind["public_surrogate_id"])
        serialized = json.dumps(blind)
        self.assertNotIn(self.row["system_prompt"], serialized)
        self.assertNotIn(self.row["context"], serialized)
        self.assertNotIn("Do not admit it.", serialized)

        contaminated = copy.deepcopy(blind)
        contaminated["system_prompt"] = "must be rejected"
        with self.assertRaisesRegex(ValueError, "extra=.*system_prompt"):
            validate_blind_row(contaminated)

    def test_mixed_lineage_and_normalized_policy_leaks_fail_before_stage_one(self):
        mixed = copy.deepcopy(self.public_pool)
        mixed[0]["lineage_sha256"] = "ef" * 32
        with self.assertRaisesRegex(ValueError, "different transformation lineages"):
            prepare_blind_inputs([self.row], mixed)

        leaked = copy.deepcopy(self.public_pool)
        leaked[0]["surrogate_guardrails"] = ["Do   not, admit it!"]
        with self.assertRaisesRegex(ValueError, "copies private toy policy text"):
            prepare_blind_inputs([self.row], leaked)

    def test_suffix_is_limited_to_at_most_32_tokenizer_tokens(self):
        self.assertEqual(len(suffix_token_ids(ToyTokenizer(), "one two three", 32)), 3)
        with self.assertRaisesRegex(ValueError, "33 tokens"):
            suffix_token_ids(ToyTokenizer(), " ".join(["x"] * 33), 32)

    def test_target_mask_uses_target_slices_not_suffix_positions(self):
        mask = target_span_mask([slice(1, 3), slice(0, 1)], [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        self.assertEqual(mask, [
            [False, True, True, False, False],
            [True, False, False, False, False],
        ])
        self.assertFalse(mask[0][-1], "query suffix position must not become the deletion target")

    def test_query_suffix_batching_forces_right_padding(self):
        tokenizer = ToyTokenizer()
        self.assertEqual(tokenizer.padding_side, "left")
        self.assertIs(configure_right_padding(tokenizer), tokenizer)
        self.assertEqual(tokenizer.padding_side, "right")

    def test_query_suffix_batches_cannot_be_empty_or_truncated(self):
        with self.assertRaisesRegex(ValueError, "non-empty and equal-length"):
            validate_prompt_batch([], [], [])
        with self.assertRaisesRegex(ValueError, r"\(2, 1, 2\)"):
            validate_prompt_batch(["a", "b"], ["rule"], ["word", "word"])
        validate_prompt_batch(["a"], ["rule"], ["word"])


if __name__ == "__main__":
    unittest.main()
