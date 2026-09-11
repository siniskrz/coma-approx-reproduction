import subprocess
import sys
import unittest
from pathlib import Path

import run_guardrail_attack
import run_surrogate_mismatch


ROOT = Path(__file__).resolve().parents[1]


class LegacySPCEntrypointTest(unittest.TestCase):
    def test_guardrail_runner_fails_closed(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "run_guardrail_attack.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("modifies trusted system prompt", result.stderr)
        self.assertIn("run_spc_stage2.py", result.stderr)

    def test_guardrail_runner_has_no_callable_legacy_attack(self):
        with self.assertRaisesRegex(RuntimeError, "query-suffix"):
            run_guardrail_attack.reject_legacy_spc()
        self.assertFalse(hasattr(run_guardrail_attack, "compute_guardrail_target"))
        self.assertFalse(hasattr(run_guardrail_attack, "run_extractive_guardrail"))

    def test_surrogate_mismatch_rejects_spc_before_loading_data(self):
        with self.assertRaisesRegex(ValueError, "dedicated query-suffix"):
            run_surrogate_mismatch.reject_legacy_spc_task("spc")
        run_surrogate_mismatch.reject_legacy_spc_task("qa")

    def test_batch_launchers_cannot_invoke_legacy_spc(self):
        rq1 = (ROOT / "scripts" / "reproduce_rq1.sh").read_text(encoding="utf-8")
        guardrail = (ROOT / "scripts" / "run_guardrail_attack.sh").read_text(
            encoding="utf-8"
        )
        mismatch = (ROOT / "run_surrogate_mismatch.sh").read_text(encoding="utf-8")
        rq3 = (ROOT / "scripts" / "reproduce_rq3.sh").read_text(encoding="utf-8")
        rq5 = (ROOT / "scripts" / "reproduce_rq5.sh").read_text(encoding="utf-8")
        self.assertNotIn("run_cmd python run_guardrail_attack.py", rq1)
        self.assertIn('""|spc)', rq1)
        self.assertIn("exit 2", rq1)
        self.assertNotIn("python run_guardrail_attack.py", guardrail)
        self.assertIn("exit 2", guardrail)
        self.assertNotIn("TASKS+=(spc)", mismatch)
        self.assertIn("legacy SPC surrogate-mismatch runs are disabled", mismatch)
        self.assertIn("legacy SPC surrogate-mismatch runs are disabled", rq3)
        self.assertNotIn("guardrail_extractive_results.jsonl", rq5)

    def test_readme_names_only_query_suffix_route_as_reportable_spc(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertTrue(
            (ROOT / "run_spc_stage2.py").is_file(),
            "README must not advertise a missing Stage-II runner",
        )
        self.assertIn("only route in this repository", readme)
        self.assertIn("python run_spc_stage1.py", readme)
        self.assertIn("python run_spc_stage2.py", readme)
        self.assertIn("python run_spc_asr.py", readme)
        self.assertNotIn("python run_guardrail_attack.py \\", readme)


if __name__ == "__main__":
    unittest.main()
