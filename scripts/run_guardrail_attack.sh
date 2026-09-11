#!/usr/bin/env bash
set -euo pipefail

echo "ERROR: this legacy SPC launcher is disabled because it edited the trusted system prompt." >&2
echo "Use prepare_spc_blind_inputs.py, run_spc_stage1.py, run_spc_stage2.py, and run_spc_asr.py." >&2
exit 2
