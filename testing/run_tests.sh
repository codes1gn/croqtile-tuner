#!/usr/bin/env bash
# run_tests.sh — Entry point: run all croq-tune harness tests.
#
# USAGE (from repo root):
#   bash testing/run_tests.sh
#
# USAGE (from anywhere, using cursor-agent):
#   cursor-agent --run "bash /path/to/croqtile-tuner/testing/run_tests.sh"
#
# EXIT: 0 = all pass, 1 = one or more failed

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
bash testing/harness/run_all.sh
