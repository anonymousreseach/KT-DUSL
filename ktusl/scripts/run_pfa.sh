#!/bin/bash -l
set -euo pipefail

cd "$HOME/kt_repo"
source venv/bin/activate

# PFA (Performance Factors Analysis)
python -m ktusl.experiments.run ktusl/config/pfa_multi.yaml --no-progress
