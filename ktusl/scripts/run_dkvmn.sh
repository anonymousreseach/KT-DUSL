#!/bin/bash -l
set -euo pipefail

cd "$HOME/kt_repo"
source venv/bin/activate

# DKVMN (Dynamic Key-Value Memory Network for KT)
python -m ktusl.experiments.run ktusl/config/dkvmn_multi.yaml --no-progress