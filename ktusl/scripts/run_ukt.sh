#!/bin/bash -l
set -euo pipefail

cd "$HOME/kt_repo"
source venv/bin/activate

# UKT (uncertainty-aware)
python -m ktusl.experiments.run ktusl/config/ukt_multi.yaml --no-progress
