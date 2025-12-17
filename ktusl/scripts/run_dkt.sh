#!/bin/bash -l
set -euo pipefail

cd "$HOME/kt_repo"
source venv/bin/activate

# DKT (LSTM)
python -m ktusl.experiments.run ktusl/config/dkt_multi.yaml --no-progress
