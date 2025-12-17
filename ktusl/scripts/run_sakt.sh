#!/bin/bash -l
set -euo pipefail

cd "$HOME/kt_repo"
source venv/bin/activate

# SAKT (Self-Attentive KT)
python -m ktusl.experiments.run ktusl/config/sakt_multi.yaml --no-progress
